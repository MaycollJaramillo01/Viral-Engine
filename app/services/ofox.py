import math
import os
import time
from typing import Any, Mapping
from urllib.parse import quote_plus

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call

from app.config import config
from app.services import generation_progress
from app.models.schema import MaterialInfo, VideoAspect


DEFAULT_BASE_URL = "https://api.ofox.ai/v1"
DEFAULT_MODEL_ID = "bytedance/seedance-2.0-fast"
DEFAULT_RESOLUTION = "720p"
# Por defecto se fija el canal de proveedores internacionales: para audiencias globales la política de contenido es más coherente; si se configura explícitamente vacío se devuelve
# al gateway, que reparte por pesos entre los proveedores disponibles.
DEFAULT_PROVIDER_TYPE = "byteplus"
# El modelo por defecto bytedance/seedance-2.0-fast solo acepta 4-15 segundos (valor validado por el servidor en pruebas reales).
# Otros modelos tienen intervalos distintos (por ejemplo alibaba/wan-2.7 es 2-15 segundos); al cambiar de modelo hay que
# ajustar el intervalo en la configuración; las peticiones fuera del intervalo la API las rechaza con un 400 claro, sin cobro.
DEFAULT_MIN_DURATION_SECONDS = 4
DEFAULT_MAX_DURATION_SECONDS = 15
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
MAX_POLL_RETRIES = 5
RETRY_BASE_SECONDS = 1.0
MAX_ERROR_TEXT_LENGTH = 500
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
TERMINAL_SUCCESS_STATUSES = frozenset({"completed", "succeeded"})
TERMINAL_FAILURE_STATUSES = frozenset(
    {"failed", "error", "cancelled", "canceled", "expired"}
)
# Ruta de éxito oficial: pending (recibido, pendiente de envío al upstream) → queued → in_progress → completed
ACTIVE_STATUSES = frozenset({"pending", "queued", "in_progress"})


class OFoxError(RuntimeError):
    """Errores deterministas de configuración, petición o respuesta."""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message)
        # En cuanto se crea la tarea remota, todos los tipos de error llevan el ID de la tarea. La capa superior no necesita
        # mantener lógica de recuperación según la subclase de excepción, y la WebUI/API pueden mostrar de forma estable la información para diagnosticar.
        self.task_id = task_id


class OFoxUnconfirmedTaskError(OFoxError):
    """El remoto puede haber creado ya una tarea de pago, pero esta máquina no puede confirmar su estado final."""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message, task_id=task_id)


class OFoxDownloadError(OFoxError):
    """La tarea de pago remota tuvo éxito, pero el video generado no se pudo descargar a esta máquina."""

    def __init__(self, message: str, task_id: str):
        super().__init__(message, task_id=task_id)


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    """
    Lee las credenciales de OFox con una prioridad clara y única.

    La clave específica del archivo de configuración tiene la máxima prioridad; la única variable de entorno de ejecución admitida es la de significado claro
    ``OFOX_API_KEY``.
    """
    settings = config.app if settings is None else settings
    configured = str(settings.get("ofox_api_key", "") or "").strip()
    environment_key = os.getenv("OFOX_API_KEY", "").strip()
    return configured or environment_key


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    return bool(get_api_key(settings))


def _base_url() -> str:
    return str(
        config.app.get("ofox_base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
    ).rstrip("/")


def _model_id() -> str:
    return str(
        config.app.get("ofox_text_to_video_model", DEFAULT_MODEL_ID) or DEFAULT_MODEL_ID
    ).strip()


def _resolution() -> str:
    """
    Lee la resolución de generación.

    OFox valida la resolución en el servidor según el modelo (por ejemplo, el modelo Seedance fast por defecto solo acepta
    480p/720p); un valor inválido recibe un rechazo 400 claro y no crea ninguna tarea de pago,
    así que aquí solo se quitan espacios, sin mantener una lista blanca local: la lista blanca caducaría cuando cambiara
    el catálogo de modelos remoto. Un valor vacío vuelve al valor por defecto, para no enviar una cadena vacía al remoto.
    """
    value = str(config.app.get("ofox_resolution", DEFAULT_RESOLUTION) or "").strip()
    return value or DEFAULT_RESOLUTION


def _provider_type() -> str:
    """
    Lee la fijación de proveedor upstream (provider routing).

    OFox Algunos modelos los suministran varios proveedores upstream (por ejemplo volcengine y
    bytepluspara la serie Seedance), y cada proveedor tiene su propia política de contenido y disponibilidad regional. Por defecto se fija byteplus
    (proveedor internacional, con una política de contenido más coherente para audiencias globales y enrutado predecible); si se configura otro nombre de proveedor
    se fija ese; si se configura explícitamente una cadena vacía no se fija y el gateway reparte por pesos. Un nombre de proveedor ilegal
    la API lo rechaza con un 400 ``invalid_provider_type`` específico sin crear ninguna tarea
    de pago, así que localmente no se mantiene una lista blanca de proveedores.
    """
    value = config.app.get("ofox_provider", DEFAULT_PROVIDER_TYPE)
    if value is None:
        return DEFAULT_PROVIDER_TYPE
    return str(value).strip()


def _config_bool(key: str, default: bool) -> bool:
    value = config.app.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _bounded_float(key: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(config.app.get(key, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return min(max(value, minimum), maximum)


def _duration_bounds() -> tuple[int, int]:
    def read(key: str, default: int) -> int:
        try:
            value = int(config.app.get(key, default))
        except (TypeError, ValueError, OverflowError):
            # TOML accepts `inf`; int(inf) raises OverflowError, which is not
            # a provider error and would abort the task with a traceback.
            return default
        return value if value >= 1 else default

    minimum = read("ofox_min_duration", DEFAULT_MIN_DURATION_SECONDS)
    maximum = read("ofox_max_duration", DEFAULT_MAX_DURATION_SECONDS)
    return minimum, max(minimum, maximum)


def _tls_verify() -> bool:
    return _config_bool("tls_verify", True)


def _status_code(response: Any) -> int:
    try:
        return int(getattr(response, "status_code", 200))
    except (TypeError, ValueError):
        return 200


def _redact_secret(value: Any, secret: str) -> str:
    text = str(value or "")
    if secret:
        text = text.replace(secret, "***")
        encoded = quote_plus(secret)
        if encoded != secret:
            text = text.replace(encoded, "***")
    for proxy_url in config.proxy.values():
        proxy_secret = str(proxy_url or "")
        if proxy_secret:
            text = text.replace(proxy_secret, "***")
    return text[:MAX_ERROR_TEXT_LENGTH]


def _response_error(response: Any, api_key: str) -> str:
    try:
        payload = response.json()
    except Exception:
        return f"HTTP {_status_code(response)}"
    if not isinstance(payload, dict):
        return f"HTTP {_status_code(response)}"
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code") or payload.get("code")
        message = error.get("message") or payload.get("message")
    else:
        code = payload.get("code")
        message = payload.get("message") or error
    detail = ": ".join(str(item) for item in (code, message) if item not in (None, ""))
    return _redact_secret(detail or f"HTTP {_status_code(response)}", api_key)


def _is_retryable_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True
    response = getattr(error, "response", None)
    return response is not None and _status_code(response) in RETRYABLE_STATUS_CODES


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    """Envía una tarea de texto a video de OFox y espera la dirección de un resultado descargable."""
    api_key = get_api_key()
    if not api_key:
        raise OFoxError("OFox video generation requires an OFox API key")

    term = str(search_term or "").strip()
    if not term:
        # Un prompt vacío puede deberse a un fallo en la división del guion en el upstream. Una fuente de generación de pago no puede enviarlo al remoto,
        # o aunque la interfaz acepte la petición solo se obtendría un video inutilizable y ya cobrado.
        raise OFoxError("OFox search term must not be empty")

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    try:
        requested_duration = max(int(minimum_duration), 1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OFoxError("OFox clip duration must be a positive integer") from exc
    minimum, maximum = _duration_bounds()
    duration = min(max(requested_duration, minimum), maximum)
    if duration != requested_duration:
        # Generar más de lo pedido no afecta al video final: el flujo de edición sigue recortando según la duración del clip; generar menos de lo pedido
        # menos solo ocurre cuando la petición supera el máximo del modelo, y en ese caso solo se puede ajustar al máximo.
        logger.info(
            f"ofox clip duration clamped to the configured model range: "
            f"requested={requested_duration}s, using={duration}s "
            f"(configured {minimum}-{maximum}s)"
        )
    payload = {
        "model": _model_id(),
        "prompt": term,
        "duration": duration,
        "resolution": _resolution(),
        "aspect_ratio": aspect.value,
    }
    provider_type = _provider_type()
    if provider_type:
        payload["provider"] = {"type": provider_type}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    videos_url = f"{_base_url()}/videos"
    logger.info(
        "generating video with OFox: "
        f"model={payload['model']}, term={term!r}, duration={duration}s"
    )

    # La interfaz de envío no se reintenta automáticamente: un tiempo agotado o un 5xx pueden ocurrir después de crear la tarea de pago,
    # y reintentar a ciegas causaría cobros duplicados. Solo se considera un fallo determinista cuando se obtiene una respuesta de rechazo clara.
    generation_progress.submitting('ofox')
    try:
        response = trace_http_call(
            "POST", requests.post,
            videos_url,
            json=payload,
            headers=headers,
            proxies=config.proxy,
            verify=_tls_verify(),
            timeout=(30, 60),
            allow_redirects=False,
        )
    except Exception as exc:
        raise OFoxUnconfirmedTaskError(
            "OFox submission returned no response; a paid task may already "
            "exist remotely: "
            f"error={type(exc).__name__}, detail={_redact_secret(exc, api_key)}"
        ) from exc

    status_code = _status_code(response)
    if 300 <= status_code < 400:
        raise OFoxUnconfirmedTaskError(
            "OFox submission returned a redirect; the paid task state is "
            "unknown and the request was not replayed"
        )
    if status_code >= 500:
        raise OFoxUnconfirmedTaskError(
            f"OFox submission failed with HTTP {status_code}; a paid task may "
            "already exist remotely"
        )
    if not 200 <= status_code < 300:
        # 4xx es un rechazo claro (como duration/resolution fuera del rango admitido por el modelo); el remoto
        # no creó la tarea y no hay riesgo de cobro duplicado; el mensaje de error incluye el
        # rango de valores válidos dado por el servidor, y se lanza directamente al usuario para que corrija la configuración.
        raise OFoxError(
            "OFox video generation request rejected: "
            f"HTTP {status_code}, {_response_error(response, api_key)}"
        )
    try:
        body = response.json()
    except Exception as exc:
        raise OFoxUnconfirmedTaskError(
            "OFox submission returned an unreadable response; a paid task may "
            f"already exist remotely: error={type(exc).__name__}"
        ) from exc
    task_id = str(body.get("id") or "").strip() if isinstance(body, dict) else ""
    if not task_id:
        raise OFoxUnconfirmedTaskError(
            "OFox accepted the submission without returning a task id"
        )
    logger.info(f"OFox video task created: id={task_id}")

    task = _wait_for_task(
        task_id=task_id,
        videos_url=videos_url,
        headers=headers,
        api_key=api_key,
    )
    if task is None:
        return []

    # La recomendación oficial es usar primero mirror_urls (dirección firmada persistente de la CDN de OFox, solo se devuelve si el upstream activa el espejo
    # ); si falta, se recurre a unsigned_urls (enlace directo temporal del upstream, que puede caducar en 24 horas
    # ). Ambas deben conservarse completas y usarse de inmediato para descargar, sin escribirse en el source_info a largo plazo.
    video_url = ""
    for field in ("mirror_urls", "unsigned_urls"):
        urls = task.get(field)
        for candidate in urls if isinstance(urls, list) else []:
            if isinstance(candidate, str) and candidate.startswith(
                ("http://", "https://")
            ):
                video_url = candidate
                break
        if video_url:
            break
    if not video_url:
        raise OFoxError(
            f"OFox task completed without a downloadable video: id={task_id}",
            task_id=task_id,
        )

    return [
        MaterialInfo(
            provider="ofox",
            url=video_url,
            duration=duration,
            source_info={
                "provider": "ofox",
                "search_term": term,
                "asset_id": task_id,
                "rendition": {
                    "id": task_id,
                    "width": video_width,
                    "height": video_height,
                },
            },
        )
    ]


def _wait_for_task(
    *,
    task_id: str,
    videos_url: str,
    headers: dict[str, str],
    api_key: str,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + _bounded_float(
        "ofox_run_timeout",
        DEFAULT_RUN_TIMEOUT_SECONDS,
        60.0,
        7200.0,
    )
    poll_interval = _bounded_float(
        "ofox_poll_interval",
        DEFAULT_POLL_INTERVAL_SECONDS,
        0.5,
        60.0,
    )
    generation_progress.remote('ofox', task_id)
    consecutive_failures = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OFoxUnconfirmedTaskError(
                "OFox task is still running after the configured local wait "
                f"timeout: id={task_id}",
                task_id=task_id,
            )

        # requests cronometra por separado connect/read timeout, así que cada uno usa la
        # mitad del tiempo total restante. Aunque la conexión y la lectura lleguen al límite, una ronda de petición no supera a propósito el plazo total;
        # la biblioteca de red puede tener un pequeño error de planificación, y la siguiente comprobación de deadline impide otro reintento.
        phase_timeout = max(min(remaining / 2.0, 30.0), 0.001)
        try:
            response = trace_http_call(
                "GET", requests.get,
                f"{videos_url}/{task_id}",
                headers=headers,
                proxies=config.proxy,
                verify=_tls_verify(),
                timeout=(phase_timeout, phase_timeout),
                allow_redirects=False,
            )
            status_code = _status_code(response)
            if status_code in RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(
                    f"HTTP {status_code}", response=response
                )
            if not 200 <= status_code < 300:
                raise OFoxUnconfirmedTaskError(
                    "OFox task status is unknown: "
                    f"http_status={status_code}, "
                    f"detail={_response_error(response, api_key)}",
                    task_id=task_id,
                )
            body = response.json()
            if not isinstance(body, dict):
                raise OFoxUnconfirmedTaskError(
                    "OFox task status response is malformed", task_id=task_id
                )
        except OFoxUnconfirmedTaskError:
            raise
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise OFoxUnconfirmedTaskError(
                    "OFox polling failed and the paid task state is unknown: "
                    f"error={type(exc).__name__}, "
                    f"detail={_redact_secret(exc, api_key)}",
                    task_id=task_id,
                ) from exc

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OFoxUnconfirmedTaskError(
                    "OFox task is still running after the configured local wait "
                    f"timeout: id={task_id}",
                    task_id=task_id,
                ) from exc
            consecutive_failures += 1
            generation_progress.remote('ofox', task_id, retrying=True)
            if consecutive_failures > MAX_POLL_RETRIES:
                raise OFoxUnconfirmedTaskError(
                    "OFox polling failed after retries; the paid task may still "
                    f"be running remotely: id={task_id}",
                    task_id=task_id,
                ) from exc
            delay = min(RETRY_BASE_SECONDS * consecutive_failures, remaining)
            logger.warning(
                "OFox polling hit a transient error; retrying the same task: "
                f"id={task_id}, attempt={consecutive_failures}/{MAX_POLL_RETRIES}, "
                f"retry_in={delay:.1f}s"
            )
            time.sleep(delay)
            continue

        consecutive_failures = 0
        status = str(body.get("status") or "").strip().lower()
        generation_progress.remote('ofox', task_id, status)
        if status in TERMINAL_SUCCESS_STATUSES:
            return body
        if status in TERMINAL_FAILURE_STATUSES:
            error_detail = body.get("error")
            logger.error(
                "OFox task did not produce a video: "
                f"id={task_id}, status={status}, "
                f"detail={_redact_secret(error_detail, api_key)}"
            )
            # Un fallo claro del remoto significa que la tarea terminó y se puede continuar con seguridad con los fragmentos siguientes;
            # el llamador decide si reintentar con otra palabra clave.
            return None
        if status not in ACTIVE_STATUSES:
            raise OFoxUnconfirmedTaskError(
                f"OFox returned an unknown task status: id={task_id}, "
                f"status={status!r}",
                task_id=task_id,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OFoxUnconfirmedTaskError(
                "OFox task is still running after the configured local wait "
                f"timeout: id={task_id}",
                task_id=task_id,
            )
        time.sleep(min(poll_interval, remaining))
