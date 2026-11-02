import base64
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import io
import math
import os
import random
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, List
from urllib.parse import quote_plus, urlencode, urlsplit, urlunsplit

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image, UnidentifiedImageError

from app.config import config
from app.models.schema import ImageMotion, MaterialInfo, VideoAspect, VideoConcatMode
from app.services import (
    codex_image,
    generation_progress,
    infistar_video,
    jalapeno_video,
    material_cache,
    material_upload,
    metaso_minimax,
    muapi,
    ofox,
    task_artifacts,
    video,
    volcengine_seedance,
)
from app.utils import logging_utils, utils

# Thread-safe counter for API key rotation
_api_key_counter = 0
_api_key_lock = threading.Lock()

# A provider URL can point to an unexpectedly large object or never-ending stream.
# Short stock and generated clips should stay well below this conservative cap.
MAX_VIDEO_DOWNLOAD_BYTES = 512 * 1024 * 1024

# Por defecto se mantiene en serie, igual que la versión anterior; si hace falta, se puede aumentar en la configuración la concurrencia de material de stock.
_DEFAULT_MATERIAL_CONCURRENCY = 1


def _get_material_concurrency() -> int:
    try:
        concurrency = int(config.app.get("material_concurrency", _DEFAULT_MATERIAL_CONCURRENCY))
    except (TypeError, ValueError):
        concurrency = _DEFAULT_MATERIAL_CONCURRENCY
    return max(1, min(8, concurrency))


class _OpenAIImageDecodeError(ValueError):
    """Indica que los bytes devueltos por la interfaz compatible no se pueden decodificar como imagen; no incluye fallos de escritura de archivos locales."""


class OpenAIImagePaidResultError(RuntimeError):
    """A paid image request cannot provide a usable local video material."""


class OpenAIImageUnconfirmedError(OpenAIImagePaidResultError):
    """A paid image request may have succeeded without returning a response."""


class CodexImageMaterialError(OpenAIImagePaidResultError):
    """Stop the image loop while retaining a translatable, sanitized error."""

    def __init__(self, error):
        self.key = error.key
        self.params = error.params
        messages = {
            "codex.image.save_failed": "Could not save the generated Codex image. Generation stopped without retrying.",
            "codex.image.render_failed": "Could not render the generated Codex image. Generation stopped without retrying.",
        }
        super().__init__(messages.get(self.key, str(error)))


def _safe_public_url(value: Any) -> str | None:
    """
    Solo conserva direcciones de páginas HTTP(S) que se puedan mostrar públicamente, eliminando los parámetros de consulta y las credenciales.

    Las direcciones de descarga del material pueden llevar API Keys, JWT firmados o tokens temporales. El listado de la tarea solo necesita
    ayudar al usuario a volver a la página pública del material del proveedor y no debe guardar parámetros de autenticación; la forma con información de usuario URL
    también se rechaza, para evitar que algo como ``https://user:pass@example.com`` se guarde en disco.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _creator_info(value: Any) -> dict[str, str] | None:
    """Extrae campos públicos unificados de las distintas estructuras de autor de cada proveedor."""
    if isinstance(value, str) and value.strip():
        return {"name": value.strip()}
    if not isinstance(value, dict):
        return None

    creator: dict[str, str] = {}
    creator_id = value.get("id")
    creator_name = value.get("name") or value.get("username")
    creator_page = _safe_public_url(
        value.get("url") or value.get("profile_url") or value.get("profile_page")
    )
    if creator_id is not None:
        creator["id"] = str(creator_id)
    if creator_name:
        creator["name"] = str(creator_name)
    if creator_page:
        creator["profile_page"] = creator_page
    return creator or None


def _material_source_record(item: MaterialInfo, local_path: str) -> dict[str, Any]:
    """
    Genera un registro de origen ligero para el material descargado con éxito.

    ``source_info`` puede venir de la caché o incluso de un ``MaterialInfo`` construido externamente, así que
    no se puede escribir tal cual. Aquí se reconstruye según una lista blanca, conservando solo la página pública, los identificadores de negocio y las dimensiones,
    y registrando solo el nombre de archivo local, para que el directorio del usuario o las rutas montadas por Docker no entren en el archivo de la tarea.
    """
    source = item.source_info if isinstance(item.source_info, dict) else {}
    record: dict[str, Any] = {
        "provider": str(item.provider or source.get("provider") or ""),
        "local_file": Path(local_path).name,
        "duration": int(item.duration),
    }

    search_term = source.get("search_term")
    asset_id = source.get("asset_id")
    source_page = _safe_public_url(source.get("source_page"))
    if isinstance(search_term, str) and search_term.strip():
        record["search_term"] = search_term.strip()
    if asset_id not in (None, ""):
        record["asset_id"] = str(asset_id)
    if source_page:
        record["source_page"] = source_page

    creator = _creator_info(source.get("creator"))
    if creator:
        record["creator"] = creator

    raw_rendition = source.get("rendition")
    if isinstance(raw_rendition, dict):
        rendition = {}
        for field in ("id", "width", "height"):
            value = raw_rendition.get(field)
            if value not in (None, ""):
                rendition[field] = str(value) if field == "id" else value
        if rendition:
            record["rendition"] = rendition
    return record


def _persist_material_sources(
    task_id: str,
    material_sources: list[dict[str, Any]],
) -> None:
    """
    Añade al listado de la tarea los orígenes del material realmente descargado con éxito.

    El registro de la tarea es una función auxiliar: no puede cambiar el valor devuelto por la función de descarga de video, ni, por un fallo de escritura en disco,
    interrumpir el flujo principal de composición. ``patch_script_data`` se encarga de la sustitución atómica y del log de excepciones; aquí solo
    se registra la cantidad tras el éxito, para confirmar que la información de trazabilidad de la tarea ya se guardó en disco.
    """
    try:
        saved = task_artifacts.patch_script_data(
            task_id,
            material_sources=material_sources,
        )
        if saved:
            logger.info(
                f"saved material source records: "
                f"task_id={task_id}, count={len(material_sources)}"
            )
    except Exception as exc:
        # task_artifacts ya está diseñado para degradarse ante fallos; aquí se mantiene un último aislamiento,
        # para evitar que futuros cambios de implementación o fallos de resolución de directorios afecten por accidente al valor devuelto por la descarga de material.
        logger.warning(
            "failed to persist material source records: "
            f"task_id={task_id}, error={type(exc).__name__}, detail={exc}"
        )


def _get_tls_verify() -> bool:
    # La validación de certificados TLS está activada por defecto, para evitar que la búsqueda y descarga de material sea manipulada por un intermediario.
    # Solo en casos claramente necesarios, como proxies corporativos o certificados autofirmados, se permite al usuario
    # `config.toml` desactivarla temporalmente con `tls_verify = false` de forma explícita.
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")

    if not tls_verify:
        logger.warning(
            "TLS certificate verification is disabled by config.app.tls_verify=false. "
            "Only use this in trusted proxy environments."
        )

    return bool(tls_verify)


def get_api_key(cfg_key: str):
    api_keys = config.app.get(cfg_key)
    if not api_keys:
        raise ValueError(
            f"\n\n##### {cfg_key} is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )

    # if only one key is provided, return it
    if isinstance(api_keys, str):
        return api_keys

    global _api_key_counter
    with _api_key_lock:
        _api_key_counter += 1
        return api_keys[_api_key_counter % len(api_keys)]


def _redact_secret(message: str, secret: str) -> str:
    """
    Anonimiza mínimamente el texto de excepción que se va a escribir en el log.

    requests Las excepciones de conexión pueden incluir la URL completa de la petición, y la API Key de Pixabay se pasa como parámetro
    de consulta. Aquí se reemplazan tanto el valor original como el codificado en URL, conservando la información del error de red para diagnosticar
    y evitando que la clave llegue a los archivos de log.
    """
    safe_message = str(message)
    if not secret:
        return safe_message

    safe_message = safe_message.replace(secret, "***")
    encoded_secret = quote_plus(secret)
    if encoded_secret != secret:
        safe_message = safe_message.replace(encoded_secret, "***")
    return safe_message


def _redact_request_error(error: Exception, *secrets: str) -> str:
    """
    Conserva la información diagnosticable de las excepciones de red, eliminando la API Key y las credenciales del proxy.

    Registrar solo el tipo de excepción pierde contexto clave como DNS, certificados o tiempos de espera; registrar la excepción original
    puede repetir la URL completa de la petición. Una entrada unificada permite que los tres proveedores de material usen las mismas reglas de anonimización.
    """
    safe_message = str(error)
    for secret in secrets:
        safe_message = _redact_secret(safe_message, str(secret or ""))
    for proxy_url in config.proxy.values():
        safe_message = _redact_secret(safe_message, str(proxy_url))
    return safe_message


def _is_cloudflare_challenge(response: requests.Response) -> bool:
    """
    Detecta el HTML Challenge que devuelve Cloudflare, en lugar de tratarlo como JSON de Pixabay.

    Cloudflare normalmente establece `cf-mitigated: challenge`; algunos despliegues solo devuelven HTML con
    "Just a moment" o challenge-platform, así que se conserva la detección por contenido como respaldo.
    El cuerpo de la respuesta solo se evalúa en memoria y no se escribe en el log, para no registrar grandes bloques de HTML sin valor.
    """
    headers = getattr(response, "headers", {}) or {}
    if str(headers.get("cf-mitigated", "")).lower() == "challenge":
        return True

    content_type = str(headers.get("content-type", "")).lower()
    if "text/html" not in content_type:
        return False

    body = str(getattr(response, "text", "")).lower()
    return "just a moment" in body or "/cdn-cgi/challenge-platform/" in body


def _stock_search_results(
    provider: str, response: Any, results_key: str
) -> list | None:
    """
    Return the result list of a Pexels or Coverr search response.

    When the search failed, log why and return None. As in the Pixabay search,
    the log gets the status and content type but not the response body, and a
    401 names the setting that holds the rejected key.
    """
    status_code = int(getattr(response, "status_code", 200))
    headers = getattr(response, "headers", {}) or {}
    content_type = str(headers.get("content-type", "")) or "unknown"
    if status_code == 401:
        logger.error(
            f"{provider} rejected the API key: status=401. "
            f"Check {provider}_api_keys in config.toml."
        )
        return None
    if status_code == 429:
        logger.error(
            f"{provider} API rate limit exceeded: "
            f"status=429, retry_after={headers.get('retry-after') or 'unknown'}"
        )
        return None
    if status_code >= 400:
        logger.error(
            f"{provider} search request failed: "
            f"status={status_code}, content_type={content_type}"
        )
        return None
    try:
        body = response.json()
    except ValueError:
        logger.error(
            f"{provider} returned an unexpected non-JSON response: "
            f"status={status_code}, content_type={content_type}"
        )
        return None
    if not isinstance(body, dict) or not isinstance(body.get(results_key), list):
        logger.error(f"{provider} video search returned an unsupported response")
        return None
    return body[results_key]


def _matches_video_aspect(
    width: Any,
    height: Any,
    video_aspect: VideoAspect,
    *,
    is_vertical: Any = None,
) -> bool:
    """
    Determina si el material remoto coincide con la orientación de imagen objetivo.

    Pexels, Pixabay y Coverr no tienen campos de respuesta uniformes, así que primero se usa el ancho y alto para una evaluación fiable;
    Coverr cuando a algunas respuestas antiguas les faltan las dimensiones, se recurre al booleano explícito ``is_vertical``.
    El material cuya orientación no se puede confirmar se omite, para que las tareas verticales no mezclen material horizontal y produzcan bandas negras en el video final.
    """
    aspect = VideoAspect(video_aspect)
    try:
        normalized_width = int(float(width))
        normalized_height = int(float(height))
    except (OverflowError, TypeError, ValueError):
        normalized_width = 0
        normalized_height = 0

    if normalized_width > 0 and normalized_height > 0:
        if aspect == VideoAspect.portrait:
            return normalized_height > normalized_width
        if aspect == VideoAspect.landscape:
            return normalized_width > normalized_height
        return normalized_width == normalized_height

    if isinstance(is_vertical, bool) and aspect != VideoAspect.square:
        return is_vertical == (aspect == VideoAspect.portrait)
    return False


def _filter_materials_by_aspect(
    items: List[MaterialInfo],
    video_aspect: VideoAspect,
) -> List[MaterialInfo]:
    """
    Vuelve a validar la orientación de los resultados en caché.

    La caché de búsqueda de material se conserva hasta 24 horas, y la caché escrita antes de actualizar puede contener material con orientación incorrecta.
    Filtrar en la entrada unificada de caché hace que la corrección se aplique de inmediato y también protege frente a Providers de terceros o caché antigua
    que omitieran el filtrado remoto. Las entradas antiguas cuyas dimensiones de rendition no se pueden leer se tratan como no verificadas y se omiten.
    """
    aspect = VideoAspect(video_aspect)
    if aspect == VideoAspect.square:
        # Pixabay y Coverr rara vez ofrecen material cuadrado nativo. La salida cuadrada mantiene el comportamiento existente:
        # acepta los candidatos disponibles y deja el recorte a la fase de composición del video, para que tras actualizar las tareas 1:1 no se queden sin material.
        return list(items)

    filtered_items = []
    for item in items:
        source_info = item.source_info if isinstance(item.source_info, dict) else {}
        rendition = source_info.get("rendition")
        rendition = rendition if isinstance(rendition, dict) else {}
        if _matches_video_aspect(
            rendition.get("width"),
            rendition.get("height"),
            aspect,
        ):
            filtered_items.append(item)
    return filtered_items


def search_videos_pexels(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    api_key = get_api_key("pexels_api_keys")
    headers = {
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }
    # Build URL
    params = {"query": search_term, "per_page": 20}
    if aspect != VideoAspect.square:
        params["orientation"] = aspect.name
    query_url = f"https://api.pexels.com/v1/videos/search?{urlencode(params)}"
    logger.info(f"searching videos on pexels: term={search_term!r}")

    try:
        r = trace_http_call(
            "GET", requests.get,
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        videos = _stock_search_results("pexels", r, "videos")
        if videos is None:
            return []
        video_items = []
        # loop through each video in the result
        for v in videos:
            if not isinstance(v, dict):
                continue
            duration = v.get("duration")
            # check if video has desired minimum duration
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < minimum_duration
            ):
                continue
            video_files = v.get("video_files")
            if not isinstance(video_files, list):
                continue
            # Prefer the smallest rendition that can fill the output canvas.
            # Square output can crop either orientation, as other providers do.
            renditions = []
            for video in video_files:
                if not isinstance(video, dict):
                    continue
                try:
                    w = int(video.get("width"))
                    h = int(video.get("height"))
                except (OverflowError, TypeError, ValueError):
                    continue
                video_url = video.get("link")
                if not isinstance(video_url, str) or not video_url:
                    continue
                if (
                    (aspect == VideoAspect.square or _matches_video_aspect(w, h, aspect))
                    and w >= video_width
                    and h >= video_height
                ):
                    renditions.append((w * h, video, w, h, video_url))
            if renditions:
                _, video, w, h, video_url = min(renditions, key=lambda candidate: candidate[0])
                item = MaterialInfo()
                item.provider = "pexels"
                item.url = video_url
                item.duration = duration
                item.source_info = {
                    "provider": "pexels",
                    "search_term": search_term,
                    "asset_id": (
                        str(v.get("id")) if v.get("id") is not None else None
                    ),
                    "source_page": _safe_public_url(v.get("url")),
                    "creator": _creator_info(v.get("user")),
                    "rendition": {
                        "id": (
                            str(video.get("id"))
                            if video.get("id") is not None
                            else None
                        ),
                        "width": w,
                        "height": h,
                    },
                }
                video_items.append(item)
        return video_items
    except Exception as e:
        logger.error(
            "pexels video search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )

    return []


def search_videos_pixabay(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)

    video_width, video_height = aspect.to_resolution()

    api_key = get_api_key("pixabay_api_keys")
    # Build URL
    params = {
        "q": search_term,
        "video_type": "all",  # Accepted values: "all", "film", "animation"
        "per_page": 50,
        "key": api_key,
    }
    query_url = f"https://pixabay.com/api/videos/?{urlencode(params)}"
    logger.info(
        f"searching videos on pixabay: term={search_term!r}, "
        f"proxy_enabled={bool(config.proxy)}"
    )

    try:
        r = trace_http_call(
            "GET", requests.get,
            query_url, proxies=config.proxy, verify=_get_tls_verify(), timeout=(30, 60)
        )
        status_code = int(getattr(r, "status_code", 200))
        headers = getattr(r, "headers", {}) or {}
        content_type = str(headers.get("content-type", ""))
        retry_after = headers.get("retry-after")
        cf_ray = headers.get("cf-ray")

        if _is_cloudflare_challenge(r):
            logger.error(
                "pixabay search was blocked by a Cloudflare challenge: "
                f"status={status_code}, cf_ray={cf_ray or 'unknown'}. "
                "Check the server network or proxy, or use Pexels/Coverr instead."
            )
            return []

        if status_code == 429:
            logger.error(
                "pixabay API rate limit exceeded: "
                f"status=429, retry_after={retry_after or 'unknown'}"
            )
            return []

        if status_code >= 400:
            logger.error(
                "pixabay search request failed: "
                f"status={status_code}, content_type={content_type or 'unknown'}"
            )
            return []

        try:
            response = r.json()
        except ValueError:
            logger.error(
                "pixabay returned an unexpected non-JSON response: "
                f"status={status_code}, content_type={content_type or 'unknown'}"
            )
            return []

        video_items = []
        if not isinstance(response, dict) or not isinstance(
            response.get("hits"), list
        ):
            logger.error("pixabay video search returned an unsupported response")
            return video_items
        videos = response["hits"]
        # loop through each video in the result
        for v in videos:
            if not isinstance(v, dict):
                continue
            duration = v.get("duration")
            # check if video has desired minimum duration
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < minimum_duration
            ):
                continue
            video_files = v.get("videos")
            if not isinstance(video_files, dict):
                continue
            # loop through each url to determine the best quality
            for video_type, video in video_files.items():
                if not isinstance(video, dict):
                    continue
                try:
                    w = int(video["width"])
                    h = int(video["height"])
                except (KeyError, OverflowError, TypeError, ValueError):
                    continue
                video_url = video.get("url")
                if not isinstance(video_url, str) or not video_url:
                    continue
                # Pixabay rara vez devuelve videos cuadrados nativos; la salida 1:1 sigue aceptando candidatos que cumplan la resolución
                # y la fase de composición los recorta. En horizontal y vertical debe coincidir estrictamente la orientación objetivo.
                orientation_matches = aspect == VideoAspect.square or (
                    _matches_video_aspect(w, h, aspect)
                )
                if orientation_matches and w >= video_width and h >= video_height:
                    item = MaterialInfo()
                    item.provider = "pixabay"
                    item.url = video_url
                    item.duration = duration
                    item.source_info = {
                        "provider": "pixabay",
                        "search_term": search_term,
                        "asset_id": (
                            str(v.get("id")) if v.get("id") is not None else None
                        ),
                        "source_page": _safe_public_url(v.get("pageURL")),
                        "creator": _creator_info(
                            {
                                "id": v.get("user_id"),
                                "name": v.get("user"),
                            }
                        ),
                        "rendition": {
                            "id": video_type,
                            "width": w,
                            "height": video.get("height"),
                        },
                    }
                    video_items.append(item)
                    break
        return video_items
    except Exception as e:
        error_message = _redact_request_error(e, api_key)
        logger.error(
            "pixabay search request failed: "
            f"error={type(e).__name__}, detail={error_message}"
        )

    return []


def search_videos_coverr(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    """
    Coverr (https://coverr.co) - free HD/4K stock videos,
    subject to Coverr license terms (https://coverr.co/license).

    Coverr API notes (based on official docs at api.coverr.co/docs/):
      - Autenticación: Authorization: Bearer <api_key>
      - Endpoint de búsqueda: GET /videos?query=..., estructura de la respuesta {"hits": [...], ...}
      - añadir ?urls=true devuelve directamente en la respuesta de búsqueda el enlace directo mp4
      - URL es un JWT firmado (ligado a la API key, sin caducidad)
      - Coverr admite filtrar material horizontal/vertical con filter=is_vertical:true/false;
        tras recibir la respuesta se valida localmente según max_width/max_height o is_vertical
      - duration el campo existe en dos formas, number y string; esta función acepta ambas

    esta función usa el campo urls.mp4_download como dirección de descarga; según la documentación oficial de Coverr
    (https://api.coverr.co/docs/videos/#download-a-video) según indica,
    GET Coverr cuenta esta misma URL como un evento de descarga legítimo en sus estadísticas,
    así que no hace falta llamar a PATCH /videos/:id/stats/downloads.
    """
    aspect = VideoAspect(video_aspect)
    api_key = get_api_key("coverr_api_keys")
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "query": search_term,
        "page_size": 20,
        "urls": "true",
        "sort": "popular",
    }
    # El filtrado de orientación en el servidor puede devolver el material objetivo directamente desde todos los resultados de búsqueda, evitando obtener primero los más populares
    # y filtrar localmente dejando vacíos los candidatos verticales. El material cuadrado no tiene condición booleana equivalente y sigue dependiendo de la validación local de ancho y alto.
    if aspect == VideoAspect.portrait:
        params["filter"] = "is_vertical:true"
    elif aspect == VideoAspect.landscape:
        params["filter"] = "is_vertical:false"
    query_url = f"https://api.coverr.co/videos?{urlencode(params)}"
    logger.info(f"searching videos on coverr: term={search_term!r}")

    try:
        r = trace_http_call(
            "GET", requests.get,
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        hits = _stock_search_results("coverr", r, "hits")
        if hits is None:
            return []
        video_items: List[MaterialInfo] = []

        for v in hits:
            if not isinstance(v, dict):
                continue
            # duration en distintas respuestas puede ser number(11.625) o string("10.500000")
            try:
                duration = int(float(v.get("duration") or 0))
            except (OverflowError, TypeError, ValueError):
                continue
            if duration < minimum_duration:
                continue

            video_id = v.get("id")
            urls = v.get("urls")
            if not isinstance(urls, dict):
                continue
            mp4_download_url = urls.get("mp4_download")
            if (
                not video_id
                or not isinstance(mp4_download_url, str)
                or not mp4_download_url
            ):
                continue
            if aspect != VideoAspect.square and not _matches_video_aspect(
                v.get("max_width"),
                v.get("max_height"),
                aspect,
                is_vertical=v.get("is_vertical"),
            ):
                continue

            item = MaterialInfo()
            item.provider = "coverr"
            item.url = mp4_download_url
            item.duration = duration
            item.source_info = {
                "provider": "coverr",
                "search_term": search_term,
                "asset_id": str(video_id),
                "source_page": _safe_public_url(v.get("canonical_url") or v.get("url")),
                "creator": _creator_info(v.get("creator") or v.get("author")),
                "rendition": {
                    "id": "mp4_download",
                    "width": v.get("max_width"),
                    "height": v.get("max_height"),
                },
            }
            video_items.append(item)
        return video_items
    except Exception as e:
        logger.error(
            "coverr video search failed: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        )

    return []


# WaveSpeed AI (https://wavespeed.ai) Genera material directamente con un modelo de texto a video a partir de las palabras clave del guion,
# compartiendo con las tres fuentes de material de stock la estructura de resultados MaterialInfo y el flujo posterior de descarga y edición.
WAVESPEED_API_BASE_URL = "https://api.wavespeed.ai/api/v3"
WAVESPEED_DEFAULT_T2V_MODEL = "bytedance/seedance-2.0-fast/text-to-video"
WAVESPEED_POLL_INTERVAL_SECONDS = 2.0
WAVESPEED_RUN_TIMEOUT_SECONDS = 600.0
# El modelo por defecto bytedance/seedance-2.0-fast/text-to-video solo acepta 4-15 segundos; las peticiones fuera
# de ese rango las rechaza directamente la API. La duración de clip por defecto de la WebUI es de 3 segundos, así que antes de enviar
# hay que ajustarla al intervalo admitido por el modelo; el tiempo sobrante lo recorta el flujo de edición existente según la duración del clip.
WAVESPEED_MIN_DURATION_SECONDS = 4
WAVESPEED_MAX_DURATION_SECONDS = 15
# Los tres estados de fallo tienen significados distintos (error del modelo / cancelación del usuario / tiempo agotado en la plataforma), pero para el flujo de material todos significan
# que esta palabra clave no tiene resultado, así que se tratan como resultado vacío y la capa superior omite ese clip y sigue generando.
WAVESPEED_FAILURE_STATUSES = frozenset({"failed", "cancelled", "timeout"})
# Mismo criterio que el SDK oficial de Python de WaveSpeed / nodo de n8n: 429 y 5xx son fallos
# temporales que merecen un número limitado de reintentos con espera; 4xx son errores claros del cliente y fallan rápido.
WAVESPEED_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Número de fallos temporales consecutivos permitidos en un sondeo. Un GET con mala suerte no puede hacer perder una tarea ya cobrada.
WAVESPEED_MAX_POLL_RETRIES = 5
# Base de la espera lineal: el reintento n espera base * n segundos.
WAVESPEED_RETRY_BASE_SECONDS = 1.0
# Número de reintentos sobre la misma dirección firmada cuando falla la descarga del resultado. El material ya se generó pagando, así que se prioriza reintentar
# la dirección original, en lugar de volver a enviar una tarea de generación de pago por una sola descarga inestable.
WAVESPEED_MAX_DOWNLOAD_RETRIES = 2


class WaveSpeedUnconfirmedTaskError(RuntimeError):
    """
    La tarea de generación de pago ya se envió, pero su estado final no se puede confirmar localmente.

    Esta excepción nunca equivale a «la tarea falló y se puede repetir»: la tarea remota puede seguir ejecutándose o haber
    terminado y cobrado. El flujo de material debe detenerse aquí, sin enviar nuevas tareas de pago para las palabras clave siguientes,
    y pasar el prediction id ya enviado al estado de la tarea para recuperarlo manualmente.
    """

    def __init__(self, message: str, prediction_id: str = ""):
        super().__init__(message)
        self.prediction_id = prediction_id


class WaveSpeedDownloadError(RuntimeError):
    """A paid prediction completed, but its video could not be saved locally."""

    def __init__(self, message: str, prediction_id: str = ""):
        super().__init__(message)
        self.prediction_id = prediction_id


def _wavespeed_status_code(response: Any) -> int:
    """Lee el código de estado de la respuesta; si el doble de prueba o el objeto de excepción no tiene ese campo, se trata como 200."""
    try:
        return int(getattr(response, "status_code", 200))
    except (TypeError, ValueError):
        return 200


def _is_wavespeed_retryable_error(error: Exception) -> bool:
    """
    Determina si vale la pena reintentar una excepción de sondeo.

    Las excepciones de red como conexión o tiempo de espera no tienen código de estado y se tratan como fallos temporales; las respuestas con código de estado solo se reintentan con
    429 y 5xx, igual que el conjunto de reintentos del SDK oficial.
    """
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
    if response is not None:
        return _wavespeed_status_code(response) in WAVESPEED_RETRYABLE_STATUS_CODES
    return False


def _wavespeed_duration_bounds() -> tuple[int, int]:
    """
    Devuelve el intervalo de duración de generación (segundos) que admite el modelo actual.

    El intervalo por defecto corresponde al modelo Seedance por defecto; si el usuario cambia a otro modelo de texto a video, puede
    ajustar el intervalo en la configuración. Cualquier configuración anómala vuelve al valor por defecto, garantizando min <= max,
    para no convertir la entrada del usuario en una petición remota que fallará seguro.
    """

    def read_bound(key: str, fallback: int) -> int:
        try:
            value = int(config.app.get(key, fallback))
        except (TypeError, ValueError):
            return fallback
        return value if value >= 1 else fallback

    min_duration = read_bound("wavespeed_min_duration", WAVESPEED_MIN_DURATION_SECONDS)
    max_duration = read_bound("wavespeed_max_duration", WAVESPEED_MAX_DURATION_SECONDS)
    return min_duration, max(max_duration, min_duration)


def generate_videos_wavespeed(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    """
    Genera un clip de material para una palabra clave del guion con un modelo de texto a video de WaveSpeed.

    Mantiene la misma firma y la convención de lista vacía ante fallos que search_videos_* de las fuentes de stock,
    para poder conectarse directamente al flujo genérico de descarga y cálculo de duración de ``download_videos``.
    ``minimum_duration`` en el contexto de generación es la duración del clip objetivo (segundos).
    """
    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    api_key = get_api_key("wavespeed_api_keys")
    model_id = (
        str(
            config.app.get("wavespeed_text_to_video_model", "")
            or WAVESPEED_DEFAULT_T2V_MODEL
        )
        .strip()
        .strip("/")
    )
    headers = {"Authorization": f"Bearer {api_key}"}
    requested_duration = max(int(minimum_duration), 1)
    min_duration, max_duration = _wavespeed_duration_bounds()
    duration = min(max(requested_duration, min_duration), max_duration)
    if duration != requested_duration:
        # Generar más de lo pedido no afecta al video final: el flujo de edición sigue recortando según la duración del clip; generar menos de lo pedido
        # solo ocurre cuando la petición supera el máximo del modelo, y en ese caso solo se puede ajustar al máximo.
        logger.info(
            f"wavespeed clip duration clamped to model-supported range: "
            f"requested={requested_duration}s, using={duration}s "
            f"(supported {min_duration}-{max_duration}s)"
        )
    payload = {
        "prompt": search_term,
        "aspect_ratio": aspect.value,
        "duration": duration,
    }
    logger.info(
        f"generating video on wavespeed: model={model_id}, "
        f"term={search_term!r}, duration={duration}s"
    )

    # El POST de envío nunca se reintenta automáticamente: la petición puede haber creado ya una tarea de pago en el remoto, y reenviarla causaría
    # generación y cobro duplicados (igual que la estrategia de submission del SDK oficial).
    generation_progress.submitting("wavespeed")
    try:
        submit_response = trace_http_call(
            "POST", requests.post,
            f"{WAVESPEED_API_BASE_URL}/{model_id}",
            json=payload,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
    except Exception as e:
        # No recibir respuesta no significa que la tarea no se haya creado. En ese momento el estado es incierto y hay que abortar todo el flujo
        # de generación, en lugar de seguir enviando nuevas tareas de pago para la siguiente palabra clave.
        raise WaveSpeedUnconfirmedTaskError(
            "wavespeed submission did not return a response, the task may "
            "already exist remotely: "
            f"error={type(e).__name__}, detail={_redact_request_error(e, api_key)}"
        ) from e

    submit_status = _wavespeed_status_code(submit_response)
    if submit_status >= 500:
        # 5xx puede ocurrir después de crear la tarea, sin poder saber si ya se cobró.
        raise WaveSpeedUnconfirmedTaskError(
            f"wavespeed submission failed with HTTP {submit_status}, "
            "the task may already exist remotely"
        )
    try:
        submit_body = submit_response.json()
    except Exception as e:
        if 400 <= submit_status < 500:
            # Los 4xx devueltos por el gateway o la capa de autenticación pueden ser texto plano/HTML. Un 4xx sigue siendo un rechazo claro,
            # el remoto no creó la tarea, y no se debe informar por error como «posiblemente cobrado» abortando toda la tarea.
            logger.error(
                "wavespeed video generation request rejected: "
                f"http_status={submit_status}, unreadable body: {type(e).__name__}"
            )
            return []
        raise WaveSpeedUnconfirmedTaskError(
            "wavespeed submission returned an unreadable response, the task "
            f"may already exist remotely: error={type(e).__name__}"
        ) from e

    submit_data = submit_body.get("data") if isinstance(submit_body, dict) else None
    if not isinstance(submit_body, dict) or submit_body.get("code") != 200:
        # 4xx y los códigos de error de negocio son rechazos claros; el remoto no creó la tarea y no hay riesgo de cobro
        # duplicado, así que se devuelve un resultado vacío según la convención de las fuentes de material y se continúa.
        logger.error(
            "wavespeed video generation request rejected: "
            f"http_status={submit_status}, "
            f"code={submit_body.get('code') if isinstance(submit_body, dict) else None}, "
            f"detail={_redact_secret(str((submit_body or {}).get('message') or ''), api_key)}"
        )
        return []
    prediction_id = (
        str(submit_data.get("id") or "") if isinstance(submit_data, dict) else ""
    )
    if not prediction_id:
        # El envío fue aceptado pero no se obtuvo ID: la tarea puede existir sin poder rastrearse, así que no se puede seguir pidiendo.
        raise WaveSpeedUnconfirmedTaskError(
            "wavespeed accepted the submission without returning a prediction id"
        )
    # El envío exitoso de la tarea de generación produce un cobro remoto, así que primero se registra el ID de la tarea en el log;
    # aunque el sondeo posterior falle, el usuario puede recuperar el resultado en la consola de WaveSpeed con ese ID.
    logger.info(f"wavespeed prediction created: id={prediction_id}")

    result_data = _wait_for_wavespeed_prediction(
        prediction_id=prediction_id,
        headers=headers,
        api_key=api_key,
    )
    if result_data is None:
        return []

    try:
        video_items = []
        outputs = result_data.get("outputs")
        for output in outputs if isinstance(outputs, list) else []:
            # La URL del resultado es una dirección de descarga temporal firmada y debe conservarse completa (no se pueden quitar los parámetros de
            # consulta), así que no se escribe en source_info y solo se usa para la descarga inmediata.
            if not isinstance(output, str) or not output.startswith(
                ("http://", "https://")
            ):
                continue
            item = MaterialInfo()
            item.provider = "wavespeed"
            item.url = output
            item.duration = duration
            item.source_info = {
                "provider": "wavespeed",
                "search_term": search_term,
                "asset_id": prediction_id,
                "rendition": {
                    "id": None,
                    "width": video_width,
                    "height": video_height,
                },
            }
            video_items.append(item)
        if not video_items:
            logger.error(
                "wavespeed prediction completed without downloadable outputs: "
                f"id={prediction_id}"
            )
        return video_items
    except Exception as e:
        # El resultado ya se generó y se cobró; aquí las excepciones solo pueden venir del análisis local. Se registran y se devuelve un resultado vacío
        # para que la capa superior omita ese clip; el estado de la tarea en sí es definitivo y se puede continuar con los clips siguientes.
        logger.error(
            "wavespeed output parsing failed: "
            f"id={prediction_id}, error={type(e).__name__}, "
            f"detail={_redact_request_error(e, api_key)}"
        )

    return []


def _wait_for_wavespeed_prediction(
    *,
    prediction_id: str,
    headers: dict,
    api_key: str,
) -> dict | None:
    """
    Sondea el mismo prediction id hasta obtener un resultado definitivo.

    Devuelve los data de ``completed``; cuando el remoto falla claramente (failed / cancelled / timeout)
    devuelve None, indicando que la tarea terminó y se puede continuar con seguridad con los clips siguientes. Los fallos temporales
    se reintentan sobre el mismo ID con espera lineal, sin volver a enviar nunca la tarea; si el estado nunca se puede confirmar se lanza
    :class:`WaveSpeedUnconfirmedTaskError`, y el llamador aborta todo el flujo de generación.
    """
    deadline = time.monotonic() + WAVESPEED_RUN_TIMEOUT_SECONDS
    generation_progress.remote("wavespeed", prediction_id)
    consecutive_failures = 0
    while True:
        try:
            response = trace_http_call(
                "GET", requests.get,
                f"{WAVESPEED_API_BASE_URL}/predictions/{prediction_id}/result",
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 60),
            )
            status_code = _wavespeed_status_code(response)
            if status_code in WAVESPEED_RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(
                    f"HTTP {status_code}", response=response
                )
            result_body = response.json()
            result_data = (
                result_body.get("data") if isinstance(result_body, dict) else None
            )
            if not isinstance(result_body, dict) or result_body.get("code") != 200:
                # Cuando el sondeo es rechazado claramente (como un 4xx), el estado de la tarea sigue siendo desconocido: la tarea ya se envió,
                # solo que localmente no se encuentra el resultado, así que tampoco se pueden enviar nuevas tareas de pago.
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction status is unknown: "
                    f"http_status={status_code}, "
                    f"code={result_body.get('code') if isinstance(result_body, dict) else None}, "
                    f"detail={_redact_secret(str((result_body or {}).get('message') or ''), api_key)}",
                    prediction_id=prediction_id,
                )
            if not isinstance(result_data, dict):
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction result payload is malformed",
                    prediction_id=prediction_id,
                )
        except WaveSpeedUnconfirmedTaskError:
            raise
        except Exception as e:
            if not _is_wavespeed_retryable_error(e):
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction polling failed and the task state is "
                    f"unknown: error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, api_key)}",
                    prediction_id=prediction_id,
                ) from e
            consecutive_failures += 1
            generation_progress.remote("wavespeed", prediction_id, retrying=True)
            if consecutive_failures > WAVESPEED_MAX_POLL_RETRIES:
                raise WaveSpeedUnconfirmedTaskError(
                    "wavespeed prediction polling failed after "
                    f"{WAVESPEED_MAX_POLL_RETRIES + 1} attempts, the task may "
                    "still be running remotely: "
                    f"error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, api_key)}",
                    prediction_id=prediction_id,
                ) from e
            delay = WAVESPEED_RETRY_BASE_SECONDS * consecutive_failures
            logger.warning(
                "wavespeed prediction polling hit a transient error, retry the "
                f"same task: id={prediction_id}, "
                f"attempt={consecutive_failures}/{WAVESPEED_MAX_POLL_RETRIES}, "
                f"error={type(e).__name__}, retry_in={delay:.1f}s"
            )
            time.sleep(delay)
            continue

        # Al obtener una respuesta válida se reinicia el contador; solo los fallos consecutivos consumen la cuota de reintentos.
        consecutive_failures = 0
        status = str(result_data.get("status") or "")
        generation_progress.remote("wavespeed", prediction_id, status)
        if status == "completed":
            return result_data
        if status in WAVESPEED_FAILURE_STATUSES:
            logger.error(
                "wavespeed prediction did not produce a video: "
                f"id={prediction_id}, status={status}, "
                f"detail={_redact_secret(str(result_data.get('error') or ''), api_key)}"
            )
            return None
        if time.monotonic() > deadline:
            # La tarea remota sigue ejecutándose y su estado final no se puede confirmar localmente; hay que dejar de hacer pedidos.
            raise WaveSpeedUnconfirmedTaskError(
                f"wavespeed prediction is still {status or 'pending'} after "
                f"{WAVESPEED_RUN_TIMEOUT_SECONDS:.0f}s of local waiting",
                prediction_id=prediction_id,
            )
        time.sleep(WAVESPEED_POLL_INTERVAL_SECONDS)


def _save_generated_video_with_retry(
    video_url: str, save_dir: str, provider: str
) -> str:
    """
    Descarga el resultado ya generado y pagado; si falla, se reintenta primero la misma dirección.

    Volver a generar una tarea remota cuesta pagar otra vez, así que una descarga inestable debe reintentarse primero sobre la dirección original
    un número limitado de veces con espera; al agotar los reintentos el llamador informa de un fallo recuperable de tarea de pago.
    """
    for attempt in range(WAVESPEED_MAX_DOWNLOAD_RETRIES + 1):
        try:
            saved_video_path = save_video(video_url=video_url, save_dir=save_dir)
            if saved_video_path:
                return saved_video_path
            failure_detail = "empty result"
        except Exception as e:
            failure_detail = (
                f"error={type(e).__name__}, "
                f"detail={_redact_request_error(e, video_url)}"
            )
        if attempt >= WAVESPEED_MAX_DOWNLOAD_RETRIES:
            break
        delay = WAVESPEED_RETRY_BASE_SECONDS * (attempt + 1)
        logger.warning(
            "failed to download generated video, retry the same url: "
            f"provider={provider}, "
            f"attempt={attempt + 1}/{WAVESPEED_MAX_DOWNLOAD_RETRIES}, "
            f"{failure_detail}, retry_in={delay:.1f}s"
        )
        time.sleep(delay)
    logger.error(
        "failed to download generated video after "
        f"{WAVESPEED_MAX_DOWNLOAD_RETRIES + 1} attempts: "
        f"provider={provider}, {failure_detail}"
    )
    return ""


def _get_downloaded_video_duration(video_path: str) -> float:
    """Read the usable duration from the downloaded media file."""
    clip = None
    try:
        clip = VideoFileClip(video_path)
        duration = float(clip.duration or 0)
    except Exception as exc:
        raise ValueError(
            f"downloaded video duration could not be measured: {video_path}"
        ) from exc
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception as close_error:
                logger.warning(
                    "failed to close downloaded video after duration probe: "
                    f"path={video_path}, error={type(close_error).__name__}, "
                    f"detail={close_error}"
                )
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            f"downloaded video duration is not positive and finite: {video_path}"
        )
    return duration


# requests Durante la descarga en streaming no se produce ningún log. En redes lentas un solo material 4K tarda varios minutos en descargarse,
# y la tarea parece colgada. Si pasado este intervalo aún no ha terminado, se registran el tamaño descargado y la velocidad; los archivos pequeños que
# terminan dentro del intervalo no producen logs adicionales, para no inundar el log.
_DOWNLOAD_HEARTBEAT_SECONDS = 15.0
_BYTES_PER_MEGABYTE = 1024 * 1024


def _describe_download_progress(
    downloaded_bytes: int, declared_size: int, elapsed_seconds: float
) -> str:
    """Formatea el progreso de descarga como texto de log; si el servidor no declara el tamaño no se da porcentaje."""
    downloaded_mb = downloaded_bytes / _BYTES_PER_MEGABYTE
    speed = downloaded_mb / elapsed_seconds if elapsed_seconds > 0 else 0.0
    if declared_size > 0:
        percent = min(100, round(downloaded_bytes * 100 / declared_size))
        return (
            f"{downloaded_mb:.1f} of {declared_size / _BYTES_PER_MEGABYTE:.1f} MB "
            f"({percent}%), {speed:.2f} MB/s"
        )
    return f"{downloaded_mb:.1f} MB, {speed:.2f} MB/s"


def _is_valid_video_file(video_path: str) -> bool:
    """Return whether a video satisfies the cache publication contract."""
    clip = None
    try:
        # Decode the complete stream first. Unlike MoviePy's metadata probe,
        # this validator distinguishes confirmed bad media from an FFmpeg
        # execution failure through MaterialUploadError/MaterialServiceError.
        material_upload._validate_video(video_path)
    except material_upload.MaterialServiceError:
        raise
    except material_upload.MaterialUploadError as exc:
        logger.warning(f"invalid video file: {video_path} => {str(exc)}")
        return False
    except Exception as exc:
        raise material_upload.MaterialServiceError(
            "unexpected failure while validating video content"
        ) from exc

    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        fps = clip.fps
        return bool(duration > 0 and fps > 0)
    except Exception as exc:
        # Complete decoding already succeeded. A MoviePy failure at this point
        # is an unavailable metadata probe (for example, its configured FFmpeg
        # executable is missing), not proof that the cached bytes are corrupt.
        raise material_upload.MaterialServiceError(
            "failed to probe video metadata"
        ) from exc
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception as close_error:
                logger.warning(
                    f"failed to close video clip: {video_path}, "
                    f"error: {str(close_error)}"
                )


def save_video(video_url: str, save_dir: str = "") -> str:
    generation_progress.downloading()
    if not save_dir:
        save_dir = utils.storage_dir("cache_videos")

    os.makedirs(save_dir, exist_ok=True)

    # Query parameters can identify the asset itself (for example,
    # /download?file_id=123). Dropping the query makes unrelated paid videos
    # share one cache entry and silently reuse the first scene. Fragments are
    # not sent in HTTP requests, so they do not affect the downloaded bytes.
    url_hash = utils.md5(video_url.split("#", 1)[0])
    video_id = f"vid-{url_hash}"
    video_path = f"{save_dir}/{video_id}.mp4"

    # A nonempty path can still be a truncated file left by an older release,
    # a manual copy, or storage corruption. Validate it before declaring a hit
    # so the cache can repair itself instead of failing on every later render.
    try:
        cached_stat = os.stat(video_path)
    except FileNotFoundError:
        cached_stat = None
    if cached_stat is not None and cached_stat.st_size > 0:
        generation_progress.verifying()
        try:
            cached_video_is_valid = _is_valid_video_file(video_path)
        except material_upload.MaterialServiceError as exc:
            # Do not destroy the only local copy because the validation service
            # is temporarily unavailable. Reusing the existing nonempty cache
            # preserves the pre-validation behavior and avoids an unrecoverable
            # download attempt when a signed source URL has expired.
            logger.warning(
                f"video cache validation unavailable; preserving existing file: "
                f"{video_path} => {str(exc)}"
            )
            return video_path
        if cached_video_is_valid:
            logger.info(f"video already exists: {video_path}")
            return video_path

        logger.warning(f"discarding invalid cached video: {video_path}")
        try:
            current_stat = os.stat(video_path)
            cached_identity = (
                cached_stat.st_dev,
                cached_stat.st_ino,
                cached_stat.st_size,
                cached_stat.st_mtime_ns,
            )
            current_identity = (
                current_stat.st_dev,
                current_stat.st_ino,
                current_stat.st_size,
                current_stat.st_mtime_ns,
            )
            # Do not delete a valid replacement published by another worker
            # while this process was validating the stale entry.
            if current_identity == cached_identity:
                os.remove(video_path)
        except FileNotFoundError:
            pass

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    # A nonempty file is treated as a cache hit above. Keep the final path
    # unpublished until the complete download has passed media validation, so a
    # failed download cannot poison this or another concurrent task's cache.
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{video_id}-",
            suffix=".mp4",
            dir=save_dir,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            with trace_http_call(
                "GET", requests.get,
                video_url,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(60, 240),
                stream=True,
            ) as response:
                response.raise_for_status()
                headers = getattr(response, "headers", {}) or {}
                try:
                    declared_size = int(headers.get("Content-Length", ""))
                except (TypeError, ValueError):
                    declared_size = 0
                if declared_size > MAX_VIDEO_DOWNLOAD_BYTES:
                    raise ValueError("video download exceeds 512 MB limit")

                downloaded_bytes = 0
                started_at = time.monotonic()
                last_report_at = started_at
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > MAX_VIDEO_DOWNLOAD_BYTES:
                            raise ValueError("video download exceeds 512 MB limit")
                        temp_file.write(chunk)
                        now = time.monotonic()
                        if now - last_report_at >= _DOWNLOAD_HEARTBEAT_SECONDS:
                            last_report_at = now
                            generation_progress.downloading(downloaded_bytes, declared_size)
                            # Solo se registra el nombre del archivo de caché: la dirección de descarga puede llevar firma o clave.
                            logger.info(
                                f"downloading video {video_id}.mp4: "
                                + _describe_download_progress(
                                    downloaded_bytes,
                                    declared_size,
                                    now - started_at,
                                )
                            )

        if os.path.getsize(temp_path) == 0:
            return ""

        generation_progress.verifying()
        if not _is_valid_video_file(temp_path):
            return ""

        os.replace(temp_path, video_path)
        temp_path = ""
        return video_path
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass
            except OSError as remove_error:
                logger.warning(
                    f"failed to remove temporary video file: {temp_path}, "
                    f"error: {str(remove_error)}"
                )


# OpenAI El texto a imagen compatible (Issue #1274) genera material de imagen para las palabras clave del guion mediante el protocolo /images/generations;
# puede apuntar a un gateway local ComfyUI/SD o a cualquier servicio intermediario con protocolo OpenAI
# . La imagen generada se renderiza de inmediato como un clip mp4 con el mismo "zoom lento" que el material local, siendo
# completamente transparente para el flujo de edición posterior.
OPENAI_IMAGE_ENDPOINT_PATH = "images/generations"
# OpenAI La interfaz oficial de imágenes solo acepta los tamaños definidos por el modelo; no se puede pasar directamente la resolución del video (como 1080x1920).
# Si openai_image_size se deja vacío, se usan los siguientes valores compatibles según el formato; los gateways locales pueden sobrescribirlo explícitamente.
OPENAI_IMAGE_DEFAULT_SIZES = {
    VideoAspect.portrait: "1024x1536",
    VideoAspect.landscape: "1536x1024",
    VideoAspect.square: "1024x1024",
}
# Mismo criterio de reintentos que WaveSpeed: 429 y 5xx son fallos temporales y se reintentan un número limitado de veces con espera.
OPENAI_IMAGE_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# 401/403 significa que la key actual fue rechazada claramente. get_api_key rota la key en cada llamada; si hay varias configuradas
# key el reintento cambia de key automáticamente; si solo hay una key falla rápido, sin reintentos inútiles.
OPENAI_IMAGE_KEY_ERROR_STATUS_CODES = frozenset({401, 403})
OPENAI_IMAGE_MAX_ATTEMPTS = 3
OPENAI_IMAGE_MAX_BYTES = 25 * 1024 * 1024
OPENAI_IMAGE_MAX_PIXELS = 50_000_000
# Generación de imágenes en serie + espera lineal, compatible con las ventanas de recuperación de límite de tasa habituales en servicios intermediarios.
OPENAI_IMAGE_RETRY_BACKOFF_SECONDS = (5, 15, 30)
# La interfaz de generación síncrona puede tardar decenas de segundos en devolver la imagen; el tiempo de espera de lectura tiene margen suficiente.
OPENAI_IMAGE_REQUEST_TIMEOUT = (30, 300)
# Reintento de descarga tras cobrarse la imagen: se reintenta primero la dirección original, en lugar de volver a generar la misma imagen.
OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS = 3
OPENAI_IMAGE_DOWNLOAD_BACKOFF_SECONDS = 2


def is_openai_image_enabled(app_config: dict | None = None) -> bool:
    """
    Determina si la fuente de material de texto a imagen compatible con OpenAI tiene la configuración mínima.

    API Key se permite vacío: los gateways ComfyUI/SD totalmente locales normalmente no requieren autenticación; si está vacío
    la petición no lleva cabecera Authorization. Sirve para que la comprobación previa de tareas y la WebUI, antes de consumir cuota de LLM y TTS,
    intercepten las tareas con configuración incompleta.
    """
    app_config = config.app if app_config is None else app_config
    return bool(
        str(app_config.get("openai_image_base_url", "") or "").strip()
        and str(app_config.get("openai_image_model", "") or "").strip()
    )


def _openai_image_endpoint() -> tuple[str, str]:
    """
    Lee el endpoint y el nombre del modelo de texto a imagen; si faltan, lanza un error con indicaciones de configuración.
    """
    base_url = (
        str(config.app.get("openai_image_base_url", "") or "").strip().rstrip("/")
    )
    model = str(config.app.get("openai_image_model", "") or "").strip()
    if not base_url:
        raise ValueError(
            "\n\n##### openai_image_base_url is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )
    if not model:
        raise ValueError(
            "\n\n##### openai_image_model is not set #####\n\n"
            f"Please set it in the config.toml file: {config.config_file}\n"
        )
    return f"{base_url}/{OPENAI_IMAGE_ENDPOINT_PATH}", model


def _openai_image_size(video_aspect: VideoAspect) -> str:
    """
    Resuelve el size de imagen solicitado.

    OpenAI La interfaz oficial solo acepta los tamaños definidos por el modelo (como 1024x1536); pasar directamente la resolución del video
    (como 1080x1920) devuelve 400. Por defecto se usa un tamaño compatible según el formato;``openai_image_size``
    se puede sobrescribir explícitamente, para gateways locales que admiten cualquier resolución (como SD WebUI).
    """
    configured = str(config.app.get("openai_image_size", "") or "").strip()
    if configured:
        return configured
    return OPENAI_IMAGE_DEFAULT_SIZES.get(VideoAspect(video_aspect), "1024x1024")


def _openai_image_prompt(search_term: str) -> str:
    """
    Envuelve la palabra clave del guion en el prompt final.

    La configuración opcional ``openai_image_prompt_template`` admite el marcador ``{term}``,
    para añadir de forma unificada modificadores de estilo (calidad, composición, lenguaje de cámara) y mejorar la correspondencia imagen-texto:

    .. code-block:: toml

        openai_image_prompt_template = "cinematic photo of {term}, photorealistic"

    si está vacía o no contiene el marcador se usa la palabra clave original, con el mismo comportamiento que la versión anterior. Si la sustitución
    del marcador falla (por ejemplo, la plantilla contiene sintaxis de formato errónea) también se usa el original, para que un error de configuración no interrumpa
    toda la tarea de generación.
    """
    template = str(config.app.get("openai_image_prompt_template", "") or "").strip()
    if not template or "{term}" not in template:
        return search_term
    try:
        return template.replace("{term}", search_term)
    except Exception:
        return search_term


def _response_json_safely(response: Any) -> Any:
    """Lee el JSON de la respuesta; devuelve None si falla el análisis del doble de prueba o de una respuesta anómala."""
    try:
        return response.json()
    except Exception:
        return None


def _openai_image_response_message(body: Any) -> str:
    """
    Extrae una descripción de error legible de una respuesta compatible con OpenAI.

    El formato estándar es ``{"error": {"message": ...}}``; los servicios intermediarios a menudo degradan a
    ``{"message": ...}`` o dan directamente una cadena. Si no se obtiene ninguno se devuelve una cadena vacía, y el llamador
    decide si recurrir al cuerpo de la respuesta.
    """
    if not isinstance(body, dict):
        return str(body or "")[:300]
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")[:300]
    if error is not None:
        return str(error)[:300]
    return str(body.get("message") or "")[:300]


def _openai_image_http_failure(response: Any, status: int, api_key: str) -> str:
    """Convierte una respuesta HTTP de error en una descripción legible y anonimizada para el log."""
    message = _openai_image_response_message(_response_json_safely(response))
    if not message:
        message = str(getattr(response, "text", "") or "")[:300]
    return f"HTTP {status}: {_redact_secret(message, api_key)}"


def _openai_image_download_bytes(
    image_url: str,
    api_key: str,
) -> tuple[bytes | None, str]:
    """
    Descarga la URL temporal de la imagen ya generada.

    La imagen ya se cobró por unidad; si la descarga falla se reintenta primero la dirección original, en lugar de volver a generarla,
    para no pagar dos veces por la misma imagen. Si fallan todos los reintentos se lanza un error, impidiendo que la capa exterior compre la siguiente imagen.
    """
    failure_detail = "no download attempt was made"
    for attempt in range(1, OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS + 1):
        response = None
        try:
            response = trace_http_call(
                "GET", requests.get,
                image_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0.0.0 Safari/537.36"
                },
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 120),
                stream=True,
            )
            if response.status_code == 200:
                try:
                    declared_size = int(response.headers.get("Content-Length", 0))
                except (TypeError, ValueError):
                    declared_size = 0
                if declared_size > OPENAI_IMAGE_MAX_BYTES:
                    return None, "generated image exceeds the 25 MB download limit"
                image_bytes = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    if len(image_bytes) + len(chunk) > OPENAI_IMAGE_MAX_BYTES:
                        return None, "generated image exceeds the 25 MB download limit"
                    image_bytes.extend(chunk)
                if image_bytes:
                    return bytes(image_bytes), ""
                failure_detail = "generated image download was empty"
            else:
                failure_detail = f"HTTP {response.status_code} while downloading image"
        except Exception as e:
            failure_detail = (
                f"error={type(e).__name__}, "
                f"detail={_redact_request_error(e, api_key, image_url)}"
            )
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception as exc:
                    logger.warning(
                        "failed to close generated image download response: "
                        f"error={type(exc).__name__}"
                    )
        if attempt < OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS:
            logger.warning(
                "generated image download failed, retrying the same url: "
                f"attempt={attempt}/{OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS}, "
                f"{failure_detail}"
            )
            time.sleep(OPENAI_IMAGE_DOWNLOAD_BACKOFF_SECONDS)
    raise OpenAIImagePaidResultError(
        "generated image could not be downloaded after retries"
    )


def _parse_openai_image_response(
    response: Any,
    api_key: str,
) -> tuple[bytes | None, str]:
    """
    Analiza la respuesta de /images/generations y obtiene los datos de imagen url o b64_json.

    Un fallo de análisis es un rechazo de negocio claro (como la política de contenido) o un formato de respuesta anómalo; se devuelve directamente
    la descripción del error, sin reintentos con espera: reenviar la misma petición solo daría el mismo resultado.
    """
    body = _response_json_safely(response)
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        return None, _redact_secret(_openai_image_response_message(body), api_key)

    entry = data[0]
    if not isinstance(entry, dict):
        return None, "invalid image data entry"

    b64_payload = entry.get("b64_json")
    if b64_payload:
        if (
            not isinstance(b64_payload, str)
            or len(b64_payload) > 4 * ((OPENAI_IMAGE_MAX_BYTES + 2) // 3)
        ):
            return None, "generated image exceeds the 25 MB response limit"
        try:
            image_bytes = base64.b64decode(b64_payload)
        except Exception as e:
            return None, f"invalid b64_json payload: {type(e).__name__}"
        if len(image_bytes) > OPENAI_IMAGE_MAX_BYTES:
            return None, "generated image exceeds the 25 MB response limit"
        return image_bytes, ""

    image_url = entry.get("url")
    if isinstance(image_url, str) and image_url.startswith(("http://", "https://")):
        return _openai_image_download_bytes(image_url, api_key)

    return None, "image response has neither url nor b64_json"


def _request_openai_image(endpoint: str, payload: dict) -> tuple[bytes | None, str]:
    """
    Llama a la interfaz /images/generations compatible con OpenAI, con reintentos con espera y rotación de key.

    429/5xx se reintentan con espera como fallos temporales; 401/403 solo se reintentan si hay varias keys configuradas
    (cambiando de key con el mecanismo de rotación de get_api_key); los demás 4xx son rechazos claros y fallan rápido,
    dejando que la capa superior omita esa palabra clave.

    Seguridad de cobro: el tiempo de espera de lectura y la interrupción de la conexión del POST se consideran estado "no confirmado": el servidor puede haber
    generado y cobrado, solo que la respuesta no volvió; reenviar automáticamente podría causar generación y cobro
    duplicados, así que se lanza una excepción específica que aborta la tarea local. Solo el tiempo agotado en la fase de conexión (ConnectTimeout,
    la petición seguro que no llegó al servidor) confirma que no se creó ninguna tarea de generación y se puede reintentar con seguridad.

    API Key se permite vacío: los gateways ComfyUI/SD totalmente locales normalmente no requieren autenticación; si está vacío
    no se envía la cabecera Authorization.
    """
    api_keys = config.app.get("openai_image_api_keys")
    if isinstance(api_keys, (list, tuple)):
        configured_keys = [k for k in api_keys if str(k or "").strip()]
    elif str(api_keys or "").strip():
        configured_keys = [api_keys]
    else:
        configured_keys = []

    failure_detail = "no request attempt was made"
    for attempt in range(1, OPENAI_IMAGE_MAX_ATTEMPTS + 1):
        api_key = get_api_key("openai_image_api_keys") if configured_keys else ""
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        retryable = False
        try:
            response = trace_http_call(
                "POST", requests.post,
                endpoint,
                json=payload,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=OPENAI_IMAGE_REQUEST_TIMEOUT,
                allow_redirects=False,
            )
        except requests.exceptions.ConnectTimeout as e:
            # Tiempo agotado en la fase de conexión: la petición seguro que no llegó al servidor y no se creó ninguna tarea de generación,
            # así que se puede reintentar con seguridad.
            failure_detail = (
                f"connect timeout: detail={_redact_request_error(e, api_key)}"
            )
            retryable = True
        except Exception as e:
            # El tiempo de espera de lectura, la interrupción de la conexión, etc. son estado "no confirmado": el servidor puede haber aceptado y cobrado,
            # y reenviar automáticamente o procesar las siguientes palabras clave podría cobrar dos veces, así que hay que abortar la tarea local.
            raise OpenAIImageUnconfirmedError(
                "unconfirmed image request (no retry to avoid double billing): "
                f"{type(e).__name__}, detail={_redact_request_error(e, api_key)}"
            ) from e
        else:
            status = int(getattr(response, "status_code", 200) or 200)
            if 300 <= status < 400:
                response.close()
                raise OpenAIImageUnconfirmedError(
                    "image submission returned a redirect; no redirected request "
                    "or retry was sent because the paid outcome is unconfirmed"
                )
            if status in OPENAI_IMAGE_KEY_ERROR_STATUS_CODES:
                failure_detail = _openai_image_http_failure(response, status, api_key)
                # Solo con varias keys configuradas el reintento puede rotar a una key válida.
                retryable = len(configured_keys) > 1
            elif status in OPENAI_IMAGE_RETRYABLE_STATUS_CODES:
                failure_detail = _openai_image_http_failure(response, status, api_key)
                retryable = True
            elif status >= 400:
                return None, _openai_image_http_failure(response, status, api_key)
            else:
                image_bytes, parse_error = _parse_openai_image_response(
                    response, api_key
                )
                if image_bytes is not None:
                    return image_bytes, ""
                failure_detail = parse_error

        if retryable and attempt < OPENAI_IMAGE_MAX_ATTEMPTS:
            backoff_seconds = OPENAI_IMAGE_RETRY_BACKOFF_SECONDS[
                min(attempt - 1, len(OPENAI_IMAGE_RETRY_BACKOFF_SECONDS) - 1)
            ]
            logger.warning(
                "openai image request failed, retrying: "
                f"attempt={attempt}/{OPENAI_IMAGE_MAX_ATTEMPTS}, "
                f"next_retry_in={backoff_seconds}s, detail={failure_detail}"
            )
            time.sleep(backoff_seconds)
            continue
        return None, failure_detail

    return None, failure_detail


def _save_openai_image_file(
    image_bytes: bytes,
    save_dir: str,
) -> tuple[str, int, int]:
    """
    Normaliza el resultado generado como PNG en disco y devuelve (ruta, ancho, alto).

    Convertir todo a PNG evita dos tipos de problemas: servicios intermediarios que devuelven WebP/JPEG sin una extensión
    fiable, e imágenes con metadatos anómalos que hacen fallar el análisis de MoviePy (en línea con la lógica de saneamiento
    del material local; aquí la normalización se completa ya al guardar en disco).
    """
    if not save_dir:
        save_dir = utils.storage_dir("cache_images", create=True)
    elif not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    image_path = os.path.join(save_dir, f"openai-image-{uuid.uuid4().hex[:12]}.png")
    if len(image_bytes) > OPENAI_IMAGE_MAX_BYTES:
        raise _OpenAIImageDecodeError("generated image exceeds the 25 MB limit")

    # Un fallo de decodificación de la imagen puede degradarse a «omitir la palabra clave actual», pero los fallos de permisos del directorio, espacio en disco y
    # escritura de archivos deben seguir lanzándose; de lo contrario el bucle de generación bajo demanda seguiría creando tareas de pago
    # aunque no se puedan guardar archivos localmente. Image.open solo lee bytes en memoria, así que aquí OSError es un fallo de reconocimiento
    # del formato; el OSError de image.load corresponde a datos de imagen truncados o dañados.
    image = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(image_bytes))
            if image.width * image.height > OPENAI_IMAGE_MAX_PIXELS:
                raise ValueError("generated image exceeds the 50 million pixel limit")
            image.load()
    except (
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        if image is not None:
            image.close()
        raise _OpenAIImageDecodeError(f"{type(exc).__name__}: {exc}") from exc

    with image:
        if image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
            image = image.convert("RGB")
        # save no se incluye en la zona de protección de excepciones de decodificación: un error de escritura indica que el entorno sigue sin estar disponible y se debe
        # abortar toda la tarea de inmediato, para no seguir produciendo imágenes de pago que no se pueden guardar.
        image.save(image_path, format="PNG")
        width, height = image.size
    return image_path, width, height


def generate_images_openai(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    save_dir: str = "",
) -> List[MaterialInfo]:
    """
    Genera una imagen para una palabra clave del guion con la interfaz de texto a imagen compatible con OpenAI y la guarda localmente.

    Mantiene la misma firma y convención de lista vacía ante fallos que generate_videos_wavespeed. Las imágenes no tienen
    duración nativa; ``duration`` registra la duración del clip objetivo (segundos), para que el flujo de descarga bajo demanda calcule
    si ya se cubre la duración de la voz. El tamaño real que devuelve la API puede no coincidir con el pedido, así que aquí
    se escribe en rendition el tamaño real de la imagen, sin depender de los parámetros de la petición.
    """
    aspect = VideoAspect(video_aspect)
    clip_duration = max(int(minimum_duration), 1)
    endpoint, model = _openai_image_endpoint()
    image_size = _openai_image_size(aspect)
    payload = {
        "model": model,
        "prompt": _openai_image_prompt(search_term),
        "n": 1,
        "size": image_size,
    }
    logger.info(
        f"generating image via openai-compatible endpoint: model={model}, "
        f"term={search_term!r}, size={image_size}"
    )
    generation_progress.submitting("openai_image")
    image_bytes, failure_detail = _request_openai_image(endpoint, payload)
    if image_bytes is None:
        logger.error(
            f"openai image generation failed: term={search_term!r}, "
            f"detail={failure_detail}"
        )
        return []

    try:
        image_path, width, height = _save_openai_image_file(image_bytes, save_dir)
    except _OpenAIImageDecodeError as e:
        # La capa de compatibilidad puede devolver 200 con un body que no es una imagen (como una página de error HTML disfrazada de JSON,
        # o una página de aviso de degradación del gateway). Que la imagen no se pueda decodificar significa "esta generación ya falló"; según la convención de las fuentes de material
        # se devuelve una lista vacía para que la capa superior omita esa palabra clave y continúe, en lugar de que la excepción interrumpa toda la tarea.
        logger.error(
            "openai image response is not a decodable image, skipping term: "
            f"term={search_term!r}, error={type(e).__name__}, detail={e}"
        )
        return []
    item = MaterialInfo()
    item.provider = "openai_image"
    item.url = image_path
    item.duration = clip_duration
    item.source_info = {
        "provider": "openai_image",
        "search_term": search_term,
        "rendition": {
            "id": None,
            "width": width,
            "height": height,
        },
    }
    return [item]


def generate_images_codex(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    save_dir: str = "",
    *,
    context=None,
) -> List[MaterialInfo]:
    """Obtain one Codex image using the existing PNG material representation."""
    aspect = VideoAspect(video_aspect)
    if minimum_duration <= 0:
        return []
    try:
        context = context if context is not None else codex_image.capture_context()
        codex_image.assert_current(context)
        # Independent of the OpenAI-compatible provider's template and settings.
        prompt = (
            f"Create one image depicting: {search_term}. "
            f"Compose the image for a {aspect.value} video frame."
        )
        generation_progress.submitting("codex_image")
        image_bytes = codex_image.generate_image(prompt, context)
        codex_image.assert_current(context)
        try:
            image_path, width, height = _save_openai_image_file(image_bytes, save_dir)
        except _OpenAIImageDecodeError:
            raise codex_image.CodexImageError("codex.image.invalid_result") from None
        except OSError:
            raise codex_image.CodexImageError("codex.image.save_failed") from None
        codex_image.assert_current(context)
    except codex_image.CodexImageError as error:
        raise CodexImageMaterialError(error) from None
    item = MaterialInfo()
    item.provider = "codex_image"
    item.url = image_path
    item.duration = max(int(minimum_duration), 1)
    item.source_info = {
        "provider": "codex_image",
        "search_term": search_term,
        "rendition": {"id": None, "width": width, "height": height},
    }
    return [item]


def _render_openai_image_video(
    image_path: str, clip_duration: int, image_motion: ImageMotion = ImageMotion.zoom
) -> str:
    """
    Renderiza la imagen generada como clip mp4, reutilizando la cadena "imagen → clip dinámico" del material local.

    Si el renderizado falla devuelve una cadena vacía, y el llamador detiene las siguientes peticiones de pago.
    """
    try:
        return video.render_image_zoom_video(
            image_path, clip_duration, image_motion=image_motion
        )
    except Exception as e:
        logger.error(
            "failed to render generated image as a video clip: "
            f"image={image_path}, error={type(e).__name__}, detail={e}"
        )
        return ""


def _download_videos_openai_image_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    image_motion: ImageMotion = ImageMotion.zoom,
    image_generator: Callable[..., List[MaterialInfo]] | None = None,
    provider: str = "openai_image",
    codex_image_context=None,
) -> List[str]:
    """
    Genera una a una las imágenes de texto a imagen compatibles con OpenAI en el orden de los fragmentos del guion, y se detiene en cuanto cubre la duración total necesaria.

    Misma semántica de seguridad de pago que la generación bajo demanda de WaveSpeed: el texto a imagen se cobra por imagen, y generar todo primero para luego
    elegir pagaría por imágenes que no se usarán. Cada imagen generada se renderiza de inmediato como clip mp4 y se acumula
    su duración efectiva (igual que el flujo de stock, con tope en la duración del clip); al alcanzar la duración de voz necesaria
    no se lanzan nuevas peticiones de pago. Una imagen rechazada claramente se puede omitir; si el resultado de la petición es incierto, falla la descarga
    de un resultado pagado o falla el renderizado local, hay que abortar la tarea para no volver a cobrar por las palabras clave siguientes.
    """
    # Las llamadas directas al servicio de material también deben validarse antes de la petición de pago, no descubrir tras generar que los parámetros de efecto no son válidos.
    image_motion = ImageMotion(image_motion)
    if not material_directory:
        # Las imágenes generadas se cobran por tarea y no se pueden reutilizar; por defecto se guardan en el directorio de la tarea para facilitar la trazabilidad.
        material_directory = utils.task_dir(task_id)

    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0

    # Una duración de voz no positiva haría que la condición "alcanzar la duración necesaria" perdiera sentido, así que se devuelve directamente sin nada,
    # para no seguir pagando por imagen en una tarea que nunca se completaría (igual que la semántica de comprobación previa de Seedance).
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError):
        required_duration = 0.0
    if required_duration <= 0:
        logger.warning(
            "skip openai image generation because required audio duration is "
            f"not positive: duration={audio_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    if image_generator is None:
        image_generator = generate_images_openai
    generation_options = {}
    if provider == "codex_image":
        try:
            codex_image_context = (
                codex_image_context
                if codex_image_context is not None
                else codex_image.capture_context()
            )
            codex_image.assert_current(codex_image_context)
        except codex_image.CodexImageError as error:
            raise CodexImageMaterialError(error) from None
        generation_options["context"] = codex_image_context

    for search_term in search_terms:
        try:
            items = image_generator(
                search_term=search_term,
                minimum_duration=max_clip_duration,
                video_aspect=video_aspect,
                save_dir=material_directory,
                **generation_options,
            )
        except OpenAIImagePaidResultError:
            _persist_material_sources(task_id, material_sources)
            raise
        for item in items:
            video_file = _render_openai_image_video(
                item.url, max_clip_duration, image_motion=image_motion
            )
            if not video_file:
                _persist_material_sources(task_id, material_sources)
                if provider == "codex_image":
                    raise CodexImageMaterialError(
                        codex_image.CodexImageError("codex.image.render_failed")
                    )
                raise OpenAIImagePaidResultError(
                    "generated image could not be rendered locally"
                )
            if provider == "codex_image":
                try:
                    codex_image.assert_current(codex_image_context)
                except codex_image.CodexImageError as error:
                    _persist_material_sources(task_id, material_sources)
                    raise CodexImageMaterialError(error) from None
            logger.info(f"image material rendered: {video_file}")
            video_paths.append(video_file)
            try:
                material_sources.append(_material_source_record(item, video_file))
            except Exception as source_error:
                # Igual que las fuentes de stock: una excepción en el registro de origen no puede hacer que un material ya pagado y renderizado con éxito
                # se considere fallido, y mucho menos bloquear la generación del video.
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider={provider}, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(max_clip_duration, item.duration)
            # Igual que WaveSpeed se usa >=: generar otra imagen cuando ya se alcanza justo la duración sería pagar una vez más.
            # Las dos comprobaciones, interior y exterior, deben mantener la misma semántica.
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated image materials cover the required duration, stop "
                f"generating more images: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"generated and rendered {len(video_paths)} image materials")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _search_videos_with_cache(
    provider: str,
    search_videos: Callable[..., List[MaterialInfo]],
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> List[MaterialInfo]:
    """
    Gestiona de forma unificada la caché de búsqueda de 24 horas de las tres fuentes de material en línea.

    La caché solo envuelve la API de búsqueda y no cambia la lógica posterior de descarga de video ni de deduplicación. Cuando el remoto devuelve una lista vacía no se escribe
    en caché, porque la interfaz de provider actual usa la lista vacía tanto para «sin resultados» como para «petición fallida»;
    hasta que ambos se separen en tipos de resultado claros, es mejor reintentar la próxima vez que cachear un fallo temporal durante un día.
    """
    cache_args = {
        "provider": provider,
        "search_term": search_term,
        "minimum_duration": minimum_duration,
        "video_aspect": video_aspect,
    }

    def load_cache_safely() -> List[MaterialInfo] | None:
        try:
            return material_cache.load_material_search_cache(**cache_args)
        except Exception as exc:
            # La caché es una optimización opcional; cualquier excepción de la implementación de caché debe tratarse como fallo de caché, sin bloquear
            # Pexelsla búsqueda remota normal de , Pixabay o Coverr.
            logger.warning(
                "material search cache read failed, continue with remote search: "
                f"provider={provider}, error={type(exc).__name__}, detail={exc}"
            )
            return None

    def load_matching_cache() -> tuple[List[MaterialInfo] | None, int]:
        cached_items = load_cache_safely()
        if cached_items is None:
            return None, 0

        filtered_cached_items = _filter_materials_by_aspect(
            cached_items,
            video_aspect,
        )
        ignored_count = len(cached_items) - len(filtered_cached_items)
        if ignored_count:
            # La caché de versiones antiguas puede mezclar material de otras orientaciones. Aunque queden algunas entradas útiles, hay que refrescar
            # el conjunto completo de candidatos; de lo contrario, durante la validez de la caché se reutilizaría repetidamente el mismo pequeño grupo de videos.
            return None, ignored_count
        return filtered_cached_items, 0

    cached_items, ignored_count = load_matching_cache()
    if cached_items is not None:
        return cached_items
    if ignored_count:
        logger.info(
            "material search cache contains mismatched orientations, "
            f"refresh from provider: provider={provider}, term={search_term!r}, "
            f"ignored={ignored_count}"
        )

    cache_lock = material_cache.get_material_search_cache_lock(**cache_args)
    with cache_lock:
        # Espera a que termine el hilo con las mismas condiciones de búsqueda y vuelve a leer, para evitar que varias tareas de la API consulten a la vez el remoto
        # en el primer fallo de caché, reduciendo la probabilidad de activar límites de tasa y controles de riesgo de interfaces de terceros.
        cached_items, _ = load_matching_cache()
        if cached_items is not None:
            return cached_items

        items = search_videos(
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )
        # Provider normalmente se escribe la palabra clave actual, pero dobles de prueba, extensiones de terceros o implementaciones antiguas pueden
        # omitirla o llevar un valor erróneo. La lectura de caché restaura ese campo según la clave de caché, así que el resultado remoto también se
        # corrige en la misma entrada, garantizando que el registro de origen de la tarea sea igual en la primera búsqueda y en un acierto de caché.
        for item in items:
            if isinstance(item.source_info, dict):
                item.source_info = dict(item.source_info)
                item.source_info["search_term"] = search_term
        if items:
            try:
                material_cache.save_material_search_cache(
                    **cache_args,
                    items=items,
                )
            except Exception as exc:
                logger.warning(
                    "material search cache write failed, use remote results: "
                    f"provider={provider}, error={type(exc).__name__}, detail={exc}"
                )
        return items


def _search_terms_in_parallel(
    search_terms: List[str],
    search_videos: Callable[[str, int, VideoAspect], List[MaterialInfo]],
    minimum_duration: int,
    video_aspect: VideoAspect,
) -> list[tuple[str, List[MaterialInfo]]]:
    """Busca palabras clave en paralelo; las que fallan devuelven un resultado vacío sin bloquear el resto del material."""
    if not search_terms:
        return []

    workers = min(_get_material_concurrency(), len(search_terms))
    if workers == 1:
        results = []
        for search_term in search_terms:
            try:
                items = search_videos(
                    search_term=search_term,
                    minimum_duration=minimum_duration,
                    video_aspect=video_aspect,
                )
            except Exception as exc:
                logger.error(
                    "failed to search material videos: "
                    f"search_term={search_term!r}, "
                    f"error={type(exc).__name__}, detail={exc}"
                )
                items = []
            logger.info(f"found {len(items)} videos for '{search_term}'")
            results.append((search_term, items))
        return results

    futures = {}
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="material-search",
    ) as executor:
        for search_term in search_terms:
            futures[executor.submit(
                copy_context().run,
                logging_utils.bind_log_scope(search_videos),
                search_term,
                minimum_duration,
                video_aspect,
            )] = search_term

        results = []
        for future in futures:
            search_term = futures[future]
            try:
                items = future.result()
            except Exception as exc:
                logger.error(
                    "failed to search material videos: "
                    f"search_term={search_term!r}, "
                    f"error={type(exc).__name__}, detail={exc}"
                )
                items = []
            logger.info(f"found {len(items)} videos for '{search_term}'")
            results.append((search_term, items))
        return results


def _report_material_downloaded(
    position: int,
    total: int,
    item: MaterialInfo,
    saved_video_path: str,
    on_downloaded: Callable[[MaterialInfo], None] | None,
) -> None:
    """
    Registra que un material terminó de descargarse y notifica al llamador para actualizar el progreso.

    Se llama siempre en el hilo que inició la descarga; el log por archivo y la actualización de progreso no dependen de la concurrencia de descarga.
    """
    description = os.path.basename(saved_video_path)
    try:
        size_mb = os.path.getsize(saved_video_path) / _BYTES_PER_MEGABYTE
        description += f" ({size_mb:.1f} MB)"
    except OSError:
        # El tamaño del archivo es solo información auxiliar; si no se puede leer no afecta al resultado de la descarga.
        pass
    logger.info(f"downloaded material {position}/{total}: {description}")
    if on_downloaded is None:
        return
    try:
        on_downloaded(item)
    except Exception as exc:
        # El progreso es solo información de visualización. Un fallo del callback (por ejemplo, el backend de estado no disponible temporalmente) no puede invalidar
        # material ya descargado con éxito, y mucho menos interrumpir toda la tarea.
        logger.warning(
            "failed to report material download progress: "
            f"error={type(exc).__name__}, detail={exc}"
        )


def _covered_duration_reporter(
    progress_callback: Callable[[float], None] | None,
    required_duration: float,
    max_clip_duration: int,
) -> Callable[[MaterialInfo], None] | None:
    """
    Genera el callback por material: convierte la «duración de voz cubierta» en un progreso de descarga de 0~1.

    El criterio coincide con la condición de parada del bucle de descarga: cada material aporta como máximo la duración de un clip, y al cubrir
    la duración necesaria se completa, así que el exceso se limita a 1.0. Sin callback de progreso devuelve None.
    """
    if progress_callback is None:
        return None
    covered_duration = 0.0

    def report(item: MaterialInfo) -> None:
        nonlocal covered_duration
        covered_duration += min(max_clip_duration, item.duration)
        fraction = 1.0
        if required_duration > 0:
            fraction = min(1.0, covered_duration / required_duration)
        progress_callback(fraction)

    return report


def _download_materials_in_parallel(
    materials: List[tuple[str, MaterialInfo]],
    material_directory: str,
    on_downloaded: Callable[[MaterialInfo], None] | None = None,
) -> list[tuple[str, MaterialInfo, str]]:
    """
    Descarga en paralelo una ronda de material, conservando el orden de candidatos que pasó el llamador.

    Tras descargar con éxito cada material se registra una línea de log y se llama a ``on_downloaded``, con lo que el llamador actualiza
    el progreso de la tarea. Tanto el callback como el log ocurren en el hilo que llama a esta función.
    """
    if not materials:
        return []

    total = len(materials)
    workers = min(_get_material_concurrency(), total)
    if workers == 1:
        downloaded = []
        for position, (search_term, item) in enumerate(materials, start=1):
            try:
                saved_video_path = save_video(
                    video_url=item.url,
                    save_dir=material_directory,
                )
            except Exception as exc:
                logger.error(
                    "failed to download material video: "
                    f"provider={item.provider}, "
                    f"error={type(exc).__name__}, "
                    f"detail={_redact_request_error(exc, item.url)}"
                )
                continue
            if saved_video_path:
                downloaded.append((search_term, item, saved_video_path))
                _report_material_downloaded(
                    position, total, item, saved_video_path, on_downloaded
                )
        return downloaded

    futures = {}
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="material-download",
    ) as executor:
        for search_term, item in materials:
            futures[executor.submit(
                copy_context().run,
                logging_utils.bind_log_scope(save_video),
                item.url,
                material_directory,
            )] = (search_term, item)

        downloaded = []
        for position, future in enumerate(futures, start=1):
            search_term, item = futures[future]
            try:
                saved_video_path = future.result()
            except Exception as exc:
                logger.error(
                    "failed to download material video: "
                    f"provider={item.provider}, "
                    f"error={type(exc).__name__}, "
                    f"detail={_redact_request_error(exc, item.url)}"
                )
                continue
            if saved_video_path:
                downloaded.append((search_term, item, saved_video_path))
                _report_material_downloaded(
                    position, total, item, saved_video_path, on_downloaded
                )
        return downloaded


def _select_materials_until_duration(
    materials: List[tuple[str, MaterialInfo]],
    max_clip_duration: int,
    audio_duration: float,
    current_duration: float = 0.0,
) -> List[tuple[str, MaterialInfo]]:
    """
    Elige una ronda de material en el orden de candidatos, para no descargar en concurrencia candidatos que ya superan la duración de la voz.

    Devuelve un subconjunto prefijo de ``materials``: acumula desde el primer candidato y se detiene cuando la duración acumulada
    supera por primera vez la duración de la voz (incluyendo el que la supera). Con esta propiedad de prefijo el llamador sabe qué
    candidatos se intentaron realmente en esta ronda.
    """
    selected = []
    total_duration = current_duration
    for material in materials:
        selected.append(material)
        total_duration += min(max_clip_duration, material[1].duration)
        if total_duration > audio_duration:
            break
    return selected


def download_videos(
    task_id: str,
    search_terms: List[str],
    source: str = "pexels",
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
    match_script_order: bool = False,
    progress_callback: Callable[[float], None] | None = None,
    image_motion: ImageMotion = ImageMotion.zoom,
    jalapeno_video_endpoint: volcengine_seedance.SeedanceEndpoint | None = None,
    codex_image_context=None,
) -> List[str]:
    """
    Busca y descarga el material necesario para cubrir la duración de la voz, y devuelve las rutas de los archivos locales.

    ``progress_callback`` Opcional: el material de stock (Pexels, Pixabay, Coverr) llama una vez por cada
    archivo descargado con la proporción completada de 0~1, para que la capa de tareas avance la barra de progreso. Las fuentes de material de pago
    generado bajo demanda no informan del progreso de descarga.
    """
    provider = "pexels"
    remote_search_videos = search_videos_pexels
    if source == "pixabay":
        provider = "pixabay"
        remote_search_videos = search_videos_pixabay
    elif source == "coverr":
        provider = "coverr"
        remote_search_videos = search_videos_coverr

    def search_videos(
        search_term: str,
        minimum_duration: int,
        video_aspect: VideoAspect,
    ) -> List[MaterialInfo]:
        return _search_videos_with_cache(
            provider=provider,
            search_videos=remote_search_videos,
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )

    material_directory = config.app.get("material_directory", "").strip()
    if material_directory == "task":
        material_directory = utils.task_dir(task_id)
    elif material_directory and not os.path.isdir(material_directory):
        material_directory = ""

    if source == "infistar":
        return _download_videos_infistar_on_demand(
            task_id=task_id, search_terms=search_terms, video_aspect=video_aspect,
            audio_duration=audio_duration, max_clip_duration=max_clip_duration,
        )
    if source == "wavespeed":
        # AI La generación se cobra por unidad, así que no puede seguir el flujo de stock de "obtener primero candidatos para todas las palabras clave y luego elegir"
        # », o pagaría por clips que no se usarán. Las fuentes de generación generan fragmento a fragmento bajo demanda y se detienen
        # en cuanto cubren la duración necesaria; tampoco participan en la caché de búsqueda de 24 horas: la URL del resultado es una dirección firmada
        # que caduca, y reutilizar la caché haría que distintas tareas obtuvieran repetidamente el mismo video generado.
        return _download_videos_wavespeed_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source in {"volcengine_seedance", "jalapeno"}:
        # Igual que WaveSpeed, la interfaz oficial de Ark crea tareas de pago asíncronas. Hay que generar
        # fragmento a fragmento bajo demanda, comprando solo el material que realmente necesita la duración de voz actual.
        return _download_videos_seedance_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            source=source,
            endpoint_snapshot=jalapeno_video_endpoint,
        )
    if source == "ofox":
        # Misma semántica de pago bajo demanda que WaveSpeed/Ark: /v1/videos del gateway OFox crea
        # tareas de pago asíncronas, así que hay que generar fragmento a fragmento y detenerse al cubrir la duración necesaria; la dirección del resultado es un enlace directo temporal
        # que caduca y tampoco participa en la caché de búsqueda de 24 horas.
        return _download_videos_ofox_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "muapi":
        # MuAPI el endpoint del modelo crea tareas de pago asíncronas. Se envían fragmento a fragmento y se detiene al cubrir la duración de la voz,
        # para no enviar al remoto palabras clave del guion que no se usarán.
        return _download_videos_muapi_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source == "metaso_minimax":
        # Metaso MiniMax también cobra por tarea asíncrona remota. Su cuerpo de petición se parece al de Volcengine Ark,
        # pero la ruta de consulta de tareas y la estructura de respuesta son distintas, así que solo comparte la semántica local de generación bajo demanda, sin reutilizar
        # el cliente del proveedor, para que las diferencias de protocolo no se filtren a la capa de orquestación de material.
        return _download_videos_metaso_minimax_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
        )
    if source in {"openai_image", "codex_image"}:
        # Misma semántica de pago bajo demanda que WaveSpeed: el texto a imagen se cobra por imagen; se genera fragmento a fragmento y se detiene al cubrir
        # la duración necesaria. El resultado es un archivo de imagen local de un solo uso y tampoco participa en la caché de búsqueda 24
        # de 24 horas: la caché haría que distintas tareas obtuvieran repetidamente la misma imagen.
        image_options = {}
        if source == "codex_image":
            image_options = {
                "image_generator": generate_images_codex,
                "provider": source,
                "codex_image_context": codex_image_context,
            }
        return _download_videos_openai_image_on_demand(
            task_id=task_id,
            search_terms=search_terms,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            image_motion=image_motion,
            **image_options,
        )

    if match_script_order:
        return _download_videos_by_script_order(
            task_id=task_id,
            search_terms=search_terms,
            search_videos=search_videos,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            progress_callback=progress_callback,
        )

    valid_video_items = []
    valid_video_urls = []
    found_duration = 0.0
    for search_term, video_items in _search_terms_in_parallel(
        search_terms=search_terms,
        search_videos=search_videos,
        minimum_duration=max_clip_duration,
        video_aspect=video_aspect,
    ):
        for item in video_items:
            if item.url not in valid_video_urls:
                valid_video_items.append(item)
                valid_video_urls.append(item.url)
                found_duration += item.duration

    logger.info(
        f"found total videos: {len(valid_video_items)}, required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )
    video_paths = []
    material_sources: list[dict[str, Any]] = []

    concat_mode_value = getattr(video_concat_mode, "value", video_concat_mode)
    if concat_mode_value == VideoConcatMode.random.value:
        random.shuffle(valid_video_items)

    total_duration = 0.0
    on_downloaded = _covered_duration_reporter(
        progress_callback, audio_duration, max_clip_duration
    )
    pending_items = list(valid_video_items)
    # La ruta por defecto/aleatoria también está controlada por la configuración de concurrencia de material: preselecciona los candidatos de cada ronda según la condición de parada
    # de la lógica en serie (acumulación en orden, parar al superar por primera vez la duración de la voz). Si todas las descargas tienen éxito, el conjunto descargado en cada ronda
    # es idéntico al de la lógica en serie; los candidatos que fallan se completan en la siguiente ronda con candidatos posteriores.
    while pending_items and total_duration <= audio_duration:
        batch = []
        projected_duration = total_duration
        for item in pending_items:
            batch.append(item)
            projected_duration += min(max_clip_duration, item.duration)
            if projected_duration > audio_duration:
                break
        for item in batch:
            source_info = item.source_info if isinstance(item.source_info, dict) else {}
            logger.info(
                f"downloading {item.provider} video: "
                f"asset_id={source_info.get('asset_id') or 'unknown'}"
            )
        downloaded_materials = _download_materials_in_parallel(
            materials=[("", item) for item in batch],
            material_directory=material_directory,
            on_downloaded=on_downloaded,
        )
        pending_items = pending_items[len(batch):]
        for _, item, saved_video_path in downloaded_materials:
            try:
                if saved_video_path:
                    logger.info(f"video saved: {saved_video_path}")
                    video_paths.append(saved_video_path)
                    try:
                        material_sources.append(
                            _material_source_record(item, saved_video_path)
                        )
                    except Exception as source_error:
                        # Una excepción en el registro de origen no puede hacer que un material ya descargado con éxito se considere fallido, y mucho menos
                        # bloquear la generación del video; se conservan el proveedor y el tipo de excepción para localizar el problema después.
                        logger.warning(
                            "failed to prepare material source record: "
                            f"provider={item.provider}, "
                            f"error={type(source_error).__name__}, detail={source_error}"
                        )
                    seconds = min(max_clip_duration, item.duration)
                    total_duration += seconds
                    if total_duration > audio_duration:
                        logger.info(
                            f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                        )
                        break
            except Exception as e:
                logger.error(
                    "failed to download material video: "
                    f"provider={item.provider}, error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, item.url)}"
                )
    logger.success(f"downloaded {len(video_paths)} videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_wavespeed_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """
    Genera fragmento a fragmento el material de WaveSpeed en el orden del guion, y se detiene en cuanto cubre la duración total necesaria.

    Cada palabra clave corresponde de forma natural a un fragmento del guion, y generar implica pagar: generar todo primero para luego elegir pagaría
    por fragmentos que no se usarán. Aquí, cada fragmento generado se descarga de inmediato y se acumula su duración efectiva (igual que el flujo
    de stock, con tope en la duración del clip); al cubrir la duración de voz necesaria no se lanzan más peticiones de
    generación. Los fragmentos que fallan claramente en el remoto se pueden omitir; las tareas de pago con estado incierto o descarga fallida deben
    conservar el ID y abortar la tarea local.
    """
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("WaveSpeed audio duration must be finite") from exc
    if not math.isfinite(required_duration):
        raise ValueError("WaveSpeed audio duration must be finite")
    if required_duration <= 0:
        logger.warning(
            "skip WaveSpeed paid generation because required audio duration "
            f"is not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("WaveSpeed clip duration must be a positive integer") from exc
    if clip_duration <= 0:
        raise ValueError("WaveSpeed clip duration must be a positive integer")

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = generate_videos_wavespeed(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except WaveSpeedUnconfirmedTaskError as e:
            # El estado de la tarea de pago ya enviada es incierto: el remoto puede seguir ejecutándose o haber terminado y cobrado.
            # Seguir pidiendo para las palabras clave siguientes causaría generación y cobro duplicados, así que se detiene aquí mismo
            # y se entrega el prediction id a la capa de tareas para escribirlo en el estado de fallo, para que el usuario recupere el resultado.
            logger.error(
                "stop submitting new wavespeed tasks, the last submitted task "
                f"is unconfirmed: prediction_id={e.prediction_id or 'unknown'}, "
                f"detail={e}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "wavespeed"
            )
            if not saved_video_path:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                prediction_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise WaveSpeedDownloadError(
                    "WaveSpeed generated a paid video but the result could not be "
                    f"downloaded: id={prediction_id or 'unknown'}",
                    prediction_id=prediction_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                # Igual que las fuentes de stock: una excepción en el registro de origen no puede hacer que un material ya pagado y descargado con éxito
                # se considere fallido, y mucho menos bloquear la generación del video.
                logger.warning(
                    "failed to prepare material source record: "
                    f"provider={item.provider}, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            # Se usa >=: cuando la duración acumulada es justo igual a la necesaria ya es suficiente, y generar otra vez
            # costaría un pago más. Las dos comprobaciones, interior y exterior, deben mantener la misma semántica.
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated materials cover the required duration, stop "
                f"generating more clips: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break
    logger.success(f"generated and downloaded {len(video_paths)} videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_seedance_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    source: str = "volcengine_seedance",
    endpoint_snapshot: volcengine_seedance.SeedanceEndpoint | None = None,
) -> List[str]:
    """Genera en orden el material de Seedance y deja de hacer pedidos de pago en cuanto cubre la duración de la voz."""
    service = jalapeno_video if source == "jalapeno" else volcengine_seedance
    generation_options = (
        {"endpoint_snapshot": endpoint_snapshot}
        if source == "jalapeno" and endpoint_snapshot is not None else {}
    )
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    # El bucle de generación de pago debe validar primero las dos duraciones que controlan el número de iteraciones. NaN/Infinity harían que
    # ``total_duration >= audio_duration`` nunca se cumpliera, y una duración de clip no positiva haría que
    # el acumulado no pudiera crecer; ambos podrían crear tareas de pago inútiles para todas las palabras clave.
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance audio duration must be a finite number"
        ) from exc
    if not math.isfinite(required_duration):
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance audio duration must be a finite number"
        )
    if required_duration <= 0:
        logger.warning(
            "skip Seedance paid generation because required audio duration is "
            f"not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance clip duration must be a positive integer"
        ) from exc
    if clip_duration <= 0:
        raise volcengine_seedance.VolcEngineSeedanceError(
            "Seedance clip duration must be a positive integer"
        )

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = service.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
                **generation_options,
            )
        except volcengine_seedance.VolcEngineSeedanceUnconfirmedTaskError as exc:
            # La tarea de pago remota aún puede tener éxito. Se deja de pedir de inmediato y se conserva el ID de la tarea, para que
            # el usuario lo confirme o recupere después en la consola de Ark.
            logger.error(
                "stop submitting new Seedance tasks because the last paid task "
                f"is unconfirmed: task_id={exc.task_id or 'unknown'}, detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except volcengine_seedance.VolcEngineSeedanceError as exc:
            logger.error(f"Seedance generation failed before completion: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, source
            )
            if not saved_video_path:
                # La tarea remota terminó y generó un coste; si la descarga local falla, hay que llevar la tarea remota ID
                # al estado de la tarea, para que el usuario recupere el resultado en la consola de Ark. Aquí se lanza directamente
                # un error específico, que además impide que las palabras clave siguientes creen nuevas tareas de pago.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise volcengine_seedance.VolcEngineSeedanceDownloadError(
                    "Seedance generated a paid video but the result could not be "
                    f"downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            if source == "jalapeno":
                # Aun siendo 480p, distintos gateways pueden devolver 854 u 864 de ancho. Se miden el tamaño y la duración del video
                # descargado, para que el registro de origen sea preciso y no se sobreestime la cobertura del material.
                try:
                    with VideoFileClip(saved_video_path) as downloaded:
                        actual_duration = float(downloaded.duration)
                        width, height = downloaded.size
                    if not math.isfinite(actual_duration) or actual_duration <= 0:
                        raise ValueError("invalid generated video duration")
                except Exception as exc:
                    remote_task_id = str((item.source_info or {}).get("asset_id") or "")
                    _persist_material_sources(task_id, material_sources)
                    raise volcengine_seedance.VolcEngineSeedanceDownloadError(
                        "Jalapeno Cloud generated a paid video but its local media could not be read",
                        task_id=remote_task_id,
                    ) from exc
                item.duration = actual_duration
                item.source_info = dict(item.source_info or {})
                item.source_info["rendition"] = {
                    **item.source_info.get("rendition", {}),
                    "width": width, "height": height,
                }
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider={source}, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated Seedance materials cover the required duration; stop "
                f"submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(
        f"generated and downloaded {len(video_paths)} {source} videos"
    )
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_ofox_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Genera en orden el material de OFox y deja de hacer pedidos de pago en cuanto cubre la duración de la voz."""
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    # El bucle de generación de pago debe validar primero las dos duraciones que controlan el número de iteraciones. NaN/Infinity harían que
    # ``total_duration >= audio_duration`` nunca se cumpliera, y una duración de clip no positiva haría que
    # el acumulado no pudiera crecer; ambos podrían crear tareas de pago inútiles para todas las palabras clave.
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise ofox.OFoxError("OFox audio duration must be a finite number") from exc
    if not math.isfinite(required_duration):
        raise ofox.OFoxError("OFox audio duration must be a finite number")
    if required_duration <= 0:
        logger.warning(
            "skip OFox paid generation because required audio duration is "
            f"not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ofox.OFoxError("OFox clip duration must be a positive integer") from exc
    if clip_duration <= 0:
        raise ofox.OFoxError("OFox clip duration must be a positive integer")

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = ofox.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except ofox.OFoxUnconfirmedTaskError as exc:
            # La tarea de pago remota aún puede tener éxito. Se deja de pedir de inmediato y se conserva el ID de la tarea, para que
            # el usuario lo confirme o recupere después en la consola de OFox.
            logger.error(
                "stop submitting new OFox tasks because the last paid task "
                f"is unconfirmed: task_id={exc.task_id or 'unknown'}, detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except ofox.OFoxError as exc:
            logger.error(f"OFox generation failed before completion: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        # Cuando el remoto declara fallida claramente una palabra clave (por ejemplo, por moderación de contenido) se devuelve una lista vacía: la tarea ya
        # terminó y no hay dudas de cobro, así que se omite ese fragmento y se siguen generando las palabras clave siguientes.
        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "ofox"
            )
            if not saved_video_path:
                # La tarea remota terminó y generó un coste; si la descarga local falla, hay que llevar la tarea remota ID
                # al estado de la tarea, para que el usuario recupere el resultado en la consola de OFox. Aquí se lanza directamente
                # un error específico, que además impide que las palabras clave siguientes creen nuevas tareas de pago.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise ofox.OFoxDownloadError(
                    "OFox generated a paid video but the result could not be "
                    f"downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=ofox, "
                    f"error={type(source_error).__name__}, detail={source_error}"
                )
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated OFox materials cover the required duration; stop "
                f"submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"generated and downloaded {len(video_paths)} OFox videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_muapi_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Generate MuAPI materials until the narration duration is covered."""
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise muapi.MuAPIError(
            "MuAPI audio duration must be a finite number"
        ) from exc
    if not math.isfinite(required_duration):
        raise muapi.MuAPIError("MuAPI audio duration must be a finite number")
    if required_duration <= 0:
        logger.warning(
            "skip MuAPI paid generation because required audio duration is "
            f"not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise muapi.MuAPIError(
            "MuAPI clip duration must be a positive integer"
        ) from exc
    if clip_duration <= 0:
        raise muapi.MuAPIError("MuAPI clip duration must be a positive integer")

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = muapi.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except muapi.MuAPIUnconfirmedTaskError as exc:
            # An ambiguous paid submission must stop the loop: retrying with the
            # next keyword could create a duplicate charge.  The task ID is kept
            # for task-service recovery when the provider returned one.
            logger.error(
                "stop submitting new MuAPI tasks because the last paid task "
                f"is unconfirmed: task_id={exc.task_id or 'unknown'}, "
                f"detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except muapi.MuAPIError as exc:
            logger.error(f"MuAPI generation failed before completion: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "muapi"
            )
            if not saved_video_path:
                # The remote job completed and may already have incurred a
                # charge.  Do not submit a replacement task after download
                # failure; return the remote ID for manual recovery instead.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise muapi.MuAPIDownloadError(
                    "MuAPI generated a paid video but the result could not be "
                    f"downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=muapi, error={type(source_error).__name__}, "
                    f"detail={source_error}"
                )
            try:
                downloaded_duration = _get_downloaded_video_duration(saved_video_path)
            except Exception as duration_error:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise muapi.MuAPIDownloadError(
                    "MuAPI generated a paid video but its downloaded duration "
                    "could not be measured: "
                    f"id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                ) from duration_error
            total_duration += min(clip_duration, downloaded_duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated MuAPI materials cover the required duration; stop "
                f"submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"generated and downloaded {len(video_paths)} MuAPI videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_metaso_minimax_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
) -> List[str]:
    """Genera en orden el material de Metaso MiniMax y deja de hacer pedidos de pago en cuanto cubre la duración de la voz."""
    video_paths: List[str] = []
    material_sources: list[dict[str, Any]] = []

    # El remoto genera como mínimo 4 segundos, pero localmente se sigue recortando y acumulando según la duración de clip del usuario. Se validan antes los parámetros
    # de control del bucle, para evitar que NaN, Infinity o valores no positivos impidan cumplir la condición de parada y acaben
    # enviando todas las palabras clave como tareas de pago.
    try:
        required_duration = float(audio_duration)
    except (TypeError, ValueError) as exc:
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax audio duration must be a finite number"
        ) from exc
    if not math.isfinite(required_duration):
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax audio duration must be a finite number"
        )
    if required_duration <= 0:
        logger.warning(
            "skip Metaso MiniMax paid generation because required audio duration "
            f"is not positive: duration={required_duration}"
        )
        _persist_material_sources(task_id, material_sources)
        return video_paths

    try:
        clip_duration = int(max_clip_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax clip duration must be a positive integer"
        ) from exc
    if clip_duration <= 0:
        raise metaso_minimax.MetasoMiniMaxError(
            "Metaso MiniMax clip duration must be a positive integer"
        )

    total_duration = 0.0
    for search_term in search_terms:
        try:
            video_items = metaso_minimax.generate_videos(
                search_term=search_term,
                minimum_duration=clip_duration,
                video_aspect=video_aspect,
            )
        except metaso_minimax.MetasoMiniMaxUnconfirmedTaskError as exc:
            # Si el estado de la petición o del sondeo es incierto, la tarea remota aún puede tener éxito y cobrarse. Se detiene de inmediato todo el
            # bucle de generación, para que las palabras clave siguientes no sigan pidiendo, y se entrega el ID de la tarea al servicio de tareas para guardarlo.
            logger.error(
                "stop submitting new Metaso MiniMax tasks because the last paid "
                f"task is unconfirmed: task_id={exc.task_id or 'unknown'}, "
                f"detail={exc}"
            )
            _persist_material_sources(task_id, material_sources)
            raise
        except metaso_minimax.MetasoMiniMaxError as exc:
            logger.error(f"Metaso MiniMax generation failed before completion: {exc}")
            _persist_material_sources(task_id, material_sources)
            raise

        for item in video_items:
            saved_video_path = _save_generated_video_with_retry(
                item.url, material_directory, "metaso_minimax"
            )
            if not saved_video_path:
                # La generación exitosa ya produjo un coste; si la descarga falla no se puede seguir creando nuevas tareas para sustituirla.
                # Lanza un error específico con el ID remoto, para el estado de la tarea y la recuperación manual.
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                remote_task_id = str(source_info.get("asset_id") or "").strip()
                _persist_material_sources(task_id, material_sources)
                raise metaso_minimax.MetasoMiniMaxDownloadError(
                    "Metaso MiniMax generated a paid video but the result could "
                    f"not be downloaded: id={remote_task_id or 'unknown'}",
                    task_id=remote_task_id,
                )
            logger.info(f"video saved: {saved_video_path}")
            video_paths.append(saved_video_path)
            try:
                material_sources.append(_material_source_record(item, saved_video_path))
            except Exception as source_error:
                logger.warning(
                    "failed to prepare generated material source record: "
                    f"provider=metaso_minimax, error={type(source_error).__name__}, "
                    f"detail={source_error}"
                )

            # El video final local solo usa la duración de clip elegida por el usuario; aunque H3, por la restricción de duración mínima, haya generado
            # material más largo, la parte no usada no puede contarse como cobertura y hacer que se genere menos imagen de la necesaria.
            total_duration += min(clip_duration, item.duration)
            if total_duration >= required_duration:
                break
        if total_duration >= required_duration:
            logger.info(
                "generated Metaso MiniMax materials cover the required duration; "
                f"stop submitting paid tasks: generated={total_duration:.1f}s, "
                f"required={required_duration:.1f}s"
            )
            break

    logger.success(f"generated and downloaded {len(video_paths)} Metaso MiniMax videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


def _download_videos_by_script_order(
    task_id: str,
    search_terms: List[str],
    search_videos,
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    progress_callback: Callable[[float], None] | None = None,
) -> List[str]:
    """
    Descarga el material en el orden del texto del guion.

    La lógica de descarga por defecto une los candidatos de todas las palabras clave en una gran lista; si la primera
    palabra clave devuelve muchos resultados, la descarga final podría consumir solo material de esa palabra clave, y los temas
    posteriores del guion no entrarían en la línea de tiempo. Aquí se agrupa por palabra clave y se descarga por turnos:
    la ronda 1 toma el 1.er candidato de cada palabra clave, la ronda 2 toma el 2.º candidato de cada palabra clave.
    Así, sin reescribir el motor de composición de video, se intenta que el orden del material siga el orden del texto.
    """
    logger.info("downloading videos with script-order material matching")
    candidate_groups = []
    valid_video_urls = set()
    found_duration = 0.0

    for search_term, video_items in _search_terms_in_parallel(
        search_terms=search_terms,
        search_videos=search_videos,
        minimum_duration=max_clip_duration,
        video_aspect=video_aspect,
    ):
        term_items = []
        for item in video_items:
            if item.url in valid_video_urls:
                continue
            term_items.append(item)
            valid_video_urls.add(item.url)
            found_duration += item.duration

        if term_items:
            candidate_groups.append((search_term, term_items))

    logger.info(
        f"found total ordered video candidates: {sum(len(items) for _, items in candidate_groups)}, "
        f"required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )

    video_paths = []
    material_sources: list[dict[str, Any]] = []
    total_duration = 0.0
    on_downloaded = _covered_duration_reporter(
        progress_callback, audio_duration, max_clip_duration
    )
    # Cada palabra clave avanza su índice de candidatos de forma independiente: solo avanzan los candidatos realmente elegidos para descargar en esta ronda;
    # los no elegidos se conservan para la siguiente ronda, para que no se salten con un índice común a toda la ronda.
    next_candidate_indices = [0] * len(candidate_groups)
    while candidate_groups and total_duration <= audio_duration:
        round_materials = [
            (
                group_position,
                search_term,
                term_items[next_candidate_indices[group_position]],
            )
            for group_position, (search_term, term_items) in enumerate(
                candidate_groups
            )
            if next_candidate_indices[group_position] < len(term_items)
        ]
        if not round_materials:
            break

        selected_materials = _select_materials_until_duration(
            materials=[
                (search_term, item) for _, search_term, item in round_materials
            ],
            max_clip_duration=max_clip_duration,
            audio_duration=audio_duration,
            current_duration=total_duration,
        )
        # _select_materials_until_duration devuelve un subconjunto prefijo de la lista de candidatos,
        # así que los primeros len(selected_materials) son los candidatos realmente intentados en esta ronda,
        # y solo se avanza el índice de los grupos de esos candidatos.
        for group_position, _, _ in round_materials[: len(selected_materials)]:
            next_candidate_indices[group_position] += 1

        downloaded_materials = _download_materials_in_parallel(
            materials=selected_materials,
            material_directory=material_directory,
            on_downloaded=on_downloaded,
        )
        for search_term, item, saved_video_path in downloaded_materials:
            try:
                source_info = (
                    item.source_info if isinstance(item.source_info, dict) else {}
                )
                logger.info(
                    f"downloading ordered {item.provider} video for {search_term!r}: "
                    f"asset_id={source_info.get('asset_id') or 'unknown'}"
                )
                if saved_video_path:
                    logger.info(f"video saved: {saved_video_path}")
                    video_paths.append(saved_video_path)
                    try:
                        material_sources.append(
                            _material_source_record(item, saved_video_path)
                        )
                    except Exception as source_error:
                        logger.warning(
                            "failed to prepare ordered material source record: "
                            f"provider={item.provider}, "
                            f"error={type(source_error).__name__}, "
                            f"detail={source_error}"
                        )
                    total_duration += min(max_clip_duration, item.duration)
                    if total_duration > audio_duration:
                        logger.info(
                            f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                        )
                        break
            except Exception as e:
                logger.error(
                    "failed to download ordered material video: "
                    f"provider={item.provider}, error={type(e).__name__}, "
                    f"detail={_redact_request_error(e, item.url)}"
                )

    logger.success(f"downloaded {len(video_paths)} ordered videos")
    _persist_material_sources(task_id, material_sources)
    return video_paths


if __name__ == "__main__":
    download_videos(
        "test123", ["Money Exchange Medium"], audio_duration=100, source="pixabay"
    )


def _download_videos_infistar_on_demand(
    *, task_id, search_terms, video_aspect, audio_duration, max_clip_duration,
):
    """Pide fragmento a fragmento; se detiene cuando hay material suficiente, y ante estado desconocido o fallo de descarga no compra el siguiente fragmento."""
    try:
        required = float(audio_duration)
        clip_duration = int(max_clip_duration)
    except (ValueError, TypeError, OverflowError) as exc:
        raise infistar_video.InfistarVideoError("Infistar duration must be finite and positive") from exc
    if not math.isfinite(required) or clip_duration <= 0:
        raise infistar_video.InfistarVideoError("Infistar duration must be finite and positive")
    paths = []
    sources = []
    covered = 0.0
    try:
        for term in search_terms:
            if covered >= required:
                break
            item = infistar_video.generate_video(
                search_term=term, video_aspect=video_aspect,
                output_dir=utils.task_dir(task_id),
            )
            paths.append(item.url)
            sources.append(_material_source_record(item, item.url))
            covered += min(clip_duration, item.duration)
    finally:
        _persist_material_sources(task_id, sources)
    return paths
