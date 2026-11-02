"""Application implementation - ASGI."""

import math
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import config
from app.controllers import base
from app.models.exception import HttpException
from app.router import root_api_router
from app.utils import utils


@asynccontextmanager
async def application_lifespan(_: FastAPI):
    """Gestiona de forma centralizada la recuperación al arrancar y los registros de cierre del proceso de la API."""
    logger.info("startup event")

    configured_api_key = config.app.get("api_key", "")
    if configured_api_key in (None, ""):
        logger.warning(
            "API key authentication is disabled; keep the API on a trusted network"
        )
    elif isinstance(configured_api_key, str):
        # Solo registra el ámbito de protección; nunca debe mostrar la Key, su longitud ni un resumen, para evitar que las credenciales entren en el sistema de registros.
        logger.info("API key authentication is enabled for /api/v1 and /tasks")
    else:
        logger.error(
            "API key authentication is misconfigured: app.api_key must be a string"
        )

    # La publicación multiplataforma la ejecuta el pool de hilos del proceso actual y no se recupera tras reiniciar el servicio. Al arrancar, los estados activos Redis
    # cuyo proceso de ejecución se confirma perdido se convierten en fallidos, para evitar tareas que nunca se puedan eliminar.
    from app.services import task as task_service

    task_service.recover_interrupted_cross_posts()

    # Redis queue entries persist across API restarts. No task_done callback
    # exists in the new process to dispatch them, so fill its worker slots now.
    from app.controllers.manager.redis_manager import RedisTaskManager
    from app.controllers.v1 import video as video_controller

    redis_task_manager = None
    if isinstance(video_controller.task_manager, RedisTaskManager):
        redis_task_manager = video_controller.task_manager
        redis_task_manager.resume_queued_tasks()
        redis_task_manager.start_queue_dispatcher()
    try:
        yield
    finally:
        if redis_task_manager is not None:
            redis_task_manager.stop_queue_dispatcher()
        logger.info("shutdown event")


def exception_handler(request: Request, e: HttpException):
    return JSONResponse(
        status_code=e.status_code,
        content=utils.get_response(e.status_code, e.data, e.message),
    )


def validation_exception_handler(request: Request, e: RequestValidationError):
    # Rejected inputs can contain NaN/Infinity, and custom validators attach
    # exception objects to ctx. Neither can be emitted by JSONResponse directly.
    errors = jsonable_encoder(
        e.errors(),
        custom_encoder={
            float: lambda value: value if math.isfinite(value) else str(value),
            Exception: str,
        },
    )
    # Describe the first rejected field instead of always claiming a missing
    # field: out-of-range values and malformed JSON are not "field required".
    message = "invalid request parameters"
    if errors and isinstance(errors[0], dict):
        location = ".".join(str(part) for part in errors[0].get("loc") or ())
        detail = errors[0].get("msg") or message
        message = f"{location}: {detail}" if location else str(detail)
    return JSONResponse(
        status_code=400,
        content=utils.get_response(status=400, data=errors, message=message),
    )


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _normalize_allowed_origin(raw_origin: str) -> str | None:
    """Normaliza el elemento de configuración a la forma de Origin que envía realmente el navegador.

    Origin La cabecera está fijada por el RFC 6454 como ``scheme://host[:port]``: sin ruta, sin barra
    final, scheme y host ya normalizados en minúsculas, y omitiendo el puerto por defecto del protocolo. Lo que el usuario copia de la barra de direcciones
    del navegador, como ``https://frontend.example/``, o lo que escribe con el puerto por defecto explícito
    ``https://frontend.example:443``, no coincide carácter a carácter con lo que el navegador envía realmente
    ``https://frontend.example`` , y la lista blanca falla en silencio: el frontend solo ve
    CORS el error, el servidor solo deja una línea de log blocked, y ninguno de los dos apunta a la propia configuración.

    Devolver ``None`` significa que esa forma no puede coincidir con ningún Origin (falta el scheme, el scheme no es
    http/https, el host está vacío o contiene espacios, el literal IPv6 está mal formado, o el puerto es ilegal o está fuera de rango); el llamador debe
    descartarla y emitir una advertencia. Los fallos de análisis deben convertirse en ``None``: esta función se ejecuta al importar el módulo, y dejar que
    ``urlsplit()`` el ``ValueError`` se propague equivale a que toda la API no pueda arrancar.
    """

    if raw_origin == "*":
        return raw_origin

    try:
        parsed = urlsplit(raw_origin)
        scheme = parsed.scheme.lower()
        # netloc solo se valida al leer hostname / port: los literales IPv6 mal formados los urlsplit()
        # lanza, y los puertos no numéricos o fuera de 0-65535 lanzan al leer port.
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None

    # Solo http/https tienen semántica de Origin del navegador, y ambos tienen un puerto por defecto omitible.
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    if any(character.isspace() for character in host):
        return None

    # Minúsculas explícitas, sin depender de cómo trate urlsplit las mayúsculas en distintas versiones de Python.
    host = host.lower()
    # Al serializar el Origin, el navegador conserva los corchetes de los literales IPv6; ``:443``, ``:0443``,
    # ``:80`` y formas similares del puerto por defecto deben normalizarse sin puerto para coincidir con el frontend real.
    if ":" in host:
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def parse_cors_allowed_origins(raw_origins: str | None) -> list[str]:
    """Analiza la lista blanca de orígenes cross-origin del navegador.

    CORS Solo restringe el JavaScript cross-origin en el navegador; no afecta a curl, Postman,n8n
    ni a SDKs del lado del servidor. Si no está configurado devuelve una lista vacía, es decir, por defecto no se permite el acceso cross-origin; cuando el usuario realmente
    despliega un frontend web independiente, debe habilitarlo explícitamente con ``CORS_ALLOWED_ORIGINS``.

    Cada origen se normaliza primero a la forma canónica de la cabecera Origin, por lo que ``https://a.example/``,
    ``HTTPS://A.Example`` y ``https://a.example:443`` con el puerto por defecto explícito producen el mismo
    resultado; las formas que no pueden coincidir con ningún Origin se descartan con una advertencia. Una entrada mal formada solo se descarta a sí misma
    y no interrumpe el análisis: esta función se llama al importar el módulo y una excepción haría que la API no pudiera arrancar.
    """

    if not raw_origins:
        return []

    # Elimina los espacios a ambos lados de los elementos separados por comas e ignora los vacíos, para evitar que formatos habituales de variables de entorno
    # ``https://a.example, https://b.example,`` produzcan orígenes que nunca coincidirán.
    origins: list[str] = []
    for candidate in raw_origins.split(","):
        item = candidate.strip()
        if not item:
            continue
        origin = _normalize_allowed_origin(item)
        if origin is None:
            logger.warning(
                f"ignoring configured CORS origin that cannot match a browser "
                f"Origin header: {item!r}"
            )
            continue
        if origin not in origins:
            origins.append(origin)
    return origins


def configure_cors(instance: FastAPI, allowed_origins: list[str]) -> None:
    """Configura CORS según la lista blanca explícita; una lista vacía mantiene la política same-origin por defecto."""

    if not allowed_origins:
        logger.info(
            "browser cross-origin API access is disabled; set "
            "CORS_ALLOWED_ORIGINS to enable trusted origins"
        )
        return

    allow_all_origins = "*" in allowed_origins
    configured_api_key = config.app.get("api_key", "")
    if allow_all_origins and configured_api_key in (None, ""):
        # ``*`` es un modo de compatibilidad elegido explícitamente por el usuario, así que no se fuerza el rechazo del arranque; pero sin autenticación
        # permite que cualquier web lea y llame a la API, por lo que debe dejar una advertencia de seguridad localizable.
        logger.warning(
            "CORS allows every browser origin while API key authentication is "
            "disabled; configure app.api_key or restrict CORS_ALLOWED_ORIGINS"
        )

    instance.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        # Starlette refleja cualquier Origin cuando ``*`` y credentials están activados a la vez.
        # El modo comodín no necesita autenticación por cookies, así que desactiva credentials; los orígenes explícitos
        # mantienen el comportamiento anterior para no afectar al modo de credentials de los frontends web independientes existentes.
        allow_credentials=not allow_all_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        # Cuando un frontend HTTPS remoto accede a una API local o de red local, los navegadores modernos envían además
        # Private Network Access una petición preflight. Solo los orígenes exactos de la lista blanca obtienen permiso;
        # el modo comodín sigue rechazándola, para evitar que cualquier web sondee los servicios de red privada del usuario.
        allow_private_network=not allow_all_origins,
    )


def is_browser_origin_allowed(
    request: Request, allowed_origins: list[str]
) -> bool:
    """Determina si el origen de la petición del navegador es el mismo origen o un origen explícito de la lista blanca."""

    origin = request.headers.get("origin")
    if not origin:
        # curl, Postman, n8n y los SDKs de servidor normalmente no envían Origin. Estas peticiones se conservan
        # para evitar que una corrección de seguridad cambie por error el contrato de llamada de los clientes API existentes.
        return True
    if "*" in allowed_origins or origin in allowed_origins:
        return True

    # Los navegadores también pueden enviar Origin en un POST del mismo origen. Solo se comparan scheme + authority, ignorando
    # la ruta y los parámetros de consulta; si un proxy inverso no reenvía correctamente el scheme/host públicos, se puede
    # CORS_ALLOWED_ORIGINS declarar explícitamente el origen externo, evitando depender de cabeceras reenviadas no confiables.
    request_url = urlsplit(str(request.url))
    request_origin = f"{request_url.scheme}://{request_url.netloc}"
    return origin == request_origin


def configure_browser_access(instance: FastAPI, allowed_origins: list[str]) -> None:
    """Configura a la vez la protección de Origin del servidor y la política de respuesta CORS del navegador."""

    @instance.middleware("http")
    async def reject_untrusted_browser_origin(request: Request, call_next):
        """Rechaza activamente orígenes de navegador no confiables, cubriendo peticiones simples que no requieren preflight CORS."""

        if not is_browser_origin_allowed(request, allowed_origins):
            origin = request.headers.get("origin", "")
            logger.warning(
                f"blocked untrusted browser origin: method={request.method}, "
                f"path={request.url.path}, origin={origin}"
            )
            return JSONResponse(
                status_code=403,
                content=utils.get_response(
                    status=403,
                    message="cross-origin browser request is not allowed",
                ),
            )

        return await call_next(request)

    # CORS El middleware registrado al final queda por fuera de la protección de Origin: los preflight confiables pueden tener éxito directamente,
    # los no confiables los rechaza CORS; las peticiones reales sin preflight siguen pasando por la protección 403 de arriba.
    configure_cors(instance, allowed_origins)


def get_application() -> FastAPI:
    """Initialize FastAPI application.

    Returns:
       FastAPI: Application object instance.

    """
    instance = FastAPI(
        title=config.project_name,
        description=config.project_description,
        version=config.project_version,
        debug=False,
        lifespan=application_lifespan,
    )
    instance.include_router(root_api_router)
    instance.add_exception_handler(HttpException, exception_handler)
    instance.add_exception_handler(RequestValidationError, validation_exception_handler)
    return instance


app = get_application()


@app.middleware("http")
async def protect_generated_task_files(request: Request, call_next):
    """Protege la ruta estática de artefactos de tareas, evitando descargas directas que se salten la autenticación de la API.

    ``/tasks`` se monta de forma independiente con StaticFiles y no puede reutilizar las dependencias de APIRouter,
    por eso se llama al mismo verify_token dentro del middleware. La función de autenticación deja pasar
    api_key cuando no está configurado; las peticiones preflight OPTIONS también se dejan al middleware CORS.
    """

    request_path = request.url.path
    is_task_file = request_path == "/tasks" or request_path.startswith("/tasks/")
    if is_task_file and request.method != "OPTIONS":
        try:
            base.verify_token(request)
        except HttpException as exception:
            return exception_handler(request, exception)

    return await call_next(request)


# Por defecto se sigue la política same-origin del navegador; solo se abre el acceso cross-origin cuando el usuario configura explícitamente orígenes web confiables.
cors_allowed_origins = parse_cors_allowed_origins(
    os.getenv("CORS_ALLOWED_ORIGINS", "")
)
configure_browser_access(app, cors_allowed_origins)

task_dir = utils.task_dir()
app.mount("/tasks", StaticFiles(directory=task_dir, html=True), name="")

public_dir = utils.public_dir()
app.mount("/", StaticFiles(directory=public_dir, html=True), name="")
