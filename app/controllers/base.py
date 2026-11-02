import secrets
from typing import Annotated
from uuid import uuid4

from fastapi import Header, Request

from app.config import config
from app.models.exception import HttpException

MAX_TASK_ID_LENGTH = 128


def normalize_task_id(value: object) -> str:
    """Return a log-safe request ID, replacing invalid client input with a UUID."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_TASK_ID_LENGTH
        or not value.isprintable()
    ):
        return str(uuid4())
    return value


def get_task_id(request: Request) -> str:
    return normalize_task_id(request.headers.get("x-task-id"))


def get_api_key(request: Request):
    api_key = request.headers.get("x-api-key")
    return api_key


def get_api_key_values(request: Request) -> list[str]:
    """Devuelve todas las cabeceras API Key de la petición, conservando los valores duplicados para la validación de seguridad."""

    # Starlette Headers ofrece getlist(), que permite distinguir cabeceras duplicadas enviadas por un proxy o cliente.
    # Los dobles ligeros de Request en las pruebas unitarias solo usan un dict normal, así que se mantiene un fallback compatible.
    get_list = getattr(request.headers, "getlist", None)
    if callable(get_list):
        return [value for value in get_list("x-api-key") if isinstance(value, str)]

    api_key = get_api_key(request)
    return [api_key] if isinstance(api_key, str) else []


def verify_token(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
):
    """Decide según la configuración si se valida la API Key.

    Una Key vacía conserva el modo local sin autenticación; cuando el administrador configura explícitamente una Key no vacía,API
    las rutas y las descargas de artefactos de tareas exigen que el cliente envíe ese mismo
    valor en la cabecera ``x-api-key``. La declaración del parámetro también hace que Swagger muestre esa cabecera, facilitando la depuración en entornos protegidos.
    """

    configured_key = config.app.get("api_key", "")
    if configured_key in (None, ""):
        return None

    # El elemento de configuración debe ser una cadena. Aquí se rechazan listas, números y otros tipos erróneos, para evitar que la conversión implícita a cadena
    # produzca comportamientos de autenticación difíciles de detectar; el mensaje de error tampoco incluye la Key real.
    if not isinstance(configured_key, str):
        raise HttpException(
            task_id=get_task_id(request),
            status_code=500,
            message="API authentication is misconfigured",
        )

    # FastAPI El parámetro sirve para declarar x-api-key en OpenAPI; la validación real siempre lee el Request,
    # para poder detectar cuando se envía repetida una cabecera con el mismo nombre. Los clientes normales y los proxies inversos pueden ordenar de forma distinta Header
    # los valores duplicados, por lo que hay que rechazarlos en lugar de adoptar implícitamente el primero o el último.
    token_values = get_api_key_values(request)
    if not token_values and isinstance(x_api_key, str):
        token_values = [x_api_key]

    if len(token_values) != 1:
        raise HttpException(
            task_id=get_task_id(request),
            status_code=401,
            message="invalid API key",
        )

    # compare_digest solo admite ASCII para str. Las cabeceras de la petición son entrada no confiable y un atacante
    # puede enviar caracteres Latin-1 para provocar un TypeError. Codificar todo como bytes UTF-8 conserva
    # la comparación en tiempo constante y admite también Keys Unicode válidas en TOML.
    token = token_values[0]
    if not secrets.compare_digest(
        token.encode("utf-8"), configured_key.encode("utf-8")
    ):
        raise HttpException(
            task_id=get_task_id(request),
            status_code=401,
            message="invalid API key",
        )

    return None
