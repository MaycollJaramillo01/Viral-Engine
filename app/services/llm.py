import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from time import perf_counter
from typing import List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config
from app.models.llm_provider import DEFAULT_LLM_PROVIDER_ID, get_llm_provider
from app.utils import utils
from app.utils.request_logging import trace_sdk_call

_max_retries = 5
MIN_SCRIPT_PARAGRAPH_NUMBER = 1
MAX_SCRIPT_PARAGRAPH_NUMBER = 10
MAX_SCRIPT_PROMPT_LENGTH = 2000
MAX_SCRIPT_SYSTEM_PROMPT_LENGTH = 8000
# «Probar conexión del modelo» solo envía una petición mínima. El SDK de OpenAI tiene por defecto un tiempo de espera de 600 segundos y reintenta automáticamente
# 2 veces; cuando el endpoint no es accesible (fallo de proxy, cortafuegos o DNS), el botón de prueba se queda colgado mucho tiempo
# y al final solo deja un "Request timed out." (ver issue #1642). Aquí se da a la prueba de conexión un
# tiempo de espera corto y acotado y se desactivan los reintentos del SDK, para que el fallo sea rápido y diagnosticable; la cadena formal de generación de guiones mantiene
# SDK el comportamiento por defecto, para no perjudicar a modelos de inferencia lentos ni al arranque en frío de Ollama local.
LLM_CONNECTION_TEST_TIMEOUT_SECONDS = 30.0
LLM_CONNECTION_TEST_MAX_RETRIES = 0
# Estos fragmentos provienen de APITimeoutError / APIConnectionError del SDK de OpenAI y de
# httpx los textos de excepciones de conexión de bajo nivel; si coincide cualquiera se trata como «endpoint no accesible».
_CONNECTION_FAILURE_MARKERS = (
    "timed out",
    "timeout",
    "connection error",
    "connection refused",
    "connection reset",
    "connection aborted",
    "failed to connect",
)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_UNOPENED_THINK_BLOCK_RE = re.compile(r"^.*</think\s*>", re.IGNORECASE | re.DOTALL)
_URL_USERINFO_RE = re.compile(
    r"((?:https?|wss?)://)([^/\s?#@]+@)", re.IGNORECASE
)
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password)=)([^&#\s]+)",
    re.IGNORECASE,
)

DEFAULT_SCRIPT_SYSTEM_PROMPT = """
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.
""".strip()

# Con script_refine_rounds > 0, el modelo critica su propio borrador con esta lista y lo reescribe.
SCRIPT_REFINE_PROMPT = """
# Role: Short Video Script Editor

Critique the draft below against this checklist, then rewrite it fixing every weak point:
1. the first sentence is a hook (a shocking fact, a question or a twist) that can be spoken in under 3 seconds.
2. short, natural sentences meant to be read aloud; no filler, no cliches, no greetings.
3. build tension and land a turn or a reveal before the end.
4. {facts_rule}
5. inform without glorifying crime: never praise criminals or explain how to commit a crime, and present accusations as accusations.
6. close with a question that invites the viewer to comment.
7. keep the language, the number of paragraphs and the requirements of the brief.

Return only the rewritten script as plain text: no critique, no title, no markdown.
""".strip()
SCRIPT_RESEARCH_DEFAULT_MODEL = "gemini-flash-latest"

# Claude Code CLI usa por defecto el prompt de sistema del agente de programación, con muchas restricciones ajenas a la redacción de textos
# que desviarían la generación de guiones y palabras clave, así que al llamarlo se reemplaza por completo.
CLAUDE_CODE_SYSTEM_PROMPT = (
    "You are a concise copywriter. Follow the user's instructions and output "
    "format exactly, and output nothing else."
)
CLAUDE_CODE_DEFAULT_TIMEOUT = 300.0
# `--tools ""` desactiva todas las herramientas integradas; `--safe-mode` desactiva CLAUDE.md, skills, hooks,
# plugins, MCP y toda personalización a nivel de usuario, manteniendo la autenticación, la selección de modelo y los permisos funcionando con normalidad.
# Ambos requieren un CLI reciente; las versiones antiguas salen con "unknown option", y el llamador lo convierte en un aviso claro.
CLAUDE_CODE_MIN_CLI_VERSION = "2.1.260"
# Estas variables de entorno hacen que el CLI use una API Key o un proveedor de terceros (Bedrock, Vertex, Foundry,
# Mantle, Gateway, etc.), saltándose el inicio de sesión de la suscripción y generando cobros adicionales. Enumerarlas una a una facilita olvidar alguna,
# y además el CLI seguirá añadiendo proveedores, así que se eliminan por clases según el prefijo:
#   ANTHROPIC_*           API Key, Auth Token, Base URL, endpoints de cada proveedor y Profile
#   CLAUDE_CODE_USE_*     interruptores de proveedor
#   CLAUDE_CODE_SKIP_*_AUTH  interruptores para omitir la autenticación del proveedor
CLAUDE_CODE_CONFLICTING_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_")
CLAUDE_CODE_CONFLICTING_ENV_VARS = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "CLAUDE_CODE_GATEWAY_TOKEN_FILE_DESCRIPTOR",
)
# Estas dos clases de variables no se pueden eliminar:
#   CLAUDE_CODE_OAUTH_TOKEN es la única forma de autenticación de suscripción dentro del contenedor (no coincide con los prefijos de arriba);
#   *_CONFIG_DIR solo indica dónde se guardan las credenciales; eliminarla invalidaría la suscripción ya iniciada.
CLAUDE_CODE_PRESERVED_ENV_VARS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_CONFIG_DIR",
    "CLAUDE_CONFIG_DIR",
)


def _is_conflicting_claude_code_env(name: str) -> bool:
    """Determina si una variable de entorno haría que el CLI pase del inicio de sesión de suscripción a otra forma de autenticación."""
    if name in CLAUDE_CODE_PRESERVED_ENV_VARS:
        return False
    if name in CLAUDE_CODE_CONFLICTING_ENV_VARS:
        return True
    if name.startswith(CLAUDE_CODE_CONFLICTING_ENV_PREFIXES):
        return True
    return name.startswith("CLAUDE_CODE_SKIP_") and name.endswith("_AUTH")


def coerce_claude_code_timeout(value, config_key: str = "claude_code_timeout"):
    """
    Convierte el valor de tiempo de espera de la configuración en un número de segundos positivo y finito.

    TOML puede escribirse como `claude_code_timeout = 300` (int/float) o como
    `"300"`(cadena), así que no se puede llamar directamente a `strip()`. nan / inf harían que
    `subprocess.run(timeout=...)` se bloqueara para siempre, así que también se rechazan.
    """
    if value is None:
        return CLAUDE_CODE_DEFAULT_TIMEOUT

    if isinstance(value, bool):
        # bool es una subclase de int, pero True segundos claramente no es el tiempo de espera que quiere el usuario.
        raise ValueError(f"{config_key} must be a number of seconds, got {value!r}")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return CLAUDE_CODE_DEFAULT_TIMEOUT
        try:
            seconds = float(text)
        except ValueError:
            raise ValueError(
                f"{config_key} must be a number of seconds, got {value!r}"
            ) from None
    elif isinstance(value, (int, float)):
        seconds = float(value)
    else:
        raise ValueError(f"{config_key} must be a number of seconds, got {value!r}")

    if not math.isfinite(seconds):
        raise ValueError(f"{config_key} must be a finite number, got {value!r}")
    if seconds <= 0:
        raise ValueError(f"{config_key} must be greater than 0, got {value!r}")
    return seconds


def _resolve_provider_field_value(raw_value, default_value):
    """
    Solo se recurre al valor por defecto del Registry cuando «no está configurado».

    Antes se usaba `raw or default_value`, que también trataba como no configurados valores válidos como 0 y false
    y los reemplazaba: `claude_code_timeout = 0` se cambiaba en silencio a 300, mientras que `"0"` daba error. El valor por defecto
    solo se aplica con None o una cadena en blanco, para que la validación de configuración sea coherente con todas las formas.
    """
    if raw_value is None:
        return default_value
    if isinstance(raw_value, str) and not raw_value.strip():
        return default_value
    return raw_value


def build_claude_code_env(base_env=None):
    """
    Construye el entorno del subproceso dependiendo solo del inicio de sesión de la suscripción.

    Devuelve (diccionario de variables de entorno, lista de nombres de variables eliminadas). Se eliminan las variables que cambiarían la forma de autenticación o el proveedor
    ; `CLAUDE_CODE_OAUTH_TOKEN` debe conservarse: dentro del contenedor no hay keychain y el CLI solo puede
    completar la autenticación de suscripción con ella.
    """
    env = dict(os.environ if base_env is None else base_env)
    removed = sorted(name for name in env if _is_conflicting_claude_code_env(name))
    for name in removed:
        env.pop(name, None)
    return env, removed


def _normalize_text_response(content, llm_provider: str) -> str:
    # Distintos SDKs de LLM pueden devolver None, una cadena vacía
    # o incluso objetos que no son cadenas ante excepciones o bloqueos. Aquí se hace una validación de respaldo unificada, para evitar que al llamar después directamente a
    # `.replace()` se lance un error de atributo como `NoneType`.
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    # MiniMax M3, los modelos reasoning como DeepSeek R1 pueden envolver su razonamiento interno en
    # `<think>...</think>` al devolverlo. El guion y las palabras clave del video solo necesitan el texto final legible;
    # si no se limpia de forma unificada en la capa de servicio, la WebUI, los subtítulos y la voz tratarían el proceso de pensamiento como texto.
    content = _THINK_BLOCK_RE.sub("", content)
    # DeepSeek R1, las plantillas de conversación de modelos como QwQ ponen el `<think>` inicial en el prompt,
    # vLLM/Ollama y el texto que devuelven gateways como solo queda como `razonamiento...</think>respuesta`. Tras limpiar los pares,
    # todo lo que esté antes de una etiqueta de cierre restante es razonamiento y tampoco puede entrar en el guion.
    content = _UNOPENED_THINK_BLOCK_RE.sub("", content)
    content = _UNCLOSED_THINK_BLOCK_RE.sub("", content).strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    # El ``strip()`` anterior ya limpió los espacios al principio y al final. Aquí deben conservarse los saltos de línea simples y
    # dobles del texto: la generación del guion distingue párrafos con dobles saltos de línea, y el procesamiento de subtítulos también lee el texto del usuario por líneas.
    return content


def _sanitize_error_message(error: object) -> str:
    """
    Limpia los mensajes de error devueltos a la WebUI/API, para evitar filtrar credenciales de un base_url personalizado.

    Algunos SDKs OpenAI-compatible incluyen la URL de la petición tal cual en el mensaje de excepción. Si el usuario configuró
    `https://user:pass@example.com/v1` para un gateway proxy, devolver directamente `str(e)`
    expondría la contraseña a la página, al llamador de la API o a logs posteriores. Aquí solo se procesa el texto del error, sin cambiar
    la dirección real de la petición, para no afectar a la cadena de llamadas normal.
    """
    message = str(error)
    message = _URL_USERINFO_RE.sub(r"\1***:***@", message)
    message = _SENSITIVE_QUERY_RE.sub(r"\1***", message)
    return message


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # OpenAI Las interfaces compatibles, en casos anómalos, pueden devolver un objeto de respuesta sin choices,
    # o con choices/message/content vacíos.
    # Aquí se valida la estructura de forma unificada, para evitar `NoneType is not subscriptable`
    # este tipo de errores de acceso a atributos de bajo nivel.
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _get_response_field(value, key: str):
    """Lee campos de forma compatible tanto con dict como con objetos de respuesta del SDK."""
    if isinstance(value, dict):
        return value.get(key)

    try:
        return value[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(value, key, None)


def _sdk_transport_options(request_timeout, max_retries) -> dict:
    """Convierte las restricciones opcionales de tiempo de espera/reintentos en parámetros del constructor del cliente del SDK de OpenAI.

    Si ambos parámetros son None se devuelve un diccionario vacío y el cliente sigue usando los valores por defecto del SDK (600 segundos de espera,
    2 reintentos automáticos), así que las cadenas normales como la generación de guiones no cambian; solo cuando un llamador como la prueba de conexión
    que necesita fallar rápido pasa explícitamente valores acotados se reduce el presupuesto de la petición.
    """
    options = {}
    if request_timeout is not None:
        options["timeout"] = request_timeout
    if max_retries is not None:
        options["max_retries"] = max_retries
    return options


def _explain_connection_test_failure(error_message: str) -> str:
    """Añade pistas de diagnóstico a los fallos de tiempo de espera/conexión, en lugar de mostrar solo el texto original del SDK.

    issue #1642 la API del usuario funcionaba; el fallo se debía al enlace entre la máquina local y el endpoint
    (proxy, cortafuegos o DNS), pero "Request timed out." no apuntaba a ninguna causa posible.
    Los demás errores (autenticación, cuota, etc.) se devuelven con el texto original.
    """
    normalized = (error_message or "").lower()
    if not any(marker in normalized for marker in _CONNECTION_FAILURE_MARKERS):
        return error_message
    return (
        f"{error_message} (the request did not complete within "
        f"{LLM_CONNECTION_TEST_TIMEOUT_SECONDS:.0f}s). Please check that the "
        "configured base URL is reachable from this machine, and that no proxy "
        "or firewall is blocking the request."
    )


def _extract_qwen_generation_text(response) -> str:
    """
    Extrae el texto de la respuesta de DashScope Generation.

    Qwen Al llamar con `messages` se devuelve una estructura de chat:
    `output.choices[0].message.content`; solo la forma antigua de completion devuelve
    `output.text`. Aquí se admiten ambas rutas, para que cuando `output.text` sea None
    no se siga con `.replace()` provocando un AttributeError imposible de diagnosticar.
    """
    output = _get_response_field(response, "output")
    choices = _get_response_field(output, "choices") if output else None
    if choices is not None:
        if not choices:
            logger.warning("Qwen returned an empty choices list")
            raise ValueError("[qwen] returned empty choices")

        first_choice = choices[0]
        message = _get_response_field(first_choice, "message")
        content = _get_response_field(message, "content") if message else None
        if content is not None:
            return _normalize_text_response(content, "qwen")

    text = _get_response_field(output, "text") if output else None
    return _normalize_text_response(text, "qwen")


def _generate_response(
    prompt: str,
    app_config=None,
    request_timeout: float | None = None,
    max_retries: int | None = None,
) -> str:
    sdk_client = None
    sdk_stream = None
    try:
        # WebUI Permite al usuario preparar el siguiente texto durante la generación del video. El llamador puede pasar la instantánea de configuración
        # del momento del envío, para que durante los reintentos de la petición al modelo no se cambie, porque una tarea en segundo plano terminó y aplicó nueva configuración,
        # a otro Provider, Base URL o modelo.
        runtime_app_config = app_config if app_config is not None else config.app
        llm_provider = str(
            runtime_app_config.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
        ).lower()
        provider = get_llm_provider(llm_provider)
        if provider is None:
            raise ValueError(f"{llm_provider}: unsupported llm provider")

        logger.info(f"llm provider: {llm_provider}")
        api_key = runtime_app_config.get(provider.config_key("api_key"), "")
        configured_model = runtime_app_config.get(provider.config_key("model_name"), "")
        model_name = provider.resolve_model_name(configured_model)
        if configured_model and model_name != configured_model:
            logger.warning(
                f"{llm_provider} model '{configured_model}' is deprecated, "
                f"fallback to '{model_name}'"
            )
        configured_base_url = runtime_app_config.get(
            provider.config_key("base_url"), ""
        )
        base_url = provider.resolve_base_url(configured_base_url)
        if configured_base_url and configured_base_url.strip().rstrip("/") in {
            url.rstrip("/") for url in provider.deprecated_base_urls
        }:
            logger.warning(
                f"{llm_provider} base URL '{configured_base_url}' is deprecated, "
                f"fallback to '{base_url}'"
            )
        adapter = provider.adapter
        api_version = ""

        # Ollama La dirección por defecto depende de si se ejecuta dentro de un contenedor, así que no se puede guardar como valor Registry
        # estático; el Registry sigue encargándose del modelo y las reglas obligatorias, y las diferencias del entorno de ejecución se resuelven aquí.
        if llm_provider == "ollama":
            api_key = "ollama"
            if not base_url:
                base_url = config.get_default_ollama_base_url()

        if adapter == "azure":
            api_version = runtime_app_config.get(
                provider.config_key("api_version"), "2024-02-15-preview"
            )

        extra_values = {
            field.config_suffix: _resolve_provider_field_value(
                runtime_app_config.get(provider.config_key(field.config_suffix)),
                field.default_value,
            )
            for field in provider.extra_fields
        }

        if provider.requires_api_key and not api_key:
            raise ValueError(
                f"{llm_provider}: api_key is not set, please set it in the config.toml file."
            )
        if provider.requires_model_name and not model_name:
            raise ValueError(
                f"{llm_provider}: model_name is not set, please set it in the config.toml file."
            )
        if provider.requires_base_url and not base_url:
            raise ValueError(
                f"{llm_provider}: base_url is not set, please set it in the config.toml file."
            )

        for field in provider.extra_fields:
            if field.required and not extra_values[field.config_suffix]:
                raise ValueError(
                    f"{llm_provider}: {field.config_suffix} is not set, "
                    "please set it in the config.toml file."
                )

        if adapter == "qwen":
            import dashscope
            from dashscope.api_entities.dashscope_response import GenerationResponse

            response = trace_sdk_call(
                llm_provider, base_url, dashscope.Generation.call,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
            )
            if response:
                if isinstance(response, GenerationResponse):
                    status_code = response.status_code
                    if status_code != 200:
                        raise Exception(
                            f'[{llm_provider}] returned an error response: "{response}"'
                        )

                    return _extract_qwen_generation_text(response)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}"'
                    )
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        if adapter == "gemini":
            from google import genai
            from google.genai import types

            http_options = types.HttpOptions(base_url=base_url) if base_url else None
            generation_config = types.GenerateContentConfig(
                temperature=0.5,
                top_p=1,
                top_k=1,
                max_output_tokens=2048,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                ],
            )

            try:
                # El nuevo google-genai expone los servicios del modelo mediante un Client unificado. El gestor de contexto
                # cierra la conexión HTTP subyacente al terminar la petición, para no acumular conexiones al generar con frecuencia.
                with genai.Client(
                    api_key=api_key,
                    http_options=http_options,
                ) as client:
                    response = trace_sdk_call(
                        llm_provider, base_url, client.models.generate_content,
                        model=model_name,
                        contents=prompt,
                        config=generation_config,
                    )
                generated_text = response.text
            except (AttributeError, IndexError, ValueError) as e:
                logger.warning(f"gemini returned invalid response content: {str(e)}")
                raise ValueError(f"[{llm_provider}] returned invalid response content")

            return _normalize_text_response(generated_text, llm_provider)

        if adapter == "cloudflare_ai_gateway":
            account_id = extra_values["account_id"]
            gateway_id = extra_values["gateway_id"]
            # Cloudflare La API REST de AI Gateway recomendada actualmente es compatible con el SDK de OpenAI.
            # Account ID se usa para construir el endpoint unificado, y el Gateway ID se elige mediante una cabecera; aquí
            # ya no se llama a la interfaz específica /ai/run/{model} de Workers AI.
            client = sdk_client = OpenAI(
                api_key=api_key,
                base_url=(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
                ),
                default_headers={"cf-aig-gateway-id": gateway_id},
                **_sdk_transport_options(request_timeout, max_retries),
            )
            response = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "litellm":
            import litellm

            if not model_name:
                raise ValueError(
                    f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                )

            response = trace_sdk_call(
                llm_provider, base_url, litellm.completion,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                drop_params=True,
            )

            if not response:
                raise ValueError(f"[{llm_provider}] returned empty response")
            if not getattr(response, "choices", None):
                raise ValueError(f"[{llm_provider}] returned empty response")

            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "azure":
            # Azure OpenAI SDK usa `azure_endpoint` y `api_version` para generar una dirección de petición específica,
            # y no puede reutilizar la lógica de inicialización con `base_url` del OpenAI-compatible normal de abajo.
            # Aquí la petición se completa dentro de la rama de Azure y se devuelve de inmediato, para evitar que el cliente sea sobrescrito después fallback
            # , haciendo que las credenciales de Azure configuradas pasen la validación pero no se usen en la petición real.
            logger.info(f"requesting azure chat completion, model: {model_name}")
            client = sdk_client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
                **_sdk_transport_options(request_timeout, max_retries),
            )
            response = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    return _extract_chat_completion_text(response, llm_provider)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                        f"connection and try again."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                )

        if adapter in {"openai_codex", "openai_chatgpt"}:
            if adapter == "openai_chatgpt":
                from app.services import chatgpt_oauth as subscription_provider
            else:
                from app.services import codex_oauth as subscription_provider

            timeout_seconds = coerce_claude_code_timeout(
                extra_values.get("timeout"), provider.config_key("timeout")
            )
            if request_timeout is not None:
                timeout_seconds = min(timeout_seconds, request_timeout)
            return _normalize_text_response(
                subscription_provider.generate_text(prompt, model_name, timeout_seconds),
                llm_provider,
            )

        if adapter == "claude_code":
            # Claude Las suscripciones (Pro / Max / Team) no emiten API Keys; sus credenciales solo las puede usar
            # Claude Code el cliente oficial. Aquí no se llama directamente a la API de Anthropic,
            # sino que se invoca en modo headless el claude CLI ya autenticado en la máquina (`claude -p`);
            # el CLI completa la autenticación y la generación del guion solo consume el texto que devuelve.
            configured_cli = (extra_values.get("cli_path") or "").strip() or "claude"
            cli_path = shutil.which(configured_cli)
            if not cli_path and os.path.isfile(configured_cli):
                cli_path = configured_cli
            if not cli_path:
                raise ValueError(
                    f"{llm_provider}: claude CLI not found ('{configured_cli}'), "
                    f"install it in the runtime or set "
                    f"{provider.config_key('cli_path')} in the config.toml file."
                )

            try:
                timeout_seconds = coerce_claude_code_timeout(
                    extra_values.get("timeout"), provider.config_key("timeout")
                )
            except ValueError as timeout_error:
                raise ValueError(f"{llm_provider}: {timeout_error}") from None

            # prompt se pasa por stdin y no por la línea de comandos: en Windows, lo instalado con npm
            # claude es claude.cmd, y cmd.exe corta los argumentos en el primer salto de línea, perdiendo el texto multilínea
            # prompt y los argumentos de aislamiento posteriores.
            command = [
                cli_path,
                "-p",
                "--output-format",
                "json",
                "--system-prompt",
                CLAUDE_CODE_SYSTEM_PROMPT,
                # Desactiva todas las herramientas integradas, para que solo genere texto.
                "--tools",
                "",
                # Desactiva CLAUDE.md, skills, hooks, plugins, MCP y demás personalizaciones de usuario;
                # la autenticación y la selección de modelo no se ven afectadas (no se puede usar --bare, que desactiva OAuth).
                "--safe-mode",
            ]
            # Si el nombre del modelo está vacío se usa el modelo por defecto del propio CLI, para que un modelo codificado aquí ID
            # no deje de funcionar cuando cambien los modelos disponibles en la suscripción.
            if model_name:
                command += ["--model", model_name]

            cli_env, removed_env = build_claude_code_env()
            if removed_env:
                # Solo se registran los nombres de las variables, no sus valores, para no escribir secretos en los logs.
                logger.warning(
                    f"{llm_provider}: ignoring conflicting environment variables "
                    f"so the subscription login is used: {', '.join(removed_env)}"
                )

            logger.info(f"invoking claude cli, model: {model_name or 'cli default'}")
            # CLI lee el CLAUDE.md y la configuración del proyecto en el directorio de trabajo, y ese contenido contaminaría
            # el texto resultante, así que se ejecuta siempre en un directorio temporal vacío.
            with tempfile.TemporaryDirectory() as work_dir:
                try:
                    completed = subprocess.run(
                        command,
                        input=prompt,
                        capture_output=True,
                        text=True,
                        # The CLI always emits UTF-8. Without an explicit encoding,
                        # text=True decodes with the system locale (e.g. cp1252 on
                        # non-English Windows), so every non-ASCII character reaches
                        # the script as mojibake.
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout_seconds,
                        cwd=work_dir,
                        env=cli_env,
                    )
                except subprocess.TimeoutExpired:
                    raise Exception(
                        f"[{llm_provider}] claude cli timed out after "
                        f"{timeout_seconds:.0f}s"
                    )

            # Los fallos como sesión no iniciada o uso agotado también devuelven JSON (`is_error` verdadero,
            # `result` es el motivo legible), solo que con código de salida distinto de 0. Por eso primero se analiza stdout,
            # y solo si no se obtiene JSON se recurre al código de salida y a stderr.
            stdout = (completed.stdout or "").strip()
            try:
                payload = json.loads(stdout) if stdout else None
            except json.JSONDecodeError:
                payload = None

            if payload is None:
                detail = (completed.stderr or stdout or "").strip()
                if "unknown option" in detail.lower():
                    raise Exception(
                        f"[{llm_provider}] the installed claude CLI does not support "
                        f"the required isolation flags; upgrade to "
                        f"{CLAUDE_CODE_MIN_CLI_VERSION} or newer: {detail[:300]}"
                    )
                if completed.returncode != 0:
                    raise Exception(
                        f"[{llm_provider}] claude cli exited with code "
                        f"{completed.returncode}: {detail[:500]}"
                    )
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{detail[:500]}"'
                )

            if payload.get("is_error") or completed.returncode != 0:
                reason = str(payload.get("result") or "").strip() or (
                    f"claude cli exited with code {completed.returncode}"
                )
                # En el contenedor no se puede ejecutar el /login interactivo, así que aquí se indica directamente la forma de autenticación disponible.
                if "login" in reason.lower():
                    reason += (
                        " (run `claude setup-token` on the host and pass the token "
                        "to the container as CLAUDE_CODE_OAUTH_TOKEN)"
                    )
                raise Exception(
                    f'[{llm_provider}] returned an error response: "{reason[:500]}"'
                )

            return _normalize_text_response(payload.get("result"), llm_provider)

        if adapter == "kimi_code":
            # Kimi La suscripción (beneficio Kimi Code de Plus+) no emite API Keys de pago por uso; la autenticación la completa
            # OAuth la autorización de dispositivo: la WebUI guía al usuario para confirmar con un clic en el navegador y se guarda localmente
            # refresh token; el access token caduca en 15 minutos y kimi_code_oauth lo renueva de forma transparente
            # . Las peticiones van al endpoint oficial compatible con OpenAI, con una cuota independiente de la API de pago por uso.
            from app.services import kimi_code_oauth

            try:
                timeout_seconds = kimi_code_oauth.coerce_timeout(
                    extra_values.get("timeout")
                )
            except ValueError as timeout_error:
                raise ValueError(f"{llm_provider}: {timeout_error}") from None

            access_token = kimi_code_oauth.get_valid_access_token()
            # La región de autorización la determinan las credenciales: el token solo es válido en la región que lo emitió. Si el usuario cambia
            # API Platform tras autorizar y se enviaran las peticiones según la región de la interfaz, un token emitido en China iría al endpoint
            # Global (daría 401 seguro), así que la dirección de la petición sigue primero la región de emisión del token.
            authorized_region = kimi_code_oauth.authorized_region()
            if authorized_region:
                request_base_url = kimi_code_oauth.api_base_url(authorized_region)
                if base_url and authorized_region != kimi_code_oauth.resolve_region(
                    base_url
                ):
                    logger.warning(
                        f"kimi_code: API Platform selection differs from the "
                        f"authorized region; using '{authorized_region}' to match "
                        "the issued token"
                    )
            else:
                request_base_url = base_url or kimi_code_oauth.api_base_url(
                    kimi_code_oauth.resolve_region(base_url)
                )
            # Si el nombre del modelo está vacío se usa por defecto kimi-for-coding: el modelo de programación incluido en la suscripción,
            # para evitar que un id de modelo genérico codificado deje de funcionar cuando cambien los beneficios.
            request_model = model_name or "kimi-for-coding"
            logger.info(f"requesting kimi code chat completion, model: {request_model}")
            # Provider El tiempo de espera holgado propio debe reducirse al presupuesto de la prueba en el escenario de prueba de conexión.
            effective_timeout = timeout_seconds
            if request_timeout is not None:
                effective_timeout = min(effective_timeout, request_timeout)
            client = sdk_client = OpenAI(
                api_key=access_token,
                base_url=request_base_url,
                timeout=effective_timeout,
                # La cadena anti-scraping del servidor exige cabeceras de identidad de dispositivo iguales a las del CLI.
                default_headers=kimi_code_oauth.api_headers(),
                **_sdk_transport_options(None, max_retries),
            )
            response = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=request_model, messages=[{"role": "user", "content": prompt}]
            )
            if response and isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            raise Exception(
                f'[{llm_provider}] returned an invalid response: "{response}"'
            )

        if adapter == "modelscope":
            content = ""
            client = sdk_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                **_sdk_transport_options(request_timeout, max_retries),
            )
            response = sdk_stream = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": False},
                stream=True,
            )
            if response:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content += delta.content

                if not content.strip():
                    raise ValueError("Empty content in stream response")

                return _normalize_text_response(content, llm_provider)
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        client = sdk_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            **_sdk_transport_options(request_timeout, max_retries),
        )

        response = trace_sdk_call(
            llm_provider, base_url, client.chat.completions.create,
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        if response:
            if isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            else:
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                    f"connection and try again."
                )
        else:
            raise Exception(
                f"[{llm_provider}] returned an empty response, please check your network connection and try again."
            )

    except Exception as e:
        return f"Error: {_sanitize_error_message(e)}"

    finally:
        for resource in (sdk_stream, sdk_client):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as cleanup_error:
                    logger.warning(
                        f"could not close LLM transport: {type(cleanup_error).__name__}"
                    )


def test_connection() -> tuple[bool, str, float]:
    """
    Hace una petición mínima con la configuración actual del Provider, para verificar si la cadena real de generación funciona.

    La prueba de conexión reutiliza directamente `_generate_response()`, así que cubre la API Key, la Base URL,
    el nombre del modelo y los campos específicos del Provider, pero no entra en la lógica de reintentos de la generación del guion ni envía
    el tema del video ni el texto del usuario. La petición usa un tiempo de espera corto y acotado y desactiva los reintentos automáticos del SDK: cuando el endpoint
    no es accesible (fallo de proxy, cortafuegos o DNS) falla rápido y añade pistas de diagnóstico, en lugar de
    quedarse colgado varios minutos y dejar solo "Request timed out." (issue #1642). Los valores devueltos son, en orden,
    el estado de éxito, el mensaje de error y el tiempo de la petición.
    """
    started_at = perf_counter()
    response = _generate_response(
        prompt="Reply with exactly: OK",
        request_timeout=LLM_CONNECTION_TEST_TIMEOUT_SECONDS,
        max_retries=LLM_CONNECTION_TEST_MAX_RETRIES,
    )
    elapsed = perf_counter() - started_at

    if not response:
        error_message = "LLM returned an empty response"
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    if response.startswith("Error:"):
        error_message = _explain_connection_test_failure(
            response.removeprefix("Error:").strip()
        )
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    logger.info(f"llm connection test succeeded, elapsed: {elapsed:.2f}s")
    return True, "", elapsed


def _limit_script_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API La capa ya valida la longitud con Pydantic; aquí se mantiene un respaldo para proteger
    # WebUI que, si un servicio interno llama directamente a generate_script, no se envíen prompts demasiado largos al modelo,
    # evitando costes de tokens anómalos y fallos de petición.
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _normalize_script_paragraph_number(paragraph_number: int | None) -> int:
    try:
        value = int(paragraph_number or MIN_SCRIPT_PARAGRAPH_NUMBER)
    except (TypeError, ValueError):
        value = MIN_SCRIPT_PARAGRAPH_NUMBER

    if value < MIN_SCRIPT_PARAGRAPH_NUMBER or value > MAX_SCRIPT_PARAGRAPH_NUMBER:
        # WebUI y la API limitan el rango; aquí se cubren las llamadas internas, para que parámetros anómalos no aumenten directamente
        # LLM el coste de generación ni generen resultados vacíos.
        logger.warning(
            f"script paragraph_number is out of range and will be clamped: {value}"
        )
        return max(MIN_SCRIPT_PARAGRAPH_NUMBER, min(value, MAX_SCRIPT_PARAGRAPH_NUMBER))

    return value


def build_script_prompt(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )

    # Se concatenan por separado las «reglas de generación del guion» y el «contexto de ejecución». Así, aunque un usuario avanzado sobrescriba el
    # system promptpor defecto, no se pierden el tema del video, el idioma, el número de párrafos y demás parámetros que cada generación debe incluir.
    prompt = custom_system_prompt or DEFAULT_SCRIPT_SYSTEM_PROMPT
    prompt += f"""

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".rstrip()
    if language:
        prompt += f"\n- language: {language}"
    if video_script_prompt:
        prompt += f"""

# Additional User Requirements:
{video_script_prompt}
""".rstrip()

    return prompt


def research_subject(video_subject: str, language: str = "", app_config=None) -> str:
    """Busca en Google, con Gemini, hechos verificables sobre el tema del video.

    Devuelve los hechos como lista de texto, o "" si falta la clave de Gemini o la
    búsqueda falla (cuota, red), para que el guion se genere igualmente sin ellos.
    """
    runtime_app_config = app_config if app_config is not None else config.app
    api_key = runtime_app_config.get("gemini_api_key", "")
    if not api_key:
        logger.warning("script research skipped: gemini_api_key is not set")
        return ""
    model = (
        str(runtime_app_config.get("script_research_model", "") or "").strip()
        or SCRIPT_RESEARCH_DEFAULT_MODEL
    )
    prompt = (
        f"Search the web for recent, verifiable facts about: {video_subject}\n"
        "List 5 to 8 facts, one per line starting with '- ', each with concrete "
        "names, numbers, places or dates and its source in brackets. Use only facts "
        "confirmed by the search results, mark accusations and open cases as such, "
        "and add nothing else. "
        f"Write the facts in {language or 'the language of the subject'}."
    )
    try:
        from google import genai
        from google.genai import types

        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                ),
            )
        facts = (response.text or "").strip()
        metadata = (
            response.candidates[0].grounding_metadata if response.candidates else None
        )
        sources = [
            chunk.web.title
            for chunk in (metadata.grounding_chunks if metadata else None) or []
            if chunk.web and chunk.web.title
        ]
    except Exception as e:
        logger.warning(f"script research failed, continuing without facts: {e}")
        return ""

    logger.info(
        f"script research with {model}, sources: {', '.join(sources) or 'none'}\n{facts}"
    )
    return facts


def generate_script(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
    app_config=None,
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )
    prompt = build_script_prompt(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
    )
    runtime_app_config = app_config if app_config is not None else config.app
    facts = (
        research_subject(video_subject, language, app_config)
        if runtime_app_config.get("script_research", False)
        else ""
    )
    if facts:
        prompt += (
            "\n\n# Verified Facts (use only these for names, numbers and dates):\n"
            f"{facts}"
        )
    final_script = ""
    logger.info(
        "generating video script: "
        f"subject={video_subject}, paragraph_number={paragraph_number}, "
        f"has_custom_prompt={bool(video_script_prompt.strip())}, "
        f"has_custom_system_prompt={bool(custom_system_prompt.strip())}"
    )

    def format_response(response):
        # Clean the script
        # Remove emphasis and heading markers without changing inline names
        # or musical notes such as C#, F# and G#.
        response = response.replace("*", "")
        response = re.sub(
            r"(?m)^[ \t]{0,3}#{1,6}[ \t]+(.*?)[ \t]*$",
            lambda match: re.sub(r"[ \t]+#+[ \t]*$", "", match.group(1)),
            response,
        )

        # Link labels are narration; discard destinations before stripping
        # bracketed stage directions. Common parenthesized URL components fit.
        response = re.sub(
            r"\[([^\]\n]+)\]\((?:[^()\n]|\([^()\n]*\))*\)",
            lambda match: match.group(1),
            response,
        )

        # Remove markdown syntax.  Use non-greedy .*? so each bracket/paren
        # group is removed independently; the greedy form would eat all text
        # between the first opener and the last closer on the same line.
        response = re.sub(r"\[.*?\]", "", response)
        response = re.sub(r"\(.*?\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        # selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        return "\n\n".join(paragraphs)

    def ask(text):
        if app_config is None:
            return _generate_response(prompt=text)
        return _generate_response(prompt=text, app_config=app_config)

    for i in range(_max_retries):
        try:
            response = ask(prompt)
            if isinstance(response, str) and response.startswith("Error: "):
                # _generate_response returns provider failures as text. Passing
                # that text through would make the task treat it as narration.
                raise ValueError(response)
            if response:
                candidate = format_response(response).strip()
            else:
                logging.error("gpt returned an empty response")
                candidate = ""

            # Some upstream providers may return quota errors as plain text.
            if candidate and "当日额度已消耗完" in candidate:
                raise ValueError(candidate)

            if candidate:
                final_script = candidate
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")

        if i < _max_retries - 1:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")

    facts_rule = (
        "use only the verified facts of the brief for names, numbers and dates, and remove any other."
        if facts
        else "do not add names, numbers or dates, and remove any you are not sure about."
    )
    refine_rounds = int(runtime_app_config.get("script_refine_rounds", 0) or 0)
    for round_number in range(1, refine_rounds + 1):
        if not final_script:
            break
        try:
            response = ask(
                SCRIPT_REFINE_PROMPT.format(facts_rule=facts_rule)
                + f"\n\n# Brief:\n{prompt}\n\n# Draft:\n{final_script}"
            )
            if not isinstance(response, str) or response.startswith("Error: "):
                raise ValueError(response)
            candidate = format_response(response).strip()
            if not candidate:
                raise ValueError("empty response")
        except Exception as e:
            # Una ronda fallida (cuota, tiempo de espera) deja la versión anterior, que ya es válida.
            logger.warning(
                f"script refine round {round_number} failed, keeping the previous version: {e}"
            )
            break
        final_script = candidate
        logger.info(f"script refine round {round_number} done")

    if not final_script:
        logger.error("failed to generate video script after retries")
    else:
        logger.success(f"completed: \n{final_script}")
    return final_script.strip()


def _strip_code_fence(text: str) -> str:
    """Strip a surrounding markdown code fence from an LLM response.

    Non-OpenAI providers (Claude, Gemini, …) frequently wrap JSON output in a
    ```json … ``` fence even when asked to return raw JSON. Removing it lets the
    first json.loads() succeed instead of falling through to the regex recovery
    path (and spuriously logging a warning). Mirrors the DOTALL handling already
    used in _parse_social_metadata().
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_embedded_json(text: str, opening: str):
    """Decode one complete JSON value without absorbing later prose/examples."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text or ""):
        if character != opening:
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError("response contains no complete JSON value")


def generate_terms(
    video_subject: str,
    video_script: str,
    amount: int = 5,
    match_script_order: bool = False,
    app_config=None,
) -> List[str]:
    video_script = utils.remove_pause_tags(video_script or "").strip()
    if match_script_order:
        goal = (
            f"Generate {amount} chronological stock-video search terms that follow "
            "the order of topics in the video script."
        )
        ordering_rule = (
            "6. keep the terms in the same order as the script narration; "
            "earlier terms must describe earlier visual moments."
        )
        # En el modo de palabras clave ordenadas, el número de ejemplos debe coincidir con amount, para que el modelo no se deje engañar por
        # los 4 ejemplos fijos y devuelva pocas palabras clave para textos largos, afectando a la cobertura de material.
        example_terms = [
            "opening visual topic",
            *[f"script visual topic {index}" for index in range(2, max(amount, 1))],
            "final visual topic",
        ]
        output_example = json.dumps(example_terms[:amount], ensure_ascii=False)
    else:
        goal = (
            f"Generate {amount} search terms for stock videos, depending on the "
            "subject of a video."
        )
        ordering_rule = ""
        output_example = (
            '["search term 1", "search term 2", "search term 3",'
            '"search term 4", "search term 5"]'
        )

    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
{goal}

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words naming a concrete, filmable scene that stock libraries carry (e.g. "counting cash", "police lights night"); add the main subject only when it can be filmed itself.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video and match the mood of the script (dark, tense scenes for crime or danger; bright scenes for happy topics); avoid shots of signs, posters or text.
5. reply with english search terms only.
{ordering_rule}

## Output Example:
{output_example}

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}, match_script_order: {match_script_order}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        search_terms = []
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                # generate_terms el tipo de retorno público es List[str]. Si se devolviera tal cual el texto de error
                # del Provider, un consumidor que solo comprueba si está vacío tomaría una cadena no vacía como éxito,
                # y el bucle de descarga de material recorrería el texto de error carácter a carácter, generando peticiones externas sin sentido.
                # Aquí se devuelve siempre una lista vacía, para que la capa de orquestación termine la tarea de inmediato en el punto real del fallo.
                logger.error(f"failed to generate video terms: {response}")
                return []
            search_terms = json.loads(_strip_code_fence(response))
        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                try:
                    search_terms = _extract_embedded_json(response, "[")
                except ValueError as e:
                    logger.warning(f"failed to generate video terms: {str(e)}")

        # Apply the same contract to direct JSON and prose-wrapped recovery.
        # Otherwise a nonempty array of numbers or objects reaches material search.
        if not isinstance(search_terms, list) or not all(
            isinstance(term, str) for term in search_terms
        ):
            logger.error("response is not a list of strings.")
            search_terms = []

        # The model may ignore the requested count or return blank strings.
        # Enforce the prompt contract before material providers are contacted;
        # repeated topics remain meaningful in chronological mode.
        search_terms = [term.strip() for term in search_terms if term.strip()][:amount]
        if search_terms:
            break
        if i < _max_retries - 1:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    logger.success(f"completed: \n{search_terms}")
    return search_terms


# =============================================================================
# Social publishing metadata
#
# Genera, a partir del tema y el guion del video, el title, el caption y los hashtags habituales al publicar en plataformas de video corto.
# Esta función solo reutiliza el LLM provider existente, no se conecta a ningún servicio de publicación externo ni afecta a la cadena principal de generación de video.
# =============================================================================

# Cada plataforma prefiere una longitud de texto y un número de hashtags distintos. Aquí se usan límites conservadores, para que el llamador no tenga que
# recortar de nuevo si el modelo devuelve contenido demasiado largo.
SOCIAL_PLATFORMS = {
    "tiktok": {"title_max": 100, "caption_max": 2200, "hashtag_count": 5},
    "youtube_shorts": {"title_max": 100, "caption_max": 5000, "hashtag_count": 3},
    "instagram_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 8},
    "facebook_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 5},
}
DEFAULT_SOCIAL_PLATFORM = "tiktok"
DEFAULT_SOCIAL_LANGUAGE = "auto"
MAX_SOCIAL_SUBJECT_LENGTH = 500
MAX_SOCIAL_SCRIPT_LENGTH = 8000
MAX_SOCIAL_LANGUAGE_LENGTH = 64

SOCIAL_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube_shorts": "YouTube Shorts",
    "instagram_reels": "Instagram Reels",
    "facebook_reels": "Facebook Reels",
}

# LLM Etiquetas genéricas de respaldo cuando no está disponible. A propósito no se ligan a ningún país ni idioma, para garantizar API
# que se devuelva una estructura utilizable en chino, inglés, vietnamita y otros escenarios.
DEFAULT_SOCIAL_HASHTAGS = [
    "#shorts",
    "#viral",
    "#trending",
    "#fyp",
    "#video",
    "#reels",
    "#creator",
    "#content",
]


def _resolve_social_platform(platform: str | None) -> str:
    value = (platform or "").strip().lower()
    return value if value in SOCIAL_PLATFORMS else DEFAULT_SOCIAL_PLATFORM


def _normalize_social_language(language: str | None) -> str:
    value = (language or DEFAULT_SOCIAL_LANGUAGE).strip()
    if len(value) > MAX_SOCIAL_LANGUAGE_LENGTH:
        logger.warning(
            "social metadata language is too long and will be truncated to "
            f"{MAX_SOCIAL_LANGUAGE_LENGTH} characters."
        )
        value = value[:MAX_SOCIAL_LANGUAGE_LENGTH]
    return value or DEFAULT_SOCIAL_LANGUAGE


def _limit_social_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API La capa limita la longitud; aquí se mantiene un respaldo para proteger las llamadas internas o futuras WebUI
    # llamadas directas, para no enviar contenido demasiado largo al modelo y evitar costes de tokens anómalos.
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _social_language_instruction(language: str | None) -> str:
    language = _normalize_social_language(language)
    if language.lower() == DEFAULT_SOCIAL_LANGUAGE:
        return (
            "Use the same language as the video subject and script. If the subject "
            "and script use different languages, prefer the script language."
        )

    return f'Write "title" and "caption" in this language: {language}.'


def _clamp_text(text, max_length: int) -> str:
    value = ("" if text is None else str(text)).strip()
    if max_length and len(value) > max_length:
        return value[:max_length].rstrip()
    return value


def _normalize_hashtags(raw, count: int) -> List[str]:
    """
    Normaliza los hashtags devueltos por el LLM al formato `#tag`.

    LLM puede devolver cadenas, arrays, frases con espacios, etiquetas duplicadas o contenido con signos de puntuación.
    Limpiar aquí de forma centralizada hace que la estructura de respuesta de la interfaz sea estable y evita, al publicar, etiquetas vacías,
    duplicadas o hashtags con formato poco habitual.
    """
    if isinstance(raw, str):
        # CJK text separates tags with ，、； and no spaces; # starts a new tag.
        candidates = re.split(r"[\s,，、;；#]+", raw)
    elif isinstance(raw, (list, tuple)):
        # Cada elemento del array se trata como una etiqueta completa, así que "du lich" se convierte en
        # "#dulich", en lugar de separarse en dos etiquetas.
        candidates = [str(entry) for entry in raw]
    else:
        candidates = []

    seen = set()
    result: List[str] = []
    for item in candidates:
        # El hindi, el tailandés y otros dependen de caracteres combinantes, que no se pueden eliminar directamente como \w. Pero los combinantes deben
        # ir unidos a un carácter conservado; cuando se elimina un emoji, su selector de variante también debe descartarse, de lo contrario
        # #☕️ quedarían etiquetas con solo caracteres invisibles, y #coffee☕️ no podría deduplicarse con #coffee.
        # Los dígitos se conservan, así que también hay que limpiar los selectores de presentación de emoji y los combinantes de teclas,
        # para que #1️⃣ sea igual que #1; los combinantes de otros idiomas y las variantes de glifos CJK se conservan según la regla de unión.
        characters = []
        can_attach_mark = False
        for character in item:
            if character in {"\ufe0e", "\ufe0f", "\u20e3"}:
                can_attach_mark = False
            elif character.isalnum() or character == "_":
                characters.append(character)
                can_attach_mark = True
            elif can_attach_mark and unicodedata.category(character).startswith("M"):
                characters.append(character)
            else:
                can_attach_mark = False
        tag = "".join(characters)
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(f"#{tag}")
        if count and len(result) >= count:
            break
    return result


def build_social_metadata_prompt(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> str:
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    platform = _resolve_social_platform(platform)
    spec = SOCIAL_PLATFORMS[platform]
    label = SOCIAL_PLATFORM_LABELS.get(platform, platform)
    language_instruction = _social_language_instruction(language)

    prompt = f"""
# Role: Short-Video Social Media Copywriter

## Goal
Write engaging publishing metadata for a short video that will be posted on {label}.

## Constraints
1. Respond ONLY with a single valid minified JSON object. No markdown, no code fences, no commentary.
2. The JSON must contain exactly these keys: "title", "caption", "hashtags".
3. "title": a catchy hook, at most {spec["title_max"]} characters.
4. "caption": an engaging description that ends with a call to action, at most {spec["caption_max"]} characters. Do not put hashtags inside the caption.
5. "hashtags": a JSON array of exactly {spec["hashtag_count"]} strings. Each must start with "#", contain no spaces, and be relevant to the topic and to {label}.
6. {language_instruction}

## Output Example
{{"title":"...","caption":"...","hashtags":["#example","#video"]}}

## Context
### Video Subject
{video_subject}

### Video Script
{video_script}
""".strip()
    return prompt


def _parse_social_metadata(response: str, platform: str) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]

    data = None
    try:
        data = json.loads(_strip_code_fence(response))
    except Exception:
        # Algunos modelos envuelven el JSON con texto explicativo o un bloque markdown.
        # API El llamador solo necesita una estructura estable, así que aquí se intenta extraer el primer objeto JSON.
        data = _extract_embedded_json(response, "{")

    if not isinstance(data, dict):
        raise ValueError("social metadata response is not a JSON object")

    title = _clamp_text(data.get("title", ""), spec["title_max"])
    caption = _clamp_text(data.get("caption", ""), spec["caption_max"])
    hashtags = _normalize_hashtags(data.get("hashtags", []), spec["hashtag_count"])

    if not title and not caption:
        raise ValueError("social metadata response is missing both title and caption")

    return {"title": title, "caption": caption, "hashtags": hashtags}


def _fallback_social_metadata(
    video_subject: str, video_script: str, platform: str
) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]
    subject = (video_subject or "").strip()
    script = (video_script or "").strip()

    title = subject
    if not title and script:
        # En chino y japonés normalmente no hay espacio tras el final de frase; en inglés se sigue exigiendo un espacio, para no partir números como 2.0.
        title = re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*", script, maxsplit=1)[0]

    return {
        "title": _clamp_text(title, spec["title_max"]),
        "caption": _clamp_text(script or subject, spec["caption_max"]),
        "hashtags": _normalize_hashtags(DEFAULT_SOCIAL_HASHTAGS, spec["hashtag_count"]),
    }


def generate_social_metadata(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> dict:
    """
    Genera los metadatos del texto de publicación del video corto.

    La estructura devuelta es siempre `{"title": str, "caption": str, "hashtags": List[str]}`.
    Si el LLM no está disponible o devuelve un formato anómalo, se degrada a un resultado heurístico genérico, garantizando API
    que el llamador siempre reciba una estructura de datos que se pueda mostrar y editar antes de publicar.
    """
    platform = _resolve_social_platform(platform)
    language = _normalize_social_language(language)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    prompt = build_social_metadata_prompt(
        video_subject=video_subject,
        video_script=video_script,
        language=language,
        platform=platform,
    )
    logger.info(f"generating social metadata: platform={platform}, language={language}")

    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and response.startswith("Error: "):
                logger.error(f"failed to generate social metadata: {response}")
                break
            metadata = _parse_social_metadata(response, platform)
            logger.success(f"completed: \n{metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"failed to parse social metadata: {str(e)}")

        if i < _max_retries - 1:
            logger.warning(
                f"failed to generate social metadata, trying again... {i + 1}"
            )

    logger.warning("falling back to heuristic social metadata")
    return _fallback_social_metadata(video_subject, video_script, platform)


if __name__ == "__main__":
    video_subject = "¿Cuál es el sentido de la vida?"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)
