import hashlib
import html
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import requests
import streamlit as st
from loguru import logger
from streamlit_tour import Tour

# WebUI Al ejecutarse como punto de entrada independiente, el directorio raíz del proyecto debe tener prioridad sobre las dependencias de terceros,
# para que un paquete app homónimo de las dependencias no oculte el paquete app propio de MoneyPrinterTurbo.
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if root_dir in sys.path:
    sys.path.remove(root_dir)
sys.path.insert(0, root_dir)

from app.config import config
from app.models import const
from app.models.llm_provider import (
    DEFAULT_LLM_PROVIDER_ID,
    LLM_PROVIDER_REGISTRY,
    get_llm_provider,
    normalize_provider_override,
)
from app.models.schema import (
    ImageMotion,
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import bgm as bgm_service
from app.services import material_upload as material_upload_service
from app.services import (
    cache_manager,
    llm,
    loomloom,
    material,
    metaso_minimax,
    muapi,
    infistar_video,
    jalapeno_video,
    ofox,
    subtitle,
    video,
    volcengine_seedance,
    voice,
    webui_task,
)
from app.services import elevenlabs_music as elevenlabs_music_service
from app.services import sonilo as sonilo_service
from app.services import state as sm
from app.services import task as tm
from app.services import version_checker
from app.utils.logging_utils import (
    LOG_LEVELS,
    configure_terminal_logger,
    get_logging_options,
)
from app.utils import utils
from app.utils.request_logging import trace_http_call

st.set_page_config(
    page_title="MoneyPrinterTurbo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Report a bug": "https://github.com/harry0703/MoneyPrinterTurbo/issues",
        "About": "# MoneyPrinterTurbo\nSimply provide a topic or keyword for a video, and it will "
        "automatically generate the video copy, video materials, video subtitles, "
        "and video background music before synthesizing a high-definition short "
        "video.\n\nhttps://github.com/harry0703/MoneyPrinterTurbo",
    },
)


# Streamlit 1.59 muestra por defecto en la esquina superior derecha accesos de plataforma como Deploy o skills nudge.
# MoneyPrinterTurbo es una herramienta local para usuarios finales; estos accesos dejan un gran espacio en blanco arriba
# y hacen creer a los usuarios nuevos que necesitan instalar componentes adicionales. Aquí se oculta de forma unificada la barra de herramientas de Streamlit
# y se reduce el margen superior del contenedor principal, dejando solo el título del proyecto, la selección de idioma y la zona de ajustes de negocio.
style_file = Path(__file__).with_name("styles.css")
streamlit_style = f"<style>{style_file.read_text(encoding='utf-8')}</style>"
st.markdown(streamlit_style, unsafe_allow_html=True)
# Definir directorios de recursos
font_dir = utils.font_dir()
song_dir = utils.song_dir()
i18n_dir = os.path.join(root_dir, "webui", "i18n")
config_file = os.path.join(root_dir, "webui", ".streamlit", "webui.toml")
# La lista de idiomas debe estar disponible antes de inicializar el estado de sesión, para que en la primera visita se pueda mapear el locale del navegador a
# los idiomas que el proyecto admite de verdad; el resultado de la detección automática solo afecta a la sesión actual, sin modificar la configuración global.
locales = utils.load_locales(i18n_dir)
DEFAULT_CHATTERBOX_BASE_URL = "http://127.0.0.1:4123/v1"
DEFAULT_CHATTERBOX_MODEL = "chatterbox"
DEFAULT_CHATTERBOX_VOICES = ["default-Female"]
DEFAULT_KOKORO_BASE_URL = "http://127.0.0.1:8880/v1"
DEFAULT_KOKORO_MODEL = "kokoro"
# empty = ask the server for its voice list (GET {base_url}/audio/voices)
DEFAULT_KOKORO_VOICES: list[str] = []
DEFAULT_VOXCPM_BASE_URL = voice.VOXCPM_DEFAULT_BASE_URL
DEFAULT_VOXCPM_VOICE = voice.VOXCPM_DEFAULT_VOICE
VOXCPM_REFERENCE_AUDIO_SESSION_KEY = "voxcpm_reference_audio"
VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY = "voxcpm_reference_audio_error"
VOXCPM_PROMPT_AUDIO_SESSION_KEY = "voxcpm_prompt_audio"
VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY = "voxcpm_prompt_audio_error"
VOXCPM_PROMPT_TEXT_SESSION_KEY = "voxcpm_prompt_text_input"
VOXCPM_HIGH_FIDELITY_SESSION_KEY = "voxcpm_high_fidelity_enabled"
VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY = "voxcpm_separate_prompt_audio_enabled"
VOXCPM_PROMPT_EXAMPLE_MODE_SESSION_KEY = "voxcpm_prompt_example_mode"
ONBOARDING_TOUR_KEY = "mpt-onboarding-v1"
CUSTOM_LLM_ENDPOINT_ID = "custom"
VOICE_MODE_TTS = "tts"
VOICE_MODE_UPLOAD = "upload"
VOICE_MODE_NONE = "none"
LOOMLOOM_MAX_POLL_FAILURES = 5
# WebUI Muestra los orígenes de video agrupados por capacidad de material, pero internamente sigue guardando el valor video_source original.
# AI El grupo de video y la página de ajustes comparten el mismo orden de negocio: primero los proveedores colaboradores, en el orden Metaso, OFox, Shengsuanyun,
# Volcengine, Infistar; después el resto de servicios. Así el orden es el mismo en ambos accesos, y además
# no cambia la semántica de los campos en config.toml, las tareas históricas ni las peticiones de API, y los usuarios antiguos no tienen que migrar la configuración.
VIDEO_SOURCE_GROUPS = {
    "stock_video": ("pexels", "pixabay", "coverr", "wikimedia"),
    "ai_video": (
        "metaso_minimax",
        "ofox",
        "loomloom",
        "volcengine_seedance",
        "infistar",
        "jalapeno",
        "wavespeed",
        "muapi",
    ),
    "ai_image": ("openai_image", "codex_image"),
    "local": ("local",),
}
# Upload-Post la API Key y el usuario de publicación se gestionan en dos páginas distintas, y el nombre de usuario de publicación
# no es el correo de inicio de sesión. Mantener los accesos centralizados evita desviaciones por URLs codificadas en cada texto multilingüe,
# y permite al usuario completar desde la WebUI la primera configuración y el mantenimiento posterior de la cuenta.
UPLOAD_POST_API_KEYS_URL = "https://app.upload-post.com/api-keys"
UPLOAD_POST_MANAGE_USERS_URL = "https://app.upload-post.com/manage-users"
# Los ajustes de material y la descripción del origen de video comparten el acceso promocional, para que los parámetros del enlace no difieran entre ambos sitios.
OFOX_REFERRAL_URL = (
    "https://ofox.ai/?utm_source=github"
    "&utm_medium=sponsorship&utm_content=moneyprinterturbo"
)
# «Por defecto» es un centinela exclusivo de la WebUI; no se escribe en config.toml ni se pasa a FFmpeg.
# Cuando video_codec no está configurado, el backend sigue usando el estable libx264; conservar este centinela aparte permite distinguir
# «seguir la estrategia por defecto del proyecto» de «el usuario fija explícitamente libx264», facilitando ajustar con seguridad la estrategia por defecto en el futuro.
DEFAULT_VIDEO_CODEC_OPTION = "__default__"
# LoomLoom la interfaz de capacidades solo devuelve el ID del modelo y el nombre mostrado, sin precios. Aquí solo se mantienen
# los precios de referencia confirmados por el usuario, para ayudar a elegir modelo; el coste final se liquida según el uso real del modelo. Los alias cubren
# tanto el nombre mostrado actual como los IDs de modelo habituales; los modelos nuevos no incluidos devuelven simplemente un precio vacío, sin afectar a la selección ni a la cotización.
LOOMLOOM_VIDEO_MODEL_PRICES = (
    (("veo31fast", "googleveo31fastpreview"), "￥0.700/seg", "￥0.700/seg"),
    (
        (
            "通义万相22图生视频fastlora",
            "通义万相22文生视频fastlora",
            "tongyiwanxiang22i2vfastlora",
            "tongyiwanxiang22t2vfastlora",
            "wanx22i2vfastlora",
            "wanx22t2vfastlora",
        ),
        "￥0.350–0.770/video",
        "￥0.350/video (480P); ￥0.770/video (720P)",
    ),
    (("即梦30文生视频720p", "jimeng30t2v720p"), "￥0.230/seg", "￥0.230/seg"),
    (("即梦30pro视频", "jimeng30pro视频", "jimeng30provideo"), "￥1.000/seg", "￥1.000/seg"),
    (("veo3", "googleveo3"), "￥1.400/seg", "￥1.400/seg"),
    (("veo31", "googleveo31"), "￥1.400/seg", "￥1.400/seg"),
    (
        ("klingv2", "可灵v2"),
        "￥10.00–20.00/video",
        "￥10.00/video (5 s); ￥20.00/video (10 s)",
    ),
    (
        ("klingv21master", "可灵v21master"),
        "￥10.00–20.00/video",
        "￥10.00/video (5 s); ￥20.00/video (10 s)",
    ),
    (
        ("viduq3pro",),
        "￥0.440–1.000/seg",
        "￥0.440/seg (540P); ￥0.940/seg (720P); ￥1.000/seg (1080P)",
    ),
)
DEFAULT_SUBTITLE_SETTINGS = {
    "subtitle_enabled": True,
    "font_name": "MicrosoftYaHeiBold.ttc",
    "subtitle_position": "bottom",
    "subtitle_display_mode": "sentence",
    "subtitle_animation": "none",
    "custom_position": 70.0,
    "text_fore_color": "#FFFFFF",
    "font_size": 60,
    "stroke_color": "#000000",
    "stroke_width": 1.5,
    "subtitle_background_enabled": False,
    "subtitle_background_color": "#000000",
    "rounded_subtitle_background": False,
}
LOCAL_MATERIAL_EXTENSIONS = set(material_upload_service.SUPPORTED_MATERIAL_EXTENSIONS)
CUSTOM_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
_FINAL_VIDEO_PATTERN = re.compile(
    r"^final-(?P<index>\d+)\.(?P<extension>mp4|mov|mkv|webm)$",
    re.IGNORECASE,
)
# Comparte con las reglas de nombre de archivo de subida de app/services/bgm.py y app/services/material_upload.py
# la misma lista: además de los separadores de ruta y los caracteres ilegales de Win32, deben eliminarse todos los caracteres de control y los que cambian la forma
# de visualización. Antes solo se bloqueaba ord < 32 (caracteres de control C0), y faltaba la otra mitad del mismo problema:
# * C1 caracteres de control U+007F-U+009F: entran tal cual en el nombre de archivo de descarga, los logs y la interfaz.
# * caracteres de control de texto bidireccional U+200E/U+200F/U+202A-U+202E/U+2066-U+2069: los caracteres después de U+202E
#   se renderizan al revés, y ``holiday<U+202E>4pm`` parece otra extensión.
# * U+2028/U+2029(separadores Unicode de línea/párrafo) pueden crear saltos de línea visuales dentro de una línea de log.
# El tema del video es texto libre del usuario, que puede pegar directamente estos caracteres, así que deben filtrarse aquí.
# La cadena no usa el prefijo r'': ``\\`` representa la propia barra invertida a excluir, y los escapes ``\x``/``\u`` los
# Python resuelve, para no escribir texto de escape en el conjunto de caracteres.
_DOWNLOAD_FILENAME_INVALID_PATTERN = re.compile(
    "[<>:\"/\\\\|?*\\x00-\\x1f\\x7f-\\x9f"
    "\\u200e\\u200f\\u2028\\u2029\\u202a-\\u202e\\u2066-\\u2069]"
)
_WINDOWS_RESERVED_FILENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {
        f"{prefix}{number}"
        for prefix in ("COM", "LPT")
        for number in range(1, 10)
    }
    # Win32 también reconoce los superíndices Latin-1 ¹, ², ³ como números de dispositivo. Aunque esos temas
    # son raros, siguen haciendo fallar la descarga en Windows, así que se tratan igual que los nombres reservados con dígitos normales.
    | {
        f"{prefix}{number}"
        for prefix in ("COM", "LPT")
        for number in ("¹", "²", "³")
    }
)
_RUNTIME_CONFIG_SECTIONS = {
    "app": config.app,
    "azure": config.azure,
    "chatterbox": config.chatterbox,
    "kokoro": config.kokoro,
    "elevenlabs": config.elevenlabs,
    "minimax_tts": config.minimax_tts,
    "siliconflow": config.siliconflow,
    "fish_audio": config.fish_audio,
    "voxcpm": config.voxcpm,
    "ui": config.ui,
}
# Los preajustes y la copia de seguridad de claves usan cada uno su propio identificador de archivo. Al importar se valida primero el schema y la versión,
# para no tomar registros de tareas, config.toml u otros JSON como archivos exportados de esta función.
SETTINGS_PRESET_SCHEMA = "moneyprinterturbo.settings-preset"
SETTINGS_PRESET_VERSION = 1
SETTINGS_PRESET_FILE_NAME = "moneyprinterturbo-settings.json"
KEY_BACKUP_SCHEMA = "moneyprinterturbo.key-backup"
KEY_BACKUP_VERSION = 1
KEY_BACKUP_FILE_NAME = "moneyprinterturbo-keys.json"
# Export files contain only settings or credentials, not media. Reject oversized
# uploads before decoding and parsing them in the Streamlit process.
MAX_SETTINGS_TRANSFER_BYTES = 2 * 1024 * 1024
# Los preajustes solo describen parámetros de generación. El material, la voz y la música son rutas de archivos locales, y los preajustes suelen importarse en otra
# máquina u otro contenedor, donde esas rutas solo apuntarían a archivos inexistentes.
PRESET_EXCLUDED_PARAM_KEYS = frozenset(
    {
        "video_materials",
        "custom_audio_file",
        "bgm_file",
    }
)
# Las claves se reconocen por el sufijo del nombre del ajuste. Los Providers nuevos que sigan la nomenclatura existente entrarán automáticamente
# en la copia, sin tener que mantener una segunda lista de claves.
CREDENTIAL_KEY_SUFFIXES = (
    "api_key",
    "api_keys",
    "api_token",
    "access_key",
    "secret_key",
    "speech_key",
)
# Si solo se restauran las claves y no los ajustes asociados, las credenciales siguen sin funcionar. Estos ajustes asociados se copian junto con las claves.
CREDENTIAL_COMPANION_KEYS = {
    # Azure la voz debe conocer también la región.
    "azure": ("speech_region",),
    # Provider los campos adicionales los declara el Registry, por ejemplo, de Cloudflare AI Gateway, el
    # Account ID y el Gateway ID. Si solo se restaura la API Key y se pierden estos campos, en otra
    # máquina ese Provider seguiría sin poder llamarse. Leerlos del Registry permite que los
    # Provider que se añadan en el futuro entren automáticamente en la copia, sin mantener aquí una segunda lista de campos.
    "app": tuple(
        provider.config_key(field.config_suffix)
        for provider in LLM_PROVIDER_REGISTRY
        for field in provider.extra_fields
    ),
}

NON_LLM_COMPANION_KEYS = {
    "app": ("upload_post_username",)
}
# Una misma clave puede usar distintas key de control en distintos paneles: el panel de audio edita directamente las claves LLM de Gemini y
# MiMo . Al restaurar la copia deben borrarse todos los alias, o el valor antiguo residual
# sobrescribiría la clave recién restaurada en el siguiente rerun.
CREDENTIAL_WIDGET_STATE_ALIASES = {
    ("app", "gemini_api_key"): ("gemini_tts_api_key_input",),
    ("app", "mimo_api_key"): ("mimo_tts_api_key_input",),
}
# ui la sección solo guarda preferencias de interfaz, sin credenciales, y se omite entera en la copia.
KEY_BACKUP_EXCLUDED_SECTIONS = frozenset({"ui"})


# -----------------------------------------------------------------------------
# Configuración de arranque, estado de sesión y localización
# -----------------------------------------------------------------------------


def _set_runtime_config(section_name, key, value):
    """
    Actualiza la configuración de la WebUI sin esperar a la tarea en segundo plano que está generando el video.

    Hasta que termina la tarea en segundo plano, la capa de configuración solo conserva el último valor de cada ajuste; cuando la tarea libera el bloqueo de configuración se
    aplica y guarda automáticamente. Los valores de los controles de la página los sigue manteniendo session_state de Streamlit, así que durante el almacenamiento temporal el
    rerun no restablece a la configuración antigua lo que el usuario acaba de escribir.
    """
    config_section = _RUNTIME_CONFIG_SECTIONS[section_name]
    updated = config.update_config_nonblocking(config_section, key, value)
    if not updated:
        logger.debug(f"deferred WebUI config update: section={section_name}, key={key}")
    return updated


def _delete_runtime_config(section_name, key):
    """Elimina un ajuste de la WebUI; si la tarea en segundo plano ocupa la configuración, se ejecuta más tarde."""
    config_section = _RUNTIME_CONFIG_SECTIONS[section_name]
    deleted = config.delete_config_nonblocking(config_section, key)
    if not deleted:
        logger.debug(f"deferred WebUI config delete: section={section_name}, key={key}")
    return deleted


def _save_runtime_config():
    """Pide guardar la configuración de la WebUI; si la tarea en segundo plano ocupa la configuración, vuelve de inmediato."""
    saved = config.try_save_config()
    if not saved:
        logger.debug("deferred WebUI config save until active task completes")
    return saved


def _saved_ui_choice(key, options, default, section=None):
    """Lee una elección persistente y degrada a valor por defecto la configuración antigua o los valores ilegales editados a mano."""
    options = list(options)
    section = config.ui if section is None else section
    saved = section.get(key, default)
    numeric_default = isinstance(default, (int, float)) and not isinstance(
        default, bool
    )
    # bool es una subclase de int, ``True == 1``. Si una opción numérica se escribe a mano como TOML
    # booleano debe rechazarse, sin dejar que se haga pasar por la primera option numérica.
    if numeric_default and isinstance(saved, bool):
        return default
    for option in options:
        if saved == option:
            # Devuelve el valor real de options y, de paso, normaliza el equivalente TOML 1.0 a
            # la opción entera 1, para que el tipo del parámetro aguas abajo no varíe según cómo se escriba la configuración.
            return option

    # TOML los valores numéricos suelen conservar su tipo original; sigue siendo compatible con cadenas escritas a mano por el usuario.
    if numeric_default and isinstance(saved, str):
        try:
            converted = type(default)(saved)
        except (TypeError, ValueError):
            converted = None
        for option in options:
            if converted == option:
                return option
    return default


def _saved_ui_number(key, default, minimum, maximum, number_type=float):
    """Lee y limita los valores numéricos persistentes, para que una configuración ilegal no rompa el slider de Streamlit."""
    try:
        saved = config.ui.get(key, default)
        if isinstance(saved, bool):
            raise ValueError("boolean is not a numeric setting")
        value = number_type(saved)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite value")
    except (TypeError, ValueError, OverflowError):
        value = default
    return min(maximum, max(minimum, value))


def _saved_ui_bool(key, default):
    """Compatible con booleanos TOML y cadenas manuales habituales; rechaza valores antiguos de significado ambiguo."""
    value = config.ui.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _saved_ui_color(key, default):
    """Solo pasa al color picker de Streamlit colores hexadecimales estándar de seis dígitos."""
    value = str(config.ui.get(key, default) or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return value
    return default


def _saved_ui_text(key, default="", max_length=None):
    """Lee texto persistente respetando el límite de longitud del control de la WebUI correspondiente."""
    value = str(config.ui.get(key, default) or default)
    if max_length is not None:
        value = value[:max_length]
    return value


def _run_llm_read_operation(operation_name, operation):
    """
    Ejecuta peticiones de solo lectura con una configuración LLM actual estable, sin esperar a las tareas de generación de video.

    Si se puede obtener de inmediato el bloqueo de configuración se mantiene la protección de exclusión mutua original; si lo tiene la tarea de video en segundo plano,
    la configuración global no cambiará hasta que termine la tarea, así que se puede copiar con seguridad la configuración actual y superponer el Provider, el modelo y la clave
    que la página aún no ha guardado en disco. Así el texto nuevo usa la última elección de la interfaz, y a la vez
    no cambia la tarea de video que se está generando.
    """
    with config.try_runtime_config_lock() as lock_acquired:
        # La capa de configuración tiene el bloqueo de cola mientras copia los valores globales y superpone los pendientes, así que la instantánea solo puede ver
        # el estado completo de antes o de después de la actualización, sin mezclar dos conjuntos de parámetros de Provider.
        app_config_snapshot = config.snapshot_config_with_pending(config.app)
        if lock_acquired:
            return operation(app_config_snapshot)

    logger.info(
        f"run read-only LLM operation with active task configuration: "
        f"operation={operation_name}"
    )
    return operation(app_config_snapshot)


def _parse_chatterbox_voices(voices):
    # Chatterbox es un servicio autoalojado, y el usuario escribe a mano la lista de voces en la WebUI.
    # Aquí se admiten de forma unificada arrays TOML y cadenas separadas por comas del campo de entrada, para evitar que el desplegable,
    # el botón de vista previa y el flujo de generación posterior usen formatos distintos y el estado sea incoherente.
    if isinstance(voices, str):
        return [v.strip() for v in voices.split(",") if v.strip()]
    return [str(v).strip() for v in voices or [] if str(v).strip()]


def _sync_chatterbox_config_from_session_state():
    # Streamlit el botón provoca un rerun de toda la página, y el campo de configuración de Chatterbox está
    # después del botón «Probar síntesis de voz». Si la vista previa solo leyera config.chatterbox, podría no obtener
    # el base_url/model/voices que el usuario acaba de escribir. Sincronizar primero desde session_state
    # garantiza que la lógica del botón y la del campo de entrada usen la misma configuración más reciente.
    _set_runtime_config(
        "chatterbox",
        "base_url",
        (
            st.session_state.get(
                "chatterbox_base_url_input",
                config.chatterbox.get("base_url") or DEFAULT_CHATTERBOX_BASE_URL,
            )
            or ""
        ).strip(),
    )
    _set_runtime_config(
        "chatterbox",
        "api_key",
        st.session_state.get(
            "chatterbox_api_key_input", config.chatterbox.get("api_key", "")
        ),
    )
    _set_runtime_config(
        "chatterbox",
        "model_id",
        (
            st.session_state.get(
                "chatterbox_model_input",
                config.chatterbox.get("model_id") or DEFAULT_CHATTERBOX_MODEL,
            )
            or DEFAULT_CHATTERBOX_MODEL
        ).strip(),
    )
    _set_runtime_config(
        "chatterbox",
        "voices",
        _parse_chatterbox_voices(
            st.session_state.get(
                "chatterbox_voices_input",
                config.chatterbox.get("voices") or DEFAULT_CHATTERBOX_VOICES,
            )
        ),
    )


def _sync_kokoro_config_from_session_state():
    # El catálogo de voces se renderiza antes que los campos de ajustes; se sincroniza primero el estado del navegador, para que este mismo rerun use
    # el nuevo endpoint y las voces configuradas a mano, sin tener que tocar otra vez el control.
    _set_runtime_config(
        "kokoro",
        "base_url",
        (
            st.session_state.get(
                "kokoro_base_url_input",
                config.kokoro.get("base_url") or DEFAULT_KOKORO_BASE_URL,
            )
            or ""
        ).strip(),
    )
    _set_runtime_config(
        "kokoro",
        "api_key",
        st.session_state.get(
            "kokoro_api_key_input", config.kokoro.get("api_key", "")
        ),
    )
    _set_runtime_config(
        "kokoro",
        "model_id",
        (
            st.session_state.get(
                "kokoro_model_input",
                config.kokoro.get("model_id") or DEFAULT_KOKORO_MODEL,
            )
            or DEFAULT_KOKORO_MODEL
        ).strip(),
    )
    _set_runtime_config(
        "kokoro",
        "voices",
        _parse_chatterbox_voices(
            st.session_state.get(
                "kokoro_voices_input",
                config.kokoro.get("voices") or DEFAULT_KOKORO_VOICES,
            )
        ),
    )


def _get_kokoro_voice_options(saved_voice_name: str) -> list[str]:
    """Caché corta del catálogo remoto en la sesión; al desconectarse se conserva la última elección, sin tomar el fallo como un cambio de voz del usuario."""
    if config.kokoro.get("voices"):
        return voice.get_kokoro_voices()

    # Solo se conserva una entrada de caché del servicio actual. Al cambiar de endpoint/credenciales se vuelve a consultar de inmediato, y la caché no guarda la Key en claro;
    # 30 las demás operaciones de UI dentro de esos segundos no vuelven a bloquearse 5 segundos esperando a un servicio que ya se sabe desconectado.
    signature = (
        (config.kokoro.get("base_url") or "").strip().rstrip("/"),
        _credential_signature(config.kokoro.get("api_key", "")),
    )
    catalog = st.session_state.get("kokoro_voice_catalog", {})
    if catalog.get("signature") != signature:
        catalog = {"signature": signature, "voices": [], "checked_at": None}
    now = time.monotonic()
    if catalog["checked_at"] is None or now - catalog["checked_at"] >= 30:
        fetched = voice.get_kokoro_voices(fallback=False)
        catalog.update(checked_at=now, available=bool(fetched))
        if fetched:
            catalog["voices"] = fetched
        st.session_state["kokoro_voice_catalog"] = catalog

    options = list(catalog["voices"])
    if not catalog["available"]:
        st.warning(tr("Kokoro Voices Unavailable"))
        # En la primera apertura puede no haber caché, y se conserva la elección real del archivo de configuración; tras recuperar la conexión,
        # solo un catálogo nuevo devuelto con éxito puede determinar que una voz antigua fue realmente eliminada por el servidor.
        if voice.is_kokoro_voice(saved_voice_name) and saved_voice_name not in options:
            options.insert(0, saved_voice_name)
    return options or [f"kokoro:{voice.KOKORO_DEFAULT_VOICE}"]


def _detect_audio_mime(audio_file: str, audio_bytes: bytes) -> str:
    # Algunos servicios TTS OpenAI-compatible, como travisvn/chatterbox-tts-api,
    # devuelven contenido WAV aunque se pida response_format=mp3. Si la vista previa de la WebUI usara siempre
    # audio/mp3, el navegador podría no reproducirlo, así que aquí se reconoce el formato real por la cabecera del archivo.
    header = audio_bytes[:12]
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "audio/wav"
    if header.startswith(b"ID3") or header[:2] in (
        b"\xff\xfb",
        b"\xff\xf3",
        b"\xff\xf2",
    ):
        return "audio/mp3"
    if header.startswith(b"OggS"):
        return "audio/ogg"
    ext = os.path.splitext(audio_file)[1].lower()
    return {
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }.get(ext, "audio/mp3")


def _build_uploaded_file_path(uploaded_file, target_dir, allowed_extensions, prefix):
    """Genera una ruta de guardado controlada en el servidor para los archivos subidos desde el navegador."""
    original_name = os.path.basename(str(uploaded_file.name or ""))
    extension = os.path.splitext(original_name)[1].lower()
    if extension not in allowed_extensions:
        logger.warning(
            f"reject unsupported uploaded file extension: {original_name or '<empty>'}"
        )
        raise ValueError("unsupported uploaded file type")

    normalized_target_dir = os.path.realpath(target_dir)
    os.makedirs(normalized_target_dir, exist_ok=True)
    # No reutiliza el nombre de archivo que envía el navegador, para evitar separadores de ruta, caracteres de control o sobrescrituras por nombre igual. El UUID solo se usa
    # para guardar en el servidor, sin cambiar el nombre original que el usuario ve en el control de subida.
    file_path = os.path.realpath(
        os.path.join(normalized_target_dir, f"{prefix}-{uuid4().hex}{extension}")
    )
    if os.path.commonpath([normalized_target_dir, file_path]) != normalized_target_dir:
        logger.warning(f"invalid uploaded file path: {file_path}")
        raise ValueError("invalid uploaded file path")
    return file_path


def _save_uploaded_local_materials(uploaded_files):
    """Validate a WebUI material batch and undo earlier files on failure."""
    local_videos_dir = utils.storage_dir("local_videos", create=True)
    materials = []
    persisted = []
    saved_paths = []
    try:
        for uploaded_file in uploaded_files:
            stored_name = material_upload_service.save_material_upload(
                uploaded_file.name, uploaded_file
            )
            file_path = os.path.join(local_videos_dir, stored_name)
            saved_paths.append(file_path)
            material_info = MaterialInfo()
            material_info.provider = "local"
            material_info.url = file_path
            materials.append(material_info)
            persisted.append(
                {
                    "provider": material_info.provider,
                    "url": material_info.url,
                    "duration": material_info.duration,
                }
            )
    except Exception:
        for file_path in saved_paths:
            try:
                os.remove(file_path)
            except OSError as exc:
                logger.warning(
                    f"failed to remove local material after batch error: "
                    f"path={file_path}, error={exc}"
                )
        raise
    return materials, persisted


def _initialize_session_state():
    """Inicializa de forma centralizada el estado de página que se conserva entre reruns."""
    if not st.session_state.get("cross_post_recovery_checked"):
        # WebUI puede ejecutarse sin FastAPI, así que también hay que tratar en la primera inicialización de sesión
        # los estados de publicación que dejó un reinicio del proceso. Si la recuperación falla no se escribe la marca, y un rerun posterior lo volverá a intentar.
        recovered = tm.recover_interrupted_cross_posts()
        if recovered is not None:
            st.session_state["cross_post_recovery_checked"] = True

    saved_ui_language = config.ui.get("language", "")
    browser_locale = st.context.locale
    initial_ui_language = utils.resolve_ui_language(
        saved_language=saved_ui_language,
        browser_locale=browser_locale,
        supported_languages=locales.keys(),
    )

    defaults = {
        "video_subject": "",
        "video_script": "",
        "video_terms": "",
        "paragraph_number_input": _saved_ui_number(
            "paragraph_number",
            1,
            llm.MIN_SCRIPT_PARAGRAPH_NUMBER,
            llm.MAX_SCRIPT_PARAGRAPH_NUMBER,
            int,
        ),
        "video_script_prompt": _saved_ui_text(
            "video_script_prompt",
            max_length=llm.MAX_SCRIPT_PROMPT_LENGTH,
        ),
        "custom_system_prompt": _saved_ui_text(
            "custom_system_prompt",
            llm.DEFAULT_SCRIPT_SYSTEM_PROMPT,
            llm.MAX_SCRIPT_SYSTEM_PROMPT_LENGTH,
        ),
        "match_materials_to_script": bool(
            config.app.get("match_materials_to_script", False)
        ),
        "custom_bgm_file_input": _saved_ui_text("custom_bgm_file"),
        "sonilo_bgm_prompt_input": _saved_ui_text(
            "sonilo_bgm_prompt",
            max_length=sonilo_service.MAX_PROMPT_LENGTH,
        ),
        "elevenlabs_music_prompt_input": _saved_ui_text(
            "elevenlabs_music_prompt",
            max_length=elevenlabs_music_service.MAX_PROMPT_LENGTH,
        ),
        "subtitle_enabled_checkbox": _saved_ui_bool("subtitle_enabled", True),
        "stroke_color_picker": _saved_ui_color("stroke_color", "#000000"),
        "stroke_width_slider": _saved_ui_number(
            "stroke_width", 1.5, 0.0, 10.0
        ),
        "loomloom_candidate_count": _saved_ui_number(
            "loomloom_candidate_count",
            3,
            1,
            loomloom.MAX_SCRIPT_CANDIDATES,
            int,
        ),
        "loomloom_script_duration_seconds": _saved_ui_number(
            "loomloom_script_duration_seconds", 60, 10, 600, int
        ),
        "ui_language": initial_ui_language,
        # El material local ya guardado en disco puede seguir reutilizándose si el usuario solo cambia el texto.
        "local_video_materials": [],
        # El callback del botón de generar registra primero la tarea, para que el acceso superior muestre de inmediato el número en ejecución.
        "active_generation_tasks": {},
        # La última tarea enviada desde la página actual. Desde que la generación se ejecuta en segundo plano, la página Fragment
        # consulta el estado con este ID; al refrescar ya no depende del script de página antiguo en ejecución.
        "current_generation_task_id": "",
        # LoomLoom La consulta de precio y la ejecución deben conservar entre reruns de Streamlit exactamente la misma entrada y
        # clientRequestId, para que los reintentos de red no generen tareas de pago duplicadas.
        "loomloom_script_batch": None,
        "loomloom_script_quote": None,
        "loomloom_script_input_signature": "",
        "loomloom_client_request_id": "",
        "loomloom_run_id": "",
        "loomloom_run_status": "",
        "loomloom_run_error": "",
        "loomloom_poll_failure_count": 0,
        "loomloom_poll_retry_after": 0.0,
        "loomloom_poll_paused": False,
        "loomloom_script_candidates": (),
        "loomloom_candidate_errors": (),
        "loomloom_selected_candidate": 0,
        "loomloom_video_batch": None,
        "loomloom_video_quote": None,
        "loomloom_video_input_signature": "",
        "loomloom_video_client_request_id": "",
        "loomloom_video_confirm_charge": False,
        "loomloom_video_quote_error_signature": "",
        "loomloom_video_quote_error": "",
        "loomloom_video_capability": None,
        "loomloom_video_capability_fingerprint": "",
        "loomloom_video_capability_load_attempt": "",
        "loomloom_video_capability_error": "",
        "loomloom_video_model_id": "",
        # Cuando el texto o la voz completa se acaban de generar, este resumen se consume antes de crear el control de número de videos y se
        # rellena automáticamente el número recomendado de materiales; tras consumirlo se vacía, para no sobrescribir ajustes manuales posteriores del usuario.
        "loomloom_video_scene_autofill_digest": "",
        "wavespeed_confirm_charge": False,
        "volcengine_seedance_confirm_charge": False,
        "infistar_confirm_charge": False,
        "jalapeno_confirm_charge": False,
        "ofox_confirm_charge": False,
        "metaso_minimax_confirm_charge": False,
        "muapi_confirm_charge": False,
        # AI El video se cobra por tramo de material; por defecto solo se genera un tramo, y el usuario aumenta la cantidad tras confirmar el resultado.
        "loomloom_video_scene_count": _saved_ui_number(
            "loomloom_video_scene_count",
            1,
            1,
            loomloom.MAX_VIDEO_SCENES,
            int,
        ),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_initialize_session_state()


def tr(key):
    loc = locales.get(st.session_state["ui_language"], {})
    value = loc.get("Translation", {}).get(key)
    if value is not None:
        return value
    # Las funciones nuevas mantienen primero español e inglés. Cuando a otros idiomas les falta una traducción, vuelven de forma unificada al inglés, para no copiar
    # locale el mismo inglés en varios y perder la sincronía a largo plazo; solo si el inglés tampoco tiene la clave se muestra la key original.
    return locales.get("en", {}).get("Translation", {}).get(key, key)


# -----------------------------------------------------------------------------
# Gestión de tareas: escaneo del historial, estado de ejecución, restauración de parámetros e interacción con la lista
# -----------------------------------------------------------------------------


def _format_task_time(timestamp):
    if not timestamp:
        return "-"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _format_task_subject(subject, max_length=30):
    subject = str(subject or "").replace("\n", " ").strip()
    if len(subject) <= max_length:
        return subject or "-"
    return f"{subject[:max_length]}..."


def _safe_load_task_script(task_path):
    script_file = os.path.join(task_path, "script.json")
    if not os.path.isfile(script_file):
        return {}

    try:
        with open(script_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            logger.warning(f"task script data is not an object: {script_file}")
            return {}
        return payload
    except Exception as e:
        logger.warning(f"failed to read task script data: {script_file}, {e}")
        return {}


def _find_final_task_video(task_path: str) -> str:
    """
    Devuelve el video final de menor número del directorio de la tarea.

    El flujo de composición también genera archivos combined, temp-clip y temporales de MoviePy, que no pueden
    indicar que la tarea terminó con éxito, así que aquí solo se acepta ``final-<número>.<extensión>``.
    """
    try:
        files = os.listdir(task_path)
    except OSError:
        return ""

    candidates = []
    for file_name in files:
        match = _FINAL_VIDEO_PATTERN.fullmatch(file_name)
        if match and os.path.isfile(os.path.join(task_path, file_name)):
            candidates.append((int(match.group("index")), file_name))

    if not candidates:
        return ""

    _, file_name = min(candidates, key=lambda item: item[0])
    return os.path.join(task_path, file_name)


def _build_restore_upload_requirements(params: Mapping) -> dict:
    """
    Registra las dependencias de archivos subidos de las tareas históricas que Streamlit no puede restaurar automáticamente.

    El navegador no permite que un programa vuelva a rellenar file_uploader, así que al restaurar una tarea hay que registrar aparte las dependencias de
    material local y audio personalizado, y comprobar antes de que el usuario regenere si ya las completó o sustituyó.
    """
    return {
        "local_materials": params.get("video_source") == "local",
        "custom_audio": bool(params.get("custom_audio_file")),
        "original_voice_name": params.get("voice_name") or "",
    }


def _get_unmet_restore_upload_requirements(
    requirements: Mapping | None,
    *,
    video_source: str,
    voice_name: str,
    has_local_materials: bool,
    has_custom_audio: bool,
    voice_mode: str | None = None,
) -> set[str]:
    """Devuelve las dependencias de archivos subidos históricos que el formulario actual aún no cumple."""
    requirements = requirements or {}
    unmet = set()

    if (
        requirements.get("local_materials")
        and video_source == "local"
        and not has_local_materials
    ):
        unmet.add("local_materials")

    if requirements.get("custom_audio") and not has_custom_audio:
        if voice_mode is not None:
            # La nueva WebUI usa un modo de voz explícito. Que el usuario cambie a voz automática o sin voz indica
            # que ya sustituyó el audio subido histórico; solo si sigue eligiendo el modo de subida se exige volver a subir.
            if voice_mode == VOICE_MODE_UPLOAD:
                unmet.add("custom_audio")
        elif voice_name == requirements.get("original_voice_name", ""):
            # Mantiene el comportamiento compatible de los llamadores antiguos que deciden por la voz, para no afectar a la API ni a las herramientas de prueba existentes.
            unmet.add("custom_audio")

    return unmet


def _queue_task_restore(task_id):
    # La lista de tareas se ejecuta en un fragment y no puede modificar directamente el estado de los controles del formulario principal ya creados.
    # Aquí solo se registra la tarea candidata y se lanza un rerun de toda la página; la confirmación y la restauración de parámetros las gestiona de forma unificada la página principal.
    st.session_state["task_restore_candidate_id"] = task_id
    st.session_state["task_manager_popover_nonce"] = (
        st.session_state.get("task_manager_popover_nonce", 0) + 1
    )
    st.rerun(scope="app")


def _normalize_task_state(state):
    if state in (
        const.TASK_STATE_COMPLETE,
        const.TASK_STATE_FAILED,
        const.TASK_STATE_PROCESSING,
    ):
        return state
    try:
        return int(state)
    except (TypeError, ValueError):
        return state


def _active_generation_tasks():
    tasks = st.session_state.setdefault("active_generation_tasks", {})
    if not isinstance(tasks, dict):
        tasks = {}
        st.session_state["active_generation_tasks"] = tasks
    return tasks


def _add_active_generation_task(task_id, subject=None):
    tasks = _active_generation_tasks()
    task = tasks.setdefault(task_id, {})
    task["subject"] = subject or task.get("subject") or task_id
    task["mtime"] = task.get("mtime") or datetime.now().timestamp()


def _remove_active_generation_task(task_id):
    tasks = _active_generation_tasks()
    if task_id in tasks:
        del tasks[task_id]
    if st.session_state.get("pending_generation_task_id") == task_id:
        del st.session_state["pending_generation_task_id"]


def _prepare_generation_task():
    # st.button el on_click se dispara antes de volver a ejecutar el script de la página. Aquí se genera el ID de tarea por adelantado,
    # para que el acceso superior de gestión de tareas muestre el número «generando» en el mismo rerun.
    current_id = _current_generation_task_id()
    if current_id and webui_task.is_generation_active(current_id):
        st.session_state["pending_generation_task_id"] = current_id
        return
    task_id = str(uuid4())
    st.session_state["pending_generation_task_id"] = task_id
    subject = st.session_state.get("video_subject") or st.session_state.get(
        "video_script"
    )
    _add_active_generation_task(task_id, subject=subject)


def _task_state_label(state, has_video):
    normalized_state = _normalize_task_state(state)
    if normalized_state == const.TASK_STATE_COMPLETE:
        return tr("Task Status Complete")
    if normalized_state == const.TASK_STATE_FAILED:
        return tr("Task Status Failed")
    if normalized_state == const.TASK_STATE_PROCESSING:
        return tr("Task Status Processing")
    if has_video:
        return tr("Task Status Complete")
    return tr("Task Status History")


def _task_state_filter_key(task):
    normalized_state = _normalize_task_state(task.get("state"))
    if normalized_state == const.TASK_STATE_PROCESSING:
        return "processing"
    if normalized_state == const.TASK_STATE_FAILED:
        return "failed"
    if normalized_state == const.TASK_STATE_COMPLETE or task["video_file"]:
        return "complete"
    return "history"


def _scan_history_tasks(limit=30):
    tasks_root = utils.task_dir()
    if not os.path.isdir(tasks_root):
        return []

    # El fragment de gestión de tareas se refresca cada dos segundos. Primero solo lee los metadatos de directorio, que son baratos, y recorta las tareas
    # más recientes, y luego analiza script.json y la lista de videos, para no escanear todo una y otra vez cuando hay muchas tareas históricas.
    task_entries = []
    try:
        with os.scandir(tasks_root) as entries:
            for entry in entries:
                try:
                    if entry.name.startswith(".") or not entry.is_dir(
                        follow_symlinks=False
                    ):
                        continue
                    task_entries.append(
                        (
                            entry.stat(follow_symlinks=False).st_mtime,
                            entry.name,
                            entry.path,
                        )
                    )
                except OSError as e:
                    # Un directorio de tarea concreto puede estar borrándose, y eso no debería invalidar todo el panel de tareas.
                    logger.debug(f"skip unavailable task directory: {entry.path}, {e}")
    except OSError as e:
        logger.warning(f"failed to scan task directory: {tasks_root}, {e}")
        return []

    task_entries.sort(key=lambda item: item[0], reverse=True)
    tasks = []
    for mtime, name, task_path in task_entries[:limit]:
        script_data = _safe_load_task_script(task_path)
        params_data = script_data.get("params", {})
        if not isinstance(params_data, dict):
            params_data = {}
        script_text = script_data.get("script", "")
        if not isinstance(script_text, str):
            script_text = ""
        video_file = _find_final_task_video(task_path)
        subject = (
            params_data.get("video_subject")
            or script_text[:40]
            or name
        )
        tasks.append(
            {
                "task_id": name,
                "subject": subject,
                "state": const.TASK_STATE_COMPLETE if video_file else None,
                "progress": 100 if video_file else 0,
                "mtime": mtime,
                "task_path": task_path,
                "video_file": video_file,
                "source": "history",
            }
        )

    return tasks


def _collect_task_summaries(limit=20):
    history_tasks = {task["task_id"]: task for task in _scan_history_tasks(limit=50)}
    active_tasks = _active_generation_tasks()

    try:
        runtime_tasks, _ = sm.state.get_all_tasks(1, 50)
    except Exception as e:
        logger.warning(f"failed to load runtime tasks: {e}")
        runtime_tasks = []

    # The paginated state view can omit this session's newer tasks after 50
    # older records. Read those active IDs directly so a completed or failed
    # task cannot remain labelled as processing forever.
    runtime_ids = {task.get("task_id") for task in runtime_tasks}
    for task_id in active_tasks:
        if task_id in runtime_ids:
            continue
        try:
            task = sm.state.get_task(task_id)
        except Exception as e:
            logger.warning(f"failed to load active task {task_id}: {e}")
            continue
        if task:
            runtime_tasks.append(task)

    for task in runtime_tasks:
        task_id = task.get("task_id", "")
        if not task_id:
            continue

        task_path = os.path.join(utils.task_dir(), task_id)
        history_task = history_tasks.get(task_id, {})
        video_files = task.get("videos") or []
        video_file = (
            video_files[0] if video_files else history_task.get("video_file", "")
        )
        subject = (
            task.get("video_subject")
            or history_task.get("subject")
            or (task.get("script", "")[:40] if task.get("script") else "")
            or task_id
        )
        task_mtime = active_tasks.get(task_id, {}).get("mtime") or history_task.get(
            "mtime", 0
        )
        if os.path.isdir(task_path):
            try:
                task_mtime = os.path.getmtime(task_path)
            except OSError:
                # Another session can delete this directory between isdir and
                # getmtime. Keep rendering the persisted task state.
                pass

        history_tasks[task_id] = {
            "task_id": task_id,
            "subject": subject,
            "state": task.get("state"),
            "cross_post_state": task.get("cross_post_state"),
            "progress": int(task.get("progress", 0) or 0),
            "mtime": task_mtime,
            "task_path": task_path,
            "video_file": video_file,
            "source": "runtime",
        }

    for task_id, active_task in active_tasks.items():
        history_task = history_tasks.get(task_id, {})
        if history_task and _task_state_filter_key(history_task) in {
            "complete",
            "failed",
        }:
            # La marca active de la sesión solo cubre la ventana muy corta antes de que la tarea llegue al almacén de estado.
            # Al terminar la tarea en segundo plano manda el estado final real, sin volver a mostrar como «generando» una tarea fallida.
            continue

        task_path = os.path.join(utils.task_dir(), task_id)
        history_tasks[task_id] = {
            "task_id": task_id,
            "subject": active_task.get("subject")
            or history_task.get("subject")
            or task_id,
            "state": const.TASK_STATE_PROCESSING,
            "progress": history_task.get("progress", 0),
            "mtime": active_task.get("mtime")
            or history_task.get("mtime", datetime.now().timestamp()),
            "task_path": task_path,
            "video_file": history_task.get("video_file", ""),
            "source": "active",
        }

    tasks = list(history_tasks.values())
    return sorted(tasks, key=lambda item: item["mtime"], reverse=True)[:limit]


def _is_headless_server():
    # Docker o en despliegues en servidores sin escritorio, el proceso de la WebUI no tiene acceso al entorno de escritorio del usuario:
    # xdg-open / webbrowser solo fallaría en silencio dentro del contenedor. En ese caso hay que previsualizar en el navegador
    # el video y sustituir abrir el directorio por un aviso con la ruta. Los despliegues de escritorio macOS/Windows no se ven afectados.
    if sys.platform == "darwin" or sys.platform.startswith("win"):
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _open_task_path(task_path):
    tasks_root = os.path.abspath(utils.task_dir())
    normalized_path = os.path.abspath(task_path)
    if not normalized_path.startswith(tasks_root + os.sep):
        logger.warning(f"invalid task folder path: {normalized_path}")
        return
    if not os.path.isdir(normalized_path):
        return
    if _is_headless_server():
        # storage el directorio suele estar montado como volumen en el host, así que basta con indicar la ruta relativa para localizar el archivo.
        rel_path = os.path.relpath(normalized_path, os.path.dirname(tasks_root))
        st.toast(f"{tr('Open Task Folder')}: ./storage/{rel_path}", icon="📂")
        return
    webbrowser.open(f"file://{normalized_path}")


def _open_task_video(video_file):
    tasks_root = os.path.abspath(utils.task_dir())
    normalized_file = os.path.abspath(video_file)

    # La ruta del video viene del escaneo del directorio de tareas o del estado en tiempo de ejecución. Aun así se limita a abrir solo archivos dentro del
    # directorio de tareas, para que las acciones de la UI no se conviertan, con rutas anómalas, en la capacidad de abrir cualquier archivo local.
    if not normalized_file.startswith(tasks_root + os.sep):
        logger.warning(f"invalid task video path: {normalized_file}")
        return
    if not os.path.isfile(normalized_file):
        logger.warning(f"task video does not exist: {normalized_file}")
        return

    if _is_headless_server():
        # Sin entorno de escritorio se incrusta un reproductor de vista previa en el panel de tareas, en lugar de llamar al reproductor del sistema.
        st.session_state["task_preview_video_file"] = normalized_file
        return

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", normalized_file])
        elif sys.platform.startswith("win"):
            os.startfile(normalized_file)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", normalized_file])
    except Exception as e:
        logger.error(f"failed to open task video: {normalized_file}, {e}")


def _delete_task(task_id, task_path, task_state=None):
    # El estado mostrado en la página puede ir por detrás de la tarea en segundo plano. Antes de borrar se comprueban el estado recibido, las tareas activas
    # de la sesión actual y el estado más reciente, para no borrar por error una tarea recién iniciada o que ya produjo videos intermedios.
    current_task = None
    try:
        current_task = sm.state.get_task(task_id)
    except Exception as e:
        logger.exception(f"failed to verify task state before deletion: {task_id}, {e}")
        return False

    task_snapshot = dict(current_task or {})
    task_snapshot.setdefault("state", task_state)
    if task_id in _active_generation_tasks():
        task_snapshot["state"] = const.TASK_STATE_PROCESSING

    if tm.is_task_busy(task_snapshot):
        logger.warning(f"refused to delete running task: {task_id}")
        return False

    tasks_root = os.path.abspath(utils.task_dir())
    normalized_path = os.path.abspath(task_path)

    # Borrar una tarea elimina su estado y los archivos generados localmente. Aquí debe limitarse a storage/tasks
    # , para que un task_path anómalo no borre por error otros directorios locales.
    if not normalized_path.startswith(tasks_root + os.sep):
        logger.warning(f"invalid task folder path for deletion: {normalized_path}")
        return False

    try:
        if os.path.isdir(normalized_path):
            shutil.rmtree(normalized_path)
        # Keep the persisted task available for retry when file removal fails
        # (for example, a video is still locked by another process on Windows).
        if hasattr(sm.state, "delete_task"):
            sm.state.delete_task(task_id)
        logger.info(f"deleted task: {task_id}")
        return True
    except Exception as e:
        logger.exception(f"failed to delete task: {task_id}, {e}")
        return False


def _count_processing_tasks(tasks):
    # El acceso superior de gestión de tareas solo necesita mostrar el número de tareas «generando».
    # Aquí se reutiliza la key de estado interna, para no depender de textos multilingües que darían recuentos distintos según el idioma.
    processing_task_ids = {
        task["task_id"]
        for task in tasks
        if _task_state_filter_key(task) == "processing"
    }
    return len(processing_task_ids)


def _task_manager_label(processing_count):
    label = tr("Task Manager")
    if processing_count <= 0:
        return label
    return f"{label} · {processing_count}"


def _build_video_download_name(subject, index, total):
    """Genera, a partir del tema del video, un nombre de archivo de descarga seguro en todas las plataformas."""
    safe_subject = _DOWNLOAD_FILENAME_INVALID_PATTERN.sub(" ", str(subject or ""))
    safe_subject = re.sub(r"\s+", " ", safe_subject).strip(" .")[:80].rstrip(" .")
    if not safe_subject:
        safe_subject = "video"
    # Win32 ignora los espacios y puntos finales antes de la extensión al reconocer nombres de dispositivo. Coincide con las reglas existentes
    # de subida de música de fondo, para evitar que ``CON .topic`` se salte la protección de nombres reservados.
    windows_basename = safe_subject.split(".", 1)[0].rstrip(" .").upper()
    if windows_basename in _WINDOWS_RESERVED_FILENAMES:
        safe_subject = f"_{safe_subject}"

    suffix = f"-{index}" if total > 1 else ""
    return f"{safe_subject}{suffix}.mp4"


def _render_task_table(filtered_tasks, key_prefix):
    with st.container(key=f"task_table_header_{key_prefix}"):
        header_cols = st.columns([1.1, 1.7, 3.0, 0.8, 1.6], vertical_alignment="center")
        header_cols[0].caption(tr("Task Status"))
        header_cols[1].caption(tr("Task Updated At"))
        header_cols[2].caption(tr("Task Subject"))
        header_cols[3].caption(tr("Task Progress"))
        header_cols[4].caption(tr("Task Actions"))

    if not filtered_tasks:
        st.info(tr("No Tasks Match Filter"))
        return

    visible_tasks = filtered_tasks[:12]
    list_height = min(390, max(96, len(visible_tasks) * 58))
    with st.container(height=list_height, border=False):
        for task in visible_tasks:
            task_id = task["task_id"]
            has_video = bool(task["video_file"] and os.path.isfile(task["video_file"]))
            is_processing = _task_state_filter_key(task) == "processing"
            is_busy = is_processing or tm.is_task_busy(task)
            has_restore_data = os.path.isfile(
                os.path.join(task["task_path"], "script.json")
            )
            safe_task_key = "".join(ch if ch.isalnum() else "_" for ch in task_id)[:40]

            # Usa bordered container + columns nativos de Streamlit para conservar las acciones de cada fila.
            # Frente a una tabla HTML/CSS personalizada, es más estable ante cambios de versión de Streamlit;
            # frente a dataframe, conserva acciones en línea como reproducir, abrir directorio o borrar.
            with st.container(
                key=f"task_row_{key_prefix}_{safe_task_key}", border=True
            ):
                row_cols = st.columns(
                    [1.1, 1.7, 3.0, 0.8, 1.6],
                    vertical_alignment="center",
                )
                row_cols[0].write(_task_state_label(task["state"], has_video))
                row_cols[1].write(_format_task_time(task["mtime"]))
                row_cols[2].write(_format_task_subject(task["subject"]))
                row_cols[3].write(f"{task['progress']}%")

                action_cols = row_cols[4].columns(
                    4,
                    vertical_alignment="center",
                    gap="small",
                )
                with action_cols[0]:
                    play_label = tr("Play")
                    if st.button(
                        play_label,
                        key=f"play_task_{key_prefix}_{task_id}",
                        use_container_width=True,
                        icon=":material/play_arrow:",
                        help=play_label,
                        disabled=not has_video,
                    ):
                        _open_task_video(task["video_file"])

                with action_cols[1]:
                    open_label = tr("Open Task Folder")
                    if st.button(
                        open_label,
                        key=f"open_task_{key_prefix}_{task_id}",
                        use_container_width=True,
                        icon=":material/folder_open:",
                        help=open_label,
                    ):
                        _open_task_path(task["task_path"])

                with action_cols[2]:
                    restore_label = tr("Regenerate Task")
                    if st.button(
                        restore_label,
                        key=f"restore_task_{key_prefix}_{task_id}",
                        use_container_width=True,
                        icon=":material/replay:",
                        help=restore_label,
                        disabled=is_processing or not has_restore_data,
                    ):
                        _queue_task_restore(task_id)

                with action_cols[3]:
                    delete_label = tr("Delete Task")
                    delete_help = (
                        f"{delete_label} ({tr('Task Status Processing')})"
                        if is_busy
                        else delete_label
                    )
                    if st.button(
                        delete_label,
                        key=f"delete_task_{key_prefix}_{task_id}",
                        use_container_width=True,
                        icon=":material/delete:",
                        help=delete_help,
                        disabled=is_busy,
                    ):
                        if _delete_task(task_id, task["task_path"], task["state"]):
                            st.toast(tr("Task Deleted"))
                            st.rerun()
                        else:
                            st.error(tr("Task Delete Failed"))


def _render_task_manager_panel(tasks=None):
    tasks = tasks if tasks is not None else _collect_task_summaries()
    if not tasks:
        st.info(tr("No Tasks Yet"))
        return

    # Streamlit 1.59 Admite el renderizado diferido de Tabs con estado. Al cambiar solo se reconstruye la lista actual,
    # para que el Fragment periódico no cree cada dos segundos cuatro juegos de filas de tareas y botones.
    status_tabs = [
        ("all", tr("All Tasks")),
        ("processing", tr("Task Status Processing")),
        ("complete", tr("Task Status Complete")),
        ("failed", tr("Task Status Failed")),
    ]
    tabs = st.tabs(
        [label for _, label in status_tabs],
        key="task_manager_status_tabs",
        on_change="rerun",
    )
    for (status_key, _), tab in zip(status_tabs, tabs):
        if not tab.open:
            continue
        with tab:
            filtered_tasks = [
                task
                for task in tasks
                if status_key == "all" or _task_state_filter_key(task) == status_key
            ]
            _render_task_table(filtered_tasks, status_key)

    _render_task_video_preview()


def _render_task_video_preview():
    # Alternativa en el navegador del botón «Reproducir» en despliegues sin escritorio: renderiza el reproductor al final del panel de tareas.
    preview_file = st.session_state.get("task_preview_video_file")
    if not preview_file:
        return

    tasks_root = os.path.abspath(utils.task_dir())
    if not (
        preview_file.startswith(tasks_root + os.sep) and os.path.isfile(preview_file)
    ):
        st.session_state.pop("task_preview_video_file", None)
        return

    st.divider()
    preview_cols = st.columns([5, 1], vertical_alignment="center")
    task_name = os.path.basename(os.path.dirname(preview_file))
    preview_cols[0].caption(f"{os.path.basename(preview_file)} · {task_name}")
    closed = preview_cols[1].button(
        "✕",
        key="close_task_video_preview",
        use_container_width=True,
        help=tr("Close"),
    )
    if closed:
        st.session_state.pop("task_preview_video_file", None)
        return
    st.video(preview_file)


@st.fragment(run_every="2s")
def _render_task_manager_entry():
    # La generación puede lanzarse desde la página actual u otras. El acceso se refresca periódicamente con un fragment propio,
    # actualizando solo el número de tareas y el contenido del popover, sin interrumpir lo que se escribe en el formulario principal.
    task_summaries = _collect_task_summaries()
    processing_task_count = _count_processing_tasks(task_summaries)
    with st.container(key="task_manager_entry", width="content"):
        with st.popover(
            _task_manager_label(processing_task_count),
            width="content",
            key=(
                "task_manager_popover_"
                f"{st.session_state.get('task_manager_popover_nonce', 0)}"
            ),
        ):
            _render_task_manager_panel(task_summaries)


def _load_task_restore_payload(task_id):
    tasks_root = os.path.realpath(utils.task_dir())
    task_path = os.path.realpath(os.path.join(tasks_root, str(task_id)))
    try:
        if os.path.commonpath([tasks_root, task_path]) != tasks_root:
            raise ValueError("task path is outside the task directory")
    except ValueError as e:
        logger.warning(f"invalid task restore path: {task_id}, {e}")
        return None

    script_data = _safe_load_task_script(task_path)
    raw_params = script_data.get("params")
    if not isinstance(raw_params, dict):
        logger.warning(f"task has no restorable parameters: {task_id}")
        return None

    params_input = dict(raw_params)
    if script_data.get("script"):
        params_input["video_script"] = script_data["script"]
    if script_data.get("search_terms"):
        params_input["video_terms"] = script_data["search_terms"]

    try:
        params = VideoParams.model_validate(params_input).model_dump(mode="json")
    except Exception as e:
        logger.warning(f"failed to validate task restore parameters: {task_id}, {e}")
        return None

    return {
        "task_id": str(task_id),
        "subject": params.get("video_subject") or script_data.get("script") or task_id,
        "params": params,
    }


def _infer_tts_server_from_voice(voice_name):
    if voice.is_no_voice(voice_name):
        return voice.NO_VOICE_NAME
    if voice.is_siliconflow_voice(voice_name):
        return "siliconflow"
    if voice.is_gemini_voice(voice_name):
        return "gemini-tts"
    if voice.is_mimo_voice(voice_name):
        return "mimo-tts"
    if voice.is_minimax_voice(voice_name):
        return "minimax-tts"
    if voice.is_elevenlabs_voice(voice_name):
        return "elevenlabs"
    if voice.is_chatterbox_voice(voice_name):
        return "chatterbox"
    if voice.is_kokoro_voice(voice_name):
        return "kokoro"
    if voice.is_fish_audio_voice(voice_name):
        return "fish_audio"
    if voice.is_voxcpm_voice(voice_name):
        return "voxcpm"
    if voice.is_azure_v2_voice(voice_name):
        return "azure-tts-v2"
    return "azure-tts-v1"


def _set_stable_widget_value(key, value):
    if value is not None:
        st.session_state[localized_widget_key(key)] = value


def _apply_pending_task_restore():
    payload = st.session_state.pop("task_restore_payload", None)
    if not payload:
        return False

    _apply_restored_params(payload["params"])
    st.session_state["task_restore_succeeded"] = True
    logger.info(f"restored task configuration: {payload['task_id']}")
    return True


def _apply_restored_params(params):
    """
    Escribe un conjunto completo de parámetros de generación en el estado de los controles de la página.

    La restauración de tareas históricas y la importación de preajustes usan el mismo modelo de parámetros, así que comparten una implementación, para evitar
    que al añadir campos solo se actualice una de las rutas. El llamador debe ejecutarlo antes de renderizar cualquier control, o
    Streamlit rechazará modificar el estado de controles ya instanciados.
    """
    video_terms = params.get("video_terms") or ""
    if isinstance(video_terms, list):
        video_terms = ", ".join(str(term) for term in video_terms)

    # Ajustes de texto y de guion avanzados.
    st.session_state["video_subject"] = params.get("video_subject") or ""
    st.session_state["video_script"] = params.get("video_script") or ""
    st.session_state["video_terms"] = str(video_terms)
    _set_stable_widget_value(
        "script_language_select", params.get("video_language") or ""
    )
    st.session_state["paragraph_number_input"] = params.get("paragraph_number", 1)
    st.session_state["video_script_prompt"] = params.get("video_script_prompt") or ""
    st.session_state["custom_system_prompt"] = (
        params.get("custom_system_prompt") or llm.DEFAULT_SCRIPT_SYSTEM_PROMPT
    )

    # Ajustes de video. El control de subida de material no se puede escribir desde el servidor, así que el usuario debe volver a elegir el material local.
    video_source = params.get("video_source") or "pexels"
    _set_stable_widget_value("video_source_select", video_source)
    _set_stable_widget_value(
        "video_concat_mode_select", params.get("video_concat_mode") or "random"
    )
    _set_stable_widget_value(
        "video_transition_mode_select",
        params.get("video_transition_mode") or VideoTransitionMode.none.value,
    )
    _set_stable_widget_value(
        f"video_aspect_for_{video_source}",
        params.get("video_aspect") or VideoAspect.portrait.value,
    )
    _set_stable_widget_value(
        "video_fit_mode_select",
        params.get("video_fit_mode") or VideoFitMode.cover.value,
    )
    _set_stable_widget_value(
        "image_motion_select", params.get("image_motion") or ImageMotion.zoom.value
    )
    _set_stable_widget_value(
        "video_clip_duration_select", params.get("video_clip_duration", 3)
    )
    _set_stable_widget_value(
        "video_clip_speed_slider",
        # API puede escribir velocidades fuera del rango de la WebUI; la fase de generación de tareas las normaliza con seguridad, pero
        # el historial puede conservar el valor original. Se vuelve a normalizar antes de restaurar la tarea, para no Streamlit
        # slider inyectar valores fuera de rango, NaN o infinitos que dejen el estado del control en mal estado.
        utils.normalize_clip_speed(params.get("video_clip_speed", 1.0)),
    )
    _set_stable_widget_value("video_count_select", params.get("video_count", 1))
    st.session_state["match_materials_to_script"] = bool(
        params.get("match_materials_to_script", False)
    )

    # Ajustes de audio. El TTS server no se guardaba en las tareas antiguas; se deduce del voice_name histórico.
    voice_name = params.get("voice_name") or voice.NO_VOICE_NAME
    tts_server = _infer_tts_server_from_voice(voice_name)
    if params.get("custom_audio_file"):
        voice_mode = VOICE_MODE_UPLOAD
    elif voice.is_no_voice(voice_name):
        voice_mode = VOICE_MODE_NONE
    else:
        voice_mode = VOICE_MODE_TTS
    _set_stable_widget_value("voice_mode_control", voice_mode)
    if tts_server != voice.NO_VOICE_NAME:
        _set_stable_widget_value("tts_server_select", tts_server)
        _set_stable_widget_value(f"speech_synthesis_select_{tts_server}", voice_name)
    _set_stable_widget_value("voice_volume_select", params.get("voice_volume", 1.0))
    _set_stable_widget_value("voice_rate_select", params.get("voice_rate", 1.0))
    bgm_type = params.get("bgm_type") or ""
    _set_stable_widget_value("bgm_type_select", bgm_type)
    _set_stable_widget_value("bgm_volume_select", params.get("bgm_volume", 0.2))
    if bgm_type == "preset" and params.get("bgm_file"):
        # El control de canciones predefinidas usa el nombre de archivo como valor de negocio estable. Las tareas históricas pueden guardar una ruta absoluta o
        # relativa; tomando siempre el basename se puede emparejar con la lista de canciones enumerada de forma segura.
        _set_stable_widget_value(
            "preset_song_select", os.path.basename(str(params["bgm_file"]))
        )
    st.session_state["custom_bgm_file_input"] = params.get("bgm_file") or ""
    st.session_state["sonilo_bgm_prompt_input"] = (
        params.get("video_music_prompt") or params.get("sonilo_bgm_prompt") or ""
    )
    st.session_state["elevenlabs_music_prompt_input"] = (
        params.get("video_music_prompt") or ""
    )

    # Ajustes de subtítulos. Se hace una limitación mínima de los valores fuera de rango de tareas antiguas, para que el Slider pueda inicializarse.
    st.session_state["subtitle_enabled_checkbox"] = bool(
        params.get("subtitle_enabled", True)
    )
    st.session_state["opening_title_enabled_checkbox"] = bool(
        params.get("opening_title_enabled") or params.get("opening_title")
    )
    st.session_state["opening_title_input"] = params.get("opening_title") or ""
    _set_stable_widget_value("font_name_select", params.get("font_name") or "")
    _set_stable_widget_value(
        "subtitle_position_select", params.get("subtitle_position") or "bottom"
    )
    _set_stable_widget_value(
        "subtitle_display_mode_select", params.get("subtitle_display_mode") or "sentence"
    )
    _set_stable_widget_value(
        "subtitle_animation_select", params.get("subtitle_animation") or "none"
    )
    custom_position = min(100.0, max(0.0, float(params.get("custom_position", 70.0))))
    st.session_state["custom_position_input"] = str(custom_position)
    st.session_state["font_color_picker"] = params.get("text_fore_color") or "#FFFFFF"
    st.session_state["font_size_slider"] = min(
        100, max(30, int(params.get("font_size", 60)))
    )
    st.session_state["stroke_color_picker"] = params.get("stroke_color") or "#000000"
    st.session_state["stroke_width_slider"] = min(
        10.0, max(0.0, float(params.get("stroke_width", 1.5)))
    )
    background_color = params.get("text_background_color")
    background_enabled = bool(background_color)
    st.session_state["subtitle_background_enabled_checkbox"] = background_enabled
    if isinstance(background_color, str):
        st.session_state["subtitle_background_color_picker"] = background_color
    st.session_state["rounded_subtitle_background_checkbox"] = bool(
        params.get("rounded_subtitle_background", False) and background_enabled
    )

    st.session_state.pop("local_video_materials_uploader", None)
    # Las tareas históricas solo guardan las rutas de material, sin garantía de que esos archivos sigan existiendo en el entorno actual.
    # También vacía el material subido en caché de la página actual, para no usar por error los archivos de otra tarea tras restaurar.
    st.session_state["local_video_materials"] = []
    st.session_state.pop("custom_audio_file_uploader", None)
    st.session_state.pop("voxcpm_reference_audio_uploader", None)
    st.session_state.pop(VOXCPM_REFERENCE_AUDIO_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY, None)
    st.session_state.pop("voxcpm_prompt_audio_uploader", None)
    st.session_state.pop(VOXCPM_PROMPT_AUDIO_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_PROMPT_TEXT_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_HIGH_FIDELITY_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_PROMPT_EXAMPLE_MODE_SESSION_KEY, None)
    st.session_state.pop("custom_bgm_uploader", None)
    st.session_state.pop("custom_bgm_validation", None)
    st.session_state["task_restore_upload_requirements"] = (
        _build_restore_upload_requirements(params)
    )

    return True


def _dismiss_task_restore_dialog():
    st.session_state.pop("task_restore_candidate_id", None)


@st.dialog(
    tr("Regenerate Task"),
    width="small",
    on_dismiss=_dismiss_task_restore_dialog,
)
def _render_task_restore_dialog(task_id):
    payload = _load_task_restore_payload(task_id)
    if payload is None:
        st.error(tr("Task Restore Failed"))
        if st.button(tr("Cancel"), key="cancel_invalid_task_restore"):
            st.session_state.pop("task_restore_candidate_id", None)
            st.rerun(scope="app")
        return

    st.write(tr("Regenerate Task Confirmation"))
    st.caption(_format_task_subject(payload["subject"], max_length=80))
    cancel_col, load_col = st.columns(2)
    if cancel_col.button(
        tr("Cancel"),
        key="cancel_task_restore",
        use_container_width=True,
    ):
        st.session_state.pop("task_restore_candidate_id", None)
        st.rerun(scope="app")
    if load_col.button(
        tr("Load Task Configuration"),
        key="confirm_task_restore",
        type="primary",
        use_container_width=True,
    ):
        st.session_state["task_restore_payload"] = payload
        st.session_state.pop("task_restore_candidate_id", None)
        st.rerun(scope="app")


def _dismiss_settings_dialog():
    """Cierra el diálogo de ajustes y asegura que el siguiente rerun de toda la página no lo vuelva a abrir automáticamente."""
    st.session_state["settings_dialog_open"] = False


def _open_settings_dialog(target_tab=None):
    """Abre el diálogo de ajustes, con la posibilidad de situarse directamente en una pestaña de negocio concreta."""
    st.session_state["settings_dialog_open"] = True
    if not target_tab:
        target_tab = "general"
    if target_tab:
        # Aquí solo se guarda el ID de negocio estable, no el texto traducido; justo antes de crear las tabs se resuelve
        # el label según el idioma actual de la interfaz, para que el texto antiguo no se convierta en una opción ilegal tras cambiar de idioma.
        st.session_state["settings_dialog_target_tab"] = target_tab


def _open_material_settings_dialog():
    """Para el callback del componente de origen de video: abre directamente los ajustes del servicio de material."""
    _open_settings_dialog("material")


def _render_brand(available_update: str | None = None):
    """Renderiza el nombre del proyecto, la versión actual y el acceso opcional de actualización."""
    update_link = ""
    if available_update:
        update_label = html.escape(
            tr("Update Available").format(version=available_update)
        )
        # Streamlit sigue analizando con Markdown el HTML recibido. Aquí se mantiene el enlace en una sola línea,
        # para que la sangría de cadenas multilínea no se interprete como bloque de código y la página muestre directamente el código HTML.
        update_link = (
            '<a class="mpt-brand__update" '
            f'href="{version_checker.LATEST_RELEASE_PAGE_URL}" '
            'target="_blank" rel="noopener noreferrer" '
            f'aria-label="{update_label}" title="{update_label}">'
            f"{update_label}</a>"
        )
    st.markdown(
        f"""
        <h1 class="mpt-brand">
            <span class="mpt-brand__name">MoneyPrinterTurbo</span>
            <a class="mpt-brand__version"
               href="https://github.com/harry0703/MoneyPrinterTurbo"
               target="_blank"
               rel="noopener noreferrer"
               aria-label="Open MoneyPrinterTurbo on GitHub"
               title="Open project on GitHub">v{html.escape(str(config.project_version))}</a>
            {update_link}
        </h1>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every="1s")
def _render_pending_version_check():
    """Mientras la comprobación no ha terminado solo se refresca la zona de marca, para no bloquear ni volver a ejecutar todo el formulario de la página."""
    snapshot = version_checker.poll_available_update(config.project_version)
    if snapshot.complete:
        # Al terminar la comprobación se refresca una vez toda la página, para que la barra superior pase a renderizado estático y deje de sondear el fragment.
        # Ese refresco ocurre después de completar la petición en segundo plano y no retrasa el resto del contenido inicial de la página.
        st.rerun(scope="app")
    _render_brand()


def _render_top_bar():
    """Renderiza la barra superior de la página, formada por la marca, la gestión de tareas, los ajustes y el cambio de idioma."""
    # La barra superior se divide en dos zonas independientes: marca y acciones. En pantallas estrechas, Streamlit
    # ambas zonas saltan de línea como bloque, y dentro de la zona de acciones se salta de línea automáticamente según el ancho restante.
    with st.container(key="top_bar"):
        brand_col, actions_col = st.columns(
            [3.5, 2.0],
            vertical_alignment="center",
            gap="small",
        )

    with brand_col:
        update_snapshot = version_checker.poll_available_update(config.project_version)
        if update_snapshot.complete:
            _render_brand(update_snapshot.available_version)
        else:
            _render_pending_version_check()

    with actions_col:
        with st.container(
            key="top_bar_actions",
            horizontal=True,
            horizontal_alignment="right",
            vertical_alignment="center",
            gap="small",
            width="stretch",
        ):
            _render_task_manager_entry()

            st.button(
                tr("Settings"),
                key="open_settings_dialog_button",
                type="secondary",
                icon=":material/settings:",
                width="content",
                on_click=_open_settings_dialog,
            )

            language_codes = list(locales.keys())
            selected_index = 0
            for i, code in enumerate(language_codes):
                if code == st.session_state.get("ui_language", ""):
                    selected_index = i

            selected_language_code = st.selectbox(
                "Language / Idioma",
                options=language_codes,
                index=selected_index,
                format_func=lambda code: locales[code].get("Language", code),
                key="top_language_code_selector",
                label_visibility="collapsed",
                width=180,
            )
            if selected_language_code:
                previous_language = st.session_state.get("ui_language", "")
                if selected_language_code != previous_language:
                    logger.info(
                        "UI language changed by user: "
                        f"previous_language={previous_language or '<empty>'}, "
                        f"selected_language={selected_language_code}"
                    )
                    st.session_state["ui_language"] = selected_language_code
                    # La detección automática del navegador solo afecta a la sesión actual; solo cuando el usuario cambia el desplegable por su cuenta
                    # se escribe en config.toml, y las sesiones nuevas posteriores usarán con prioridad esa elección explícita.
                    _set_runtime_config("ui", "language", selected_language_code)
                    _save_runtime_config()
                    # Cambiar de idioma vuelve a ejecutar primero la barra superior, y lanza el rerun antes de renderizar los controles del cuerpo.
                    # Conserva explícitamente el contenido creado, para que, tras limpiar Streamlit el estado de los controles antiguos,
                    # la página en el nuevo idioma no deje vacíos el tema, el texto y las palabras clave ya introducidos.
                    for content_key in ("video_subject", "video_script", "video_terms"):
                        if content_key in st.session_state:
                            st.session_state[content_key] = st.session_state[content_key]
                    # Tras cambiar de idioma se fuerza un refresco, para que el selectbox no siga mostrando el texto del idioma anterior.
                    st.rerun()


support_locales = [
    "ca-ES",
    "zh-CN",
    "zh-HK",
    "zh-TW",
    "de-DE",
    "en-US",
    "es-ES",
    "fr-FR",
    "id-ID",
    "it-IT",
    "ru-RU",
    "vi-VN",
    "th-TH",
    "tr-TR",
]


# -----------------------------------------------------------------------------
# Componentes de UI comunes, caché de recursos y logs
# -----------------------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def get_all_fonts():
    # El directorio de fuentes cambia poco, pero Streamlit hace rerun de la página en cada interacción con un control. Una caché de ciclo corto
    # evita os.walk repetidos seguidos, y garantiza que las fuentes nuevas se detecten en 30 segundos como máximo.
    fonts = []
    for root, dirs, files in os.walk(font_dir):
        for file in files:
            if file.endswith(".ttf") or file.endswith(".ttc"):
                # Keep each option tied to its file, including nested fonts.
                relative_path = os.path.relpath(os.path.join(root, file), font_dir)
                fonts.append(relative_path.replace(os.sep, "/"))
    fonts.sort()
    return fonts


@st.cache_data(ttl=30, show_spinner=False)
def get_all_songs():
    # La música de fondo usa la misma estrategia de ciclo corto que las fuentes, sin caché permanente, equilibrando el rendimiento del rerun y
    # el caso de que el usuario añada archivos de música a mano mientras está en marcha.
    songs = []
    for root, dirs, files in os.walk(song_dir):
        for file in files:
            if file.endswith(".mp3"):
                songs.append(file)
    return songs


def open_task_folder(task_id):
    try:
        # task_id debe ser siempre un UUID generado por el servidor. Aquí se valida primero el formato, para que valores anómalos no
        # accedan, concatenando rutas, a ubicaciones fuera del directorio de tareas, ni provoquen al abrir el directorio
        # que la shell de la plataforma interprete caracteres especiales.
        normalized_task_id = str(UUID(str(task_id)))
        tasks_root = os.path.abspath(utils.task_dir())
        path = os.path.abspath(os.path.join(tasks_root, normalized_task_id))

        # Aunque el UUID pase la validación, se vuelve a confirmar que la ruta final sigue dentro del directorio raíz de tareas, para no
        # introducir riesgos de path traversal si en el futuro un llamador cambia el origen de task_id.
        if not path.startswith(tasks_root + os.sep):
            logger.warning(f"invalid task folder path: {path}")
            return

        if os.path.isdir(path):
            webbrowser.open(f"file://{path}")
    except Exception as e:
        logger.exception(f"failed to open task folder: task_id={task_id}, error={e}")


@st.cache_resource
def init_log():
    # El Handler de log básico es un recurso a nivel de proceso, no estado de la sesión de la página. Streamlit hace rerun del script de la página en cada
    # interacción con un componente, y la recarga en caliente del código también puede invalidar la caché. La inicialización de logs solo puede
    # sustituir con precisión el Handler del terminal, sin vaciar el Handler temporal de la WebUI que usan las tareas en generación.
    _lvl = get_logging_options()[0]

    return configure_terminal_logger(
        sys.stdout,
        level=_lvl,
        colorize=True,
    )


init_log()


def tr_optional(key, fallback_language=""):
    loc = locales.get(st.session_state["ui_language"], {})
    value = loc.get("Translation", {}).get(key, "")
    if not value and fallback_language:
        fallback_loc = locales.get(fallback_language, {})
        value = fallback_loc.get("Translation", {}).get(key, "")
    return value if value else ""


def render_onboarding_tour():
    # La guía solo cubre tres accesos estables, sin intentar controlar Dialog, Tabs ni formularios de negocio. Así
    # los usuarios nuevos entienden el flujo completo, sin acoplar el estado de la guía al ciclo de vida dinámico de los componentes de Streamlit.
    steps = [
        Tour.bind(
            "open_settings_dialog_button",
            title=tr("Onboarding Model Settings Title"),
            desc=tr("Onboarding Model Settings Description"),
            side="bottom",
            align="end",
        ),
        Tour.bind(
            "main_settings_grid",
            title=tr("Onboarding Creation Settings Title"),
            desc=tr("Onboarding Creation Settings Description"),
            side="top",
            align="center",
        ),
        Tour.bind(
            "generate_video_button",
            title=tr("Onboarding Generate Video Title"),
            desc=tr("Onboarding Generate Video Description"),
            side="top",
            align="center",
        ),
    ]

    # streamlit-tour 1.1.0 no expone los textos de navegación en los parámetros de construcción de Python, pero internamente
    # Driver.js permite sobrescribir el texto de los botones en la configuración del popover de cada paso. Aquí se inyectan de forma unificada los textos
    # localizados, escapando el contenido HTML, porque el componente renderiza estos campos mediante innerHTML.
    previous_text = html.escape(tr("Onboarding Previous"))
    next_text = html.escape(tr("Onboarding Next"))
    done_text = html.escape(tr("Onboarding Done"))
    for index, step in enumerate(steps):
        step.popover["prevBtnText"] = f"&larr; {previous_text}"
        # Driver.js al fusionar la configuración de cada paso sobrescribe la plantilla de progreso con las variables ya sustituidas, así que se
        # escriben directamente el paso actual y el total de pasos, para que la página no muestre el marcador {{current}} sin resolver.
        step.popover["progressText"] = f"{index + 1} / {len(steps)}"
        if index == len(steps) - 1:
            step.popover["doneBtnText"] = done_text
        else:
            step.popover["nextBtnText"] = f"{next_text} &rarr;"

    tour = Tour(
        steps=steps,
        key=ONBOARDING_TOUR_KEY,
        show_progress=True,
        animate=True,
        overlay_opacity=0.55,
        one_time_tour=True,
    )

    # Cada sesión de Streamlit solo la lanza una vez. Si ya se completó lo decide el componente mediante el navegador
    # localStorage , para que los reruns de la página o las interacciones normales con controles no vuelvan a mostrar la guía.
    auto_start_key = f"{ONBOARDING_TOUR_KEY}-auto-started"
    if not st.session_state.get(auto_start_key, False):
        st.session_state[auto_start_key] = True
        tour.start()


def _render_generation_logs(task_id):
    """Renderiza la instantánea de logs de la tarea en segundo plano, sin acceder al estado de sesión de Streamlit desde el hilo de trabajo."""
    if config.ui.get("hide_log", False):
        return

    log_records = webui_task.get_task_logs(task_id)
    if not log_records:
        return

    st.code("\n".join(log_records))


def _current_generation_task_id():
    """URL Solo guarda el ID de tarea local, para que refrescar/reconectar recupere la tarea original, sin guardar configuración ni credenciales."""
    task_id = st.session_state.get("current_generation_task_id", "")
    if not task_id:
        try:
            task_id = str(UUID(st.query_params.get("generation_task", "")))
        except (ValueError, TypeError, AttributeError):
            return ""
        st.session_state["current_generation_task_id"] = task_id
    return task_id


def _generation_is_busy(task_id):
    if not task_id:
        return False
    if webui_task.is_generation_active(task_id):
        return True
    try:
        task = sm.state.get_task(task_id)
    except Exception:
        # Ante una excepción de consulta no se puede tomar el «estado desconocido» como completado y permitir envíos duplicados.
        return True
    return _normalize_task_state((task or {}).get("state")) == const.TASK_STATE_PROCESSING


def _render_generation_activity(task):
    activity = task.get("generation_activity") or {}
    phase = activity.get("phase", "")
    labels = {
        "queued": "Generation Waiting In Queue",
        "preparing": "Generation Preparing",
        "script": "Generation Writing Script",
        "terms": "Generation Preparing Terms",
        "audio": "Generation Preparing Audio",
        "subtitle": "Generation Preparing Subtitles",
        "materials": "Generation Preparing Materials",
        "submitting": "Generation Submitting",
        "checking": "Generation Checking",
        "remote_queued": "Generation Remote Queued",
        "generating": "Generation Remote Running",
        "generated": "Generation Remote Completed",
        "downloading": "Generation Downloading",
        "verifying": "Generation Verifying",
        "rendering": "Generation Rendering",
    }
    st.info(tr(labels.get(phase, "Generating Video")))
    now = time.time()
    started = (task.get("generation_submitted_at") or task.get("generation_started_at")
               or activity.get("started_at") or now)
    elapsed = max(0, int(now - float(started)))
    st.caption(tr("Generation Elapsed").format(time=f"{elapsed // 60:02d}:{elapsed % 60:02d}"))
    if phase in {"submitting", "remote_queued", "generating", "checking"}:
        st.caption(tr("Generation Long Wait Hint"))
        if activity.get("retrying"):
            st.warning(tr("Generation Poll Retry Hint"))
        if activity.get("confirmed_at"):
            age = max(0, int(now - float(activity["confirmed_at"])))
            st.caption(tr("Generation Last Checked").format(seconds=age))
    if phase == "downloading" and activity.get("downloaded_bytes"):
        downloaded = activity["downloaded_bytes"] / (1024 * 1024)
        st.caption(tr("Generation Downloaded").format(size=f"{downloaded:.1f}"))
    if activity.get("remote_id"):
        st.caption(tr("Generation Remote Task").format(task_id=activity["remote_id"]))


def _render_generation_task_snapshot(task_id, task):
    """Renderiza el progreso, el motivo de fallo o el video final según la instantánea del almacén de estado."""
    if not task:
        st.info(tr("Generating Video"))
        _render_generation_logs(task_id)
        return

    state = _normalize_task_state(task.get("state"))
    progress = max(0, min(100, int(task.get("progress", 0) or 0)))
    if state == const.TASK_STATE_PROCESSING:
        _render_generation_activity(task)
        st.progress(
            progress,
            text=f"{tr('Task Progress')}: {progress}%",
        )
        _render_generation_logs(task_id)
        return

    video_files = task.get("videos") or []
    if state == const.TASK_STATE_FAILED:
        error = str(task.get("error") or "").strip()
        error_key = task.get("error_key")
        error_params = task.get("error_params") or {}
        if (
            isinstance(error_key, str)
            and error_key.startswith("codex.")
            and isinstance(error_params, Mapping)
        ):
            error = tr(error_key).format(**error_params)
        message = tr("Video Generation Failed")
        st.error(f"{message}: {error}" if error else message)
        activity = task.get("generation_activity") or {}
        if activity.get("remote_id"):
            st.warning(tr("Generation Existing Task Hint").format(task_id=activity["remote_id"]))
        if not video_files:
            _render_generation_logs(task_id)
            return
    elif state != const.TASK_STATE_COMPLETE or not video_files:
        st.error(tr("Video Generation Failed"))
        _render_generation_logs(task_id)
        return

    if state == const.TASK_STATE_COMPLETE:
        st.success(tr("Video Generation Completed"))
    for warning in task.get("warnings") or []:
        if isinstance(warning, Mapping) and warning.get("code") == "batch_materials_reused":
            st.warning(
                tr("Batch Material Reuse Warning").format(
                    index=warning.get("video_index", ""), count=warning.get("count", 0)
                )
            )
        elif isinstance(warning, Mapping) and warning.get("code") == "sonilo_bgm_failed":
            st.warning(
                tr("Sonilo BGM Fallback Warning").format(
                    index=warning.get("video_index", "")
                )
            )
        elif (
            isinstance(warning, Mapping)
            and warning.get("code") == "elevenlabs_bgm_failed"
        ):
            st.warning(
                tr("ElevenLabs BGM Fallback Warning").format(
                    index=warning.get("video_index", "")
                )
            )
        else:
            st.warning(str(warning))

    available_videos = [
        (index, url)
        for index, url in enumerate(video_files)
        if os.path.isfile(url)
    ]
    for index, url in enumerate(video_files):
        if os.path.isfile(url):
            continue
        logger.warning(
            f"generated video is unavailable: "
            f"task_id={task_id}, video_file={url}"
        )

    try:
        if not available_videos:
            st.warning(tr("Generated Video Files Unavailable"))
        else:
            player_cols = st.columns(len(available_videos) * 2 + 1)
            for player_index, (video_index, url) in enumerate(available_videos):
                with player_cols[player_index * 2 + 1]:
                    st.video(url)
                    download_label = tr("Download Video")
                    if len(video_files) > 1:
                        download_label = f"{download_label} {video_index + 1}"
                    download_name = _build_video_download_name(
                        task.get("video_subject"),
                        video_index + 1,
                        len(video_files),
                    )
                    with open(url, "rb") as video_file:
                        st.download_button(
                            download_label,
                            data=video_file,
                            file_name=download_name,
                            mime=mimetypes.guess_type(url)[0] or "video/mp4",
                            key=f"download_generated_video_{task_id}_{video_index}",
                            icon=":material/download:",
                            on_click="ignore",
                            use_container_width=True,
                        )
    except Exception as exc:
        logger.exception(
            f"failed to render generated video preview: task_id={task_id}, "
            f"video_files={video_files}, error={exc}"
        )

    _render_generation_logs(task_id)
    if (
        state == const.TASK_STATE_COMPLETE
        and st.session_state.get("handled_generation_task_id") != task_id
    ):
        # Fragment puede renderizar varias veces la misma tarea completada. Esté o no activada la apertura automática del directorio,
        # cada tarea solo procesa una vez el evento de finalización, para no abrir repetidamente el explorador de archivos ni escribir logs duplicados.
        st.session_state["handled_generation_task_id"] = task_id
        if config.ui.get("open_task_folder_on_completion", True):
            open_task_folder(task_id)
        logger.info(f"{tr('Video Generation Completed')}: task_id={task_id}")


@st.fragment(run_every=webui_task.TASK_LOG_REFRESH_INTERVAL_SECONDS)
def _render_running_generation_task(task_id):
    """Solo sondea mientras la tarea está en ejecución; al terminar vuelve al resultado estático, deteniendo refrescos periódicos innecesarios."""
    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.exception(
            f"failed to query WebUI generation task: task_id={task_id}, error={exc}"
        )
        st.warning(tr("Generation Status Unavailable"))
        return

    if not task and not webui_task.is_generation_active(task_id):
        st.warning(tr("Generation Task Unavailable"))
        return
    state = _normalize_task_state((task or {}).get("state"))
    if state in {const.TASK_STATE_COMPLETE, const.TASK_STATE_FAILED}:
        _remove_active_generation_task(task_id)
        # El script de la página completa ya no tiene lógica de generación lenta, así que puede hacer rerun con seguridad y pasar el resultado a renderizado
        # estático. Así, al terminar la tarea el navegador no conserva para siempre un Fragment que sondea cada dos segundos.
        st.rerun(scope="app")

    _render_generation_task_snapshot(task_id, task)
    if not webui_task.is_generation_active(task_id):
        # El estado persistente puede pertenecer a antes del reinicio o a otro proceso de servicio, y no se puede afirmar que la tarea remota se detuvo.
        # Solo permite dejar de seguirla en la página actual, sin cancelar la tarea ni reescribir su estado real.
        st.warning(tr("Generation Untracked Worker Hint"))
        if st.button(tr("Stop Tracking Generation"), key=f"stop_tracking_generation_{task_id}"):
            _remove_active_generation_task(task_id)
            st.session_state["current_generation_task_id"] = ""
            st.query_params.pop("generation_task", None)
            st.rerun(scope="app")


def _render_current_generation_task():
    """Restaura debajo del botón de generar la UI consultable de la última tarea enviada desde la página actual."""
    task_id = _current_generation_task_id()
    if not task_id:
        return

    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.exception(
            f"failed to query current WebUI task: task_id={task_id}, error={exc}"
        )
        # Aunque falle la primera lectura, hay que registrar el Fragment, para que continúe automáticamente cuando se recupere el backend de estado.
        _render_running_generation_task(task_id)
        return

    if not task and not webui_task.is_generation_active(task_id):
        st.warning(tr("Generation Task Unavailable"))
        return
    state = _normalize_task_state((task or {}).get("state"))
    if state in {const.TASK_STATE_COMPLETE, const.TASK_STATE_FAILED}:
        _remove_active_generation_task(task_id)
        _render_generation_task_snapshot(task_id, task)
        return

    _render_running_generation_task(task_id)


def get_llm_provider_tips(provider_id, **kwargs):
    # LLM provider Los textos explicativos usan de forma unificada la regla `llm_provider_tips.<provider_id>`.
    # Así, al añadir un provider solo hay que completar el texto en el locale; sin texto no se muestra el bloque de aviso,
    # evitando seguir acumulando en Main.py muchas explicaciones codificadas en dos idiomas.
    provider = get_llm_provider(provider_id)
    if provider is None:
        return ""

    # Provider Las instrucciones de configuración mantienen por ahora dos plantillas estándar, en español e inglés; los demás idiomas de interfaz
    # usan el inglés de forma unificada, para no copiar el inglés en el locale y perder la sincronía a largo plazo. Cuando un idioma complete
    # la traducción completa, se añadirá aquí a su ámbito de mantenimiento independiente.
    ui_language = st.session_state.get("ui_language", "en")
    tips_language = ui_language if ui_language in {"es", "en"} else "en"
    tips = (
        locales.get(tips_language, {}).get("Translation", {}).get(provider.tips_key, "")
    )
    if not tips:
        return tips

    service_endpoint = provider.preferred_service_endpoint(
        prefer_international=True
    )
    api_key_url = (
        service_endpoint.api_key_url
        if service_endpoint
        else provider.effective_api_key_url()
    )
    format_context = {
        "api_key_url": api_key_url,
        "default_model": provider.default_model,
        "default_base_url": (
            service_endpoint.base_url
            if service_endpoint
            else provider.effective_default_base_url
        ),
        "model_docs_url": (
            service_endpoint.model_docs_url
            if service_endpoint and service_endpoint.model_docs_url
            else provider.effective_model_docs_url(
                prefer_international=True
            )
        ),
        **{
            f"default_{field.config_suffix}": field.default_value
            for field in provider.extra_fields
        },
        **kwargs,
    }
    try:
        return tips.format(**format_context)
    except Exception as e:
        logger.warning(f"format llm provider tips failed: {provider_id}, {e}")
        return tips


def format_llm_connection_error(provider_id, base_url, error):
    """Añade sugerencias de revisión de configuración a los errores de autenticación claramente localizables, conservando la respuesta original."""
    error_text = str(error or "").strip()
    if provider_id in {"openai_codex", "openai_chatgpt"}:
        if provider_id == "openai_chatgpt":
            from app.services.chatgpt_oauth import ERROR_MESSAGES
            message_prefix = "chatgpt.oauth"
        else:
            from app.services.codex_oauth import ERROR_MESSAGES
            message_prefix = "codex.oauth"

        for key, message in ERROR_MESSAGES.items():
            if error_text == message:
                return tr(f"{message_prefix}.{key}")
        status_match = re.fullmatch(
            r"OpenAI - (?:Codex|ChatGPT \(Native OAuth\)) returned HTTP (\d+)\. Check the selected model and your account access\.",
            error_text,
        )
        if status_match:
            return tr(f"{message_prefix}.http_error").format(status=status_match[1])
    normalized_error = error_text.lower()
    authentication_markers = (
        "401",
        "authentication",
        "invalid api key",
        "invalid_api_key",
        "unauthorized",
    )
    provider = get_llm_provider(provider_id)
    if provider is None or not provider.service_endpoints or not any(
        marker in normalized_error for marker in authentication_markers
    ):
        return error_text

    message = tr_optional(
        provider.authentication_error_key,
        fallback_language="en",
    )
    if not message:
        return error_text
    return message.format(base_url=base_url or "-", error=error_text)


def get_llm_provider_label(provider):
    return tr_optional(provider.label_key) or provider.default_label


def get_tts_provider_tips(provider_id):
    # TTS Las instrucciones de configuración siguen la misma estrategia de mantenimiento que el LLM Provider: solo se mantienen español e inglés,
    # y los demás idiomas de interfaz vuelven al inglés, para no perder la sincronía tras copiarlos.
    ui_language = st.session_state.get("ui_language", "en")
    tips_language = ui_language if ui_language in {"es", "en"} else "en"
    return (
        locales.get(tips_language, {})
        .get("Translation", {})
        .get(f"tts_provider_tips.{provider_id}", "")
    )


def localized_widget_key(name, *parts):
    # Algunos selectbox de Streamlit recuerdan la selección con una key estable, pero el texto mostrado viene del locale.
    # Al cambiar de idioma se incluye también el idioma en la key, forzando a reconstruir el control para que la opción elegida no siga mostrándose en el idioma anterior.
    language = st.session_state.get("ui_language", config.ui.get("language", ""))
    suffix_parts = [name, language, *[str(part) for part in parts if part]]
    return "_".join(suffix_parts)


def stable_selectbox(label, options, default_value, key, format_func=None, **kwargs):
    # Streamlit 1.59 es más sensible a la reutilización de estado del selectbox: si el control no tiene key fija,
    # o las opciones reales son solo índices temporales, tras un rerun es fácil que se sobrescriba con un index recalculado,
    # y el usuario ve que su primera elección no surte efecto y tiene que elegir otra vez. Este helper usa de forma unificada valores de negocio estables
    # como opciones reales y guarda ese valor en session_state; el texto mostrado solo se obtiene mediante format_func
    # la conversión, para que los textos traducidos, el orden de opciones o los cambios de configuración aguas arriba no afecten al estado de selección.
    options = list(options)
    if not options:
        raise ValueError(f"selectbox options cannot be empty: {key}")

    if default_value not in options:
        default_value = options[0]

    widget_key = localized_widget_key(key)
    selected_value = st.session_state.get(widget_key)
    accepts_custom_value = bool(kwargs.get("accept_new_options"))
    has_valid_custom_value = (
        accepts_custom_value
        and isinstance(selected_value, str)
        and bool(selected_value.strip())
    )
    if selected_value not in options and not has_valid_custom_value:
        # Si cambian las opciones aguas arriba (por ejemplo, la lista de voces cambia al cambiar de TTS provider),
        # el valor antiguo ya no es válido. Se inicializa session_state directamente antes de crear el control, y después solo se deja que key
        # gestione el estado, sin pasar también index. Así se evita que Streamlit, en el rerun,
        # sobrescriba con el index recalculado el valor que el usuario acaba de elegir, haciendo que la primera elección no surta efecto.
        st.session_state[widget_key] = default_value

    if format_func is None:
        format_func = str

    return st.selectbox(
        label,
        options=options,
        format_func=format_func,
        key=widget_key,
        **kwargs,
    )


# Streamlit El selectbox nativo aún no admite HTML optgroup. Aquí se usa el que trae la 1.59
# Components v2 para envolver el <select>/<optgroup> nativo, sin introducir dependencias de frontend, conservando la navegación por teclado,
# la semántica de accesibilidad y la experiencia de selección móvil nativas del navegador. El componente solo pasa valores de negocio fijos y textos traducidos,
# sin aceptar HTML arbitrario, evitando desde el límite que el contenido de configuración entre en innerHTML.
_GROUPED_SELECT_COMPONENT = st.components.v2.component(
    "mpt_grouped_select",
    html="""
        <div class="mpt-grouped-select">
            <div class="mpt-grouped-select__label-row">
                <label class="mpt-grouped-select__label"></label>
                <button class="mpt-grouped-select__settings" type="button"></button>
            </div>
            <div class="mpt-grouped-select__control">
                <select></select>
            </div>
        </div>
    """,
    css="""
        .mpt-grouped-select {
            width: 100%;
            color: var(--st-text-color);
            font-family: var(--st-font);
        }

        .mpt-grouped-select__label-row {
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: 0.45rem;
            margin-bottom: 0.35rem;
        }

        .mpt-grouped-select__label {
            font-size: 0.875rem;
            line-height: 1.25rem;
        }

        .mpt-grouped-select__settings {
            padding: 0;
            border: 0;
            background: transparent;
            color: var(--st-link-color);
            font: inherit;
            font-size: 0.8rem;
            line-height: 1.25rem;
            cursor: pointer;
        }

        .mpt-grouped-select__settings:hover {
            text-decoration: underline;
            text-underline-offset: 0.15rem;
        }

        .mpt-grouped-select__settings:focus-visible {
            border-radius: 0.2rem;
            outline: 2px solid var(--st-primary-color);
            outline-offset: 2px;
        }

        .mpt-grouped-select__control {
            position: relative;
        }

        .mpt-grouped-select__control::after {
            position: absolute;
            top: 50%;
            right: 1rem;
            width: 0.55rem;
            height: 0.55rem;
            border-right: 2px solid currentColor;
            border-bottom: 2px solid currentColor;
            content: "";
            pointer-events: none;
            transform: translateY(-70%) rotate(45deg);
        }

        .mpt-grouped-select select {
            width: 100%;
            min-height: 2.5rem;
            padding: 0.45rem 2.75rem 0.45rem 0.75rem;
            border: 1px solid color-mix(in srgb, currentColor 20%, transparent);
            border-radius: 0.5rem;
            outline: none;
            appearance: none;
            background: var(--st-secondary-background-color);
            color: inherit;
            font: inherit;
            cursor: pointer;
        }

        .mpt-grouped-select select:hover {
            border-color: color-mix(in srgb, currentColor 36%, transparent);
        }

        .mpt-grouped-select select:focus-visible {
            border-color: var(--st-primary-color);
            box-shadow: 0 0 0 1px var(--st-primary-color);
        }
    """,
    js="""
        export default function(component) {
            const { data, parentElement, setTriggerValue } = component;
            const label = parentElement.querySelector("label");
            const settings = parentElement.querySelector(".mpt-grouped-select__settings");
            const select = parentElement.querySelector("select");

            label.textContent = data.label;
            settings.textContent = data.settingsLabel;
            settings.hidden = !data.settingsLabel;
            select.id = data.controlId;
            label.htmlFor = data.controlId;
            select.setAttribute("aria-label", data.label);
            select.replaceChildren();

            for (const groupData of data.groups) {
                const group = document.createElement("optgroup");
                group.label = groupData.label;
                for (const optionData of groupData.options) {
                    const option = document.createElement("option");
                    option.value = optionData.value;
                    option.textContent = optionData.label;
                    group.appendChild(option);
                }
                select.appendChild(group);
            }

            select.value = data.value;
            const handleChange = () => {
                setTriggerValue("selected", select.value);
            };
            const handleSettings = () => {
                setTriggerValue("settings", true);
            };
            select.addEventListener("change", handleChange);
            settings.addEventListener("click", handleSettings);

            return () => {
                select.removeEventListener("change", handleChange);
                settings.removeEventListener("click", handleSettings);
            };
        }
    """,
)


def grouped_selectbox(
    label,
    groups,
    default_value,
    key,
    format_func=None,
    settings_label="",
    on_settings=None,
):
    """Renderiza un único desplegable con títulos de grupo no seleccionables y devuelve un valor de negocio estable."""
    if format_func is None:
        format_func = str

    normalized_groups = []
    valid_values = []
    for group_label, options in groups:
        normalized_options = []
        for option in options:
            valid_values.append(option)
            normalized_options.append(
                {"value": option, "label": str(format_func(option))}
            )
        if normalized_options:
            normalized_groups.append(
                {"label": str(group_label), "options": normalized_options}
            )

    if not valid_values:
        raise ValueError(f"grouped selectbox options cannot be empty: {key}")
    if len(set(valid_values)) != len(valid_values):
        raise ValueError(f"grouped selectbox options must be unique: {key}")
    if default_value not in valid_values:
        default_value = valid_values[0]

    # La selección de negocio se guarda en la misma session key que el selectbox antiguo, así que la restauración de preajustes y
    # la lógica de cambio de idioma no tienen que bifurcarse; el componente usa su propia key independiente, para no chocar con el estado de negocio.
    widget_key = localized_widget_key(key)
    if widget_key not in st.session_state:
        st.session_state[widget_key] = default_value
    selected_value = st.session_state[widget_key]
    if selected_value not in valid_values:
        selected_value = default_value
        st.session_state[widget_key] = selected_value

    result = _GROUPED_SELECT_COMPONENT(
        key=f"{widget_key}_component",
        data={
            "label": label,
            "settingsLabel": settings_label,
            # Asocia explícitamente el label visible con el select nativo. La key del componente la forman un nombre de negocio fijo y
            # el código de idioma, única en la página, lo que facilita enfocar el control al hacer clic en la etiqueta,
            # sin introducir IDs aleatorios que reconstruirían el estado del frontend en cada rerun.
            "controlId": f"{widget_key}_control",
            "value": selected_value,
            "groups": normalized_groups,
        },
        on_selected_change=lambda: None,
        on_settings_change=on_settings or (lambda: None),
    )
    changed_value = getattr(result, "selected", None)
    if changed_value in valid_values and changed_value != selected_value:
        st.session_state[widget_key] = changed_value
        # Components v2 al devolver el evento en la ronda actual del script, la data que se pasa al frontend en esta ronda sigue siendo
        # el valor antiguo de antes del evento. Se hace un rerun automático inmediato, para que el componente y los controles que dependen de video_source
        # reciban a la vez el valor nuevo; si no, el desplegable quedaría sobrescrito brevemente por la data antigua y el usuario tendría que elegir otra vez.
        st.rerun()

    return selected_value


def sync_script_order_concat_mode():
    """Con la correspondencia en el orden del texto activada se usa siempre la concatenación secuencial, y al desactivarla se restaura la elección original."""
    widget_key = localized_widget_key("video_concat_mode_select")
    previous_key = "video_concat_mode_before_script_order_match"
    match_script_order = bool(st.session_state.get("match_materials_to_script", False))

    if match_script_order:
        current_mode = st.session_state.get(widget_key, VideoConcatMode.random.value)
        if current_mode != VideoConcatMode.sequential.value:
            st.session_state[previous_key] = current_mode
        st.session_state[widget_key] = VideoConcatMode.sequential.value
        return

    previous_mode = st.session_state.pop(previous_key, None)
    if previous_mode in {
        VideoConcatMode.sequential.value,
        VideoConcatMode.random.value,
    }:
        st.session_state[widget_key] = previous_mode


def reset_script_system_prompt():
    """Restaura el prompt de sistema de los ajustes de guion avanzados al contenido por defecto de la versión actual."""
    st.session_state["custom_system_prompt"] = llm.DEFAULT_SCRIPT_SYSTEM_PROMPT


def reset_subtitle_settings():
    """Restaura los valores por defecto en los controles de subtítulos de la WebUI y en la configuración persistente."""
    defaults = DEFAULT_SUBTITLE_SETTINGS
    st.session_state["subtitle_enabled_checkbox"] = defaults["subtitle_enabled"]
    _set_stable_widget_value("font_name_select", defaults["font_name"])
    _set_stable_widget_value("subtitle_position_select", defaults["subtitle_position"])
    _set_stable_widget_value(
        "subtitle_display_mode_select", defaults["subtitle_display_mode"]
    )
    _set_stable_widget_value(
        "subtitle_animation_select", defaults["subtitle_animation"]
    )
    st.session_state["custom_position_input"] = str(defaults["custom_position"])
    st.session_state["font_color_picker"] = defaults["text_fore_color"]
    st.session_state["font_size_slider"] = defaults["font_size"]
    st.session_state["stroke_color_picker"] = defaults["stroke_color"]
    st.session_state["stroke_width_slider"] = defaults["stroke_width"]
    st.session_state["subtitle_background_enabled_checkbox"] = defaults[
        "subtitle_background_enabled"
    ]
    st.session_state["subtitle_background_color_picker"] = defaults[
        "subtitle_background_color"
    ]
    st.session_state["rounded_subtitle_background_checkbox"] = defaults[
        "rounded_subtitle_background"
    ]

    # Sincroniza las opciones de UI persistentes, para que tras restaurar y refrescar la página se mantengan los ajustes por defecto.
    for key in (
        "subtitle_enabled",
        "font_name",
        "subtitle_position",
        "subtitle_display_mode",
        "subtitle_animation",
        "custom_position",
        "text_fore_color",
        "font_size",
        "stroke_color",
        "stroke_width",
        "subtitle_background_enabled",
        "subtitle_background_color",
        "rounded_subtitle_background",
    ):
        if key in defaults:
            _set_runtime_config("ui", key, defaults[key])


@st.dialog(tr("Final Prompt Preview"), width="large")
def render_script_prompt_preview(prompt):
    """Muestra el prompt completo de generación de guion que se enviará al modelo grande."""
    st.code(prompt, language="markdown", wrap_lines=True)


def stable_segmented_control(
    label, options, default_value, key, format_func=None, **kwargs
):
    """Crea un control segmentado de selección única con valores de negocio estables, para que el estado no se sobrescriba con el texto mostrado al cambiar de idioma."""
    options = list(options)
    if not options:
        raise ValueError(f"segmented control options cannot be empty: {key}")

    if default_value not in options:
        default_value = options[0]

    widget_key = localized_widget_key(key)
    if st.session_state.get(widget_key) not in options:
        st.session_state[widget_key] = default_value

    return st.segmented_control(
        label,
        options=options,
        selection_mode="single",
        required=True,
        format_func=format_func or str,
        key=widget_key,
        **kwargs,
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_groq_model_ids(api_key: str, base_url: str) -> list[str]:
    if not api_key:
        return []

    normalized_base_url = (
        (base_url or "https://api.groq.com/openai/v1").strip().rstrip("/")
    )
    models_url = f"{normalized_base_url}/models"

    try:
        response = trace_http_call(
            "GET", requests.get,
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])

        model_ids = []
        for item in data:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    model_ids.append(model_id.strip())

        return sorted(set(model_ids))
    except Exception as e:
        logger.warning(f"failed to fetch groq models: {e}")
        return []


def _get_material_api_keys(config_key):
    """Convierte de forma unificada las API Key de material de la configuración en cadenas editables en la WebUI."""
    api_keys = config.snapshot_config_with_pending(config.app).get(config_key, [])
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    return ", ".join(api_keys)


def _save_material_api_keys(config_key, value):
    """Guarda las API Key de material separadas por comas, permitiendo al usuario vaciar explícitamente la configuración antigua."""
    normalized_value = value.replace(" ", "")
    _set_runtime_config(
        "app",
        config_key,
        normalized_value.split(",") if normalized_value else [],
    )


def _format_file_size(size_bytes):
    """Formatea un número de bytes como un texto de capacidad compacto adecuado para la página de ajustes."""
    size = float(max(0, size_bytes))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


@st.cache_data(ttl=30, show_spinner=False)
def _get_video_cache_stats_data(max_age_days=None):
    """
    Caché de ciclo corto para las estadísticas de directorio, para que las interacciones normales en el diálogo de ajustes no escaneen repetidamente muchos archivos.

    La clave de caché incluye los días de limpieza, así que cambiar de rango solo escanea una vez por rango; al refrescar o limpiar manualmente
    se vacía explícitamente, y la caché de hasta 30 segundos no afecta al segundo escaneo del borrado real.

    Aquí solo se cachea un dict puro, no instancias de VideoCacheStats: st.cache_data necesita pickle
    serializar el valor devuelto, y pickle, al guardar clases personalizadas, las referencia por «nombre de módulo + nombre de clase» y comprueba que la clase resuelta
    sea el mismo objeto que la clase de la instancia. Cuando el vigilante de código de Streamlit detecta cambios en los archivos fuente locales
    (incluidos eventos esporádicos en Windows causados por antivirus o indexadores), vacía de sys.modules los
    módulos vigilados y los vuelve a importar; el diálogo de ajustes que siga abierto en ese momento (Dialog hereda el comportamiento de fragment,
    y sus interacciones internas no reconstruyen las referencias de nivel superior de Main.py) aún tiene la clase del módulo antiguo, y pickle lanzaría
    PicklingError: it's not the same object as ..., que Streamlit envuelve como
    UnserializableReturnValueError(defecto conocido correspondiente al issue oficial #14593 de Streamlit).
    Un dict puro se serializa por valor, sin referencias a clases, así que no le afecta la reimportación de módulos.
    """
    stats = cache_manager.get_video_cache_stats(max_age_days=max_age_days)
    # Los nombres de campo coinciden exactamente con VideoCacheStats, para que _get_video_cache_stats, más abajo, los expanda directamente
    # y los restaure; si en el futuro VideoCacheStats añade o quita campos, hay que mantener aquí el mapeo de campos.
    return {
        "file_count": stats.file_count,
        "total_size": stats.total_size,
        "oldest_mtime": stats.oldest_mtime,
        "newest_mtime": stats.newest_mtime,
    }


def _get_video_cache_stats(max_age_days=None) -> cache_manager.VideoCacheStats:
    """Restaura los datos puros a VideoCacheStats fuera del límite de la caché; el acceso a atributos del llamador no cambia."""

    # Al restaurar se usa el módulo cache_manager vigente; aunque el módulo se reimporte, solo se reconstruye
    # un dataclass ligero, sin volver a activar la comprobación de «identidad de referencia de clase» de pickle.
    return cache_manager.VideoCacheStats(
        **_get_video_cache_stats_data(max_age_days=max_age_days)
    )


def _render_cache_management_settings(panel):
    """Renderiza las estadísticas, la vista previa y las operaciones de limpieza segura de la caché de material de video en línea por defecto."""
    with panel:
        cleanup_message = st.session_state.pop("video_cache_cleanup_message", None)
        if cleanup_message:
            message_type, message = cleanup_message
            if message_type == "success":
                st.success(message)
            else:
                st.warning(message)

        st.caption(tr("Video Cache Directory"))
        st.code(cache_manager.video_cache_dir(), language="text")

        total_stats = _get_video_cache_stats()
        metric_count, metric_size, metric_oldest = st.columns(3)
        metric_count.metric(tr("Cache File Count"), total_stats.file_count)
        metric_size.metric(
            tr("Cache Total Size"), _format_file_size(total_stats.total_size)
        )
        oldest_text = (
            datetime.fromtimestamp(total_stats.oldest_mtime).strftime("%Y-%m-%d")
            if total_stats.oldest_mtime is not None
            else "-"
        )
        metric_oldest.metric(tr("Oldest Cache Date"), oldest_text)

        st.caption(tr("Video Cache Management Help"))
        cleanup_options = (30, 7, 90, None)
        cleanup_labels = {
            30: tr("Cache Older Than 30 Days"),
            7: tr("Cache Older Than 7 Days"),
            90: tr("Cache Older Than 90 Days"),
            None: tr("All Video Cache"),
        }
        max_age_days = st.selectbox(
            tr("Cache Cleanup Range"),
            options=cleanup_options,
            format_func=lambda value: cleanup_labels[value],
            key="video_cache_cleanup_range",
        )
        cleanup_preview = _get_video_cache_stats(max_age_days=max_age_days)
        st.info(
            tr("Cache Cleanup Preview").format(
                count=cleanup_preview.file_count,
                size=_format_file_size(cleanup_preview.total_size),
            )
        )

        confirm_nonce = st.session_state.get("video_cache_cleanup_confirm_nonce", 0)
        confirmed = st.checkbox(
            tr("Confirm Cache Cleanup"),
            key=f"video_cache_cleanup_confirm_{confirm_nonce}",
        )
        refresh_col, open_col, cleanup_col = st.columns(3)
        if refresh_col.button(
            tr("Refresh Cache Stats"),
            key="refresh_video_cache_stats",
            use_container_width=True,
            icon=":material/refresh:",
        ):
            _get_video_cache_stats_data.clear()
            st.rerun(scope="fragment")

        if open_col.button(
            tr("Open Cache Directory"),
            key="open_video_cache_directory",
            use_container_width=True,
            icon=":material/folder_open:",
        ):
            webbrowser.open(Path(cache_manager.video_cache_dir()).as_uri())

        cleanup_disabled = not confirmed or cleanup_preview.file_count == 0
        if cleanup_col.button(
            tr("Clean Cache Now"),
            key="clean_video_cache_now",
            type="primary",
            disabled=cleanup_disabled,
            use_container_width=True,
            icon=":material/delete_sweep:",
        ):
            result = cache_manager.clean_video_cache(max_age_days=max_age_days)
            message_key = (
                "Cache Cleanup Completed With Failures"
                if result.failed_count
                else "Cache Cleanup Completed"
            )
            st.session_state["video_cache_cleanup_message"] = (
                "warning" if result.failed_count else "success",
                tr(message_key).format(
                    count=result.deleted_count,
                    size=_format_file_size(result.deleted_size),
                    failed=result.failed_count,
                ),
            )
            # Streamlit no permite modificar el session_state del mismo nombre tras instanciar el control. Incrementando
            # nonce se hace que el siguiente rerun del fragment cree un control nuevo sin marcar, para que tras la limpieza no
            # se conserve el estado de confirmación peligrosa.
            st.session_state["video_cache_cleanup_confirm_nonce"] = confirm_nonce + 1
            _get_video_cache_stats_data.clear()
            st.rerun(scope="fragment")


# -----------------------------------------------------------------------------
# Exportación e importación de preajustes y copia de seguridad de claves
# -----------------------------------------------------------------------------


def _is_credential_config_key(key):
    """Determina si el nombre de un ajuste representa una credencial."""
    return str(key).endswith(CREDENTIAL_KEY_SUFFIXES)


def _is_backup_config_key(section_name, key):
    """Tanto las credenciales como sus ajustes asociados forman parte de la copia de seguridad de claves."""
    if _is_credential_config_key(key):
        return True
    if key in CREDENTIAL_COMPANION_KEYS.get(section_name, ()):
        return True
    return key in NON_LLM_COMPANION_KEYS.get(section_name, ())


def _credential_widget_state_keys(section_name, key):
    """
    Devuelve todas las key de control de Streamlit correspondientes a un ajuste de credencial.

    todos los campos de contraseña llevan key, y en Streamlit el valor de session_state tiene prioridad sobre el parámetro value
    del control. Tras restaurar la copia deben borrarse estos estados residuales, o la página seguiría mostrando la clave antigua,
    y en el siguiente rerun volvería a escribir el valor antiguo en la configuración, como si la restauración no hubiera surtido efecto. Cuando varios paneles
    comparten la misma clave, cada uno tiene su estado de control, así que se devuelven la key por defecto y todos los alias.
    """
    if section_name == "app":
        default_widget_key = f"{key}_input"
    else:
        default_widget_key = f"{section_name}_{key}_input"
    return (
        default_widget_key,
        *CREDENTIAL_WIDGET_STATE_ALIASES.get((section_name, key), ()),
    )


def _normalize_backup_value(value):
    """Normaliza los valores de la copia, descartando cadenas y listas vacías, para no sobrescribir con configuración vacía al restaurar."""
    if isinstance(value, list):
        items = [
            str(item).strip()
            for item in value
            if isinstance(item, (str, int, float)) and str(item).strip()
        ]
        return items or None
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value).strip()
        return text or None
    return None


def _collect_key_backup(config_sections):
    """Reúne de las secciones de configuración en ejecución todas las claves rellenadas y sus ajustes asociados."""
    backup = {}
    for section_name, section in config_sections.items():
        if section_name in KEY_BACKUP_EXCLUDED_SECTIONS:
            continue
        entries = {}
        for key, value in section.items():
            if not _is_backup_config_key(section_name, key):
                continue
            normalized_value = _normalize_backup_value(value)
            if normalized_value is not None:
                entries[key] = normalized_value
        if entries:
            backup[section_name] = entries
    return backup


def _count_backup_keys(backup):
    """Cuenta los ajustes de la copia, para avisos en la interfaz y para deshabilitar exportaciones vacías."""
    return sum(len(entries) for entries in backup.values())


def _build_key_backup_payload(config_sections, app_version):
    """Construye el contenido del archivo de copia de seguridad de claves."""
    return {
        "schema": KEY_BACKUP_SCHEMA,
        "version": KEY_BACKUP_VERSION,
        "app_version": str(app_version),
        "keys": _collect_key_backup(config_sections),
    }


def _load_transfer_payload(raw_bytes, schema, version):
    """
    Analiza el archivo exportado y verifica que proviene realmente de la misma versión de esta función.

    El usuario puede subir cualquier JSON. Aquí solo se aceptan archivos que declaran el schema y la versión correctos, para que el aviso
    de error se quede en el punto de importación, en lugar de escribir contenido no reconocible en la configuración o el estado de los controles.
    Windows Los editores pueden guardar JSON con BOM, así que se decodifica como utf-8-sig.
    """
    if len(raw_bytes) > MAX_SETTINGS_TRANSFER_BYTES:
        raise ValueError("settings import exceeds the 2 MB limit")
    payload = json.loads(raw_bytes.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("exported file must contain a JSON object")
    if payload.get("schema") != schema:
        raise ValueError(f"unexpected schema: {payload.get('schema')!r}")
    if payload.get("version") != version:
        raise ValueError(f"unsupported version: {payload.get('version')!r}")
    return payload


def _parse_key_backup(raw_bytes, config_sections):
    """
    Analiza el archivo de copia de claves, conservando solo las secciones y ajustes que reconoce la versión actual.

    El archivo de copia puede editarse a mano o venir de una versión más nueva. Las secciones desconocidas o los ajustes que no son claves se
    ignoran siempre, para no reescribir mediante la importación configuración no relacionada con credenciales.
    """
    payload = _load_transfer_payload(raw_bytes, KEY_BACKUP_SCHEMA, KEY_BACKUP_VERSION)
    keys = payload.get("keys")
    if not isinstance(keys, dict):
        raise ValueError("key backup file has no keys object")

    restored = {}
    for section_name, entries in keys.items():
        if section_name not in config_sections:
            continue
        if section_name in KEY_BACKUP_EXCLUDED_SECTIONS:
            continue
        if not isinstance(entries, dict):
            continue
        section_entries = {}
        for key, value in entries.items():
            if not _is_backup_config_key(section_name, key):
                continue
            normalized_value = _normalize_backup_value(value)
            if normalized_value is not None:
                section_entries[key] = normalized_value
        if section_entries:
            restored[section_name] = section_entries

    if not restored:
        raise ValueError("key backup file contains no restorable keys")
    return restored


def _build_settings_preset_payload(params, app_version):
    """Construye el contenido del archivo de preajuste de parámetros de generación."""
    preset_params = {
        key: value
        for key, value in params.items()
        if key not in PRESET_EXCLUDED_PARAM_KEYS
    }
    if params.get("bgm_type") == "preset" and params.get("bgm_file"):
        try:
            builtin_bgm_path = bgm_service.resolve_builtin_bgm_file(
                str(params["bgm_file"])
            )
        except ValueError:
            # Los archivos personalizados son recursos locales y no pueden entrar en preajustes portables. En casos anómalos se mantiene
            # el comportamiento de exclusión existente, para que el archivo exportado no contenga rutas absolutas ni UUIDs que no existen en otro dispositivo.
            pass
        else:
            preset_params["bgm_file"] = Path(builtin_bgm_path).name
    return {
        "schema": SETTINGS_PRESET_SCHEMA,
        "version": SETTINGS_PRESET_VERSION,
        "app_version": str(app_version),
        "params": preset_params,
    }


def _parse_settings_preset(raw_bytes):
    """
    Analiza el archivo de preajuste y lo pasa a la validación de VideoParams.

    Los preajustes pueden generarse en otras máquinas o editarse a mano. Pasar siempre por la validación del modelo permite reutilizar
    las restricciones de rango existentes, y los preajustes ilegales se rechazan al importar, en lugar de fallar al generar la tarea.
    """
    payload = _load_transfer_payload(
        raw_bytes, SETTINGS_PRESET_SCHEMA, SETTINGS_PRESET_VERSION
    )
    preset_params = payload.get("params")
    if not isinstance(preset_params, dict):
        raise ValueError("settings preset file has no params object")

    params_input = {
        key: value
        for key, value in preset_params.items()
        if key not in PRESET_EXCLUDED_PARAM_KEYS
    }
    if preset_params.get("bgm_type") == "preset" and preset_params.get("bgm_file"):
        # Los preajustes solo pueden restaurar canciones incluidas que existan de verdad en la versión actual. La capa de servicio también rechaza separadores de directorio
        # y archivos subidos por el usuario, para que un archivo importado no use la vista previa para leer rutas locales arbitrarias.
        builtin_bgm_path = bgm_service.resolve_builtin_bgm_file(
            str(preset_params["bgm_file"])
        )
        params_input["bgm_file"] = Path(builtin_bgm_path).name
    # video_subject es un campo obligatorio de VideoParams, pero el preajuste permite guardar solo los ajustes de estilo.
    params_input.setdefault("video_subject", "")
    return VideoParams.model_validate(params_input).model_dump(mode="json")


def _apply_key_backup(restored_keys):
    """Escribe las claves analizadas en la configuración de ejecución y borra el estado residual de los controles correspondientes."""
    restored_count = 0
    for section_name, entries in restored_keys.items():
        for key, value in entries.items():
            _set_runtime_config(section_name, key, value)
            for widget_key in _credential_widget_state_keys(section_name, key):
                st.session_state.pop(widget_key, None)
            restored_count += 1
    # ElevenLabs La lista de voces se cachea por clave; al usar otra copia de seguridad hay que volver a obtenerla.
    for cache_key in list(st.session_state.keys()):
        if str(cache_key).startswith("elevenlabs_voices_"):
            del st.session_state[cache_key]
    return restored_count


def _apply_pending_settings_preset():
    """Aplica el preajuste importado antes de renderizar cualquier control."""
    preset_params = st.session_state.pop("settings_preset_payload", None)
    if not preset_params:
        return False

    _apply_restored_params(preset_params)
    logger.info("applied imported settings preset")
    return True


def _render_settings_transfer(params):
    """Renderiza los accesos de exportación e importación de preajustes de parámetros de generación."""
    with st.expander(tr("Settings Preset"), expanded=False):
        st.caption(tr("Settings Preset Help"))
        preset_payload = _build_settings_preset_payload(
            params.model_dump(mode="json"), config.project_version
        )
        st.download_button(
            tr("Export Settings"),
            data=json.dumps(preset_payload, ensure_ascii=False, indent=2).encode(
                "utf-8"
            ),
            file_name=SETTINGS_PRESET_FILE_NAME,
            mime="application/json",
            use_container_width=True,
            key="export_settings_preset_button",
            icon=":material/download:",
        )
        uploaded_preset = st.file_uploader(
            tr("Import Settings"),
            type=["json"],
            key="settings_preset_uploader",
        )
        if uploaded_preset is None:
            return
        # El archivo subido vuelve a aparecer en cada rerun posterior. Se registra el identificador del archivo ya procesado,
        # para que el mismo preajuste no sobrescriba una y otra vez los controles que el usuario ya cambió.
        if st.session_state.get("settings_preset_file_id") == uploaded_preset.file_id:
            return

        st.session_state["settings_preset_file_id"] = uploaded_preset.file_id
        try:
            preset_params = _parse_settings_preset(uploaded_preset.getvalue())
        except Exception as e:
            logger.warning(f"failed to import settings preset: {e}")
            st.error(tr("Settings Preset Import Failed"))
            return

        st.session_state["settings_preset_payload"] = preset_params
        st.rerun()


def _render_key_backup_settings(panel):
    """Renderiza los accesos de exportación y restauración de la copia de seguridad de claves."""
    with panel:
        backup_message = st.session_state.pop("key_backup_message", None)
        if backup_message:
            message_type, message = backup_message
            if message_type == "success":
                st.success(message)
            else:
                st.error(message)

        st.caption(tr("Key Backup Help"))
        st.warning(tr("Key Backup Warning"))

        backup_payload = _build_key_backup_payload(
            _RUNTIME_CONFIG_SECTIONS, config.project_version
        )
        backup_key_count = _count_backup_keys(backup_payload["keys"])
        st.caption(tr("Key Backup Summary").format(count=backup_key_count))
        st.download_button(
            tr("Export Keys"),
            data=json.dumps(backup_payload, ensure_ascii=False, indent=2).encode(
                "utf-8"
            ),
            file_name=KEY_BACKUP_FILE_NAME,
            mime="application/json",
            disabled=backup_key_count == 0,
            use_container_width=True,
            key="export_key_backup_button",
            icon=":material/download:",
        )

        uploaded_backup = st.file_uploader(
            tr("Import Keys"),
            type=["json"],
            key="key_backup_uploader",
        )
        if uploaded_backup is None:
            return
        if st.session_state.get("key_backup_file_id") == uploaded_backup.file_id:
            return

        st.session_state["key_backup_file_id"] = uploaded_backup.file_id
        try:
            restored_keys = _parse_key_backup(
                uploaded_backup.getvalue(), _RUNTIME_CONFIG_SECTIONS
            )
        except Exception as e:
            logger.warning(f"failed to import key backup: {e}")
            st.session_state["key_backup_message"] = (
                "error",
                tr("Key Restore Failed"),
            )
        else:
            restored_count = _apply_key_backup(restored_keys)
            _save_runtime_config()
            logger.info(f"restored keys from backup file: count={restored_count}")
            st.session_state["key_backup_message"] = (
                "success",
                tr("Keys Restored").format(count=restored_count),
            )
        # Los campos de clave TTS de la página principal también tienen que leer la configuración restaurada, así que se refresca toda la página.
        # El estado abierto del diálogo de ajustes se guarda en session_state, y tras refrescar se vuelve a desplegar.
        st.rerun(scope="app")


# -----------------------------------------------------------------------------
# Diálogos de ajustes y de prompt
# -----------------------------------------------------------------------------


def _render_logging_settings():
    level, log_requests = get_logging_options()
    with st.expander(tr("Diagnostic Logging")):
        st.caption(tr("Diagnostic Logging Help"))
        with st.form("diagnostic_logging_form"):
            selected_level = st.selectbox(
                tr("Log Level"),
                LOG_LEVELS,
                index=LOG_LEVELS.index(level),
                key="diagnostic_log_level",
            )
            selected_trace = st.checkbox(
                tr("Trace API Requests"),
                value=log_requests,
                key="diagnostic_log_requests",
                help=tr("Trace API Requests Help"),
            )
            apply = st.form_submit_button(tr("Apply Logging Settings"))
        if apply:
            # El diagnóstico no cambia los parámetros de generación y surte efecto de inmediato; la persistencia sigue pasando por la cola de configuración no bloqueante.
            config.update_logging_options(selected_level, selected_trace)
            _save_runtime_config()
            st.success(tr("Logging Settings Applied"))


# Los ajustes son una operación poco frecuente; se usa un Dialog de tamaño medio para no ocupar durante mucho tiempo el espacio vertical de la página principal,
# y a la vez se controla el ancho de lectura, para que el diálogo no parezca demasiado disperso en pantallas anchas.
# Dialog hereda el comportamiento de fragment, y las interacciones con controles internos solo redibujan el diálogo; al final de la función se guarda aparte la configuración,
# y al cerrar se lanza mediante callback la sincronización de toda la página, para que el flujo de generación lea el Provider y los ajustes de interfaz más recientes.
def _remember_settings_tab(widget_key, targets):
    selected = st.session_state.get(widget_key)
    for target, label in targets.items():
        if selected == label:
            st.session_state["settings_active_tab"] = target
            break


def _remember_settings_selection(memory_key, widget_key):
    st.session_state[memory_key] = st.session_state[widget_key]


def _remember_material_provider(group, widget_key):
    remembered = dict(st.session_state.get("settings_material_providers", {}))
    remembered[group] = st.session_state[widget_key]
    st.session_state["settings_material_group"] = group
    st.session_state["settings_material_providers"] = remembered


def _initialize_material_navigation(target, settings):
    groups = {
        group: providers
        for group, providers in VIDEO_SOURCE_GROUPS.items()
        if group != "local"
    }
    source = settings.get("video_source", "pexels")
    source_group = next(
        (group for group, providers in groups.items() if source in providers), None
    )
    explicit = {"stock": "stock_video", "video": "ai_video", "image": "ai_image"}.get(
        target
    )
    group = explicit or (source_group if target == "material" else None)
    group = group or st.session_state.get(
        "settings_material_group", source_group or "stock_video"
    )
    if group not in groups:
        group = "stock_video"
    remembered = dict(st.session_state.get("settings_material_providers", {}))
    if target == "material" and source_group:
        remembered[group] = source
    elif group not in remembered or remembered[group] not in groups[group]:
        remembered[group] = source if source in groups[group] else groups[group][0]
    st.session_state["settings_material_group"] = group
    st.session_state["settings_material_providers"] = remembered
    st.session_state[
        localized_widget_key(f"settings_material_provider_{group}_select")
    ] = remembered[group]


def _material_provider_labels():
    return {
        "wikimedia": tr("Documentary Portraits"),
        "pexels": tr("Pexels"),
        "pixabay": tr("Pixabay"),
        "coverr": tr("Coverr"),
        "metaso_minimax": tr("Metaso MiniMax H3"),
        "ofox": tr("OFox AI Video"),
        "loomloom": tr("Shengsuan Cloud AI Video"),
        "volcengine_seedance": tr("Volcano Engine Seedance"),
        "infistar": tr("Infistar AI Video"),
        "jalapeno": tr("Jalapeno Cloud AI Video"),
        "wavespeed": tr("WaveSpeed AI Video"),
        "muapi": tr("MuAPI AI Video"),
        "openai_image": tr("OpenAI Compatible Text-to-Image"),
        "codex_image": tr("codex.image.label"),
        "local": tr("Local file"),
    }


def _general_provider_readiness(settings):
    from app.services import chatgpt_oauth, codex_oauth, kimi_code_oauth
    from webui.settings_providers import (
        configured_llm_providers,
        configured_material_providers,
    )

    oauth = {}
    for provider, read_status in (
        ("openai_codex", codex_oauth.account_status),
        ("openai_chatgpt", chatgpt_oauth.account_status),
        ("kimi_code", kimi_code_oauth.is_authorized),
    ):
        try:
            oauth[provider] = bool(read_status())
        except (ValueError, OSError):
            oauth[provider] = False
    return {
        "llm": configured_llm_providers(settings, oauth),
        **configured_material_providers(
            settings, VIDEO_SOURCE_GROUPS, oauth["openai_codex"]
        ),
    }


def _sync_general_provider_widgets(name, value):
    prefix = f"{name}_"
    for widget_key in list(st.session_state):
        if widget_key.startswith(prefix) and not widget_key.endswith("_component"):
            st.session_state[widget_key] = value
    st.session_state[localized_widget_key(name)] = value


def _activate_general_provider(section):
    settings = config.snapshot_config_with_pending(config.app)
    selected = st.session_state.get(
        localized_widget_key(f"settings_general_{section}_provider_select")
    )
    if selected not in _general_provider_readiness(settings).get(section, ()):
        st.session_state["settings_general_activation_error"] = True
        return
    st.session_state.pop("settings_general_activation_error", None)
    if section == "llm":
        _set_runtime_config("app", "llm_provider", selected)
        _sync_general_provider_widgets("llm_provider_select", selected)
    else:
        _set_runtime_config("app", "video_source", selected)
        _sync_general_provider_widgets("video_source_select", selected)
        remembered = dict(st.session_state.get("settings_material_providers", {}))
        remembered[section] = selected
        st.session_state["settings_material_providers"] = remembered
        st.session_state["settings_material_group"] = section
        _sync_general_provider_widgets(
            f"settings_material_provider_{section}_select", selected
        )
    _save_runtime_config()


def _render_general_settings(settings):
    configured = _general_provider_readiness(settings)
    material_labels = _material_provider_labels()
    llm_labels = {
        provider.provider_id: get_llm_provider_label(provider)
        for provider in LLM_PROVIDER_REGISTRY
    }
    st.subheader(tr("General Providers Title"))
    st.caption(tr("General Providers Help"))
    if st.session_state.pop("settings_general_activation_error", False):
        st.warning(tr("Provider Configuration Changed"))
    cards = (
        ("llm", "LLM Provider", "psychology", "llm"),
        ("ai_video", "AI Video", "movie", "video"),
        ("ai_image", "AI Image", "image", "image"),
        ("stock_video", "Stock Video", "video_library", "stock"),
    )
    for section, title, icon, target in cards:
        labels = llm_labels if section == "llm" else material_labels
        active = (
            settings.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
            if section == "llm"
            else settings.get("video_source", "pexels")
        )
        is_active_category = section == "llm" or active in VIDEO_SOURCE_GROUPS[section]
        options = configured[section]
        widget_key = localized_widget_key(f"settings_general_{section}_provider_select")
        memory_key = f"settings_general_{section}_provider"
        active_key = f"settings_general_{section}_active_snapshot"
        active_value = active if is_active_category else None
        if st.session_state.get(active_key) != active_value and active_value in options:
            st.session_state[memory_key] = active_value
            st.session_state[widget_key] = active_value
        st.session_state[active_key] = active_value
        remembered = st.session_state.get(memory_key)
        if st.session_state.get(widget_key) not in options:
            st.session_state[widget_key] = (
                remembered
                if remembered in options
                else active
                if active in options
                else options[0]
                if options
                else None
            )
        with st.container(border=True):
            st.subheader(f":material/{icon}: {tr(title)}")
            if is_active_category and active in options:
                st.caption(
                    tr("Active Provider").format(
                        provider=labels.get(active, str(active))
                    )
                )
            if not options:
                st.info(tr("No Configured Providers"))
                st.button(
                    tr("Configure Providers"),
                    key=f"settings_general_configure_{section}",
                    width="stretch",
                    on_click=_open_settings_dialog,
                    args=(target,),
                )
                continue
            selection = st.selectbox(
                tr("Configured Providers"),
                options=options,
                key=widget_key,
                format_func=labels.get,
                index=None,
                on_change=_remember_settings_selection,
                args=(memory_key, widget_key),
            )
            if selection is not None:
                st.session_state[memory_key] = selection
            use_column, configure_column = st.columns(2)
            use_column.button(
                tr("Use Provider"),
                key=f"settings_general_use_{section}",
                type="primary",
                width="stretch",
                disabled=selection is None,
                on_click=_activate_general_provider,
                args=(section,),
            )
            configure_column.button(
                tr("Configure Providers"),
                key=f"settings_general_configure_{section}",
                width="stretch",
                on_click=_open_settings_dialog,
                args=(target,),
            )


def _save_settings_change(section, config_key, widget_key, transform):
    value = st.session_state.get(widget_key)
    if value is not None:
        _set_runtime_config(section, config_key, transform(value))


def _save_settings_endpoint(provider, widget_key):
    endpoint_id = st.session_state.get(widget_key)
    endpoint = next(
        (
            item
            for item in provider.service_endpoints
            if item.endpoint_id == endpoint_id
        ),
        None,
    )
    if endpoint:
        _set_runtime_config(
            "app",
            provider.config_key("base_url"),
            normalize_provider_override(
                endpoint.base_url, provider.effective_default_base_url
            ),
        )


def _save_settings_api_keys(config_key, widget_key):
    _save_material_api_keys(config_key, st.session_state.get(widget_key, ""))


def _render_material_provider_settings(provider):
    app_settings = config.snapshot_config_with_pending(config.app)
    if provider == "wikimedia":
        st.caption(tr("Documentary Portraits Help"))
    elif provider in VIDEO_SOURCE_GROUPS["stock_video"]:
        labels = {
            "pexels": "Pexels API Key",
            "pixabay": "Pixabay API Key",
            "coverr": "Coverr API Key",
        }
        config_key = f"{provider}_api_keys"
        value = st.text_input(
            tr(labels[provider]),
            value=_get_material_api_keys(config_key),
            type="password",
            key=f"{config_key}_input",
            on_change=_save_settings_api_keys,
            args=(config_key, f"{config_key}_input"),
        )
        _save_material_api_keys(config_key, value)
    elif provider == "metaso_minimax":
        st.markdown(f"**{tr('Metaso MiniMax H3')}**")
        metaso_api_key = st.text_input(
            tr("Metaso MiniMax API Key"),
            value=str(app_settings.get("metaso_minimax_api_key", "") or "").strip(),
            type="password",
            help=tr("Metaso MiniMax API Key Help"),
            key="metaso_minimax_api_key_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "metaso_minimax_api_key",
                "metaso_minimax_api_key_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config("app", "metaso_minimax_api_key", metaso_api_key.strip())
        configured_metaso_base_url = str(
            app_settings.get(
                "metaso_minimax_base_url",
                metaso_minimax.DEFAULT_BASE_URL,
            )
            or metaso_minimax.DEFAULT_BASE_URL
        ).strip()
        metaso_base_url = st.text_input(
            tr("Metaso MiniMax Base URL"),
            value=(
                ""
                if configured_metaso_base_url == metaso_minimax.DEFAULT_BASE_URL
                else configured_metaso_base_url
            ),
            placeholder=metaso_minimax.DEFAULT_BASE_URL,
            key="metaso_minimax_base_url_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "metaso_minimax_base_url",
                "metaso_minimax_base_url_input",
                lambda value: value.strip() or metaso_minimax.DEFAULT_BASE_URL,
            ),
        )
        _set_runtime_config(
            "app",
            "metaso_minimax_base_url",
            metaso_base_url.strip() or metaso_minimax.DEFAULT_BASE_URL,
        )
        configured_metaso_resolution = (
            str(
                app_settings.get(
                    "metaso_minimax_resolution",
                    metaso_minimax.DEFAULT_RESOLUTION,
                )
            )
            .strip()
            .upper()
        )
        metaso_resolution_options = sorted(
            metaso_minimax.SUPPORTED_RESOLUTIONS,
            key=lambda value: value != metaso_minimax.DEFAULT_RESOLUTION,
        )
        resolution_is_valid = (
            configured_metaso_resolution in metaso_minimax.SUPPORTED_RESOLUTIONS
        )
        if not resolution_is_valid:
            # la resolución afecta directamente al coste. Si la configuración manual es errónea, se conserva el valor original y se pide al usuario
            # que elija, sin cambiarlo en silencio al 2K, más caro, al abrir el diálogo de ajustes.
            st.error(
                tr("Metaso MiniMax Invalid Resolution").format(
                    value=configured_metaso_resolution,
                    supported=", ".join(metaso_resolution_options),
                )
            )
        metaso_resolution = st.selectbox(
            tr("Metaso MiniMax Resolution"),
            options=metaso_resolution_options,
            index=(
                metaso_resolution_options.index(configured_metaso_resolution)
                if resolution_is_valid
                else None
            ),
            key="metaso_minimax_resolution_input",
            help=tr("Metaso MiniMax Resolution Help"),
            placeholder=tr("Select Metaso MiniMax Resolution"),
            on_change=_save_settings_change,
            args=(
                "app",
                "metaso_minimax_resolution",
                "metaso_minimax_resolution_input",
                lambda value: value,
            ),
        )
        if metaso_resolution is not None:
            _set_runtime_config("app", "metaso_minimax_resolution", metaso_resolution)
    elif provider == "ofox":
        st.markdown("**OfoxAI**")
        st.caption(f"[OfoxAI]({OFOX_REFERRAL_URL}) · {tr('OFox AI Video Help')}")
        ofox_api_key = st.text_input(
            tr("OFox API Key"),
            value=str(app_settings.get("ofox_api_key", "") or ""),
            type="password",
            key="ofox_api_key_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "ofox_api_key",
                "ofox_api_key_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config("app", "ofox_api_key", ofox_api_key.strip())
        ofox_model = st.text_input(
            tr("OFox Text-to-Video Model"),
            value=str(
                app_settings.get(
                    "ofox_text_to_video_model",
                    ofox.DEFAULT_MODEL_ID,
                )
                or ofox.DEFAULT_MODEL_ID
            ),
            key="ofox_text_to_video_model_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "ofox_text_to_video_model",
                "ofox_text_to_video_model_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config("app", "ofox_text_to_video_model", ofox_model.strip())
        configured_ofox_base_url = str(
            app_settings.get("ofox_base_url", ofox.DEFAULT_BASE_URL)
            or ofox.DEFAULT_BASE_URL
        ).strip()
        ofox_base_url = st.text_input(
            tr("OFox Base URL"),
            value=(
                ""
                if configured_ofox_base_url == ofox.DEFAULT_BASE_URL
                else configured_ofox_base_url
            ),
            placeholder=ofox.DEFAULT_BASE_URL,
            key="ofox_base_url_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "ofox_base_url",
                "ofox_base_url_input",
                lambda value: value.strip() or ofox.DEFAULT_BASE_URL,
            ),
        )
        _set_runtime_config(
            "app",
            "ofox_base_url",
            ofox_base_url.strip() or ofox.DEFAULT_BASE_URL,
        )
        ofox_vendor_options = [
            (tr("OFox Vendor BytePlus"), "byteplus"),
            (tr("OFox Vendor Volcengine"), "volcengine"),
            (tr("OFox Vendor Auto"), ""),
        ]
        configured_ofox_vendor = str(
            app_settings.get("ofox_provider", ofox.DEFAULT_PROVIDER_TYPE) or ""
        ).strip()
        if configured_ofox_vendor not in {value for _, value in ofox_vendor_options}:
            # Si el usuario fijó a mano en config.toml otro nombre de fabricante, se conserva esa elección,
            # para que abrir la página de ajustes no la sobrescriba con el valor por defecto del desplegable.
            ofox_vendor_options.append((configured_ofox_vendor, configured_ofox_vendor))
        selected_ofox_vendor = stable_selectbox(
            tr("OFox Upstream Vendor"),
            options=[value for _, value in ofox_vendor_options],
            default_value=configured_ofox_vendor,
            key="ofox_provider_select",
            format_func=lambda value: dict(
                (v, label) for label, v in ofox_vendor_options
            )[value],
            help=tr("OFox Upstream Vendor Help"),
            on_change=_save_settings_change,
            args=(
                "app",
                "ofox_provider",
                localized_widget_key("ofox_provider_select"),
                lambda value: value,
            ),
        )
        _set_runtime_config("app", "ofox_provider", selected_ofox_vendor)
    elif provider == "loomloom":
        st.markdown(f"**{tr('Shengsuan Cloud AI Video')}**")
        app_config_snapshot = config.snapshot_config_with_pending(config.app)
        if (
            str(app_config_snapshot.get("llm_provider", "") or "").lower()
            == "shengsuanyun"
        ):
            # Si el Provider de modelo grande elegido es Shengsuanyun, la generación de video reutiliza directamente
            # la misma clave, sin mostrar un campo de entrada independiente que se presta a confusión.
            st.caption(tr("Shengsuan Cloud API Key Reused"))
        else:
            configured_loomloom_token = str(
                app_config_snapshot.get("loomloom_api_token", "") or ""
            ).strip()
            loomloom_api_token = st.text_input(
                tr("Shengsuan Cloud API Key"),
                value=configured_loomloom_token,
                type="password",
                key="loomloom_api_token_input",
                help=tr("Shengsuan Cloud API Key Help"),
                placeholder=tr("Shengsuan Cloud API Key Placeholder"),
                on_change=_save_settings_change,
                args=(
                    "app",
                    "loomloom_api_token",
                    "loomloom_api_token_input",
                    lambda value: value.strip(),
                ),
            ).strip()
            _set_runtime_config("app", "loomloom_api_token", loomloom_api_token)
    elif provider == "volcengine_seedance":
        seedance_api_key_value = str(
            app_settings.get("volcengine_seedance_api_key", "") or ""
        ).strip()
        shared_ark_api_key = str(
            app_settings.get("volcengine_api_key", "") or ""
        ).strip()
        environment_ark_api_key = os.getenv("VOLCENGINE_ARK_API_KEY", "").strip()
        seedance_reuses_llm_key = bool(
            not seedance_api_key_value
            and not environment_ark_api_key
            and shared_ark_api_key
        )
        seedance_title = f"**{tr('Volcano Engine Seedance')}**"
        if seedance_reuses_llm_key:
            # Solo cuando se reutiliza la clave del modelo grande no se aprecia directamente en el campo actual; conservar este aviso
            # evita que el usuario crea que debe rellenarla de nuevo; el estado de configuración normal no se repite.
            seedance_title += f" :blue[{tr('Reusing LLM API Key')}]"
        st.markdown(seedance_title)
        seedance_api_key = st.text_input(
            tr("Volcano Engine Ark API Key"),
            value=seedance_api_key_value,
            type="password",
            help=tr("Volcano Engine Ark API Key Help"),
            key="volcengine_seedance_api_key_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "volcengine_seedance_api_key",
                "volcengine_seedance_api_key_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config(
            "app", "volcengine_seedance_api_key", seedance_api_key.strip()
        )
        configured_seedance_model = str(
            app_settings.get(
                "volcengine_seedance_model",
                volcengine_seedance.DEFAULT_MODEL_ID,
            )
            or volcengine_seedance.DEFAULT_MODEL_ID
        ).strip()
        seedance_model = st.text_input(
            tr("Volcano Engine Seedance Model"),
            # Los valores por defecto incluidos se muestran con placeholder, y el modelo o ID de punto de acceso
            # personalizado por el usuario se sigue mostrando y guardando como valor real.
            value=(
                ""
                if configured_seedance_model == volcengine_seedance.DEFAULT_MODEL_ID
                else configured_seedance_model
            ),
            placeholder=volcengine_seedance.DEFAULT_MODEL_ID,
            key="volcengine_seedance_model_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "volcengine_seedance_model",
                "volcengine_seedance_model_input",
                lambda value: value.strip() or volcengine_seedance.DEFAULT_MODEL_ID,
            ),
        )
        _set_runtime_config(
            "app",
            "volcengine_seedance_model",
            seedance_model.strip() or volcengine_seedance.DEFAULT_MODEL_ID,
        )
        configured_seedance_base_url = str(
            app_settings.get(
                "volcengine_seedance_base_url",
                volcengine_seedance.DEFAULT_BASE_URL,
            )
            or volcengine_seedance.DEFAULT_BASE_URL
        ).strip()
        seedance_base_url = st.text_input(
            tr("Volcano Engine Ark Base URL"),
            value=(
                ""
                if configured_seedance_base_url == volcengine_seedance.DEFAULT_BASE_URL
                else configured_seedance_base_url
            ),
            placeholder=volcengine_seedance.DEFAULT_BASE_URL,
            key="volcengine_seedance_base_url_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "volcengine_seedance_base_url",
                "volcengine_seedance_base_url_input",
                lambda value: value.strip() or volcengine_seedance.DEFAULT_BASE_URL,
            ),
        )
        _set_runtime_config(
            "app",
            "volcengine_seedance_base_url",
            seedance_base_url.strip() or volcengine_seedance.DEFAULT_BASE_URL,
        )
    elif provider == "infistar":
        st.markdown(f"**{tr('Infistar AI Video')}**")
        st.caption(tr("Infistar Video Key Help"))
        infistar_video_key = st.text_input(
            tr("Infistar Video API Key"),
            value=str(app_settings.get("infistar_video_api_key", "") or ""),
            type="password",
            key="infistar_video_api_key_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "infistar_video_api_key",
                "infistar_video_api_key_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config("app", "infistar_video_api_key", infistar_video_key.strip())
        infistar_video_base_url = st.text_input(
            tr("Infistar Video Base URL"),
            value=str(app_settings.get("infistar_video_base_url", "") or ""),
            placeholder=infistar_video.DEFAULT_BASE_URL,
            key="infistar_video_base_url_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "infistar_video_base_url",
                "infistar_video_base_url_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config(
            "app", "infistar_video_base_url", infistar_video_base_url.strip()
        )
    elif provider == "jalapeno":
        _render_jalapeno_provider_settings()
    elif provider == "wavespeed":
        wavespeed_api_key = _get_material_api_keys("wavespeed_api_keys")
        st.markdown("**WaveSpeed**")
        wavespeed_api_key = st.text_input(
            tr("WaveSpeed API Key"),
            value=wavespeed_api_key,
            type="password",
            key="wavespeed_api_keys_input",
            on_change=_save_settings_api_keys,
            args=("wavespeed_api_keys", "wavespeed_api_keys_input"),
        )
        _save_material_api_keys("wavespeed_api_keys", wavespeed_api_key)
    elif provider == "muapi":
        st.markdown(f"**{tr('MuAPI AI Video')}**")
        st.caption(tr("MuAPI AI Video Help"))
        muapi_api_key = st.text_input(
            tr("MuAPI API Key"),
            value=str(app_settings.get("muapi_api_key", "") or ""),
            type="password",
            help=tr("MuAPI API Key Help"),
            key="muapi_api_key_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "muapi_api_key",
                "muapi_api_key_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config("app", "muapi_api_key", muapi_api_key.strip())
        configured_muapi_base_url = str(
            app_settings.get("muapi_base_url", muapi.DEFAULT_BASE_URL)
            or muapi.DEFAULT_BASE_URL
        ).strip()
        muapi_base_url = st.text_input(
            tr("MuAPI Base URL"),
            value=(
                ""
                if configured_muapi_base_url == muapi.DEFAULT_BASE_URL
                else configured_muapi_base_url
            ),
            placeholder=muapi.DEFAULT_BASE_URL,
            key="muapi_base_url_input",
            help=tr("MuAPI Base URL Help"),
            on_change=_save_settings_change,
            args=(
                "app",
                "muapi_base_url",
                "muapi_base_url_input",
                lambda value: value.strip() or muapi.DEFAULT_BASE_URL,
            ),
        )
        _set_runtime_config(
            "app",
            "muapi_base_url",
            muapi_base_url.strip() or muapi.DEFAULT_BASE_URL,
        )
        configured_muapi_endpoint = str(
            app_settings.get("muapi_video_endpoint", muapi.DEFAULT_ENDPOINT)
            or muapi.DEFAULT_ENDPOINT
        ).strip()
        muapi_endpoint = st.text_input(
            tr("MuAPI Video Endpoint"),
            value=(
                ""
                if configured_muapi_endpoint == muapi.DEFAULT_ENDPOINT
                else configured_muapi_endpoint
            ),
            placeholder=muapi.DEFAULT_ENDPOINT,
            key="muapi_video_endpoint_input",
            help=tr("MuAPI Video Endpoint Help"),
            on_change=_save_settings_change,
            args=(
                "app",
                "muapi_video_endpoint",
                "muapi_video_endpoint_input",
                lambda value: value.strip() or muapi.DEFAULT_ENDPOINT,
            ),
        )
        _set_runtime_config(
            "app",
            "muapi_video_endpoint",
            muapi_endpoint.strip() or muapi.DEFAULT_ENDPOINT,
        )
        configured_muapi_resolution = str(
            app_settings.get("muapi_resolution", muapi.DEFAULT_RESOLUTION)
            or muapi.DEFAULT_RESOLUTION
        ).strip()
        muapi_resolution = st.text_input(
            tr("MuAPI Resolution"),
            value=(
                ""
                if configured_muapi_resolution == muapi.DEFAULT_RESOLUTION
                else configured_muapi_resolution
            ),
            placeholder=muapi.DEFAULT_RESOLUTION,
            key="muapi_resolution_input",
            help=tr("MuAPI Resolution Help"),
            on_change=_save_settings_change,
            args=(
                "app",
                "muapi_resolution",
                "muapi_resolution_input",
                lambda value: value.strip() or muapi.DEFAULT_RESOLUTION,
            ),
        )
        _set_runtime_config(
            "app",
            "muapi_resolution",
            muapi_resolution.strip() or muapi.DEFAULT_RESOLUTION,
        )
    elif provider == "codex_image":
        st.markdown(f"**{tr('codex.image.label')}**")
        st.caption(tr("codex.image.help"))
        from webui.codex_models import render_codex_model_selector

        codex_image_model = render_codex_model_selector(
            str(app_settings.get("codex_image_model_name", "") or ""),
            tr,
            prefix="codex_image",
            on_change=lambda model: _set_runtime_config(
                "app", "codex_image_model_name", model
            ),
        )
        _set_runtime_config("app", "codex_image_model_name", codex_image_model)
        codex_image_timeout = st.text_input(
            tr("codex.image.timeout"),
            value=str(app_settings.get("codex_image_timeout") or "300"),
            placeholder="300",
            key="codex_image_timeout_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "codex_image_timeout",
                "codex_image_timeout_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config("app", "codex_image_timeout", codex_image_timeout.strip())
        from webui.codex_login import render_codex_login

        with st.container():
            render_codex_login(tr)
    elif provider == "openai_image":
        st.markdown(f"**{tr('OpenAI Compatible Text-to-Image')}**")

        openai_image_base_url = st.text_input(
            tr("OpenAI Image Base URL"),
            value=str(app_settings.get("openai_image_base_url", "") or ""),
            placeholder="https://api.openai.com/v1",
            key="openai_image_base_url_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "openai_image_base_url",
                "openai_image_base_url_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config(
            "app", "openai_image_base_url", openai_image_base_url.strip()
        )

        openai_image_api_key = _get_material_api_keys("openai_image_api_keys")
        openai_image_api_key = st.text_input(
            tr("OpenAI Image API Key"),
            value=openai_image_api_key,
            type="password",
            help=tr("OpenAI Image API Key Help"),
            key="openai_image_api_keys_input",
            on_change=_save_settings_api_keys,
            args=("openai_image_api_keys", "openai_image_api_keys_input"),
        )
        _save_material_api_keys("openai_image_api_keys", openai_image_api_key)

        openai_image_model = st.text_input(
            tr("OpenAI Image Model"),
            value=str(app_settings.get("openai_image_model", "") or ""),
            placeholder="gpt-image-2",
            key="openai_image_model_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "openai_image_model",
                "openai_image_model_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config("app", "openai_image_model", openai_image_model.strip())
        # Solo muestra el valor de referencia, sin escribir el endpoint oficial de OpenAI como configuración por defecto.
        # La Base URL y el ID de modelo de los servicios compatibles no tienen un valor unificado; dejarlo vacío evita que
        # el usuario se conecte sin saberlo a la interfaz oficial de pago, y tampoco sobrescribe la configuración antigua.
        st.caption(tr("OpenAI Image Configuration Example"))

        with st.expander(tr("OpenAI Image Advanced Settings"), expanded=False):
            openai_image_size = st.text_input(
                tr("OpenAI Image Size"),
                value=str(app_settings.get("openai_image_size", "") or ""),
                placeholder="1024x1536",
                help=tr("OpenAI Image Size Help"),
                key="openai_image_size_input",
                on_change=_save_settings_change,
                args=(
                    "app",
                    "openai_image_size",
                    "openai_image_size_input",
                    lambda value: value.strip(),
                ),
            )
            _set_runtime_config("app", "openai_image_size", openai_image_size.strip())

            openai_image_prompt_template = st.text_input(
                tr("OpenAI Image Prompt Template"),
                value=str(app_settings.get("openai_image_prompt_template", "") or ""),
                placeholder="cinematic photo of {term}, photorealistic",
                help=tr("OpenAI Image Prompt Template Help"),
                key="openai_image_prompt_template_input",
                on_change=_save_settings_change,
                args=(
                    "app",
                    "openai_image_prompt_template",
                    "openai_image_prompt_template_input",
                    lambda value: value.strip(),
                ),
            )
            _set_runtime_config(
                "app",
                "openai_image_prompt_template",
                openai_image_prompt_template.strip(),
            )


@st.dialog(
    tr("Settings"),
    width="medium",
    on_dismiss=_dismiss_settings_dialog,
)
def _render_settings_dialog():
    app_settings = config.snapshot_config_with_pending(config.app)
    ui_settings = config.snapshot_config_with_pending(config.ui)
    with st.container():
        # El hide_config histórico solo servía para ocultar el antiguo panel de ajustes básicos. Al pasar a un acceso de ajustes fijo, ese valor
        # ya no tiene significado visible para el usuario, y se migra siempre a false, para que la configuración antigua no afecte a versiones posteriores.
        _set_runtime_config("app", "hide_config", False)
        tab_keys = {
            "general": "General Settings Tab",
            "llm": "LLM Settings Tab",
            "material": "Material API Tab",
            "publish": "Auto-Publish Settings",
            "interface": "Interface Settings Tab",
            "keys": "Key Backup Tab",
            "cache": "Cache Management Tab",
        }
        settings_tab_targets = {key: tr(label) for key, label in tab_keys.items()}
        settings_tabs_key = localized_widget_key("settings_dialog_tabs")
        target_tab = st.session_state.pop("settings_dialog_target_tab", None)
        if target_tab in {"stock", "video", "image", "material"}:
            _initialize_material_navigation(target_tab, app_settings)
        if target_tab in {"stock", "video", "image"}:
            target_tab = "material"
        active = target_tab or st.session_state.get("settings_active_tab", "general")
        if active not in settings_tab_targets:
            active = "general"
        if (
            target_tab
            or st.session_state.get(settings_tabs_key)
            not in settings_tab_targets.values()
        ):
            st.session_state[settings_tabs_key] = settings_tab_targets[active]
        panels = st.tabs(
            list(settings_tab_targets.values()),
            key=settings_tabs_key,
            on_change=_remember_settings_tab,
            args=(settings_tabs_key, settings_tab_targets),
        )
        (
            general_config_panel,
            middle_config_panel,
            right_config_panel,
            publish_config_panel,
            left_config_panel,
            key_backup_panel,
            cache_config_panel,
        ) = panels
        _remember_settings_tab(settings_tabs_key, settings_tab_targets)
        with general_config_panel:
            if general_config_panel.open:
                _render_general_settings(app_settings)

        with publish_config_panel:
            if publish_config_panel.open:
                st.write(
                    tr(
                        "Automatically publish generated videos to social media using upload-post.com"
                    )
                )
                st.info(
                    tr("Upload-Post Setup Guide").format(
                        api_keys_url=UPLOAD_POST_API_KEYS_URL,
                        manage_users_url=UPLOAD_POST_MANAGE_USERS_URL,
                    )
                )

                is_enabled = app_settings.get("upload_post_enabled", False)
                is_auto = app_settings.get("upload_post_auto_upload", False)

                # Ambas claves son independientes: enabled permite que flujos externos llamen a Upload-Post,
                # auto_upload solo decide si se publica automáticamente al terminar el renderizado. Fusionarlas en una sola casilla haría que,
                # con una configuración en la que ambas claves no coinciden, solo abrir el diálogo de ajustes reescribiera enabled como False.
                upload_post_enabled = st.checkbox(
                    tr("Enable Upload-Post Integration"),
                    value=is_enabled,
                    key="upload_post_enabled_checkbox",
                    on_change=_save_settings_change,
                    args=(
                        "app",
                        "upload_post_enabled",
                        "upload_post_enabled_checkbox",
                        lambda value: value,
                    ),
                )
                if upload_post_enabled != is_enabled:
                    _set_runtime_config(
                        "app", "upload_post_enabled", upload_post_enabled
                    )

                upload_post_auto_upload = st.checkbox(
                    tr("Enable Auto-Publish"),
                    value=is_auto,
                    key="upload_post_auto_upload_checkbox",
                    on_change=_save_settings_change,
                    args=(
                        "app",
                        "upload_post_auto_upload",
                        "upload_post_auto_upload_checkbox",
                        lambda value: value,
                    ),
                )
                if upload_post_auto_upload != is_auto:
                    _set_runtime_config(
                        "app", "upload_post_auto_upload", upload_post_auto_upload
                    )

                upload_post_api_key = st.text_input(
                    tr("Upload-Post API Key"),
                    value=app_settings.get("upload_post_api_key", ""),
                    type="password",
                    help=tr("Upload-Post API Key Help").format(
                        api_keys_url=UPLOAD_POST_API_KEYS_URL
                    ),
                    key="upload_post_api_key_input",
                    on_change=_save_settings_change,
                    args=(
                        "app",
                        "upload_post_api_key",
                        "upload_post_api_key_input",
                        lambda value: value,
                    ),
                )
                if upload_post_api_key != app_settings.get("upload_post_api_key", ""):
                    _set_runtime_config(
                        "app", "upload_post_api_key", upload_post_api_key
                    )

                upload_post_username = st.text_input(
                    tr("Upload-Post Profile Username"),
                    value=app_settings.get("upload_post_username", ""),
                    help=tr("Upload-Post Profile Username Help").format(
                        manage_users_url=UPLOAD_POST_MANAGE_USERS_URL
                    ),
                    key="upload_post_username_input",
                    on_change=_save_settings_change,
                    args=(
                        "app",
                        "upload_post_username",
                        "upload_post_username_input",
                        lambda value: value,
                    ),
                )
                if upload_post_username != app_settings.get("upload_post_username", ""):
                    _set_runtime_config(
                        "app", "upload_post_username", upload_post_username
                    )

                upload_post_platforms = st.multiselect(
                    tr("Platforms"),
                    options=["tiktok", "instagram", "youtube"],
                    default=app_settings.get(
                        "upload_post_platforms", ["tiktok", "instagram"]
                    ),
                    help="Select platforms to publish to",
                    key="upload_post_platforms_multiselect",
                    on_change=_save_settings_change,
                    args=(
                        "app",
                        "upload_post_platforms",
                        "upload_post_platforms_multiselect",
                        lambda value: value,
                    ),
                )
                if upload_post_platforms != app_settings.get(
                    "upload_post_platforms", ["tiktok", "instagram"]
                ):
                    _set_runtime_config(
                        "app", "upload_post_platforms", upload_post_platforms
                    )

                if "youtube" in upload_post_platforms:
                    yt_status_options = ["public", "private", "unlisted"]
                    yt_saved = app_settings.get(
                        "upload_post_youtube_privacy_status", "public"
                    )
                    if yt_saved not in yt_status_options:
                        yt_saved = "public"
                    upload_post_youtube_privacy_status = st.selectbox(
                        tr("YouTube Privacy Status"),
                        options=yt_status_options,
                        index=yt_status_options.index(yt_saved),
                        key="upload_post_youtube_privacy_status_selectbox",
                        on_change=_save_settings_change,
                        args=(
                            "app",
                            "upload_post_youtube_privacy_status",
                            "upload_post_youtube_privacy_status_selectbox",
                            lambda value: value,
                        ),
                    )
                    if upload_post_youtube_privacy_status != app_settings.get(
                        "upload_post_youtube_privacy_status", "public"
                    ):
                        _set_runtime_config(
                            "app",
                            "upload_post_youtube_privacy_status",
                            upload_post_youtube_privacy_status,
                        )

                    # La declaración de audiencia solo afecta a la publicación en YouTube, sin cambiar el contenido generado ni las peticiones de otras plataformas.
                    # Usa opciones booleanas reales, para no tomar el texto mostrado o cadenas como parámetros de la API.
                    saved_audience = app_settings.get(
                        "upload_post_youtube_made_for_kids", False
                    )
                    audience_labels = {
                        False: tr("Not Made for Kids"),
                        True: tr("Made for Kids"),
                    }
                    made_for_kids = st.selectbox(
                        tr("YouTube Audience"),
                        options=[False, True],
                        # Una configuración ilegal se mantiene sin seleccionar, sin cambiarla por su cuenta a la declaración «no infantil» al abrir los ajustes.
                        index=int(saved_audience)
                        if isinstance(saved_audience, bool)
                        else None,
                        format_func=audience_labels.get,
                        help=tr("YouTube Audience Help"),
                        key="upload_post_youtube_made_for_kids_selectbox",
                        on_change=_save_settings_change,
                        args=(
                            "app",
                            "upload_post_youtube_made_for_kids",
                            "upload_post_youtube_made_for_kids_selectbox",
                            lambda value: value,
                        ),
                    )
                    if isinstance(made_for_kids, bool):
                        _set_runtime_config(
                            "app", "upload_post_youtube_made_for_kids", made_for_kids
                        )

        # Panel izquierdo - ajustes de logs
        with left_config_panel:
            if left_config_panel.open:
                hide_log = st.checkbox(
                    tr("Hide Log"),
                    value=ui_settings.get("hide_log", False),
                    key="hide_log_checkbox",
                    on_change=_save_settings_change,
                    args=("ui", "hide_log", "hide_log_checkbox", lambda value: value),
                )
                _set_runtime_config("ui", "hide_log", hide_log)
                _render_logging_settings()

        if cache_config_panel.open:
            _render_cache_management_settings(cache_config_panel)
        # La restauración de claves escribe la configuración y borra el estado de los controles de contraseña, así que debe ejecutarse antes de renderizar esos controles más abajo.
        if key_backup_panel.open:
            _render_key_backup_settings(key_backup_panel)

        # Panel central - ajustes de LLM

        with middle_config_panel:
            if middle_config_panel.open:
                # El orden del desplegable, el label por defecto y el provider id estable vienen todos del Registry;locale
                # solo se sobrescriben los textos mostrados, y Main.py ya no mantiene una segunda lista de Providers.
                llm_provider_ids = [
                    provider.provider_id for provider in LLM_PROVIDER_REGISTRY
                ]
                llm_provider_labels = {
                    provider.provider_id: get_llm_provider_label(provider)
                    for provider in LLM_PROVIDER_REGISTRY
                }
                saved_llm_provider = app_settings.get(
                    "llm_provider", DEFAULT_LLM_PROVIDER_ID
                ).lower()
                if saved_llm_provider not in llm_provider_ids:
                    saved_llm_provider = DEFAULT_LLM_PROVIDER_ID

                llm_provider = stable_selectbox(
                    tr("LLM Provider"),
                    options=llm_provider_ids,
                    default_value=saved_llm_provider,
                    key="llm_provider_select",
                    format_func=lambda provider_id: llm_provider_labels[provider_id],
                    on_change=_save_settings_change,
                    args=(
                        "app",
                        "llm_provider",
                        localized_widget_key("llm_provider_select"),
                        lambda value: value,
                    ),
                )
                # El formulario de configuración y la descripción del Provider se muestran lado a lado, reduciendo los saltos de línea de las descripciones largas en columnas estrechas,
                # y aprovechando a la vez el espacio horizontal del panel de ajustes básicos.
                llm_form_panel, llm_help_panel = st.columns(
                    [0.9, 1.1],
                    gap="large",
                    vertical_alignment="top",
                )
                llm_helper = llm_help_panel.container()
                _set_runtime_config("app", "llm_provider", llm_provider)
                llm_provider_spec = get_llm_provider(llm_provider)
                if llm_provider_spec is None:
                    # Normalmente todas las opciones del desplegable vienen del Registry y no se entra en esta rama; se conserva
                    # un error explícito para diagnosticar un session state dañado o integraciones olvidadas en el futuro.
                    raise RuntimeError(f"unsupported llm provider: {llm_provider}")

                llm_api_key = app_settings.get(
                    llm_provider_spec.config_key("api_key"), ""
                )
                configured_llm_base_url = app_settings.get(
                    llm_provider_spec.config_key("base_url"), ""
                )
                llm_default_base_url = llm_provider_spec.effective_default_base_url
                llm_base_url = configured_llm_base_url or llm_default_base_url
                llm_model_name = llm_provider_spec.resolve_model_name(
                    app_settings.get(llm_provider_spec.config_key("model_name"), "")
                )

                provider_tip_context = {}
                selected_service_endpoint = None
                if llm_provider_spec.service_endpoints:
                    # Kimi y otros Providers usan sistemas de cuentas distintos en el sitio de China y el internacional. Solo se deja que el usuario
                    # elija la región del servicio, y el Registry sincroniza el acceso de solicitud de API y la Base URL,
                    # evitando combinaciones manuales erróneas. Las configuraciones existentes con Base URL vacía siguen usando el sitio de China, y solo
                    # las configuraciones totalmente nuevas sin Key recomiendan el acceso correspondiente según el idioma de la interfaz.
                    selected_service_endpoint = (
                        llm_provider_spec.select_service_endpoint(
                            configured_llm_base_url,
                            has_api_key=bool(str(llm_api_key).strip()),
                            prefer_international=(
                                st.session_state.get("ui_language", "en") != "zh"
                            ),
                        )
                    )
                    endpoint_options = [
                        endpoint.endpoint_id
                        for endpoint in llm_provider_spec.service_endpoints
                    ] + [CUSTOM_LLM_ENDPOINT_ID]
                    default_endpoint_id = (
                        selected_service_endpoint.endpoint_id
                        if selected_service_endpoint
                        else CUSTOM_LLM_ENDPOINT_ID
                    )
                    endpoint_labels = {
                        endpoint.endpoint_id: (
                            tr_optional(
                                llm_provider_spec.endpoint_label_key(
                                    endpoint.endpoint_id
                                ),
                                fallback_language="en",
                            )
                            or endpoint.default_label
                        )
                        for endpoint in llm_provider_spec.service_endpoints
                    }
                    endpoint_labels[CUSTOM_LLM_ENDPOINT_ID] = (
                        tr_optional("Custom API Endpoint", fallback_language="en")
                        or "Custom API Endpoint"
                    )
                    with llm_form_panel:
                        selected_endpoint_id = stable_selectbox(
                            tr_optional(
                                llm_provider_spec.endpoint_selector_label_key,
                                fallback_language="en",
                            )
                            or tr("API Platform"),
                            options=endpoint_options,
                            default_value=default_endpoint_id,
                            key=f"{llm_provider}_service_endpoint_select",
                            on_change=_save_settings_endpoint,
                            args=(
                                llm_provider_spec,
                                localized_widget_key(
                                    f"{llm_provider}_service_endpoint_select"
                                ),
                            ),
                            format_func=lambda endpoint_id: endpoint_labels[
                                endpoint_id
                            ],
                            help=(
                                tr_optional(
                                    llm_provider_spec.endpoint_selector_help_key,
                                    fallback_language="en",
                                )
                                or None
                            ),
                        )
                    selected_service_endpoint = next(
                        (
                            endpoint
                            for endpoint in llm_provider_spec.service_endpoints
                            if endpoint.endpoint_id == selected_endpoint_id
                        ),
                        None,
                    )
                    if selected_service_endpoint:
                        llm_base_url = selected_service_endpoint.base_url
                        provider_tip_context.update(
                            {
                                "api_key_url": selected_service_endpoint.api_key_url,
                                "default_base_url": selected_service_endpoint.base_url,
                                "model_docs_url": selected_service_endpoint.model_docs_url,
                            }
                        )
                    else:
                        # El modo personalizado solo conserva la dirección que el usuario guardó explícitamente, sin disfrazar una región estándar
                        # de valor personalizado. Si la entrada está vacía, la configuración no se persiste y la próxima vez vuelve al valor por defecto compatible.
                        llm_base_url = str(configured_llm_base_url or "").strip()

                if llm_provider == "ollama":
                    llm_default_base_url = config.get_default_ollama_base_url()
                    if not llm_base_url:
                        llm_base_url = llm_default_base_url
                    docker_hint = ""
                    if config.is_running_in_container():
                        docker_hint = tr_optional(
                            "llm_provider_tips.ollama.docker_hint",
                            fallback_language="en",
                        )
                    provider_tip_context["docker_hint"] = docker_hint

                tips = get_llm_provider_tips(llm_provider, **provider_tip_context)
                if tips:
                    with llm_helper:
                        st.info(tips)

                st_llm_api_key = llm_api_key
                if llm_provider_spec.show_api_key:
                    st_llm_api_key = llm_form_panel.text_input(
                        tr("API Key"),
                        value=llm_api_key,
                        type="password",
                        key=f"{llm_provider}_api_key_input",
                        on_change=_save_settings_change,
                        args=(
                            "app",
                            llm_provider_spec.config_key("api_key"),
                            f"{llm_provider}_api_key_input",
                            lambda value: value,
                        ),
                    )

                st_llm_base_url = llm_base_url
                if llm_provider_spec.show_base_url:
                    st_llm_base_url = llm_form_panel.text_input(
                        tr("Base Url"),
                        value=llm_base_url,
                        key=(
                            f"{llm_provider}_base_url_"
                            f"{selected_service_endpoint.endpoint_id}_input"
                            if selected_service_endpoint
                            else f"{llm_provider}_base_url_custom_input"
                        ),
                        disabled=selected_service_endpoint is not None,
                        on_change=_save_settings_change,
                        args=(
                            "app",
                            llm_provider_spec.config_key("base_url"),
                            f"{llm_provider}_base_url_"
                            f"{selected_service_endpoint.endpoint_id}_input"
                            if selected_service_endpoint
                            else f"{llm_provider}_base_url_custom_input",
                            lambda value, llm_default_base_url=llm_default_base_url: (
                                normalize_provider_override(value, llm_default_base_url)
                            ),
                        ),
                    )
                st_llm_model_name = ""
                if llm_provider == "openai_codex":
                    from webui.codex_models import render_codex_model_selector

                    with llm_form_panel:
                        st_llm_model_name = render_codex_model_selector(
                            llm_model_name,
                            tr,
                            on_change=lambda model, provider=llm_provider_spec: (
                                _set_runtime_config(
                                    "app",
                                    provider.config_key("model_name"),
                                    normalize_provider_override(
                                        model, provider.default_model
                                    ),
                                )
                            ),
                        )
                elif llm_provider == "openai_chatgpt":
                    from webui.chatgpt_login import render_chatgpt_login
                    from webui.chatgpt_models import render_chatgpt_model_selector

                    with llm_form_panel:
                        if middle_config_panel.open:
                            # Keep the nested polling fragment's delta paths separate
                            # from the model selector and connection-test controls.
                            with st.container(key="chatgpt_login_panel"):
                                render_chatgpt_login(tr)
                        st_llm_model_name = render_chatgpt_model_selector(
                            llm_model_name,
                            tr,
                            on_change=lambda model, provider=llm_provider_spec: (
                                _set_runtime_config(
                                    "app",
                                    provider.config_key("model_name"),
                                    normalize_provider_override(
                                        model, provider.default_model
                                    ),
                                )
                            ),
                        )
                elif llm_provider == "groq":
                    effective_api_key = st_llm_api_key or llm_api_key
                    effective_base_url = st_llm_base_url or llm_base_url
                    groq_models = get_groq_model_ids(
                        api_key=effective_api_key,
                        base_url=effective_base_url,
                    )

                    if groq_models:
                        selected_index = 0
                        if llm_model_name in groq_models:
                            selected_index = groq_models.index(llm_model_name)

                        st_llm_model_name = llm_form_panel.selectbox(
                            tr("Model Name"),
                            options=groq_models,
                            index=selected_index,
                            key="groq_model_name_select",
                            on_change=_save_settings_change,
                            args=(
                                "app",
                                llm_provider_spec.config_key("model_name"),
                                "groq_model_name_select",
                                lambda value, llm_provider_spec=llm_provider_spec: (
                                    normalize_provider_override(
                                        value, llm_provider_spec.default_model
                                    )
                                ),
                            ),
                        )
                    else:
                        st_llm_model_name = llm_form_panel.text_input(
                            tr("Model Name"),
                            value=llm_model_name,
                            key="groq_model_name_input",
                            on_change=_save_settings_change,
                            args=(
                                "app",
                                llm_provider_spec.config_key("model_name"),
                                "groq_model_name_input",
                                lambda value, llm_provider_spec=llm_provider_spec: (
                                    normalize_provider_override(
                                        value, llm_provider_spec.default_model
                                    )
                                ),
                            ),
                        )
                        if effective_api_key:
                            llm_form_panel.caption(tr("Groq Model List Load Failed"))
                        else:
                            llm_form_panel.caption(
                                tr("Groq API Key Required for Model List")
                            )
                else:
                    st_llm_model_name = llm_form_panel.text_input(
                        tr("Model Name"),
                        value=llm_model_name,
                        key=f"{llm_provider}_model_name_input",
                        on_change=_save_settings_change,
                        args=(
                            "app",
                            llm_provider_spec.config_key("model_name"),
                            f"{llm_provider}_model_name_input",
                            lambda value, llm_provider_spec=llm_provider_spec: (
                                normalize_provider_override(
                                    value, llm_provider_spec.default_model
                                )
                            ),
                        ),
                    )
                # El campo muestra el valor por defecto del Registry, pero la configuración solo guarda las sobrescrituras reales del usuario.
                # Así, cuando se actualicen el modelo por defecto o la Base URL, los usuarios que no los personalizaron los seguirán automáticamente.
                _set_runtime_config(
                    "app",
                    llm_provider_spec.config_key("api_key"),
                    st_llm_api_key,
                )
                _set_runtime_config(
                    "app",
                    llm_provider_spec.config_key("base_url"),
                    normalize_provider_override(
                        st_llm_base_url,
                        llm_default_base_url,
                    ),
                )
                _set_runtime_config(
                    "app",
                    llm_provider_spec.config_key("model_name"),
                    normalize_provider_override(
                        st_llm_model_name,
                        llm_provider_spec.default_model,
                    ),
                )

                # Provider Los campos específicos también los declara el Registry. Por ejemplo, Cloudflare AI Gateway
                # necesita Account ID; al añadir campos similares en el futuro no hará falta añadir comprobaciones en Main.py.
                for field in llm_provider_spec.extra_fields:
                    field_config_key = llm_provider_spec.config_key(field.config_suffix)
                    field_value = llm_form_panel.text_input(
                        tr(field.label_key),
                        value=(
                            app_settings.get(field_config_key, "")
                            or field.default_value
                        ),
                        type="password" if field.secret else "default",
                        key=f"{llm_provider}_{field.config_suffix}_input",
                        on_change=_save_settings_change,
                        args=(
                            "app",
                            field_config_key,
                            f"{llm_provider}_{field.config_suffix}_input",
                            lambda value, field=field: normalize_provider_override(
                                value, field.default_value
                            ),
                        ),
                    )
                    _set_runtime_config(
                        "app",
                        field_config_key,
                        normalize_provider_override(
                            field_value,
                            field.default_value,
                        ),
                    )

                if llm_provider == "openai_codex" and middle_config_panel.open:
                    from webui.codex_login import render_codex_login

                    with llm_form_panel.container():
                        render_codex_login(tr)

                if llm_provider == "kimi_code" and middle_config_panel.open:
                    from app.services import kimi_code_oauth
                    from webui.kimi_login import render_kimi_login

                    # Da al fragment anidado un contenedor propio, para que el redibujado por sondeo no ocupe la posición de los controles siguientes.
                    with llm_form_panel.container():
                        render_kimi_login(
                            kimi_code_oauth.resolve_region(st_llm_base_url), tr
                        )

                if llm_form_panel.button(
                    tr("Test LLM Connection"),
                    key="test_llm_connection_button",
                    use_container_width=True,
                    type="secondary",
                    icon=":material/network_check:",
                ):
                    with config.try_runtime_config_lock() as lock_acquired:
                        if not lock_acquired:
                            llm_form_panel.warning(tr("Runtime Configuration Busy"))
                        else:
                            with llm_form_panel.spinner(tr("Testing LLM Connection")):
                                connection_ok, connection_error, connection_elapsed = (
                                    llm.test_connection()
                                )

                    if not lock_acquired:
                        connection_ok = None
                    elif connection_ok:
                        llm_form_panel.success(
                            tr("LLM Connection Test Succeeded").format(
                                provider=llm_provider_labels[llm_provider],
                                model=st_llm_model_name or "-",
                                elapsed=f"{connection_elapsed:.2f}",
                            )
                        )
                    else:
                        connection_error = format_llm_connection_error(
                            llm_provider,
                            st_llm_base_url,
                            connection_error,
                        )
                        llm_form_panel.error(
                            tr("LLM Connection Test Failed").format(
                                error=connection_error
                            )
                        )

        # Panel derecho - ajustes de claves de API
        with right_config_panel:
            if right_config_panel.open:
                if "settings_material_group" not in st.session_state:
                    _initialize_material_navigation("material", app_settings)
                sections = (
                    (
                        "ai_video",
                        "AI Video Generation APIs",
                        "AI Video Generation APIs Help",
                    ),
                    (
                        "ai_image",
                        "AI Image Generation APIs",
                        "AI Image Generation APIs Help",
                    ),
                    ("stock_video", "Stock Video APIs", "Stock Video APIs Help"),
                )
                remembered = dict(
                    st.session_state.get("settings_material_providers", {})
                )
                for group, title_key, help_key in sections:
                    providers = VIDEO_SOURCE_GROUPS[group]
                    default_provider = remembered.get(group, providers[0])
                    if default_provider not in providers:
                        default_provider = providers[0]
                    provider_key = localized_widget_key(
                        f"settings_material_provider_{group}_select"
                    )
                    st.session_state[provider_key] = default_provider
                    with st.container(border=True):
                        st.subheader(tr(title_key))
                        st.caption(tr(help_key))
                        provider = st.selectbox(
                            tr(title_key),
                            options=providers,
                            key=provider_key,
                            format_func=_material_provider_labels().get,
                            label_visibility="collapsed",
                            on_change=_remember_material_provider,
                            args=(group, provider_key),
                        )
                        remembered[group] = provider
                        _render_material_provider_settings(provider)
                st.session_state["settings_material_providers"] = remembered

    _save_runtime_config()


# -----------------------------------------------------------------------------
# Formulario principal de generación: paneles de texto, video, audio y subtítulos
# -----------------------------------------------------------------------------


def _create_loomloom_script_backend():
    """Crea el cliente de texto por lotes a partir de la configuración actual de la WebUI/config.toml."""
    app_config_snapshot = config.snapshot_config_with_pending(config.app)
    settings = loomloom.LoomLoomSettings.from_mapping(app_config_snapshot)
    return loomloom.LoomLoomScriptBackend(settings)


def _create_loomloom_video_backend():
    """Crea el cliente de video con el SkillBot por defecto del proyecto y las credenciales vigentes."""
    app_config_snapshot = config.snapshot_config_with_pending(config.app)
    settings = loomloom.video_settings_from_mapping(app_config_snapshot)
    return loomloom.LoomLoomVideoBackend(settings)


def _effective_loomloom_api_token():
    """Lee la API Key de Shengsuanyun aún no guardada por la WebUI o la de config.toml."""
    app_config_snapshot = config.snapshot_config_with_pending(config.app)
    return loomloom.resolve_api_token(app_config_snapshot)


def _effective_script_generation_backend():
    """Lee el modo de generación de texto incluyendo los cambios de la WebUI pendientes de guardar."""
    app_config_snapshot = config.snapshot_config_with_pending(config.app)
    backend = str(
        app_config_snapshot.get("script_generation_backend", "local") or "local"
    ).strip()
    return backend if backend in {"local", "loomloom"} else "local"


def _script_generation_method_help(selected_backend):
    """Hace que el contenido del signo de interrogación de «Modo de generación de texto» siga estrictamente la elección actual."""
    if selected_backend != "loomloom":
        return tr("Script Generation Method Help")

    app_config_snapshot = config.snapshot_config_with_pending(config.app)
    guidance = [tr("LoomLoom Batch Script Generation Help")]
    if (
        str(app_config_snapshot.get("llm_provider", "") or "").strip().lower()
        == "shengsuanyun"
    ):
        guidance.append(tr("Shengsuan Cloud API Key Reused"))
    guidance.append(tr("Shengsuan Cloud API Key Link"))
    return "\n\n".join(guidance)


def _loomloom_video_scene_prompts(video_terms, subject, scene_count):
    """Genera un número limitado de descripciones de escena según las palabras clave del material, para que el modelo de video genere el material tramo a tramo."""
    if isinstance(video_terms, str):
        terms = [
            term.strip() for term in re.split(r"[,，\n]", video_terms) if term.strip()
        ]
    elif isinstance(video_terms, list):
        terms = [
            str(term or "").strip() for term in video_terms if str(term or "").strip()
        ]
    else:
        terms = []
    fallback = str(subject or "").strip()
    if not terms and fallback:
        terms = [fallback]
    if not terms:
        return ()
    return tuple(
        (
            terms[index % len(terms)]
            if index < len(terms)
            else f"{terms[index % len(terms)]}; alternative camera angle {index + 1}"
        )
        for index in range(int(scene_count))
    )


def _loomloom_video_signature(batch, credential_fingerprint):
    """Incluye en la firma todas las entradas facturables y el resumen de credenciales; si cambian los parámetros se fuerza una nueva cotización."""
    payload = {
        "inputRows": [dict(row) for row in batch.input_rows],
        "credentialFingerprint": str(credential_fingerprint or "").strip(),
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _loomloom_video_account_signature(token):
    """La dirección del servicio y las credenciales aíslan juntas el catálogo de modelos y la cotización; no se puede reutilizar un estado confirmado entre endpoints."""
    values = config.snapshot_config_with_pending(config.app)
    base_url = str(values.get("loomloom_base_url") or loomloom.DEFAULT_BASE_URL).strip().rstrip("/")
    payload = json.dumps([base_url, str(token or "").strip()])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_loomloom_video_capability(token, *, force=False):
    """Cachea el Profile por credencial actual; si falla la actualización se conserva el último resultado exitoso de la misma credencial."""
    normalized_token = str(token or "").strip()
    if not normalized_token:
        return None

    fingerprint = _loomloom_video_account_signature(normalized_token)
    if st.session_state.get("loomloom_video_capability_fingerprint") != fingerprint:
        st.session_state["loomloom_video_capability"] = None
        st.session_state["loomloom_video_capability_fingerprint"] = fingerprint
        st.session_state["loomloom_video_capability_load_attempt"] = ""
        st.session_state["loomloom_video_capability_error"] = ""

    should_load = force or (
        st.session_state.get("loomloom_video_capability_load_attempt") != fingerprint
    )
    if should_load:
        st.session_state["loomloom_video_capability_load_attempt"] = fingerprint
        try:
            capability = _create_loomloom_video_backend().resolve_video_capability()
        except (loomloom.LoomLoomError, ValueError) as exc:
            logger.warning(
                f"failed to load LoomLoom video capability: error={type(exc).__name__}"
            )
            st.session_state["loomloom_video_capability_error"] = str(exc)
        else:
            st.session_state["loomloom_video_capability"] = capability
            st.session_state["loomloom_video_capability_error"] = ""

    capability = st.session_state.get("loomloom_video_capability")
    return (
        capability if isinstance(capability, loomloom.LoomLoomVideoCapability) else None
    )


def _normalize_loomloom_model_identifier(value):
    """Unifica separadores y mayúsculas de los nombres mostrados y de los IDs de modelo, para emparejarlos con seguridad con la tabla de precios local."""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())


def _loomloom_video_model_price(model):
    """Devuelve, para modelos conocidos, (precio corto del desplegable, precio de referencia completo al elegirlo); los modelos desconocidos devuelven vacío."""
    identifiers = {
        _normalize_loomloom_model_identifier(model.model_id),
        _normalize_loomloom_model_identifier(model.display_name),
    }
    for aliases, compact_price, detailed_price in LOOMLOOM_VIDEO_MODEL_PRICES:
        if identifiers.intersection(aliases):
            return compact_price, detailed_price
    return "", ""


def _format_loomloom_video_model_option(model):
    """Muestra el precio corto a la derecha del nombre del modelo, para que los precios de varias resoluciones no ensanchen demasiado el desplegable."""
    compact_price, _ = _loomloom_video_model_price(model)
    return f"{model.display_name} · {compact_price}" if compact_price else model.display_name


def _effective_voice_rate_before_audio_panel():
    """El panel de video está antes que el de audio, así que la velocidad actual debe leerse del estado de controles existente o de la configuración."""
    raw_rate = st.session_state.get(
        localized_widget_key("voice_rate_select"),
        config.ui.get("voice_rate", 1.0),
    )
    try:
        rate = float(raw_rate)
    except (TypeError, ValueError, OverflowError):
        return 1.0
    return rate if math.isfinite(rate) and rate > 0 else 1.0


def _matching_full_voice_preview_duration(script, voice_rate):
    """Solo se usa la duración real de la vista previa completa si no cambiaron ni el texto, ni el Provider, ni la voz, ni la velocidad."""
    cached = st.session_state.get("voice_preview_audio")
    if not isinstance(cached, dict) or cached.get("preview_type") != "full":
        return None
    script_digest = hashlib.sha256(str(script or "").encode("utf-8")).hexdigest()
    if cached.get("content_digest") != script_digest:
        return None

    current_tts_server = st.session_state.get(
        localized_widget_key("tts_server_select"),
        config.ui.get("tts_server", "azure-tts-v1"),
    )
    current_voice_name = st.session_state.get(
        localized_widget_key(f"speech_synthesis_select_{current_tts_server}"),
        config.ui.get("voice_name", ""),
    )
    try:
        cached_voice_rate = float(cached.get("voice_rate", 0))
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        cached.get("tts_server") != current_tts_server
        or cached.get("voice_name") != current_voice_name
        or not math.isfinite(cached_voice_rate)
        or not math.isclose(cached_voice_rate, voice_rate)
    ):
        return None

    duration = cached.get("duration")
    if (
        not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        return None
    return float(duration)


def _loomloom_video_coverage_plan(params):
    """Recomienda el número de materiales según la duración real o estimada de la narración; lo que falte lo sigue completando la lógica de bucle original."""
    script = str(params.video_script or "").strip()
    if not script:
        return None

    voice_rate = _effective_voice_rate_before_audio_panel()
    actual_duration = _matching_full_voice_preview_duration(script, voice_rate)
    if actual_duration is not None:
        duration_min = duration_max = actual_duration
        basis_key = "AI Video Duration Basis Actual"
    else:
        estimated = _estimate_voiceover_duration_range(script, voice_rate)
        if not estimated:
            return None
        duration_min, duration_max = estimated
        basis_key = "AI Video Duration Basis Estimated"

    clip_duration = max(float(params.video_clip_duration or 1), 1.0)
    needed_min = max(math.ceil(duration_min / clip_duration), 1)
    needed_max = max(math.ceil(duration_max / clip_duration), needed_min)
    return {
        "script_digest": hashlib.sha256(script.encode("utf-8")).hexdigest(),
        "basis_key": basis_key,
        "duration_min": float(duration_min),
        "duration_max": float(duration_max),
        "clip_duration": clip_duration,
        "needed_min": needed_min,
        "needed_max": needed_max,
        # El valor recomendado prioriza cubrir el límite superior conservador, pero nunca supera el máximo de tareas de pago que permite el servidor.
        "recommended_count": min(needed_max, loomloom.MAX_VIDEO_SCENES),
    }


def _format_numeric_range(minimum, maximum, digits=1):
    if math.isclose(float(minimum), float(maximum)):
        return f"{float(maximum):.{digits}f}"
    return f"{float(minimum):.{digits}f}–{float(maximum):.{digits}f}"


def _selected_loomloom_video_model(capability):
    """Devuelve la elección del usuario si sigue entre los candidatos del Profile actual, sin degradarla en silencio."""
    selected_model_id = str(
        st.session_state.get("loomloom_video_model_id", "") or ""
    ).strip()
    eligible_model_ids = {model.model_id for model in capability.models}
    return selected_model_id if selected_model_id in eligible_model_ids else ""


def _current_loomloom_video_quote_context(params):
    """Construye el lote de cotización de video del SkillBot por defecto según los parámetros actuales de la página."""
    token = _effective_loomloom_api_token()
    fingerprint = _loomloom_video_account_signature(token) if token else ""
    capability = st.session_state.get("loomloom_video_capability")
    if (
        not isinstance(capability, loomloom.LoomLoomVideoCapability)
        or st.session_state.get("loomloom_video_capability_fingerprint") != fingerprint
    ):
        return None, ""
    model_id = _selected_loomloom_video_model(capability)
    scene_count = int(st.session_state.get("loomloom_video_scene_count", 1) or 1)
    prompts = _loomloom_video_scene_prompts(
        params.video_terms,
        params.video_subject or params.video_script,
        scene_count,
    )
    aspect_ratio = str(
        params.video_aspect.value
        if isinstance(params.video_aspect, VideoAspect)
        else params.video_aspect
    )
    if (
        not token
        or not model_id
        or not prompts
        or aspect_ratio not in capability.aspect_ratios
    ):
        return None, ""
    try:
        batch = _create_loomloom_video_backend().prepare_video_batch(
            subject=params.video_subject or params.video_script,
            scene_prompts=prompts,
            model_id=model_id,
            aspect_ratio=aspect_ratio,
        )
    except (loomloom.LoomLoomError, ValueError):
        return None, ""
    return batch, _loomloom_video_signature(batch, fingerprint)


def _retry_loomloom_video_quote():
    """Cuando el usuario reintenta por su cuenta se quita el bloqueo de fallo; no se reutiliza la confirmación de pago anterior."""
    st.session_state["loomloom_video_quote_error_signature"] = ""
    st.session_state["loomloom_video_quote_error"] = ""
    st.session_state["loomloom_video_confirm_charge"] = False


def _render_loomloom_video_settings(params):
    """Renderiza el flujo de cotización, caducidad de la cotización y confirmación de pago del SkillBot de video por defecto."""
    st.caption(tr("Shengsuan Cloud AI Video Help"))
    if (
        str(
            config.snapshot_config_with_pending(config.app).get("llm_provider", "")
            or ""
        ).lower()
        == "shengsuanyun"
    ):
        st.caption(tr("Shengsuan Cloud API Key Reused"))

    token = _effective_loomloom_api_token()

    refresh_models = st.button(
        tr("Refresh AI Video Models"),
        key="loomloom_refresh_video_models",
        use_container_width=True,
        disabled=not token,
    )
    capability = _load_loomloom_video_capability(token, force=refresh_models)
    capability_error = str(
        st.session_state.get("loomloom_video_capability_error", "") or ""
    ).strip()
    if capability_error:
        st.warning(tr("AI Video Model List Load Failed").format(error=capability_error))

    if capability is not None:
        models_by_id = {model.model_id: model for model in capability.models}
        selected_model_id = str(
            st.session_state.get("loomloom_video_model_id", "") or ""
        ).strip()
        if not selected_model_id:
            selected_model_id = capability.default_model_id
            st.session_state["loomloom_video_model_id"] = selected_model_id

        model_options = list(models_by_id)
        if selected_model_id not in models_by_id:
            # Conserva la elección original ya caducada, para que el usuario vea claramente el estado y vuelva a elegir. Cambiar directamente el control
            # al nuevo modelo por defecto haría que la cotización antigua no coincidiera con lo que el usuario cree.
            model_options.insert(0, selected_model_id)

        selected_model_id = stable_selectbox(
            tr("AI Video Model"),
            options=model_options,
            default_value=selected_model_id,
            key="loomloom_video_model_select",
            format_func=lambda model_id: (
                _format_loomloom_video_model_option(models_by_id[model_id])
                if model_id in models_by_id
                else tr("Unavailable AI Video Model").format(model=model_id)
            ),
        )
        st.session_state["loomloom_video_model_id"] = selected_model_id
        if selected_model_id not in models_by_id:
            st.error(tr("Selected AI Video Model Unavailable"))
        else:
            _, detailed_price = _loomloom_video_model_price(
                models_by_id[selected_model_id]
            )
            if detailed_price:
                st.caption(
                    tr("AI Video Model Reference Price").format(price=detailed_price)
                )

        current_aspect_ratio = str(
            params.video_aspect.value
            if isinstance(params.video_aspect, VideoAspect)
            else params.video_aspect
        )
        if current_aspect_ratio not in capability.aspect_ratios:
            st.error(tr("Selected AI Video Ratio Unavailable"))

    coverage_plan = _loomloom_video_coverage_plan(params)
    pending_autofill_digest = str(
        st.session_state.get("loomloom_video_scene_autofill_digest", "") or ""
    )
    if (
        coverage_plan is not None
        and pending_autofill_digest == coverage_plan["script_digest"]
    ):
        # Solo se recomienda una vez, «justo tras generar el texto» o «justo tras obtener la duración real de la vista previa completa».
        # Tras consumir la marca ya no se sobrescribe, y los ajustes manuales posteriores del número de tramos se conservan por completo.
        st.session_state["loomloom_video_scene_count"] = coverage_plan[
            "recommended_count"
        ]
        st.session_state["loomloom_video_scene_autofill_digest"] = ""

    scene_count = st.number_input(
        tr("AI Video Scene Count"),
        min_value=1,
        max_value=loomloom.MAX_VIDEO_SCENES,
        step=1,
        key="loomloom_video_scene_count",
    )
    _set_runtime_config("ui", "loomloom_video_scene_count", int(scene_count))
    if coverage_plan is not None:
        coverage_seconds = int(scene_count) * coverage_plan["clip_duration"]
        shortfall_min = max(
            coverage_plan["duration_min"] - coverage_seconds, 0.0
        )
        shortfall_max = max(
            coverage_plan["duration_max"] - coverage_seconds, 0.0
        )
        duration_basis = tr(coverage_plan["basis_key"]).format(
            duration=_format_numeric_range(
                coverage_plan["duration_min"], coverage_plan["duration_max"]
            )
        )
        coverage_message = tr("AI Video Material Coverage").format(
            basis=duration_basis,
            clip=_format_numeric_range(
                coverage_plan["clip_duration"], coverage_plan["clip_duration"]
            ),
            needed=_format_numeric_range(
                coverage_plan["needed_min"], coverage_plan["needed_max"], digits=0
            ),
            count=int(scene_count),
            coverage=_format_numeric_range(coverage_seconds, coverage_seconds),
            shortfall=_format_numeric_range(shortfall_min, shortfall_max),
        )
        if shortfall_max > 0:
            st.warning(coverage_message)
        else:
            st.caption(coverage_message)

    batch, input_signature = _current_loomloom_video_quote_context(params)
    if not token:
        st.warning(tr("Shengsuan Cloud API Key Required"))

    quote_result = st.session_state.get("loomloom_video_quote")
    quoted_batch = st.session_state.get("loomloom_video_batch")
    quote_is_current = bool(
        quote_result is not None
        and quoted_batch is not None
        and st.session_state.get("loomloom_video_input_signature") == input_signature
    )
    # Tras un fallo con el mismo conjunto de parámetros se pausan las peticiones automáticas, para que las interacciones normales no esperen repetidamente el tiempo de espera del servicio.
    # La firma incluye la cuenta, el endpoint y todas las entradas facturables; se vuelve a cotizar si cambian los parámetros o el usuario reintenta.
    if st.session_state.get("loomloom_video_quote_error_signature") != input_signature:
        st.session_state["loomloom_video_quote_error_signature"] = ""
        st.session_state["loomloom_video_quote_error"] = ""
    quote_failed = bool(st.session_state.get("loomloom_video_quote_error"))
    # Quote No crea tareas de pago; la ejecución real sigue requiriendo que el usuario marque explícitamente la confirmación.
    if (token and batch is not None and not quote_is_current and not quote_failed
            and not _generation_is_busy(_current_generation_task_id())):
        st.session_state["loomloom_video_confirm_charge"] = False
        try:
            quote_result = _create_loomloom_video_backend().quote(batch)
        except (loomloom.LoomLoomError, ValueError) as exc:
            logger.warning(f"failed to quote LoomLoom videos: error={exc}")
            st.session_state["loomloom_video_quote_error_signature"] = input_signature
            st.session_state["loomloom_video_quote_error"] = str(exc) or type(exc).__name__
        else:
            st.session_state["loomloom_video_batch"] = batch
            st.session_state["loomloom_video_quote"] = quote_result
            st.session_state["loomloom_video_input_signature"] = input_signature
            st.session_state["loomloom_video_client_request_id"] = (
                f"mpt-video-{uuid4()}"
            )
            st.session_state["loomloom_video_confirm_charge"] = False
            logger.info(
                "LoomLoom video quote ready: "
                f"tasks={quote_result.task_count}, currency={quote_result.currency}, "
                f"estimated_payable_t={quote_result.estimated_buyer_payable_t}"
            )

    if st.session_state.get("loomloom_video_quote_error"):
        st.error(st.session_state["loomloom_video_quote_error"])
        st.button(
            tr("Retry AI Video Quote"),
            key="loomloom_retry_video_quote",
            on_click=_retry_loomloom_video_quote,
        )

    quote_result = st.session_state.get("loomloom_video_quote")
    quoted_batch = st.session_state.get("loomloom_video_batch")
    if quote_result is not None and quoted_batch is not None:
        display_amount = (
            quote_result.estimated_buyer_payable_amount
            or f"{quote_result.estimated_buyer_payable_t} T"
        )
        if quote_result.estimated_buyer_payable_t == 0:
            st.warning(tr("AI Video Quote Estimate Incomplete"))
        else:
            st.success(
                tr(
                    "AI Video Quote Summary Singular"
                    if quote_result.task_count == 1
                    else "AI Video Quote Summary"
                ).format(
                    tasks=quote_result.task_count,
                    amount=display_amount,
                    currency=quote_result.currency,
                )
            )
        quote_is_current = (
            st.session_state.get("loomloom_video_input_signature") == input_signature
        )
        if not quote_is_current:
            st.warning(tr("LoomLoom Quote Changed Warning"))
        st.checkbox(
            tr("Confirm AI Video Charge"),
            key="loomloom_video_confirm_charge",
            help=tr("Confirm AI Video Charge Help"),
            disabled=not quote_is_current,
        )


def _loomloom_script_signature(
    *,
    subject,
    language,
    candidate_count,
    duration_seconds,
    style,
    credential_fingerprint,
):
    payload = {
        "subject": str(subject or "").strip(),
        "language": str(language or "auto").strip() or "auto",
        "candidateCount": int(candidate_count),
        "durationSeconds": int(duration_seconds),
        "style": str(style or "").strip(),
        "credentialFingerprint": str(credential_fingerprint or "").strip(),
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _render_local_script_generation(params):
    """Conserva la ruta original de generación de guion con LLM local de MoneyPrinterTurbo."""
    if not st.button(
        tr("Generate Video Script and Keywords"),
        key="auto_generate_script",
        use_container_width=True,
        type="secondary",
        icon=":material/auto_awesome:",
    ):
        return

    if not params.video_subject:
        st.toast(tr("Please Enter the Video Subject First"))
        st.warning(tr("Please Enter the Video Subject First"))
        return

    with st.spinner(tr("Generating Video Script and Keywords")):

        def generate_script_and_terms(app_config_snapshot):
            script = llm.generate_script(
                video_subject=params.video_subject,
                language=params.video_language,
                paragraph_number=params.paragraph_number,
                video_script_prompt=params.video_script_prompt,
                custom_system_prompt=params.custom_system_prompt,
                app_config=app_config_snapshot,
            )
            terms = llm.generate_terms(
                params.video_subject,
                script,
                amount=8 if params.match_materials_to_script else 5,
                match_script_order=params.match_materials_to_script,
                **({"real_people": True} if params.video_source == "wikimedia" else {}),
                app_config=app_config_snapshot,
            )
            return script, terms

        script, terms = _run_llm_read_operation(
            "generate_script_and_terms",
            generate_script_and_terms,
        )
        if "Error: " in script:
            st.error(tr(script))
        elif "Error: " in terms:
            st.error(tr(terms))
        else:
            st.session_state["video_script"] = script
            st.session_state["video_terms"] = ", ".join(terms)
            st.session_state["loomloom_video_scene_autofill_digest"] = (
                hashlib.sha256(script.strip().encode("utf-8")).hexdigest()
            )


def _render_loomloom_candidates():
    candidates = tuple(st.session_state.get("loomloom_script_candidates") or ())
    errors = tuple(st.session_state.get("loomloom_candidate_errors") or ())
    if errors:
        st.warning(
            tr("LoomLoom Candidate Errors").format(
                count=len(errors),
                details="; ".join(
                    f"#{error.row_index + 1}: {error.message}" for error in errors
                ),
            )
        )
    if not candidates:
        return

    selected_index = st.radio(
        tr("Choose Script Candidate"),
        options=list(range(len(candidates))),
        key="loomloom_selected_candidate",
        format_func=lambda index: (
            f"#{candidates[index].row_index + 1} {candidates[index].script[:80]}"
        ),
    )
    selected = candidates[selected_index]
    st.code(selected.script, language=None, wrap_lines=True)
    st.caption(", ".join(selected.video_terms))
    if st.button(
        tr("Use Selected Candidate"),
        key="loomloom_apply_candidate",
        type="primary",
        use_container_width=True,
    ):
        st.session_state["video_script"] = selected.script
        st.session_state["video_terms"] = ", ".join(selected.video_terms)
        # Igual que la generación de texto con el modelo grande normal: tras aplicar un candidato nuevo solo se recomienda una vez el número de materiales.
        st.session_state["loomloom_video_scene_autofill_digest"] = (
            hashlib.sha256(selected.script.strip().encode("utf-8")).hexdigest()
        )
        st.toast(tr("LoomLoom Candidate Applied"))


def _handle_loomloom_poll_error(run_id, exc):
    """Aplica una espera limitada a los errores de sondeo de la tarea de guion, y detiene el sondeo de inmediato ante errores deterministas."""
    logger.warning(f"failed to poll LoomLoom run: run_id={run_id}, error={exc}")
    failure_count = int(st.session_state.get("loomloom_poll_failure_count", 0) or 0) + 1
    retryable = isinstance(exc, loomloom.LoomLoomAPIError) and exc.retryable
    if not retryable or failure_count >= LOOMLOOM_MAX_POLL_FAILURES:
        st.session_state["loomloom_run_error"] = str(exc)
        st.session_state["loomloom_poll_failure_count"] = 0
        st.session_state["loomloom_poll_retry_after"] = 0.0
        # Un fallo de consulta no equivale a un fallo de la tarea de pago remota. Se conserva run_id y se pausa el sondeo automático, para que el usuario
        # pueda seguir consultando la misma tarea; si se descartara el ID y se reenviara, podría pagarse por duplicado.
        st.session_state["loomloom_poll_paused"] = True
        st.rerun(scope="app")
        return

    retry_delay = min(2**failure_count, 30)
    st.session_state["loomloom_poll_failure_count"] = failure_count
    st.session_state["loomloom_poll_retry_after"] = time.monotonic() + retry_delay
    st.warning(
        tr("LoomLoom Poll Retry Warning").format(
            attempt=failure_count,
            max_attempts=LOOMLOOM_MAX_POLL_FAILURES,
        )
    )


@st.fragment(run_every="2s")
def _render_loomloom_run_progress():
    run_id = str(st.session_state.get("loomloom_run_id", "") or "").strip()
    if not run_id or st.session_state.get("loomloom_poll_paused", False):
        return
    retry_after = float(st.session_state.get("loomloom_poll_retry_after", 0.0) or 0.0)
    retry_wait_seconds = max(0, int(math.ceil(retry_after - time.monotonic())))
    if retry_wait_seconds > 0:
        st.info(
            tr("LoomLoom Poll Retry Pending").format(
                seconds=retry_wait_seconds,
            )
        )
        return
    try:
        backend = _create_loomloom_script_backend()
        run = backend.get_run(run_id)
    except loomloom.LoomLoomError as exc:
        _handle_loomloom_poll_error(run_id, exc)
        return

    st.session_state["loomloom_run_status"] = run.status
    if run.status == "completed":
        try:
            result = backend.get_script_results(run_id)
        except loomloom.LoomLoomError as exc:
            _handle_loomloom_poll_error(run_id, exc)
            return
        st.session_state["loomloom_poll_failure_count"] = 0
        st.session_state["loomloom_poll_retry_after"] = 0.0
        st.session_state["loomloom_poll_paused"] = False
        st.session_state["loomloom_script_candidates"] = result.candidates
        st.session_state["loomloom_candidate_errors"] = result.errors
        st.session_state["loomloom_selected_candidate"] = 0
        st.session_state["loomloom_run_id"] = ""
        st.rerun(scope="app")
        return
    if run.status in {"failed", "cancelled", "canceled"}:
        st.session_state["loomloom_run_error"] = run.first_error_message or run.status
        st.session_state["loomloom_run_id"] = ""
        st.session_state["loomloom_poll_paused"] = False
        st.rerun(scope="app")
        return

    st.session_state["loomloom_poll_failure_count"] = 0
    st.session_state["loomloom_poll_retry_after"] = 0.0
    st.info(
        tr("LoomLoom Run Progress").format(
            completed=run.completed_tasks,
            total=run.total_tasks,
        )
    )


def _render_loomloom_script_generation(params):
    st.caption(tr("LoomLoom Batch Script Generation Help"))
    effective_token = _effective_loomloom_api_token()
    if not effective_token:
        st.warning(tr("Shengsuan Cloud API Key Required"))

    candidate_col, duration_col = st.columns(2)
    candidate_count = candidate_col.number_input(
        tr("Script Candidate Count"),
        min_value=1,
        max_value=loomloom.MAX_SCRIPT_CANDIDATES,
        step=1,
        key="loomloom_candidate_count",
    )
    duration_seconds = duration_col.number_input(
        tr("Target Script Duration Seconds"),
        min_value=10,
        max_value=600,
        step=10,
        key="loomloom_script_duration_seconds",
    )
    _set_runtime_config("ui", "loomloom_candidate_count", int(candidate_count))
    _set_runtime_config(
        "ui", "loomloom_script_duration_seconds", int(duration_seconds)
    )
    input_signature = _loomloom_script_signature(
        subject=params.video_subject,
        language=params.video_language,
        candidate_count=candidate_count,
        duration_seconds=duration_seconds,
        style=params.video_script_prompt,
        credential_fingerprint=(
            hashlib.sha256(effective_token.encode("utf-8")).hexdigest()
            if effective_token
            else ""
        ),
    )

    if st.button(
        tr("Get LoomLoom Quote"),
        key="loomloom_quote_scripts",
        use_container_width=True,
        type="secondary",
        icon=":material/request_quote:",
        disabled=not effective_token or bool(st.session_state.get("loomloom_run_id")),
    ):
        if not params.video_subject:
            st.toast(tr("Please Enter the Video Subject First"))
            st.warning(tr("Please Enter the Video Subject First"))
        else:
            try:
                backend = _create_loomloom_script_backend()
                batch = backend.prepare_script_batch(
                    subject=params.video_subject,
                    candidate_count=int(candidate_count),
                    language=params.video_language,
                    duration_seconds=int(duration_seconds),
                    style=params.video_script_prompt,
                )
                quote_result = backend.quote(batch)
            except (loomloom.LoomLoomError, ValueError) as exc:
                logger.warning(f"failed to quote LoomLoom scripts: error={exc}")
                st.error(str(exc))
            else:
                st.session_state["loomloom_script_batch"] = batch
                st.session_state["loomloom_script_quote"] = quote_result
                st.session_state["loomloom_script_input_signature"] = input_signature
                st.session_state["loomloom_client_request_id"] = f"mpt-{uuid4()}"
                st.session_state["loomloom_run_id"] = ""
                st.session_state["loomloom_run_status"] = "quoted"
                st.session_state["loomloom_run_error"] = ""
                st.session_state["loomloom_poll_failure_count"] = 0
                st.session_state["loomloom_poll_retry_after"] = 0.0
                st.session_state["loomloom_poll_paused"] = False
                st.session_state["loomloom_script_candidates"] = ()
                st.session_state["loomloom_candidate_errors"] = ()
                st.session_state["loomloom_confirm_charge"] = False
                logger.info(
                    "LoomLoom script quote ready: "
                    f"tasks={quote_result.task_count}, currency={quote_result.currency}, "
                    f"estimated_payable_t={quote_result.estimated_buyer_payable_t}"
                )

    quote_result = st.session_state.get("loomloom_script_quote")
    batch = st.session_state.get("loomloom_script_batch")
    if quote_result is not None and batch is not None:
        display_amount = (
            quote_result.estimated_buyer_payable_amount
            or f"{quote_result.estimated_buyer_payable_t} T"
        )
        st.success(
            tr(
                "LoomLoom Quote Summary Singular"
                if quote_result.task_count == 1
                else "LoomLoom Quote Summary"
            ).format(
                tasks=quote_result.task_count,
                amount=display_amount,
                currency=quote_result.currency,
            )
        )
        quote_is_current = (
            st.session_state.get("loomloom_script_input_signature") == input_signature
        )
        if not quote_is_current:
            st.warning(tr("LoomLoom Quote Changed Warning"))
        confirm_charge = st.checkbox(
            tr("Confirm LoomLoom Charge"),
            key="loomloom_confirm_charge",
            disabled=not quote_is_current,
        )
        run_in_progress = bool(st.session_state.get("loomloom_run_id"))
        if st.button(
            tr("Run LoomLoom Batch"),
            key="loomloom_execute_scripts",
            use_container_width=True,
            type="primary",
            disabled=(not quote_is_current or not confirm_charge or run_in_progress),
        ):
            try:
                execution = _create_loomloom_script_backend().execute(
                    batch,
                    client_request_id=st.session_state["loomloom_client_request_id"],
                    listing_version_id=quote_result.listing_version_id,
                    confirm=True,
                )
            except (loomloom.LoomLoomError, ValueError) as exc:
                logger.warning(f"failed to execute LoomLoom scripts: error={exc}")
                st.error(str(exc))
            else:
                st.session_state["loomloom_run_id"] = execution.run_id
                st.session_state["loomloom_run_status"] = "running"
                st.session_state["loomloom_poll_paused"] = False
                # Cada cotización solo permite lanzar un lote de pago. El estado en segundo plano solo depende de run_id, así que tras enviar
                # se pueden descartar la cotización y el ID de petición idempotente; tras un fallo el usuario debe volver a cotizar y reintentar.
                st.session_state["loomloom_script_batch"] = None
                st.session_state["loomloom_script_quote"] = None
                st.session_state["loomloom_script_input_signature"] = ""
                st.session_state["loomloom_client_request_id"] = ""
                logger.info(
                    f"LoomLoom script run submitted: run_id={execution.run_id}, "
                    f"tasks={len(batch.input_rows)}"
                )
                st.toast(tr("LoomLoom Run Submitted"))

    run_error = str(st.session_state.get("loomloom_run_error", "") or "").strip()
    if run_error:
        st.error(tr("LoomLoom Run Failed").format(error=run_error))
    run_id = str(st.session_state.get("loomloom_run_id", "") or "").strip()
    if run_id and st.session_state.get("loomloom_poll_paused", False):
        retry_col, stop_col = st.columns(2)
        if retry_col.button(
            tr("Resume LoomLoom Status Check"),
            key="loomloom_resume_status_check",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state["loomloom_run_error"] = ""
            st.session_state["loomloom_poll_failure_count"] = 0
            st.session_state["loomloom_poll_retry_after"] = 0.0
            st.session_state["loomloom_poll_paused"] = False
            st.rerun(scope="app")
        if stop_col.button(
            tr("Stop Tracking LoomLoom Run"),
            key="loomloom_stop_tracking_run",
            use_container_width=True,
            type="secondary",
            help=tr("Stop Tracking LoomLoom Run Help"),
        ):
            # Aquí solo se detiene la consulta de estado local, sin afirmar que se cancela la ejecución remota. Solo cuando el usuario confirma que deja de seguirla
            # se borra run_id, y la siguiente ejecución de pago seguirá necesitando nueva cotización y confirmación.
            st.session_state["loomloom_run_id"] = ""
            st.session_state["loomloom_run_error"] = ""
            st.session_state["loomloom_poll_paused"] = False
            st.rerun(scope="app")
    # Solo los lotes realmente en ejecución lanzan el sondeo de dos segundos; las fases de cotización y de visualización de resultados no crean
    # fragments periódicos, para no generar peticiones de red ni reruns sin sentido mientras el usuario está en la página.
    if run_id and not st.session_state.get("loomloom_poll_paused", False):
        _render_loomloom_run_progress()
    _render_loomloom_candidates()


def _render_script_settings(panel, params):
    """Renderiza los ajustes de texto y actualiza los parámetros de generación."""
    with panel:
        with st.container(border=True):
            st.write(tr("Video Script Settings"))
            # la línea de la etiqueta debe incluir el acceso «Configurar modelo grande», así que no se puede seguir usando text_area
            # la etiqueta incluida. Al meter la etiqueta y el campo en un mismo contenedor de campo se pueden sobrescribir los espaciados internos,
            # y mantener a la vez el mismo ritmo externo que los demás controles de formulario de la página.
            with st.container(key="video_subject_field"):
                with st.container(
                    key="video_subject_label_row",
                    horizontal=True,
                    vertical_alignment="center",
                    gap="small",
                ):
                    st.markdown(
                        tr("Video Subject"),
                        help=tr("Video Subject Help"),
                        width="content",
                    )
                    st.button(
                        tr("Configure LLM"),
                        key="open_llm_settings_from_subject",
                        type="tertiary",
                        on_click=_open_settings_dialog,
                        args=("llm",),
                    )
                params.video_subject = st.text_area(
                    tr("Video Subject"),
                    placeholder=tr("Video Subject Placeholder"),
                    height=96,
                    key="video_subject",
                    label_visibility="collapsed",
                ).strip()

            video_languages = [
                (tr("Auto Detect"), ""),
            ]
            for code in support_locales:
                video_languages.append((code, code))

            selected_language_code = stable_selectbox(
                tr("Script Language"),
                options=[value for _, value in video_languages],
                default_value=_saved_ui_choice(
                    "video_language",
                    [value for _, value in video_languages],
                    "",
                ),
                key="script_language_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_languages
                )[value],
            )
            params.video_language = selected_language_code
            _set_runtime_config("ui", "video_language", params.video_language)

            # Usa un contenedor local con key para limitar el estilo del acceso plegable, manteniendo la interacción nativa del expander,
            # y evitando que el estilo afecte a otras zonas plegables como «Ajustes básicos» en la parte superior de la página.
            with st.container(key="advanced_settings_script"):
                with st.expander(tr("Advanced Script Settings"), expanded=False):
                    script_backend_options = ["local", "loomloom"]
                    script_backend_labels = {
                        "local": tr("Local LLM Script Generation"),
                        "loomloom": tr("Shengsuan Cloud Batch Script Generation"),
                    }
                    script_backend_widget_key = localized_widget_key(
                        "script_generation_backend_select"
                    )
                    current_script_backend = st.session_state.get(
                        script_backend_widget_key,
                        _effective_script_generation_backend(),
                    )
                    script_generation_backend = stable_selectbox(
                        tr("Script Generation Method"),
                        options=script_backend_options,
                        default_value=_effective_script_generation_backend(),
                        key="script_generation_backend_select",
                        format_func=lambda value: script_backend_labels[value],
                        help=_script_generation_method_help(current_script_backend),
                    )
                    _set_runtime_config(
                        "app", "script_generation_backend", script_generation_backend
                    )

                    params.paragraph_number = st.slider(
                        tr("Script Paragraph Number"),
                        min_value=llm.MIN_SCRIPT_PARAGRAPH_NUMBER,
                        max_value=llm.MAX_SCRIPT_PARAGRAPH_NUMBER,
                        key="paragraph_number_input",
                    )
                    _set_runtime_config(
                        "ui", "paragraph_number", params.paragraph_number
                    )
                    params.video_script_prompt = st.text_area(
                        tr("Custom Script Requirements"),
                        height=100,
                        max_chars=llm.MAX_SCRIPT_PROMPT_LENGTH,
                        placeholder=tr("Custom Script Requirements Placeholder"),
                        key="video_script_prompt",
                    ).strip()
                    _set_runtime_config(
                        "ui", "video_script_prompt", params.video_script_prompt
                    )

                    system_prompt = st.text_area(
                        tr("Custom System Prompt"),
                        height=240,
                        max_chars=llm.MAX_SCRIPT_SYSTEM_PROMPT_LENGTH,
                        key="custom_system_prompt",
                    ).strip()
                    # El contenido por defecto lo mantiene la capa de servicio. Aunque la interfaz muestra directamente el prompt por defecto, solo
                    # se envía con la tarea cuando el usuario lo modifica de verdad, para que las tareas históricas no fijen las reglas por defecto de versiones antiguas.
                    params.custom_system_prompt = (
                        ""
                        if system_prompt == llm.DEFAULT_SCRIPT_SYSTEM_PROMPT.strip()
                        else system_prompt
                    )
                    _set_runtime_config(
                        "ui", "custom_system_prompt", params.custom_system_prompt
                    )

                    restore_prompt_col, preview_prompt_col = st.columns(2)
                    if restore_prompt_col.button(
                        tr("Restore Default System Prompt"),
                        key="restore_default_system_prompt",
                        icon=":material/restart_alt:",
                        on_click=reset_script_system_prompt,
                        use_container_width=True,
                    ):
                        st.toast(tr("Default System Prompt Restored"))
                    if preview_prompt_col.button(
                        tr("Preview Final Prompt"),
                        key="preview_final_script_prompt",
                        icon=":material/preview:",
                        use_container_width=True,
                    ):
                        render_script_prompt_preview(
                            llm.build_script_prompt(
                                video_subject=params.video_subject,
                                language=params.video_language,
                                paragraph_number=params.paragraph_number,
                                video_script_prompt=params.video_script_prompt,
                                custom_system_prompt=params.custom_system_prompt,
                            )
                        )

            # El descubrimiento de modelos solo mejora el material de video, sin cambiar el Provider de texto elegido explícitamente por el usuario.
            if _effective_script_generation_backend() == "loomloom":
                _render_loomloom_script_generation(params)
            else:
                _render_local_script_generation(params)
            params.video_script = st.text_area(
                tr("Video Script"),
                help=tr("Video Script Help"),
                height=180,
                key="video_script",
            )
            if _effective_script_generation_backend() == "loomloom":
                st.caption(tr("LoomLoom Video Terms Reuse Help"))
            elif st.button(
                tr("Generate Video Keywords"),
                key="auto_generate_terms",
                use_container_width=True,
                type="secondary",
                icon=":material/auto_awesome:",
            ):
                if not params.video_script:
                    # Las palabras clave de video deben extraerse del texto; si el texto está vacío se avisa antes y se omite la llamada al modelo.
                    st.toast(tr("Please Enter the Video Subject"))
                    st.warning(tr("Please Enter the Video Subject"))
                else:
                    with st.spinner(tr("Generating Video Keywords")):
                        terms = _run_llm_read_operation(
                            "generate_terms",
                            lambda app_config_snapshot: llm.generate_terms(
                                params.video_subject,
                                params.video_script,
                                amount=8 if params.match_materials_to_script else 5,
                                match_script_order=params.match_materials_to_script,
                                **({"real_people": True} if params.video_source == "wikimedia" else {}),
                                app_config=app_config_snapshot,
                            ),
                        )
                        if "Error: " in terms:
                            st.error(tr(terms))
                        else:
                            st.session_state["video_terms"] = ", ".join(terms)

            params.video_terms = st.text_area(
                tr("Video Keywords"),
                help=tr("Video Keywords Help"),
                key="video_terms",
            )


def _render_video_settings(panel, params):
    """Renderiza los ajustes de video y devuelve el material local elegido esta vez."""
    uploaded_files = []
    with panel:
        with st.container(border=True):
            st.write(tr("Video Settings"))
            video_concat_modes = [
                (tr("Sequential"), "sequential"),
                (tr("Random"), "random"),
            ]
            video_source_labels = {
                "wikimedia": tr("Documentary Portraits"),
                "pexels": tr("Pexels"),
                "pixabay": tr("Pixabay"),
                "coverr": tr("Coverr"),
                "wavespeed": tr("WaveSpeed AI Video"),
                "volcengine_seedance": tr("Volcano Engine Seedance"),
                "infistar": tr("Infistar AI Video"),
                "jalapeno": tr("Jalapeno Cloud AI Video"),
                "ofox": tr("OFox AI Video"),
                "metaso_minimax": tr("Metaso MiniMax H3"),
                "muapi": tr("MuAPI AI Video"),
                "loomloom": tr("Shengsuan Cloud AI Video"),
                "openai_image": tr("OpenAI Compatible Text-to-Image"),
                "codex_image": tr("codex.image.label"),
                "local": tr("Local file"),
            }
            saved_video_source_name = str(
                config.snapshot_config_with_pending(config.app).get("video_source", "pexels")
                or "pexels"
            )
            params.video_source = grouped_selectbox(
                tr("Video Source"),
                groups=(
                    (tr("Stock Video"), VIDEO_SOURCE_GROUPS["stock_video"]),
                    (tr("AI Video"), VIDEO_SOURCE_GROUPS["ai_video"]),
                    (tr("AI Image"), VIDEO_SOURCE_GROUPS["ai_image"]),
                    (tr("Local Material"), VIDEO_SOURCE_GROUPS["local"]),
                ),
                default_value=saved_video_source_name,
                key="video_source_select",
                format_func=video_source_labels.get,
                settings_label=tr("Configure Material Sources"),
                on_settings=_open_material_settings_dialog,
            )
            _set_runtime_config("app", "video_source", params.video_source)

            if params.video_source == "wikimedia":
                st.caption(tr("Documentary Portraits Help"))
                st.session_state["match_materials_to_script"] = True
                st.session_state["opening_title_enabled_checkbox"] = True
            params.opening_title_enabled = st.checkbox(
                tr("Opening Title Enabled"),
                value=_saved_ui_bool("opening_title_enabled", params.video_source == "wikimedia"),
                key="opening_title_enabled_checkbox",
                disabled=params.video_source == "wikimedia",
            )
            _set_runtime_config("ui", "opening_title_enabled", params.opening_title_enabled)
            if params.opening_title_enabled:
                params.opening_title = st.text_input(
                    tr("Opening Title"), value="", max_chars=80,
                    help=tr("Opening Title Help"), key="opening_title_input",
                )

            loomloom_video_capability = None
            if params.video_source == "loomloom":
                # Lee la caché lo antes posible, para que el control de proporción de imagen de abajo quede restringido directamente por el Profile actual.
                # Tras introducir la Key por primera vez Streamlit hace rerun, y aquí se carga una vez enseguida.
                loomloom_video_capability = _load_loomloom_video_capability(
                    _effective_loomloom_api_token()
                )

            if params.video_source == "infistar":
                st.caption(tr("Infistar AI Video Help"))
                selected_infistar_model = _render_infistar_model_selector()
            if params.video_source == "jalapeno":
                st.caption(tr("Jalapeno Cloud Video Help"))
                snapshot = config.snapshot_config_with_pending(config.app)
                st.caption(tr("Jalapeno Cloud Video Specification").format(
                    model=str(snapshot.get("jalapeno_video_model", "") or jalapeno_video.DEFAULT_MODEL),
                    resolution=str(snapshot.get("jalapeno_video_resolution", "") or jalapeno_video.DEFAULT_RESOLUTION),
                ))
            if params.video_source == "wavespeed":
                st.caption(tr("WaveSpeed AI Video Help"))
            if params.video_source == "volcengine_seedance":
                st.caption(tr("Volcano Engine Seedance Help"))
            if params.video_source == "ofox":
                st.caption(f"[OfoxAI]({OFOX_REFERRAL_URL}) · {tr('OFox AI Video Help')}")
            if params.video_source == "metaso_minimax":
                st.caption(tr("Metaso MiniMax H3 Help"))
            if params.video_source == "muapi":
                st.caption(tr("MuAPI AI Video Help"))
            if params.video_source == "local":
                # Streamlit la validación del tipo de archivo distingue mayúsculas en la extensión, así que aquí se admiten ambas formas.
                local_file_types = sorted(
                    extension.removeprefix(".")
                    for extension in LOCAL_MATERIAL_EXTENSIONS
                )
                uploaded_files = st.file_uploader(
                    tr("Upload Local Files"),
                    type=local_file_types
                    + [file_type.upper() for file_type in local_file_types],
                    accept_multiple_files=True,
                    key="local_video_materials_uploader",
                )

            # La correspondencia en el orden del texto mantiene el orden narrativo desde la generación de palabras clave hasta la composición final, así que al activarla
            # la concatenación secuencial es la única opción coherente con la lógica de ejecución real. Sincronizar el valor del control evita que la interfaz siga mostrando
            # «concatenación aleatoria», conservando la elección original del usuario, que se restaura automáticamente al desactivarla.
            sync_script_order_concat_mode()
            selected_concat_mode = stable_selectbox(
                tr("Video Concat Mode"),
                options=[value for _, value in video_concat_modes],
                default_value=_saved_ui_choice(
                    "video_concat_mode",
                    [value for _, value in video_concat_modes],
                    VideoConcatMode.random.value,
                ),
                key="video_concat_mode_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_concat_modes
                )[value],
                disabled=bool(st.session_state.get("match_materials_to_script", False)),
            )
            params.video_concat_mode = VideoConcatMode(selected_concat_mode)

            params.match_materials_to_script = st.checkbox(
                tr("Match Materials to Script Order"),
                help=tr("Match Materials to Script Order Help"),
                key="match_materials_to_script",
                on_change=sync_script_order_concat_mode,
                disabled=params.video_source == "wikimedia",
            )
            _set_runtime_config(
                "app",
                "match_materials_to_script",
                params.match_materials_to_script,
            )
            # Con la correspondencia en orden activada, sequential es un valor forzado derivado y no debe sobrescribir la preferencia de concatenación que el usuario eligió
            # con la función desactivada; al desactivarla se puede restaurar el random/sequential anterior.
            if not params.match_materials_to_script:
                _set_runtime_config(
                    "ui", "video_concat_mode", params.video_concat_mode.value
                )

            # Modo de transición de video
            video_transition_modes = [
                (tr("None"), VideoTransitionMode.none.value),
                (tr("Shuffle"), VideoTransitionMode.shuffle.value),
                (tr("FadeIn"), VideoTransitionMode.fade_in.value),
                (tr("FadeOut"), VideoTransitionMode.fade_out.value),
                (tr("SlideIn"), VideoTransitionMode.slide_in.value),
                (tr("SlideOut"), VideoTransitionMode.slide_out.value),
                (tr("ZoomIn"), VideoTransitionMode.zoom_in.value),
                (tr("ZoomOut"), VideoTransitionMode.zoom_out.value),
            ]
            selected_transition_mode = stable_selectbox(
                tr("Video Transition Mode"),
                options=[value for _, value in video_transition_modes],
                default_value=_saved_ui_choice(
                    "video_transition_mode",
                    [value for _, value in video_transition_modes],
                    VideoTransitionMode.none.value,
                ),
                key="video_transition_mode_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_transition_modes
                )[value],
            )
            params.video_transition_mode = VideoTransitionMode(selected_transition_mode)
            _set_runtime_config(
                "ui",
                "video_transition_mode",
                params.video_transition_mode.value,
            )

            video_aspect_ratios = [
                (tr("Portrait"), VideoAspect.portrait.value),
                (tr("Landscape"), VideoAspect.landscape.value),
            ]
            if loomloom_video_capability is not None:
                ratio_labels = {value: label for label, value in video_aspect_ratios}
                video_aspect_ratios = [
                    (ratio_labels[value], value)
                    for value in loomloom_video_capability.aspect_ratios
                ]
            # Coverr el 99% de la biblioteca es horizontal 16:9, y el vertical por defecto dejaría la imagen rodeada de grandes bandas negras.
            # usa una widget key específica por source para que cada source recuerde su propia elección de aspect:
            #   - al cambiar por primera vez a coverr → por defecto Landscape(index=1)
            #   - los demás source mantienen Portrait(index=0)
            #   - si el usuario cambió a mano el aspect en un source, session_state lo recuerda,
            #     y la próxima vez que vuelva a ese source se respeta su elección, sin volver a forzarla.
            default_aspect_index = 1 if params.video_source == "coverr" else 0
            video_aspect_values = [value for _, value in video_aspect_ratios]
            video_aspect_config_key = f"video_aspect_{params.video_source}"
            selected_aspect_ratio = stable_selectbox(
                tr("Video Ratio"),
                options=video_aspect_values,
                default_value=_saved_ui_choice(
                    video_aspect_config_key,
                    video_aspect_values,
                    video_aspect_ratios[default_aspect_index][1],
                ),
                key=f"video_aspect_for_{params.video_source}",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_aspect_ratios
                )[value],
            )
            params.video_aspect = VideoAspect(selected_aspect_ratio)
            _set_runtime_config(
                "ui", video_aspect_config_key, params.video_aspect.value
            )

            video_fit_modes = [
                (tr("Fill and Crop"), VideoFitMode.cover.value),
                (tr("Fit with Black Bars"), VideoFitMode.contain.value),
            ]
            selected_fit_mode = stable_selectbox(
                tr("Video Fit Mode"),
                options=[value for _, value in video_fit_modes],
                default_value=_saved_ui_choice(
                    "video_fit_mode",
                    [value for _, value in video_fit_modes],
                    VideoFitMode.cover.value,
                ),
                key="video_fit_mode_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_fit_modes
                )[value],
                help=tr("Video Fit Mode Help"),
            )
            params.video_fit_mode = VideoFitMode(selected_fit_mode)
            _set_runtime_config(
                "ui", "video_fit_mode", params.video_fit_mode.value
            )

            if params.video_source in {"local", "openai_image", "codex_image", "wikimedia"}:
                image_motion_labels = {
                    ImageMotion.static.value: tr("Static Image"),
                    ImageMotion.zoom.value: tr("Slow Zoom"),
                }
                params.image_motion = ImageMotion(
                    stable_selectbox(
                        tr("Image Motion"),
                        options=list(image_motion_labels),
                        default_value=_saved_ui_choice(
                            "image_motion", list(image_motion_labels), ImageMotion.zoom.value,
                            # Mientras la tarea de generación tiene el bloqueo, cambiar de idioma o de origen de material reconstruye los controles;
                            # en ese momento debe restaurarse la última elección aún no guardada en disco, sin escribir el valor de configuración antiguo.
                            section=config.snapshot_config_with_pending(config.ui),
                        ),
                        key="image_motion_select",
                        format_func=image_motion_labels.get,
                        help=tr("Image Motion Help"),
                    )
                )
                _set_runtime_config("ui", "image_motion", params.image_motion.value)

            # MiniMax H3 el rango de duración remoto es de 4 a 15 segundos. Al elegir Metaso se usa el rango de capacidad completo
            # , evitando que 2/3 segundos se cobren como 4 y manteniendo la WebUI coherente con la CLI y la capa de servicio.
            if params.video_source == "infistar":
                # Cada modelo ofrece primero especificaciones fijas comprobadas; no se envían a la interfaz de pago duraciones que la API rechaza.
                video_clip_durations = [infistar_video.MODELS.get(selected_infistar_model, {"seconds": 5})["seconds"]]
            elif params.video_source == "metaso_minimax":
                video_clip_durations = list(
                    range(
                        metaso_minimax.DEFAULT_MIN_DURATION_SECONDS,
                        metaso_minimax.DEFAULT_MAX_DURATION_SECONDS + 1,
                    )
                )
            elif params.video_source == "muapi":
                video_clip_durations = list(
                    range(
                        muapi.DEFAULT_MIN_DURATION_SECONDS,
                        muapi.DEFAULT_MAX_DURATION_SECONDS + 1,
                    )
                )
            else:
                video_clip_durations = [2, 3, 4, 5, 6, 7, 8, 9, 10]
            params.video_clip_duration = stable_selectbox(
                tr("Clip Duration"),
                options=video_clip_durations,
                default_value=_saved_ui_choice(
                    "video_clip_duration",
                    video_clip_durations,
                    5
                    if params.video_source in {"metaso_minimax", "muapi"}
                    else 3,
                ),
                key="video_clip_duration_select",
                help=tr("Clip Duration Help"),
            )
            _set_runtime_config(
                "ui", "video_clip_duration", params.video_clip_duration
            )
            clip_speed_key = localized_widget_key("video_clip_speed_slider")
            # session_state puede venir de tareas antiguas, parámetros de la API o estado de página de versiones antiguas. Antes de crear el control
            # se normaliza de forma unificada, conservando las elecciones legales y asegurando que el slider reciba siempre un float finito dentro del rango 0.5-2.0
            # .
            st.session_state[clip_speed_key] = utils.normalize_clip_speed(
                st.session_state.get(
                    clip_speed_key,
                    _saved_ui_number("video_clip_speed", 1.0, 0.5, 2.0),
                )
            )
            params.video_clip_speed = st.slider(
                tr("Clip Speed"),
                min_value=0.5,
                max_value=2.0,
                step=0.05,
                format="%.2fx",
                key=clip_speed_key,
                help=tr("Clip Speed Help"),
            )
            _set_runtime_config("ui", "video_clip_speed", params.video_clip_speed)
            video_count_options = [1, 2, 3, 4, 5]
            params.video_count = stable_selectbox(
                tr("Number of Videos Generated Simultaneously"),
                options=video_count_options,
                default_value=_saved_ui_choice(
                    "video_count", video_count_options, 1
                ),
                key="video_count_select",
            )
            _set_runtime_config("ui", "video_count", params.video_count)

            video_codec_options = [
                (tr("Default Video Encoder"), DEFAULT_VIDEO_CODEC_OPTION),
                ("libx264 (CPU)", "libx264"),
                ("NVIDIA NVENC (h264_nvenc)", "h264_nvenc"),
                ("AMD AMF (h264_amf)", "h264_amf"),
                ("Intel QSV (h264_qsv)", "h264_qsv"),
                ("Windows MediaFoundation (h264_mf)", "h264_mf"),
                ("macOS VideoToolbox (h264_videotoolbox)", "h264_videotoolbox"),
            ]
            saved_video_codec = config.app.get(
                "video_codec", DEFAULT_VIDEO_CODEC_OPTION
            )
            saved_video_codec_values = [item[1] for item in video_codec_options]
            if saved_video_codec not in saved_video_codec_values:
                # Versiones antiguas o configuraciones manuales pueden dejar valores inválidos. La UI vuelve a «Por defecto» en lugar de fijar
                # un codificador por el usuario, y el backend lo seguirá resolviendo como libx264 con la estrategia estable.
                saved_video_codec = DEFAULT_VIDEO_CODEC_OPTION
            selected_video_codec = stable_selectbox(
                tr("Video Encoder"),
                options=saved_video_codec_values,
                default_value=saved_video_codec,
                key="video_encoder_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_codec_options
                )[value],
                help=tr("Video Encoder Help"),
            )
            if selected_video_codec == DEFAULT_VIDEO_CODEC_OPTION:
                # El modo por defecto no persiste un codificador concreto, para que la configuración exprese «seguir el valor por defecto del proyecto».
                _delete_runtime_config("app", "video_codec")
            else:
                _set_runtime_config("app", "video_codec", selected_video_codec)

            concurrency_options = [1, 2, 4, 6, 8]
            if params.video_source in {"pexels", "pixabay", "coverr"}:
                selected_material_concurrency = stable_selectbox(
                    tr("Material Concurrency"),
                    options=concurrency_options,
                    default_value=_saved_ui_choice(
                        "material_concurrency", concurrency_options, 1, config.app
                    ),
                    key="material_concurrency_select",
                    help=tr("Material Concurrency Help"),
                )
                _set_runtime_config(
                    "app", "material_concurrency", selected_material_concurrency
                )

            selected_clip_concurrency = stable_selectbox(
                tr("Clip Rendering Concurrency"),
                options=concurrency_options,
                default_value=_saved_ui_choice(
                    "video_clip_concurrency", concurrency_options, 1, config.app
                ),
                key="clip_rendering_concurrency_select",
                help=tr("Clip Rendering Concurrency Help"),
            )
            _set_runtime_config(
                "app", "video_clip_concurrency", selected_clip_concurrency
            )

            if params.video_source == "loomloom":
                _render_loomloom_video_settings(params)

            if params.video_source == "wavespeed":
                _render_wavespeed_video_settings(params)
            if params.video_source == "volcengine_seedance":
                _render_seedance_video_settings(params)
            if params.video_source == "infistar":
                st.warning(tr("Infistar Video Billing Notice"))
                st.checkbox(tr("Confirm Infistar Charge"), key="infistar_confirm_charge")
            if params.video_source == "jalapeno":
                _sync_jalapeno_charge_confirmation(params)
                st.warning(tr("Jalapeno Cloud Video Billing Notice"))
                st.checkbox(tr("Confirm Jalapeno Cloud Charge"), key="jalapeno_confirm_charge")
            if params.video_source == "ofox":
                _render_ofox_video_settings(params)
            if params.video_source == "metaso_minimax":
                _render_metaso_minimax_video_settings(params)
            if params.video_source == "muapi":
                _render_muapi_video_settings(params)
    return uploaded_files


def _render_wavespeed_video_settings(params):
    """
    Renderiza la estimación del número de generaciones de WaveSpeed y la confirmación de cobro.

    La generación se cobra por unidad; antes de enviar el usuario debe ver aproximadamente cuántos tramos se generarán. La estimación se hace totalmente en local
    : se divide el rango estimado de duración de la voz entre la duración del fragmento para obtener los fragmentos a cubrir. El flujo de material
    genera por sí mismo tramo a tramo según se necesita y se detiene al reunir la duración necesaria, así que el número real de generaciones es el del tiempo de ejecución,
    y la estimación solo sirve como indicación de magnitud, sin participar en la ejecución de la tarea.
    """
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    estimated_range = _estimate_voiceover_duration_range(
        str(params.video_script or ""),
        params.voice_rate,
    )
    if estimated_range:
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("WaveSpeed Billing Notice").format(min=min_clips, max=max_clips)
        )
    else:
        st.warning(tr("WaveSpeed Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm WaveSpeed Charge"),
        key="wavespeed_confirm_charge",
        help=tr("Confirm WaveSpeed Charge Help"),
    )


def _render_seedance_video_settings(params):
    """Muestra el número previsto de tareas de pago y exige que el usuario confirme explícitamente el coste de generación de Ark."""
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    estimated_range = _estimate_voiceover_duration_range(
        str(params.video_script or ""), params.voice_rate
    )
    if estimated_range:
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("Volcano Engine Seedance Billing Notice").format(
                min=min_clips, max=max_clips
            )
        )
    else:
        st.warning(tr("Volcano Engine Seedance Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm Volcano Engine Seedance Charge"),
        key="volcengine_seedance_confirm_charge",
        help=tr("Confirm Volcano Engine Seedance Charge Help"),
    )


def _render_jalapeno_provider_settings():
    """Los valores por defecto se rellenan directamente; la configuración de video independiente no depende del Provider de modelo grande elegido."""
    snapshot = config.snapshot_config_with_pending(config.app)
    st.markdown(f"**{tr('Jalapeno Cloud AI Video')}**")
    key = st.text_input(
        tr("Jalapeno Cloud Video API Key"),
        value=str(snapshot.get("jalapeno_video_api_key", "") or ""),
        type="password",
        key="jalapeno_video_api_key_input",
        help=tr("Jalapeno Cloud Video Key Help"),
        on_change=_save_settings_change,
        args=(
            "app",
            "jalapeno_video_api_key",
            "jalapeno_video_api_key_input",
            lambda value: value.strip(),
        ),
    ).strip()
    _set_runtime_config("app", "jalapeno_video_api_key", key)
    if (
        not key
        and not os.getenv("JALAPENO_VIDEO_API_KEY", "").strip()
        and str(snapshot.get("jalapeno_api_key", "") or "").strip()
    ):
        st.caption(tr("Jalapeno Cloud Video Key Reused"))
    model = st.text_input(
        tr("Jalapeno Cloud Video Model"),
        value=str(
            snapshot.get("jalapeno_video_model", "") or jalapeno_video.DEFAULT_MODEL
        ),
        key="jalapeno_video_model_input",
        on_change=_save_settings_change,
        args=(
            "app",
            "jalapeno_video_model",
            "jalapeno_video_model_input",
            lambda value: value.strip(),
        ),
    ).strip()
    _set_runtime_config("app", "jalapeno_video_model", model)
    base = st.text_input(
        tr("Jalapeno Cloud Video Base URL"),
        value=str(
            snapshot.get("jalapeno_video_base_url", "")
            or jalapeno_video.DEFAULT_TASKS_URL
        ),
        help=tr("Jalapeno Cloud Video Base URL Help"),
        key="jalapeno_video_base_url_input",
        on_change=_save_settings_change,
        args=(
            "app",
            "jalapeno_video_base_url",
            "jalapeno_video_base_url_input",
            lambda value: value.strip(),
        ),
    ).strip()
    _set_runtime_config("app", "jalapeno_video_base_url", base)
    resolution = stable_selectbox(
        tr("Jalapeno Cloud Video Resolution"),
        options=sorted(
            volcengine_seedance.SUPPORTED_RESOLUTIONS, key=lambda value: int(value[:-1])
        ),
        default_value=str(
            snapshot.get("jalapeno_video_resolution", "")
            or jalapeno_video.DEFAULT_RESOLUTION
        ),
        key="jalapeno_video_resolution_select",
        on_change=_save_settings_change,
        args=(
            "app",
            "jalapeno_video_resolution",
            localized_widget_key("jalapeno_video_resolution_select"),
            lambda value: value,
        ),
    )
    _set_runtime_config("app", "jalapeno_video_resolution", resolution)


def _sync_jalapeno_charge_confirmation(params):
    """Si cambian la dirección de acceso, el modelo, la especificación o la Key vigente, se vuelve a confirmar, para no reutilizar una autorización caducada."""
    snapshot = config.snapshot_config_with_pending(config.app)
    signature = _jalapeno_charge_signature(params, snapshot)
    if st.session_state.get("jalapeno_charge_signature") != signature:
        st.session_state["jalapeno_confirm_charge"] = False
        st.session_state["jalapeno_charge_signature"] = signature


def _jalapeno_charge_signature(params, snapshot):
    """Usa la misma instantánea para comprobar la confirmación de coste, para no leer configuraciones distintas entre el renderizado y el clic de envío."""
    return hashlib.sha256(json.dumps([
        jalapeno_video.get_api_key(snapshot),
        snapshot.get("jalapeno_video_base_url", ""),
        snapshot.get("jalapeno_video_model", ""),
        snapshot.get("jalapeno_video_resolution", ""),
        str(params.video_aspect), params.video_clip_duration, params.video_count,
    ]).encode()).hexdigest()


def _render_infistar_model_selector():
    """El orden de modelos es fijo: Seedance, MiniMax, Kling; al cambiar de modelo se vuelve a confirmar el coste."""
    snapshot = config.snapshot_config_with_pending(config.app)
    configured = str(snapshot.get("infistar_video_model", "") or infistar_video.DEFAULT_MODEL)
    options = list(infistar_video.MODELS)
    if configured not in options:
        options.append(configured)
        st.error(tr("Infistar Video Invalid Model"))
    model = stable_selectbox(
        tr("Infistar Video Model"), options=options,
        default_value=configured, key="infistar_video_model_select",
        format_func=lambda value: infistar_video.MODELS.get(value, {"label": value})["label"],
    )
    if st.session_state.get("infistar_previous_video_model") != model:
        st.session_state["infistar_confirm_charge"] = False
        st.session_state["infistar_previous_video_model"] = model
    _set_runtime_config("app", "infistar_video_model", model)
    profile = infistar_video.MODELS.get(model)
    if profile:
        st.caption(tr("Infistar Video Specification").format(seconds=profile["seconds"], size=profile["size"]))
        if model == "MiniMax-H3":
            st.info(tr("Infistar MiniMax Landscape Help"))
    return model


def _render_ofox_video_settings(params):
    """Muestra el número previsto de tareas de pago y exige que el usuario confirme explícitamente el coste de generación de OFox."""
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    estimated_range = _estimate_voiceover_duration_range(
        str(params.video_script or ""), params.voice_rate
    )
    if estimated_range:
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("OFox Billing Notice").format(min=min_clips, max=max_clips)
        )
    else:
        st.warning(tr("OFox Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm OFox Charge"),
        key="ofox_confirm_charge",
        help=tr("Confirm OFox Charge Help"),
    )


def _render_metaso_minimax_video_settings(params):
    """Muestra el número previsto de tareas de pago y exige que el usuario confirme el coste de generación de Metaso MiniMax."""
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    voice_mode = st.session_state.get(
        localized_widget_key("voice_mode_control"),
        config.ui.get("voice_mode"),
    )
    if voice_mode == VOICE_MODE_UPLOAD:
        # Los ajustes de video se renderizan antes que los de audio, y en ese momento no se puede leer de forma fiable la duración real del archivo recién subido en esta ronda
        # . El modo de subida ya no muestra la cifra deducida del texto del guion, para que el usuario no crea que un audio de
        # 5 segundos también creará varias tareas de pago según un texto más largo; en ejecución sigue mandando la duración real del archivo.
        st.warning(
            tr("Metaso MiniMax Billing Notice Uploaded Audio").format(
                resolution=str(
                    config.app.get(
                        "metaso_minimax_resolution",
                        metaso_minimax.DEFAULT_RESOLUTION,
                    )
                    or metaso_minimax.DEFAULT_RESOLUTION
                ),
                duration=clip_duration,
                count=video_count,
            )
        )
    elif estimated_range := _estimate_voiceover_duration_range(
        str(params.video_script or ""), params.voice_rate
    ):
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("Metaso MiniMax Billing Notice").format(
                min=min_clips,
                max=max_clips,
                resolution=str(
                    config.app.get(
                        "metaso_minimax_resolution",
                        metaso_minimax.DEFAULT_RESOLUTION,
                    )
                    or metaso_minimax.DEFAULT_RESOLUTION
                ),
            )
        )
    else:
        st.warning(tr("Metaso MiniMax Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm Metaso MiniMax Charge"),
        key="metaso_minimax_confirm_charge",
        help=tr("Confirm Metaso MiniMax Charge Help"),
    )


def _render_muapi_video_settings(params):
    """Show an estimated MuAPI task count and require explicit billing consent."""
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    if estimated_range := _estimate_voiceover_duration_range(
        str(params.video_script or ""), params.voice_rate
    ):
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("MuAPI Billing Notice").format(min=min_clips, max=max_clips)
        )
    else:
        st.warning(tr("MuAPI Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm MuAPI Charge"),
        key="muapi_confirm_charge",
        help=tr("Confirm MuAPI Charge Help"),
    )


def _estimate_voiceover_duration_range(
    text: str, voice_rate: float
) -> tuple[float, float] | None:
    """
    Estima en local la duración completa de la voz y devuelve límites inferior y superior conservadores en segundos.

    Esta estimación solo ayuda al usuario a valorar la magnitud del texto antes de llamar a un TTS de pago, sin participar en la ejecución de la tarea.
    El chino, el japonés y el coreano se estiman por velocidad de caracteres; los demás idiomas que separan palabras con espacios, por velocidad de palabras,
    y luego se suman las pausas de puntuación habituales. Distintos Providers, voces y tonos causan desviaciones reales, así que la interfaz
    debe mostrar un intervalo y no un único resultado falsamente preciso.
    """
    normalized_text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized_text:
        return None

    script_chars = re.findall(
        r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]",
        normalized_text,
    )
    remaining_text = re.sub(
        r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]",
        " ",
        normalized_text,
    )
    words = re.findall(r"\b[\w]+(?:[-'’][\w]+)*\b", remaining_text, re.UNICODE)
    punctuation_count = len(re.findall(r"[,，.。!?！？;；:：]", normalized_text))

    # 4.2 caracteres/s y 2.6 palabras/s se acercan a la velocidad de una narración cotidiana; la puntuación añade una ligera pausa de 0.12 segundos.
    # voice_rate solo sirve como corrección de la estimación. Algunos TTS generativos no aplican el multiplicador de forma estricta, así que al final
    # se mantiene un intervalo de ±15%, para que el usuario no crea que este valor equivale al resultado real del servidor.
    base_seconds = len(script_chars) / 4.2 + len(words) / 2.6 + punctuation_count * 0.12
    if base_seconds <= 0:
        return None

    normalized_rate = max(float(voice_rate or 1.0), 0.1)
    estimated_seconds = base_seconds / normalized_rate
    return (
        round(max(estimated_seconds * 0.85, 1.0), 1),
        round(max(estimated_seconds * 1.15, 1.0), 1),
    )


def _get_voice_preview_sample(voice_name: str) -> str:
    """Devuelve un texto de vista previa corto adecuado para la voz actual, sin usar el texto completo del video del usuario."""
    # ElevenLabs Cuando la voz no tiene un campo de idioma explícito, se elige según los caracteres vietnamitas del nombre mostrado
    # el texto de vista previa, para no juzgar el efecto de la voz con un idioma claramente inadecuado.
    if voice.is_elevenlabs_voice(voice_name):
        parts = voice_name.split(":", 2)
        display = parts[2] if len(parts) >= 3 else ""
        vietnamese_chars = set("àáâãèéêìíòóôõùúýăđơưÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ")
        if any(char in vietnamese_chars for char in display):
            return "Xin chào, đây là đoạn âm thanh thử nghiệm giọng nói."
    return tr("Voice Example")


def _voice_preview_fingerprint(
    *,
    preview_type: str,
    content: str,
    tts_server: str,
    voice_name: str,
    voice_rate: float,
    voice_volume: float,
    provider_signature: dict,
) -> str:
    """Genera la huella de caché de la vista previa; si cambia cualquier parámetro de voz, la vista previa antigua se invalida automáticamente."""
    payload = {
        "preview_type": preview_type,
        "content": content,
        "tts_server": tts_server,
        "voice_name": voice_name,
        "voice_rate": voice_rate,
        "voice_volume": voice_volume,
        "provider_signature": provider_signature,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _credential_signature(value: str) -> str:
    """
    Genera un resumen de credenciales que solo sirve para decidir la invalidación de caché.

    El resumen no se escribe en la configuración, los logs ni los archivos de tarea. Si el usuario cambia la API Key, el resumen cambia, y así
    se fuerza a llamar de nuevo al servicio de voz actual, para que la caché de vista previa antigua no haga parecer válidas unas credenciales nuevas inválidas.
    """
    normalized_value = str(value or "")
    if not normalized_value:
        return ""
    return hashlib.sha256(normalized_value.encode("utf-8")).hexdigest()


def _get_voxcpm_reference_audio() -> bytes | None:
    """Return the current session's normalized reference audio, if any."""
    payload = st.session_state.get(VOXCPM_REFERENCE_AUDIO_SESSION_KEY)
    if not isinstance(payload, dict):
        return None
    audio_bytes = payload.get("audio_bytes")
    return bytes(audio_bytes) if isinstance(audio_bytes, bytes) else None


def _get_voxcpm_reference_audio_digest() -> str:
    payload = st.session_state.get(VOXCPM_REFERENCE_AUDIO_SESSION_KEY)
    if not isinstance(payload, dict):
        return ""
    digest = payload.get("audio_digest")
    return str(digest) if isinstance(digest, str) else ""


def _get_voxcpm_prompt_audio() -> bytes | None:
    payload = st.session_state.get(VOXCPM_PROMPT_AUDIO_SESSION_KEY)
    if not isinstance(payload, dict):
        return None
    audio_bytes = payload.get("audio_bytes")
    return bytes(audio_bytes) if isinstance(audio_bytes, bytes) else None


def _get_voxcpm_prompt_audio_digest() -> str:
    payload = st.session_state.get(VOXCPM_PROMPT_AUDIO_SESSION_KEY)
    if not isinstance(payload, dict):
        return ""
    digest = payload.get("audio_digest")
    return str(digest) if isinstance(digest, str) else ""


def _get_voxcpm_prompt_text() -> str:
    return str(st.session_state.get(VOXCPM_PROMPT_TEXT_SESSION_KEY, "") or "").strip()


def _clear_voxcpm_separate_prompt_audio() -> None:
    st.session_state.pop("voxcpm_prompt_audio_uploader", None)
    st.session_state.pop(VOXCPM_PROMPT_AUDIO_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY, None)


def _clear_voxcpm_prompt_state() -> None:
    _clear_voxcpm_separate_prompt_audio()
    _clear_voxcpm_prompt_transcript()
    st.session_state.pop(VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY, None)
    st.session_state.pop(VOXCPM_PROMPT_EXAMPLE_MODE_SESSION_KEY, None)


def _clear_voxcpm_prompt_transcript() -> None:
    st.session_state.pop(VOXCPM_PROMPT_TEXT_SESSION_KEY, None)


def _sync_voxcpm_prompt_example_mode(use_separate_prompt_audio: bool) -> None:
    """Invalidate the transcript whenever its effective example changes mode."""
    previous_mode = st.session_state.get(VOXCPM_PROMPT_EXAMPLE_MODE_SESSION_KEY)
    current_mode = bool(use_separate_prompt_audio)
    if previous_mode is not None and bool(previous_mode) != current_mode:
        _clear_voxcpm_prompt_transcript()
    st.session_state[VOXCPM_PROMPT_EXAMPLE_MODE_SESSION_KEY] = current_mode


def _get_voxcpm_effective_prompt_audio() -> bytes | None:
    if not st.session_state.get(VOXCPM_HIGH_FIDELITY_SESSION_KEY, False):
        return None
    if st.session_state.get(VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY, False):
        return _get_voxcpm_prompt_audio()
    return _get_voxcpm_reference_audio()


def _get_voxcpm_effective_prompt_audio_digest() -> str:
    if not st.session_state.get(VOXCPM_HIGH_FIDELITY_SESSION_KEY, False):
        return ""
    if st.session_state.get(VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY, False):
        return _get_voxcpm_prompt_audio_digest()
    return _get_voxcpm_reference_audio_digest()


def _get_voxcpm_prompt_validation_error() -> str:
    if not st.session_state.get(VOXCPM_HIGH_FIDELITY_SESSION_KEY, False):
        return ""
    upload_error = st.session_state.get(VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY)
    if upload_error:
        return str(upload_error)
    prompt_text = _get_voxcpm_prompt_text()
    if not prompt_text:
        return tr("VoxCPM Prompt Text Required")
    if (
        st.session_state.get(VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY, False)
        and not _get_voxcpm_prompt_audio()
    ):
        return tr("VoxCPM Prompt Pair Required")
    return ""


def _get_voxcpm_preview_validation_error(
    selected_tts_server: str,
    voice_name: str,
) -> str:
    if selected_tts_server != "voxcpm" and not voice.is_voxcpm_voice(voice_name):
        return ""
    reference_error = st.session_state.get(VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY)
    if reference_error:
        return str(reference_error)
    return _get_voxcpm_prompt_validation_error()


def _sync_voxcpm_reference_audio(uploaded_file) -> bytes | None:
    """Normalize a new upload once and keep only bounded WAV bytes in session."""
    if uploaded_file is None:
        st.session_state.pop(VOXCPM_REFERENCE_AUDIO_SESSION_KEY, None)
        st.session_state.pop(VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY, None)
        _clear_voxcpm_prompt_state()
        st.session_state.pop(VOXCPM_HIGH_FIDELITY_SESSION_KEY, None)
        return None

    upload_size = int(getattr(uploaded_file, "size", 0) or 0)
    if upload_size <= 0:
        st.session_state.pop(VOXCPM_REFERENCE_AUDIO_SESSION_KEY, None)
        _clear_voxcpm_prompt_state()
        st.session_state[VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY] = tr(
            "VoxCPM Reference Audio Empty"
        )
        return None
    if upload_size > voice.VOXCPM_REFERENCE_AUDIO_MAX_UPLOAD_BYTES:
        st.session_state.pop(VOXCPM_REFERENCE_AUDIO_SESSION_KEY, None)
        _clear_voxcpm_prompt_state()
        st.session_state[VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY] = tr(
            "VoxCPM Reference Audio Upload Too Large"
        ).format(limit=voice.VOXCPM_REFERENCE_AUDIO_MAX_UPLOAD_BYTES // (1024 * 1024))
        return None

    previous_digest = _get_voxcpm_reference_audio_digest()
    raw_audio = uploaded_file.getvalue()
    upload_digest = hashlib.sha256(raw_audio).hexdigest()
    cached = st.session_state.get(VOXCPM_REFERENCE_AUDIO_SESSION_KEY)
    if isinstance(cached, dict) and cached.get("upload_digest") == upload_digest:
        st.session_state.pop(VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY, None)
        return _get_voxcpm_reference_audio()

    try:
        with st.spinner(tr("Validating VoxCPM Reference Audio")):
            wav_audio = voice.prepare_voxcpm_reference_audio(
                raw_audio,
                Path(str(getattr(uploaded_file, "name", ""))).suffix,
            )
    except ValueError as exc:
        # Do not retain a prior clip after the user selects a replacement that
        # fails validation. Otherwise preview and generation could silently use
        # a different person's voice than the one currently shown in the UI.
        st.session_state.pop(VOXCPM_REFERENCE_AUDIO_SESSION_KEY, None)
        _clear_voxcpm_prompt_state()
        st.session_state[VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY] = str(exc)
        return None

    st.session_state[VOXCPM_REFERENCE_AUDIO_SESSION_KEY] = {
        "upload_digest": upload_digest,
        "audio_digest": hashlib.sha256(wav_audio).hexdigest(),
        "audio_bytes": wav_audio,
    }
    if (
        previous_digest != hashlib.sha256(wav_audio).hexdigest()
        and not st.session_state.get(VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY, False)
    ):
        _clear_voxcpm_prompt_transcript()
    st.session_state.pop(VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY, None)
    return bytes(wav_audio)


def _sync_voxcpm_prompt_audio(uploaded_file) -> bytes | None:
    """Normalize the optional delivery example without persisting its content."""
    if uploaded_file is None:
        st.session_state.pop(VOXCPM_PROMPT_AUDIO_SESSION_KEY, None)
        st.session_state.pop(VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY, None)
        return None

    upload_size = int(getattr(uploaded_file, "size", 0) or 0)
    if upload_size <= 0:
        st.session_state.pop(VOXCPM_PROMPT_AUDIO_SESSION_KEY, None)
        st.session_state[VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY] = tr(
            "VoxCPM Prompt Audio Empty"
        )
        return None
    if upload_size > voice.VOXCPM_REFERENCE_AUDIO_MAX_UPLOAD_BYTES:
        st.session_state.pop(VOXCPM_PROMPT_AUDIO_SESSION_KEY, None)
        st.session_state[VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY] = tr(
            "VoxCPM Prompt Audio Upload Too Large"
        ).format(limit=voice.VOXCPM_REFERENCE_AUDIO_MAX_UPLOAD_BYTES // (1024 * 1024))
        return None

    previous_digest = _get_voxcpm_prompt_audio_digest()
    raw_audio = uploaded_file.getvalue()
    upload_digest = hashlib.sha256(raw_audio).hexdigest()
    cached = st.session_state.get(VOXCPM_PROMPT_AUDIO_SESSION_KEY)
    if isinstance(cached, dict) and cached.get("upload_digest") == upload_digest:
        st.session_state.pop(VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY, None)
        return _get_voxcpm_prompt_audio()

    try:
        with st.spinner(tr("Validating VoxCPM Prompt Audio")):
            wav_audio = voice.prepare_voxcpm_reference_audio(
                raw_audio,
                Path(str(getattr(uploaded_file, "name", ""))).suffix,
            )
    except ValueError as exc:
        st.session_state.pop(VOXCPM_PROMPT_AUDIO_SESSION_KEY, None)
        st.session_state[VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY] = str(exc)
        return None

    st.session_state[VOXCPM_PROMPT_AUDIO_SESSION_KEY] = {
        "upload_digest": upload_digest,
        "audio_digest": hashlib.sha256(wav_audio).hexdigest(),
        "audio_bytes": wav_audio,
    }
    if previous_digest != hashlib.sha256(wav_audio).hexdigest():
        _clear_voxcpm_prompt_transcript()
    st.session_state.pop(VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY, None)
    return bytes(wav_audio)


def _get_voice_preview_provider_signature(tts_server: str) -> dict:
    """
    Devuelve la configuración no sensible del Provider que afecta al resultado de la vista previa.

    API Key Solo participa en la huella de caché como resumen unidireccional; las credenciales originales no entran en la caché ni en los logs. Si cambian el modelo,
    la dirección del servicio, la región o las credenciales, debe regenerarse la vista previa, o la interfaz podría seguir reproduciendo
    audio de la configuración de Provider anterior, haciendo creer al usuario que los ajustes actuales ya surtieron efecto.
    """
    if tts_server == "azure-tts-v2":
        return {
            "speech_region": config.azure.get("speech_region", ""),
            "credential": _credential_signature(config.azure.get("speech_key", "")),
        }
    if tts_server == "siliconflow":
        return {
            "credential": _credential_signature(config.siliconflow.get("api_key", ""))
        }
    if tts_server == "gemini-tts":
        return {
            "credential": _credential_signature(config.app.get("gemini_api_key", ""))
        }
    if tts_server == "mimo-tts":
        return {"credential": _credential_signature(config.app.get("mimo_api_key", ""))}
    if tts_server == "minimax-tts":
        return {
            "base_url": voice.get_minimax_tts_endpoint(),
            "model_id": config.minimax_tts.get("model_id", ""),
            "voice_id": config.minimax_tts.get("voice_id", ""),
            "credential": _credential_signature(voice.get_minimax_tts_api_key()),
        }
    if tts_server == "elevenlabs":
        return {
            "model_id": config.elevenlabs.get("model_id", ""),
            "credential": _credential_signature(config.elevenlabs.get("api_key", "")),
        }
    if tts_server == "chatterbox":
        return {
            "base_url": config.chatterbox.get("base_url", ""),
            "model_id": config.chatterbox.get("model_id", ""),
            "credential": _credential_signature(config.chatterbox.get("api_key", "")),
        }
    if tts_server == "kokoro":
        return {
            "base_url": config.kokoro.get("base_url", ""),
            "model_id": config.kokoro.get("model_id", ""),
            "credential": _credential_signature(config.kokoro.get("api_key", "")),
        }
    if tts_server == "voxcpm":
        return {
            "base_url": config.voxcpm.get("base_url", ""),
            "model_id": config.voxcpm.get("model_id", ""),
            "voice_id": config.voxcpm.get("voice_id", "default"),
            "credential": _credential_signature(config.voxcpm.get("api_key", "")),
            "reference_audio": _get_voxcpm_reference_audio_digest(),
            "prompt_audio": _get_voxcpm_effective_prompt_audio_digest(),
            "prompt_text": _credential_signature(
                _get_voxcpm_prompt_text()
                if st.session_state.get(VOXCPM_HIGH_FIDELITY_SESSION_KEY, False)
                else ""
            ),
        }
    return {}


def _synthesize_voice_preview(
    *,
    content: str,
    preview_type: str,
    selected_tts_server: str,
    voice_name: str,
    voice_rate: float,
    voice_volume: float,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
) -> dict | None:
    """Genera una vista previa y la pasa a caché en memoria; los archivos temporales no se conservan a largo plazo entre sesiones."""
    if selected_tts_server == "chatterbox":
        _sync_chatterbox_config_from_session_state()
    if selected_tts_server == "kokoro":
        _sync_kokoro_config_from_session_state()

    temp_dir = utils.storage_dir("temp", create=True)
    audio_file = os.path.join(temp_dir, f"tmp-voice-{str(uuid4())}.mp3")
    logger.info(
        f"generating {preview_type} voice preview: "
        f"voice={voice_name}, rate={voice_rate}, volume={voice_volume}, "
        f"text_length={len(content)}"
    )
    try:
        with config.try_runtime_config_lock() as lock_acquired:
            if not lock_acquired:
                return {"busy": True}
            tts_kwargs = {
                "text": content,
                "voice_name": voice_name,
                "voice_rate": voice_rate,
                "voice_file": audio_file,
                "voice_volume": voice_volume,
            }
            if voxcpm_reference_audio is not None:
                tts_kwargs["voxcpm_reference_audio"] = voxcpm_reference_audio
            if voxcpm_prompt_audio is not None:
                tts_kwargs["voxcpm_prompt_audio"] = voxcpm_prompt_audio
                tts_kwargs["voxcpm_prompt_text"] = voxcpm_prompt_text
            sub_maker = voice.tts(**tts_kwargs)
        if not sub_maker or not os.path.exists(audio_file):
            logger.error(f"{preview_type} voice preview did not produce an audio file")
            return None

        with open(audio_file, "rb") as file:
            audio_bytes = file.read()
        if not audio_bytes:
            logger.error(f"voice preview audio file is empty: {audio_file}")
            return None

        duration = voice.get_audio_duration(audio_file)
        if (
            not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            logger.warning(
                f"voice preview duration is unavailable: "
                f"preview_type={preview_type}, voice={voice_name}"
            )
            duration = None

        return {
            "audio_bytes": audio_bytes,
            "mime_type": _detect_audio_mime(audio_file, audio_bytes),
            "duration": duration,
            "preview_type": preview_type,
            "sub_maker": sub_maker,
            # Hace que el panel de video, situado antes del panel de audio, solo use la duración de la vista previa completa que coincide exactamente
            # con los ajustes actuales; una vista previa corta o un texto antiguo nunca pueden cambiar el número recomendado de materiales.
            "content_digest": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "tts_server": selected_tts_server,
            "voice_name": voice_name,
            "voice_rate": float(voice_rate),
        }
    finally:
        # El reproductor del navegador usa bytes en memoria, y el archivo se puede limpiar tras leerlo, para no acumular temporales con vistas previas frecuentes.
        try:
            os.remove(audio_file)
        except FileNotFoundError:
            pass
        except OSError as exc:
            # Un fallo de limpieza no debe sobrescribir la respuesta o excepción real del TTS, pero hay que conservar la ruta y el error del sistema,
            # para diagnosticar problemas del entorno como permisos o sistemas de archivos de solo lectura.
            logger.warning(
                f"failed to delete voice preview file {audio_file}: {str(exc)}"
            )


def _render_voice_preview(params, friendly_names, selected_tts_server, voice_name):
    """Renderiza la vista previa corta de bajo coste, la estimación de duración del texto completo y la vista previa completa de la voz bajo demanda."""
    if not friendly_names:
        return

    script_content = str(params.video_script or "").strip()
    estimated_range = _estimate_voiceover_duration_range(
        script_content,
        params.voice_rate,
    )
    if estimated_range:
        st.caption(
            tr("Estimated Voiceover Duration").format(
                min=estimated_range[0],
                max=estimated_range[1],
            )
        )
    else:
        st.caption(tr("Voiceover Script Required"))

    sample_content = _get_voice_preview_sample(voice_name)
    provider_signature = _get_voice_preview_provider_signature(selected_tts_server)
    preview_validation_error = _get_voxcpm_preview_validation_error(
        selected_tts_server,
        voice_name,
    )
    preview_columns = st.columns(2)
    short_preview_requested = preview_columns[0].button(
        tr("Play Voice"),
        key="play_voice_button",
        icon=":material/graphic_eq:",
        use_container_width=True,
        disabled=bool(preview_validation_error),
    )
    full_preview_requested = preview_columns[1].button(
        tr("Generate Full Voiceover Preview"),
        key="generate_full_voiceover_preview_button",
        icon=":material/article:",
        help=tr("Full Voiceover Preview Cost Hint"),
        use_container_width=True,
        disabled=not bool(script_content) or bool(preview_validation_error),
    )

    preview_type = ""
    preview_content = ""
    if short_preview_requested:
        preview_type = "sample"
        preview_content = sample_content
    elif full_preview_requested:
        preview_type = "full"
        preview_content = script_content

    sample_fingerprint = _voice_preview_fingerprint(
        preview_type="sample",
        content=sample_content,
        tts_server=selected_tts_server,
        voice_name=voice_name,
        voice_rate=params.voice_rate,
        voice_volume=params.voice_volume,
        provider_signature=provider_signature,
    )
    full_fingerprint = (
        _voice_preview_fingerprint(
            preview_type="full",
            content=script_content,
            tts_server=selected_tts_server,
            voice_name=voice_name,
            voice_rate=params.voice_rate,
            voice_volume=params.voice_volume,
            provider_signature=provider_signature,
        )
        if script_content
        else ""
    )

    if preview_type:
        requested_fingerprint = (
            sample_fingerprint if preview_type == "sample" else full_fingerprint
        )
        cached_preview = st.session_state.get("voice_preview_audio")
        if (
            not cached_preview
            or cached_preview.get("fingerprint") != requested_fingerprint
        ):
            try:
                with st.spinner(tr("Synthesizing Voice")):
                    preview_result = _synthesize_voice_preview(
                        content=preview_content,
                        preview_type=preview_type,
                        selected_tts_server=selected_tts_server,
                        voice_name=voice_name,
                        voice_rate=params.voice_rate,
                        voice_volume=params.voice_volume,
                        voxcpm_reference_audio=(
                            _get_voxcpm_reference_audio()
                            if selected_tts_server == "voxcpm"
                            else None
                        ),
                        voxcpm_prompt_audio=(
                            _get_voxcpm_effective_prompt_audio()
                            if selected_tts_server == "voxcpm"
                            else None
                        ),
                        voxcpm_prompt_text=(
                            _get_voxcpm_prompt_text()
                            if selected_tts_server == "voxcpm"
                            else ""
                        ),
                    )
            except Exception as exc:
                logger.exception(f"failed to generate {preview_type} voice preview")
                st.error(tr("Voice Preview Failed").format(error=str(exc)))
            else:
                if preview_result and preview_result.get("busy"):
                    st.warning(tr("Voice Preview Busy"))
                elif preview_result:
                    preview_result["fingerprint"] = requested_fingerprint
                    st.session_state["voice_preview_audio"] = preview_result
                    if (
                        preview_type == "full"
                        and params.video_source == "loomloom"
                        and isinstance(preview_result.get("duration"), (int, float))
                        and math.isfinite(preview_result["duration"])
                        and preview_result["duration"] > 0
                    ):
                        # Los ajustes de video se renderizan antes que los de audio. Tras una vista previa completa exitosa se lanza un rerun,
                        # para que el número de materiales de arriba se recomiende de nuevo al instante según la duración real de la narración y se actualice el aviso de cobertura.
                        st.session_state["loomloom_video_scene_autofill_digest"] = (
                            preview_result["content_digest"]
                        )
                        st.rerun()
                else:
                    st.error(tr("Voice Preview No Audio"))

    cached_preview = st.session_state.get("voice_preview_audio")
    valid_fingerprints = {sample_fingerprint, full_fingerprint}
    if (
        cached_preview
        and cached_preview.get("fingerprint") in valid_fingerprints
        and cached_preview.get("audio_bytes")
    ):
        # Solo se reproduce automáticamente cuando el usuario hace clic explícitamente en «Probar voz» en esta ocasión. Los demás controles de Streamlit
        # también provocan reruns de la página; si se activara autoplay de forma permanente para el audio en caché, cambiar cualquier ajuste
        # podría hacer que la vista previa antigua sonara desde el principio. La vista previa completa sigue siendo de reproducción manual, para que un audio largo no
        # interrumpa al usuario por sorpresa al terminar de generarse.
        should_autoplay = bool(
            short_preview_requested
            and cached_preview.get("preview_type") == "sample"
            and cached_preview.get("fingerprint") == sample_fingerprint
        )
        st.audio(
            cached_preview["audio_bytes"],
            format=cached_preview.get("mime_type", "audio/mp3"),
            autoplay=should_autoplay,
        )
        if cached_preview.get("preview_type") == "full":
            duration = cached_preview.get("duration")
            if isinstance(duration, (int, float)) and duration > 0:
                st.caption(
                    tr("Actual Voiceover Duration").format(duration=f"{duration:.1f}")
                )
            else:
                st.warning(tr("Voice Preview Duration Unavailable"))


def _get_reusable_full_voice_preview(params, voice_mode: str) -> dict | None:
    """
    Devuelve la caché de vista previa completa que coincide exactamente con los parámetros de generación actuales.

    Solo se reutiliza la vista previa del texto completo; las muestras cortas de voz nunca pueden entrar en la tarea real. La huella cubre de forma unificada el texto,
    Provider, la voz, la velocidad, el volumen y el resumen de configuración no sensible; cualquier cambio de parámetro vuelve de forma natural al
    flujo TTS normal. La línea de tiempo de subtítulos y la duración válida también son condiciones necesarias, para que reutilizar solo el audio no deje
    Edge a la cadena de subtítulos sin SubMaker.
    """
    if voice_mode != VOICE_MODE_TTS:
        return None

    script_content = str(params.video_script or "").strip()
    selected_tts_server = config.ui.get("tts_server", "azure-tts-v1")
    if (
        not script_content
        or not params.voice_name
        # El video real aplica el volumen de voz de forma unificada en la fase de composición de MoviePy; algunos Providers además
        # escriben la ganancia de volumen directamente en la fase TTS. Con un volumen no predeterminado, reutilizar la vista previa podría aplicar la ganancia dos veces,
        # así que se vuelve de forma conservadora al flujo original, sin introducir casos especiales por Provider para unos pocos escenarios.
        or not math.isclose(float(params.voice_volume), 1.0)
    ):
        return None

    expected_fingerprint = _voice_preview_fingerprint(
        preview_type="full",
        content=script_content,
        tts_server=selected_tts_server,
        voice_name=params.voice_name,
        voice_rate=params.voice_rate,
        voice_volume=params.voice_volume,
        provider_signature=_get_voice_preview_provider_signature(selected_tts_server),
    )
    cached_preview = st.session_state.get("voice_preview_audio")
    if (
        not cached_preview
        or cached_preview.get("fingerprint") != expected_fingerprint
        or cached_preview.get("preview_type") != "full"
        or not cached_preview.get("audio_bytes")
        or cached_preview.get("sub_maker") is None
    ):
        return None

    duration = cached_preview.get("duration")
    if (
        not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        return None

    return {
        "audio_bytes": bytes(cached_preview["audio_bytes"]),
        "audio_digest": hashlib.sha256(cached_preview["audio_bytes"]).hexdigest(),
        "duration": float(duration),
        "sub_maker": cached_preview["sub_maker"],
        "script": script_content,
        "voice_name": params.voice_name,
        "voice_rate": float(params.voice_rate),
        "voice_volume": float(params.voice_volume),
    }


def _sync_minimax_tts_api_key_input():
    """
    Sincroniza el control de contraseña de MiniMax TTS y devuelve la Key vigente.

    TTS Si la Key exclusiva está vacía, se permite reutilizar la MiniMax LLM Key. La Key compartida solo se usa para el control y la petición actuales,
    sin copiarse automáticamente a [minimax_tts], para no mantener la misma credencial duplicada en el archivo de configuración.
    """
    widget_key = "minimax_tts_api_key_input"
    configured_key = str(config.minimax_tts.get("api_key", "") or "").strip()
    shared_key = str(
        config.app.get("minimax_api_key", "") or os.getenv("MINIMAX_API_KEY", "") or ""
    ).strip()
    effective_key = configured_key or shared_key
    had_widget_state = widget_key in st.session_state
    entered_key = str(st.session_state.get(widget_key, "") or "").strip()

    if not entered_key and effective_key:
        # La reconexión del navegador puede reproducir un estado de contraseña vacía. Se restauran las credenciales configuradas, para que el vacío no sobrescriba la configuración,
        # asegurando a la vez que la petición de vista previa de este rerun pueda usar directamente la Key vigente.
        st.session_state[widget_key] = effective_key
        entered_key = effective_key
        if had_widget_state:
            logger.debug("restored MiniMax TTS API key after empty session replay")
    elif not had_widget_state:
        st.session_state[widget_key] = effective_key
        entered_key = effective_key

    if entered_key and entered_key != effective_key:
        _set_runtime_config("minimax_tts", "api_key", entered_key)

    return entered_key


def _get_cached_minimax_voices(api_key: str, endpoint: str) -> list[dict[str, str]]:
    """Lee, por sitio y resumen de credenciales, el resultado de la consulta de voces de MiniMax de la sesión actual."""
    cache = st.session_state.get("minimax_tts_voice_catalog_cache", {})
    cache_key = f"{endpoint}|{_credential_signature(api_key)}"
    cached_voices = cache.get(cache_key, [])
    return cached_voices if isinstance(cached_voices, list) else []


def _cache_minimax_voices(
    api_key: str,
    endpoint: str,
    voices: list[dict[str, str]],
):
    """Cachea las voces consultadas explícitamente, para no volver a pedirlas a MiniMax tras reruns de controles normales."""
    cache = st.session_state.setdefault("minimax_tts_voice_catalog_cache", {})
    cache_key = f"{endpoint}|{_credential_signature(api_key)}"
    cache[cache_key] = voices


def _render_minimax_tts_settings() -> tuple[list[str], dict[str, str]]:
    """Renderiza la configuración de MiniMax TTS y devuelve las opciones y textos que usa el selector de voces unificado."""
    effective_api_key = _sync_minimax_tts_api_key_input()
    effective_api_key = st.text_input(
        tr("MiniMax TTS API Key"),
        type="password",
        key="minimax_tts_api_key_input",
    ).strip()

    dedicated_key = str(config.minimax_tts.get("api_key", "") or "").strip()
    minimax_tts_endpoints = [voice.MINIMAX_TTS_GLOBAL_URL, voice.MINIMAX_TTS_CN_URL]
    effective_endpoint = voice.get_minimax_tts_endpoint()
    if effective_endpoint not in minimax_tts_endpoints:
        effective_endpoint = voice.MINIMAX_TTS_GLOBAL_URL
    minimax_tts_base_url = stable_selectbox(
        tr("MiniMax TTS Endpoint"),
        options=minimax_tts_endpoints,
        default_value=effective_endpoint,
        key="minimax_tts_endpoint_select",
        # Al reutilizar la LLM Key hay que seguir la región del LLM, para que la interfaz no permita elegir una dirección que en realidad
        # no funcionaría; tras introducir una TTS Key independiente se puede elegir el sitio por separado.
        disabled=not dedicated_key,
    )
    if dedicated_key:
        _set_runtime_config("minimax_tts", "base_url", minimax_tts_base_url)

    configured_model = config.minimax_tts.get(
        "model_id", voice.MINIMAX_TTS_DEFAULT_MODEL
    )
    if configured_model not in voice.MINIMAX_TTS_MODELS:
        configured_model = voice.MINIMAX_TTS_DEFAULT_MODEL
    minimax_tts_model = stable_selectbox(
        tr("MiniMax TTS Model"),
        options=list(voice.MINIMAX_TTS_MODELS),
        default_value=configured_model,
        key="minimax_tts_model_select",
    )
    _set_runtime_config("minimax_tts", "model_id", minimax_tts_model)

    if st.button(
        tr("Load MiniMax Voices"),
        key="load_minimax_voices_button",
        icon=":material/refresh:",
        use_container_width=True,
    ):
        try:
            available_voices = voice.get_minimax_voice_catalog(
                api_key=effective_api_key,
                endpoint=minimax_tts_base_url,
                voice_type="all",
            )
        except Exception as exc:
            # Aquí hay que mostrar la excepción al usuario y registrarla en el log. Que la región de la cuenta no coincida, que la Key no tenga permisos
            # o que falle la red son casos habituales, y devolver en silencio una lista vacía haría creer al usuario que la cuenta no tiene voces.
            logger.warning(f"load MiniMax voices failed: {exc}")
            st.error(tr("MiniMax Voices Load Failed").format(error=str(exc)))
        else:
            _cache_minimax_voices(
                effective_api_key,
                minimax_tts_base_url,
                available_voices,
            )
            st.success(tr("MiniMax Voices Loaded").format(count=len(available_voices)))

    available_voices = _get_cached_minimax_voices(
        effective_api_key,
        minimax_tts_base_url,
    )
    voice_labels = {
        f"minimax:{item['voice_id']}": (
            f"{item['voice_name']} ({item['voice_id']})"
            if item["voice_name"] != item["voice_id"]
            else item["voice_id"]
        )
        for item in available_voices
    }
    configured_voice_id = str(
        config.minimax_tts.get("voice_id", voice.MINIMAX_TTS_DEFAULT_VOICE)
        or voice.MINIMAX_TTS_DEFAULT_VOICE
    ).strip()
    configured_voice = f"minimax:{configured_voice_id}"
    # Si aún no se ha pulsado obtener voces, la interfaz no está disponible temporalmente o la configuración usa voces clonadas fuera de la lista, se conserva
    # el Voice ID actual, para que el flujo de generación original no dependa del resultado de la consulta remota de voces.
    voice_labels.setdefault(configured_voice, configured_voice_id)
    return list(voice_labels), voice_labels


def _sync_elevenlabs_api_key_input():
    """
    Sincroniza el control de contraseña de ElevenLabs, la configuración persistente y las variables de entorno, y devuelve la Key vigente.

    Streamlit Cuando una pestaña del navegador se conecta al servicio reiniciado, puede reproducir un estado vacío del control de contraseña
    . Ese vacío no se puede distinguir de forma fiable de un borrado intencionado del usuario, así que, si el archivo de configuración o las variables de entorno aún tienen
    Key , se prioriza restaurar el valor vigente, para que el estado vacío no sobrescriba la configuración y este rerun pueda cargar de inmediato
    las voces. Para eliminar la Key por completo hay que modificar el archivo de configuración o las variables de entorno, evitando errores por reconexión.
    """
    widget_key = "elevenlabs_api_key_input"
    configured_key = str(config.elevenlabs.get("api_key", "") or "").strip()
    env_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    effective_key = configured_key or env_key
    had_widget_state = widget_key in st.session_state
    entered_key = str(st.session_state.get(widget_key, "") or "").strip()

    if not entered_key and effective_key:
        # El estado vacío tras reconectar no puede sobrescribir credenciales vigentes, y debe restaurarse antes de renderizar la lista de voces,
        # o, aunque el archivo de configuración no se haya vaciado, la página actual seguiría pidiendo a ElevenLabs con una Key vacía.
        st.session_state[widget_key] = effective_key
        entered_key = effective_key
        if had_widget_state:
            logger.debug("restored ElevenLabs API key after empty session replay")
    elif not had_widget_state:
        # Se inicializa antes de crear el control, para no pasar a la vez value y session_state y provocar Streamlit
        # la advertencia de conflicto de valores por defecto; sin ninguna Key basta con inicializar a vacío.
        st.session_state[widget_key] = entered_key

    if entered_key and entered_key != effective_key:
        # Solo los valores nuevos introducidos por el usuario van a config.toml. Cuando una variable de entorno se rellena como valor vigente no
        # se copia al archivo, y las claves inyectadas por el contenedor o la plataforma de despliegue se quedan solo en el entorno de ejecución.
        for cache_key in list(st.session_state.keys()):
            if str(cache_key).startswith("elevenlabs_voices_"):
                del st.session_state[cache_key]
        _set_runtime_config("elevenlabs", "api_key", entered_key)

    return entered_key


def _sync_voxcpm_api_key_input():
    """Restaura el estado vacío que Streamlit reproduce en el control de contraseña de VoxCPM al reconectar."""
    widget_key = "voxcpm_api_key_input"
    configured_key = str(config.voxcpm.get("api_key", "") or "").strip()
    had_widget_state = widget_key in st.session_state
    entered_key = str(st.session_state.get(widget_key, "") or "").strip()

    if not entered_key and configured_key:
        # La reconexión del navegador puede reproducir un estado de contraseña vacía. Se conservan las credenciales guardadas, para que esta vez no se rerun
        # sobrescriba con vacío, mediante _set_runtime_config, la Key vigente de config.toml.
        st.session_state[widget_key] = configured_key
        entered_key = configured_key
        if had_widget_state:
            logger.debug("restored VoxCPM API key after empty session replay")
    elif not had_widget_state:
        st.session_state[widget_key] = entered_key

    return entered_key


def _render_elevenlabs_api_key_input(label_key):
    """
    Renderiza el único estado de entrada de API Key que comparten ElevenLabs TTS y la música.

    Si en la misma página el TTS y la música usaran dos widget key distintas, Streamlit conservaría el valor antiguo de cada una,
    y el campo renderizado después sobrescribiría además la configuración compartida. Aquí se usa una sola key, y se gestionan de forma centralizada el relleno desde variables de entorno
    , la actualización de la configuración y la invalidación de la caché de voces, para que la interfaz y las tareas en segundo plano lean siempre el mismo valor.
    """
    _sync_elevenlabs_api_key_input()
    return st.text_input(
        tr(label_key),
        type="password",
        key="elevenlabs_api_key_input",
    ).strip()


def _render_background_music_settings(params, elevenlabs_api_key_rendered=False):
    """Renderiza el origen y el volumen de la música de fondo, y devuelve el archivo subido pendiente de guardar esta vez."""
    uploaded_bgm_file = None
    previous_bgm_type = st.session_state.get("last_rendered_bgm_type")
    st.divider()
    bgm_options = [
        (tr("No Background Music"), ""),
        (tr("Random Background Music"), "random"),
        (tr("Preset Song"), "preset"),
        (tr("Custom Background Music"), "custom"),
        (tr("Sonilo Background Music"), "sonilo"),
        (tr("ElevenLabs Background Music"), "elevenlabs"),
    ]
    selected_bgm_type = stable_selectbox(
        tr("Background Music Source"),
        options=[value for _, value in bgm_options],
        default_value=_saved_ui_choice(
            "bgm_type",
            [value for _, value in bgm_options],
            "random",
        ),
        key="bgm_type_select",
        format_func=lambda value: dict((v, label) for label, v in bgm_options)[value],
    )
    params.bgm_type = selected_bgm_type
    _set_runtime_config("ui", "bgm_type", params.bgm_type)
    if params.bgm_type == "sonilo":
        configured_key = str(config.app.get("sonilo_api_key", "") or "").strip()
        effective_key = configured_key or os.getenv("SONILO_API_KEY", "").strip()
        entered_key = st.text_input(
            tr("Sonilo API Key"),
            value=effective_key,
            type="password",
            key="sonilo_api_key_input",
        ).strip()
        # El usuario pidió que la Key configurada se rellene directamente en el campo de contraseña. El valor de configuración tiene prioridad sobre las variables de entorno;
        # solo se escribe de vuelta si el usuario realmente cambió la entrada o ya usaba la configuración, para no copiar sin intervención Key
        # el de las variables de entorno a config.toml.
        if configured_key or entered_key != effective_key:
            _set_runtime_config("app", "sonilo_api_key", entered_key)
    elif params.bgm_type == "elevenlabs":
        if elevenlabs_api_key_rendered:
            # TTS Si la región ya renderizó el campo compartido, no se crea un segundo widget, para que dos valores independientes
            # session_state no se sobrescriban entre sí. El texto explicativo ayuda al usuario a localizar la configuración compartida de arriba.
            st.caption(tr("ElevenLabs API Key Help"))
        else:
            _render_elevenlabs_api_key_input("ElevenLabs Music API Key")

    bgm_volume_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    params.bgm_volume = stable_selectbox(
        tr("Background Music Volume"),
        options=bgm_volume_options,
        default_value=_saved_ui_choice("bgm_volume", bgm_volume_options, 0.2),
        key="bgm_volume_select",
        format_func=lambda value: f"{int(value * 100)}%",
        disabled=not params.bgm_type,
    )
    _set_runtime_config("ui", "bgm_volume", params.bgm_volume)
    bgm_enabled = bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)

    if params.bgm_type == "custom":
        uploaded_bgm_file = st.file_uploader(
            tr("Upload Background Music"),
            type=[
                extension.removeprefix(".")
                for extension in bgm_service.SUPPORTED_BGM_EXTENSIONS
            ],
            accept_multiple_files=False,
            key="custom_bgm_uploader",
            help=tr("Upload Background Music Help"),
            # Streamlit por defecto muestra en el control el límite global de 200MB. Aquí debe coincidir con el
            # 30MB límite estricto de la capa de servicio, para que la interfaz no permita elegir algo que el servidor rechazará al enviar.
            max_upload_size=bgm_service.MAX_BGM_UPLOAD_BYTES // (1024 * 1024),
        )
        if uploaded_bgm_file is not None and bgm_enabled:
            try:
                safe_name = bgm_service.sanitize_upload_filename(uploaded_bgm_file.name)
                # Streamlit vuelve a ejecutar la página tras ajustar cualquier control, como el volumen. Se usa un hash del contenido
                # para distinguir los archivos subidos y se cachea en la sesión actual el resultado completo de la decodificación, para no reutilizar un resultado antiguo solo por coincidir el nombre
                # y el tamaño del archivo, y evitar a la vez llamar a FFmpeg en cada rerun.
                validation_key = (
                    safe_name,
                    uploaded_bgm_file.size,
                    hashlib.sha256(uploaded_bgm_file.getbuffer()).hexdigest(),
                )
                cached_validation = st.session_state.get("custom_bgm_validation")
                if (
                    not cached_validation
                    or cached_validation.get("key") != validation_key
                ):
                    try:
                        bgm_service.validate_bgm_upload(
                            uploaded_bgm_file.name, uploaded_bgm_file
                        )
                    except bgm_service.BgmUploadError as exc:
                        cached_validation = {
                            "key": validation_key,
                            "error": str(exc),
                            "error_type": "upload",
                        }
                        # Los resultados fallidos de una misma huella de archivo entran en la caché de sesión, así que aquí solo
                        # se registra una vez, en la primera validación real, para que los reruns de controles normales no llenen la pantalla.
                        logger.warning(
                            "WebUI background music validation rejected: "
                            f"name={safe_name}, error={str(exc)}"
                        )
                    except bgm_service.BgmServiceError as exc:
                        cached_validation = {
                            "key": validation_key,
                            "error": str(exc),
                            "error_type": "service",
                        }
                        logger.error(
                            "WebUI background music validation failed: "
                            f"name={safe_name}, error={str(exc)}"
                        )
                    else:
                        cached_validation = {
                            "key": validation_key,
                            "error": "",
                            "error_type": "",
                        }
                    st.session_state["custom_bgm_validation"] = cached_validation

                if cached_validation.get("error"):
                    if cached_validation.get("error_type") == "service":
                        raise bgm_service.BgmServiceError(cached_validation["error"])
                    raise bgm_service.BgmUploadError(cached_validation["error"])
            except bgm_service.BgmUploadError:
                # Un archivo ilegal no puede reutilizar el nombre de la última subida válida, o los parámetros de la tarea podrían seguir apuntando
                # a la BGM histórica. Se conserva el valor devuelto de UploadedFile, para que al pulsar generar lo bloquee la validación final
                # del servidor, en lugar de generar en silencio un video sin música de fondo.
                params.bgm_file = ""
                st.error(tr("Invalid Background Music"))
            except bgm_service.BgmServiceError:
                params.bgm_file = ""
                st.error(tr("Background Music Validation Failed"))
            else:
                # Solo tras pasar la validación completa de decodificación se muestran el reproductor y «Listo». El archivo sigue persistiéndose solo al pulsar
                # generar, y si el usuario solo previsualiza o quita después el archivo, no se ensucia storage/bgm.
                uploaded_mime_type = str(getattr(uploaded_bgm_file, "type", "") or "")
                preview_mime_type = (
                    uploaded_mime_type
                    if uploaded_mime_type.startswith("audio/")
                    else mimetypes.guess_type(safe_name)[0] or "audio/mpeg"
                )
                st.audio(uploaded_bgm_file, format=preview_mime_type)
                st.info(f"{tr('Background Music Ready')}: {safe_name}")
                params.bgm_file = safe_name

        # Streamlit limpia el widget state de los controles condicionales cuando dejan de renderizarse temporalmente.
        # Al volver desde otro origen de BGM se restaura con el valor persistido; dentro del mismo origen,
        # cuando el usuario lo vacía, previous_bgm_type no cambia, así que el valor antiguo no rebota.
        if previous_bgm_type != "custom":
            st.session_state["custom_bgm_file_input"] = _saved_ui_text(
                "custom_bgm_file"
            )
        custom_bgm_file = st.text_input(
            tr("Custom Background Music File"),
            key="custom_bgm_file_input",
            disabled=uploaded_bgm_file is not None,
        )
        _set_runtime_config(
            "ui", "custom_bgm_file", custom_bgm_file.strip()
        )
        if uploaded_bgm_file is None and custom_bgm_file and bgm_enabled:
            # El nombre de archivo lo mapea la capa de servicio a storage/bgm o resource/songs y luego lo valida,
            # UI sin aceptar rutas arbitrarias fuera de los dos directorios de lista blanca.
            params.bgm_file = custom_bgm_file.strip()
        elif not bgm_enabled:
            # El control de subida sigue conservando el archivo que eligió el usuario, y el siguiente rerun tras subir el volumen hará automáticamente
            # la validación completa; los parámetros de la tarea actual deben vaciarse, para que una tarea con volumen 0 no guarde ni analice ese archivo.
            params.bgm_file = ""

    if params.bgm_type == "preset":
        # La capa de servicio ya hace de forma unificada la validación de extensiones, archivos temporales y enlaces simbólicos. Aquí se reutiliza directamente su
        # resultado, para que la UI no mantenga un segundo conjunto de reglas de enumeración y no haya diferencias al añadir formatos en el futuro.
        available_song_paths = bgm_service.list_builtin_bgm_files()
        songs_by_name = {
            os.path.basename(song_path): song_path for song_path in available_song_paths
        }
        available_songs = list(songs_by_name)
        if not available_songs:
            st.warning(tr("No Background Music Available"))
            params.bgm_file = ""
        else:
            default_preset_song = _saved_ui_text("preset_song", available_songs[0])
            requested_preset_song = st.session_state.get(
                localized_widget_key("preset_song_select"), default_preset_song
            )
            if requested_preset_song not in available_songs:
                # Las tareas históricas o los ajustes exportados por otras versiones pueden referirse a canciones que no existen en la instalación actual.
                # Tras un aviso explícito, stable_selectbox vuelve a la primera canción, para no cambiar de canción en silencio.
                st.warning(tr("Selected Background Music Unavailable"))
            selected_song = stable_selectbox(
                tr("Preset Song"),
                options=available_songs,
                default_value=(
                    default_preset_song
                    if default_preset_song in available_songs
                    else available_songs[0]
                ),
                key="preset_song_select",
            )
            _set_runtime_config("ui", "preset_song", selected_song)
            # Tras elegir una canción se ofrece de inmediato la escucha en línea. El reproductor lee la ruta real recién obtenida tras la validación de lista blanca
            # de la capa de servicio, sin aceptar rutas de archivo arbitrarias introducidas en la página.
            selected_song_path = songs_by_name[selected_song]
            preview_mime_type = (
                mimetypes.guess_type(selected_song_path)[0] or "audio/mpeg"
            )
            preview_available = True
            try:
                # Streamlit al fallar la lectura de la ruta envuelve OSError en una excepción interna, lo que hace que lo de abajo
                # no pueda tratarlo como error de archivo. Leer primero los bytes por cuenta propia mantiene el comportamiento del reproductor y hace que
                # Docker los fallos temporales de montaje o los cambios de permisos caigan de forma estable en una rama controlada.
                selected_song_bytes = Path(selected_song_path).read_bytes()
            except OSError as exc:
                preview_available = False
                # El archivo puede borrarlo otro proceso después de enumerarlo. Un fallo de escucha no puede interrumpir la página ni la edición de parámetros
                # del video, pero hay que conservar el log para localizar problemas del entorno de ejecución y del montaje.
                logger.warning(
                    "failed to preview preset background music: "
                    f"name={selected_song}, error={str(exc)}"
                )
                st.warning(tr("Background Music Preview Failed"))
            else:
                st.audio(selected_song_bytes, format=preview_mime_type)
            if bgm_enabled and preview_available:
                params.bgm_file = selected_song
            else:
                params.bgm_file = ""

    if params.bgm_type == "sonilo":
        if previous_bgm_type != "sonilo":
            st.session_state["sonilo_bgm_prompt_input"] = _saved_ui_text(
                "sonilo_bgm_prompt",
                max_length=sonilo_service.MAX_PROMPT_LENGTH,
            )
        params.video_music_prompt = st.text_input(
            tr("Sonilo Music Prompt"),
            key="sonilo_bgm_prompt_input",
            max_chars=sonilo_service.MAX_PROMPT_LENGTH,
            help=tr("Sonilo Music Prompt Help"),
        ).strip()
        _set_runtime_config(
            "ui", "sonilo_bgm_prompt", params.video_music_prompt
        )
        if params.video_count > 1:
            st.warning(tr("Sonilo Multiple Videos Warning"))
        if st.button(
            tr("Test Sonilo Connection"),
            key="test_sonilo_connection_button",
            use_container_width=True,
        ):
            try:
                sonilo_service.test_connection()
            except sonilo_service.SoniloError as exc:
                logger.warning(f"Sonilo connection test failed: {exc}")
                st.error(tr("Sonilo Connection Test Failed").format(error=str(exc)))
            else:
                st.success(tr("Sonilo Connection Test Succeeded"))
    elif params.bgm_type == "elevenlabs":
        if previous_bgm_type != "elevenlabs":
            st.session_state["elevenlabs_music_prompt_input"] = _saved_ui_text(
                "elevenlabs_music_prompt",
                max_length=elevenlabs_music_service.MAX_PROMPT_LENGTH,
            )
        params.video_music_prompt = st.text_input(
            tr("ElevenLabs Music Prompt"),
            key="elevenlabs_music_prompt_input",
            max_chars=elevenlabs_music_service.MAX_PROMPT_LENGTH,
            help=tr("ElevenLabs Music Prompt Help"),
        ).strip()
        _set_runtime_config(
            "ui", "elevenlabs_music_prompt", params.video_music_prompt
        )
        if params.video_count > 1:
            st.warning(tr("ElevenLabs Multiple Videos Warning"))
        if st.button(
            tr("Test ElevenLabs Connection"),
            key="test_elevenlabs_music_connection_button",
            use_container_width=True,
        ):
            try:
                elevenlabs_music_service.test_connection()
            except elevenlabs_music_service.ElevenLabsPaidPlanRequiredError:
                st.error(tr("ElevenLabs Paid Plan Required"))
            except elevenlabs_music_service.ElevenLabsMusicError as exc:
                logger.warning(f"ElevenLabs connection test failed: {exc}")
                st.error(tr("ElevenLabs Connection Test Failed").format(error=str(exc)))
            else:
                st.success(tr("ElevenLabs Connection Test Succeeded"))
    if params.bgm_type == "sonilo" and bgm_enabled and not sonilo_service.is_enabled():
        # Con el volumen a 0 la capa de tareas no genera ni mezcla la música de Sonilo, así que no hace falta avisar de la Key;
        # esta comprobación comparte con el punto de entrada de tareas la regla de la capa de servicio, para que el aviso de la interfaz y la condición real de ejecución no diverjan.
        st.warning(tr("Sonilo API Key Required"))
    elif (
        params.bgm_type == "elevenlabs"
        and bgm_enabled
        and not elevenlabs_music_service.is_enabled()
    ):
        st.warning(tr("ElevenLabs API Key Required"))
    st.session_state["last_rendered_bgm_type"] = params.bgm_type
    return uploaded_bgm_file


def _voice_rate_not_supported_help(tts_server, voice_name):
    """
    Return the help for the disabled speed selector when the provider has no
    numeric speed parameter (Gemini TTS, MiMo TTS and VoxCPM), or "" when it
    applies the speed. The disabled selector keeps the saved speed.
    """
    if tts_server == "gemini-tts" or (voice_name and voice.is_gemini_voice(voice_name)):
        return tr("Gemini TTS Speed Not Supported")
    if tts_server == "mimo-tts" or (voice_name and voice.is_mimo_voice(voice_name)):
        return tr("MiMo TTS Speed Not Supported")
    if tts_server == "voxcpm" or (voice_name and voice.is_voxcpm_voice(voice_name)):
        return tr("VoxCPM Speed Not Supported")
    return ""


def _render_audio_settings(panel, params):
    """Renderiza los ajustes de audio y devuelve el audio subido y el modo de voz actual."""
    with panel:
        with st.container(border=True):
            st.write(tr("Audio Settings"))

            # El modo de voz es el estado de primer nivel de los ajustes de audio, y distingue claramente voz automática, subida por el usuario y sin voz.
            # Cuando una configuración antigua no tiene voice_mode, se mantiene la compatibilidad según el centinela sin voz del tts_server original.
            saved_tts_server = config.ui.get("tts_server", "azure-tts-v1")
            saved_voice_mode = config.ui.get("voice_mode")
            if saved_voice_mode not in {
                VOICE_MODE_TTS,
                VOICE_MODE_UPLOAD,
                VOICE_MODE_NONE,
            }:
                saved_voice_mode = (
                    VOICE_MODE_NONE
                    if saved_tts_server == voice.NO_VOICE_NAME
                    else VOICE_MODE_TTS
                )
            voice_mode_options = [VOICE_MODE_TTS, VOICE_MODE_UPLOAD, VOICE_MODE_NONE]
            voice_mode_labels = {
                VOICE_MODE_TTS: tr("Automatic Voiceover"),
                VOICE_MODE_UPLOAD: tr("Upload Voiceover"),
                VOICE_MODE_NONE: tr("No Voiceover"),
            }
            voice_mode = stable_segmented_control(
                tr("Voiceover Mode"),
                options=voice_mode_options,
                default_value=saved_voice_mode,
                key="voice_mode_control",
                format_func=lambda value: voice_mode_labels[value],
                width="stretch",
            )
            _set_runtime_config("ui", "voice_mode", voice_mode)
            tts_mode_enabled = voice_mode == VOICE_MODE_TTS

            # Provider El desplegable solo elige el servicio de voz automática; el modo sin voz ya lo controla el modo de arriba,
            # y ya no se mezcla en la lista como TTS Provider, para que dos accesos no expresen el mismo estado.
            tts_servers = [
                ("azure-tts-v1", "Azure TTS V1 (Edge TTS)"),
                ("azure-tts-v2", "Azure TTS V2"),
                ("siliconflow", "SiliconFlow TTS"),
                ("gemini-tts", "Google Gemini TTS"),
                ("mimo-tts", "Xiaomi MiMo TTS"),
                ("minimax-tts", "MiniMax TTS"),
                ("elevenlabs", "ElevenLabs TTS"),
                ("chatterbox", "Chatterbox TTS"),
                ("kokoro", "Kokoro TTS"),
                ("fish_audio", "Fish Audio TTS"),
                ("voxcpm", "VoxCPM TTS"),
            ]

            tts_server_values = [server_value for server_value, _ in tts_servers]
            if saved_tts_server not in tts_server_values:
                saved_tts_server = "azure-tts-v1"

            if tts_mode_enabled:
                selected_tts_server = stable_selectbox(
                    tr("Voiceover Service"),
                    options=tts_server_values,
                    default_value=saved_tts_server,
                    key="tts_server_select",
                    format_func=lambda value: dict(
                        (v, label) for v, label in tts_servers
                    )[value],
                )
            else:
                # Los modos no automáticos no renderizan los controles TTS, pero conservan la última elección para seguir usándola al volver.
                selected_tts_server = saved_tts_server

            _set_runtime_config("ui", "tts_server", selected_tts_server)

            # La descripción del servicio va justo tras elegir el Provider: primero se dice al usuario qué debe preparar, y luego se pasa a la voz y
            # la configuración de credenciales. Los Providers sin descripción no renderizan un bloque de aviso vacío.
            if tts_mode_enabled:
                provider_tips = get_tts_provider_tips(selected_tts_server)
                if provider_tips:
                    st.info(provider_tips)

            # MiniMax solo reutiliza el selector genérico «Voz» de abajo. La función de configuración del Provider se encarga de
            # actualizar las voces remotas y devolver textos amigables, sin renderizar además Voice ID ni un desplegable de voces.
            minimax_voices = []
            minimax_voice_labels = {}
            if tts_mode_enabled and selected_tts_server == "minimax-tts":
                minimax_voices, minimax_voice_labels = _render_minimax_tts_settings()

            # Obtener la lista de voces según el servidor TTS elegido
            filtered_voices = []
            saved_voice_name = config.ui.get("voice_name", "")
            elevenlabs_api_key_rendered = False

            if not tts_mode_enabled:
                # Los modos de audio subido y sin voz no cargan voces remotas, reduciendo peticiones de red inútiles y ruido en la interfaz.
                filtered_voices = []
            elif selected_tts_server == "siliconflow":
                # Obtiene la lista de voces de SiliconFlow
                filtered_voices = voice.get_siliconflow_voices()
            elif selected_tts_server == "gemini-tts":
                # Obtener la lista de voces de Gemini TTS
                filtered_voices = voice.get_gemini_voices()
            elif selected_tts_server == "mimo-tts":
                # Obtener la lista de voces predefinidas de Xiaomi MiMo TTS
                filtered_voices = voice.get_mimo_voices()
            elif selected_tts_server == "minimax-tts":
                filtered_voices = minimax_voices
            elif selected_tts_server == "elevenlabs":
                # La lista de voces se renderiza antes del campo de Key, así que primero hay que restaurar de forma unificada el estado de reconexión y leer
                # la configuración/variables de entorno, o la página cargaría con una Key vacía y cachearía una lista de voces vacía.
                saved_elevenlabs_api_key = _sync_elevenlabs_api_key_input()
                cache_key = f"elevenlabs_voices_{saved_elevenlabs_api_key}"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = voice.get_elevenlabs_voices(
                        saved_elevenlabs_api_key
                    )
                filtered_voices = st.session_state[cache_key]
            elif selected_tts_server == "chatterbox":
                # Voces predefinidas del servicio Chatterbox autoalojado (de la configuración [chatterbox] voices)
                _sync_chatterbox_config_from_session_state()
                filtered_voices = voice.get_chatterbox_voices()
            elif selected_tts_server == "kokoro":
                # Voces del servicio Kokoro autoalojado: si [kokoro] voices está vacío se leen de /audio/voices del servidor
                _sync_kokoro_config_from_session_state()
                filtered_voices = _get_kokoro_voice_options(saved_voice_name)
            elif selected_tts_server == "fish_audio":
                filtered_voices = voice.get_fish_audio_voices()
            elif selected_tts_server == "voxcpm":
                filtered_voices = voice.get_voxcpm_voices()
            else:
                # Obtener la lista de voces de Azure
                all_voices = voice.get_all_azure_voices(filter_locals=None)

                # Filtrar las voces según el servidor TTS elegido
                for v in all_voices:
                    if selected_tts_server == "azure-tts-v2":
                        # V2el nombre de voz de la versión contiene"v2"
                        if "V2" in v:
                            filtered_voices.append(v)
                    else:
                        # V1el nombre de voz de la versión no contiene"v2"
                        if "V2" not in v:
                            filtered_voices.append(v)

            def _friendly(v):
                if voice.is_no_voice(v):
                    return tr("No Voice Selected")
                if voice.is_elevenlabs_voice(v):
                    parts = v.split(":", 2)
                    return parts[2] if len(parts) >= 3 else v
                if voice.is_chatterbox_voice(v) or voice.is_kokoro_voice(v):
                    name = v.split(":", 1)[1] if ":" in v else v
                    return name.replace("-Female", "").replace("-Male", "")
                if voice.is_minimax_voice(v):
                    return minimax_voice_labels.get(v, v.split(":", 1)[1])
                if voice.is_fish_audio_voice(v):
                    parts = v.split(":", 2)
                    display_name = parts[2] if len(parts) >= 3 else v
                    return (
                        display_name.replace("Female", tr("Female"))
                        .replace("Male", tr("Male"))
                    )
                if voice.is_voxcpm_voice(v):
                    return v.split(":", 1)[1] or DEFAULT_VOXCPM_VOICE
                return (
                    v.replace("Female", tr("Female"))
                    .replace("Male", tr("Male"))
                    .replace("Neural", "")
                )

            friendly_names = {v: _friendly(v) for v in filtered_voices}

            # Gemini El catálogo antiguo ponía el género deducido en el valor (por ejemplo, Charon-Male). Se mapea por el nombre base
            # voice name al nuevo valor de estilo oficial, para conservar la voz original del usuario tras actualizar.
            if (
                selected_tts_server == "gemini-tts"
                and saved_voice_name not in friendly_names
            ):
                saved_gemini_voice = voice.parse_gemini_voice_name(saved_voice_name)
                saved_voice_name = next(
                    (
                        candidate
                        for candidate in filtered_voices
                        if voice.parse_gemini_voice_name(candidate)
                        == saved_gemini_voice
                    ),
                    saved_voice_name,
                )

            saved_voice_name_index = 0

            # Comprobar si la voz guardada está en la lista de voces filtrada actual
            if saved_voice_name in friendly_names:
                saved_voice_name_index = list(friendly_names.keys()).index(
                    saved_voice_name
                )
            else:
                # Si no está, elegir una voz por defecto según el idioma actual de la UI
                for i, v in enumerate(filtered_voices):
                    if v.lower().startswith(st.session_state["ui_language"].lower()):
                        saved_voice_name_index = i
                        break

            # Si no se encuentra una voz que coincida, usar la primera
            if saved_voice_name_index >= len(friendly_names) and friendly_names:
                saved_voice_name_index = 0

            # Asegurar que haya voces para elegir
            if tts_mode_enabled and friendly_names:
                voice_name = stable_selectbox(
                    tr("Voiceover Voice"),
                    options=list(friendly_names.keys()),
                    default_value=list(friendly_names.keys())[saved_voice_name_index],
                    key=f"speech_synthesis_select_{selected_tts_server}",
                    format_func=lambda value: friendly_names.get(
                        value,
                        str(value).removeprefix("minimax:"),
                    ),
                    # MiniMax permite al usuario introducir directamente IDs de voces clonadas o generadas que no están en la lista; los demás
                    # Provider mantienen el comportamiento original del selector, sin ampliar el alcance de este cambio.
                    accept_new_options=selected_tts_server == "minimax-tts",
                )

                if selected_tts_server == "minimax-tts":
                    custom_voice_id = str(voice_name or "").strip()
                    if custom_voice_id and not voice.is_minimax_voice(custom_voice_id):
                        voice_name = f"minimax:{custom_voice_id}"
                    if voice.is_minimax_voice(voice_name):
                        _set_runtime_config(
                            "minimax_tts",
                            "voice_id",
                            voice_name.split(":", 1)[1],
                        )

                params.voice_name = voice_name
                if not voice.is_no_voice(voice_name):
                    # El sentinel de marcador solo se usa para la visualización deshabilitada de los modos no automáticos, sin sobrescribir la última voz
                    # que eligió realmente el usuario, que se puede restaurar al volver a la voz automática.
                    _set_runtime_config("ui", "voice_name", voice_name)
            elif tts_mode_enabled:
                # Si no hay voces para elegir, mostrar un aviso
                st.warning(
                    tr(
                        "No voices available for the selected TTS server. Please select another server."
                    )
                )
                voice_name = ""
                params.voice_name = ""
                _set_runtime_config("ui", "voice_name", "")
            else:
                # Los modos de voz no automáticos no muestran el control de voz, y solo reutilizan el valor guardado para mantener estable la estructura de parámetros.
                voice_name = saved_voice_name or voice.NO_VOICE_NAME
                params.voice_name = voice_name

            # Cuando se elige la versión V2 o la voz es V2, mostrar la región del servicio y el campo de API key
            if tts_mode_enabled and (
                selected_tts_server == "azure-tts-v2"
                or (voice_name and voice.is_azure_v2_voice(voice_name))
            ):
                saved_azure_speech_region = config.azure.get("speech_region", "")
                saved_azure_speech_key = config.azure.get("speech_key", "")
                azure_speech_region = st.text_input(
                    tr("Speech Region"),
                    value=saved_azure_speech_region,
                    key="azure_speech_region_input",
                )
                azure_speech_key = st.text_input(
                    tr("Speech Key"),
                    value=saved_azure_speech_key,
                    type="password",
                    key="azure_speech_key_input",
                )
                _set_runtime_config("azure", "speech_region", azure_speech_region)
                _set_runtime_config("azure", "speech_key", azure_speech_key)

            if tts_mode_enabled and selected_tts_server == "gemini-tts":
                # Gemini TTS Comparte la misma clave que Gemini LLM; el panel de audio ofrece un acceso directo,
                # y el usuario no tiene que cambiar antes de LLM Provider para completar la configuración de voz.
                gemini_tts_api_key = st.text_input(
                    tr("Gemini API Key"),
                    value=config.app.get("gemini_api_key", ""),
                    type="password",
                    key="gemini_tts_api_key_input",
                )
                _set_runtime_config("app", "gemini_api_key", gemini_tts_api_key)

            # Cuando se elige SiliconFlow, mostrar el campo de API key y la descripción
            if tts_mode_enabled and (
                selected_tts_server == "siliconflow"
                or (voice_name and voice.is_siliconflow_voice(voice_name))
            ):
                saved_siliconflow_api_key = config.siliconflow.get("api_key", "")

                siliconflow_api_key = st.text_input(
                    tr("SiliconFlow API Key"),
                    value=saved_siliconflow_api_key,
                    type="password",
                    key="siliconflow_api_key_input",
                )

                _set_runtime_config("siliconflow", "api_key", siliconflow_api_key)

            # Cuando se elige Xiaomi MiMo TTS, reutilizar la API Key del LLM provider de MiMo.
            # Así, si el usuario usa MiMo tanto para el texto como para la voz, solo mantiene una clave.
            if tts_mode_enabled and (
                selected_tts_server == "mimo-tts"
                or (voice_name and voice.is_mimo_voice(voice_name))
            ):
                saved_mimo_api_key = config.app.get("mimo_api_key", "")

                mimo_api_key = st.text_input(
                    tr("MiMo API Key"),
                    value=saved_mimo_api_key,
                    type="password",
                    key="mimo_tts_api_key_input",
                )

                _set_runtime_config("app", "mimo_api_key", mimo_api_key)

            # ElevenLabs API key section
            if tts_mode_enabled and (
                selected_tts_server == "elevenlabs"
                or (voice_name and voice.is_elevenlabs_voice(voice_name))
            ):
                _render_elevenlabs_api_key_input(
                    "ElevenLabs API Key",
                )
                elevenlabs_api_key_rendered = True

                _elevenlabs_models = [
                    "eleven_multilingual_v2",
                    "eleven_flash_v2_5",
                    "eleven_v3",
                ]
                saved_elevenlabs_model = config.elevenlabs.get(
                    "model_id", "eleven_multilingual_v2"
                )
                if saved_elevenlabs_model not in _elevenlabs_models:
                    saved_elevenlabs_model = "eleven_multilingual_v2"
                elevenlabs_model = stable_selectbox(
                    tr("ElevenLabs Model"),
                    options=_elevenlabs_models,
                    default_value=saved_elevenlabs_model,
                    key="elevenlabs_model_select",
                )
                _set_runtime_config("elevenlabs", "model_id", elevenlabs_model)

            # Fish Audio API settings section
            if tts_mode_enabled and (
                selected_tts_server == "fish_audio"
                or (voice_name and voice.is_fish_audio_voice(voice_name))
            ):
                saved_fish_api_key = (
                    config.fish_audio.get("api_key", "")
                    if hasattr(config, "fish_audio") and isinstance(config.fish_audio, dict)
                    else ""
                )
                fish_audio_api_key = st.text_input(
                    tr("Fish Audio API Key"),
                    value=saved_fish_api_key,
                    type="password",
                    key="fish_audio_api_key_input",
                )
                _set_runtime_config("fish_audio", "api_key", fish_audio_api_key)

                _fish_audio_models = [
                    "s2.1-pro-free",
                    "s2.1-pro",
                    "s2-pro",
                ]
                saved_fish_model = (
                    config.fish_audio.get("model", "s2.1-pro-free")
                    if hasattr(config, "fish_audio") and isinstance(config.fish_audio, dict)
                    else "s2.1-pro-free"
                )
                if saved_fish_model not in _fish_audio_models:
                    saved_fish_model = "s2.1-pro-free"
                fish_model = stable_selectbox(
                    tr("Fish Audio Model"),
                    options=_fish_audio_models,
                    default_value=saved_fish_model,
                    key="fish_audio_model_select",
                )
                _set_runtime_config("fish_audio", "model", fish_model)

            # ModelBest hosts VoxCPM behind its streaming Audio Speech API.
            # The fixed provider endpoint is still editable for compatible
            # gateways, while the user only has to supply an API key and a
            # speech_synthesis-capable model id for the standard platform.
            if tts_mode_enabled and (
                selected_tts_server == "voxcpm"
                or (voice_name and voice.is_voxcpm_voice(voice_name))
            ):
                _sync_voxcpm_api_key_input()
                voxcpm_api_key = st.text_input(
                    tr("VoxCPM API Key"),
                    type="password",
                    key="voxcpm_api_key_input",
                )
                _set_runtime_config("voxcpm", "api_key", voxcpm_api_key)

                voxcpm_model = st.text_input(
                    tr("VoxCPM Model ID"),
                    value=config.voxcpm.get("model_id", ""),
                    key="voxcpm_model_id_input",
                    placeholder=tr("VoxCPM Model ID Placeholder"),
                )
                _set_runtime_config("voxcpm", "model_id", voxcpm_model.strip())

                voxcpm_base_url = st.text_input(
                    tr("VoxCPM Base URL"),
                    value=config.voxcpm.get("base_url") or DEFAULT_VOXCPM_BASE_URL,
                    key="voxcpm_base_url_input",
                    placeholder=DEFAULT_VOXCPM_BASE_URL,
                )
                _set_runtime_config(
                    "voxcpm",
                    "base_url",
                    (voxcpm_base_url or DEFAULT_VOXCPM_BASE_URL).strip().rstrip("/"),
                )

                uploaded_reference_audio = st.file_uploader(
                    tr("VoxCPM Reference Audio"),
                    type=list(voice.VOXCPM_REFERENCE_AUDIO_FILE_TYPES),
                    key="voxcpm_reference_audio_uploader",
                    help=tr("VoxCPM Reference Audio Help"),
                )
                reference_audio = _sync_voxcpm_reference_audio(
                    uploaded_reference_audio
                )
                st.caption(tr("VoxCPM Reference Audio Notice"))
                reference_audio_error = st.session_state.get(
                    VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY
                )
                if reference_audio_error:
                    st.error(
                        tr("VoxCPM Reference Audio Invalid").format(
                            error=reference_audio_error
                        )
                    )

                if reference_audio:
                    high_fidelity_enabled = st.toggle(
                        tr("VoxCPM High Fidelity Delivery"),
                        key=VOXCPM_HIGH_FIDELITY_SESSION_KEY,
                        help=tr("VoxCPM High Fidelity Delivery Help"),
                    )
                    if high_fidelity_enabled:
                        use_separate_prompt_audio = st.toggle(
                            tr("VoxCPM Separate Prompt Audio"),
                            key=VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY,
                            help=tr("VoxCPM Separate Prompt Audio Help"),
                        )
                        _sync_voxcpm_prompt_example_mode(use_separate_prompt_audio)
                        if use_separate_prompt_audio:
                            prompt_audio_file = st.file_uploader(
                                tr("VoxCPM Prompt Audio"),
                                type=list(voice.VOXCPM_REFERENCE_AUDIO_FILE_TYPES),
                                key="voxcpm_prompt_audio_uploader",
                                help=tr("VoxCPM Prompt Audio Help"),
                            )
                            _sync_voxcpm_prompt_audio(prompt_audio_file)
                        else:
                            _clear_voxcpm_separate_prompt_audio()
                        effective_prompt_audio = _get_voxcpm_effective_prompt_audio()
                        if st.button(
                            tr("Transcribe VoxCPM Prompt Audio"),
                            key="transcribe_voxcpm_prompt_audio_button",
                            icon=":material/transcribe:",
                            help=tr("Transcribe VoxCPM Prompt Audio Help"),
                            use_container_width=True,
                            disabled=not bool(effective_prompt_audio),
                        ):
                            try:
                                with st.spinner(tr("Transcribing VoxCPM Prompt Audio")):
                                    recognized_text = subtitle.transcribe_audio_bytes(
                                        effective_prompt_audio
                                    )
                            except Exception:
                                logger.exception(
                                    "failed to transcribe VoxCPM prompt audio"
                                )
                                recognized_text = ""
                            if recognized_text:
                                st.session_state[VOXCPM_PROMPT_TEXT_SESSION_KEY] = (
                                    recognized_text
                                )
                                st.toast(tr("VoxCPM Prompt Audio Transcribed"))
                            else:
                                st.error(tr("VoxCPM Prompt Audio Transcription Failed"))
                        st.text_area(
                            tr("VoxCPM Prompt Text"),
                            key=VOXCPM_PROMPT_TEXT_SESSION_KEY,
                            help=tr("VoxCPM Prompt Text Help"),
                            height=100,
                        )
                        st.caption(tr("VoxCPM Prompt Transcript Review"))
                        prompt_error = _get_voxcpm_prompt_validation_error()
                        if prompt_error:
                            st.error(
                                tr("VoxCPM Prompt Invalid").format(
                                    error=prompt_error
                                )
                            )
                    else:
                        _clear_voxcpm_prompt_state()

            # Chatterbox API settings section (self-hosted, OpenAI-compatible)
            if tts_mode_enabled and (
                selected_tts_server == "chatterbox"
                or (voice_name and voice.is_chatterbox_voice(voice_name))
            ):
                chatterbox_base_url = st.text_input(
                    tr("Chatterbox Base URL"),
                    value=config.chatterbox.get("base_url")
                    or DEFAULT_CHATTERBOX_BASE_URL,
                    key="chatterbox_base_url_input",
                    placeholder=tr("Chatterbox Base URL Placeholder"),
                )
                _set_runtime_config(
                    "chatterbox", "base_url", (chatterbox_base_url or "").strip()
                )

                chatterbox_api_key = st.text_input(
                    tr("Chatterbox API Key"),
                    value=config.chatterbox.get("api_key", ""),
                    type="password",
                    key="chatterbox_api_key_input",
                )
                _set_runtime_config("chatterbox", "api_key", chatterbox_api_key)

                chatterbox_model = st.text_input(
                    tr("Chatterbox Model"),
                    value=config.chatterbox.get("model_id") or DEFAULT_CHATTERBOX_MODEL,
                    key="chatterbox_model_input",
                )
                _set_runtime_config(
                    "chatterbox",
                    "model_id",
                    (chatterbox_model or DEFAULT_CHATTERBOX_MODEL).strip(),
                )

                _saved_chatterbox_voices = (
                    _parse_chatterbox_voices(config.chatterbox.get("voices"))
                    or DEFAULT_CHATTERBOX_VOICES
                )
                if isinstance(_saved_chatterbox_voices, list):
                    _saved_chatterbox_voices = ", ".join(_saved_chatterbox_voices)
                chatterbox_voices = st.text_input(
                    tr("Chatterbox Voices"),
                    value=str(_saved_chatterbox_voices or ""),
                    key="chatterbox_voices_input",
                    placeholder=tr("Chatterbox Voices Placeholder"),
                )
                _set_runtime_config(
                    "chatterbox",
                    "voices",
                    _parse_chatterbox_voices(chatterbox_voices),
                )

            # Kokoro API settings section (self-hosted, OpenAI-compatible; voices listed from the server when left empty)
            if tts_mode_enabled and (
                selected_tts_server == "kokoro"
                or (voice_name and voice.is_kokoro_voice(voice_name))
            ):
                kokoro_base_url = st.text_input(
                    tr("Kokoro Base URL"),
                    value=config.kokoro.get("base_url")
                    or DEFAULT_KOKORO_BASE_URL,
                    key="kokoro_base_url_input",
                    placeholder=tr("Kokoro Base URL Placeholder"),
                )
                _set_runtime_config(
                    "kokoro", "base_url", (kokoro_base_url or "").strip()
                )

                kokoro_api_key = st.text_input(
                    tr("Kokoro API Key"),
                    value=config.kokoro.get("api_key", ""),
                    type="password",
                    key="kokoro_api_key_input",
                )
                _set_runtime_config("kokoro", "api_key", kokoro_api_key)

                kokoro_model = st.text_input(
                    tr("Kokoro Model"),
                    value=config.kokoro.get("model_id") or DEFAULT_KOKORO_MODEL,
                    key="kokoro_model_input",
                )
                _set_runtime_config(
                    "kokoro",
                    "model_id",
                    (kokoro_model or DEFAULT_KOKORO_MODEL).strip(),
                )

                _saved_kokoro_voices = (
                    _parse_chatterbox_voices(config.kokoro.get("voices"))
                    or DEFAULT_KOKORO_VOICES
                )
                if isinstance(_saved_kokoro_voices, list):
                    _saved_kokoro_voices = ", ".join(_saved_kokoro_voices)
                kokoro_voices = st.text_input(
                    tr("Kokoro Voices"),
                    value=str(_saved_kokoro_voices or ""),
                    key="kokoro_voices_input",
                    placeholder=tr("Kokoro Voices Placeholder"),
                )
                _set_runtime_config(
                    "kokoro",
                    "voices",
                    _parse_chatterbox_voices(kokoro_voices),
                )

            # Los tres modos solo renderizan los controles que la tarea actual necesita de verdad. La voz automática permite ajustar volumen y velocidad;
            # el audio subido solo necesita el archivo y el volumen; sin voz ya no se muestran ajustes inútiles.
            params.voice_name = (
                voice.NO_VOICE_NAME if voice_mode == VOICE_MODE_NONE else voice_name
            )
            params.voice_volume = 1.0
            params.voice_rate = 1.0
            uploaded_audio_file = None
            voice_volume_options = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0]
            voice_rate_options = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]

            if tts_mode_enabled:
                voice_control_cols = st.columns(2)
                with voice_control_cols[0]:
                    params.voice_volume = stable_selectbox(
                        tr("Voiceover Volume"),
                        options=voice_volume_options,
                        default_value=_saved_ui_choice(
                            "voice_volume", voice_volume_options, 1.0
                        ),
                        key="voice_volume_select",
                        format_func=lambda value: f"{int(value * 100)}%",
                        help=tr("Voiceover Volume Help"),
                    )

                with voice_control_cols[1]:
                    speed_not_supported_help = _voice_rate_not_supported_help(
                        selected_tts_server, voice_name
                    )
                    params.voice_rate = stable_selectbox(
                        tr("Voiceover Speed"),
                        options=voice_rate_options,
                        default_value=_saved_ui_choice(
                            "voice_rate", voice_rate_options, 1.0
                        ),
                        key="voice_rate_select",
                        format_func=lambda value: f"{value:.1f}×",
                        help=speed_not_supported_help or tr("Voiceover Speed Help"),
                        disabled=bool(speed_not_supported_help),
                    )
                _set_runtime_config("ui", "voice_volume", params.voice_volume)
                _set_runtime_config("ui", "voice_rate", params.voice_rate)

                # La vista previa debe ir después de los controles de volumen y velocidad, para que la llamada use los valores actuales de los controles.
                _render_voice_preview(
                    params,
                    friendly_names,
                    selected_tts_server,
                    voice_name,
                )
            elif voice_mode == VOICE_MODE_UPLOAD:
                custom_audio_file_types = sorted(
                    extension.removeprefix(".") for extension in CUSTOM_AUDIO_EXTENSIONS
                )
                uploaded_audio_file = st.file_uploader(
                    tr("Upload Voiceover File"),
                    type=custom_audio_file_types
                    + [file_type.upper() for file_type in custom_audio_file_types],
                    accept_multiple_files=False,
                    key="custom_audio_file_uploader",
                    help=tr("Upload Voiceover File Help"),
                )
                params.voice_volume = stable_selectbox(
                    tr("Voiceover Volume"),
                    options=voice_volume_options,
                    default_value=_saved_ui_choice(
                        "voice_volume", voice_volume_options, 1.0
                    ),
                    key="voice_volume_select",
                    format_func=lambda value: f"{int(value * 100)}%",
                    help=tr("Voiceover Volume Help"),
                )
                _set_runtime_config("ui", "voice_volume", params.voice_volume)
                if uploaded_audio_file:
                    st.audio(uploaded_audio_file, format="audio/mp3")
                    st.info(
                        tr(
                            "Custom audio will be used directly. TTS synthesis will be skipped for this task."
                        )
                    )
            uploaded_bgm_file = _render_background_music_settings(
                params,
                elevenlabs_api_key_rendered=elevenlabs_api_key_rendered,
            )
    return uploaded_audio_file, uploaded_bgm_file, voice_mode


def _render_subtitle_settings(panel, params):
    """Renderiza los ajustes de subtítulos y actualiza los parámetros de generación."""
    with panel:
        with st.container(border=True):
            st.write(tr("Subtitle Settings"))
            st.session_state.setdefault(
                "subtitle_enabled_checkbox",
                _saved_ui_bool(
                    "subtitle_enabled",
                    DEFAULT_SUBTITLE_SETTINGS["subtitle_enabled"],
                ),
            )
            params.subtitle_enabled = st.checkbox(
                tr("Enable Subtitles"),
                key="subtitle_enabled_checkbox",
            )
            _set_runtime_config("ui", "subtitle_enabled", params.subtitle_enabled)
            subtitle_settings_disabled = not params.subtitle_enabled
            font_names = get_all_fonts()
            saved_font_name = config.ui.get(
                "font_name", DEFAULT_SUBTITLE_SETTINGS["font_name"]
            )
            saved_font_name_index = 0
            if saved_font_name in font_names:
                saved_font_name_index = font_names.index(saved_font_name)
            params.font_name = stable_selectbox(
                tr("Font"),
                options=font_names,
                default_value=font_names[saved_font_name_index] if font_names else "",
                key="font_name_select",
                disabled=subtitle_settings_disabled,
            )
            _set_runtime_config("ui", "font_name", params.font_name)

            subtitle_positions = [
                (tr("Top"), "top"),
                (tr("Center"), "center"),
                (tr("Bottom"), "bottom"),
                (tr("2/3 from Bottom"), "two_thirds_bottom"),
                (tr("Custom"), "custom"),
            ]
            saved_subtitle_position = config.ui.get(
                "subtitle_position", DEFAULT_SUBTITLE_SETTINGS["subtitle_position"]
            )
            saved_position_index = 2
            for i, (_, pos_value) in enumerate(subtitle_positions):
                if pos_value == saved_subtitle_position:
                    saved_position_index = i
                    break
            selected_subtitle_position = stable_selectbox(
                tr("Position"),
                options=[value for _, value in subtitle_positions],
                default_value=subtitle_positions[saved_position_index][1],
                key="subtitle_position_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in subtitle_positions
                ).get(value, value),
                disabled=subtitle_settings_disabled,
            )
            params.subtitle_position = selected_subtitle_position
            _set_runtime_config("ui", "subtitle_position", params.subtitle_position)

            # Subtitle Display Mode (Sentence vs Single Word)
            subtitle_display_modes = [
                (tr("Sentence by Sentence"), "sentence"),
                (tr("Single Word (Word by Word)"), "word_by_word"),
            ]
            saved_display_mode = config.ui.get(
                "subtitle_display_mode",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_display_mode"],
            )
            saved_mode_idx = 0
            for i, (_, mode_val) in enumerate(subtitle_display_modes):
                if mode_val == saved_display_mode:
                    saved_mode_idx = i
                    break
            selected_display_mode = stable_selectbox(
                tr("Display Mode"),
                options=[val for _, val in subtitle_display_modes],
                default_value=subtitle_display_modes[saved_mode_idx][1],
                key="subtitle_display_mode_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in subtitle_display_modes
                ).get(value, value),
                help=tr("Word-by-word Timing Help"),
                disabled=subtitle_settings_disabled,
            )
            params.subtitle_display_mode = selected_display_mode
            _set_runtime_config(
                "ui", "subtitle_display_mode", params.subtitle_display_mode
            )

            # Subtitle Animation (None vs Pop Spring)
            subtitle_animations = [
                (tr("None (Animation)"), "none"),
                (tr("Pop Up (Spring)"), "pop_spring"),
            ]
            saved_anim = config.ui.get(
                "subtitle_animation",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_animation"],
            )
            saved_anim_idx = 0
            for i, (_, anim_val) in enumerate(subtitle_animations):
                if anim_val == saved_anim:
                    saved_anim_idx = i
                    break
            selected_anim = stable_selectbox(
                tr("Subtitle Animation"),
                options=[val for _, val in subtitle_animations],
                default_value=subtitle_animations[saved_anim_idx][1],
                key="subtitle_animation_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in subtitle_animations
                ).get(value, value),
                disabled=subtitle_settings_disabled,
            )
            params.subtitle_animation = selected_anim
            _set_runtime_config(
                "ui", "subtitle_animation", params.subtitle_animation
            )

            if params.subtitle_position == "custom":
                saved_custom_position = config.ui.get(
                    "custom_position", DEFAULT_SUBTITLE_SETTINGS["custom_position"]
                )
                st.session_state.setdefault(
                    "custom_position_input", str(saved_custom_position)
                )
                custom_position = st.text_input(
                    tr("Custom Position (% from top)"),
                    key="custom_position_input",
                    disabled=subtitle_settings_disabled,
                )
                try:
                    params.custom_position = float(custom_position)
                    if params.custom_position < 0 or params.custom_position > 100:
                        st.error(tr("Please enter a value between 0 and 100"))
                    else:
                        _set_runtime_config(
                            "ui", "custom_position", params.custom_position
                        )
                except ValueError:
                    st.error(tr("Please enter a valid number"))

            # Las etiquetas de color de los idiomas que no son chino suelen ser más largas que en chino. Se reserva un ancho adecuado para el selector de color,
            # evitando que la etiqueta salte de línea, y dejando a la vez suficiente espacio operable para el slider del tamaño de letra.
            font_cols = st.columns([0.42, 0.58])
            with font_cols[0]:
                saved_text_fore_color = config.ui.get(
                    "text_fore_color", DEFAULT_SUBTITLE_SETTINGS["text_fore_color"]
                )
                st.session_state.setdefault("font_color_picker", saved_text_fore_color)
                params.text_fore_color = st.color_picker(
                    tr("Font Color"),
                    key="font_color_picker",
                    disabled=subtitle_settings_disabled,
                )
                _set_runtime_config("ui", "text_fore_color", params.text_fore_color)

            with font_cols[1]:
                saved_font_size = config.ui.get(
                    "font_size", DEFAULT_SUBTITLE_SETTINGS["font_size"]
                )
                st.session_state.setdefault("font_size_slider", saved_font_size)
                params.font_size = st.slider(
                    tr("Font Size"),
                    30,
                    100,
                    key="font_size_slider",
                    disabled=subtitle_settings_disabled,
                )
                _set_runtime_config("ui", "font_size", params.font_size)

            stroke_cols = st.columns([0.42, 0.58])
            with stroke_cols[0]:
                st.session_state.setdefault(
                    "stroke_color_picker",
                    _saved_ui_color(
                        "stroke_color", DEFAULT_SUBTITLE_SETTINGS["stroke_color"]
                    ),
                )
                params.stroke_color = st.color_picker(
                    tr("Stroke Color"),
                    key="stroke_color_picker",
                    disabled=subtitle_settings_disabled,
                )
                _set_runtime_config("ui", "stroke_color", params.stroke_color)
            with stroke_cols[1]:
                st.session_state.setdefault(
                    "stroke_width_slider",
                    _saved_ui_number(
                        "stroke_width",
                        DEFAULT_SUBTITLE_SETTINGS["stroke_width"],
                        0.0,
                        10.0,
                    ),
                )
                params.stroke_width = st.slider(
                    tr("Stroke Width"),
                    0.0,
                    10.0,
                    key="stroke_width_slider",
                    disabled=subtitle_settings_disabled,
                )
                _set_runtime_config("ui", "stroke_width", params.stroke_width)

            # El nombre localizado del interruptor de fondo suele ser más largo que la etiqueta de color, así que el interruptor ocupa algo más de espacio.
            subtitle_bg_cols = st.columns([0.55, 0.45])
            saved_subtitle_background_enabled = config.ui.get(
                "subtitle_background_enabled",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_background_enabled"],
            )
            st.session_state.setdefault(
                "subtitle_background_enabled_checkbox",
                saved_subtitle_background_enabled,
            )
            with subtitle_bg_cols[0]:
                subtitle_background_enabled = st.checkbox(
                    tr("Enable Subtitle Background"),
                    key="subtitle_background_enabled_checkbox",
                    disabled=subtitle_settings_disabled,
                )
            _set_runtime_config(
                "ui",
                "subtitle_background_enabled",
                subtitle_background_enabled,
            )

            # El color de fondo y el estilo redondeado dependen del interruptor de fondo del subtítulo. Los subcontroles siempre permanecen en la página,
            # y se deshabilitan juntos cuando el interruptor padre está apagado, para evitar saltos de diseño porque un control desaparezca y otro se deshabilite.
            # El valor de color se sigue guardando en la configuración de UI, y al volver a activar el fondo se puede restaurar la elección anterior del usuario;
            # el parámetro que se pasa al servicio de generación se pone a False, para que el estado apagado no renderice el fondo.
            saved_subtitle_background_color = config.ui.get(
                "subtitle_background_color",
                DEFAULT_SUBTITLE_SETTINGS["subtitle_background_color"],
            )
            st.session_state.setdefault(
                "subtitle_background_color_picker",
                saved_subtitle_background_color,
            )
            with subtitle_bg_cols[1]:
                selected_subtitle_background_color = st.color_picker(
                    tr("Subtitle Background Color"),
                    key="subtitle_background_color_picker",
                    disabled=subtitle_settings_disabled
                    or not subtitle_background_enabled,
                )
            _set_runtime_config(
                "ui",
                "subtitle_background_color",
                selected_subtitle_background_color,
            )
            params.text_background_color = (
                selected_subtitle_background_color
                if subtitle_background_enabled
                else False
            )

            saved_rounded_subtitle_background = config.ui.get(
                "rounded_subtitle_background",
                DEFAULT_SUBTITLE_SETTINGS["rounded_subtitle_background"],
            )
            # Con el fondo apagado, el fondo redondeado no tiene color base que renderizar. Aquí se deshabilita el control pero se conserva la configuración original,
            # y cuando el usuario vuelva a activar el fondo del subtítulo podrá seguir usando la preferencia de redondeado guardada.
            rounded_background_disabled = (
                subtitle_settings_disabled or not subtitle_background_enabled
            )
            st.session_state.setdefault(
                "rounded_subtitle_background_checkbox",
                saved_rounded_subtitle_background,
            )
            selected_rounded_subtitle_background = st.checkbox(
                tr("Rounded Subtitle Background"),
                help=tr("Rounded Subtitle Background Help"),
                disabled=rounded_background_disabled,
                key="rounded_subtitle_background_checkbox",
            )
            params.rounded_subtitle_background = (
                selected_rounded_subtitle_background
                if subtitle_background_enabled
                else False
            )
            if not subtitle_settings_disabled and subtitle_background_enabled:
                _set_runtime_config(
                    "ui",
                    "rounded_subtitle_background",
                    selected_rounded_subtitle_background,
                )

            if video.subtitle_colors_are_indistinguishable(params):
                # Una configuración del mismo color sigue siendo una elección válida del usuario, así que solo se avisa junto a la zona de ajustes de subtítulos,
                # sin impedir la generación. El usuario puede decidir si continúa según sus necesidades visuales reales.
                st.warning(tr("Subtitle Colors Are Indistinguishable"))

            subtitle_preview_text = params.video_script or params.video_subject
            selected_font_path = os.path.join(font_dir, params.font_name)
            if (
                params.subtitle_enabled
                and subtitle_preview_text
                and not video.subtitle_font_supports_text(
                    selected_font_path, subtitle_preview_text
                )
            ):
                st.warning(tr("Subtitle Font Does Not Support Text"))

            if st.button(
                tr("Restore Default Subtitle Settings"),
                key="restore_default_subtitle_settings",
                icon=":material/restart_alt:",
                on_click=reset_subtitle_settings,
                use_container_width=True,
            ):
                st.toast(tr("Default Subtitle Settings Restored"))


def _stage_task_audio(audio_path, audio_bytes):
    """Publish task-local audio only after the complete write succeeds."""
    descriptor = None
    staged_path = None
    try:
        descriptor, staged_path = tempfile.mkstemp(
            dir=os.path.dirname(audio_path), prefix=".task-audio-", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as file:
            descriptor = None  # The file context now owns the descriptor.
            file.write(audio_bytes)
        os.replace(staged_path, audio_path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if staged_path is not None:
            try:
                os.unlink(staged_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(f"failed to remove staged task audio: {exc}")


def _render_generation_controls(
    params, uploaded_files, uploaded_audio_file, uploaded_bgm_file, voice_mode
):
    """
    Valida las dependencias de generación, envía la tarea y renderiza los logs y el video final.

    Devuelve si esta ejecución de la página envió con éxito una tarea nueva. Antes de enviar ya se pidió un guardado no bloqueante, y el llamador
    se salta por ello la petición repetida al final de la página. El script principal debe terminar a tiempo para que el Fragment periódico pueda seguir
    refrescando el progreso y los logs de la tarea.
    """
    restore_upload_requirements = st.session_state.get(
        "task_restore_upload_requirements", {}
    )
    has_local_materials = bool(
        uploaded_files or st.session_state.get("local_video_materials", [])
    )
    has_custom_audio = bool(uploaded_audio_file)
    unmet_restore_requirements = _get_unmet_restore_upload_requirements(
        restore_upload_requirements,
        video_source=params.video_source,
        voice_name=params.voice_name or "",
        has_local_materials=has_local_materials,
        has_custom_audio=has_custom_audio,
        voice_mode=voice_mode,
    )
    if "local_materials" in unmet_restore_requirements:
        st.warning(tr("Task Restore Local Materials Warning"))
    if "custom_audio" in unmet_restore_requirements:
        st.warning(tr("Task Restore Custom Audio Warning"))
    if restore_upload_requirements and not unmet_restore_requirements:
        # El usuario ya volvió a subir el archivo, o cambió por su cuenta el origen de material/la voz. En ese caso la dependencia de subida de la tarea histórica
        # ya se resolvió explícitamente; se borra la marca, para que las generaciones normales posteriores no sigan mostrando el aviso antiguo.
        st.session_state.pop("task_restore_upload_requirements", None)

    _render_settings_transfer(params)

    generation_busy = _generation_is_busy(_current_generation_task_id())
    start_button = st.button(
        tr("Generating Video") if generation_busy else tr("Generate Video"),
        use_container_width=True,
        type="primary",
        key="generate_video_button",
        on_click=_prepare_generation_task,
        disabled=generation_busy,
    )
    render_onboarding_tour()
    if start_button:
        _save_runtime_config()
        task_id = st.session_state.get("pending_generation_task_id") or str(uuid4())
        _add_active_generation_task(
            task_id,
            subject=params.video_subject or params.video_script or task_id,
        )
        if not params.video_subject and not params.video_script:
            _remove_active_generation_task(task_id)
            st.error(tr("Video Script and Subject Cannot Both Be Empty"))
            st.stop()

        voxcpm_reference_audio = None
        voxcpm_prompt_audio = None
        voxcpm_prompt_text = ""
        if voice.is_voxcpm_voice(params.voice_name or ""):
            reference_audio_error = st.session_state.get(
                VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY
            )
            if reference_audio_error:
                _remove_active_generation_task(task_id)
                st.error(
                    tr("VoxCPM Reference Audio Invalid").format(
                        error=reference_audio_error
                    )
                )
                st.stop()
            voxcpm_reference_audio = _get_voxcpm_reference_audio()
            prompt_validation_error = _get_voxcpm_prompt_validation_error()
            if prompt_validation_error:
                _remove_active_generation_task(task_id)
                st.error(
                    tr("VoxCPM Prompt Invalid").format(
                        error=prompt_validation_error
                    )
                )
                st.stop()
            voxcpm_prompt_audio = _get_voxcpm_effective_prompt_audio()
            voxcpm_prompt_text = _get_voxcpm_prompt_text()

        if params.video_source not in [
            "infistar",
            "jalapeno",
            "pexels",
            "pixabay",
            "coverr",
            "wavespeed",
            "volcengine_seedance",
            "ofox",
            "metaso_minimax",
            "muapi",
            "loomloom",
            "openai_image",
            "codex_image",
            "local",
        ]:
            _remove_active_generation_task(task_id)
            st.error(tr("Please Select a Valid Video Source"))
            st.stop()

        if params.video_source == "pexels" and not config.app.get(
            "pexels_api_keys", ""
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the Pexels API Key"))
            st.stop()

        if params.video_source == "pixabay" and not config.app.get(
            "pixabay_api_keys", ""
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the Pixabay API Key"))
            st.stop()

        if params.video_source == "coverr" and not config.app.get(
            "coverr_api_keys", ""
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the Coverr API Key"))
            st.stop()

        if params.video_source == "wavespeed" and not config.app.get(
            "wavespeed_api_keys", ""
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the WaveSpeed API Key"))
            st.stop()

        if params.video_source == "wavespeed" and not st.session_state.get(
            "wavespeed_confirm_charge", False
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Confirm WaveSpeed Charge Required"))
            st.stop()

        if params.video_source == "volcengine_seedance" and not (
            volcengine_seedance.is_enabled(
                config.snapshot_config_with_pending(config.app)
            )
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the Volcano Engine Ark API Key"))
            st.stop()

        if params.video_source == "volcengine_seedance" and not st.session_state.get(
            "volcengine_seedance_confirm_charge", False
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Confirm Volcano Engine Seedance Charge Required"))
            st.stop()

        if params.video_source == "jalapeno":
            try:
                snapshot = config.snapshot_config_with_pending(config.app)
                jalapeno_video_endpoint = jalapeno_video.endpoint(snapshot)
                if (not st.session_state.get("jalapeno_confirm_charge", False)
                        or _jalapeno_charge_signature(params, snapshot)
                        != st.session_state.get("jalapeno_charge_signature")):
                    raise jalapeno_video.JalapenoVideoError(tr("Confirm Jalapeno Cloud Charge Required"))
            except jalapeno_video.JalapenoVideoError as exc:
                _remove_active_generation_task(task_id)
                st.error(str(exc))
                st.stop()

        if params.video_source == "infistar":
            try:
                snapshot = config.snapshot_config_with_pending(config.app)
                if not infistar_video.is_enabled(snapshot):
                    raise infistar_video.InfistarVideoError(tr("Please Enter the Infistar API Key"))
                infistar_video.model_id(snapshot)
                if not st.session_state.get("infistar_confirm_charge", False):
                    raise infistar_video.InfistarVideoError(tr("Confirm Infistar Charge Required"))
            except infistar_video.InfistarVideoError as exc:
                _remove_active_generation_task(task_id)
                st.error(str(exc))
                st.stop()

        if params.video_source == "ofox" and not (
            ofox.is_enabled(config.snapshot_config_with_pending(config.app))
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the OFox API Key"))
            st.stop()

        if params.video_source == "ofox" and not st.session_state.get(
            "ofox_confirm_charge", False
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Confirm OFox Charge Required"))
            st.stop()

        if params.video_source == "metaso_minimax" and not (
            metaso_minimax.is_enabled(
                config.snapshot_config_with_pending(config.app)
            )
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the Metaso MiniMax API Key"))
            st.stop()

        if params.video_source == "metaso_minimax" and not st.session_state.get(
            "metaso_minimax_confirm_charge", False
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Confirm Metaso MiniMax Charge Required"))
            st.stop()

        if params.video_source == "muapi" and not (
            muapi.is_enabled(config.snapshot_config_with_pending(config.app))
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Enter the MuAPI API Key"))
            st.stop()

        if params.video_source == "muapi" and not st.session_state.get(
            "muapi_confirm_charge", False
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Confirm MuAPI Charge Required"))
            st.stop()

        if params.video_source == "codex_image":
            from app.services import codex_image, codex_oauth

            try:
                codex_image.capture_context(
                    config.snapshot_config_with_pending(config.app)
                )
            except codex_oauth.CodexError as error:
                _remove_active_generation_task(task_id)
                st.error(tr(error.key).format(**error.params))
                st.stop()

        if params.video_source == "openai_image" and not material.is_openai_image_enabled(
            config.snapshot_config_with_pending(config.app)
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Please Configure the OpenAI Image Source"))
            st.stop()

        loomloom_video_request = None
        if params.video_source == "loomloom":
            current_batch, current_signature = _current_loomloom_video_quote_context(
                params
            )
            quoted_batch = st.session_state.get("loomloom_video_batch")
            quote_result = st.session_state.get("loomloom_video_quote")
            quote_is_current = bool(
                current_batch is not None
                and isinstance(quoted_batch, loomloom.LoomLoomVideoBatch)
                and quote_result is not None
                and st.session_state.get("loomloom_video_input_signature")
                == current_signature
            )
            if not quote_is_current:
                _remove_active_generation_task(task_id)
                st.error(tr("AI Video Quote Required"))
                st.stop()
            if not st.session_state.get("loomloom_video_confirm_charge", False):
                _remove_active_generation_task(task_id)
                st.error(tr("Confirm AI Video Charge Required"))
                st.stop()
            try:
                video_backend = _create_loomloom_video_backend()
                loomloom_video_request = loomloom.LoomLoomConfirmedVideoRequest(
                    settings=video_backend.settings,
                    batch=current_batch,
                    listing_version_id=quote_result.listing_version_id,
                    client_request_id=st.session_state[
                        "loomloom_video_client_request_id"
                    ],
                )
                loomloom_video_request.validate()
            except (loomloom.LoomLoomError, ValueError) as exc:
                _remove_active_generation_task(task_id)
                st.error(str(exc))
                st.stop()

        if (
            params.bgm_type == "sonilo"
            and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
            and not sonilo_service.is_enabled()
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("Sonilo API Key Required"))
            st.stop()

        if (
            params.bgm_type == "elevenlabs"
            and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
            and not elevenlabs_music_service.is_enabled()
        ):
            _remove_active_generation_task(task_id)
            st.error(tr("ElevenLabs API Key Required"))
            st.stop()

        if params.video_source == "local" and not has_local_materials:
            # Si el material local está vacío y se sigue adelante, primero se generarían TTS/subtítulos y solo al final fallaría en la fase de preprocesado del material.
            # Bloquearlo antes de arrancar la tarea evita llamadas de API inútiles y archivos intermedios.
            _remove_active_generation_task(task_id)
            st.error(tr("Please Upload Local Materials First"))
            st.stop()

        if voice_mode == VOICE_MODE_UPLOAD and not uploaded_audio_file:
            # El audio subido es un modo de voz elegido explícitamente por el usuario; si falta el archivo no se puede volver en silencio al TTS.
            # Se bloquea antes de arrancar la tarea, para no producir un video final distinto de lo que eligió el usuario.
            _remove_active_generation_task(task_id)
            st.error(tr("Please Upload Voiceover File First"))
            st.stop()

        if "custom_audio" in unmet_restore_requirements:
            # El audio personalizado histórico no puede rellenarse automáticamente. Si el usuario aún no lo volvió a subir ni cambió de voz por su cuenta,
            # debe impedirse volver en silencio al TTS, o el resultado regenerado no coincidiría con la voz de la tarea original.
            _remove_active_generation_task(task_id)
            st.error(tr("Task Restore Custom Audio Warning"))
            st.stop()

        if uploaded_bgm_file and bgm_service.should_use_bgm(
            params.bgm_type, params.bgm_volume
        ):
            try:
                saved_bgm_name = bgm_service.save_bgm_upload(
                    uploaded_bgm_file.name, uploaded_bgm_file
                )
            except bgm_service.BgmUploadError as exc:
                _remove_active_generation_task(task_id)
                logger.warning(f"WebUI background music upload rejected: {str(exc)}")
                st.error(tr("Invalid Background Music"))
                st.stop()
            except bgm_service.BgmServiceError as exc:
                _remove_active_generation_task(task_id)
                logger.error(f"WebUI background music upload failed: {str(exc)}")
                st.error(tr("Background Music Validation Failed"))
                st.stop()
            # Tras guardar con éxito solo se escribe el nombre de archivo en los parámetros de la tarea. El servicio de video lo vuelve a resolver en los dos directorios de lista blanca
            # de BGM, para no persistir ni mostrar al usuario rutas absolutas del servidor.
            params.bgm_file = saved_bgm_name
        elif uploaded_bgm_file:
            # 0 con volumen , el servicio de video no usa ninguna BGM, así que ya no se persiste en storage el archivo subido
            # ya previsualizado. Si el usuario sube después el volumen, puede volver a pulsar generar para completar el guardado.
            params.bgm_file = ""

        if uploaded_audio_file:
            try:
                task_dir = utils.task_dir(task_id)
                custom_audio_path = _build_uploaded_file_path(
                    uploaded_audio_file,
                    task_dir,
                    CUSTOM_AUDIO_EXTENSIONS,
                    "custom-audio",
                )
                # Voiceover uploads previously bypassed the same full-decode
                # and size checks used for background-music uploads. Validate
                # before storing or scheduling the generation task.
                bgm_service.validate_bgm_upload(
                    uploaded_audio_file.name, uploaded_audio_file
                )
                _stage_task_audio(custom_audio_path, uploaded_audio_file.getbuffer())
            except OSError as exc:
                _remove_active_generation_task(task_id)
                logger.error(f"failed to persist uploaded task audio: {exc}")
                st.error(tr("Video Generation Failed"))
                st.stop()
            except bgm_service.BgmUploadError as exc:
                _remove_active_generation_task(task_id)
                logger.warning(f"WebUI custom audio upload rejected: {exc}")
                st.error(str(exc))
                st.stop()
            except bgm_service.BgmServiceError as exc:
                _remove_active_generation_task(task_id)
                logger.error(f"WebUI custom audio validation failed: {exc}")
                st.error(str(exc))
                st.stop()
            except ValueError:
                _remove_active_generation_task(task_id)
                st.error(tr("Unsupported Upload File Type"))
                st.stop()
            params.custom_audio_file = custom_audio_path

        if uploaded_files:
            # En cada nueva subida manda el material elegido esta vez, para que el material antiguo no se siga añadiendo repetidamente.
            try:
                params.video_materials, persisted_local_materials = (
                    _save_uploaded_local_materials(uploaded_files)
                )
            except material_upload_service.MaterialUploadError as exc:
                _remove_active_generation_task(task_id)
                logger.warning(f"WebUI local material upload rejected: {exc}")
                st.error(str(exc))
                st.stop()
            except material_upload_service.MaterialServiceError as exc:
                _remove_active_generation_task(task_id)
                logger.error(f"WebUI local material upload failed: {exc}")
                st.error(str(exc))
                st.stop()
            # Escribe en la sesión el material de video subido y guardado en local, para reutilizarlo directamente cuando luego solo se cambie el texto.
            st.session_state["local_video_materials"] = persisted_local_materials
        elif (
            params.video_source == "local" and st.session_state["local_video_materials"]
        ):
            # Si el usuario no vuelve a subir archivos, se reutiliza la última lista de material local guardada en disco.
            params.video_materials = []
            for material_entry in st.session_state["local_video_materials"]:
                m = MaterialInfo()
                m.provider = material_entry.get("provider", "local")
                m.url = material_entry.get("url", "")
                m.duration = material_entry.get("duration", 0)
                if m.url:
                    params.video_materials.append(m)

        reusable_voice_preview = _get_reusable_full_voice_preview(
            params,
            voice_mode,
        )
        if reusable_voice_preview:
            # La caché de vista previa solo existe en la sesión actual de Streamlit. Antes de enviar, el audio se escribe en el directorio de la tarea de destino,
            # y el hilo en segundo plano solo lee después los archivos de su propia tarea; aunque la página haga rerun, el navegador se cierre o
            # el usuario pruebe otras voces, la tarea de generación ya encolada no se ve afectada.
            try:
                preview_audio_file = os.path.join(
                    utils.task_dir(task_id),
                    "audio.mp3",
                )
                _stage_task_audio(preview_audio_file, reusable_voice_preview["audio_bytes"])
            except OSError as exc:
                _remove_active_generation_task(task_id)
                logger.error(f"failed to persist preview task audio: {exc}")
                st.error(tr("Video Generation Failed"))
                st.stop()
            reusable_voice_preview.pop("audio_bytes")
            reusable_voice_preview["audio_file"] = preview_audio_file
            logger.info(
                f"reuse full voice preview for task: "
                f"task_id={task_id}, duration={reusable_voice_preview['duration']:.2f}s"
            )

        try:
            st.toast(tr("Generating Video"))
            logger.info(tr("Start Generating Video"))
            logger.info(utils.to_json(params))
            submitted_task_id = webui_task.submit_generation(
                task_id=task_id,
                params=params,
                capture_logs=not config.ui.get("hide_log", False),
                voice_preview=reusable_voice_preview,
                loomloom_video_request=loomloom_video_request,
                voxcpm_reference_audio=voxcpm_reference_audio,
                voxcpm_prompt_audio=voxcpm_prompt_audio,
                voxcpm_prompt_text=voxcpm_prompt_text,
                **({"jalapeno_video_endpoint": jalapeno_video_endpoint}
                   if params.video_source == "jalapeno" else {}),
            )
            if loomloom_video_request is not None:
                # Cada cotización solo permite un envío. La petición en segundo plano lleva su propio ID idempotente estable; tras enviar con éxito
                # se borra la cotización de la página, y la siguiente generación debe volver a cotizar y confirmar.
                st.session_state["loomloom_video_batch"] = None
                st.session_state["loomloom_video_quote"] = None
                st.session_state["loomloom_video_input_signature"] = ""
                st.session_state["loomloom_video_client_request_id"] = ""
        except Exception:
            _remove_active_generation_task(task_id)
            st.error(tr("Video Generation Failed"))
            st.stop()

        if isinstance(submitted_task_id, str) and submitted_task_id != task_id:
            _remove_active_generation_task(task_id)
            task_id = submitted_task_id
            _add_active_generation_task(task_id, subject=params.video_subject)
        st.session_state["current_generation_task_id"] = task_id
        st.query_params["generation_task"] = task_id
        logger.info(f"WebUI generation task submitted: task_id={task_id}")
        if webui_task.is_generation_active(task_id):
            st.rerun()

    _render_current_generation_task()
    return start_button


def _render_application():
    """Renderiza en un orden fijo la barra superior, los diálogos, el formulario de generación y los resultados de la tarea."""
    _render_top_bar()

    if st.session_state.get("settings_dialog_open", False):
        _render_settings_dialog()

    if _apply_pending_settings_preset():
        st.success(tr("Settings Preset Imported"))

    restore_applied = _apply_pending_task_restore()
    restore_candidate_id = st.session_state.get("task_restore_candidate_id")
    if restore_candidate_id:
        _render_task_restore_dialog(restore_candidate_id)
    restore_succeeded = st.session_state.pop("task_restore_succeeded", False)
    if restore_applied or restore_succeeded:
        st.success(tr("Task Configuration Loaded"))

    with st.container(key="main_settings_grid"):
        panel = st.columns(4)
    left_panel = panel[0]
    middle_panel = panel[1]
    audio_panel = panel[2]
    right_panel = panel[3]

    params = VideoParams(video_subject="")
    params.match_materials_to_script = bool(
        st.session_state.get("match_materials_to_script", False)
    )
    _render_script_settings(left_panel, params)

    uploaded_files = _render_video_settings(middle_panel, params)
    uploaded_audio_file, uploaded_bgm_file, voice_mode = _render_audio_settings(
        audio_panel, params
    )

    _render_subtitle_settings(right_panel, params)

    generation_submitted = _render_generation_controls(
        params,
        uploaded_files,
        uploaded_audio_file,
        uploaded_bgm_file,
        voice_mode,
    )

    # La rama de generación ya pidió guardar antes de arrancar el hilo en segundo plano. Las interacciones normales con controles siguen pidiendo guardados no bloqueantes;
    # si la tarea en segundo plano está usando la configuración, la capa de configuración aplicará y guardará en disco automáticamente los últimos valores al terminar la tarea.
    if not generation_submitted:
        _save_runtime_config()


_render_application()
