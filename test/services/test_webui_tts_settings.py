import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from app.config import config
from app.services import voice


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"
I18N_DIR = ROOT_DIR / "webui" / "i18n"
LOCALES = ("de", "en", "es", "id", "pt", "ru", "tr", "vi")

# Cada proveedor mantiene un único acceso oficial. Chatterbox es un servicio autoalojado sin una Key
# plataforma de obtención unificada, así que enlaza a las instrucciones de configuración del servicio compatible que realmente se usa, para no inducir al usuario a registrarse en cuentas de terceros.
TTS_API_KEY_LABELS = {
    "Speech Key": "portal.azure.com",
    "SiliconFlow API Key": "cloud.siliconflow.cn/account/ak",
    "Gemini API Key": "aistudio.google.com/app/apikey",
    "MiMo API Key": "mimo.mi.com/docs/",
    "MiniMax TTS API Key": "platform.minimaxi.com",
    "ElevenLabs API Key": "elevenlabs.io/app/settings/api-keys",
    "Chatterbox API Key": "github.com/travisvn/chatterbox-tts-api",
    "VoxCPM API Key": "platform.modelbest.cn/console/keys",
}

TTS_PROVIDER_WIDGETS = {
    "azure-tts-v2": ("azure_speech_key_input", "Speech Key"),
    "siliconflow": ("siliconflow_api_key_input", "SiliconFlow API Key"),
    "gemini-tts": ("gemini_tts_api_key_input", "Gemini API Key"),
    "mimo-tts": ("mimo_tts_api_key_input", "MiMo API Key"),
    "minimax-tts": ("minimax_tts_api_key_input", "MiniMax TTS API Key"),
    "elevenlabs": ("elevenlabs_api_key_input", "ElevenLabs API Key"),
    "chatterbox": ("chatterbox_api_key_input", "Chatterbox API Key"),
    "voxcpm": ("voxcpm_api_key_input", "VoxCPM API Key"),
}


def _load_translation(locale: str) -> dict:
    """Lee directamente los archivos de idioma, asegurando que la aserción cubre la etiqueta Markdown final que realmente ve el usuario."""
    data = json.loads((I18N_DIR / f"{locale}.json").read_text(encoding="utf-8"))
    return data["Translation"]


def _widget_by_key(elements, key: str):
    """Streamlit Las etiquetas de los controles se traducen; se usa una key de negocio estable para localizar el campo de entrada real."""
    return next(
        item
        for item in elements
        if str(getattr(item, "key", "")) == key
        or str(getattr(item, "key", "")).startswith(f"{key}_")
    )


def test_all_tts_api_key_labels_include_an_official_configuration_link():
    """Todos los idiomas deben conservar el nombre del proveedor y un acceso clicable, para no perder enlaces al traducir."""
    for locale in LOCALES:
        translations = _load_translation(locale)
        for label_key, expected_host in TTS_API_KEY_LABELS.items():
            label = translations[label_key]
            assert expected_host in label, f"{locale}: {label_key}"
            assert "](" in label, f"{locale}: {label_key}"


def test_tts_provider_inputs_render_the_standardized_labels():
    """Cambia de verdad a cada TTS Provider, confirmando que los campos de entrada no se saltan las etiquetas traducidas unificadas."""
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="",
    )
    translations = _load_translation("es")

    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
        patch.object(voice, "get_all_azure_voices", return_value=[]),
        patch.object(voice, "get_siliconflow_voices", return_value=[]),
        patch.object(voice, "get_gemini_voices", return_value=[]),
        patch.object(voice, "get_mimo_voices", return_value=[]),
        patch.object(voice, "get_elevenlabs_voices", return_value=[]),
        patch.object(voice, "get_chatterbox_voices", return_value=[]),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "es"
        app.run()

        for provider, (widget_key, label_key) in TTS_PROVIDER_WIDGETS.items():
            provider_select = _widget_by_key(app.selectbox, "tts_server_select")
            provider_select.set_value(provider).run()

            api_key_input = _widget_by_key(app.text_input, widget_key)
            assert api_key_input.label == translations[label_key]
            assert api_key_input.proto.type == api_key_input.proto.PASSWORD
            assert not getattr(api_key_input.proto, "help", "")

    assert [str(item.value) for item in app.exception] == []


def test_voxcpm_settings_render_model_and_endpoint_fields():
    test_config = dict(
        config.voxcpm,
        api_key="",
        model_id="speech-model",
        base_url=voice.VOXCPM_DEFAULT_BASE_URL,
        voice_id=voice.VOXCPM_DEFAULT_VOICE,
    )
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="voxcpm",
        voice_name="voxcpm:default",
    )

    with (
        patch.object(config, "voxcpm", test_config),
        patch.object(config, "ui", test_ui),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "en"
        app.run()

    api_key_input = _widget_by_key(app.text_input, "voxcpm_api_key_input")
    assert api_key_input.proto.type == api_key_input.proto.PASSWORD
    assert _widget_by_key(app.text_input, "voxcpm_model_id_input").value == "speech-model"
    assert (
        _widget_by_key(app.text_input, "voxcpm_base_url_input").value
        == voice.VOXCPM_DEFAULT_BASE_URL
    )
    speed_input = _widget_by_key(app.selectbox, "voice_rate_select")
    assert speed_input.disabled
    assert "does not support numeric speed" in speed_input.help
    assert [str(item.value) for item in app.exception] == []


@pytest.mark.parametrize(
    ("tts_server", "voice_name", "disabled", "speed_help"),
    [
        (
            "gemini-tts",
            "gemini:Zephyr-Bright",
            True,
            "Gemini TTS does not support numeric speed adjustment.",
        ),
        (
            "mimo-tts",
            "mimo:mimo_default-Female",
            True,
            "MiMo TTS does not support numeric speed adjustment.",
        ),
        (
            "azure-tts-v1",
            "en-US-AriaNeural-Female",
            False,
            "Applies only to automatic voiceover. 1.0× is the normal speed.",
        ),
    ],
)
def test_speed_selector_is_disabled_for_providers_without_a_speed_setting(
    tts_server, voice_name, disabled, speed_help
):
    """Gemini and MiMo, like VoxCPM, have no speed setting; the saved speed is kept."""
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server=tts_server,
        voice_name=voice_name,
        voice_rate=1.5,
    )

    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "en"
        app.run()

    speed_input = _widget_by_key(app.selectbox, "voice_rate_select")
    assert speed_input.disabled == disabled
    assert speed_input.help == speed_help
    assert speed_input.value == 1.5
    assert test_ui["voice_rate"] == 1.5
    assert [str(item.value) for item in app.exception] == []


def test_voxcpm_settings_render_reference_audio_with_cloud_notice():
    test_config = dict(
        config.voxcpm,
        api_key="",
        model_id="speech-model",
        base_url=voice.VOXCPM_DEFAULT_BASE_URL,
        voice_id=voice.VOXCPM_DEFAULT_VOICE,
    )
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="voxcpm",
        voice_name="voxcpm:default",
    )

    with (
        patch.object(config, "voxcpm", test_config),
        patch.object(config, "ui", test_ui),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "en"
        app.run()

    uploader = _widget_by_key(app.file_uploader, "voxcpm_reference_audio_uploader")
    assert uploader.label == "VoxCPM Reference Audio (Optional)"
    assert "short reference clip" in uploader.help
    assert any("sent to ModelBest" in str(item.value) for item in app.caption)
    assert "voxcpm_reference_audio" not in test_config
    assert [str(item.value) for item in app.exception] == []


def test_voxcpm_reconnect_restores_saved_key_instead_of_clearing_it():
    """Streamlit Al reproducir un estado de contraseña vacía tras reconectar, la VoxCPM Key debe conservarse."""
    test_config = dict(config.voxcpm, api_key="saved-voxcpm-key")
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="voxcpm",
        voice_name="voxcpm:default",
    )

    with (
        patch.object(config, "voxcpm", test_config),
        patch.object(config, "ui", test_ui),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "en"
        app.session_state["voxcpm_api_key_input"] = ""
        app.run()

    assert test_config["api_key"] == "saved-voxcpm-key"
    assert app.session_state["voxcpm_api_key_input"] == "saved-voxcpm-key"
    assert [str(item.value) for item in app.exception] == []


def test_elevenlabs_reconnect_restores_saved_key_before_loading_voices():
    """
    Tras reiniciar el servicio, el navegador puede reproducir un estado de contraseña vacía; la WebUI debe conservar la configuración y, en el mismo rerun,
    cargar las voces con la Key guardada, en lugar de solo evitar escribir el vacío y seguir pidiendo al servicio con una Key vacía.
    """
    test_config = dict(config.elevenlabs, api_key="saved-key")
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="elevenlabs",
        voice_name="",
    )

    with (
        patch.object(config, "elevenlabs", test_config),
        patch.object(config, "ui", test_ui),
        patch.object(config, "try_save_config", return_value=True),
        patch.object(voice, "get_elevenlabs_voices", return_value=[]) as get_voices,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "en"
        app.session_state["elevenlabs_api_key_input"] = ""
        app.run()

    assert test_config["api_key"] == "saved-key"
    assert app.session_state["elevenlabs_api_key_input"] == "saved-key"
    assert get_voices.call_count >= 1
    assert all(call.args == ("saved-key",) for call in get_voices.call_args_list)
    assert [str(item.value) for item in app.exception] == []


def test_elevenlabs_environment_key_is_used_without_persisting_it():
    """Las variables de entorno pueden alimentar la carga de voces, pero la WebUI no puede copiarlas automáticamente a config.toml."""
    test_config = dict(config.elevenlabs, api_key="")
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="elevenlabs",
        voice_name="",
    )

    with (
        patch.object(config, "elevenlabs", test_config),
        patch.object(config, "ui", test_ui),
        patch.object(config, "try_save_config", return_value=True),
        patch.dict(os.environ, {"ELEVENLABS_API_KEY": "env-key"}),
        patch.object(voice, "get_elevenlabs_voices", return_value=[]) as get_voices,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "en"
        app.run()

    assert test_config["api_key"] == ""
    assert app.session_state["elevenlabs_api_key_input"] == "env-key"
    assert get_voices.call_count >= 1
    assert all(call.args == ("env-key",) for call in get_voices.call_args_list)
    assert [str(item.value) for item in app.exception] == []


def test_minimax_reconnect_restores_saved_tts_key():
    """El estado vacío tras reconectar el navegador no puede borrar la MiniMax TTS Key ya guardada."""
    test_config = dict(config.minimax_tts, api_key="saved-tts-key", base_url=voice.MINIMAX_TTS_GLOBAL_URL)
    test_ui = dict(config.ui, voice_mode="tts", tts_server="minimax-tts", voice_name="")

    with (
        patch.object(config, "minimax_tts", test_config),
        patch.object(config, "ui", test_ui),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "en"
        app.session_state["minimax_tts_api_key_input"] = ""
        app.run()

    assert test_config["api_key"] == "saved-tts-key"
    assert app.session_state["minimax_tts_api_key_input"] == "saved-tts-key"
    assert [str(item.value) for item in app.exception] == []


def test_minimax_shared_llm_key_is_not_duplicated_in_tts_config():
    """La Key de LLM compartida debe ajustarse automáticamente a la región, pero no puede copiarse a la configuración exclusiva de TTS."""
    test_config = dict(config.minimax_tts, api_key="", base_url="")
    test_app_config = dict(
        config.app,
        minimax_api_key="shared-cn-key",
        minimax_base_url="https://api.minimaxi.com/v1",
    )
    test_ui = dict(config.ui, voice_mode="tts", tts_server="minimax-tts", voice_name="")

    with (
        patch.object(config, "minimax_tts", test_config),
        patch.object(config, "app", test_app_config),
        patch.object(config, "ui", test_ui),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "en"
        app.run()

    api_key_input = _widget_by_key(app.text_input, "minimax_tts_api_key_input")
    endpoint_select = _widget_by_key(app.selectbox, "minimax_tts_endpoint_select")
    assert api_key_input.value == "shared-cn-key"
    assert test_config["api_key"] == ""
    assert endpoint_select.value == voice.MINIMAX_TTS_CN_URL
    assert endpoint_select.disabled
    assert [str(item.value) for item in app.exception] == []


def test_minimax_voice_selector_accepts_a_custom_voice_id():
    """MiniMax El selector de voces genérico debe permitir introducir Voice IDs que no están en la lista."""
    test_config = dict(
        config.minimax_tts,
        api_key="test-key",
        base_url=voice.MINIMAX_TTS_GLOBAL_URL,
        voice_id="old-voice",
    )
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="minimax-tts",
        voice_name="minimax:old-voice",
    )

    with (
        patch.object(config, "minimax_tts", test_config),
        patch.object(config, "ui", test_ui),
        patch.object(config, "try_save_config", return_value=True),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "en"
        app.run()
        voice_select = _widget_by_key(
            app.selectbox,
            "speech_synthesis_select_minimax-tts",
        )

    assert voice_select.proto.accept_new_options
    assert voice_select.value == "minimax:old-voice"
    assert [str(item.value) for item in app.exception] == []


def test_minimax_voices_load_only_on_demand_and_sync_the_selected_voice():
    """La lista de voces solo se carga cuando el usuario hace clic, y la selección debe sincronizarse con la configuración y con el control de voz genérico."""
    test_config = dict(
        config.minimax_tts,
        api_key="test-key",
        base_url=voice.MINIMAX_TTS_CN_URL,
        voice_id="old-voice",
    )
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="minimax-tts",
        voice_name="minimax:old-voice",
    )
    catalog = [
        {
            "voice_id": "Chinese (Mandarin)_News_Anchor",
            "voice_name": "voz femenina de noticias",
            "voice_type": "system",
        },
        {
            "voice_id": "English_expressive_narrator",
            "voice_name": "Expressive Narrator",
            "voice_type": "system",
        },
    ]

    with (
        patch.object(config, "minimax_tts", test_config),
        patch.object(config, "ui", test_ui),
        patch.object(config, "try_save_config", return_value=True),
        patch.object(
            voice,
            "get_minimax_voice_catalog",
            return_value=catalog,
        ) as get_catalog,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "es"
        app.run()

        # Un rerun normal de la página no puede consumir por su cuenta la API de MiniMax; solo se consulta al hacer clic en el botón.
        get_catalog.assert_not_called()
        _widget_by_key(app.button, "load_minimax_voices_button").click().run()
        get_catalog.assert_called_once_with(
            api_key="test-key",
            endpoint=voice.MINIMAX_TTS_CN_URL,
            voice_type="all",
        )

        voice_select = _widget_by_key(
            app.selectbox,
            "speech_synthesis_select_minimax-tts",
        )
        voice_select.set_value("minimax:Chinese (Mandarin)_News_Anchor").run()

        assert test_config["voice_id"] == "Chinese (Mandarin)_News_Anchor"
        assert voice_select.value == "minimax:Chinese (Mandarin)_News_Anchor"

    voice_select = _widget_by_key(app.selectbox, "speech_synthesis_select_minimax-tts")
    assert voice_select.proto.accept_new_options
    assert test_config["voice_id"] == "Chinese (Mandarin)_News_Anchor"
    assert test_ui["voice_name"] == "minimax:Chinese (Mandarin)_News_Anchor"
    assert voice_select.value == "minimax:Chinese (Mandarin)_News_Anchor"
    assert get_catalog.call_count == 1
    assert not any(item.label == "MiniMax TTS Voice ID" for item in app.text_input)
    assert not any(item.label == "MiniMax Voice Catalog" for item in app.selectbox)
    assert [str(item.value) for item in app.exception] == []
