import ast
import hashlib
import re
import shutil
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.config import config
from app.models.schema import VideoParams
from app.services import task as tm
from app.services import voice
from app.services import webui_task
from app.utils import utils


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


def _load_duration_estimator():
    """Solo carga la función pura de estimación, para que la prueba unitaria no importe ni ejecute toda la página de Streamlit."""
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_estimate_voiceover_duration_range"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"re": re}
    exec(compile(module, str(WEBUI_MAIN), "exec"), namespace)
    return namespace["_estimate_voiceover_duration_range"]


def _load_provider_signature(test_config, session_state=None):
    """Carga las funciones de resumen de credenciales y de huella del Provider, verificando por separado las reglas de invalidación de caché."""
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_credential_signature",
            "_get_voxcpm_reference_audio_digest",
            "_get_voxcpm_prompt_audio_digest",
            "_get_voxcpm_prompt_text",
            "_get_voxcpm_effective_prompt_audio_digest",
            "_get_voice_preview_provider_signature",
        }
    ]
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {
        "hashlib": hashlib,
        "config": test_config,
        "st": SimpleNamespace(
            session_state={} if session_state is None else session_state
        ),
        "VOXCPM_REFERENCE_AUDIO_SESSION_KEY": "voxcpm_reference_audio",
        "VOXCPM_PROMPT_AUDIO_SESSION_KEY": "voxcpm_prompt_audio",
        "VOXCPM_PROMPT_TEXT_SESSION_KEY": "voxcpm_prompt_text_input",
        "VOXCPM_HIGH_FIDELITY_SESSION_KEY": "voxcpm_high_fidelity_enabled",
        "VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY": (
            "voxcpm_separate_prompt_audio_enabled"
        ),
    }
    exec(compile(module, str(WEBUI_MAIN), "exec"), namespace)
    return namespace["_get_voice_preview_provider_signature"]


def _load_voxcpm_state_helpers(session_state):
    """Load upload-state helpers without executing the full Streamlit app."""
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    function_names = {
        "_get_voxcpm_reference_audio",
        "_get_voxcpm_reference_audio_digest",
        "_get_voxcpm_prompt_audio",
        "_get_voxcpm_prompt_audio_digest",
        "_get_voxcpm_prompt_text",
        "_clear_voxcpm_separate_prompt_audio",
        "_clear_voxcpm_prompt_state",
        "_clear_voxcpm_prompt_transcript",
        "_sync_voxcpm_prompt_example_mode",
        "_get_voxcpm_effective_prompt_audio",
        "_get_voxcpm_effective_prompt_audio_digest",
        "_get_voxcpm_prompt_validation_error",
        "_get_voxcpm_preview_validation_error",
        "_sync_voxcpm_reference_audio",
        "_sync_voxcpm_prompt_audio",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {
        "Path": Path,
        "hashlib": hashlib,
        "st": SimpleNamespace(
            session_state=session_state,
            spinner=lambda _label: nullcontext(),
        ),
        "tr": lambda key: key,
        "voice": SimpleNamespace(
            VOXCPM_REFERENCE_AUDIO_MAX_UPLOAD_BYTES=1024,
            prepare_voxcpm_reference_audio=lambda audio, _suffix: audio,
            is_voxcpm_voice=lambda name: str(name or "").startswith("voxcpm:"),
        ),
        "VOXCPM_REFERENCE_AUDIO_SESSION_KEY": "voxcpm_reference_audio",
        "VOXCPM_REFERENCE_AUDIO_ERROR_SESSION_KEY": "voxcpm_reference_audio_error",
        "VOXCPM_PROMPT_AUDIO_SESSION_KEY": "voxcpm_prompt_audio",
        "VOXCPM_PROMPT_AUDIO_ERROR_SESSION_KEY": "voxcpm_prompt_audio_error",
        "VOXCPM_PROMPT_TEXT_SESSION_KEY": "voxcpm_prompt_text_input",
        "VOXCPM_HIGH_FIDELITY_SESSION_KEY": "voxcpm_high_fidelity_enabled",
        "VOXCPM_SEPARATE_PROMPT_AUDIO_SESSION_KEY": (
            "voxcpm_separate_prompt_audio_enabled"
        ),
        "VOXCPM_PROMPT_EXAMPLE_MODE_SESSION_KEY": "voxcpm_prompt_example_mode",
    }
    exec(compile(module, str(WEBUI_MAIN), "exec"), namespace)
    return namespace


class _FakeAudioUpload:
    def __init__(self, name: str, payload: bytes):
        self.name = name
        self.size = len(payload)
        self._payload = payload

    def getvalue(self):
        return self._payload


def _session_audio(payload: bytes) -> dict:
    return {
        "audio_bytes": payload,
        "audio_digest": hashlib.sha256(payload).hexdigest(),
        "upload_digest": hashlib.sha256(payload).hexdigest(),
    }


def _button_by_key(app, key):
    return next(
        button
        for button in app.button
        if str(getattr(button, "key", "")).startswith(key)
    )


def test_duration_estimator_is_local_and_respects_voice_rate():
    """La estimación local debe cubrir chino e inglés, y acortarse razonablemente según la velocidad que elija el usuario."""
    estimate = _load_duration_estimator()
    script = "人工智能正在改变日常生活。它可以帮助我们整理信息，也能提高效率。"

    normal_range = estimate(script, 1.0)
    fast_range = estimate(script, 2.0)

    assert normal_range is not None
    assert fast_range is not None
    assert normal_range[0] < normal_range[1]
    assert fast_range[0] < normal_range[0]
    assert estimate("", 1.0) is None
    assert estimate("AI tools can simplify repetitive work.", 1.0) is not None


def test_provider_signature_changes_when_api_key_changes():
    """Cambiar solo la API Key también debe invalidar la caché de vista previa, sin fingir que las nuevas credenciales se validaron con éxito."""
    test_config = SimpleNamespace(
        app={"gemini_api_key": "old-gemini", "mimo_api_key": "old-mimo"},
        azure={"speech_region": "eastasia", "speech_key": "old-azure"},
        siliconflow={"api_key": "old-siliconflow"},
        elevenlabs={"api_key": "old-elevenlabs", "model_id": "eleven_v3"},
        chatterbox={
            "api_key": "old-chatterbox",
            "base_url": "http://127.0.0.1:4123/v1",
            "model_id": "chatterbox",
        },
        voxcpm={
            "api_key": "old-voxcpm",
            "base_url": "https://api.modelbest.cn/v1",
            "model_id": "speech-model",
            "voice_id": "default",
        },
    )
    provider_signature = _load_provider_signature(test_config)

    old_signature = provider_signature("elevenlabs")
    test_config.elevenlabs["api_key"] = "new-elevenlabs"
    new_signature = provider_signature("elevenlabs")

    assert old_signature != new_signature
    assert "old-elevenlabs" not in str(old_signature)
    assert "new-elevenlabs" not in str(new_signature)


def test_voxcpm_preview_signature_tracks_endpoint_model_and_credentials():
    session_state = {}
    test_config = SimpleNamespace(
        app={}, azure={}, siliconflow={}, elevenlabs={}, chatterbox={}, kokoro={},
        voxcpm={
            "api_key": "old-key",
            "base_url": "https://api.modelbest.cn/v1",
            "model_id": "speech-model-a",
            "voice_id": "default",
        },
    )
    provider_signature = _load_provider_signature(test_config, session_state)

    original = provider_signature("voxcpm")
    test_config.voxcpm["model_id"] = "speech-model-b"
    changed_model = provider_signature("voxcpm")
    test_config.voxcpm["api_key"] = "new-key"
    changed_key = provider_signature("voxcpm")
    session_state["voxcpm_reference_audio"] = {"audio_digest": "new-reference"}
    changed_reference = provider_signature("voxcpm")
    session_state["voxcpm_high_fidelity_enabled"] = True
    session_state["voxcpm_prompt_text_input"] = "delivery transcript"
    reused_reference_as_prompt = provider_signature("voxcpm")
    session_state["voxcpm_separate_prompt_audio_enabled"] = True
    session_state["voxcpm_prompt_audio"] = {"audio_digest": "new-prompt"}
    changed_prompt = provider_signature("voxcpm")

    assert original != changed_model != changed_key != changed_reference
    assert changed_reference != reused_reference_as_prompt != changed_prompt
    assert "old-key" not in str(original)
    assert "new-key" not in str(changed_key)


def test_replacing_shared_reference_clears_its_transcript():
    session_state = {
        "voxcpm_reference_audio": _session_audio(b"clip-a"),
        "voxcpm_high_fidelity_enabled": True,
        "voxcpm_separate_prompt_audio_enabled": False,
        "voxcpm_prompt_text_input": "clip A transcript",
    }
    helpers = _load_voxcpm_state_helpers(session_state)

    helpers["_sync_voxcpm_reference_audio"](
        _FakeAudioUpload("clip-b.wav", b"clip-b")
    )

    assert "voxcpm_prompt_text_input" not in session_state


def test_replacing_separate_delivery_audio_clears_transcript_but_not_identity_change():
    session_state = {
        "voxcpm_reference_audio": _session_audio(b"identity-a"),
        "voxcpm_prompt_audio": _session_audio(b"delivery-a"),
        "voxcpm_high_fidelity_enabled": True,
        "voxcpm_separate_prompt_audio_enabled": True,
        "voxcpm_prompt_text_input": "delivery A transcript",
    }
    helpers = _load_voxcpm_state_helpers(session_state)

    helpers["_sync_voxcpm_reference_audio"](
        _FakeAudioUpload("identity-b.wav", b"identity-b")
    )
    assert session_state["voxcpm_prompt_text_input"] == "delivery A transcript"

    helpers["_sync_voxcpm_prompt_audio"](
        _FakeAudioUpload("delivery-b.wav", b"delivery-b")
    )
    assert "voxcpm_prompt_text_input" not in session_state


def test_switching_prompt_example_modes_clears_transcript():
    session_state = {
        "voxcpm_prompt_example_mode": False,
        "voxcpm_prompt_text_input": "shared transcript",
    }
    helpers = _load_voxcpm_state_helpers(session_state)

    helpers["_sync_voxcpm_prompt_example_mode"](True)

    assert "voxcpm_prompt_text_input" not in session_state


def test_invalid_reference_upload_blocks_voxcpm_preview_only():
    session_state = {"voxcpm_reference_audio_error": "invalid WAV"}
    helpers = _load_voxcpm_state_helpers(session_state)

    assert (
        helpers["_get_voxcpm_preview_validation_error"](
            "voxcpm", "voxcpm:default"
        )
        == "invalid WAV"
    )
    assert (
        helpers["_get_voxcpm_preview_validation_error"](
            "azure-tts-v1", "zh-CN-XiaoxiaoNeural-Female"
        )
        == ""
    )


def test_full_voiceover_preview_is_disabled_until_script_exists():
    """La vista previa completa debe lanzarla el usuario; con el texto vacío no se puede llamar por error a un TTS comercial."""
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
    )
    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "es"
        app.run()

    full_preview = _button_by_key(
        app,
        "generate_full_voiceover_preview_button",
    )
    assert full_preview.disabled
    assert any("Añade un guion para estimar" in item.value for item in app.caption)


def test_script_shows_estimate_and_enables_full_voiceover_preview():
    """Tras escribir el texto se muestra la estimación gratuita, indicando claramente que la vista previa completa puede tener costes de API."""
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
    )
    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "es"
        app.session_state["video_script"] = (
            "人工智能正在改变日常生活。合理使用工具，可以帮助我们提高工作效率。"
        )
        app.run()

    full_preview = _button_by_key(
        app,
        "generate_full_voiceover_preview_button",
    )
    assert not full_preview.disabled
    assert any("cálculo local, sin llamar a la API" in item.value for item in app.caption)
    assert "puede consumir créditos de API" in full_preview.help
    assert [str(item.value) for item in app.exception] == []


def test_short_preview_autoplays_only_after_explicit_click_and_reuses_cache():
    """La vista previa corta debe reproducirse de inmediato; un rerun normal no la repite, y los clics repetidos no vuelven a llamar al TTS."""
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
    )

    def fake_tts(**kwargs):
        Path(kwargs["voice_file"]).write_bytes(
            b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32
        )
        return object()

    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
        patch.object(voice, "tts", side_effect=fake_tts) as synthesize,
        patch.object(voice, "get_audio_duration", return_value=3.0),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "es"
        app.run()

        _button_by_key(app, "play_voice_button").click().run()
        assert len(app.get("audio")) == 1
        assert app.get("audio")[0].proto.autoplay

        app.run()
        assert len(app.get("audio")) == 1
        assert not app.get("audio")[0].proto.autoplay

        _button_by_key(app, "play_voice_button").click().run()

    synthesize.assert_called_once()
    assert app.get("audio")[0].proto.autoplay
    assert [str(item.value) for item in app.exception] == []


def test_full_preview_uses_script_and_reuses_identical_cached_audio():
    """La vista previa completa usa el texto actual, y los clics repetidos con los mismos parámetros no pueden volver a llamar al TTS."""
    script = "Este es un texto de prueba para verificar la caché de la vista previa completa de la voz."
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
    )

    def fake_tts(**kwargs):
        # Aunque la extensión del archivo sea mp3, el TTS real puede devolver WAV; esta cabecera mínima también
        # verifica que la WebUI reconoce el MIME del reproductor por el contenido, sin fiarse ciegamente de la extensión.
        Path(kwargs["voice_file"]).write_bytes(
            b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32
        )
        return object()

    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
        patch.object(voice, "tts", side_effect=fake_tts) as synthesize,
        patch.object(voice, "get_audio_duration", return_value=12.3),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "es"
        app.session_state["video_script"] = script
        app.run()

        _button_by_key(
            app,
            "generate_full_voiceover_preview_button",
        ).click().run()
        _button_by_key(
            app,
            "generate_full_voiceover_preview_button",
        ).click().run()

    synthesize.assert_called_once()
    assert synthesize.call_args.kwargs["text"] == script
    assert len(app.get("audio")) == 1
    assert not app.get("audio")[0].proto.autoplay
    assert any("Duración real: 12.3 s" in item.value for item in app.caption)
    assert [str(item.value) for item in app.exception] == []


def test_full_preview_reports_when_tts_returns_no_audio():
    """TTS Si devuelve un resultado vacío debe dar un aviso accionable, sin dejar el clic del botón sin ninguna respuesta."""
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
    )
    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
        patch.object(voice, "tts", return_value=None),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "es"
        app.session_state["video_script"] = "Verificar la respuesta vacía del servicio de voz."
        app.run()
        _button_by_key(
            app,
            "generate_full_voiceover_preview_button",
        ).click().run()

    assert [item.value for item in app.error] == [
        "El servicio TTS no devolvió audio. Revisa la configuración y los registros."
    ]
    assert [str(item.value) for item in app.exception] == []


def test_full_preview_returns_immediately_when_runtime_config_is_busy():
    """Cuando la tarea en segundo plano tiene el bloqueo de configuración, la vista previa debe pedir reintentar más tarde en lugar de bloquear la página."""
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
    )
    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
        patch.object(
            config,
            "try_runtime_config_lock",
            return_value=nullcontext(False),
        ),
        patch.object(voice, "tts") as synthesize,
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "es"
        app.session_state["video_script"] = "Verificar que el estado ocupado no bloquea la página."
        app.run()
        _button_by_key(
            app,
            "generate_full_voiceover_preview_button",
        ).click().run()

    synthesize.assert_not_called()
    warning_messages = [item.value for item in app.warning]
    assert "Una tarea de vídeo está usando la configuración de voz. Inténtalo de nuevo en breve." in warning_messages


def test_full_preview_warns_when_audio_duration_is_unavailable():
    """Si el audio se puede reproducir pero no se puede decodificar su duración, no se puede mostrar 0.0 segundos como resultado real."""
    test_ui = dict(
        config.ui,
        voice_mode="tts",
        tts_server="azure-tts-v1",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
    )

    def fake_tts(**kwargs):
        Path(kwargs["voice_file"]).write_bytes(
            b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32
        )
        return object()

    with (
        patch.object(config, "ui", test_ui),
        patch.object(config, "save_config"),
        patch.object(voice, "tts", side_effect=fake_tts),
        patch.object(voice, "get_audio_duration", return_value=0),
    ):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
        app.session_state["ui_language"] = "es"
        app.session_state["video_script"] = "Verificar el aviso de no poder leer la duración del audio de vista previa."
        app.run()
        _button_by_key(
            app,
            "generate_full_voiceover_preview_button",
        ).click().run()

    assert len(app.get("audio")) == 1
    warning_messages = [item.value for item in app.warning]
    assert "El audio se generó, pero no se pudo leer su duración. Revisa los registros." in warning_messages


def test_task_reuses_matching_full_preview_without_calling_tts():
    """Con parámetros idénticos, la tarea real debe reutilizar el audio de vista previa y la línea de tiempo de subtítulos."""
    task_id = "reuse-full-voice-preview"
    task_dir = Path(utils.task_dir(task_id))
    audio_file = task_dir / "audio.mp3"
    audio_file.write_bytes(b"preview audio")
    sub_maker = object()
    script = "La vista previa completa y la tarea real usan el mismo texto."
    params = VideoParams(
        video_subject="preview reuse",
        video_script=script,
        voice_name="zh-CN-XiaoxiaoNeural-Female",
        voice_rate=1.2,
        voice_volume=1.0,
    )
    preview = {
        "audio_file": str(audio_file),
        "duration": 8.2,
        "sub_maker": sub_maker,
        "script": script,
        "voice_name": params.voice_name,
        "voice_rate": params.voice_rate,
        "voice_volume": params.voice_volume,
    }

    try:
        with patch.object(tm.voice, "tts") as synthesize:
            result = tm.generate_audio(
                task_id,
                params,
                script,
                voice_preview=preview,
            )
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)

    assert result == (str(audio_file.resolve()), 9, sub_maker)
    synthesize.assert_not_called()


def test_task_regenerates_audio_when_preview_parameters_changed():
    """Si cambian el texto o los parámetros de voz debe volver al TTS, sin reutilizar una vista previa completa caducada."""
    task_id = "stale-full-voice-preview"
    task_dir = Path(utils.task_dir(task_id))
    audio_file = task_dir / "audio.mp3"
    audio_file.write_bytes(b"stale preview audio")
    script = "La tarea real debe regenerar la voz con la nueva velocidad."
    params = VideoParams(
        video_subject="stale preview",
        video_script=script,
        voice_name="zh-CN-XiaoxiaoNeural-Female",
        voice_rate=1.5,
        voice_volume=1.0,
    )
    preview = {
        "audio_file": str(audio_file),
        "duration": 8.2,
        "sub_maker": object(),
        "script": script,
        "voice_name": params.voice_name,
        "voice_rate": 1.0,
        "voice_volume": params.voice_volume,
    }
    regenerated_sub_maker = object()

    try:
        with (
            patch.object(
                tm.voice,
                "tts",
                return_value=regenerated_sub_maker,
            ) as synthesize,
            patch.object(tm.voice, "get_audio_duration", return_value=6),
        ):
            result = tm.generate_audio(
                task_id,
                params,
                script,
                voice_preview=preview,
            )
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)

    assert result[1:] == (6, regenerated_sub_maker)
    synthesize.assert_called_once()
    assert synthesize.call_args.kwargs["voice_rate"] == 1.5


def test_non_default_volume_regenerates_audio_without_double_gain():
    """Un volumen no predeterminado debe volver al flujo original, para que el TTS y la composición de video no apliquen la ganancia dos veces."""
    task_id = "voice-volume-forwarding"
    task_dir = Path(utils.task_dir(task_id))
    audio_file = task_dir / "audio.mp3"
    audio_file.write_bytes(b"preview with provider-side volume")
    script = "Un volumen no predeterminado requiere generar la voz con el flujo original."
    params = VideoParams(
        video_subject="voice volume",
        video_script=script,
        voice_name="zh-CN-XiaoxiaoNeural-Female",
        voice_rate=1.2,
        voice_volume=1.5,
    )
    sub_maker = object()
    preview = {
        "audio_file": str(audio_file),
        "duration": 5.0,
        "sub_maker": object(),
        "script": script,
        "voice_name": params.voice_name,
        "voice_rate": params.voice_rate,
        "voice_volume": params.voice_volume,
    }

    try:
        with (
            patch.object(tm.voice, "tts", return_value=sub_maker) as synthesize,
            patch.object(tm.voice, "get_audio_duration", return_value=5),
        ):
            result = tm.generate_audio(
                task_id,
                params,
                script,
                voice_preview=preview,
            )
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)

    assert result[1:] == (5, sub_maker)
    synthesize.assert_called_once()
    assert "voice_volume" not in synthesize.call_args.kwargs


def test_webui_worker_forwards_voice_preview_to_pipeline():
    """La capa envolvente de la tarea en segundo plano no puede perder la caché de vista previa ya validada al enviar."""
    preview = {"audio_file": "audio.mp3", "duration": 5.0}
    with (
        patch.object(webui_task.tm, "start", return_value={"videos": []}) as start,
        patch.object(
            webui_task.config,
            "runtime_config_lock",
            return_value=nullcontext(),
        ),
    ):
        webui_task._run_generation(
            "preview-forwarding",
            VideoParams(video_subject="preview forwarding"),
            capture_logs=False,
            voice_preview=preview,
        )

    assert start.call_args.kwargs["voice_preview"] == preview
