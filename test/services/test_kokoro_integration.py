"""Prueba con el servicio real activada explícitamente; la CI normal no necesita instalar ni acceder a Kokoro.

Tras arrancar el servicio local, define MPT_KOKORO_TEST_BASE_URL=http://127.0.0.1:8880/v1
y ejecuta este archivo. La prueba solo sintetiza texto corto, sin llamar al LLM, plataformas de material ni interfaces de publicación.
"""

import os

import pytest

from app.config import config
from app.services import voice


BASE_URL = os.environ.get("MPT_KOKORO_TEST_BASE_URL", "")
pytestmark = pytest.mark.skipif(not BASE_URL, reason="requires an explicitly configured Kokoro test server")


@pytest.fixture
def live_config(monkeypatch):
    monkeypatch.setattr(config, "kokoro", {"base_url": BASE_URL, "api_key": "",
                                          "model_id": "kokoro", "voices": []})


def test_live_voice_discovery(live_config):
    options = voice.get_kokoro_voices(fallback=False)
    assert "kokoro:af_heart" in options
    assert "kokoro:zf_xiaobei" in options
    assert all(option.startswith("kokoro:") and "{" not in option for option in options)


@pytest.mark.parametrize("speaker, text", [
    ("af_heart", "Artificial intelligence is changing everyday life."),
    ("zf_xiaobei", "La inteligencia artificial está cambiando nuestras vidas. Usemos la tecnología con criterio para que todos se beneficien."),
    ("ff_siwis", "Bonjour ! L'intelligence artificielle nous aide à apprendre."),
    ("ef_dora", "La inteligencia artificial mejora nuestra vida cotidiana."),
    ("if_sara", "L'intelligenza artificiale migliora la vita di ogni giorno."),
    ("pf_dora", "A inteligência artificial melhora a nossa vida."),
    ("hf_alpha", "नमस्ते। कृत्रिम बुद्धिमत्ता हमारे जीवन को बेहतर बनाती है।"),
])
def test_live_audio_and_subtitles(live_config, tmp_path, speaker, text):
    output = tmp_path / "audio.mp3"
    maker = voice.tts(text, f"kokoro:{speaker}", 1, str(output))
    assert maker is not None
    assert output.stat().st_size > 100
    assert voice.get_audio_duration(str(output)) > 0
    subtitle = tmp_path / "subtitle.srt"
    voice.create_subtitle(maker, text, str(subtitle))
    assert "-->" in subtitle.read_text(encoding="utf-8")
    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".mp3", ".srt"]
