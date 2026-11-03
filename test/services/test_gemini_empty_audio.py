"""Offline Gemini response diagnostics; no provider calls or narration logging."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydub import AudioSegment

from app.services import voice


def _run(tmp_path, status_code, body):
    output = tmp_path / "narration.mp3"
    output.write_bytes(b"previous narration")
    response = SimpleNamespace(status_code=status_code, text="", json=lambda: body)
    messages = []
    sink = voice.logger.add(lambda message: messages.append(message.record["message"]))
    try:
        with (
            patch("app.services.voice.requests.post", return_value=response) as post,
            patch.dict(voice.config.app, {"gemini_api_key": "private-test-key"}),
            patch.object(AudioSegment, "from_file") as decode,
        ):
            result = voice.gemini_tts("private narration", "Achird", 1, str(output))
            post.assert_called_once()
            decode.assert_not_called()
    finally:
        voice.logger.remove(sink)

    assert result is None
    assert output.read_bytes() == b"previous narration"
    assert list(tmp_path.iterdir()) == [output]
    logs = "\n".join(messages)
    assert "private narration" not in logs
    assert "private provider response" not in logs
    assert "private-test-key" not in logs
    return logs


@pytest.mark.parametrize(
    "body, status",
    [
        ({"status": "failed"}, "failed"),
        ({"status": "completed", "steps": []}, "completed"),
        ({"status": "incomplete", "steps": [{"content": []}]}, "incomplete"),
        (
            {
                "status": "completed",
                "steps": [
                    {"content": [{"type": "text", "text": "private provider response"}]}
                ],
            },
            "completed",
        ),
        (
            {"status": "completed", "steps": [{"content": [{"type": "audio", "data": ""}]}]},
            "completed",
        ),
    ],
)
def test_empty_audio_reports_provider_status_without_overwriting_narration(
    tmp_path, body, status
):
    logs = _run(tmp_path, 200, body)
    assert f"No audio data received from Gemini TTS: status={status}" in logs
    assert "Gemini TTS failed" not in logs


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"message": "API key not valid. Please pass a valid API key."}},
        # Google devuelve algunos errores como lista de un objeto.
        [{"error": {"message": "API key not valid. Please pass a valid API key."}}],
    ],
)
def test_http_error_logs_provider_message(tmp_path, body):
    logs = _run(tmp_path, 400, body)
    assert "Gemini TTS failed: HTTP 400, API key not valid" in logs
