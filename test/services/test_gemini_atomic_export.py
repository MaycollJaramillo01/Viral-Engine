import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from pydub import AudioSegment
from app.services import voice


class TestGeminiAtomicExport(unittest.TestCase):
    def test_failed_encoder_preserves_previous_narration_and_cleans_staging(self):
        tone = AudioSegment.silent(duration=100).set_frame_rate(24000).set_channels(1).set_sample_width(2)
        audio = {"type": "audio", "data": base64.b64encode(tone.raw_data).decode("ascii")}
        response = SimpleNamespace(status_code=200, text="", json=lambda: {"steps": [{"content": [audio]}]})
        def failed_export(segment, out_f, **kwargs):
            Path(out_f).write_bytes(b"partial encoder output")
            raise OSError("encoder failed")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "narration.mp3"
            output.write_bytes(b"previous narration")
            with patch("app.services.voice.requests.post", return_value=response), patch.dict(voice.config.app, {"gemini_api_key": "test-key"}), patch.object(AudioSegment, "export", failed_export):
                result = voice.gemini_tts("Hello.", "Zephyr", 1, str(output))
            self.assertIsNone(result)
            self.assertEqual(output.read_bytes(), b"previous narration")
            self.assertEqual(list(Path(tmp).iterdir()), [output])
