from unittest.mock import patch

import pytest

from app.config import config


@pytest.fixture(autouse=True)
def _script_research_and_refine_off(tmp_path):
    # A local config.toml may enable these or pick a paid image source as the
    # default video_source; tests must not call Gemini for real or feed an extra
    # editor round to a mocked model. Saves go to a temp file, or a test that
    # persists settings writes these values into the real config.toml. Caption
    # tests also start from deterministic provider/chunk settings, regardless
    # of the developer's saved narration preferences.
    with (
        patch.dict(
            config.app,
            {"script_research": False, "script_refine_rounds": 0, "video_source": "pexels", "subtitle_provider": "edge"},
        ),
        patch.dict(config.ui, {"subtitle_max_words": 0}),
        patch.object(config, "config_file", str(tmp_path / "config.toml")),
    ):
        yield
