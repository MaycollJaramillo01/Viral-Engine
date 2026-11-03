from unittest.mock import patch

import pytest

from app.config import config


@pytest.fixture(autouse=True)
def _script_research_and_refine_off():
    # A local config.toml may enable these; tests must not call Gemini for real
    # or feed an extra editor round to a mocked model.
    with patch.dict(config.app, {"script_research": False, "script_refine_rounds": 0}):
        yield
