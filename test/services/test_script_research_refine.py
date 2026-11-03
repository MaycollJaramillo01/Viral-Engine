from unittest.mock import patch

from app.services import llm


def test_researched_facts_reach_the_draft_and_the_editor_rewrites_it():
    app_config = {"script_research": True, "script_refine_rounds": 1}
    prompts = []

    def fake_generate_response(prompt, app_config=None):
        prompts.append(prompt)
        return "Borrador." if len(prompts) == 1 else "Guion pulido, ¿y tú qué opinas?"

    with patch.object(
        llm, "research_subject", return_value="- 120 detenidos en 2026 [La Prensa]"
    ) as research, patch.object(
        llm, "_generate_response", side_effect=fake_generate_response
    ):
        script = llm.generate_script(
            "extorsión en Honduras", language="es", app_config=app_config
        )

    research.assert_called_once_with("extorsión en Honduras", "es", app_config)
    assert "- 120 detenidos en 2026 [La Prensa]" in prompts[0]
    assert "use only the verified facts" in prompts[1]
    assert prompts[1].endswith("# Draft:\nBorrador.")
    assert script == "Guion pulido, ¿y tú qué opinas?"


def test_failed_refine_round_keeps_the_draft_and_research_is_off_by_default():
    with patch.object(llm, "research_subject") as research, patch.object(
        llm, "_generate_response", side_effect=["Borrador.", "Error: quota exceeded"]
    ) as generate_response:
        script = llm.generate_script(
            "extorsión", app_config={"script_refine_rounds": 3}
        )

    research.assert_not_called()
    assert script == "Borrador."
    assert generate_response.call_count == 2


def test_research_returns_no_facts_when_gemini_fails():
    with patch("google.genai.Client", side_effect=RuntimeError("429 quota")):
        assert llm.research_subject("extorsión", app_config={"gemini_api_key": "k"}) == ""
    assert llm.research_subject("extorsión", app_config={"gemini_api_key": ""}) == ""
