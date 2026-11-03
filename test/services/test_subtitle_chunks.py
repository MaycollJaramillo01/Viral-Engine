import pytest

from app.utils.subtitle_reader import split_long_cues

# A whole clause as Edge TTS or the Gemini timeline writes it: 12 words in 4 s.
CLAUSE = ((2.0, 6.0), "Este año Honduras ya suma más de 12,281 casos de dengue reportados")


def test_long_cue_splits_into_balanced_chunks_that_keep_its_time_span():
    chunks = split_long_cues([CLAUSE], 4)

    assert [text for _, text in chunks] == [
        "Este año Honduras ya",
        "suma más de 12,281",
        "casos de dengue reportados",
    ]
    assert chunks[0][0][0] == 2.0
    assert chunks[-1][0][1] == 6.0
    for (_, previous_end), (next_start, _) in zip(
        (timing for timing, _ in chunks), (timing for timing, _ in chunks[1:])
    ):
        assert previous_end == pytest.approx(next_start)
    # The longest chunk is on screen longest.
    durations = [end - start for (start, end), _ in chunks]
    assert durations[2] == max(durations)


def test_short_cues_cjk_text_and_disabled_splitting_stay_unchanged():
    short = ((0.0, 1.0), "no afuera")
    cjk = ((1.0, 3.0), "你的大脑在你睡觉时监视着你")

    assert split_long_cues([short, cjk], 4) == [short, cjk]
    assert split_long_cues([CLAUSE], 0) == [CLAUSE]
