from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.config import config
from app.models.schema import VideoConcatMode, VideoParams
from app.services import llm, material, subtitle, task, voice, wikimedia


def test_short_captions_use_word_boundaries_and_preserve_silence(tmp_path):
    words = [
        SimpleNamespace(word=" Pablo", start=0.35, end=0.7),
        SimpleNamespace(word=" Escobar", start=0.7, end=1.8),
        SimpleNamespace(word=" llegó", start=1.85, end=2.1),
        SimpleNamespace(word=" al", start=3.2, end=3.4),
        SimpleNamespace(word=" Congreso.", start=3.4, end=4.0),
    ]
    segment = SimpleNamespace(words=words, text="", start=0, end=4.7)
    model = SimpleNamespace(transcribe=lambda *a, **k: (
        [segment], SimpleNamespace(language="es", language_probability=1)
    ))
    destination = tmp_path / "aligned.srt"
    with patch.object(subtitle, "WhisperModel", object()), patch.object(subtitle, "model", model):
        subtitle.create("audio.mp3", str(destination), max_words=2)
    cues = subtitle.file_to_subtitles(str(destination))
    assert [(c[1], c[2]) for c in cues] == [
        ("00:00:00,350 --> 00:00:01,800", "Pablo Escobar"),
        ("00:00:01,850 --> 00:00:02,100", "llegó"),
        ("00:00:03,200 --> 00:00:04,000", "al Congreso."),
    ]


def test_estimated_tts_timings_select_audio_alignment(tmp_path):
    maker = voice.populate_legacy_submaker_with_full_text(voice.SubMaker(), "Pablo Escobar llegó.", 4)
    params = VideoParams(video_subject="Pablo Escobar", voice_name="gemini:Charon-Informative")

    def transcribe(**kwargs):
        assert kwargs["reference_text"] == "Pablo Escobar llegó."
        assert kwargs["max_words"] == 4
        Path(kwargs["subtitle_file"]).write_text(
            "1\n00:00:00,450 --> 00:00:02,900\nPablo Escobar llegó.\n\n", encoding="utf-8"
        )

    with patch.dict(config.app, {"subtitle_provider": "edge"}), patch.dict(config.ui, {"subtitle_max_words": 4}), patch(
        "app.services.task.utils.task_dir", return_value=str(tmp_path)
    ), patch.object(subtitle, "create", side_effect=transcribe), patch.object(voice, "create_subtitle") as estimated, patch.object(subtitle, "correct") as correct:
        result = task.generate_subtitle("aligned", params, "Pablo Escobar llegó.", maker, "actual.mp3")
    assert "00:00:00,450" in Path(result).read_text(encoding="utf-8")
    estimated.assert_not_called()
    correct.assert_not_called()


def test_native_tts_timings_do_not_load_whisper(tmp_path):
    params = VideoParams(video_subject="Native timings", voice_name="en-US-AriaNeural")

    def native(**kwargs):
        Path(kwargs["subtitle_file"]).write_text(
            "1\n00:00:00,300 --> 00:00:01,200\nHello\n\n", encoding="utf-8"
        )

    with patch.dict(config.app, {"subtitle_provider": "auto"}), patch(
        "app.services.task.utils.task_dir", return_value=str(tmp_path)
    ), patch.object(voice, "create_subtitle", side_effect=native), patch.object(subtitle, "create") as whisper:
        result = task.generate_subtitle("native", params, "Hello", SimpleNamespace(), "actual.mp3")
    assert Path(result).is_file()
    whisper.assert_not_called()


def _portrait_responses(human=True, description="Police photograph", filename="Portrait.jpg"):
    return [
        {"query": {"pages": {"1": {"pageprops": {"wikibase_item": "Q123"}}}}},
        {"entities": {"Q123": {"claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5" if human else "Q43229"}}}}],
            "P18": [{"mainsnak": {"datavalue": {"value": filename}}}],
        }}}},
        {"query": {"pages": {"2": {"imageinfo": [{
            "url": "https://upload.wikimedia.org/portrait.jpg", "size": 1024,
            "width": 800, "height": 1200, "mime": "image/jpeg",
            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Portrait.jpg",
            "extmetadata": {"LicenseShortName": {"value": "Public domain"},
                            "ImageDescription": {"value": description},
                            "Artist": {"value": "<b>Archive</b>"}},
        }]}}}},
    ]


def test_portraits_require_named_human_identity_and_keep_credits():
    with patch.object(wikimedia, "_api", side_effect=_portrait_responses()) as api:
        photo = wikimedia.find_portrait("Pablo Escobar")
    assert api.call_args_list[0].kwargs["titles"] == "Pablo Escobar"
    assert photo["asset_id"] == "Q123"
    assert photo["license"] == "Public domain"
    assert photo["creator"]["name"] == "Archive"


@pytest.mark.parametrize("description", ["AI-generated portrait", "Oil painting of the person", "A statue"])
def test_generated_faces_and_artwork_are_rejected(description):
    responses = _portrait_responses(description=description)
    with patch.object(wikimedia, "_api", side_effect=responses * 2):
        with pytest.raises(wikimedia.PortraitNotFound):
            wikimedia.find_portrait("Pablo Escobar")


def test_nonhuman_and_disambiguation_pages_are_rejected():
    responses = _portrait_responses(human=False)
    with patch.object(wikimedia, "_api", side_effect=responses[:2] * 2):
        with pytest.raises(wikimedia.PortraitNotFound):
            wikimedia.find_portrait("Congress")


def test_documentary_materials_use_a_different_photo_for_every_scene(tmp_path):
    from PIL import Image
    import numpy as np

    def download(photo, directory, duration, **kwargs):
        path = tmp_path / (photo["asset_id"] + ".jpg")
        seed = sum(map(ord, photo["asset_id"]))
        Image.fromarray(np.random.default_rng(seed).integers(0, 256, (160, 160, 3), dtype=np.uint8)).save(path)
        item = material.MaterialInfo()
        item.provider, item.url, item.duration = "editorial_archive", str(path), duration
        item.source_info = {"search_term": photo["person"], "asset_id": photo["asset_id"]}
        return item

    def gallery(name, **kwargs):
        return [{"person": name, "asset_id": name + str(i)} for i in range(3)]

    with patch.object(wikimedia, "find_portraits", side_effect=gallery), patch.object(
        wikimedia, "download_portrait", side_effect=download
    ), patch.object(material, "_render_openai_image_video", side_effect=lambda image, *a, **k: image + ".mp4"), patch.object(
        material, "_persist_material_sources"
    ), patch.object(material, "generate_images_openai") as generated:
        paths = material.download_videos(
            "documentary", ["Pablo Escobar", "Rodrigo Lara Bonilla"], source="wikimedia",
            audio_duration=20, max_clip_duration=5, video_concat_mode=VideoConcatMode.random,
        )
    assert len(paths) == len(set(paths)) == 4
    assert "Pablo Escobar" in paths[0]
    assert "Rodrigo Lara Bonilla" in paths[-1]
    generated.assert_not_called()


def test_one_face_cannot_be_repeated_to_fill_the_video(tmp_path):
    from PIL import Image
    import numpy as np

    image = tmp_path / "same-photo.jpg"
    Image.fromarray(np.random.default_rng(42).integers(0, 256, (160, 160, 3), dtype=np.uint8)).save(image)
    item = material.MaterialInfo()
    item.url, item.duration = str(image), 5
    with patch.object(wikimedia, "find_portraits", return_value=[{"asset_id": "one"}, {"asset_id": "same-image-another-url"}]), patch.object(
        wikimedia, "download_portrait", return_value=item
    ), patch.object(material, "_render_openai_image_video") as render:
        with pytest.raises(wikimedia.PortraitNotFound, match="fotografías distintas"):
            material.download_videos("doc", ["Pablo Escobar"], source="wikimedia", audio_duration=20)
    render.assert_not_called()


def test_visual_duplicates_are_detected_after_resize_and_recompression(tmp_path):
    from PIL import Image
    import numpy as np

    image = Image.fromarray(np.random.default_rng(100).integers(0, 256, (400, 400, 3), dtype=np.uint8))
    image.save(tmp_path / "original.png")
    image.resize((200, 200)).save(tmp_path / "copy.jpg", quality=80)
    original = wikimedia.photo_fingerprints(str(tmp_path / "original.png"))
    copied = wikimedia.photo_fingerprints(str(tmp_path / "copy.jpg"))
    assert min((a ^ b).bit_count() for a in original for b in copied) <= 5


def test_reviewed_archive_catalog_carries_distinct_assets_and_sources():
    import json
    photos = json.loads(Path(wikimedia.__file__).with_name("documentary_photos.json").read_text(encoding="utf-8"))
    assert len({p["asset_id"] for p in photos}) == len(photos)
    assert sum(p["person"] == "Pablo Escobar" for p in photos) >= 6
    assert all(p["reviewed"] and p["source_page"].startswith("https://") and p["creator"] for p in photos)


def test_archive_gallery_survives_unavailable_wiki_lookup():
    import requests
    with patch.object(wikimedia, "find_portrait", side_effect=requests.ConnectionError):
        photos = wikimedia.find_portraits("Pablo Escobar")
    assert len(photos) >= 6
    assert photos[0]["provider"] == "editorial_archive"
    assert len({p["asset_id"] for p in photos}) == len(photos)


def test_portrait_terms_keep_names_and_repeats_in_narration_order():
    with patch.object(llm, "_generate_response", return_value='["Pablo Escobar","Rodrigo Lara Bonilla","Pablo Escobar","invented person"]') as generate:
        terms = llm.generate_terms(
            "Pablo Escobar", "Pablo Escobar y Rodrigo Lara Bonilla.", amount=40, real_people=True,
        )
    assert terms == ["Pablo Escobar", "Rodrigo Lara Bonilla", "Pablo Escobar"]
    assert "stock" not in generate.call_args.args[0].lower().split("never")[0]


def test_opening_title_is_separate_and_bounded():
    with patch.object(llm, "_generate_response", return_value="Escobar llegó al Congreso"):
        assert llm.generate_opening_title("Pablo Escobar", "Narración.", "es") == "Escobar llegó al Congreso"
    with patch.object(llm, "_generate_response", return_value="Error: provider unavailable"):
        title = llm.generate_opening_title("Pablo Escobar: el narcotraficante que llegó al Congreso", "Narración.", "es")
    assert len(title) <= 60
    assert len(title.split()) <= 8
    assert "Escobar" in title
