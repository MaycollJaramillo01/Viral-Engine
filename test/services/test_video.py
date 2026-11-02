import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from loguru import logger
from moviepy import (
    ImageClip,
    VideoFileClip,
)

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import MaterialInfo
from app.services import video as vd
from app.utils import logging_utils, utils

resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")


@contextmanager
def _capture_task_scoped_logs():
    """
    Recoge los logs con la misma regla que el log de tareas de la WebUI: solo se conservan los registros que pertenecen al hilo actual.

    Parchear directamente ``logger.info`` no permite distinguir de qué hilo viene el log, y la causa de que la WebUI pierda logs
    es justamente la pertenencia al hilo, así que aquí se verifica con un sink real de loguru y un filtro por ámbito.
    """
    messages = []
    root_thread_id = threading.get_ident()
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="DEBUG",
        filter=lambda record: (
            logging_utils.log_scope_thread_id(record["thread"].id) == root_thread_id
        ),
    )
    try:
        yield messages
    finally:
        logger.remove(handler_id)


class _FakeMoviePyClip:
    """Proporciona una interfaz MoviePy mínima para las pruebas unitarias de la mezcla final, evitando que la CI codifique de verdad videos grandes."""

    def __init__(self, *, duration=5, fps=44100):
        self.duration = duration
        self.fps = fps
        self.close_calls = 0
        self.with_audio_result = self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        self.close_calls += 1

    def with_effects(self, _effects):
        return self

    def with_audio(self, _audio):
        return self.with_audio_result


class TestVideoService(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        self.test_img_path = os.path.join(resources_dir, "1.png")
        vd._runtime_disabled_video_codecs.clear()
        vd._ffmpeg_encoder_exists.cache_clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        vd._runtime_disabled_video_codecs.clear()
        vd._ffmpeg_encoder_exists.cache_clear()

    def test_clip_processing_concurrency_defaults_to_serial(self):
        """Sin configurar o con configuración inválida se mantiene en serie; un valor explícito sigue aplicándose dentro del rango seguro."""
        with patch.dict(config.app, {}, clear=True):
            self.assertEqual(vd._get_clip_processing_concurrency(), 1)
        for value, expected in (("bad", 1), (0, 1), (4, 4), (99, 8)):
            with self.subTest(value=value):
                with patch.dict(config.app, {"video_clip_concurrency": value}):
                    self.assertEqual(vd._get_clip_processing_concurrency(), expected)

    def test_generate_video_rejects_font_outside_directory_before_opening_media(self):
        """WebUI, la CLI o las llamadas internas que se saltan la API: la capa de renderizado también debe bloquear las fuentes fuera de límites."""
        with tempfile.TemporaryDirectory() as temp_dir:
            font_dir = Path(temp_dir, "fonts")
            font_dir.mkdir()
            outside = Path(temp_dir, "outside.ttf")
            outside.write_bytes(b"not a font")

            for font_name in (str(outside), "../outside.ttf"):
                with (
                    self.subTest(font_name=font_name),
                    patch.object(vd.utils, "font_dir", return_value=str(font_dir)),
                    patch.object(vd, "_open_video_clip_quietly") as open_video,
                ):
                    params = vd.VideoParams(video_subject="Coffee", font_name=font_name)
                    with self.assertRaisesRegex(ValueError, "outside the allowed directory"):
                        vd.generate_video(
                            video_path="unused.mp4",
                            audio_path="unused.mp3",
                            subtitle_path="unused.srt",
                            output_file="unused-output.mp4",
                            params=params,
                        )
                    open_video.assert_not_called()

    def test_generate_video_accepts_bundled_font_before_opening_media(self):
        """Las fuentes incluidas deben seguir pasando la validación, sin bloquear la cadena de generación de subtítulos por defecto."""
        params = vd.VideoParams(video_subject="Coffee", font_name="STHeitiMedium.ttc")
        with patch.object(
            vd, "_open_video_clip_quietly", side_effect=RuntimeError("media reached")
        ) as open_video:
            with self.assertRaisesRegex(RuntimeError, "media reached"):
                vd.generate_video(
                    video_path="unused.mp4",
                    audio_path="unused.mp3",
                    subtitle_path="unused.srt",
                    output_file="unused-output.mp4",
                    params=params,
                )
        open_video.assert_called_once_with("unused.mp4")

    def test_subtitle_spring_animation_keeps_color_and_mask_aligned(self):
        """
        La animación de rebote debe escalar a la vez los fotogramas de color y la máscara transparente.

        La implementación antigua solo escalaba los fotogramas de color; el primer fotograma seguía usando la máscara del tamaño original, y al componer aparecía brevemente un
        contorno negro del texto. Usar una imagen blanca pura y una máscara completa permite comparar con precisión las regiones de píxeles válidos de ambos.
        """
        color_frame = vd.np.full((20, 30, 3), 255, dtype=vd.np.uint8)
        mask_frame = vd.np.ones((20, 30), dtype=float)
        clip = (
            ImageClip(color_frame)
            .with_mask(ImageClip(mask_frame, is_mask=True))
            .with_duration(1)
        )
        animated = vd._apply_subtitle_spring_animation(clip, 1)

        try:
            initial_color = vd.np.any(animated.get_frame(0) > 0, axis=2)
            initial_mask = animated.mask.get_frame(0) > 0
            vd.np.testing.assert_array_equal(initial_color, initial_mask)
            self.assertLess(initial_color.sum(), color_frame.shape[0] * color_frame.shape[1])

            # Al terminar la animación debe restaurarse exactamente el tamaño original, para que los subtítulos largos no queden borrosos ni escalados.
            settled_color = animated.get_frame(
                vd._SUBTITLE_SPRING_DURATION_SECONDS
            )
            settled_mask = animated.mask.get_frame(
                vd._SUBTITLE_SPRING_DURATION_SECONDS
            )
            vd.np.testing.assert_array_equal(settled_color, color_frame)
            vd.np.testing.assert_array_equal(settled_mask, mask_frame)
        finally:
            vd.close_clip(animated)
            vd.close_clip(clip)

    def test_subtitle_spring_scale_handles_time_boundaries(self):
        """Duración cero, tiempos negativos y el punto final de la animación no pueden producir divisiones por cero ni proporciones de escala ilegales."""
        duration = vd._SUBTITLE_SPRING_DURATION_SECONDS

        self.assertEqual(vd._get_subtitle_spring_scale(0, duration), 0.05)
        self.assertEqual(vd._get_subtitle_spring_scale(-1, duration), 0.05)
        self.assertEqual(vd._get_subtitle_spring_scale(duration, duration), 1.0)
        self.assertEqual(vd._get_subtitle_spring_scale(1, 0), 1.0)

    def test_scale_subtitle_frame_rejects_unsupported_shapes(self):
        """Canales o dimensiones anómalos deben fallar claramente, para no pasar fotogramas dañados al codificador de video."""
        with self.assertRaisesRegex(ValueError, "2D mask or 3D color"):
            vd._scale_subtitle_frame_on_canvas(vd.np.zeros((8,)), 0.5)
        with self.assertRaisesRegex(ValueError, "RGB or RGBA"):
            vd._scale_subtitle_frame_on_canvas(
                vd.np.zeros((8, 8, 2), dtype=vd.np.uint8),
                0.5,
            )

    def test_fit_clip_cover_fills_portrait_canvas_without_black_bars(self):
        source_color = [17, 34, 51]
        source = ImageClip(
            vd.np.full((90, 160, 3), source_color, dtype=vd.np.uint8)
        ).with_duration(1)
        fitted = vd._fit_clip_to_canvas(
            source,
            target_width=90,
            target_height=160,
            fit_mode=vd.VideoFitMode.cover,
        )

        try:
            self.assertEqual(tuple(fitted.size), (90, 160))
            frame = fitted.get_frame(0)
            self.assertEqual(frame[0, 45].tolist(), source_color)
            self.assertEqual(frame[-1, 45].tolist(), source_color)
        finally:
            vd.close_clip(fitted)
            vd.close_clip(source)

    def test_fit_clip_contain_preserves_legacy_black_bars(self):
        source_color = [17, 34, 51]
        source = ImageClip(
            vd.np.full((90, 160, 3), source_color, dtype=vd.np.uint8)
        ).with_duration(1)
        fitted = vd._fit_clip_to_canvas(
            source,
            target_width=90,
            target_height=160,
            fit_mode=vd.VideoFitMode.contain,
        )

        try:
            self.assertEqual(tuple(fitted.size), (90, 160))
            frame = fitted.get_frame(0)
            self.assertEqual(frame[0, 45].tolist(), [0, 0, 0])
            self.assertEqual(frame[80, 45].tolist(), source_color)
        finally:
            vd.close_clip(fitted)
            vd.close_clip(source)

    def test_delete_files_deduplicates_paths_and_ignores_missing_files(self):
        """
        Los fragmentos en bucle hacen que la misma ruta aparezca repetida en la lista de concatenación; al limpiar, cada ruta solo puede borrarse una vez.

        Que un archivo ya no exista es un estado normal de la limpieza idempotente y no debería producir logs de fallo que confundan al usuario.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_file = os.path.join(temp_dir, "temp-clip-1.mp4")
            missing_file = os.path.join(temp_dir, "already-removed.mp4")
            Path(existing_file).write_bytes(b"temporary clip")

            original_remove = os.remove
            with (
                patch.object(vd.os, "remove", wraps=original_remove) as remove,
                patch.object(vd.logger, "warning") as warning,
            ):
                vd.delete_files(
                    [
                        existing_file,
                        existing_file,
                        missing_file,
                        missing_file,
                    ]
                )

        self.assertEqual(
            [item.args[0] for item in remove.call_args_list],
            [existing_file, missing_file],
        )
        warning.assert_not_called()

    def test_delete_files_logs_actionable_os_errors(self):
        """Los fallos reales de limpieza, como los de permisos, deben conservar la ruta y el error del sistema, para localizar archivos residuales."""
        with (
            patch.object(
                vd.os,
                "remove",
                side_effect=PermissionError("permission denied"),
            ),
            patch.object(vd.logger, "warning") as warning,
        ):
            vd.delete_files(["protected-temp-clip.mp4"])

        warning.assert_called_once()
        message = warning.call_args.args[0]
        self.assertIn("protected-temp-clip.mp4", message)
        self.assertIn("permission denied", message)

    def test_generate_video_ignores_existing_subtitle_when_disabled(self):
        params = vd.VideoParams(
            video_subject="test", subtitle_enabled=False, bgm_type=""
        )
        source_video = _FakeMoviePyClip()
        voice_source = _FakeMoviePyClip()
        with tempfile.TemporaryDirectory() as tmp_dir:
            stale_subtitle = Path(tmp_dir) / "stale.srt"
            stale_subtitle.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nOld caption\n\n", encoding="utf-8"
            )
            with (
                patch.object(vd, "_open_video_clip_quietly", return_value=source_video),
                patch.object(vd, "AudioFileClip", return_value=voice_source),
                patch.object(vd, "SubtitlesClip", side_effect=AssertionError(
                    "disabled subtitles must not be parsed"
                )) as subtitle_loader,
                patch.object(vd, "TextClip") as text_renderer,
                patch.object(vd, "_write_videofile_with_codec_fallback") as writer,
                patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
            ):
                result = vd.generate_video(
                    video_path="combined.mp4", audio_path="voice.mp3",
                    subtitle_path=str(stale_subtitle), output_file="final.mp4", params=params,
                )
        self.assertTrue(result)
        subtitle_loader.assert_not_called()
        text_renderer.assert_not_called()
        writer.assert_called_once()
        self.assertEqual(source_video.close_calls, 2)
        self.assertEqual(voice_source.close_calls, 1)

    def test_generate_video_reports_successful_bgm_mix_and_closes_sources(self):
        """BGM Tras una mezcla exitosa debe devolver True y liberar todos los readers de los archivos originales."""
        params = vd.VideoParams(
            video_subject="test",
            subtitle_enabled=False,
            bgm_type="sonilo",
        )
        source_video = _FakeMoviePyClip()
        voice_source = _FakeMoviePyClip()
        bgm_source = _FakeMoviePyClip()
        mixed_audio = _FakeMoviePyClip(fps=48000)
        final_video = _FakeMoviePyClip()
        source_video.with_audio_result = final_video

        with (
            patch.object(
                vd, "_open_video_clip_quietly", return_value=source_video
            ),
            patch.object(
                vd, "AudioFileClip", side_effect=[voice_source, bgm_source]
            ),
            patch.object(vd, "CompositeAudioClip", return_value=mixed_audio),
            patch.object(vd, "_write_videofile_with_codec_fallback") as writer,
            patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
        ):
            result = vd.generate_video(
                video_path="combined.mp4",
                audio_path="voice.mp3",
                subtitle_path="",
                output_file="final.mp4",
                params=params,
                bgm_file_override="sonilo.m4a",
            )

        self.assertTrue(result)
        writer.assert_called_once()
        self.assertTrue(writer.call_args.kwargs["atomic_output"])
        self.assertEqual(writer.call_args.kwargs["audio_fps"], 48000)
        self.assertEqual(source_video.close_calls, 1)
        self.assertEqual(voice_source.close_calls, 1)
        self.assertEqual(bgm_source.close_calls, 1)
        self.assertEqual(final_video.close_calls, 1)

    def test_generate_video_keeps_output_and_reports_failed_bgm_mix(self):
        """BGM Si falla la apertura debe escribir una sola vez el video sin BGM y devolver False."""
        params = vd.VideoParams(
            video_subject="test",
            subtitle_enabled=False,
            bgm_type="sonilo",
        )
        source_video = _FakeMoviePyClip()
        voice_source = _FakeMoviePyClip()
        final_video = _FakeMoviePyClip()
        source_video.with_audio_result = final_video

        with (
            patch.object(
                vd, "_open_video_clip_quietly", return_value=source_video
            ),
            patch.object(
                vd,
                "AudioFileClip",
                side_effect=[voice_source, RuntimeError("invalid BGM")],
            ),
            patch.object(vd, "CompositeAudioClip") as composite_audio,
            patch.object(vd, "_write_videofile_with_codec_fallback") as writer,
            patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
            patch.object(vd.logger, "exception") as log_exception,
        ):
            result = vd.generate_video(
                video_path="combined.mp4",
                audio_path="voice.mp3",
                subtitle_path="",
                output_file="final.mp4",
                params=params,
                bgm_file_override="broken.m4a",
            )

        self.assertFalse(result)
        writer.assert_called_once()
        composite_audio.assert_not_called()
        log_exception.assert_called_once()
        self.assertEqual(source_video.close_calls, 1)
        self.assertEqual(voice_source.close_calls, 1)
        self.assertEqual(final_video.close_calls, 1)

    def test_generate_video_skips_every_bgm_source_when_volume_is_zero(self):
        """0 el volumen debe cortocircuitar de forma unificada la fuente actual y los proveedores futuros antes de analizar archivos."""
        test_cases = [
            ("random", None),
            ("custom", None),
            ("sonilo", "sonilo.m4a"),
            ("future_provider", "future-provider.wav"),
        ]
        for bgm_type, bgm_override in test_cases:
            with self.subTest(bgm_type=bgm_type):
                params = vd.VideoParams(
                    video_subject="test",
                    subtitle_enabled=False,
                    bgm_type=bgm_type,
                    bgm_file="missing-background.mp3",
                    bgm_volume=0.0,
                )
                source_video = _FakeMoviePyClip()
                voice_source = _FakeMoviePyClip()
                final_video = _FakeMoviePyClip()
                source_video.with_audio_result = final_video

                with (
                    patch.object(
                        vd,
                        "_open_video_clip_quietly",
                        return_value=source_video,
                    ),
                    patch.object(
                        vd, "AudioFileClip", return_value=voice_source
                    ) as audio_file_clip,
                    patch.object(vd, "get_bgm_file") as get_bgm_file,
                    patch.object(vd, "CompositeAudioClip") as composite_audio,
                    patch.object(
                        vd, "_write_videofile_with_codec_fallback"
                    ) as writer,
                    patch.object(
                        vd, "_get_configured_video_codec", return_value="libx264"
                    ),
                ):
                    result = vd.generate_video(
                        video_path="combined.mp4",
                        audio_path="voice.mp3",
                        subtitle_path="",
                        output_file="final.mp4",
                        params=params,
                        bgm_file_override=bgm_override,
                    )

                self.assertTrue(result)
                audio_file_clip.assert_called_once_with("voice.mp3")
                get_bgm_file.assert_not_called()
                composite_audio.assert_not_called()
                writer.assert_called_once()
                self.assertEqual(source_video.close_calls, 1)
                self.assertEqual(voice_source.close_calls, 1)
                self.assertEqual(final_video.close_calls, 1)

    def test_generate_video_chooses_looping_by_bgm_file_source(self):
        """La biblioteca de música por defecto necesita bucle; el archivo de ajuste de duración que proporciona la capa de tareas no debería depender del nombre del proveedor."""
        test_cases = [
            ("random", None, True),
            ("custom", None, True),
            ("sonilo", "sonilo.m4a", False),
            ("future_provider", "future-provider.wav", False),
        ]
        for bgm_type, bgm_override, should_loop in test_cases:
            with self.subTest(bgm_type=bgm_type, bgm_override=bgm_override):
                params = vd.VideoParams(
                    video_subject="test",
                    subtitle_enabled=False,
                    bgm_type=bgm_type,
                    bgm_file="library.mp3",
                    bgm_volume=0.2,
                )
                source_video = _FakeMoviePyClip()
                voice_source = _FakeMoviePyClip()
                bgm_source = _FakeMoviePyClip()
                mixed_audio = _FakeMoviePyClip()
                final_video = _FakeMoviePyClip()
                source_video.with_audio_result = final_video

                with (
                    patch.object(
                        vd,
                        "_open_video_clip_quietly",
                        return_value=source_video,
                    ),
                    patch.object(
                        vd,
                        "AudioFileClip",
                        side_effect=[voice_source, bgm_source],
                    ),
                    patch.object(vd, "get_bgm_file", return_value="library.mp3"),
                    patch.object(vd, "CompositeAudioClip", return_value=mixed_audio),
                    patch.object(vd.afx, "AudioLoop") as audio_loop,
                    patch.object(vd, "_write_videofile_with_codec_fallback"),
                    patch.object(
                        vd, "_get_configured_video_codec", return_value="libx264"
                    ),
                ):
                    result = vd.generate_video(
                        video_path="combined.mp4",
                        audio_path="voice.mp3",
                        subtitle_path="",
                        output_file="final.mp4",
                        params=params,
                        bgm_file_override=bgm_override,
                    )

                self.assertTrue(result)
                if should_loop:
                    audio_loop.assert_called_once_with(duration=source_video.duration)
                else:
                    audio_loop.assert_not_called()

    def test_preprocess_video(self):
        if not os.path.exists(self.test_img_path):
            self.fail(f"test image not found: {self.test_img_path}")

        local_videos_dir = utils.storage_dir("local_videos", create=True)
        safe_img_path = os.path.join(local_videos_dir, "test-preprocess-1.png")
        shutil.copy2(self.test_img_path, safe_img_path)

        # test preprocess_video function
        m = MaterialInfo()
        m.url = os.path.basename(safe_img_path)
        m.provider = "local"

        try:
            materials = vd.preprocess_video([m], clip_duration=4)

            # verify result
            self.assertIsNotNone(materials)
            self.assertEqual(len(materials), 1)
            self.assertTrue(materials[0].url.endswith(".mp4"))

            # moviepy get video info
            clip = VideoFileClip(materials[0].url)
            try:
                self.assertGreater(clip.duration, 0)
            finally:
                clip.close()

            # clean generated test video file
            if os.path.exists(materials[0].url):
                os.remove(materials[0].url)
        finally:
            if os.path.exists(safe_img_path):
                os.remove(safe_img_path)

    def test_image_zoom_renders_keep_distinct_clip_durations(self):
        """Two tasks must not overwrite one image render with another duration."""
        class FakeImageClip:
            mask = None

            def get_frame(self, _time):
                return np.zeros((2, 2, 3), dtype=np.uint8)

            def __init__(self, _path):
                self.duration = 0

            def with_duration(self, duration):
                self.duration = duration
                return self

            def with_position(self, _position):
                return self

            def resized(self, _scale):
                return self

        class FakeCompositeClip:
            def __init__(self, clips, bg_color=None):
                self.duration = clips[0].duration

            def write_videofile(self, output, **_kwargs):
                Path(output).write_bytes(f"duration={self.duration}".encode())

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "image.png")
            with (
                patch.object(vd, "ImageClip", FakeImageClip),
                patch.object(vd, "CompositeVideoClip", FakeCompositeClip),
            ):
                first = vd.render_image_zoom_video(image_path, clip_duration=4)
                second = vd.render_image_zoom_video(image_path, clip_duration=7)

            self.assertNotEqual(first, second)
            self.assertEqual(Path(first).read_bytes(), b"duration=4")
            self.assertEqual(Path(second).read_bytes(), b"duration=7")

    def test_failed_image_zoom_render_preserves_previous_complete_clip(self):
        """A failed rerender must leave the last verified MP4 available."""
        class FakeImageClip:
            mask = None

            def get_frame(self, _time):
                return np.zeros((2, 2, 3), dtype=np.uint8)

            duration = 0

            def __init__(self, _path):
                pass

            def with_duration(self, duration):
                self.duration = duration
                return self

            def with_position(self, _position):
                return self

            def resized(self, _scale):
                return self

        writes = 0

        class FakeCompositeClip:
            def __init__(self, _clips, bg_color=None):
                pass

            def write_videofile(self, output, **_kwargs):
                nonlocal writes
                writes += 1
                Path(output).write_bytes(b"complete" if writes == 1 else b"partial")
                if writes == 2:
                    raise RuntimeError("render interrupted")

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "image.png")
            with (
                patch.object(vd, "ImageClip", FakeImageClip),
                patch.object(vd, "CompositeVideoClip", FakeCompositeClip),
            ):
                output = vd.render_image_zoom_video(image_path, clip_duration=5)
                with self.assertRaisesRegex(RuntimeError, "render interrupted"):
                    vd.render_image_zoom_video(image_path, clip_duration=5)

            self.assertEqual(Path(output).read_bytes(), b"complete")
            self.assertEqual(
                sorted(path.name for path in Path(temp_dir).iterdir()),
                [Path(output).name],
            )

    def test_preprocess_video_rejects_material_outside_local_videos(self):
        """
        local La ruta del material viene de parámetros de la API; no puede permitirse que cualquier ruta absoluta entre en MoviePy.
        Aquí se verifica que las rutas fuera del directorio de lista blanca local_videos se omiten, para evitar la lectura de archivos arbitrarios.
        """
        m = MaterialInfo(provider="local", url=self.test_img_path)

        materials = vd.preprocess_video([m], clip_duration=4)

        self.assertEqual(materials, [])

    def test_get_bgm_file_accepts_song_directory_filename(self):
        """
        BGM La interfaz de listado ahora solo expone nombres de archivo; al generar el video, el nombre debe poder resolverse con seguridad de vuelta al
        resource/songs directorio de lista blanca, manteniendo utilizable la ruta de uso normal.
        """
        song_dir = utils.song_dir()
        bgm_path = os.path.join(song_dir, "test-safe-bgm.mp3")
        Path(bgm_path).write_bytes(b"fake-mp3")

        try:
            self.assertEqual(vd.get_bgm_file(bgm_file="test-safe-bgm.mp3"), bgm_path)
        finally:
            if os.path.exists(bgm_path):
                os.remove(bgm_path)

    def test_get_bgm_file_accepts_project_relative_song_path(self):
        """
        En la WebUI el usuario puede escribir directamente ./resource/songs/xxx.mp3. Aunque esa ruta sea
        relativa a la raíz del proyecto, el archivo real sigue dentro del directorio de lista blanca resource/songs,
        así que debe aceptarse, para que la música de fondo personalizada no se considere inexistente por error.
        """
        song_dir = utils.song_dir()
        bgm_path = os.path.join(song_dir, "test-relative-bgm.mp3")
        Path(bgm_path).write_bytes(b"fake-mp3")

        try:
            self.assertEqual(
                vd.get_bgm_file(bgm_file="./resource/songs/test-relative-bgm.mp3"),
                bgm_path,
            )
        finally:
            if os.path.exists(bgm_path):
                os.remove(bgm_path)

    def test_get_bgm_file_rejects_path_outside_song_directory(self):
        """
        El bgm_file que pasa el usuario no puede abrirse directamente como ruta local, o podría leer archivos del sistema.
        Aunque el archivo externo exista, debe rechazarse por no estar en el directorio songs.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_bgm:
            self.assertEqual(vd.get_bgm_file(bgm_file=temp_bgm.name), "")

    def test_get_ffmpeg_binary_uses_configured_env_path(self):
        """Cuando la configuración indica explícitamente ffmpeg, debe usarse esa ruta con prioridad."""
        with patch.dict(os.environ, {"IMAGEIO_FFMPEG_EXE": "/tmp/custom-ffmpeg"}, clear=True):
            self.assertEqual(utils.get_ffmpeg_binary(), "/tmp/custom-ffmpeg")

    def test_get_ffmpeg_binary_falls_back_to_imageio_ffmpeg(self):
        """
        Windows En el paquete portable puede que el PATH del sistema no tenga ffmpeg, pero la dependencia de moviepy
        imageio-ffmpeg normalmente proporciona el ejecutable. Aquí se verifica que esa ruta de respaldo funciona.
        """
        fake_imageio_ffmpeg = types.SimpleNamespace(
            get_ffmpeg_exe=lambda: "/tmp/bundled-ffmpeg"
        )

        with patch.dict(os.environ, {}, clear=True), patch.object(
            utils.shutil, "which", return_value=None
        ), patch.dict(sys.modules, {"imageio_ffmpeg": fake_imageio_ffmpeg}):
            self.assertEqual(utils.get_ffmpeg_binary(), "/tmp/bundled-ffmpeg")

    def test_get_effective_video_codec_falls_back_when_encoder_missing(self):
        """
        El codificador por hardware elegido por el usuario debe comprobarse primero en la lista de encoders de FFmpeg. Si no se detecta
        se vuelve directamente a libx264, para que la tarea de generación no falle en la fase de escritura del archivo.
        """
        config.app["video_codec"] = "h264_nvenc"

        with patch.object(vd, "_ffmpeg_encoder_exists", return_value=False):
            self.assertEqual(vd._get_effective_video_codec(), "libx264")

    def test_get_configured_video_codec_uses_stable_default_when_unset(self):
        """
        WebUI el modo «por defecto» no persiste video_codec. El backend debe, si falta la configuración, seguir
        devolviendo explícitamente libx264, sin pasar un valor vacío a MoviePy o FFmpeg para que decidan por su cuenta.
        """
        config.app.pop("video_codec", None)

        self.assertEqual(vd._get_configured_video_codec(), "libx264")

    def test_get_configured_video_codec_preserves_explicit_libx264(self):
        """
        Cuando el usuario elige explícitamente libx264 debe mantenerse fija esa elección. Actualmente da el mismo resultado que «seguir la estrategia por defecto del proyecto»,
        pero la semántica de configuración es distinta, y ajustar el valor por defecto en el futuro no puede afectar a la elección explícita.
        """
        config.app["video_codec"] = "libx264"

        self.assertEqual(vd._get_configured_video_codec(), "libx264")

    def test_ffmpeg_encoder_exists_falls_back_when_probe_fails(self):
        """
        Windows el ffmpeg configurado por el usuario puede no ejecutarse con normalidad por una ruta dañada, permisos o bloqueo del antivirus.
        Si falla la detección de encoders debe devolver False, para que la capa superior vuelva de forma estable a libx264.
        """
        with patch.object(
            vd.subprocess,
            "run",
            side_effect=OSError("permission denied"),
        ):
            self.assertFalse(vd._ffmpeg_encoder_exists("C:/ffmpeg/bin/ffmpeg.exe", "h264_nvenc"))

    def test_write_videofile_falls_back_after_runtime_encoder_failure(self):
        """
        FFmpeg Declarar soporte de un codificador por hardware no significa que la tarjeta gráfica o el driver actuales lo puedan usar.
        Tras fallar la primera codificación real, debe reintentarse de inmediato con libx264 y desactivar ese codificador en este proceso.
        """

        class _FakeClip:
            def __init__(self):
                self.codecs = []

            def write_videofile(self, output_file, codec, **kwargs):
                self.codecs.append(codec)
                if codec == "h264_nvenc":
                    raise RuntimeError("nvenc device not available")

        fake_clip = _FakeClip()

        with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True):
            used_codec = vd._write_videofile_with_codec_fallback(
                fake_clip,
                "/tmp/fake.mp4",
                codec="h264_nvenc",
                logger=None,
                fps=30,
            )

        self.assertEqual(used_codec, "libx264")
        self.assertEqual(fake_clip.codecs, ["h264_nvenc", "libx264"])
        self.assertIn("h264_nvenc", vd._runtime_disabled_video_codecs)

    def test_write_videofile_does_not_disable_codec_when_fallback_also_fails(self):
        """
        Si el respaldo libx264 también falla, la causa probablemente sea un problema general de ruta de salida, permisos, archivo en uso, etc.,
        y no puede concluirse por error que el codificador por hardware no está disponible.
        """

        class _FakeClip:
            def write_videofile(self, output_file, codec, **kwargs):
                raise RuntimeError(f"{codec} cannot write output")

        with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True):
            with self.assertRaises(RuntimeError):
                vd._write_videofile_with_codec_fallback(
                    _FakeClip(),
                    "/tmp/fake.mp4",
                    codec="h264_nvenc",
                    logger=None,
                    fps=30,
                )

        self.assertNotIn("h264_nvenc", vd._runtime_disabled_video_codecs)

    def test_failed_final_encode_keeps_previous_video_and_removes_partial_file(self):
        """A failed encode must not replace a downloadable final video with partial bytes."""

        class FailingClip:
            def write_videofile(self, output_file, codec, **_kwargs):
                Path(output_file).write_bytes(b"partial mp4")
                raise RuntimeError("encoder stopped")

        with tempfile.TemporaryDirectory() as temp_dir:
            final_path = Path(temp_dir, "final-1.mp4")
            final_path.write_bytes(b"previous complete mp4")

            with self.assertRaisesRegex(RuntimeError, "encoder stopped"):
                vd._write_videofile_with_codec_fallback(
                    FailingClip(),
                    str(final_path),
                    codec="libx264",
                    atomic_output=True,
                )

            self.assertEqual(final_path.read_bytes(), b"previous complete mp4")
            self.assertEqual(list(Path(temp_dir).iterdir()), [final_path])

    def test_final_encode_publishes_only_after_writer_returns(self):
        """Readers keep the old final video until the new encode completes."""
        test = self

        class SuccessfulClip:
            def write_videofile(self, output_file, codec, **_kwargs):
                test.assertNotEqual(Path(output_file), final_path)
                test.assertEqual(final_path.read_bytes(), b"previous complete mp4")
                Path(output_file).write_bytes(b"new complete mp4")

        with tempfile.TemporaryDirectory() as temp_dir:
            final_path = Path(temp_dir, "final-1.mp4")
            final_path.write_bytes(b"previous complete mp4")
            vd._write_videofile_with_codec_fallback(
                SuccessfulClip(),
                str(final_path),
                codec="libx264",
                atomic_output=True,
            )

            self.assertEqual(final_path.read_bytes(), b"new complete mp4")
            self.assertEqual(list(Path(temp_dir).iterdir()), [final_path])

    def test_format_ffmpeg_concat_path_normalizes_windows_path(self):
        """
        concat demuxer la lista de archivos es sensible a las barras invertidas de Windows; antes de escribir la lista se convierten
        de forma unificada a barras normales, conservando el escape de comillas simples.
        """
        with patch.object(
            vd.os.path,
            "abspath",
            return_value=r"C:\Users\Test User's Videos\clip.mp4",
        ), patch.object(vd.os, "sep", "\\"):
            self.assertEqual(
                vd._format_ffmpeg_concat_path(
                    r"C:\Users\Test User's Videos\clip.mp4"
                ),
                "C:/Users/Test User'\\''s Videos/clip.mp4",
            )

    def test_concat_video_clips_falls_back_after_runtime_encoder_failure(self):
        """
        La fase final de ffmpeg concat también debe tener la misma capacidad de respaldo. Aquí se usa mock para simular
        h264_nvenc un fallo de codificación y confirmar que se vuelve a ejecutar automáticamente con libx264.
        """
        config.app["video_codec"] = "h264_nvenc"

        def fake_run(command, capture_output, text, check, **kwargs):
            codec_index = command.index("-c:v") + 1
            codec = command[codec_index]
            if codec == "h264_nvenc":
                return types.SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="nvenc device not available",
                )
            Path(command[-1]).write_bytes(b"encoded-video")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")

            with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True):
                with patch.object(vd.subprocess, "run", side_effect=fake_run) as run:
                    vd.concat_video_clips_with_ffmpeg(
                        clip_files=[clip_file],
                        output_file=output_file,
                        threads=1,
                        output_dir=temp_dir,
                    )

        used_codecs = [
            call.args[0][call.args[0].index("-c:v") + 1]
            for call in run.call_args_list
        ]
        self.assertEqual(used_codecs, ["h264_nvenc", "libx264"])
        self.assertIn("h264_nvenc", vd._runtime_disabled_video_codecs)

    def test_concat_video_clips_does_not_disable_codec_when_fallback_also_fails(self):
        """
        concat en esa fase, si libx264 también falla, probablemente sea un problema de la lista de entrada, la ruta o los permisos de salida,
        y no se puede añadir el codificador por hardware a la lista de desactivados en tiempo de ejecución.
        """
        config.app["video_codec"] = "h264_nvenc"

        def fake_run(command, capture_output, text, check, **kwargs):
            codec_index = command.index("-c:v") + 1
            codec = command[codec_index]
            return types.SimpleNamespace(
                returncode=1,
                stdout="",
                stderr=f"{codec} cannot write output",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")

            with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True):
                with patch.object(vd.subprocess, "run", side_effect=fake_run):
                    with self.assertRaises(RuntimeError):
                        vd.concat_video_clips_with_ffmpeg(
                            clip_files=[clip_file],
                            output_file=output_file,
                            threads=1,
                            output_dir=temp_dir,
                        )

        self.assertNotIn("h264_nvenc", vd._runtime_disabled_video_codecs)

    def test_open_video_clip_quietly_suppresses_moviepy_stdout(self):
        """
        MoviePy 2.1.x el FFMPEG_VideoReader imprime directamente en stdout metadata
        y el comando ffmpeg. La capa de servicio del proyecto debe silenciar este ruido de las dependencias, para que el usuario no tome
        `audio_found: False` por error que el video final no tiene audio.
        """
        # La prueba solo comprueba si la capa de servicio silencia el ruido de lectura de MoviePy, y no debería guardar a largo plazo un fixture MP4 binario PNG
        # codificado con. Generar un video corto en tiempo de ejecución mantiene la prueba independiente y también
        # evita que el fixture, por parpadeos entre fotogramas debidos a otros parámetros de codificación, se use por error para verificar efectos visuales.
        image_path = os.path.join(resources_dir, "1.png")
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "image-fixture.mp4")
            source_clip = ImageClip(image_path).with_duration(0.2)
            try:
                source_clip.write_videofile(
                    video_path,
                    codec="libx264",
                    fps=5,
                    audio=False,
                    logger=None,
                )
            finally:
                source_clip.close()

            stdout = StringIO()
            with redirect_stdout(stdout):
                clip = vd._open_video_clip_quietly(video_path)

            try:
                self.assertEqual(stdout.getvalue(), "")
                self.assertIsNone(clip.audio)
                self.assertGreater(clip.duration, 0)
            finally:
                vd.close_clip(clip)

    def test_combine_videos_closes_audio_clip_when_duration_read_fails(self):
        """
        `combine_videos()` solo necesita leer la duración del audio de narración. Aunque al leer duration
        se produzca una excepción, debe cerrarse AudioFileClip, para evitar fugas de descriptores de archivo.
        """

        class _FakeAudioReader:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        class _BrokenAudioClip:
            def __init__(self):
                self.reader = _FakeAudioReader()

            @property
            def duration(self):
                raise RuntimeError("failed to read duration")

        fake_audio_clip = _BrokenAudioClip()

        with patch.object(vd, "AudioFileClip", return_value=fake_audio_clip):
            with self.assertRaises(RuntimeError):
                vd.combine_videos(
                    combined_video_path="/tmp/unused-combined.mp4",
                    video_paths=[],
                    audio_file="/tmp/unused-audio.mp3",
                )

        self.assertTrue(fake_audio_clip.reader.closed)

    def test_combine_videos_handles_none_transition_mode(self):
        """
        Ensure `combine_videos` safely handles
        `video_transition_mode=None`.
        """
        class _FakeAudioClip:
            @property
            def duration(self):
                return 10.0

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")
            audio_file = os.path.join(temp_dir, "audio.mp3")

            with patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()):
                # Use empty video_paths to avoid heavy video processing while
                # still exercising transition mode normalization logic.
                result = vd.combine_videos(
                    combined_video_path=combined_video_path,
                    video_paths=[],
                    audio_file=audio_file,
                    video_transition_mode=None,
                )
                self.assertEqual(result, combined_video_path)

    def _capture_source_ranges_for_clip_speed(
        self,
        *,
        source_duration,
        audio_duration,
        clip_speed,
        max_clip_duration=3,
    ):
        """Usa un video falso ligero para registrar los rangos de tiempo de origen que realmente lee combine_videos."""

        source_ranges = []
        written_durations = []

        class _FakeAudioClip:
            duration = audio_duration

            def close(self):
                pass

        class _FakeVideoClip:
            def __init__(self, duration, records_source_range=False):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920
                self.records_source_range = records_source_range

            def subclipped(self, start_time, end_time):
                # Solo registra los rangos leídos directamente del archivo origen. El recorte de seguridad tras cambiar la velocidad también llama a
                # subclipped, pero no representa un nuevo tramo de origen y no puede mezclarse en la detección de saltos.
                if self.records_source_range:
                    source_ranges.append((start_time, end_time))
                return _FakeVideoClip(end_time - start_time)

            def with_speed_scaled(self, factor):
                return _FakeVideoClip(self.duration / factor)

            def close(self):
                pass

        def _open_fake_video_clip(_video_path):
            return _FakeVideoClip(source_duration, records_source_range=True)

        def _capture_written_clip(clip, *_args, **_kwargs):
            written_durations.append(clip.duration)

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd,
                    "_open_video_clip_quietly",
                    side_effect=_open_fake_video_clip,
                ),
                patch.object(
                    vd,
                    "_write_videofile_with_codec_fallback",
                    side_effect=_capture_written_clip,
                ),
                # random el modo por defecto baraja los cortes de un mismo video origen. Aquí se mantiene el orden de generación,
                # para poder verificar con precisión si los tramos de origen adyacentes son continuos.
                patch.object(
                    vd,
                    "_prioritize_unique_source_clips",
                    side_effect=lambda subclipped_items, concat_mode: subclipped_items,
                ),
                patch.object(vd, "concat_video_clips_with_ffmpeg"),
                patch.object(vd, "delete_files"),
            ):
                vd.combine_videos(
                    combined_video_path=combined_video_path,
                    video_paths=["clip.mp4"],
                    audio_file="audio.mp3",
                    video_concat_mode=vd.VideoConcatMode.random,
                    max_clip_duration=max_clip_duration,
                    clip_speed=clip_speed,
                )

        return source_ranges, written_durations

    def test_combine_videos_slow_speed_keeps_source_timeline_continuous(self):
        """0.5 la cámara lenta x debe leer de forma continua 1,5 segundos de origen, sin saltarse imágenes intermedias."""

        source_ranges, written_durations = self._capture_source_ranges_for_clip_speed(
            source_duration=4.0,
            audio_duration=5.9,
            clip_speed=0.5,
        )

        self.assertEqual(source_ranges, [(0, 1.5), (1.5, 3.0)])
        self.assertEqual(written_durations, [3.0, 3.0])

    def test_combine_videos_fast_speed_reads_enough_source_content(self):
        """2 la cámara rápida x debe leer 6 segundos de imagen de origen, para que el fragmento final siga durando 3 segundos."""

        source_ranges, written_durations = self._capture_source_ranges_for_clip_speed(
            source_duration=8.0,
            audio_duration=2.9,
            clip_speed=2.0,
        )

        self.assertEqual(source_ranges, [(0, 6.0)])
        self.assertEqual(written_durations, [3.0])

    def test_combine_videos_keeps_small_duration_safety_margin(self):
        """
        Cuando la duración del audio y la acumulada del material son exactamente iguales, debe añadirse igualmente un fragmento corto como margen de seguridad.

        FFmpeg la concatenación según la tasa de fotogramas puede dejar el video final unas decenas de milisegundos más corto que lo teórico. Si aquí
        se parara en cuanto 10.0s == 10.0s, al final del video podría darse el caso límite de que el audio siga sonando pero
        el material de video ya haya terminado.
        """

        class _FakeAudioClip:
            duration = 10.0

            def close(self):
                pass

        class _FakeVideoClip:
            def __init__(self, duration):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920

            def subclipped(self, start_time, end_time):
                return _FakeVideoClip(end_time - start_time)

        video_durations = {
            "clip-1.mp4": 3.0,
            "clip-2.mp4": 4.0,
            "clip-3.mp4": 3.0,
            "clip-4.mp4": 2.0,
        }

        def _open_fake_video_clip(video_path):
            return _FakeVideoClip(video_durations[video_path])

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")

            with patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()):
                with patch.object(
                    vd, "_open_video_clip_quietly", side_effect=_open_fake_video_clip
                ):
                    with patch.object(
                        vd, "_write_videofile_with_codec_fallback"
                    ) as write_mock:
                        with patch.object(vd, "concat_video_clips_with_ffmpeg") as concat_mock:
                            with patch.object(vd, "delete_files"):
                                result = vd.combine_videos(
                                    combined_video_path=combined_video_path,
                                    video_paths=list(video_durations.keys()),
                                    audio_file=os.path.join(temp_dir, "audio.mp3"),
                                    video_aspect=vd.VideoAspect.portrait,
                                    video_concat_mode=vd.VideoConcatMode.sequential,
                                    video_transition_mode=None,
                                    max_clip_duration=10,
                                )

        self.assertEqual(result, combined_video_path)
        self.assertEqual(write_mock.call_count, 4)
        self.assertEqual(concat_mock.call_args.kwargs["max_duration"], 10.0)

    def test_combine_videos_cleans_temp_clips_when_concat_fails(self):
        """A failed final merge must not strand encoded clips on disk."""

        class FakeAudioClip:
            duration = 1.0

        class FakeVideoClip:
            duration = 2.0
            size = (1080, 1920)
            w = 1080
            h = 1920

            def subclipped(self, _start, _end):
                return self

        def write_clip(_clip, output_file, **_kwargs):
            Path(output_file).write_bytes(b"encoded clip")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "combined.mp4")
            with (
                patch.object(vd, "AudioFileClip", return_value=FakeAudioClip()),
                patch.object(vd, "_open_video_clip_quietly", return_value=FakeVideoClip()),
                patch.object(
                    vd, "_write_videofile_with_codec_fallback", side_effect=write_clip
                ),
                patch.object(
                    vd,
                    "concat_video_clips_with_ffmpeg",
                    side_effect=RuntimeError("concat failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "concat failed"):
                    vd.combine_videos(
                        combined_video_path=output_file,
                        video_paths=["clip.mp4"],
                        audio_file="audio.mp3",
                        video_concat_mode=vd.VideoConcatMode.sequential,
                    )

            self.assertFalse(list(Path(temp_dir).glob("temp-clip-*.mp4")))

    def test_combine_videos_cleans_failed_encoded_clip_and_reader(self):
        """A bad source must not strand a partial MP4 or an FFmpeg reader."""

        class FakeAudioClip:
            duration = 0.5

            def close(self):
                pass

        class FakeVideoClip:
            duration = 1.0
            size = (1080, 1920)
            w = 1080
            h = 1920

            def __init__(self, source):
                self.source = source
                self.close_calls = 0
                self.reader = self

            def subclipped(self, _start, _end):
                derived = FakeVideoClip(self.source)
                derived_clips.append(derived)
                return derived

            def close(self):
                self.close_calls += 1

        derived_clips = []

        def open_clip(source):
            return FakeVideoClip(source)

        def write_clip(clip, output_file, **_kwargs):
            Path(output_file).write_bytes(b"partial")
            if clip.source == "bad.mp4":
                raise RuntimeError("encode failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(vd, "AudioFileClip", return_value=FakeAudioClip()),
                patch.object(vd, "_open_video_clip_quietly", side_effect=open_clip),
                patch.object(
                    vd, "_write_videofile_with_codec_fallback", side_effect=write_clip
                ),
                patch.object(vd, "concat_video_clips_with_ffmpeg") as concat,
            ):
                vd.combine_videos(
                    combined_video_path=os.path.join(temp_dir, "combined.mp4"),
                    video_paths=["bad.mp4", "good.mp4"],
                    audio_file="audio.mp3",
                    video_concat_mode=vd.VideoConcatMode.sequential,
                )

            concat.assert_called_once()
            self.assertEqual(derived_clips[0].close_calls, 1)
            self.assertFalse(list(Path(temp_dir).glob("temp-clip-*.mp4")))

    def test_combine_videos_skips_unreadable_source_when_good_clip_remains(self):
        """A stale corrupt cache clip must not discard healthy downloaded footage."""
        class FakeAudioClip:
            duration = 0.5

        class FakeVideoClip:
            duration = 1.0
            size = (1080, 1920)
            w = 1080
            h = 1920

            def subclipped(self, _start, _end):
                return self

        def open_clip(source):
            if source == "corrupt.mp4":
                raise OSError("FFmpeg could not read video metadata")
            return FakeVideoClip()

        used_sources = []
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(vd, "AudioFileClip", return_value=FakeAudioClip()),
                patch.object(vd, "_open_video_clip_quietly", side_effect=open_clip),
                patch.object(vd, "_write_videofile_with_codec_fallback"),
                patch.object(vd, "concat_video_clips_with_ffmpeg") as concat,
                patch.object(vd, "delete_files"),
            ):
                vd.combine_videos(
                    combined_video_path=os.path.join(temp_dir, "combined.mp4"),
                    video_paths=["corrupt.mp4", "healthy.mp4"],
                    audio_file="audio.mp3",
                    video_concat_mode=vd.VideoConcatMode.sequential,
                    used_video_paths=used_sources,
                )

        concat.assert_called_once()
        self.assertEqual(used_sources, ["healthy.mp4"])

    def test_combine_videos_reports_failure_if_every_source_is_unreadable(self):
        """Never return an output path for an input set that yielded no clips."""
        class FakeAudioClip:
            duration = 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(vd, "AudioFileClip", return_value=FakeAudioClip()),
                patch.object(
                    vd,
                    "_open_video_clip_quietly",
                    side_effect=OSError("invalid cached video"),
                ),
                patch.object(vd, "concat_video_clips_with_ffmpeg") as concat,
            ):
                with self.assertRaisesRegex(RuntimeError, "no readable video clips"):
                    vd.combine_videos(
                        combined_video_path=os.path.join(temp_dir, "combined.mp4"),
                        video_paths=["corrupt.mp4"],
                        audio_file="audio.mp3",
                    )

        concat.assert_not_called()

    def test_concat_video_clips_limits_output_to_audio_duration(self):
        """En la concatenación final debe recortarse a la duración del audio, para evitar una cola de silencio evidente por el margen de seguridad."""

        def fake_run(command, capture_output, text, check, **kwargs):
            Path(command[-1]).write_bytes(b"encoded-video")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")

            with patch.object(vd.subprocess, "run", side_effect=fake_run) as run:
                vd.concat_video_clips_with_ffmpeg(
                    clip_files=[clip_file],
                    output_file=output_file,
                    threads=1,
                    output_dir=temp_dir,
                    max_duration=10.0,
                )

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-t") + 1], "10.000")
        self.assertLess(command.index("-t"), len(command) - 1)
        self.assertNotEqual(command[-1], output_file)

    def test_concat_video_clips_logs_heartbeat_while_ffmpeg_runs(self):
        """
        durante la concatenación, subprocess.run se bloquea hasta que ffmpeg termina, y mientras tanto el proyecto no produce ningún log, así que el usuario
        no puede distinguir si sigue codificando o se ha colgado (issue #1342). Durante la espera deben registrarse señales de vida.
        """

        def slow_run(command, capture_output, text, check, **kwargs):
            # Simula una concatenación lenta: en esta ventana el hilo de latido debe registrar al menos un log de actividad.
            time.sleep(0.2)
            Path(command[-1]).write_bytes(b"encoded-video")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")
            Path(output_file).write_bytes(b"x" * 2048)

            with patch.object(vd, "_FFMPEG_CONCAT_HEARTBEAT_SECONDS", 0.02):
                with patch.object(vd.subprocess, "run", side_effect=slow_run):
                    with patch.object(vd.logger, "info") as info_mock:
                        vd.concat_video_clips_with_ffmpeg(
                            clip_files=[clip_file],
                            output_file=output_file,
                            threads=1,
                            output_dir=temp_dir,
                        )

        heartbeats = [
            str(call.args[0])
            for call in info_mock.call_args_list
            if "still running" in str(call.args[0])
        ]
        self.assertTrue(heartbeats, "durante una concatenación lenta deben registrarse logs de actividad")
        self.assertRegex(heartbeats[0], r"elapsed=\d+s, output size: 0\.00 MB")

    def test_concat_heartbeat_belongs_to_the_task_log_scope(self):
        """
        El latido lo escribe un hilo independiente. La WebUI solo recoge los logs dentro del ámbito del hilo de la tarea; si el hilo de latido no
        se vincula al ámbito, el terminal ve la información de actividad pero el panel de la WebUI sigue en blanco.
        """

        def slow_run(command, capture_output, text, check, **kwargs):
            time.sleep(0.2)
            Path(command[-1]).write_bytes(b"encoded-video")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")

            with (
                patch.object(vd, "_FFMPEG_CONCAT_HEARTBEAT_SECONDS", 0.02),
                patch.object(vd.subprocess, "run", side_effect=slow_run),
                _capture_task_scoped_logs() as messages,
            ):
                vd.concat_video_clips_with_ffmpeg(
                    clip_files=[clip_file],
                    output_file=output_file,
                    threads=1,
                    output_dir=temp_dir,
                )

        self.assertTrue(
            [message for message in messages if "still running" in message],
            "los logs de latido deben pertenecer al hilo de la tarea que inició la concatenación",
        )

    def test_clip_processing_logs_belong_to_the_task_log_scope(self):
        """
        Los fragmentos siempre se procesan en el pool de hilos clip-process, incluso con concurrencia 1. Los logs por fragmento son la única
        información de progreso de esta fase y deben pertenecer al hilo de la tarea para que la WebUI muestre «procesando el fragmento N».
        """

        class _FakeAudioClip:
            duration = 4.0

            def close(self):
                pass

        class _FakeVideoClip:
            def __init__(self, duration):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920

            def subclipped(self, start_time, end_time):
                return _FakeVideoClip(end_time - start_time)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd,
                    "_open_video_clip_quietly",
                    side_effect=lambda _path: _FakeVideoClip(10.0),
                ),
                patch.object(vd, "_write_videofile_with_codec_fallback"),
                patch.object(vd, "concat_video_clips_with_ffmpeg"),
                patch.object(vd, "delete_files"),
                _capture_task_scoped_logs() as messages,
            ):
                vd.combine_videos(
                    combined_video_path=os.path.join(temp_dir, "combined.mp4"),
                    video_paths=["clip.mp4"],
                    audio_file="audio.mp3",
                    video_concat_mode=vd.VideoConcatMode.sequential,
                    max_clip_duration=2,
                )

        self.assertTrue(
            [message for message in messages if message.startswith("processing clip")],
            "los logs de procesado por fragmento deben pertenecer al hilo de la tarea que inició la composición",
        )

    def test_combine_videos_reports_covered_duration_as_progress(self):
        """
        El procesado de fragmentos es la parte más lenta de la composición (unos 20 segundos por fragmento con material 4K); antes, en toda la fase
        el progreso estaba fijo en el 50%. Cada fragmento procesado debe informar de la proporción de duración del video final cubierta, y escribir un log
        con la duración cubierta; al cubrirlo todo, la proporción se limita a 1.0.
        """

        class _FakeAudioClip:
            duration = 4.0

            def close(self):
                pass

        class _FakeVideoClip:
            def __init__(self, duration):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920

            def subclipped(self, start_time, end_time):
                return _FakeVideoClip(end_time - start_time)

            def close(self):
                pass

        fractions = []
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd,
                    "_open_video_clip_quietly",
                    side_effect=lambda _path: _FakeVideoClip(10.0),
                ),
                patch.object(vd, "_write_videofile_with_codec_fallback"),
                patch.object(vd, "concat_video_clips_with_ffmpeg"),
                patch.object(vd, "delete_files"),
                patch.object(vd.logger, "info") as info,
            ):
                vd.combine_videos(
                    combined_video_path=os.path.join(temp_dir, "combined.mp4"),
                    # En modo secuencial se toma un solo fragmento por archivo origen: tres archivos origen dan tres fragmentos.
                    video_paths=["a.mp4", "b.mp4", "c.mp4"],
                    audio_file="audio.mp3",
                    video_concat_mode=vd.VideoConcatMode.sequential,
                    max_clip_duration=2,
                    progress_callback=fractions.append,
                )

        # Voz de 4.0 segundos más 0.1 segundos de margen de seguridad, 2 segundos por fragmento: hacen falta 3 fragmentos para cubrir 4.1 segundos.
        self.assertEqual(len(fractions), 3)
        self.assertEqual(fractions, sorted(fractions))
        self.assertAlmostEqual(fractions[0], 2.0 / 4.1, places=3)
        self.assertEqual(fractions[-1], 1.0)
        processed = [
            str(call.args[0])
            for call in info.call_args_list
            if str(call.args[0]).startswith("processed clip")
        ]
        self.assertEqual(
            processed,
            [
                "processed clip 1: 2.0 of 4.1s covered",
                "processed clip 2: 4.0 of 4.1s covered",
                "processed clip 3: 6.0 of 4.1s covered",
            ],
        )

    def test_failing_clip_progress_callback_does_not_break_combine(self):
        """El progreso es solo información de visualización; un error del callback no puede invalidar fragmentos ya procesados."""

        class _FakeAudioClip:
            duration = 1.0

            def close(self):
                pass

        class _FakeVideoClip:
            def __init__(self, duration):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920

            def subclipped(self, start_time, end_time):
                return _FakeVideoClip(end_time - start_time)

            def close(self):
                pass

        def broken_callback(_fraction):
            raise RuntimeError("state backend unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            combined_video_path = os.path.join(temp_dir, "combined.mp4")
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(
                    vd,
                    "_open_video_clip_quietly",
                    side_effect=lambda _path: _FakeVideoClip(10.0),
                ),
                patch.object(vd, "_write_videofile_with_codec_fallback"),
                patch.object(vd, "concat_video_clips_with_ffmpeg") as concat,
                patch.object(vd, "delete_files"),
                patch.object(vd.logger, "warning") as warning,
            ):
                result = vd.combine_videos(
                    combined_video_path=combined_video_path,
                    video_paths=["clip.mp4"],
                    audio_file="audio.mp3",
                    video_concat_mode=vd.VideoConcatMode.sequential,
                    max_clip_duration=2,
                    progress_callback=broken_callback,
                )

        self.assertEqual(result, combined_video_path)
        concat.assert_called_once()
        self.assertTrue(
            [
                call
                for call in warning.call_args_list
                if "progress" in str(call.args[0])
            ]
        )

    def test_stage_heartbeat_logs_while_running_and_stops_afterwards(self):
        """
        Durante la codificación del video final MoviePy no produce ningún log. El latido debe escribirse a intervalos mientras corre la fase,
        pertenecer al hilo de la tarea y detenerse al terminar la fase, sin dejar hilos que sigan escribiendo logs.
        """
        with (
            patch.object(vd, "_STAGE_HEARTBEAT_SECONDS", 0.02),
            _capture_task_scoped_logs() as messages,
        ):
            with vd._stage_heartbeat("final video render"):
                time.sleep(0.2)
            heartbeats_at_exit = len(
                [m for m in messages if "still running" in m]
            )
            time.sleep(0.1)
            heartbeats_later = len([m for m in messages if "still running" in m])

        self.assertGreater(heartbeats_at_exit, 0)
        self.assertEqual(heartbeats_later, heartbeats_at_exit)
        self.assertRegex(
            next(m for m in messages if "still running" in m),
            r"^final video render still running: elapsed=\d+s$",
        )

    def test_stage_heartbeat_stops_when_the_stage_fails(self):
        """Si la fase lanza una excepción el hilo de latido también debe detenerse, y la excepción se propaga tal cual."""
        with patch.object(vd, "_STAGE_HEARTBEAT_SECONDS", 0.02):
            with _capture_task_scoped_logs() as messages:
                with self.assertRaisesRegex(RuntimeError, "encode failed"):
                    with vd._stage_heartbeat("final video render"):
                        raise RuntimeError("encode failed")
                time.sleep(0.1)

        self.assertEqual([m for m in messages if "still running" in m], [])

    def test_generate_video_reports_heartbeat_during_final_render(self):
        """la codificación final tarda varios minutos, y durante ese tiempo debe haber logs de actividad."""
        params = vd.VideoParams(
            video_subject="test", subtitle_enabled=False, bgm_type=""
        )

        def slow_write(*_args, **_kwargs):
            time.sleep(0.2)

        with (
            patch.object(vd, "_STAGE_HEARTBEAT_SECONDS", 0.02),
            patch.object(
                vd, "_open_video_clip_quietly", return_value=_FakeMoviePyClip()
            ),
            patch.object(vd, "AudioFileClip", return_value=_FakeMoviePyClip()),
            patch.object(
                vd, "_write_videofile_with_codec_fallback", side_effect=slow_write
            ),
            patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
            _capture_task_scoped_logs() as messages,
        ):
            vd.generate_video(
                video_path="combined.mp4",
                audio_path="voice.mp3",
                subtitle_path="",
                output_file="final.mp4",
                params=params,
            )

        self.assertTrue(
            [m for m in messages if m.startswith("final video render still running")]
        )

    def test_concat_timeout_fails_without_retrying_another_codec(self):
        """A stalled FFmpeg must fail the task and release the concat list file."""
        config.app["ffmpeg_concat_timeout_seconds"] = 12
        config.app["video_codec"] = "h264_nvenc"

        def timed_out_run(command, **kwargs):
            self.assertEqual(kwargs["timeout"], 12)
            raise subprocess.TimeoutExpired(
                command, kwargs["timeout"], stderr=b"stalled"
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            clip_file = os.path.join(temp_dir, "clip.mp4")
            output_file = os.path.join(temp_dir, "combined.mp4")
            Path(clip_file).write_bytes(b"fake")

            with patch.object(vd, "_ffmpeg_encoder_exists", return_value=True):
                with patch.object(vd.subprocess, "run", side_effect=timed_out_run) as run:
                    with self.assertRaisesRegex(TimeoutError, "12 seconds"):
                        vd.concat_video_clips_with_ffmpeg(
                            clip_files=[clip_file],
                            output_file=output_file,
                            threads=1,
                            output_dir=temp_dir,
                        )

            self.assertEqual(run.call_count, 1)
            self.assertFalse(Path(temp_dir, "ffmpeg-concat-list.txt").exists())

    def test_concat_video_clips_heartbeat_tolerates_missing_output_file(self):
        """
        Al empezar la concatenación el archivo de salida aún no existe, así que la descripción del latido debe degradarse con seguridad; si la excepción al detectar el tamaño del archivo
        llegara a la llamada de concatenación, una tarea que podía terminar bien acabaría fallando.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIn(
                "not available",
                vd._describe_concat_output_progress(
                    os.path.join(temp_dir, "absent.mp4")
                ),
            )
            existing = os.path.join(temp_dir, "present.mp4")
            Path(existing).write_bytes(b"x" * 2048)
            self.assertIn(
                "output size: 0.00 MB", vd._describe_concat_output_progress(existing)
            )

    def test_prioritize_unique_source_clips_uses_each_source_before_reuse(self):
        """
        En modo aleatorio, un material largo se divide en varios fragmentos. La capa de planificación debe hacer que cada material origen
        aparezca al menos una vez antes de usar otros cortes del mismo material, reduciendo la repetición que percibe el usuario.
        """
        clips = [
            vd.SubClippedVideoClip("a.mp4", 0, 4, source_file_path="a.mp4"),
            vd.SubClippedVideoClip("a.mp4", 4, 8, source_file_path="a.mp4"),
            vd.SubClippedVideoClip("b.mp4", 0, 4, source_file_path="b.mp4"),
            vd.SubClippedVideoClip("b.mp4", 4, 8, source_file_path="b.mp4"),
            vd.SubClippedVideoClip("c.mp4", 0, 4, source_file_path="c.mp4"),
        ]

        ordered_clips = vd._prioritize_unique_source_clips(
            subclipped_items=clips,
            concat_mode=vd.VideoConcatMode.random,
        )

        self.assertCountEqual(ordered_clips, clips)
        first_round_sources = [clip.source_file_path for clip in ordered_clips[:3]]
        self.assertCountEqual(first_round_sources, ["a.mp4", "b.mp4", "c.mp4"])

    def test_prioritize_unique_source_clips_keeps_sequential_order(self):
        """
        El modo secuencial solo toma el primer fragmento de cada material, y la lógica de planificación aleatoria no debe cambiar su orden.
        """
        clips = [
            vd.SubClippedVideoClip("a.mp4", 0, 4, source_file_path="a.mp4"),
            vd.SubClippedVideoClip("b.mp4", 0, 4, source_file_path="b.mp4"),
            vd.SubClippedVideoClip("c.mp4", 0, 4, source_file_path="c.mp4"),
        ]

        ordered_clips = vd._prioritize_unique_source_clips(
            subclipped_items=clips,
            concat_mode=vd.VideoConcatMode.sequential,
        )

        self.assertEqual(ordered_clips, clips)

    def test_prioritize_unique_source_clips_prefers_long_primary_clip(self):
        """
        El último corte de un mismo material origen puede ser más corto que la duración objetivo del fragmento. En la primera ronda de deduplicación deben preferirse
        los fragmentos más largos, o se reutilizaría material antes de tiempo por falta de duración acumulada.
        """
        short_tail = vd.SubClippedVideoClip(
            "a.mp4", 6, 6.5, source_file_path="a.mp4"
        )
        full_clip = vd.SubClippedVideoClip(
            "a.mp4", 0, 3, source_file_path="a.mp4"
        )
        other_source = vd.SubClippedVideoClip(
            "b.mp4", 0, 3, source_file_path="b.mp4"
        )

        ordered_clips = vd._prioritize_unique_source_clips(
            subclipped_items=[short_tail, full_clip, other_source],
            concat_mode=vd.VideoConcatMode.random,
        )

        first_a_clip = next(
            clip for clip in ordered_clips if clip.source_file_path == "a.mp4"
        )
        self.assertEqual(first_a_clip, full_clip)
    
    def test_wrap_text(self):
        """test text wrapping function"""
        try:
            font_path = os.path.join(utils.font_dir(), "STHeitiMedium.ttc")
            if not os.path.exists(font_path):
                self.fail(f"font file not found: {font_path}")
                
            # test english text wrapping
            test_text_en = "This is a test text for wrapping long sentences in english language"
            
            wrapped_text_en, text_height_en = vd.wrap_text(
                text=test_text_en,
                max_width=300,
                font=font_path,
                fontsize=30
            )
            # verify text is wrapped
            self.assertIn("\n", wrapped_text_en)
            
            # test chinese text wrapping
            test_text_zh = "这是一段用来测试中文长句换行的文本内容，应该会根据宽度限制进行换行处理"
            wrapped_text_zh, text_height_zh = vd.wrap_text(
                text=test_text_zh,
                max_width=300,
                font=font_path,
                fontsize=30
            )   
            # verify chinese text is wrapped
            self.assertIn("\n", wrapped_text_zh)
        except Exception as e:
            self.fail(f"test wrap_text failed: {str(e)}")

    def test_wrap_text_uses_stable_line_metrics_for_all_bundled_fonts(self):
        """
        La altura del subtítulo debe venir del ascent/descent de la propia fuente, y no depender del texto actual.

        El texto latino sin g/j/p/q/y solo tiene mayúsculas y x-height, y los glifos de Pillow
        bbox quedan mucho más bajos que la altura de línea real de la fuente; con varias líneas el error se acumula y acaba recortando la última línea.
        Aquí se recorren todas las fuentes incluidas, cubriendo texto en inglés con y sin trazos descendentes,
        para evitar que en el futuro se vuelva a introducir una implementación que «calcula la altura de línea según la tinta de los glifos actuales».
        """
        font_size = 60
        max_width = 360
        text_cases = {
            "without_descenders": "A man survived the Hiroshima atomic bomb blast",
            "with_descenders": "Typing quickly brings joyful progress",
        }
        font_paths = sorted(
            path
            for path in Path(utils.font_dir()).iterdir()
            if path.suffix.lower() in {".ttf", ".ttc"}
        )

        self.assertTrue(font_paths, "expected bundled subtitle fonts")
        for font_path in font_paths:
            font = vd.ImageFont.truetype(str(font_path), font_size)
            expected_line_height = sum(font.getmetrics())
            for case_name, text in text_cases.items():
                with self.subTest(font=font_path.name, case=case_name):
                    wrapped_text, text_height = vd.wrap_text(
                        text=text,
                        max_width=max_width,
                        font=str(font_path),
                        fontsize=font_size,
                    )
                    line_count = wrapped_text.count("\n") + 1

                    self.assertGreater(line_count, 1)
                    self.assertEqual(
                        text_height,
                        line_count * expected_line_height,
                    )

    def test_wrap_text_counts_existing_subtitle_line_breaks(self):
        """
        SRT El texto puede contener ya saltos de línea manuales; aunque ninguna línea necesite volver a partirse, la altura debe
        calcularse con las dos líneas finales. Si no, las frases cortas en pantallas anchas se saltarían la rama de salto automático y volverían a recortar la última línea.
        """
        font_size = 60
        font_path = os.path.join(utils.font_dir(), "MicrosoftYaHeiBold.ttc")
        text = "SAFE TEXT\nMORE SAFE"
        font = vd.ImageFont.truetype(font_path, font_size)

        wrapped_text, text_height = vd.wrap_text(
            text=text,
            max_width=972,
            font=font_path,
            fontsize=font_size,
        )

        self.assertEqual(wrapped_text, text)
        self.assertEqual(text_height, 2 * sum(font.getmetrics()))

    def test_small_subtitle_with_thick_stroke_keeps_a_bottom_margin(self):
        """
        Un tamaño de letra pequeño con contorno grueso es el límite de proporción que más fácilmente vuelve a tocar fondo. Se recorren todas las fuentes incluidas y se lee
        MoviePy la mask real, asegurando que la altura extra tenga cabida al menos para el contorno completo expandido arriba y abajo.
        """
        font_size = 24
        stroke_width = 6
        max_width = 240
        text = "A man survived the Hiroshima atomic bomb blast"
        font_paths = sorted(
            path
            for path in Path(utils.font_dir()).iterdir()
            if path.suffix.lower() in {".ttf", ".ttc"}
        )

        for font_path in font_paths:
            with self.subTest(font=font_path.name):
                wrapped_text, text_height = vd.wrap_text(
                    text=text,
                    max_width=max_width,
                    font=str(font_path),
                    fontsize=font_size,
                )
                line_count = wrapped_text.count("\n") + 1
                interline = int(font_size * 0.25)
                vertical_padding = int(font_size * 0.35)
                stroke_padding = stroke_width * 2 * line_count
                clip_height = int(
                    text_height
                    + vertical_padding
                    + interline * line_count
                    + stroke_padding
                )
                text_clip = vd.TextClip(
                    text=wrapped_text,
                    font=str(font_path),
                    font_size=font_size,
                    color="#FFFFFF",
                    stroke_color="#000000",
                    stroke_width=stroke_width,
                    interline=interline,
                    size=(max_width, clip_height),
                    text_align="center",
                )
                try:
                    mask = text_clip.mask.get_frame(0)
                    visible_rows, _ = vd.np.where(mask > 0.01)

                    self.assertGreater(len(visible_rows), 0)
                    self.assertLess(int(visible_rows.max()), clip_height - 1)
                finally:
                    text_clip.close()

    def test_multilingual_textclip_last_line_keeps_a_visible_bottom_margin(self):
        """
        Dibuja de verdad subtítulos multilingües con MoviePy, asegurando que la última línea no toque el borde inferior del lienzo.

        Comprobar solo el valor devuelto por wrap_text() pasaría por alto las diferencias combinadas de Pillow/MoviePy en baseline, contorno e
        interlineado, así que aquí se lee directamente la mask transparente de TextClip. Los textos cubiertos
        los admite por completo la fuente incluida correspondiente: inglés, vietnamita, tailandés, chino simplificado y tradicional, ruso
        y griego; si los píxeles visibles tocan la última línea, aún hay riesgo de recorte silencioso.
        """
        font_size = 60
        max_width = 360
        interline = int(font_size * 0.25)
        vertical_padding = int(font_size * 0.35)
        stroke_width = 2
        cases = (
            (
                "english_without_descenders",
                "BeVietnamPro-Bold.ttf",
                "A man survived the Hiroshima atomic bomb blast",
            ),
            (
                "vietnamese",
                "BeVietnamPro-Medium.ttf",
                "Tôi vẫn luôn tin vào một tương lai tươi sáng",
            ),
            (
                "thai",
                "Charm-Regular.ttf",
                "นี่คือข้อความสำหรับตรวจสอบบรรทัดสุดท้ายของคำบรรยาย",
            ),
            (
                "simplified_chinese",
                "MicrosoftYaHeiBold.ttc",
                "这是一个用于检查字幕最后一行是否完整显示的测试句子",
            ),
            (
                "traditional_chinese",
                "STHeitiMedium.ttc",
                "這是一個用於檢查字幕最後一行是否完整顯示的測試句子",
            ),
            (
                "cyrillic",
                "MicrosoftYaHeiNormal.ttc",
                "Это текст для проверки последней строки субтитров",
            ),
            (
                "greek",
                "STHeitiLight.ttc",
                "Αυτό είναι κείμενο για τον έλεγχο της τελευταίας γραμμής",
            ),
        )

        for language, font_name, text in cases:
            font_path = os.path.join(utils.font_dir(), font_name)
            with self.subTest(language=language, font=font_name):
                self.assertTrue(vd.subtitle_font_supports_text(font_path, text))
                wrapped_text, text_height = vd.wrap_text(
                    text=text,
                    max_width=max_width,
                    font=font_path,
                    fontsize=font_size,
                )
                line_count = wrapped_text.count("\n") + 1
                stroke_padding = stroke_width * 2 * line_count
                clip_height = int(
                    text_height
                    + vertical_padding
                    + interline * line_count
                    + stroke_padding
                )
                text_clip = vd.TextClip(
                    text=wrapped_text,
                    font=font_path,
                    font_size=font_size,
                    color="#FFFFFF",
                    stroke_color="#000000",
                    stroke_width=stroke_width,
                    interline=interline,
                    size=(max_width, clip_height),
                    text_align="center",
                )
                try:
                    mask = text_clip.mask.get_frame(0)
                    visible_rows, _ = vd.np.where(mask > 0.01)

                    self.assertGreater(line_count, 1)
                    self.assertGreater(len(visible_rows), 0)
                    self.assertLess(int(visible_rows.max()), clip_height - 1)
                finally:
                    text_clip.close()

    def test_rounded_subtitle_background_clip_has_transparent_corners(self):
        """
        El fondo de subtítulo redondeado solo se usa cuando el usuario lo activa explícitamente. Aquí se verifica directamente que el RGBA
        fondo generado tiene esquinas redondeadas transparentes y centro semitransparente, para que cambios futuros no degraden el redondeado a un rectángulo sólido.
        """
        clip = vd._rounded_subtitle_background_clip(
            width=120,
            height=48,
            color="#123456",
            alpha=140,
            radius=16,
        )
        try:
            frame = clip.get_frame(0)
            mask = clip.mask.get_frame(0)

            self.assertEqual(frame.shape[0:2], (48, 120))
            self.assertEqual(tuple(frame[24, 60]), (18, 52, 86))
            self.assertEqual(mask[0, 0], 0)
            self.assertGreater(mask[24, 60], 0.5)
            self.assertLess(mask[24, 60], 0.6)
        finally:
            clip.close()

    def test_get_temp_audio_dir_returns_system_temp_on_windows(self):
        with patch("sys.platform", "win32"):
            result = vd._get_temp_audio_dir("/some/output/dir")
            self.assertEqual(result, tempfile.gettempdir())

    def test_get_temp_audio_dir_returns_output_dir_on_non_windows(self):
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                with patch("sys.platform", platform):
                    result = vd._get_temp_audio_dir("/some/output/dir")
                    self.assertEqual(result, "/some/output/dir")


class TestMaterialResolutionTolerance(unittest.TestCase):
    def test_accepts_material_at_the_nominal_minimum(self):
        self.assertTrue(vd.is_material_resolution_acceptable(480, 480))

    def test_accepts_whatsapp_recompressed_portrait_clip(self):
        # WhatsApp delivers 9:16 clips as 478x850, two pixels under the
        # nominal 480 minimum. Rejecting them fails the whole task.
        self.assertTrue(vd.is_material_resolution_acceptable(478, 850))

    def test_accepts_material_exactly_at_the_tolerance_bound(self):
        bound = vd._MIN_MATERIAL_DIMENSION - vd._MIN_DIMENSION_TOLERANCE
        self.assertTrue(vd.is_material_resolution_acceptable(bound, bound))

    def test_rejects_material_just_below_the_tolerance_bound(self):
        bound = vd._MIN_MATERIAL_DIMENSION - vd._MIN_DIMENSION_TOLERANCE
        self.assertFalse(vd.is_material_resolution_acceptable(bound - 1, 850))
        self.assertFalse(vd.is_material_resolution_acceptable(850, bound - 1))

    def test_rejects_genuinely_low_resolution_material(self):
        self.assertFalse(vd.is_material_resolution_acceptable(320, 240))


if __name__ == "__main__":
    unittest.main()
