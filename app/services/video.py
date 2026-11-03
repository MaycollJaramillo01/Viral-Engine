import itertools
import hashlib
from concurrent.futures import ThreadPoolExecutor
import io
import math
import os
import random
import gc
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from contextlib import ExitStack, contextmanager, redirect_stdout
from functools import lru_cache
from typing import Callable, List
from loguru import logger
import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

from app.config import config
from app.models import const
from app.models.schema import (
    ImageMotion,
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import bgm as bgm_service
from app.services.utils import subtitle_shaping, video_effects
from app.utils import file_security, logging_utils, utils
from app.utils.subtitle_reader import moviepy_subtitles, split_long_cues

class SubClippedVideoClip:
    def __init__(
        self,
        file_path,
        start_time=None,
        end_time=None,
        width=None,
        height=None,
        duration=None,
        source_file_path=None,
    ):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        self.source_file_path = source_file_path or file_path
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
# Docker la combinación ffmpeg/AAC con la configuración por defecto tiende más a fluctuaciones de calidad de audio,
# así que aquí se sube explícitamente la tasa de bits del audio, para que en la fase de video final no se introduzca una distorsión notable por un valor por defecto demasiado bajo.
audio_bitrate = "192k"
fps = 30
# FFmpeg Al concatenar/transcodificar según la tasa de fotogramas, la duración final puede ser unas decenas de milisegundos más corta que la duración teórica que lee MoviePy.
# Aquí se deja al material de video un pequeño margen de seguridad, para que el final del audio no quede en negro por el redondeo de fotogramas,
# se trabe o el último trozo de la narración se quede sin imagen.
_VIDEO_DURATION_SAFETY_MARGIN = 0.1
_MIN_MATERIAL_DIMENSION = 480
# Las aplicaciones de mensajería y algunos codificadores redondean hacia abajo el tamaño de la imagen; por ejemplo, WhatsApp comprime material 9:16
# a 478x850, dos píxeles menos que 480. Exigir estrictamente 480 haría que todo ese material se
# descartara y el proceso fallara con "no valid materials found". Aquí se deja una tolerancia muy pequeña,
# que deja pasar el material ligeramente por debajo del umbral solo por el redondeo, pero sigue bloqueando el material de baja calidad real.
_MIN_DIMENSION_TOLERANCE = 10
# Por defecto los fragmentos se procesan en serie, para que tras actualizar los usuarios antiguos no vean un aumento repentino del uso de CPU y memoria.
# Si hace falta acelerar se puede aumentar la concurrencia, pero cada vía arranca una tarea de codificación independiente.
_CLIP_PROCESSING_CONCURRENCY = 1


def _get_clip_processing_concurrency() -> int:
    try:
        concurrency = int(config.app.get("video_clip_concurrency", _CLIP_PROCESSING_CONCURRENCY))
    except (TypeError, ValueError):
        concurrency = _CLIP_PROCESSING_CONCURRENCY
    return max(1, min(8, concurrency))
_DEFAULT_VIDEO_CODEC = "libx264"
# ffmpeg Durante la concatenación de fragmentos no hay logs de fase, y `subprocess.run` bloquea hasta que el proceso termina, así que una concatenación larga
# se ve en el log como «sin salida». Aquí se registra información de actividad a intervalos, para distinguir entre codificando y colgado.
_FFMPEG_CONCAT_HEARTBEAT_SECONDS = 30.0
# MoviePy Al escribir el video final tampoco hay ningún log, y un video vertical de un minuto tarda varios minutos en codificarse
# en CPU. Se sigue el mismo método que en la fase de concatenación y se registra información de actividad a intervalos.
_STAGE_HEARTBEAT_SECONDS = 30.0
_DEFAULT_FFMPEG_CONCAT_TIMEOUT_SECONDS = 3600
_SUBTITLE_SPRING_DURATION_SECONDS = 0.18
_MIN_SUBTITLE_SPRING_SCALE = 0.05
_MAX_SUBTITLE_SPRING_SCALE = 1.35
_SUPPORTED_VIDEO_CODECS = (
    "libx264",
    "h264_nvenc",
    "h264_amf",
    "h264_qsv",
    "h264_mf",
    "h264_videotoolbox",
)
_runtime_disabled_video_codecs = set()
# MoviePy pipes sRGB frames, and ffmpeg's default RGB→YUV matrix is BT.601 with no color tags,
# while players and YouTube decode untagged HD as BT.709 and shift the colors. Convert with the
# BT.709 matrix and tag the stream; setparams writes the tags the -color_* flags alone do not.
# The pixel format stays the encoder's choice: MoviePy leaves odd-sized frames to libx264.
_BT709_VIDEO_FILTER = (
    "scale=out_color_matrix=bt709:out_range=tv,"
    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
)
_BT709_FFMPEG_PARAMS = ["-vf", _BT709_VIDEO_FILTER]


def _get_subtitle_spring_scale(time_seconds: float, duration_seconds: float) -> float:
    """Devuelve la escala que usa la animación de rebote de los subtítulos en un instante dado."""
    if duration_seconds <= 0 or time_seconds >= duration_seconds:
        return 1.0

    progress = max(0.0, min(time_seconds / duration_seconds, 1.0))
    scale = 1.0 - math.exp(-6.0 * progress) * math.cos(2.5 * math.pi * progress)
    return max(
        _MIN_SUBTITLE_SPRING_SCALE,
        min(scale, _MAX_SUBTITLE_SPRING_SCALE),
    )


def _scale_subtitle_frame_on_canvas(frame: np.ndarray, scale: float) -> np.ndarray:
    """
    Escala la imagen o la máscara transparente del subtítulo alrededor del centro, manteniendo el tamaño del lienzo.

    MoviePy guarda por separado los fotogramas de color del subtítulo y la máscara transparente. La animación de rebote debe aplicar a ambos exactamente
    la misma escala y recorte; de lo contrario, el primer fotograma de la animación compondría la zona transparente como un contorno de texto negro
    sobre el video. Un array bidimensional representa una máscara con valores de 0 a 1, y uno tridimensional un fotograma de color RGB/RGBA.
    """
    if frame.ndim not in (2, 3):
        raise ValueError("subtitle frame must be a 2D mask or 3D color frame")

    height, width = frame.shape[:2]
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    offset = ((width - scaled_width) // 2, (height - scaled_height) // 2)

    if frame.ndim == 2:
        # MoviePy La máscara usa flotantes de 0 a 1 y el modo L de Pillow usa 0 a 255; tras la conversión
        # se restauran el tipo y el rango originales, para que la semántica de transparencia de CompositeVideoClip no cambie.
        mask_image = Image.fromarray(
            np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        )
        resized_mask = mask_image.resize(
            (scaled_width, scaled_height),
            Image.Resampling.BILINEAR,
        )
        mask_canvas = Image.new("L", (width, height), 0)
        mask_canvas.paste(resized_mask, offset)
        return (np.asarray(mask_canvas) / 255.0).astype(frame.dtype, copy=False)

    if frame.shape[2] not in (3, 4):
        raise ValueError("subtitle color frame must use RGB or RGBA channels")
    color_image = Image.fromarray(frame)
    resized_color = color_image.resize(
        (scaled_width, scaled_height),
        Image.Resampling.BILINEAR,
    )
    background = (0, 0, 0, 0) if frame.shape[2] == 4 else (0, 0, 0)
    color_canvas = Image.new(color_image.mode, (width, height), background)
    color_canvas.paste(resized_color, offset)
    return np.asarray(color_canvas).astype(frame.dtype, copy=False)


def _apply_subtitle_spring_animation(clip, subtitle_duration: float):
    """Escala a la vez el fotograma de color y la máscara del subtítulo, para que la animación de rebote no tenga un primer fotograma negro."""
    animation_duration = min(
        _SUBTITLE_SPRING_DURATION_SECONDS,
        max(0.0, subtitle_duration),
    )
    if animation_duration <= 0:
        return clip

    def transform_frame(get_frame, time_seconds):
        frame = get_frame(time_seconds)
        scale = _get_subtitle_spring_scale(time_seconds, animation_duration)
        if scale == 1.0:
            return frame
        return _scale_subtitle_frame_on_canvas(frame, scale)

    # apply_to=["mask"] es la clave de la corrección: MoviePy por defecto solo procesa el fotograma de color, así que la implementación antigua
    # mantenía brevemente la máscara a tamaño original al aparecer cada subtítulo, mostrando un contorno de texto negro.
    return clip.transform(transform_frame, apply_to=["mask"])


def _get_required_video_duration(audio_duration: float) -> float:
    """
    Devuelve la duración objetivo de la concatenación del material de video.

    Uso: al componer el video, la duración del material debe cubrir el audio de la narración. Si solo se iguala «justo»
    a la duración del audio, FFmpeg puede acortar ligeramente el video final por el redondeo de la tasa de fotogramas, así que se añade siempre un
    pequeño margen. La función está separada para facilitar las pruebas y ajustar el margen según la experiencia real.
    """
    return max(0.0, float(audio_duration) + _VIDEO_DURATION_SAFETY_MARGIN)


def is_material_resolution_acceptable(width: int, height: int) -> bool:
    """
    Determina si la resolución del material es suficiente para la composición.

    El mínimo nominal es 480x480, pero se permiten `_MIN_DIMENSION_TOLERANCE` píxeles menos,
    para admitir tamaños redondeados hacia abajo por codificadores o aplicaciones de mensajería (como el 478x850 de WhatsApp).
    """
    min_dimension = _MIN_MATERIAL_DIMENSION - _MIN_DIMENSION_TOLERANCE
    return width >= min_dimension and height >= min_dimension


def _prioritize_unique_source_clips(
    subclipped_items: List[SubClippedVideoClip],
    concat_mode: VideoConcatMode,
    source_usage: dict[str, int] | None = None,
    source_groups: dict[str, str] | None = None,
) -> List[SubClippedVideoClip]:
    """
    Prioriza que cada material de origen aparezca una sola vez, reduciendo la probabilidad de que el mismo material se repita en el video final.

    El material en línea a menudo tiene «un video largo cortado en varios fragmentos cortos». La lógica antigua, en
    random el modo , mezclaba directamente todos los fragmentos cortos, de modo que varios trozos del mismo video de origen podían
    repartirse al principio y en medio, y el usuario lo percibía como material repetido. Esta función solo ajusta el orden de los fragmentos:
    pone primero el fragmento más largo de cada archivo de origen y deja el resto como respaldo; cuando la duración total del material no basta,
    se sigue permitiendo que los fragmentos siguientes completen la duración del audio, sin perjudicar la tasa de éxito de la generación. Se prefiere el fragmento
    más largo para no elegir al azar trozos cortos del final del video, que harían reutilizar material antes de tiempo aunque haya suficiente.
    """
    if not subclipped_items:
        return []

    concat_mode_value = getattr(concat_mode, "value", concat_mode)
    if concat_mode_value != VideoConcatMode.random.value:
        if source_usage is None:
            return subclipped_items
        if not source_groups:
            return sorted(
                subclipped_items,
                key=lambda item: source_usage.get(item.source_file_path, 0),
            )
        # Keep keyword rounds in order while rotating candidates within each keyword.
        groups = {}
        for item in subclipped_items:
            key = source_groups.get(item.source_file_path, item.source_file_path)
            groups.setdefault(key, []).append(item)
        for items in groups.values():
            items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
        return [
            item
            for row in itertools.zip_longest(*groups.values())
            for item in row
            if item is not None
        ]

    grouped_items: dict[str, list[SubClippedVideoClip]] = {}
    for item in subclipped_items:
        grouped_items.setdefault(item.source_file_path, []).append(item)

    primary_items = []
    overflow_items = []
    for items in grouped_items.values():
        primary_item = max(items, key=lambda item: item.duration)
        primary_items.append(primary_item)
        overflow_items.extend(item for item in items if item is not primary_item)

    random.shuffle(primary_items)
    random.shuffle(overflow_items)
    if source_usage is not None:
        # Stable sorting retains randomness among equally used sources.
        primary_items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
        overflow_items.sort(key=lambda item: source_usage.get(item.source_file_path, 0))
    logger.info(
        "prioritized unique video materials, "
        f"sources: {len(grouped_items)}, "
        f"primary clips: {len(primary_items)}, "
        f"fallback clips: {len(overflow_items)}"
    )
    return primary_items + overflow_items


def get_ffmpeg_binary():
    """
    Mantiene la compatibilidad con los llamadores históricos que leen la ruta de FFmpeg directamente del servicio video.

    La lógica real de resolución ya se movió a `app.utils.utils.get_ffmpeg_binary()`; el video, la voz
    y las cadenas que se añadan deben reutilizar la misma prioridad; aquí se mantiene un envoltorio fino, para que scripts externos o
    pruebas antiguas que importan directamente `app.services.video.get_ffmpeg_binary` no den AttributeError.
    """
    return utils.get_ffmpeg_binary()


def _get_configured_video_codec() -> str:
    """
    Lee el codificador de video configurado por el usuario.

    Esta configuración está pensada para usuarios avanzados que quieran probar la codificación por hardware como NVENC/AMF/QSV/VideoToolbox
    . Aquí se permite a propósito solo una lista blanca fija, para que al no abrir parámetros arbitrarios de FFmpeg un parámetro mal escrito por el usuario
    no haga incontrolable el formato de salida, ni haga fallar la tarea de generación en una fase posterior.
    """
    configured_codec = str(
        config.app.get("video_codec", _DEFAULT_VIDEO_CODEC) or _DEFAULT_VIDEO_CODEC
    ).strip()
    if configured_codec not in _SUPPORTED_VIDEO_CODECS:
        logger.warning(
            f"unsupported video codec configured: {configured_codec}, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC
    return configured_codec


@lru_cache(maxsize=16)
def _ffmpeg_encoder_exists(ffmpeg_binary: str, codec: str) -> bool:
    """
    Comprueba si el FFmpeg actual declara soporte para el codificador indicado.

    Esto solo demuestra que FFmpeg se compiló con ese encoder, no que el hardware y los drivers de la máquina actual
    estén necesariamente disponibles. Por eso, si la codificación real falla, se sigue recurriendo a libx264.
    """
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {str(exc)}"
        )
        return False

    if result.returncode != 0:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {(result.stderr or result.stdout or '').strip()}"
        )
        return False
    return codec in result.stdout


def _get_effective_video_codec(preferred_codec: str | None = None) -> str:
    """
    Devuelve el codificador de video usado realmente en esta ocasión.

    Cuando el usuario elige un codificador por hardware, primero se comprueba la lista de encoders de FFmpeg; si en este proceso ya
    falló una codificación real, también se recurre directamente al respaldo, para que no falle repetidamente cada fragmento de una tarea.
    """
    selected_codec = preferred_codec or _get_configured_video_codec()
    if selected_codec == _DEFAULT_VIDEO_CODEC:
        return _DEFAULT_VIDEO_CODEC

    if selected_codec in _runtime_disabled_video_codecs:
        logger.warning(
            f"video codec {selected_codec} was disabled after a runtime failure, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    ffmpeg_binary = utils.get_ffmpeg_binary()
    if not _ffmpeg_encoder_exists(ffmpeg_binary, selected_codec):
        logger.warning(
            f"ffmpeg encoder {selected_codec} is not available, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    return selected_codec


def _disable_runtime_video_codec(codec: str, reason: str):
    if codec == _DEFAULT_VIDEO_CODEC:
        return
    _runtime_disabled_video_codecs.add(codec)
    logger.warning(
        f"video codec {codec} failed, fallback to {_DEFAULT_VIDEO_CODEC}. "
        f"reason: {reason}"
    )


def _get_temp_audio_dir(output_dir: str) -> str:
    """
    Return the directory to use for MoviePy's temporary audio file.

    On Windows, Windows Defender can lock files written to the task output
    directory while scanning them, causing MoviePy to fail with a
    PermissionError (WinError 32) on the TEMP_MPY_wvf_snd temp file and
    leaving the final MP4 at 0 bytes.  Using the system temp directory
    sidesteps the scan without changing behaviour on other platforms.

    On Linux/macOS/Docker the output directory is returned unchanged so
    existing behaviour is preserved.
    """
    if sys.platform == "win32":
        return tempfile.gettempdir()
    return output_dir


def _fallback_write_videofile(clip, output_file: str, failed_codec: str, reason: str, **kwargs):
    """
    Tras un fallo de codificación por hardware se reintenta con libx264, y solo si el reintento tiene éxito se desactiva ese codificador por hardware.

    Windows Las causas de fallo de FFmpeg en son complejas: puede que la tarjeta gráfica/driver no lo soporte, o que el archivo de salida
    esté en uso, haya problemas de permisos del directorio, bloqueos del antivirus u otros problemas generales de E/S. Solo cuando libx264 logra escribir
    se puede concluir que el fallo original vino probablemente del propio codificador por hardware, para no perjudicar a tareas posteriores.
    """
    clip.write_videofile(output_file, codec=_DEFAULT_VIDEO_CODEC, **kwargs)
    _disable_runtime_video_codec(failed_codec, reason)
    return _DEFAULT_VIDEO_CODEC


def _write_videofile_with_codec_fallback(
    clip, output_file: str, codec: str, atomic_output: bool = False, **kwargs
):
    """
    Escribe el video con el codificador indicado y, si falla, reintenta automáticamente una vez con libx264.

    Que un codificador por hardware esté disponible no depende solo de FFmpeg, sino también de la tarjeta gráfica, los drivers y el entorno de ejecución actual.
    La tarea de generación no puede fallar por completo porque un codificador avanzado no esté disponible, así que aquí se centraliza el respaldo.
    """
    kwargs.setdefault("ffmpeg_params", _BT709_FFMPEG_PARAMS)
    if atomic_output:
        # Final videos can be downloaded by path while they are being rendered.
        # Keep both failed encodes and in-progress writes away from that path.
        output_dir = os.path.dirname(os.path.abspath(output_file))
        descriptor, temp_output = tempfile.mkstemp(
            prefix=f".{os.path.basename(output_file)}.",
            suffix=os.path.splitext(output_file)[1] or ".mp4",
            dir=output_dir,
        )
        os.close(descriptor)
        os.unlink(temp_output)
        try:
            used_codec = _write_videofile_with_codec_fallback(
                clip, temp_output, codec, **kwargs
            )
            os.replace(temp_output, output_file)
            return used_codec
        finally:
            try:
                os.unlink(temp_output)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    f"failed to remove temporary final video: {temp_output}, "
                    f"error: {exc}"
                )

    effective_codec = _get_effective_video_codec(codec)
    try:
        clip.write_videofile(output_file, codec=effective_codec, **kwargs)
        return effective_codec
    except Exception as exc:
        if effective_codec == _DEFAULT_VIDEO_CODEC:
            raise
        return _fallback_write_videofile(
            clip,
            output_file,
            failed_codec=effective_codec,
            reason=str(exc),
            **kwargs,
        )


def _escape_ffmpeg_concat_path(file_path: str) -> str:
    # concat demuxer La ruta se envuelve en comillas simples, y las comillas simples dentro de la ruta deben escaparse primero.
    return file_path.replace("'", "'\\''")


def _format_ffmpeg_concat_path(file_path: str) -> str:
    """
    Genera la ruta en la lista de archivos del concat demuxer.

    FFmpeg La documentación oficial exige escapar los caracteres especiales y espacios en la concat list;Windows
    las barras invertidas de las rutas absolutas también se interpretan fácilmente como caracteres de escape. Aquí se convierten todas a barras normales,
    para que `C:\\Users\\...` se convierta en `C:/Users/...`, y luego se tratan las comillas simples, compatible con macOS/Linux.
    """
    absolute_path = os.path.abspath(file_path)
    # Normalize the native separator only. A backslash can be a literal
    # component of a valid POSIX path and must still identify the same file.
    return _escape_ffmpeg_concat_path(absolute_path.replace(os.sep, "/"))


def _describe_concat_output_progress(output_file: str) -> str:
    """Devuelve una descripción legible del tamaño actual del archivo de salida, para el log de latido de la concatenación."""
    try:
        size = os.path.getsize(output_file)
    except OSError:
        # Si el archivo de salida aún no se ha creado también hay que degradar con seguridad, sin afectar a la propia concatenación.
        return "output size not available"
    return f"output size: {size / (1024 * 1024):.2f} MB"


@contextmanager
def _stage_heartbeat(description: str):
    """
    Durante una fase larga sin log propio, registra logs de actividad a intervalos.

    El hilo de latido se vincula al hilo que entró en esta fase, para que el log aparezca en el panel de la WebUI de la tarea correspondiente.
    Al terminar la fase o lanzarse una excepción se detiene y espera a que salga el hilo de latido, para que tras terminar la fase no siga
    escribiendo «still running».
    """
    started_at = time.monotonic()
    stop_event = threading.Event()

    def log_heartbeat() -> None:
        while not stop_event.wait(_STAGE_HEARTBEAT_SECONDS):
            logger.info(
                f"{description} still running: "
                f"elapsed={time.monotonic() - started_at:.0f}s"
            )

    reporter = threading.Thread(
        target=logging_utils.bind_log_scope(log_heartbeat), daemon=True
    )
    reporter.start()
    try:
        yield
    finally:
        stop_event.set()
        reporter.join(timeout=1)


def _report_clip_progress(
    progress_callback: Callable[[float], None] | None,
    covered_duration: float,
    required_duration: float,
) -> None:
    """Convierte la duración de video final ya cubierta en una proporción de 0~1 y notifica al llamador."""
    if progress_callback is None:
        return
    fraction = 1.0
    if required_duration > 0:
        fraction = min(1.0, covered_duration / required_duration)
    try:
        progress_callback(fraction)
    except Exception as exc:
        # El progreso es solo información de visualización. Un fallo del callback (por ejemplo, el backend de estado no disponible temporalmente) no puede invalidar
        # invalidar los fragmentos ya procesados.
        logger.warning(
            "failed to report clip processing progress: "
            f"error={type(exc).__name__}, detail={exc}"
        )


def _run_concat_with_heartbeat(command: list[str], output_file: str):
    """
    Espera bloqueando a que termine ffmpeg, registrando logs de actividad a intervalos mientras tanto.

    ffmpeg Al concatenar fragmentos no hay logs de fase, y `subprocess.run` guarda la salida en búfer hasta que termina el proceso, así que una concatenación larga
    se ve en el log como «sin salida». Se registran el tiempo de espera y el tamaño del archivo de salida, para distinguir entre codificando y colgado.
    """
    started_at = time.monotonic()
    stop_event = threading.Event()

    def log_heartbeat() -> None:
        while not stop_event.wait(_FFMPEG_CONCAT_HEARTBEAT_SECONDS):
            logger.info(
                "ffmpeg concat still running: "
                f"elapsed={time.monotonic() - started_at:.0f}s, "
                f"{_describe_concat_output_progress(output_file)}"
            )

    reporter = threading.Thread(
        target=logging_utils.bind_log_scope(log_heartbeat), daemon=True
    )
    reporter.start()
    try:
        configured_timeout = config.app.get(
            "ffmpeg_concat_timeout_seconds", _DEFAULT_FFMPEG_CONCAT_TIMEOUT_SECONDS
        )
        try:
            timeout_seconds = float(configured_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("ffmpeg_concat_timeout_seconds must be positive") from exc
        if (
            isinstance(configured_timeout, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("ffmpeg_concat_timeout_seconds must be positive")

        try:
            # subprocess.run kills and waits for FFmpeg on timeout, so a stalled
            # encoder cannot leave an orphaned child or a permanently active task.
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"ffmpeg concat exceeded {timeout_seconds:g} seconds"
            ) from exc
    finally:
        stop_event.set()


def concat_video_clips_with_ffmpeg(
    clip_files: List[str],
    output_file: str,
    threads: int,
    output_dir: str,
    max_duration: float | None = None,
):
    # Separate renders may share a directory. Each FFmpeg process must keep its
    # own manifest until all codec attempts finish, without overwriting or
    # deleting another render's list.
    concat_list_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="ffmpeg-concat-", suffix=".txt",
            dir=output_dir, delete=False,
        ) as fp:
            concat_list_file = fp.name
            for clip_file in clip_files:
                fp.write(f"file '{_format_ffmpeg_concat_path(clip_file)}'\n")
    except Exception:
        if concat_list_file:
            delete_files(concat_list_file)
        raise

    staged_output = None

    def build_command(codec: str) -> list[str]:
        command = [
            utils.get_ffmpeg_binary(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_file,
            "-c:v",
            codec,
            "-threads",
            str(threads or 2),
            "-vf",
            _BT709_VIDEO_FILTER,
            "-pix_fmt",
            "yuv420p",
        ]
        if max_duration is not None and max_duration > 0:
            command.extend(["-t", f"{max_duration:.3f}"])
        command.append(staged_output)
        return command

    def run_concat(codec: str):
        command = build_command(codec)
        # Usa ffmpeg para hacer la concatenación y la codificación una sola vez, evitando que MoviePy recodifique repetidamente al unir fragmento a fragmento,
        # reduciendo así el riesgo de pérdida de calidad y desplazamiento de color. Durante la espera bloqueante el log de latido muestra que la tarea sigue en ejecución.
        result = _run_concat_with_heartbeat(command, staged_output)
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(error_message or "ffmpeg concat failed")
        return codec

    try:
        descriptor, staged_output = tempfile.mkstemp(
            prefix=".ffmpeg-concat-",
            suffix=os.path.splitext(output_file)[1] or ".mp4",
            dir=os.path.dirname(os.path.abspath(output_file)),
        )
        os.close(descriptor)
        effective_codec = _get_effective_video_codec()
        try:
            result_codec = run_concat(effective_codec)
        except TimeoutError:
            # A hung encoder is not evidence that another codec will work. Do
            # not spend a second timeout period retrying the same input.
            raise
        except Exception as exc:
            if effective_codec == _DEFAULT_VIDEO_CODEC:
                raise
            result_codec = run_concat(_DEFAULT_VIDEO_CODEC)
            _disable_runtime_video_codec(effective_codec, str(exc))
        # Failed attempts and in-progress output stay private until FFmpeg has
        # finished. Publication errors must not trigger another codec attempt.
        if os.path.getsize(staged_output) == 0:
            raise RuntimeError("ffmpeg concat produced no output")
        os.replace(staged_output, output_file)
        return result_codec
    finally:
        delete_files([concat_list_file, staged_output])


def _sanitize_image_file(image_path: str) -> str:
    # Algunas imágenes locales se pueden abrir con Pillow, pero por metadatos EXIF/eXIf dañados hacen que
    # ImageClip lance una excepción directamente en la fase de análisis. Aquí se vuelve a exportar una «imagen limpia», quitando los metadatos dañados.
    # Bound the intermediate basename while retaining complete source identity.
    # Different extensions must not overwrite pixels, and valid long filenames
    # must not exceed the filesystem's component limit after adding a suffix.
    source_digest = hashlib.sha256(os.fsencode(os.path.abspath(image_path)))

    temp_path = ""
    try:
        with Image.open(image_path) as image:
            with ImageOps.exif_transpose(image) as upright:
                upright.load()
                # Strip metadata after applying the camera's orientation.
                # Palette transparency belongs to pixels, not removable metadata.
                mode = "RGBA" if "A" in upright.getbands() or "transparency" in upright.info else "RGB"
                if upright.mode in {"I;16", "I;16B", "I;16L"}:
                    # 16-bit grayscale samples span0..65535; an RGB cast either
                    # clips them to white or wraps when MoviePy encodes them.
                    with upright.convert("I").point(lambda value: value / 257 + 0.5) as byte_range:
                        cleaned_image = byte_range.convert(mode)
                else:
                    cleaned_image = upright.convert(mode)
                cleaned_image.info.clear()
                source_digest.update(str(cleaned_image.size).encode())
                source_digest.update(cleaned_image.mode.encode())
                source_digest.update(cleaned_image.tobytes())
                sanitized_path = os.path.join(
                    os.path.dirname(image_path),
                    f"{source_digest.hexdigest()}.sanitized.png",
                )
                descriptor, temp_path = tempfile.mkstemp(
                    prefix=".image-sanitize-", suffix=".png",
                    dir=os.path.dirname(os.path.abspath(sanitized_path)),
                )
                os.close(descriptor)
                cleaned_image.save(temp_path)
                cleaned_image.close()
        os.replace(temp_path, sanitized_path)
        temp_path = ""
    finally:
        if temp_path:
            delete_files(temp_path)

    return sanitized_path


def _open_image_clip_with_fallback(image_path: str):
    # MoviePy does not apply camera EXIF orientation while decoding an image.
    # CMYK JPEG channels must become RGB; LA must become RGBA so MoviePy
    # creates a proper opacity mask rather than treating alpha as a color band.
    # Bilevel pixels must become byte-range RGB rather than bool values0/1.
    # Ordinary RGB inputs retain the direct path.
    try:
        with Image.open(image_path) as image:
            orientation = image.getexif().get(274, 1)
            image_mode = image.mode
    except Exception:
        orientation = 1
        image_mode = None
    if orientation in range(2, 9) or image_mode in {"CMYK", "LA", "1", "I;16", "I;16B", "I;16L"}:
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path

    # Se intenta primero abrir la imagen original; si falla por metadatos dañados, se intenta generar una copia sin metadatos.
    try:
        return ImageClip(image_path), image_path
    except Exception as exc:
        logger.warning(
            f"failed to open image directly, trying sanitized copy: {image_path}, error: {str(exc)}"
        )
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path


_moviepy_reader_open_lock = threading.Lock()


def _open_video_clip_quietly(video_path: str, audio: bool = False) -> VideoFileClip:
    """
    Abre el archivo de video en silencio, para que MoviePy 2.1.x no imprima directamente en stdout la información de detección de ffmpeg.

    Contexto:
    en la versión actual de la dependencia, `FFMPEG_VideoReader` tiene internamente `print(self.infos)` y
    `print(ffmpeg command)`, y al leer videos intermedios sin pista de audio imprime
    `audio_found: False`. Esto son solo los metadatos del material de entrada y no significa que el video final no tenga audio,
    pero confunde a los usuarios de la WebUI/terminal haciéndoles creer que la generación falló.

    Implementación:
    1. solo se redirige stdout durante la breve ventana en que se abre el VideoFileClip;
    2. por defecto `audio=False`, porque la fase de material de video del proyecto no necesita conservar el sonido original del material,
       y el audio final se monta de forma unificada en la fase `generate_video()`;
    3. si la dependencia realmente imprime contenido, se degrada a log de debug, para investigar cuando haga falta.
    """
    captured_stdout = io.StringIO()
    # redirect_stdout changes process-wide state. Overlapping reader opens can
    # restore each other's capture buffers instead of the original stdout.
    # Serialize this short construction window; clip processing stays parallel.
    with _moviepy_reader_open_lock:
        with redirect_stdout(captured_stdout):
            clip = VideoFileClip(video_path, audio=audio)

    moviepy_stdout = captured_stdout.getvalue().strip()
    if moviepy_stdout:
        logger.debug(
            "suppressed MoviePy video reader stdout for "
            f"{video_path}, chars: {len(moviepy_stdout)}"
        )

    return clip


def close_clip(clip):
    if clip is None:
        return
        
    try:
        # close main resources
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
            
        # close audio resources
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio
            
        # close mask resources
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask
            
        # handle child clips in composite clips
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)
            
        # clear clip list
        if hasattr(clip, 'clips'):
            clip.clips = []
            
    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")
    
    del clip
    gc.collect()

def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]

    # Al completar el video en bucle, la misma ruta de fragmento temporal aparece varias veces en la lista de concatenación de FFmpeg.
    # La concatenación debe conservar los duplicados, pero la limpieza solo puede borrar una vez; aquí se deduplican en el orden original, para que todos
    # los llamadores tengan un comportamiento idempotente y no se emitan FileNotFoundError seguidos tras el primer borrado exitoso.
    unique_files = dict.fromkeys(file for file in files if file)
    for file in unique_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            # La limpieza admite que el archivo ya no exista, por ejemplo en la ruta de fallo de FFmpeg o si una limpieza concurrente ya
            # recuperó el archivo; no es un problema que deba resolver el usuario y no debe ensuciar el log de generación.
            continue
        except OSError as e:
            # Los problemas de permisos, sistemas de archivos de solo lectura o anomalías del disco dejan archivos temporales reales; se conserva warning
            # para localizar problemas del entorno según la ruta concreta y el error del sistema.
            logger.warning(f"failed to delete temporary file {file}: {str(e)}")


def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file:
        try:
            resolved_bgm_file = bgm_service.resolve_bgm_file(bgm_file)
        except ValueError as exc:
            # API El bgm_file de la petición viene de la entrada del usuario y solo puede resolverse en el directorio de BGM del usuario o en el de canciones
            # integradas, impidiendo que MoviePy lea archivos arbitrarios del servidor como configuración o claves.
            logger.warning(
                f"reject unsafe bgm file: {bgm_file}, error: {str(exc)}"
            )
            return ""
        return resolved_bgm_file

    if bgm_type == "random":
        # Con música propia en storage/bgm la rotación usa solo esa, para que las canciones
        # integradas no rompan el tono del canal; sin ella se usan las integradas.
        files = bgm_service.list_uploaded_bgm_files() or bgm_service.list_bgm_files()
        # Si el directorio de música de fondo está vacío se recurre directamente a «no usar BGM», para que random.choice([]) no lance una excepción.
        if not files:
            logger.warning("no background music files found")
            return ""
        return random.choice(files)

    return ""


def _fit_clip_to_canvas(
    clip,
    *,
    target_width: int,
    target_height: int,
    fit_mode: VideoFitMode | str = VideoFitMode.cover,
):
    """Resize a clip to an exact canvas using cover/crop or contain/letterbox."""
    source_width, source_height = (int(value) for value in clip.size)
    target_width = int(target_width)
    target_height = int(target_height)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError(
            "video dimensions must be positive: "
            f"source={source_width}x{source_height}, "
            f"target={target_width}x{target_height}"
        )

    mode = VideoFitMode(fit_mode)
    if (source_width, source_height) == (target_width, target_height):
        return clip

    # Exact aspect-ratio matches do not need either a crop or a background.
    if source_width * target_height == source_height * target_width:
        return clip.resized(new_size=(target_width, target_height))

    width_scale = target_width / source_width
    height_scale = target_height / source_height

    if mode == VideoFitMode.cover:
        # ceil guarantees the resized clip covers the complete canvas despite
        # floating-point rounding. Any excess is removed symmetrically.
        scale_factor = max(width_scale, height_scale)
        resized_width = max(target_width, math.ceil(source_width * scale_factor))
        resized_height = max(target_height, math.ceil(source_height * scale_factor))
        resized_clip = clip.resized(new_size=(resized_width, resized_height))
        crop_x = max(0, (resized_width - target_width) // 2)
        crop_y = max(0, (resized_height - target_height) // 2)
        return resized_clip.cropped(
            x1=crop_x,
            y1=crop_y,
            width=target_width,
            height=target_height,
        )

    # contain preserves the legacy behavior: show the complete source frame,
    # centered over a black canvas when the aspect ratios differ.
    scale_factor = min(width_scale, height_scale)
    resized_width = max(1, min(target_width, int(source_width * scale_factor)))
    resized_height = max(1, min(target_height, int(source_height * scale_factor)))
    background = ColorClip(
        size=(target_width, target_height), color=(0, 0, 0)
    ).with_duration(clip.duration)
    resized_clip = clip.resized(
        new_size=(resized_width, resized_height)
    ).with_position("center")
    return CompositeVideoClip(
        [background, resized_clip], size=(target_width, target_height)
    ).with_duration(clip.duration)


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    clip_speed: float = 1.0,
    video_fit_mode: VideoFitMode = VideoFitMode.cover,
    source_usage: dict[str, int] | None = None,
    source_groups: dict[str, str] | None = None,
    used_video_paths: List[str] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> str:
    # Direct callers and persisted tasks can still supply the raw string.
    # Normalize it once for all downstream enum access.
    video_concat_mode = VideoConcatMode(video_concat_mode or VideoConcatMode.random)
    audio_clip = AudioFileClip(audio_file)
    try:
        # Aquí solo hace falta leer la duración del audio de la narración para decidir la longitud de concatenación del video de material; después no se vuelve a usar
        # audio_clip. Se cierra justo tras leer, para no filtrar el handle del archivo en salidas tempranas o rutas de excepción.
        audio_duration = audio_clip.duration
    finally:
        close_clip(audio_clip)
    logger.info(f"audio duration: {audio_duration} seconds")
    logger.info(f"maximum clip duration: {max_clip_duration} seconds")
    required_video_duration = _get_required_video_duration(audio_duration)
    logger.info(
        f"required video duration: {required_video_duration:.2f} seconds "
        f"(audio duration + {_VIDEO_DURATION_SAFETY_MARGIN:.2f}s safety margin)"
    )

    # Compatible con llamadas directas a la API que no pasan modo de transición, para que el acceso posterior a .value no falle.
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    normalized_clip_speed = utils.normalize_clip_speed(clip_speed)
    if normalized_clip_speed != 1.0:
        # Solo se registra una vez el valor efectivo final, lo que facilita localizar parámetros fuera de rango de la API que se normalizaron,
        # y evita repetir el mismo log en la ruta caliente de cada fragmento.
        logger.info(f"clip playback speed: {normalized_clip_speed:.2f}x")
    # max_clip_duration restringe la duración final de reproducción en el video, no la duración leída del video de origen.
    # MoviePy Reproducir 1.5 segundos de imagen de origen a 0.5x da un fragmento de 3 segundos, y reproducir a 2x
    # 6 segundos de imagen de origen también da un fragmento de 3 segundos. Por eso antes de cortar hay que deducir la duración de origen según la velocidad; si
    # se siguieran leyendo siempre 3 segundos para ralentizar y recortar, y el siguiente fragmento empezara en el segundo 3 del origen, se saltarían
    # 1.5 segundos de imagen intermedios. Este cálculo garantiza además que la línea de tiempo de origen sea continua y sin solapamientos a cualquier velocidad.
    source_clip_duration = max_clip_duration * normalized_clip_speed
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    fit_mode = VideoFitMode(video_fit_mode)
    video_width, video_height = aspect.to_resolution()

    processed_clips = []
    subclipped_items = []
    video_duration = 0
    for video_path in video_paths:
        clip = None
        try:
            clip = _open_video_clip_quietly(video_path)
            clip_duration = float(clip.duration)
            clip_w, clip_h = clip.size
            if (
                not math.isfinite(clip_duration)
                or clip_duration <= 0
                or not all(
                    math.isfinite(float(dimension)) and float(dimension) > 0
                    for dimension in (clip_w, clip_h)
                )
            ):
                raise ValueError("invalid video duration or dimensions")
        except Exception as exc:
            logger.warning(
                f"skipping unreadable video source: path={video_path}, error={exc}"
            )
            continue
        finally:
            close_clip(clip)
        
        start_time = 0

        while start_time < clip_duration:
            end_time = min(start_time + source_clip_duration, clip_duration)

            # conserva todos los segmentos válidos.
            # Así no se pierde el material «cuyo video completo ya es más corto que max_clip_duration»,
            # ni se descarta el pequeño trozo final que queda de un video largo.
            if end_time > start_time:
                subclipped_items.append(
                    SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        width=clip_w,
                        height=clip_h,
                        source_file_path=video_path,
                    )
                )

            start_time = end_time
            if video_concat_mode.value == VideoConcatMode.sequential.value:
                break

    subclipped_items = _prioritize_unique_source_clips(
        subclipped_items=subclipped_items,
        concat_mode=video_concat_mode,
        **({"source_usage": source_usage, "source_groups": source_groups}
           if source_usage is not None else {}),
    )
        
    logger.debug(f"total subclipped items: {len(subclipped_items)}")
    
    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    def process_one_clip(indexed_item):
        """Recorta, cambia la velocidad y aplica la transición a un fragmento de origen y lo escribe; si falla devuelve None."""
        index, subclipped_item = indexed_item
        source_clip = None
        clip = None
        clip_file = None
        try:
            logger.debug(
                f"processing clip {index + 1}: {subclipped_item.width}x{subclipped_item.height}, "
                f"source: {os.path.basename(subclipped_item.source_file_path)}"
            )
            source_clip = _open_video_clip_quietly(subclipped_item.file_path)
            clip = source_clip.subclipped(
                subclipped_item.start_time, subclipped_item.end_time
            )
            # La velocidad de reproducción es una propiedad del material y debe aplicarse antes de la transición. Así las transiciones de un segundo como Fade/Slide
            # no cambian a 0.5 o 2 segundos con la velocidad del material; el recorte de duración máxima posterior sigue actuando como
            # respaldo de seguridad ante errores de coma flotante o duraciones anómalas del material, garantizando que el fragmento final no supere el límite configurado.
            if normalized_clip_speed != 1.0:
                clip = clip.with_speed_scaled(normalized_clip_speed)
            # Normalize every source clip before transitions are applied. In cover mode
            # the clip fills the canvas and the excess edges are cropped; contain keeps
            # the complete source frame and uses black bars for the unused area.
            clip_w, clip_h = clip.size
            if clip_w != video_width or clip_h != video_height:
                clip_ratio = clip.w / clip.h
                video_ratio = video_width / video_height
                logger.debug(
                    "resizing clip, "
                    f"source: {clip_w}x{clip_h}, ratio: {clip_ratio:.2f}, "
                    f"target: {video_width}x{video_height}, ratio: {video_ratio:.2f}, "
                    f"fit_mode: {fit_mode.value}"
                )
                clip = _fit_clip_to_canvas(
                    clip,
                    target_width=video_width,
                    target_height=video_height,
                    fit_mode=fit_mode,
                )

            shuffle_side = random.choice(["left", "right", "top", "bottom"])
            # Short footage and faster playback can leave less than one second.
            # Keep the chosen transition inside the rendered clip's timeline.
            transition_duration = min(1.0, clip.duration)
            if transition_value == VideoTransitionMode.fade_in.value:
                clip = video_effects.fadein_transition(clip, transition_duration)
            elif transition_value == VideoTransitionMode.fade_out.value:
                clip = video_effects.fadeout_transition(clip, transition_duration)
            elif transition_value == VideoTransitionMode.slide_in.value:
                clip = video_effects.slidein_transition(clip, transition_duration, shuffle_side)
            elif transition_value == VideoTransitionMode.slide_out.value:
                clip = video_effects.slideout_transition(clip, transition_duration, shuffle_side)
            elif transition_value == VideoTransitionMode.zoom_in.value:
                clip = video_effects.zoomin_transition(clip, 1)
            elif transition_value == VideoTransitionMode.zoom_out.value:
                clip = video_effects.zoomout_transition(clip, 1)
            elif transition_value == VideoTransitionMode.shuffle.value:
                transition_funcs = [
                    lambda c: video_effects.fadein_transition(c, transition_duration),
                    lambda c: video_effects.fadeout_transition(c, transition_duration),
                    lambda c: video_effects.slidein_transition(c, transition_duration, shuffle_side),
                    lambda c: video_effects.slideout_transition(c, transition_duration, shuffle_side),
                    lambda c: video_effects.zoomin_transition(c, 1),
                    lambda c: video_effects.zoomout_transition(c, 1),
                ]
                shuffle_transition = random.choice(transition_funcs)
                clip = shuffle_transition(clip)

            if clip.duration > max_clip_duration:
                clip = clip.subclipped(0, max_clip_duration)

            # Write each candidate clip to a unique temporary file. Threads must not
            # share the same output path.
            # Distinct combinations can share a task/output directory. Reserve
            # an owned path so their encoders and cleanup never share a clip.
            with tempfile.NamedTemporaryFile(
                dir=output_dir or ".", prefix="temp-clip-", suffix=".mp4", delete=False
            ) as temporary_clip:
                clip_file = temporary_clip.name
            _write_videofile_with_codec_fallback(
                clip,
                clip_file,
                codec=_get_configured_video_codec(),
                logger=None,
                fps=fps,
            )

            clip_duration_saved = clip.duration
            processed_clip = SubClippedVideoClip(
                file_path=clip_file,
                duration=clip_duration_saved,
                width=clip_w,
                height=clip_h,
                source_file_path=subclipped_item.source_file_path,
            )
            clip_file = None
            return processed_clip
        except Exception as exc:
            logger.error(f"failed to process clip: {str(exc)}")
            return None
        finally:
            # The derived clip shares its FFmpeg reader with the source. If
            # subclipping itself failed, close the original source instead.
            close_clip(clip if clip is not None else source_clip)
            # MoviePy may leave a truncated MP4 even when encoding raises. It
            # was never returned, so concat cleanup cannot see it.
            # Close the reader first so Windows can remove the partial file.
            if clip_file:
                delete_files(clip_file)

    # Los fragmentos siempre se procesan en el pool de hilos. El log por fragmento es la única información de progreso de esta fase, y solo vinculándolo
    # al hilo que inició la composición puede el log de tareas de la WebUI recogerlo.
    process_clip_in_task_scope = logging_utils.bind_log_scope(process_one_clip)
    clip_processing_workers = 1
    if len(subclipped_items) >= 2:
        clip_processing_workers = min(_get_clip_processing_concurrency(), len(subclipped_items))
    with ThreadPoolExecutor(
        max_workers=clip_processing_workers,
        thread_name_prefix="clip-process",
    ) as executor:
        next_candidate_index = 0
        while (
            next_candidate_index < len(subclipped_items)
            and video_duration < required_video_duration
        ):
            remaining_duration = required_video_duration - video_duration
            batch = []
            batch_duration = 0.0
            candidate_index = next_candidate_index
            while candidate_index < len(subclipped_items) and batch_duration < remaining_duration:
                subclipped_item = subclipped_items[candidate_index]
                source_duration = subclipped_item.end_time - subclipped_item.start_time
                output_duration = min(
                    max_clip_duration,
                    source_duration / normalized_clip_speed,
                )
                batch.append((candidate_index, subclipped_item))
                batch_duration += output_duration
                candidate_index += 1
            if not batch:
                break
            for processed_clip in executor.map(process_clip_in_task_scope, batch):
                if processed_clip is None:
                    continue
                processed_clips.append(processed_clip)
                video_duration += processed_clip.duration
                # Cada fragmento de material 4K tarda más de diez segundos en procesarse; informar de la duración cubierta fragmento a fragmento permite que el log y la barra de progreso
                # reflejen que esta fase sigue avanzando.
                logger.info(
                    f"processed clip {len(processed_clips)}: "
                    f"{video_duration:.1f} of {required_video_duration:.1f}s covered"
                )
                _report_clip_progress(
                    progress_callback, video_duration, required_video_duration
                )
            next_candidate_index = candidate_index
    
    # loop processed clips until the video duration covers the audio duration and the small safety margin.
    if video_duration < required_video_duration:
        logger.warning(
            f"video duration ({video_duration:.2f}s) is shorter than required duration "
            f"({required_video_duration:.2f}s), looping clips to match audio length."
        )
        base_clips = processed_clips.copy()
        for clip in itertools.cycle(base_clips):
            if video_duration >= required_video_duration:
                break
            processed_clips.append(clip)
            video_duration += clip.duration
        logger.info(
            f"video duration: {video_duration:.2f}s, audio duration: {audio_duration:.2f}s, "
            f"required duration: {required_video_duration:.2f}s, "
            f"looped {len(processed_clips)-len(base_clips)} clips"
        )
     
    # merge video clips progressively, avoid loading all videos at once to avoid memory overflow
    logger.info("starting clip merging process")
    if not processed_clips:
        if video_paths:
            raise RuntimeError("no readable video clips available for merging")
        logger.warning("no clips available for merging")
        return combined_video_path
    
    clip_files = [clip.file_path for clip in processed_clips]
    logger.info(f"concatenating {len(clip_files)} clips with ffmpeg")
    try:
        concat_video_clips_with_ffmpeg(
            clip_files=clip_files,
            output_file=combined_video_path,
            threads=threads,
            output_dir=output_dir,
            max_duration=audio_duration,
        )
        if used_video_paths is not None:
            # Exclude safety-margin clips that FFmpeg trims entirely from the output.
            elapsed = 0.0
            for clip in processed_clips:
                if elapsed >= audio_duration:
                    break
                used_video_paths.append(clip.source_file_path)
                elapsed += clip.duration
    finally:
        # FFmpeg failures and timeouts must not strand one encoded MP4 per clip.
        # Repeated clips share a path; delete_files already deduplicates them.
        delete_files(clip_files)
            
    logger.info("video combining completed")
    return combined_video_path


def wrap_text(text, max_width, font="Arial", fontsize=60, *, shaped=None, fallbacks=None):
    # El salto de línea de los subtítulos debe hacerse antes de crear el TextClip; de lo contrario MoviePy solo calcula la zona de renderizado según el texto original
    # . Aquí se mide el ancho con PIL según la fuente y el tamaño actuales, para que cada línea se mantenga en lo posible
    # dentro del ancho disponible del video, evitando que tamaños grandes o frases largas en chino se salgan de la imagen.
    # Los subtítulos con letras árabes los modela y dibuja khatt por completo (ver subtitle_shaping); cada línea separada
    # por un salto de línea manual (incluidas las líneas de traducción al inglés) debe medirse con khatt.
    if shaped is None:
        shaped = subtitle_shaping.needs_shaping(text)
    if "\n" in text:
        # Hard breaks in SRT text are separate layout lines. Measuring a token
        # across a newline makes Pillow count both lines as one wide string,
        # then character wrapping can split an otherwise fitting word.
        wrapped_lines = [
            wrap_text(
                line, max_width, font=font, fontsize=fontsize, shaped=shaped, fallbacks=fallbacks
            )
            for line in text.split("\n")
        ]
        return "\n".join(line for line, _ in wrapped_lines), sum(
            height for _, height in wrapped_lines
        )

    # El texto en alfabeto árabe debe medirse según los glifos modelados y con la fuente usada realmente al dibujar (la fuente de respaldo
    # cuando faltan caracteres), para que los saltos de línea y la altura coincidan con el subtítulo final; sin modelado, Pillow mide
    # el ancho de glifos separados e independientes. Cada línea partida manualmente ya es un párrafo independiente (como el árabe seguido
    # de una línea de traducción al inglés), y las líneas partidas automáticamente llevan la marca de dirección del párrafo original, que también se incluye
    # al medir, para que la medición coincida con la dirección del renderizado. Los caracteres que la fuente de respaldo no puede dibujar (como chino intercalado) los dibuja
    # la fuente elegida, y también se cuentan al medir (ver subtitle_shaping.fallback_fonts).
    direction_mark = ""
    if shaped:
        selected_font = font
        font, _ = subtitle_shaping.resolve_font(font, text)
        if fallbacks is None:
            fallbacks = subtitle_shaping.fallback_fonts(font, selected_font)
        if text.strip():
            direction_mark = subtitle_shaping.direction_mark(text)
    shaped_font_path = font
    font = ImageFont.truetype(font, fontsize)
    max_width = int(max_width)

    # getbbox() devuelve «la altura de tinta visible de los glifos actuales», no la altura de línea de la fuente. Por ejemplo, un texto en inglés con solo
    # A, m, n y otros caracteres sin descendentes carece de descent, y con varias líneas ese error se acumula línea a línea,
    # hasta que la última línea del TextClip queda recortada por el lienzo. ascent + descent vienen de la propia fuente,
    # no dependen del idioma ni de la combinación de caracteres, y coinciden con el modelo de dibujo por baseline de MoviePy.
    ascent, descent = font.getmetrics()
    line_height = int(ascent + descent)
    if line_height <= 0:
        # Las fuentes TrueType/OpenType normales no llegan aquí; se mantienen un log diagnosticable y un respaldo de tamaño de fuente,
        # para que fuentes dañadas o poco convencionales que devuelven metrics anómalas no generen subtítulos de altura cero.
        logger.warning(
            "invalid subtitle font metrics, fallback to font size: "
            f"ascent={ascent}, descent={descent}, fontsize={fontsize}"
        )
        line_height = max(1, int(fontsize))

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        if not inner_text:
            return 0, line_height
        if shaped:
            width = subtitle_shaping.text_width(
                direction_mark + inner_text, shaped_font_path, fontsize, fallbacks
            )
            return width, line_height
        left, top, right, bottom = font.getbbox(inner_text)
        # bbox sigue sirviendo para medir el ancho real necesario para el salto de línea; la altura debe usar siempre una altura de línea de fuente estable.
        return right - left, line_height

    width, height = get_text_size(text)
    if width <= max_width:
        # SRT las entradas permiten saltos de línea manuales del autor. Aunque todo el texto no necesite volver a partirse por ancho,
        # la altura del lienzo debe calcularse con el número de líneas existente, o la segunda línea y las siguientes quedarían recortadas.
        return direction_mark + text, (text.count("\n") + 1) * line_height

    def attached_text_units(token):
        units = []
        for char in token:
            if units and unicodedata.category(char).startswith("M"):
                units[-1] += char
            else:
                units.append(char)
        return units

    def split_long_token(token):
        # Cuando un token por sí solo ya es demasiado ancho (habitual en frases largas en chino sin espacios o palabras inglesas muy largas),
        # se degrada a división por caracteres. La clave es: al detectar que candidate es demasiado ancho, primero se confirma el
        # current anterior que aún es válido y luego el carácter actual se pone en la línea siguiente; no se puede meter el carácter demasiado ancho en la línea anterior.
        lines = []
        current = ""
        for char in attached_text_units(token):
            candidate = f"{current}{char}"
            candidate_width, _ = get_text_size(candidate)
            if candidate_width <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = char
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    words = text.split(" ")
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        word_width, _ = get_text_size(word)
        if word_width <= max_width:
            current = word
        else:
            # el último trozo aún no llena la línea, y las palabras siguientes deben seguir intentando ir detrás de él,
            # o tras cada palabra demasiado larga sobraría una línea.
            pieces = split_long_token(word)
            lines.extend(pieces[:-1])
            current = pieces[-1] if pieces else ""

    if current:
        lines.append(current)

    line_start_punctuation = "，。！？；：、,.!?;:)]}）】》」』”’"
    # Las palabras en alfabeto árabe no se pueden partir (rompería la unión de letras), así que no se hace el ajuste de puntuación de abajo.
    for index in range(1, 1 if shaped else len(lines)):
        # Al partir por caracteres una frase larga en chino, el último signo de cierre como el punto o la coma puede quedar solo
        # en la línea siguiente, estirando de forma anómala el fondo del subtítulo y pareciendo un puntito caído bajo
        # el texto. Aquí, sin rediseñar el algoritmo de salto de línea, se mueve el último carácter de la línea anterior
        # delante de la línea de puntuación, para que la puntuación acompañe al texto, compatible con los signos de cierre habituales en chino e inglés.
        if not lines[index] or lines[index][0] not in line_start_punctuation:
            continue
        if len(lines[index - 1]) <= 1:
            continue

        previous_units = attached_text_units(lines[index - 1])
        if len(previous_units) <= 1:
            continue
        last_unit = previous_units[-1]
        candidate = f"{last_unit}{lines[index]}"
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            lines[index] = candidate
            lines[index - 1] = lines[index - 1][:-len(last_unit)]

    result = "\n".join(
        direction_mark + line.strip() for line in lines if line.strip()
    ).strip()
    # La altura se basa en el resultado final. Los saltos de línea explícitos del texto original pueden quedar dentro de algún token,
    # y en ese caso la longitud de la lista temporal lines no coincide con el número de líneas que renderiza realmente MoviePy.
    height = (result.count("\n") + 1) * line_height
    return result, height


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    # Match the Pillow color syntax accepted by subtitle preflight.
    # The background clip sets opacity separately from the color value.
    if isinstance(color, str):
        try:
            return ImageColor.getcolor(color, "RGB")
        except (ValueError, TypeError):
            pass
    return (0, 0, 0)


def _rounded_subtitle_background_clip(
    width: int,
    height: int,
    color: str,
    alpha: int = 140,
    radius: int = 16,
) -> ImageClip:
    # El nuevo fondo de subtítulos solo se usa cuando el usuario lo activa explícitamente: se dibuja un panel semitransparente con esquinas redondeadas como imagen RGBA,
    # y luego se entrega a MoviePy como ImageClip transparente para la composición. Así la ruta por defecto no cambia en absoluto,
    # y se pueden probar a bajo coste efectos visuales de subtítulos más suaves.
    rgb = _hex_to_rgb(color)
    safe_alpha = max(0, min(255, int(alpha)))
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, max(0, width - 1), max(0, height - 1)],
        radius=max(0, int(radius)),
        fill=(rgb[0], rgb[1], rgb[2], safe_alpha),
    )
    return ImageClip(np.array(img), transparent=True)


def _get_visible_center_position(
    text_clip: TextClip,
    container_width: int,
    container_height: int,
) -> tuple[int, int]:
    """
    Coloca el TextClip en el centro del contenedor de fondo según los píxeles realmente visibles del texto.

    MoviePy El TextClip de crea un lienzo transparente según la altura de línea y la baseline de la fuente. En muchas fuentes
    los glifos visibles no están en el centro geométrico de ese lienzo, y directamente `with_position("center")`
    centraría todo el lienzo transparente, haciendo que el subtítulo parezca desplazado hacia arriba o abajo. Aquí se lee TextClip
    la mask transparente y se calcula el desplazamiento solo según el bbox con píxeles reales, para que el texto que ve el usuario
    quede visualmente centrado en el fondo del subtítulo.
    """
    x = int(round((container_width - text_clip.w) / 2))
    y = int(round((container_height - text_clip.h) / 2))

    try:
        if text_clip.mask is None:
            return x, y

        mask_frame = text_clip.mask.get_frame(0)
        ys, _ = np.where(mask_frame > 0.01)
        if len(ys) == 0:
            return x, y

        visible_top = int(ys.min())
        visible_bottom = int(ys.max())
        visible_height = visible_bottom - visible_top + 1
        y = int(round((container_height - visible_height) / 2 - visible_top))
    except Exception as exc:
        logger.debug(f"failed to center subtitle text by visible mask: {str(exc)}")

    return x, y


def validate_subtitle_colors(params: VideoParams) -> None:
    """Reject colors Pillow cannot render, before generation services run."""
    if not params.subtitle_enabled:
        return
    colors = [("text_fore_color", params.text_fore_color)]
    # Rendering truncates the width to an integer; a zero-width stroke is unused.
    if int(params.stroke_width) > 0:
        colors.append(("stroke_color", params.stroke_color))
    if isinstance(params.text_background_color, str):
        colors.append(("text_background_color", params.text_background_color))
    for field, value in colors:
        if value is None:
            continue
        try:
            ImageColor.getcolor(value, "RGBA")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid subtitle color: {field}") from exc


def subtitle_colors_are_indistinguishable(params: VideoParams) -> bool:
    """Determina si el texto y el fondo del subtítulo son del mismo color, para avisar al usuario de que quizá no se lea el subtítulo."""
    if not params.subtitle_enabled or not params.text_background_color:
        return False

    def normalize_color(value):
        if isinstance(value, bool):
            return "#000000" if value else ""
        return str(value or "").strip().lower()

    text_color = normalize_color(params.text_fore_color)
    background_color = normalize_color(params.text_background_color)
    return bool(text_color and text_color == background_color)


@lru_cache(maxsize=64)
def _subtitle_font_supports_sample(font_path: str, sample: str) -> bool:
    """Comprueba si la fuente contiene los glifos necesarios para el texto de muestra, y guarda en caché los resultados de comprobaciones repetidas."""
    try:
        font = ImageFont.truetype(font_path, 30)
        missing_mask = font.getmask("\U0010ffff")
        missing_signature = (
            missing_mask.size,
            missing_mask.getbbox(),
            bytes(missing_mask),
        )
        for char in sample:
            char_mask = font.getmask(char)
            char_signature = (
                char_mask.size,
                char_mask.getbbox(),
                bytes(char_mask),
            )
            if char_mask.getbbox() is None or char_signature == missing_signature:
                return False
        return True
    except Exception as e:
        # Un fallo al detectar la fuente no debe impedir que el usuario genere; se conserva el log para investigar problemas de compatibilidad del entorno.
        logger.warning(f"failed to inspect subtitle font glyphs: {font_path}, {e}")
        return True


def subtitle_font_supports_text(font_path: str, text: str) -> bool:
    """Comprueba si la fuente puede dibujar las letras y números del texto, ignorando espacios y signos de puntuación."""
    sample = "".join(
        dict.fromkeys(
            char
            for char in str(text or "")
            if unicodedata.category(char)[0] in {"L", "N"}
        )
    )[:64]
    if not sample:
        return True
    if subtitle_shaping.needs_shaping(sample):
        # El texto en alfabeto árabe lo dibuja khatt, y si a la fuente elegida le faltan caracteres se usa la de respaldo:
        # se determina si la fuente usada realmente para dibujar puede dibujar todas las letras y números.
        return subtitle_shaping.supports_text(font_path, sample)
    return _subtitle_font_supports_sample(font_path, sample)


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
    bgm_file_override: str | None = None,
) -> bool:
    """
    Compone el video final y devuelve si el procesamiento de la música de fondo tuvo éxito.

    El valor devuelto solo describe el estado del procesamiento de BGM: si no se pidió BGM o se mezcló con éxito devuelve True; si se pidió
    BGM pero falló la carga, el efecto o la mezcla devuelve False. Aunque la BGM falle se sigue generando un video solo con
    la narración, y la capa de orquestación de tareas decide si mostrar al usuario una advertencia de degradación.
    """
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generating video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # https://github.com/harry0703/MoneyPrinterTurbo/issues/217
    # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'final-1.mp4.tempTEMP_MPY_wvf_snd.mp3'
    # write into the same directory as the output file
    output_dir = os.path.dirname(output_file)

    font_path = ""
    if params.subtitle_enabled or params.opening_title:
        if not params.font_name:
            params.font_name = "STHeitiMedium.ttc"
        # API Aunque la entrada ya hizo la comprobación previa, la WebUI, el CLI y las llamadas internas pueden entrar directamente en la capa de renderizado;
        # siempre se valida con la ruta real que la fuente se quede en resource/fonts, bloqueando rutas absolutas,
        # ../ recorridos y enlaces simbólicos que apunten fuera del directorio, antes de entregarla a PIL/MoviePy para abrirla.
        font_path = file_security.resolve_path_within_directory(
            utils.font_dir(), params.font_name
        )
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

    def resolve_subtitle_background_color():
        # Compatibilidad con parámetros históricos: en la API, `text_background_color` puede ser un booleano
        # o una cadena de color real. Se normaliza aquí de forma unificada, para no pasar True/False
        # directamente al TextClip y obtener un renderizado imprevisible.
        if isinstance(params.text_background_color, bool):
            return "#000000" if params.text_background_color else None
        return params.text_background_color

    # La fuente de los subtítulos en alfabeto árabe se determina una vez para todos los subtítulos del video (ver donde se cargan los subtítulos),
    # para que los subtítulos de un mismo video no alternen entre dos fuentes.
    shaped_font = font_path

    def create_text_clip(subtitle_item):
        params.font_size = int(params.font_size)
        params.stroke_width = int(params.stroke_width)
        phrase = subtitle_item[1]
        shaped = subtitle_shaping.needs_shaping(phrase)
        clip_font = shaped_font if shaped else font_path
        clip_fallbacks = subtitle_shaping.fallback_fonts(shaped_font, font_path) if shaped else []
        max_width = video_width * 0.9
        bg_color = resolve_subtitle_background_color()
        rounded_bg_enabled = bool(
            getattr(params, "rounded_subtitle_background", False) and bg_color
        )
        has_subtitle_background = bool(bg_color)
        # El fondo redondeado se genera según el ancho real del texto, con márgenes laterales más contenidos; el fondo rectangular antiguo mantiene
        # un margen de seguridad mayor, para que los subtítulos largos de configuraciones antiguas no toquen el borde ni se recorten.
        padding_ratio = 0.4 if rounded_bg_enabled else 0.6
        pad_x = int(params.font_size * padding_ratio) if has_subtitle_background else 0
        # El fondo del subtítulo necesita un relleno lateral claro para el texto. Primero se resta del ancho disponible
        # padding y luego se hace el salto de línea, para que el inglés largo o los tamaños grandes que llenan justo el 90% del ancho del video
        # no peguen el texto al borde del fondo, como si estuviera recortado. Tanto el fondo rectangular normal como el redondeado
        # siguen esta lógica; los subtítulos sin fondo mantienen el ancho máximo original.
        text_max_width = max(1, int(max_width) - 2 * pad_x)
        wrapped_txt, txt_height = wrap_text(
            phrase,
            max_width=text_max_width,
            font=clip_font,
            fontsize=params.font_size,
            fallbacks=clip_fallbacks,
        )
        interline = int(params.font_size * 0.25)
        line_count = wrapped_txt.count("\n") + 1
        vertical_padding = int(params.font_size * 0.35)
        # Pillow/MoviePy expande el contorno hacia arriba y abajo de los glifos, y cuenta esa parte en la altura de avance de cada línea
        # . Si solo se añadiera un margen de contorno una vez fuera de todo el bloque, un texto multilínea con contorno grueso
        # seguiría acumulando error línea a línea. Aquí se cuenta el espacio del contorno a ambos lados según el número real de líneas; el contorno fino por defecto solo
        # añade un poco de altura, y «tamaño pequeño + contorno grueso + varias líneas» también se muestra completo.
        stroke_padding = int(params.stroke_width * 2 * line_count)
        text_clip_margin_y = max(
            int(params.font_size * 0.3), int(params.stroke_width * 2)
        )
        # MoviePy con `method=label` reduce automáticamente la altura del cuadro de texto, y con subtítulos multilínea,
        # contornos o color de fondo tiende a recortar la mitad inferior de la última línea. Aquí se pasa explícitamente
        # una altura más conservadora, que incluye el interlineado y márgenes adicionales arriba y abajo, para que tanto el fondo
        # del subtítulo como el propio texto se rendericen completos.
        clip_h = int(
            txt_height
            + vertical_padding
            + (interline * line_count)
            + stroke_padding
        )

        if rounded_bg_enabled:
            # El fondo redondeado debe ajustarse al ancho del texto, en lugar de usar el 90% del ancho del video. Aquí primero se usa
            # PIL para medir la línea de texto más larga y se añade el relleno horizontal, evitando paneles demasiado anchos en subtítulos cortos.
            try:
                if shaped:
                    # El texto en alfabeto árabe se mide según los glifos modelados y la fuente de dibujo real.
                    text_w = max(
                        subtitle_shaping.text_width(
                            line, clip_font, params.font_size, clip_fallbacks
                        )
                        for line in wrapped_txt.split("\n")
                    )
                else:
                    font = ImageFont.truetype(font_path, params.font_size)
                    text_w = max(
                        int(font.getbbox(line)[2] - font.getbbox(line)[0])
                        for line in wrapped_txt.split("\n")
                    )
            except Exception as exc:
                logger.warning(
                    f"failed to measure subtitle text width, fallback to max width: {str(exc)}"
                )
                text_w = int(max_width)

            box_w = max(1, min(int(max_width), text_w + 2 * pad_x))
            radius = max(8, int(params.font_size * 0.4))
            text_clip = subtitle_shaping.text_clip(
                TextClip,
                shaped_font=shaped_font,
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(box_w, None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            clip_h = max(clip_h, text_clip.h)
            bg_clip = _rounded_subtitle_background_clip(
                width=box_w,
                height=clip_h,
                color=bg_color,
                alpha=140,
                radius=radius,
            )
            text_position = _get_visible_center_position(text_clip, box_w, clip_h)
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=(box_w, clip_h),
            )
        elif bg_color:
            size = (
                int(max_width),
                clip_h,
            )
            text_clip = subtitle_shaping.text_clip(
                TextClip,
                shaped_font=shaped_font,
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(int(max_width), None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            size = (size[0], max(size[1], text_clip.h))
            bg_clip = _rounded_subtitle_background_clip(
                width=size[0],
                height=size[1],
                color=bg_color,
                alpha=255,
                radius=0,
            )
            text_position = _get_visible_center_position(text_clip, size[0], size[1])
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=size,
            )
        else:
            size = (
                int(max_width),
                clip_h,
            )
            if shaped:
                # khatt para que el lienzo contenga toda la tinta (por ejemplo, los signos de pausa coránicos que sobresalen por encima de la fuente),
                # fijar la altura con la fórmula de arriba podría recortarlos, así que primero se mide la altura natural.
                with subtitle_shaping.text_clip(
                    TextClip,
                    shaped_font=shaped_font,
                    text=wrapped_txt,
                    font=font_path,
                    font_size=params.font_size,
                    stroke_width=params.stroke_width,
                    interline=interline,
                    size=(size[0], None),
                    text_align="center",
                ) as natural:
                    size = (size[0], max(size[1], natural.h))
            _clip = subtitle_shaping.text_clip(
                TextClip,
                shaped_font=shaped_font,
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=size,
                text_align="center",
            )
        duration = subtitle_item[0][1] - subtitle_item[0][0]
        _clip = _clip.with_start(subtitle_item[0][0])
        _clip = _clip.with_end(subtitle_item[0][1])
        _clip = _clip.with_duration(duration)

        # La animación de rebote solo se activa cuando el usuario la elige explícitamente; el none por defecto sigue exactamente la ruta de renderizado de subtítulos original.
        anim_type = getattr(params, "subtitle_animation", "none")
        if anim_type in ("pop_spring", "spring", "pop"):
            _clip = _apply_subtitle_spring_animation(_clip, duration)

        if params.subtitle_position == "bottom":
            _clip = _clip.with_position(("center", video_height * 0.95 - _clip.h))
        elif params.subtitle_position == "top":
            _clip = _clip.with_position(("center", video_height * 0.05))
        elif params.subtitle_position in ("two_thirds_bottom", "two_thirds", "2/3_bottom"):
            # 2/3 from the bottom = 1/3 from the top: y = (video_height - _clip.h) * (1/3)
            y_two_thirds = (video_height - _clip.h) / 3.0
            _clip = _clip.with_position(("center", y_two_thirds))
        elif params.subtitle_position == "custom":
            # Ensure the subtitle is fully within the screen bounds
            margin = 10  # Additional margin, in pixels
            max_y = video_height - _clip.h - margin
            min_y = margin
            custom_y = (video_height - _clip.h) * (params.custom_position / 100)
            custom_y = max(
                min_y, min(custom_y, max_y)
            )  # Constrain the y value within the valid range
            _clip = _clip.with_position(("center", custom_y))
        else:  # center
            _clip = _clip.with_position(("center", "center"))
        return _clip

    # MoviePy El CompositeAudioClip.close() de no cierra los AudioFileClip hijos. Aquí se usa
    # ExitStack para mantener explícitamente todos los readers de archivos originales, garantizando que en éxito, excepción de subtítulos, fallo de mezcla y
    # fallo de escritura del video se liberen los subprocesos de FFmpeg, sobre todo para evitar archivos bloqueados en Windows.
    with ExitStack() as clip_stack:
        source_video_clip = clip_stack.enter_context(
            _open_video_clip_quietly(video_path)
        )
        voice_source_clip = clip_stack.enter_context(AudioFileClip(audio_path))
        video_clip = source_video_clip
        audio_clip = voice_source_clip.with_effects(
            [afx.MultiplyVolume(1.0 if params.voice_volume is None else params.voice_volume)]
        )

        def make_textclip(text):
            return subtitle_shaping.text_clip(
                TextClip,
                shaped_font=shaped_font,
                text=text,
                font=font_path,
                font_size=params.font_size,
            )

        if params.subtitle_enabled and subtitle_path and os.path.exists(subtitle_path):
            subtitles = split_long_cues(
                moviepy_subtitles(subtitle_path),
                int(config.ui.get("subtitle_max_words", 0) or 0),
            )
            shaped_font = subtitle_shaping.video_font(
                font_path, (text for _, text in subtitles)
            )
            sub = clip_stack.enter_context(
                SubtitlesClip(
                    subtitles=subtitles,
                    encoding="utf-8",
                    make_textclip=make_textclip,
                )
            )
            text_clips = []
            for item in sub.subtitles:
                clip = create_text_clip(subtitle_item=item)
                text_clips.append(clip)
            video_clip = CompositeVideoClip([video_clip, *text_clips])
            clip_stack.callback(video_clip.close)

        if params.opening_title.strip():
            # Overlay at t=0 on the protagonist's photo. It adds no silent intro
            # and never shifts the narration or its subtitle timestamps.
            title_size = max(24, int(video_width * 0.072))
            title_font = subtitle_shaping.video_font(font_path, [params.opening_title])
            title = clip_stack.enter_context(subtitle_shaping.text_clip(
                TextClip, shaped_font=title_font, font=font_path,
                text=params.opening_title, font_size=title_size,
                color="#FFFFFF", stroke_color="#000000", stroke_width=3,
                method="caption", size=(int(video_width * 0.88), None),
                text_align="center", margin=(12, 18),
            ))
            title = title.with_start(0).with_duration(
                min(params.opening_title_duration, voice_source_clip.duration, source_video_clip.duration)
            ).with_position(("center", int(video_height * 0.08)))
            video_clip = CompositeVideoClip([video_clip, title])
            clip_stack.callback(video_clip.close)

        bgm_enabled = bgm_service.should_use_bgm(
            params.bgm_type, params.bgm_volume
        )
        if not bgm_enabled and params.bgm_type:
            # Todos los orígenes de BGM comparten esta regla de cortocircuito. Con volumen no mayor que 0 no se pueden resolver archivos aleatorios o
            # personalizados, ni cargar los archivos devueltos por los proveedores, para evitar E/S y mezclas sin sentido.
            logger.info(
                f"skipping background music because volume is not positive: "
                f"type={params.bgm_type}, volume={params.bgm_volume}"
            )

        # La música de proveedores la puede pasar directamente la capa de orquestación con el archivo correspondiente. None significa seguir la resolución aleatoria/personalizada
        # BGM , y una cadena vacía desactiva explícitamente la BGM de esta tarea; pero cualquier origen debe pasar primero la regla general de volumen.
        bgm_file = ""
        if bgm_enabled:
            bgm_file = (
                bgm_file_override
                if bgm_file_override is not None
                else get_bgm_file(
                    bgm_type=params.bgm_type,
                    bgm_file=params.bgm_file,
                )
            )
        bgm_mix_succeeded = True
        if bgm_file:
            try:
                bgm_effects = [afx.MultiplyVolume(params.bgm_volume)]
                # La música aleatoria/personalizada resuelta dentro del servicio puede ser más corta que el video y debe repetirse en bucle; el archivo que la capa de tareas
                # pasa por override indica que el proveedor ya adaptó la duración. Aquí se decide según
                # el origen del archivo si hacer bucle, para no tener que modificar una lista blanca de nombres cada vez que se añada un proveedor.
                if bgm_file_override is None:
                    bgm_effects.append(afx.AudioLoop(duration=video_clip.duration))
                # Fade the complete playback timeline, not each source loop.
                bgm_effects.append(afx.AudioFadeOut(3))
                bgm_source_clip = clip_stack.enter_context(AudioFileClip(bgm_file))
                bgm_clip = bgm_source_clip.with_effects(bgm_effects)
                audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
            except Exception:
                bgm_mix_succeeded = False
                # Registra la pila completa y un contexto estable, para distinguir entre decodificación del archivo, efectos de MoviePy y
                # CompositeAudioClip fallos; el contenido de los archivos y la API Key no entran en el log.
                logger.exception(
                    f"failed to mix background music: type={params.bgm_type}, "
                    f"file={bgm_file}"
                )

        final_video_clip = video_clip.with_audio(audio_clip)
        clip_stack.callback(final_video_clip.close)
        # Se usa explícitamente la frecuencia de muestreo del audio de entrada; si no se puede obtener, se recurre a los 44100Hz por defecto de MoviePy.
        # Así se reducen las fluctuaciones de calidad de audio que trae el remuestreo en distintos entornos, sobre todo en Docker.
        output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)
        with _stage_heartbeat("final video render"):
            _write_videofile_with_codec_fallback(
                final_video_clip,
                output_file=output_file,
                codec=_get_configured_video_codec(),
                atomic_output=True,
                audio_codec=audio_codec,
                audio_fps=output_audio_fps,
                audio_bitrate=audio_bitrate,
                temp_audiofile_path=_get_temp_audio_dir(output_dir),
                threads=params.n_threads or 2,
                logger=None,
                fps=fps,
            )
        return bgm_mix_succeeded


def render_image_zoom_video(
    image_path: str, clip_duration: int = 5, image_motion: ImageMotion = ImageMotion.zoom
) -> str:
    """
    Renderiza una sola imagen como clip mp4 estático o con zoom lento, y devuelve la ruta del archivo de salida.

    local El preprocesamiento de material y el material de texto a imagen compatible con OpenAI comparten este renderizado "imagen → clip"
    : por defecto se mantiene un zoom dinámico de aproximadamente un 3% por segundo, y el modo estático solo fija la duración de reproducción.
    Esta opción es independiente de las transiciones de clip. Las excepciones de renderizado las trata el llamador según la convención de fallo de cada fuente de material.
    """
    motion = ImageMotion(image_motion)
    clip, _ = _open_image_clip_with_fallback(image_path)
    clip = clip.with_duration(clip_duration).with_position("center")
    temp_path = ""
    try:
        # Solo el modo zoom cambia el tamaño de la imagen; el modo estático conserva la composición original y el tratamiento del fondo transparente.
        image_clip = (
            clip.resized(lambda t: 1 + (clip_duration * 0.03) * (t / clip.duration))
            if motion == ImageMotion.zoom
            else clip
        )

        # Optionally, create a composite video clip containing the zoomed clip.
        # This is useful if you want to add other elements to the video.
        final_clip = CompositeVideoClip([image_clip], bg_color=(0, 0, 0))
        try:
            # The duration changes the rendered content, so it must be part of
            # the output identity. Different tasks may render the same image
            # concurrently; only publish a complete MP4 after MoviePy closes it.
            source_digest = hashlib.sha256(os.fsencode(os.path.abspath(image_path)))
            source_frame = clip.get_frame(0)
            source_digest.update(str(source_frame.shape).encode())
            source_digest.update(source_frame.tobytes())
            if clip.mask is not None:
                source_digest.update(clip.mask.get_frame(0).tobytes())
            source_identity = source_digest.hexdigest()
            duration_identity = hashlib.sha256(str(clip_duration).encode()).hexdigest()[:16]
            video_file = os.path.join(
                os.path.dirname(image_path),
                f"{source_identity}.{motion.value}-{duration_identity}.mp4",
            )
            descriptor, temp_path = tempfile.mkstemp(
                prefix=".image-zoom-",
                suffix=".mp4",
                dir=os.path.dirname(os.path.abspath(video_file)),
            )
            os.close(descriptor)
            final_clip.write_videofile(
                temp_path, fps=30, logger=None, ffmpeg_params=_BT709_FFMPEG_PARAMS
            )
        finally:
            close_clip(final_clip)
        os.replace(temp_path, video_file)
        temp_path = ""
        return video_file
    finally:
        close_clip(clip)
        if temp_path:
            delete_files(temp_path)


def preprocess_video(
    materials: List[MaterialInfo], clip_duration=4, image_motion: ImageMotion = ImageMotion.zoom
):
    # WebUI En algunos escenarios de segunda generación puede llegar una lista de material vacía; aquí se devuelve directamente un resultado vacío, para no lanzar una excepción NoneType.
    if not materials:
        return []

    # Solo devuelve el material que pasó la validación del preprocesamiento, para que imágenes de baja resolución no sigan al flujo posterior de composición de video.
    valid_materials = []
    local_videos_dir = utils.storage_dir("local_videos", create=True)

    for material in materials:
        if not material.url:
            continue

        try:
            material_source_path = file_security.resolve_path_within_directory(
                local_videos_dir, material.url
            )
        except ValueError as exc:
            # local video_source Las rutas de material vienen de parámetros de la API y deben limitarse al directorio de material dedicado.
            # permite que el usuario pase nombres de archivo, y admite las rutas absolutas devueltas históricamente, pero no permite escapar a otros directorios del sistema
            # , para evitar la lectura de archivos arbitrarios o el sondeo de archivos locales sensibles a través de MoviePy.
            logger.warning(
                f"skip unsafe local material: {material.url}, "
                f"local_videos_dir: {local_videos_dir}, error: {str(exc)}"
            )
            continue

        ext = utils.parse_extension(material_source_path)
        is_image = ext in const.FILE_TYPE_IMAGES
        try:
            # El material de imagen se lee directamente como imagen, para no pasar primero por VideoFileClip, equivocarse y activar una rama de respaldo inestable.
            if is_image:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            else:
                clip = _open_video_clip_quietly(material_source_path)
        except Exception:
            # Con extensiones no estándar o si falla la detección se recurre al modo imagen, por compatibilidad con el paso histórico de rutas de imágenes locales.
            try:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
                # The successful decoder determines the material kind, even
                # when the uploaded filename has a video or unknown suffix.
                is_image = True
            except Exception as exc:
                logger.warning(
                    f"skip unreadable local material: {material.url}, error: {str(exc)}"
                )
                continue
        try:
            width = clip.size[0]
            height = clip.size[1]
            if not is_material_resolution_acceptable(width, height):
                logger.warning(
                    f"low resolution material: {width}x{height}, minimum "
                    f"{_MIN_MATERIAL_DIMENSION}x{_MIN_MATERIAL_DIMENSION} required "
                    f"(tolerance {_MIN_DIMENSION_TOLERANCE}px)"
                )
                # Al detectar material de baja resolución se cierran de inmediato los recursos y no se devuelve ese material al flujo posterior.
                close_clip(clip)
                continue

            if is_image:
                logger.info(f"processing image: {material_source_path}")
                # Al detectar el tamaño ya se abrió el material una vez; aquí primero se libera el handle de detección y luego se renderiza
                # Clip de imagen para exportar.
                close_clip(clip)
                video_file = render_image_zoom_video(
                    material_source_path, clip_duration, image_motion=image_motion
                )
                material.url = video_file
                logger.success(f"image processed: {video_file}")
            else:
                # Para el material de video normal solo hay que leer el tamaño para validarlo, y liberar el handle justo después de validar.
                close_clip(clip)
                # Update url to the resolved absolute path so that downstream
                # stages (combine_videos) can open the file without re-resolving.
                material.url = material_source_path
        except Exception:
            close_clip(clip)
            raise

        valid_materials.append(material)

    return valid_materials
