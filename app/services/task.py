import json
import math
import os
import re
import socket
import tempfile
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from functools import partial
from os import path
from uuid import uuid4

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams
from app.services import bgm as bgm_service
from app.services import (
    codex_image,
    codex_oauth,
    elevenlabs_music,
    generation_progress,
    infistar_video,
    jalapeno_video,
    llm,
    loomloom,
    material,
    metaso_minimax,
    muapi,
    ofox,
    sonilo,
    subtitle,
    task_artifacts,
    twelvelabs,
    video,
    volcengine_seedance,
    voice,
    wikimedia,
)
from app.services import upload_post
from app.services import state as sm
from app.utils import file_security, utils
from app.utils.subtitle_writer import staged_subtitle_file


# Una petición de publicación puede esperar hasta varios minutos y no puede seguir ocupando el cupo de concurrencia de las tareas de generación de video.
# Un pool de hilos de tamaño fijo limita el rendimiento de publicación a un rango controlable, y permite que, una vez generado el video,
# pase de inmediato al estado completado.
_cross_post_executor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="mpt-cross-post",
)
_cross_post_max_pending_tasks = max(
    1,
    int(config.app.get("upload_post_max_pending_tasks", 10)),
)
_cross_post_slots = threading.BoundedSemaphore(_cross_post_max_pending_tasks)
_cross_post_registry_lock = threading.RLock()
_cross_post_futures: dict[str, Future] = {}
_cross_post_process_owner = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex}"
_ACTIVE_CROSS_POST_STATES = {
    const.CROSS_POST_STATE_PENDING,
    const.CROSS_POST_STATE_PROCESSING,
}
_CROSS_POST_STATE_WRITE_ATTEMPTS = 3
_CROSS_POST_STATE_RETRY_DELAY_SECONDS = 0.1
_TASK_FAILURE_STATE_WRITE_ATTEMPTS = 3
_TASK_FAILURE_STATE_RETRY_DELAY_SECONDS = 0.1
_LOOMLOOM_STATE_WRITE_ATTEMPTS = 3
_LOOMLOOM_STATE_RETRY_DELAY_SECONDS = 0.1
_INTERRUPTED_CROSS_POST_ERROR = (
    "cross-posting was interrupted before the process completed"
)
# Map upload-post platform ids to the social platform names llm.py accepts.
_CROSS_POST_SOCIAL_PLATFORMS = {
    "tiktok": "tiktok",
    "instagram": "instagram_reels",
    "facebook": "facebook_reels",
}
# Los servicios de música para video solo tienen que implementar ``is_enabled`` y ``generate_bgm``. Las diferencias entre proveedores se concentran en
# la extensión de archivo, las excepciones de dominio y los códigos de advertencia de la WebUI; la orquestación de tareas, el cortocircuito de volumen 0 y la degradación ante fallos
# reutilizan todos la misma ruta, para no mantener varios flujos parecidos al añadir proveedores.
_VIDEO_MUSIC_PROVIDERS = {
    "sonilo": {
        "service": sonilo,
        "error_type": sonilo.SoniloError,
        "suffix": ".m4a",
        "warning_code": "sonilo_bgm_failed",
        "display_name": "Sonilo",
    },
    "elevenlabs": {
        "service": elevenlabs_music,
        "error_type": elevenlabs_music.ElevenLabsMusicError,
        "suffix": ".mp3",
        "warning_code": "elevenlabs_bgm_failed",
        "display_name": "ElevenLabs",
    },
}


_SUPPORTED_VIDEO_SOURCES = frozenset({
    "pexels", "pixabay", "coverr", "local", "wavespeed",
    "volcengine_seedance", "ofox", "metaso_minimax", "muapi",
    "loomloom", "openai_image", "codex_image", "infistar", "jalapeno", "wikimedia",
})


def _get_video_music_prompt(params: VideoParams) -> str:
    """
    Lee el prompt que usa realmente el proveedor de música para video actual.

    Las nuevas tareas usan siempre campos independientes del proveedor; los parámetros antiguos de Sonilo CLI y las tareas históricas pueden tener solo
    ``sonilo_bgm_prompt``, así que solo se lee el campo antiguo cuando el campo genérico de Sonilo está vacío.
    """
    prompt = str(params.video_music_prompt or "").strip()
    if params.bgm_type == "sonilo" and not prompt:
        prompt = str(params.sonilo_bgm_prompt or "").strip()
    return prompt


def is_task_busy(task: dict | None) -> bool:
    """Determina si la tarea aún se está generando o publicando, reutilizado por todas las entradas de borrado."""
    if not task:
        return False

    state = task.get("state")
    try:
        state = int(state)
    except (TypeError, ValueError):
        # The function's own docstring promises to treat unparseable
        # states as busy, so the delete API keeps a task that we
        # cannot reason about. Returning False here would let
        # delete_video() cancel a task that is still running or
        # cross-posting whenever an older Redis snapshot or a
        # legacy call site stored the state as a string.
        return True

    # Tanto la generación de video como la publicación multiplataforma pueden seguir leyendo el directorio de la tarea. Considerarlas ocupadas de forma unificada
    # evita que, si la API y la WebUI mantuvieran reglas por separado, una permitiera borrar y la otra lo prohibiera
    # , con un comportamiento incoherente.
    return (
        state == const.TASK_STATE_PROCESSING
        or task.get("cross_post_state") in _ACTIVE_CROSS_POST_STATES
    )


def _register_cross_post_future(task_id: str, future: Future) -> None:
    """Registra los Future de publicación que mantiene el proceso actual, para que la recuperación al arrancar y las pruebas determinen el estado real de ejecución."""
    with _cross_post_registry_lock:
        _cross_post_futures[task_id] = future


def _unregister_cross_post_future(task_id: str, future: Future | None = None) -> None:
    """Solo elimina el Future coincidente, para que callbacks antiguos no borren por error trabajo nuevo registrado después para la misma tarea."""
    with _cross_post_registry_lock:
        current = _cross_post_futures.get(task_id)
        if current is None or (future is not None and current is not future):
            return
        _cross_post_futures.pop(task_id, None)


def _is_cross_post_active_in_process(task_id: str) -> bool:
    """Determina si el proceso actual aún mantiene tareas de publicación sin terminar."""
    with _cross_post_registry_lock:
        future = _cross_post_futures.get(task_id)
        return future is not None and not future.done()


def _is_windows_process_alive(process_id: int) -> bool:
    """Determina el estado del proceso con una API Win32 de solo lectura, para no terminar procesos por error con os.kill."""
    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5
    error_invalid_parameter = 87
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # ctypes por defecto trata los valores de retorno no declarados como int de 32 bits. Los handles de procesos de Windows de 64 bits podrían
    # truncarse por ello, así que hay que declarar explícitamente la firma de la función Win32 antes de llamarla.
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        process_id,
    )
    if not handle:
        error_code = ctypes.get_last_error()
        if error_code == error_invalid_parameter:
            return False
        if error_code == error_access_denied:
            # Si el proceso existe pero el usuario actual no tiene permiso de consulta, hay que considerarlo vivo de forma conservadora, para no
            # reclamar por error tareas de publicación que otra cuenta está ejecutando.
            return True
        logger.warning(
            "failed to open cross-post owner process on Windows, "
            f"process_id: {process_id}, error_code: {error_code}"
        )
        return True

    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            error_code = ctypes.get_last_error()
            logger.warning(
                "failed to read cross-post owner process state on Windows, "
                f"process_id: {process_id}, error_code: {error_code}"
            )
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _is_cross_post_owner_alive(owner: str | None) -> bool:
    """Determina si el proceso local de una tarea de publicación persistida sigue existiendo."""
    if not owner:
        return False

    try:
        hostname, process_id_text, _ = owner.split(":", 2)
        process_id = int(process_id_text)
    except (TypeError, ValueError):
        logger.warning(f"invalid cross-post owner metadata: {owner}")
        return False

    # No se pueden detectar de forma fiable procesos en otros hosts. En despliegues multi-host con Redis compartido hay que
    # considerarlos en ejecución de forma conservadora, para que el nodo actual no borre por error archivos de video que otro nodo está leyendo.
    if hostname != socket.gethostname():
        return True

    # Si el proceso actual aún tiene trabajo de publicación real ya lo determina con precisión el registro de Future. Llegar
    # aquí significa que el registro no tiene un Future correspondiente; aunque el owner coincida exactamente con el proceso actual, debe
    # considerarse interrumpido; esto cubre los casos en que la escritura del estado final falla continuamente y el Future ya terminó.
    if process_id == os.getpid():
        return False

    # Windows El os.kill(pid, 0) de tiene una semántica distinta a POSIX y podría terminar directamente el proceso objetivo.
    # Se usa una API Win32 que solo solicita permiso de consulta, sin enviar ninguna señal al proceso objetivo.
    if os.name == "nt":
        return _is_windows_process_alive(process_id)

    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        logger.warning(
            f"failed to inspect cross-post owner process, owner: {owner}, error: {exc}"
        )
        return True
    return True


def _mark_task_failed(
    task_id: str,
    stage: str,
    error: str,
    details: dict | None = None,
) -> dict:
    """Registra información de fallo estructurada, conservando el progreso que la tarea había alcanzado antes de fallar."""
    existing_task = None
    try:
        existing_task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.warning(f"failed to read task state before failure update: {exc}")

    # Las funciones de servicio concretas suelen tener un motivo de error más preciso que la capa de orquestación. La comprobación posterior de resultado vacío
    # no puede sobrescribirlo con un texto genérico, o el llamador de la API seguiría viendo solo información vaga.
    if (
        existing_task
        and existing_task.get("state") == const.TASK_STATE_FAILED
        and existing_task.get("error")
    ):
        return existing_task

    message = str(error or "unknown task error").strip()
    progress = int((existing_task or {}).get("progress", 0) or 0)
    logger.error(f"task failed, task_id: {task_id}, stage: {stage}, error: {message}")
    failure = {
        "task_id": task_id,
        "state": const.TASK_STATE_FAILED,
        "progress": progress,
        "failed_stage": stage,
        "error": message,
    }
    # Algunas tareas externas ya crearon un ID remoto útil para recuperar o diagnosticar. El estado de fallo debe conservar
    # estos campos no sensibles, pero sin permitir que el llamador sobrescriba la estructura unificada de estado, progreso y error.
    # Keep completed batch outputs in both the persisted state and the direct
    # return value used by the CLI. These fields are reset before a new attempt.
    failure_details = {
        key: existing_task[key]
        for key in ("error_key", "error_params", "videos", "combined_videos", "warnings", "failed_video_index",
                    "generation_activity", "generation_started_at", "generation_submitted_at")
        if existing_task and key in existing_task
    }
    failure_details.update({
        key: value for key, value in dict(details or {}).items() if key not in failure
    })
    failure.update(failure_details)
    fields = {key: value for key, value in failure.items() if key != "task_id"}
    for attempt in range(_TASK_FAILURE_STATE_WRITE_ATTEMPTS):
        try:
            if attempt == 0:
                sm.state.update_task(task_id, **fields)
            else:
                # A lost acknowledgement may mean the first write succeeded.
                # Patch only an existing record so retries cannot revive a task
                # that the user has since deleted.
                sm.state.patch_task(task_id, **fields)
            break
        except Exception as exc:
            if attempt + 1 == _TASK_FAILURE_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    f"failed to persist task failure, task_id: {task_id}, error: {exc}"
                )
            else:
                time.sleep(_TASK_FAILURE_STATE_RETRY_DELAY_SECONDS)
    # A state outage must not discard the only completed-file snapshot held by
    # the caller. Return it even when all bounded persistence attempts fail.
    return failure


def _stage_progress_reporter(task_id: str, stage_start: float, stage_end: float):
    """
    Devuelve un callback que convierte la proporción completada de 0~1 dentro de la fase en progreso de la tarea y lo escribe en el estado.

    Antes el pipeline solo actualizaba el progreso en los límites de fase, y durante fases largas como descargar material o procesar fragmentos
    la barra de progreso no se movía en absoluto, sin que el usuario pudiera distinguir entre en ejecución y colgado. El progreso es solo información de visualización:
    si la proporción no se puede analizar se ignora, y si falla la escritura en el backend de estado solo se registra en el log, sin interrumpir la descarga ni
    la composición.
    """

    def report(fraction) -> None:
        try:
            fraction = max(0.0, min(1.0, float(fraction)))
        except (TypeError, ValueError):
            return
        try:
            sm.state.update_task(
                task_id,
                state=const.TASK_STATE_PROCESSING,
                progress=stage_start + (stage_end - stage_start) * fraction,
            )
        except Exception as exc:
            logger.warning(
                f"failed to update task progress: task_id={task_id}, "
                f"error={type(exc).__name__}, detail={exc}"
            )

    return report


def generate_script(task_id, params):
    logger.info("\n\n## generating video script")
    video_script = params.video_script.strip()
    if not video_script:
        video_script = llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
            video_script_prompt=params.video_script_prompt,
            custom_system_prompt=params.custom_system_prompt,
        )
    else:
        logger.debug(f"video script: \n{video_script}")

    if not video_script:
        _mark_task_failed(task_id, "script", "failed to generate video script")
        return None

    return video_script


def generate_terms(task_id, params, video_script):
    logger.info("\n\n## generating video terms")
    video_terms = params.video_terms
    if params.video_source == "wikimedia":
        # Keywords left over from stock/AI sources must not select generic scenes.
        return llm.generate_terms(
            video_subject=params.video_subject,
            video_script=utils.remove_pause_tags(video_script),
            amount=40,
            match_script_order=True,
            real_people=True,
        )
    if not video_terms:
        # Al activar la correspondencia del material en el orden del texto, las propias palabras clave también deben generarse en el orden narrativo del guion;
        # de lo contrario, aunque después se descargue y se una en orden, solo se podría reutilizar un conjunto de palabras clave globales,
        # sin resolver el problema de que «las imágenes del contenido posterior aparezcan antes».
        video_terms = llm.generate_terms(
            video_subject=params.video_subject,
            video_script=utils.remove_pause_tags(video_script),
            amount=8 if params.match_materials_to_script else 5,
            match_script_order=params.match_materials_to_script,
        )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms must be a string or a list of strings.")

        # Delimiter-only input must not reach paid reranking or material search.
        video_terms = [term for term in video_terms if term]

        logger.debug(f"video terms: {utils.to_json(video_terms)}")

    if not video_terms:
        _mark_task_failed(
            task_id,
            "terms",
            "failed to generate video search terms",
        )
        return None

    # Reordenación semántica opcional con TwelveLabs Marengo: si no está activada devuelve el orden original, sin ningún efecto secundario.
    # En el modo de correspondencia en orden, el orden de las palabras clave ya es el orden narrativo del guion y debe mantenerse, así que se omite.
    if not params.match_materials_to_script:
        video_terms = twelvelabs.rerank_terms_by_subject(
            video_subject=params.video_subject,
            search_terms=video_terms,
        )

    return video_terms


def save_script_data(task_id, video_script, video_terms, params):
    script_data = {
        "script": video_script,
        "search_terms": video_terms,
        "params": params,
    }
    task_artifacts.write_script_data(task_id, script_data)


def resolve_custom_audio_file(
    task_id: str,
    custom_audio_file: str | None,
    *,
    allow_server_file_input: bool = False,
) -> str:
    requested_file = (custom_audio_file or "").strip()
    if not requested_file:
        return ""

    task_dir = utils.task_dir(task_id)
    try:
        return file_security.resolve_path_within_directory(
            task_dir,
            requested_file,
        )
    except ValueError as exc:
        task_dir_error = exc

    # A missing path that otherwise stays inside the task directory is safe to
    # report precisely. Paths outside that boundary use the same generic error
    # regardless of whether they exist, so callers cannot probe the host filesystem.
    if str(task_dir_error) == "file does not exist":
        raise task_dir_error

    # HTTP requests and other untrusted callers must never turn a submitted path
    # into a server-side file read. WebUI uploads already live in the task directory;
    # only the local CLI explicitly opts into resolving files elsewhere on the host.
    if not allow_server_file_input:
        raise ValueError(
            "custom audio file must be stored within the current task directory"
        ) from task_dir_error

    server_audio_file = path.realpath(
        requested_file
        if path.isabs(requested_file)
        else path.join(utils.root_dir(), requested_file)
    )
    if not path.isabs(requested_file):
        project_root = path.realpath(utils.root_dir())
        try:
            if path.commonpath([project_root, server_audio_file]) != project_root:
                raise ValueError(
                    "relative custom audio paths must stay within the project directory"
                )
        except ValueError as exc:
            raise ValueError(
                "custom audio file must be task-local or an existing server-side file"
            ) from exc

    if not path.isfile(server_audio_file):
        raise ValueError(
            "custom audio file does not exist or is not a file"
        ) from task_dir_error

    return server_audio_file


def _resolve_reusable_voice_preview(
    task_id: str,
    params,
    video_script: str,
    voice_preview: dict | None,
) -> tuple[str, float, object] | None:
    """
    Valida y analiza la caché completa de previsualización de audio enviada por la WebUI.

    Esta carga no es un parámetro de la API pública y solo puede venir de la WebUI del proceso actual. Aun así, la tarea en segundo plano
    vuelve a comprobar el texto y todos los parámetros de voz, y limita el audio al directorio de la tarea actual; cualquier discrepancia
    vuelve al TTS normal, para que una previsualización caducada no contamine el video final.
    """
    if not voice_preview:
        return None

    expected_values = {
        "script": str(video_script or "").strip(),
        "voice_name": params.voice_name,
        "voice_rate": float(params.voice_rate),
        "voice_volume": float(params.voice_volume),
    }
    if not math.isclose(float(params.voice_volume), 1.0) or any(
        voice_preview.get(key) != value for key, value in expected_values.items()
    ):
        logger.info(
            f"skip stale voice preview cache, task_id: {task_id}, "
            "reason: voice parameters changed"
        )
        return None

    preview_file = path.realpath(str(voice_preview.get("audio_file") or ""))
    task_root = path.realpath(utils.task_dir(task_id))
    try:
        preview_is_task_local = path.commonpath([task_root, preview_file]) == task_root
    except ValueError:
        preview_is_task_local = False

    duration = voice_preview.get("duration")
    sub_maker = voice_preview.get("sub_maker")
    if (
        not preview_is_task_local
        or not path.isfile(preview_file)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
        or sub_maker is None
    ):
        logger.warning(
            f"skip invalid voice preview cache, task_id: {task_id}, "
            f"audio_file: {preview_file or '<empty>'}"
        )
        return None

    logger.info(
        f"using full voice preview audio, task_id: {task_id}, duration: {duration:.2f}s"
    )
    return preview_file, math.ceil(duration), sub_maker


def generate_audio(
    task_id,
    params,
    video_script,
    voice_preview=None,
    *,
    allow_server_file_input: bool = False,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
):
    """
    Generate audio for the video script.
    If a custom audio file is provided, it will be used directly.
    There will be no subtitle maker object returned in this case.
    Otherwise, TTS will be used to generate the audio.
    Returns:
        - audio_file: path to the generated or provided audio file
        - audio_duration: duration of the audio in seconds
        - sub_maker: subtitle maker object if TTS is used, None otherwise
    """
    logger.info("\n\n## generating audio")
    # /audio y el modelo de petición /subtitle no incluyen custom_audio_file;
    # aquí se lee de forma compatible, para evitar errores de atributo al llamar directamente a la interfaz.
    requested_custom_audio_file = getattr(params, "custom_audio_file", None)
    try:
        custom_audio_file = resolve_custom_audio_file(
            task_id,
            requested_custom_audio_file,
            allow_server_file_input=allow_server_file_input,
        )
    except ValueError as exc:
        _mark_task_failed(
            task_id,
            "audio",
            f"invalid custom audio file: {exc}",
        )
        return None, None, None

    if not custom_audio_file:
        reusable_preview = _resolve_reusable_voice_preview(
            task_id,
            params,
            video_script,
            voice_preview,
        )
        if reusable_preview:
            return reusable_preview

        logger.info("no custom audio file provided, using TTS to generate audio.")
        audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
        tts_kwargs = {
            "text": video_script,
            "voice_name": voice.parse_voice_name(params.voice_name),
            "voice_rate": params.voice_rate,
            "voice_file": audio_file,
        }
        if voxcpm_reference_audio is not None:
            tts_kwargs["voxcpm_reference_audio"] = voxcpm_reference_audio
        if voxcpm_prompt_audio is not None:
            tts_kwargs["voxcpm_prompt_audio"] = voxcpm_prompt_audio
            tts_kwargs["voxcpm_prompt_text"] = voxcpm_prompt_text
        sub_maker = voice.tts(**tts_kwargs)
        if sub_maker is None:
            _mark_task_failed(
                task_id,
                "audio",
                "failed to synthesize audio; verify the selected voice and TTS connectivity",
            )
            return None, None, None
        # Measure the real written audio_file, not sub_maker.cues[-1].end:
        # the latter is the last WORD BOUNDARY, and TTS leaves a fixed tail
        # past it (Edge TTS: ~0.88s at any length - 19% of a 7-word clip but
        # 1.4% of a 153-word one, so short scripts suffer most). The
        # under-count sizes paid generate_bgm() calls, is reported as
        # audio_duration to the API/WebUI, and under-sources
        # download_videos() material, scaled by video_count.
        file_duration = voice.get_audio_duration(audio_file)
        audio_duration = math.ceil(
            file_duration if file_duration > 0 else voice.get_audio_duration(sub_maker)
        )
        if audio_duration == 0:
            _mark_task_failed(task_id, "audio", "generated audio duration is zero")
            return None, None, None
        return audio_file, audio_duration, sub_maker
    else:
        logger.info(f"using custom audio file: {custom_audio_file}")
        audio_duration = voice.get_audio_duration(custom_audio_file)
        if audio_duration == 0:
            _mark_task_failed(
                task_id,
                "audio",
                "custom audio duration is zero",
            )
            return None, None, None
        return custom_audio_file, audio_duration, None


def generate_subtitle(task_id, params, video_script, sub_maker, audio_file):
    """
    Generate subtitle for the video script.
    If subtitle generation is disabled or no subtitle maker is provided, it will return an empty string.
    Otherwise, it will generate the subtitle using the specified provider.
    Returns:
        - subtitle_path: path to the generated subtitle file
    """
    logger.info("\n\n## generating subtitle")
    if not params.subtitle_enabled:
        return ""

    final_subtitle_path = path.join(utils.task_dir(task_id), "subtitle.srt")
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    estimated_timing = (
        getattr(sub_maker, "timing_source", "") == "estimated"
        or voice.is_gemini_voice(getattr(params, "voice_name", "") or "")
    )
    if subtitle_provider == "auto":
        subtitle_provider = "whisper" if estimated_timing or sub_maker is None else "edge"
    elif subtitle_provider == "edge" and estimated_timing:
        logger.info("TTS has estimated timings; aligning subtitles to the narration with Whisper")
        subtitle_provider = "whisper"
    logger.info(f"\n\n## generating subtitle, provider: {subtitle_provider}")

    if not subtitle_provider:
        logger.info("subtitle provider is empty, skip subtitle generation")
        return ""

    if sub_maker is None and subtitle_provider != "whisper":
        # El audio personalizado no pasa por el TTS, así que no tiene la línea de tiempo que devuelven TTS como Edge/Azure
        # sub_maker . Solo Whisper puede transcribir subtítulos directamente del archivo de audio;
        # los demás proveedores de subtítulos mantienen el comportamiento original, para no generar una línea de tiempo vacía errónea.
        logger.warning(
            "subtitle maker is missing, skip subtitle generation for provider: "
            f"{subtitle_provider}"
        )
        return ""

    is_word_level = getattr(params, "subtitle_display_mode", "sentence") == "word_by_word"

    # A failed retry must never reuse captions from an earlier narration.
    with staged_subtitle_file(final_subtitle_path) as subtitle_path:
        if subtitle_provider == "edge":
            voice.create_subtitle(
                text=video_script,
                sub_maker=sub_maker,
                subtitle_file=subtitle_path,
                word_level=is_word_level,
            )
            if not os.path.exists(subtitle_path):
                # Edge A veces los subtítulos no producen archivo porque la línea de tiempo no coincide con el texto. Aquí no se puede
                # cambiar automáticamente a Whisper, o un primer fallo descargaría sin que el usuario lo sepa varios GB
                # de modelo. Solo se permite cargar el modelo cuando Whisper está configurado explícitamente; si Edge falla se conserva
                # el video sin subtítulos y se registra el motivo, para evitar gastos inesperados de red y disco.
                logger.warning(
                    "edge subtitle generation did not produce a subtitle file; "
                    "skip subtitles without falling back to whisper"
                )
                return ""

        if subtitle_provider == "whisper":
            max_words = 0 if is_word_level else max(0, int(config.ui.get("subtitle_max_words", 0) or 0))
            subtitle.create(
                audio_file=audio_file,
                subtitle_file=subtitle_path,
                word_level=is_word_level,
                **({"max_words": max_words} if max_words else {}),
                **({"reference_text": video_script} if estimated_timing else {}),
            )
            if not subtitle.file_to_subtitles(subtitle_path):
                logger.warning("whisper produced no usable subtitle cues")
                return ""
            if not is_word_level and not max_words and not estimated_timing:
                logger.info("\n\n## correcting subtitle")
                subtitle.correct(subtitle_file=subtitle_path, video_script=video_script)

        subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
        if not subtitle_lines:
            logger.warning(f"subtitle file is invalid: {subtitle_path}")
            return ""

        os.replace(subtitle_path, final_subtitle_path)
        return final_subtitle_path


def get_video_materials(
    task_id,
    params,
    video_terms,
    audio_duration,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
    jalapeno_video_endpoint: volcengine_seedance.SeedanceEndpoint | None = None,
    codex_image_context: codex_image.CodexImageContext | None = None,
):
    if params.video_source == "local":
        logger.info("\n\n## preprocess local materials")
        materials = video.preprocess_video(
            materials=params.video_materials,
            clip_duration=params.video_clip_duration,
            image_motion=params.image_motion,
        )
        if not materials:
            _mark_task_failed(
                task_id,
                "materials",
                "no valid local video materials were found",
            )
            return None
        return [material_info.url for material_info in materials]
    elif params.video_source == "loomloom":
        if not isinstance(
            loomloom_video_request, loomloom.LoomLoomConfirmedVideoRequest
        ):
            _mark_task_failed(
                task_id,
                "materials",
                "LoomLoom video generation requires a confirmed quote",
            )
            return None

        request = loomloom_video_request
        logger.info(
            "\n\n## generating "
            f"{len(request.batch.input_rows)} video materials with LoomLoom"
        )
        run_id = ""
        try:
            request.validate()
            backend = loomloom.LoomLoomVideoBackend(request.settings)
            generation_progress.submitting("loomloom")
            execution = backend.execute(
                request.batch,
                client_request_id=request.client_request_id,
                listing_version_id=request.listing_version_id,
                confirm=True,
            )
            run_id = execution.run_id
            # execute Volver significa que el remoto ya aceptó la tarea de pago. Hay que escribir primero el run ID en el
            # log del proceso; aunque después el backend de estado, como Redis, no esté disponible, el personal de operaciones puede usar el log
            # para localizar la tarea en Shengsuanyun; no se puede dejar que el único identificador exista solo en una variable local.
            logger.info(
                "LoomLoom paid video run created: "
                f"task_id={task_id}, run_id={run_id}, "
                f"listing_version_id={request.listing_version_id}"
            )
            # En cuanto se crea la tarea de pago se registra el ID remoto. Aunque el sondeo posterior agote el tiempo, el log y el estado de la tarea
            # siguen ayudando al usuario o al soporte de la plataforma a localizar y recuperar el resultado ya generado. Un fallo del backend de estado
            # solo puede reducir la observabilidad, sin interrumpir la tarea remota ya cobrada ni la descarga del resultado.
            _record_loomloom_run_reference(
                task_id=task_id,
                run_id=run_id,
                listing_version_id=request.listing_version_id,
            )
            backend.wait_for_run(run_id)
            return list(
                backend.download_video_results(
                    run_id,
                    utils.task_dir(task_id),
                )
            )
        except (loomloom.LoomLoomError, ValueError) as exc:
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details={
                    "loomloom_run_id": run_id,
                    "loomloom_listing_version_id": request.listing_version_id,
                },
            )
            return None
    else:
        logger.info(f"\n\n## downloading videos from {params.video_source}")
        generation_options = (
            {"jalapeno_video_endpoint": jalapeno_video_endpoint}
            if params.video_source == "jalapeno" and jalapeno_video_endpoint is not None
            else {}
        )
        if params.video_source == "codex_image":
            generation_options["codex_image_context"] = codex_image_context
        # El modo de correspondencia en orden solo se aplica cuando el usuario lo activa explícitamente. Aquí se fuerza que la descarga de material se haga por turnos en el orden
        # de las palabras clave, para que una palabra clave temprana no descargue demasiado material y saque los temas posteriores del guion de la línea de tiempo final.
        try:
            downloaded_videos = material.download_videos(
                task_id=task_id,
                search_terms=video_terms,
                source=params.video_source,
                video_aspect=params.video_aspect,
                video_concat_mode=(
                    VideoConcatMode.sequential
                    if params.match_materials_to_script
                    else params.video_concat_mode
                ),
                audio_duration=audio_duration * (1 if params.video_source == "wikimedia" else params.video_count),
                max_clip_duration=params.video_clip_duration,
                match_script_order=params.match_materials_to_script,
                image_motion=params.image_motion,
                # La fase de material ocupa el 40%~50%: avanza una vez por cada archivo descargado, así que en redes lentas
                # la barra de progreso ya no se queda mucho tiempo en el 40%.
                progress_callback=_stage_progress_reporter(task_id, 40, 50),
                **generation_options,
            )
        except wikimedia.PortraitNotFound as exc:
            _mark_task_failed(task_id, "materials", str(exc))
            return None
        except volcengine_seedance.VolcEngineSeedanceError as exc:
            # Tanto el estado no confirmado como el generado pero con descarga fallida corresponden a una tarea remota recuperable en la consola de Ark
            # . Se escribe el estado de fallo de forma unificada con el task_id que lleva la excepción, para que distintas ramas de excepción
            # no mantengan cada una la información de recuperación y se vuelva a olvidar en futuras ampliaciones.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = (
                {("jalapeno_task_id" if params.video_source == "jalapeno"
                  else "volcengine_seedance_task_id"): remote_task_id}
                if remote_task_id
                else None
            )
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        except (
            material.WaveSpeedUnconfirmedTaskError,
            material.WaveSpeedDownloadError,
        ) as exc:
            # Once a paid prediction exists, a polling or download failure must
            # leave its ID in task state. Earlier successful clips cannot turn
            # this incomplete run into an apparently completed video task.
            prediction_id = str(getattr(exc, "prediction_id", "") or "").strip()
            details = {"wavespeed_prediction_id": prediction_id} if prediction_id else None
            _mark_task_failed(task_id, "materials", str(exc), details=details)
            return None
        except material.OpenAIImagePaidResultError as exc:
            # A paid request with an uncertain response, failed result download,
            # or failed local render must stop before another image order.
            details = (
                {"error_key": exc.key, "error_params": exc.params}
                if isinstance(exc, material.CodexImageMaterialError)
                else None
            )
            _mark_task_failed(task_id, "materials", str(exc), details=details)
            return None
        except ofox.OFoxError as exc:
            # Misma semántica de recuperación que Ark: tanto el estado no confirmado como el generado pero con descarga fallida corresponden a una tarea remota
            # OFox recuperable en la consola; el estado de fallo se escribe de forma unificada con el task_id que lleva la excepción.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = (
                {"ofox_task_id": remote_task_id} if remote_task_id else None
            )
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        except metaso_minimax.MetasoMiniMaxError as exc:
            # Las tareas de Metaso y de Ark usan entradas de recuperación y nombres de campo distintos, así que no se pueden unir en un
            # remote_task_id ambiguo. Se conserva un prefijo de Provider claro para que la API,WebUI
            # y los logs de operaciones localicen directamente la plataforma correspondiente.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = (
                {"metaso_minimax_task_id": remote_task_id} if remote_task_id else None
            )
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        except infistar_video.InfistarVideoError as exc:
            remote_task_id = str(exc.task_id or "").strip()
            _mark_task_failed(
                task_id, "materials", str(exc),
                details={"infistar_task_id": remote_task_id} if remote_task_id else None,
            )
            return None
        except muapi.MuAPIError as exc:
            # MuAPI tasks are billable after acceptance.  Preserve the remote
            # request ID on any ambiguous or post-completion failure so the
            # user can recover it from the provider dashboard.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = {"muapi_task_id": remote_task_id} if remote_task_id else None
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        if not downloaded_videos:
            _mark_task_failed(
                task_id,
                "materials",
                f"failed to download video materials from {params.video_source}",
            )
            return None
        return downloaded_videos


def _record_loomloom_run_reference(
    *, task_id: str, run_id: str, listing_version_id: str
) -> bool | None:
    """
    Hace todo lo posible por guardar el LoomLoom Run de pago ya creado, sin que un fallo de estado interrumpa la tarea remota.

    Devolver True significa guardado con éxito, False que el registro de la tarea ya no existe, y None que el backend de estado
    sigue sin estar disponible tras reintentos limitados. Sea cual sea el resultado, el llamador debe seguir sondeando y descargando, porque
    execute ya se produjo un efecto secundario de pago externo, y detener el flujo local solo haría más difícil recuperar el resultado.
    """
    fields = {
        "loomloom_run_id": run_id,
        "loomloom_listing_version_id": listing_version_id,
    }
    for attempt in range(1, _LOOMLOOM_STATE_WRITE_ATTEMPTS + 1):
        try:
            updated = sm.state.patch_task(task_id, **fields)
        except Exception as exc:
            if attempt >= _LOOMLOOM_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    "failed to persist LoomLoom paid run after retries: "
                    f"task_id={task_id}, run_id={run_id}, attempts={attempt}, "
                    f"error={exc}"
                )
                return None
            logger.warning(
                "retry LoomLoom paid run state update: "
                f"task_id={task_id}, run_id={run_id}, attempt={attempt}, "
                f"error={exc}"
            )
            time.sleep(_LOOMLOOM_STATE_RETRY_DELAY_SECONDS)
            continue

        if updated is False:
            logger.warning(
                "could not persist LoomLoom paid run because task is missing: "
                f"task_id={task_id}, run_id={run_id}"
            )
        return updated

    return None


def _get_material_source_groups(task_id: str, video_paths: list[str]) -> dict[str, str]:
    """Recover keyword groups from the downloaded material manifest."""
    try:
        with open(path.join(utils.task_dir(task_id), "script.json"), encoding="utf-8") as file:
            payload = json.load(file)
        groups = {
            record["local_file"]: record["search_term"]
            for record in payload.get("material_sources", [])
            if isinstance(record, dict)
            and isinstance(record.get("local_file"), str)
            and isinstance(record.get("search_term"), str)
            and record["search_term"]
        }
        return {
            file: groups[path.basename(file)]
            for file in video_paths
            if path.basename(file) in groups
        }
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        logger.warning(f"Cannot read material keyword groups: task_id={task_id}, error={exc}")
        return {}


def generate_final_videos(
    task_id, params, downloaded_videos, audio_file, subtitle_path, audio_duration,
    *, partial_results: dict | None = None,
):
    """Record completed outputs even if a later clip raises before returning."""
    final_video_paths = []
    combined_video_paths = []
    warnings = []
    if partial_results is not None:
        # The caller owns this snapshot so it can report completed files even
        # when the state backend fails during a progress/checkpoint write.
        partial_results.update(
            videos=final_video_paths, combined_videos=combined_video_paths,
            warnings=warnings,
        )
    allocate_batch_materials = params.video_count > 1 and params.video_source in {
        "pexels", "pixabay", "coverr", "local"
    }
    source_usage = {}
    material_selections = []
    source_groups = (
        _get_material_source_groups(task_id, downloaded_videos)
        if (allocate_batch_materials and params.match_materials_to_script
            and params.video_source != "local")
        else {}
    )
    video_music_provider = _VIDEO_MUSIC_PROVIDERS.get(params.bgm_type)
    video_music_requested = (
        video_music_provider is not None
        and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    )
    # Matching preserves keyword order; batch allocation varies each keyword's candidates.
    if params.match_materials_to_script or params.video_source == "wikimedia":
        video_concat_mode = VideoConcatMode.sequential
    elif params.video_count == 1:
        video_concat_mode = params.video_concat_mode
    else:
        video_concat_mode = VideoConcatMode.random
    video_transition_mode = params.video_transition_mode

    _progress = 50
    for i in range(params.video_count):
        index = i + 1
        if partial_results is not None:
            partial_results["failed_video_index"] = index
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        logger.info(f"\n\n## combining video: {index} => {combined_video_path}")
        used_video_paths = []
        batch_options = (
            {
                "source_usage": source_usage,
                "source_groups": source_groups,
                "used_video_paths": used_video_paths,
            }
            if allocate_batch_materials else {}
        )
        # Cada video ocupa una parte del 50% del progreso, con composición y codificación final a partes iguales. El procesamiento de fragmentos es
        # lo más costoso de la composición, y esa mitad del progreso avanza fragmento a fragmento.
        combine_share = 50 / params.video_count / 2
        video.combine_videos(
            combined_video_path=combined_video_path,
            video_paths=downloaded_videos,
            audio_file=audio_file,
            video_aspect=params.video_aspect,
            video_fit_mode=params.video_fit_mode,
            video_concat_mode=video_concat_mode,
            video_transition_mode=video_transition_mode,
            max_clip_duration=params.video_clip_duration,
            threads=params.n_threads,
            clip_speed=params.video_clip_speed,
            progress_callback=_stage_progress_reporter(
                task_id, _progress, _progress + combine_share
            ),
            **batch_options,
        )
        if allocate_batch_materials:
            selected_sources = list(dict.fromkeys(used_video_paths))
            reused_sources = [file for file in selected_sources if source_usage.get(file, 0)]
            for file in selected_sources:
                source_usage[file] = source_usage.get(file, 0) + 1
            material_selections.append({
                "video_index": index,
                "local_files": [path.basename(file) for file in used_video_paths],
                "reused_files": [path.basename(file) for file in reused_sources],
            })
            task_artifacts.patch_script_data(task_id, material_selections=material_selections)
            logger.info(
                f"Batch material allocation: video_index={index}, "
                f"sources={len(selected_sources)}, reused_sources={len(reused_sources)}"
            )
            if reused_sources:
                warnings.append({
                    "code": "batch_materials_reused",
                    "video_index": index,
                    "count": len(reused_sources),
                })

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_path = path.join(utils.task_dir(task_id), f"final-{index}.mp4")

        # El modo de música para video primero desactiva explícitamente el análisis de BGM por defecto, para no usar por error el bgm_file que quede de tareas antiguas
        # . Solo con volumen mayor que 0 se genera el proxy y se llama a la API de pago; con volumen 0 siempre se omite.
        bgm_file_override = "" if video_music_provider else None
        if video_music_requested:
            service = video_music_provider["service"]
            display_name = video_music_provider["display_name"]
            warning_code = video_music_provider["warning_code"]
            generated_bgm_path = path.join(
                utils.task_dir(task_id),
                (f"{params.bgm_type}-bgm-{index}{video_music_provider['suffix']}"),
            )
            try:
                service.generate_bgm(
                    video_path=combined_video_path,
                    output_path=generated_bgm_path,
                    video_duration=audio_duration,
                    prompt=_get_video_music_prompt(params),
                )
                bgm_file_override = generated_bgm_path
            except video_music_provider["error_type"] as exc:
                # Cuando el video, la narración y los subtítulos ya se generaron, un fallo temporal de la música de terceros no debe desperdiciar toda
                # la tarea. Se desactiva explícitamente la BGM del video actual y se devuelve el resultado degradado a la WebUI para avisar al usuario.
                logger.warning(
                    f"{display_name} BGM generation failed: task_id={task_id}, "
                    f"video_index={index}, error={exc}"
                )
                bgm_file_override = ""
                warnings.append({"code": warning_code, "video_index": index})

        logger.info(f"\n\n## generating video: {index} => {final_video_path}")
        bgm_mix_succeeded = video.generate_video(
            video_path=combined_video_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params,
            bgm_file_override=bgm_file_override,
        )
        if (
            video_music_provider is not None
            and bgm_file_override
            and not bgm_mix_succeeded
        ):
            # El tercero devolvió con éxito y pasó la validación de FFmpeg, pero la mezcla final de MoviePy aún puede
            # fallar por el entorno de ejecución. El servicio de video conserva el video final sin BGM; cuando falla la generación por API
            # override está vacío, así que no se añade una advertencia duplicada.
            warnings.append(
                {
                    "code": video_music_provider["warning_code"],
                    "video_index": index,
                }
            )

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)
        _progress += 50 / params.video_count / 2
        sm.state.update_task(
            task_id, progress=_progress, videos=list(final_video_paths),
            combined_videos=list(combined_video_paths), warnings=list(warnings) or None,
        )

    return final_video_paths, combined_video_paths, warnings


def _patch_cross_post_state(task_id: str, **kwargs) -> bool | None:
    """Actualiza de forma segura los campos de publicación; ante fallos breves del backend de estado reintenta un número limitado de veces."""
    for attempt in range(1, _CROSS_POST_STATE_WRITE_ATTEMPTS + 1):
        try:
            return sm.state.patch_task(task_id, **kwargs)
        except Exception as exc:
            # Redis Una desconexión breve no debe dejar la tarea para siempre en pending/processing. El estado de publicación
            # se escribe con muy poca frecuencia, así que aquí basta un número fijo de intentos con esperas cortas para cubrir fallos momentáneos,
            # evitando bloquear indefinidamente el hilo en segundo plano. El último fallo conserva la pila completa para facilitar la localización.
            if attempt >= _CROSS_POST_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    f"failed to update cross-post state after retries, "
                    f"task_id: {task_id}, fields: {', '.join(kwargs)}, "
                    f"attempts: {attempt}, error: {exc}"
                )
                return None

            logger.warning(
                f"retry cross-post state update, task_id: {task_id}, "
                f"fields: {', '.join(kwargs)}, attempt: {attempt}, error: {exc}"
            )
            time.sleep(_CROSS_POST_STATE_RETRY_DELAY_SECONDS)

    return None


def _record_cross_post_failure(
    task_id: str,
    error: Exception,
    results: list[dict] | None = None,
) -> None:
    """Hace todo lo posible por guardar el fallo de publicación; si el backend de estado no está disponible, el log conserva la información de diagnóstico."""
    updated = _patch_cross_post_state(
        task_id,
        cross_post_state=const.CROSS_POST_STATE_FAILED,
        cross_post_results=results or None,
        cross_post_error=str(error),
        cross_post_owner=None,
    )
    if updated is False:
        logger.warning(f"discard cross-post failure for missing task: {task_id}")


def _ensure_cross_post_terminal_state(task_id: str) -> None:
    """Future al terminar, convierte en fallidas las tareas que siguen en estado activo."""
    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        # Esto ya es el callback final del Future y no hay un llamador síncrono posterior que pueda manejar la excepción.
        # Cuando el backend de estado se recupere, el siguiente arranque del proceso seguirá tratando el estado pendiente mediante la lógica de recuperación.
        logger.exception(
            f"failed to verify final cross-post state, task_id: {task_id}, error: {exc}"
        )
        return

    if not task or task.get("cross_post_state") not in _ACTIVE_CROSS_POST_STATES:
        return

    logger.warning(
        f"cross-post worker ended without terminal state, task_id: {task_id}, "
        f"state: {task.get('cross_post_state')}"
    )
    _record_cross_post_failure(
        task_id,
        RuntimeError("cross-post worker ended without persisting a terminal state"),
        task.get("cross_post_results"),
    )


def recover_interrupted_cross_posts(page_size: int = 100) -> int | None:
    """
    Marca como fallidas las tareas de publicación que no se pueden recuperar tras reiniciar el proceso.

    La publicación multiplataforma usa un pool de hilos dentro del proceso actual, no una cola de tareas persistente. Al arrancar el proceso,
    Redis los pending/processing que queden no continúan automáticamente; si se siguieran considerando
    en ejecución, el usuario nunca podría eliminar la tarea. Aquí se recogen los IDs de tarea una sola vez y luego se lee el estado actual;
    no se puede avanzar por número de página en varios SCAN de Redis, o un cambio en el orden de escaneo haría perder tareas pendientes.
    Solo se tratan los registros activos sin Future correspondiente en el proceso actual, conservando los resultados de video ya generados.
    """
    recovered = 0
    try:
        task_ids = sm.state.list_task_ids(scan_count=page_size)
        for task_id in task_ids:
            # Tras el escaneo la tarea puede haberse eliminado o pasado a un estado final; se decide si recuperar según el estado más reciente,
            # sin usar la instantánea antigua del escaneo para sobrescribir resultados de publicación ya completados.
            task = sm.state.get_task(task_id)
            if (
                not task
                or task.get("cross_post_state") not in _ACTIVE_CROSS_POST_STATES
                or _is_cross_post_active_in_process(task_id)
                or _is_cross_post_owner_alive(task.get("cross_post_owner"))
            ):
                continue

            updated = _patch_cross_post_state(
                task_id,
                cross_post_state=const.CROSS_POST_STATE_FAILED,
                cross_post_error=_INTERRUPTED_CROSS_POST_ERROR,
                cross_post_owner=None,
            )
            if updated is True:
                recovered += 1
    except Exception as exc:
        logger.exception(f"failed to recover interrupted cross-post tasks: {exc}")
        return None

    if recovered:
        logger.warning(f"recovered interrupted cross-post tasks: {recovered}")
    return recovered


def _run_cross_post(
    task_id: str,
    video_paths: tuple[str, ...],
    video_subject: str,
    video_script: str,
    video_language: str,
    platforms: tuple[str, ...],
    youtube_privacy_status: str,
    youtube_made_for_kids: bool = False,
    upload_account: dict | None = None,
) -> None:
    """Ejecuta en segundo plano la publicación multiplataforma, añadiendo solo los campos de tarea relacionados con la publicación."""
    results = []
    try:
        state_updated = _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_PROCESSING,
            cross_post_error=None,
            cross_post_owner=_cross_post_process_owner,
        )
        if state_updated is not True:
            # False indica que la tarea se eliminó, y None que el backend de estado no está disponible temporalmente. En ambos casos
            # no se debe seguir llamando a interfaces de terceros, o el usuario no podría consultar ni controlar esta publicación.
            if state_updated is False:
                logger.warning(f"skip cross-post for missing task: {task_id}")
            else:
                _record_cross_post_failure(
                    task_id,
                    RuntimeError("failed to persist cross-post processing state"),
                )
            return

        logger.info(
            f"cross-post started, task_id: {task_id}, platforms: {', '.join(platforms)}"
        )
        youtube_extra = None
        post_title = video_subject or "Check out this video! #shorts #viral"
        if platforms:
            has_youtube = any(platform.startswith("youtube") for platform in platforms)
            social_platform = "youtube_shorts"
            if not has_youtube:
                first = (platforms[0] or "").strip().lower()
                # llm.py resolves unknown ids to its default platform.
                social_platform = _CROSS_POST_SOCIAL_PLATFORMS.get(first, first)
            metadata = llm.generate_social_metadata(
                video_subject=video_subject,
                video_script=video_script,
                language=video_language or "",
                platform=social_platform,
            )
            if has_youtube:
                youtube_extra = {
                    "youtube_title": metadata.get("title", video_subject),
                    "youtube_description": metadata.get("caption", ""),
                    "tags": metadata.get("hashtags", []),
                    "privacyStatus": youtube_privacy_status,
                    "selfDeclaredMadeForKids": youtube_made_for_kids,
                    "containsSyntheticMedia": True,
                }
            post_title = (
                metadata.get("caption")
                or metadata.get("title")
                or video_subject
                or "Check out this video! #shorts #viral"
            )

        for video_path in video_paths:
            pending_result_index = None

            def record_background_request(request_id: str) -> None:
                nonlocal pending_result_index
                # Persist the remote handle before polling. If this process
                # exits mid-upload, recovery can expose the ID to the user.
                pending_result_index = len(results)
                results.append(
                    {
                        "video_index": pending_result_index + 1,
                        "request_id": request_id,
                        "status": "processing",
                    }
                )
                if _patch_cross_post_state(
                    task_id, cross_post_results=list(results)
                ) is not True:
                    logger.warning(
                        "could not persist background upload request ID: "
                        f"task_id={task_id}, request_id={request_id}"
                    )

            upload_kwargs = {}
            if upload_account is not None:
                upload_kwargs["account"] = upload_account
            result = upload_post.cross_post_video(
                video_path=video_path,
                title=post_title,
                platforms=list(platforms),
                youtube_extra=youtube_extra,
                on_background_start=record_background_request,
                **upload_kwargs,
            )
            if not isinstance(result, dict):
                result = {
                    "success": False,
                    "error": "Upload-Post returned an invalid response",
                }
            if pending_result_index is None:
                results.append(result)
            else:
                results[pending_result_index] = result

        failures = [result for result in results if not result.get("success")]
        if failures:
            error_messages = [
                str(
                    result.get("error")
                    or result.get("message")
                    or "unknown upload error"
                )
                for result in failures
            ]
            cross_post_state = const.CROSS_POST_STATE_FAILED
            cross_post_error = "; ".join(error_messages)
            logger.warning(
                f"cross-post completed with failures, task_id: {task_id}, "
                f"failed: {len(failures)}, total: {len(results)}"
            )
        else:
            cross_post_state = const.CROSS_POST_STATE_COMPLETE
            cross_post_error = None
            logger.success(
                f"cross-post completed, task_id: {task_id}, videos: {len(results)}"
            )

        state_updated = _patch_cross_post_state(
            task_id,
            cross_post_state=cross_post_state,
            cross_post_results=results,
            cross_post_error=cross_post_error,
            cross_post_owner=None,
        )
        if state_updated is False:
            logger.warning(f"discard cross-post result for missing task: {task_id}")
        elif state_updated is None:
            # Si la subida ya terminó pero el resultado no se persistió, no se puede seguir manteniendo processing.
            # La escritura del estado de fallo vuelve a pasar por reintentos limitados, para que al menos el llamador obtenga un estado final claro.
            _record_cross_post_failure(
                task_id,
                RuntimeError("failed to persist final cross-post result"),
                results,
            )
    except Exception as exc:
        # Un fallo de publicación solo afecta al estado de publicación y no puede sobrescribir una tarea de video ya completada.
        # El texto original de la excepción se escribe en el estado de la tarea, para que el llamador de la API localice el problema sin acceder a los logs del servidor.
        logger.exception(f"cross-post failed, task_id: {task_id}, error: {exc}")
        _record_cross_post_failure(task_id, exc, results)


def _run_cross_post_with_slot(*args) -> None:
    """Ejecuta la tarea de publicación, garantizando que la capacidad de la cola se devuelva tanto en éxito como en fallo o excepción."""
    try:
        _run_cross_post(*args)
    except Exception as exc:
        # _run_cross_post ya maneja las excepciones previstas; esta es la última protección, para que las excepciones que lance lógica
        # añadida en el futuro no queden guardadas solo en un Future que nadie lee.
        task_id = str(args[0]) if args else "unknown"
        logger.exception(f"cross-post worker crashed, task_id: {task_id}, error: {exc}")
        if args:
            _record_cross_post_failure(task_id, exc)
    finally:
        _cross_post_slots.release()


def _finalize_cross_post_future(task_id: str, future: Future) -> None:
    """Limpia el registro del Future, garantizando que las cancelaciones, excepciones y fallos de escritura de estado se resuelvan."""
    _unregister_cross_post_future(task_id, future)

    try:
        error = future.exception()
    except CancelledError:
        logger.warning(f"cross-post future was cancelled, task_id: {task_id}")
        # Future Si se cancela antes de empezar a ejecutarse, el finally del worker no se ejecuta, así que hay que
        # devolver la capacidad de la cola en el callback y cambiar el estado persistido a fallido.
        _cross_post_slots.release()
        _record_cross_post_failure(
            task_id,
            RuntimeError("cross-post job was cancelled before execution"),
        )
        return
    except Exception as exc:
        logger.exception(
            f"failed to inspect cross-post future, task_id: {task_id}, error: {exc}"
        )
        _ensure_cross_post_terminal_state(task_id)
        return

    if error is not None:
        logger.error(
            f"cross-post future failed, task_id: {task_id}, "
            f"error: {type(error).__name__}: {error}"
        )

    _ensure_cross_post_terminal_state(task_id)


def _schedule_cross_post(
    task_id: str,
    video_paths: list[str],
    params: VideoParams,
    video_script: str,
    platforms: list[str],
    youtube_privacy_status: str,
    youtube_made_for_kids: bool = False,
) -> str | None:
    """Envía la tarea de publicación en segundo plano; si tiene éxito devuelve None, y si falla la planificación devuelve un motivo de error consultable."""
    if not _cross_post_slots.acquire(blocking=False):
        error = "cross-post queue is full; publishing was skipped"
        logger.warning(
            f"skip cross-post because queue is full, task_id: {task_id}, "
            f"capacity: {_cross_post_max_pending_tasks}"
        )
        _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_FAILED,
            cross_post_error=error,
            cross_post_owner=None,
        )
        return error

    try:
        future = _cross_post_executor.submit(
            _run_cross_post_with_slot,
            task_id,
            tuple(video_paths),
            params.video_subject or "",
            video_script,
            params.video_language or "",
            tuple(platforms),
            youtube_privacy_status,
            youtube_made_for_kids,
            upload_post.upload_post_service.snapshot_account(),
        )
        _register_cross_post_future(task_id, future)
        future.add_done_callback(partial(_finalize_cross_post_future, task_id))
    except RuntimeError as exc:
        _unregister_cross_post_future(task_id)
        _cross_post_slots.release()
        logger.exception(
            f"failed to schedule cross-post, task_id: {task_id}, error: {exc}"
        )
        _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_FAILED,
            cross_post_error=f"failed to schedule cross-post: {exc}",
            cross_post_owner=None,
        )
        return f"failed to schedule cross-post: {exc}"

    return None


def _run_pipeline(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    voice_preview: dict | None = None,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
    allow_server_file_input: bool = False,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
    jalapeno_video_endpoint: volcengine_seedance.SeedanceEndpoint | None = None,
    codex_image_context: codex_image.CodexImageContext | None = None,
):
    logger.info(f"start task: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(
        task_id, state=const.TASK_STATE_PROCESSING, progress=5,
        failed_stage=None, error=None, error_key=None, error_params=None,
        videos=[], combined_videos=[], warnings=None, failed_video_index=None,
        generation_started_at=time.time(),
    )

    generation_progress.stage("preparing")

    if stop_at == "video" and params.subtitle_enabled:
        try:
            video.validate_subtitle_colors(params)
        except ValueError as exc:
            return _mark_task_failed(task_id, "preflight", str(exc))

    if stop_at in {"materials", "video"} and params.video_source not in _SUPPORTED_VIDEO_SOURCES:
        return _mark_task_failed(
            task_id, "preflight", f"unsupported video source: {params.video_source!r}"
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "volcengine_seedance"
        and not volcengine_seedance.is_enabled()
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "Volcano Engine Seedance requires an Ark API key",
        )

    if stop_at in {"materials", "video"} and params.video_source == "codex_image":
        try:
            if codex_image_context is None:
                codex_image_context = codex_image.capture_context()
            codex_image.assert_current(codex_image_context)
        except codex_oauth.CodexError as exc:
            return _mark_task_failed(
                task_id,
                "preflight",
                str(exc),
                details={"error_key": exc.key, "error_params": exc.params},
            )

    if stop_at in {"materials", "video"} and params.video_source == "jalapeno":
        try:
            # WebUI La configuración ya confirmada acompaña a la tarea; las demás entradas la congelan una vez en la comprobación previa, para evitar que
            # los envíos de material posteriores vuelvan a leer la Key, el modelo o la especificación que otra petición haya reescrito.
            if jalapeno_video_endpoint is None:
                jalapeno_video_endpoint = jalapeno_video.endpoint()
        except jalapeno_video.JalapenoVideoError as exc:
            return _mark_task_failed(task_id, "preflight", str(exc))

    if stop_at in {"materials", "video"} and params.video_source == "infistar":
        try:
            if not infistar_video.is_enabled():
                raise infistar_video.InfistarVideoError("Infistar video generation requires an API key")
            infistar_video.model_id()
            if params.video_aspect not in {"16:9", "9:16"}:
                raise infistar_video.InfistarVideoError("Infistar video currently supports 16:9 and 9:16")
        except infistar_video.InfistarVideoError as exc:
            return _mark_task_failed(task_id, "preflight", str(exc))

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "ofox"
        and not ofox.is_enabled()
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "OFox video generation requires an OFox API key",
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "metaso_minimax"
        and not metaso_minimax.is_enabled()
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "Metaso MiniMax requires an API key",
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "muapi"
        and not muapi.is_enabled()
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "MuAPI video generation requires a MuAPI API key",
        )

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "openai_image"
        and not material.is_openai_image_enabled(
            config.snapshot_config_with_pending(config.app)
        )
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "OpenAI image source requires openai_image_base_url and "
            "openai_image_model in config.toml (openai_image_api_keys is "
            "optional for local gateways that need no auth)",
        )

    # Solo el flujo de video final completo necesita proveedor de música para video. Se bloquean pronto las tareas completas sin Key, para no
    # consumir antes cuota de LLM, TTS y servicios de material; las interfaces de productos intermedios se siguen pudiendo usar por separado.
    video_music_provider = _VIDEO_MUSIC_PROVIDERS.get(params.bgm_type)
    video_music_enabled = (
        stop_at == "video"
        and video_music_provider is not None
        and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    )
    if video_music_enabled:
        service = video_music_provider["service"]
        display_name = video_music_provider["display_name"]
        if not service.is_enabled():
            return _mark_task_failed(
                task_id,
                "preflight",
                f"{display_name} background music requires an API key",
            )

        # WebUI limita la longitud de entrada, pero la API, el CLI y las tareas históricas pueden saltarse los controles del frontend.
        # Antes de generar el guion, la voz y el material se vuelve a validar según el límite del proveedor, para que no sea tras componer el video completo
        # cuando lo rechace la petición al tercero. La capa de servicio mantiene la misma validación como última defensa en llamadas directas.
        music_prompt = _get_video_music_prompt(params)
        max_prompt_length = int(getattr(service, "MAX_PROMPT_LENGTH", 0) or 0)
        if max_prompt_length and len(music_prompt) > max_prompt_length:
            return _mark_task_failed(
                task_id,
                "preflight",
                (f"{display_name} music prompt exceeds {max_prompt_length} characters"),
            )

        # Los proveedores pueden ofrecer una comprobación previa de cuenta sin coste. La función de comprobación solo debe lanzar errores
        # deterministas; cuando la fluctuación de red o el alcance de permisos no se pueden confirmar, la capa de servicio registra una advertencia y continúa con la generación real.
        validate_access = getattr(service, "validate_generation_access", None)
        if callable(validate_access):
            try:
                validate_access()
            except video_music_provider["error_type"] as exc:
                return _mark_task_failed(task_id, "preflight", str(exc))

    # Solo los productos intermedios script/terms no necesitan FFmpeg (no generan audio ni video). La API,
    # CLI y la WebUI ejecutan tareas a través de esta entrada compartida, así que se detecta aquí de forma unificada, en lugar de
    # repetir la comprobación en cada entrada, garantizando el mismo comportamiento en las tres rutas. Se coloca tras la validación de la Key de música
    # para no cambiar el orden de "fallar primero" ni los mensajes de error originales de esas validaciones.
    if stop_at not in ("script", "terms") and not utils.check_ffmpeg_ready():
        return _mark_task_failed(
            task_id,
            "preflight",
            "ffmpeg is not available; install ffmpeg or set app.ffmpeg_path "
            "in config.toml to a working ffmpeg executable",
        )

    # 1. Generate script
    generation_progress.stage("script")
    video_script = generate_script(task_id, params)
    # The LLM adapter uses a leading "Error: " as its failure sentinel. A
    # user-provided script may legitimately quote that text anywhere, including
    # at the beginning, so only interpret it for an LLM-generated script.
    is_provider_error = (
        not (params.video_script or "").strip()
        and isinstance(video_script, str)
        and video_script.startswith("Error: ")
    )
    if not video_script or is_provider_error:
        error = (
            video_script.removeprefix("Error: ").strip()
            if is_provider_error
            else "failed to generate video script"
        )
        return _mark_task_failed(task_id, "script", error)

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

    if stop_at == "script":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=video_script
        )
        return {"script": video_script}

    generation_progress.stage("terms")
    # 2. Generate terms
    video_terms = ""
    if stop_at in {"terms", "materials", "video"} and params.video_source != "local":
        video_terms = generate_terms(task_id, params, video_script)
        if not video_terms:
            return _mark_task_failed(
                task_id,
                "terms",
                "failed to generate video search terms",
            )

    if stop_at == "video" and (params.opening_title_enabled or params.video_source == "wikimedia"):
        if not params.opening_title.strip():
            params.opening_title = llm.generate_opening_title(
                params.video_subject, video_script, params.video_language
            )
    save_script_data(task_id, video_script, video_terms, params)

    if stop_at == "terms":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
        )
        return {"script": video_script, "terms": video_terms}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    generation_progress.stage("audio")
    # 3. Generate audio
    generate_audio_kwargs = {
        "voice_preview": voice_preview,
        "allow_server_file_input": allow_server_file_input,
    }
    if voxcpm_reference_audio is not None:
        generate_audio_kwargs["voxcpm_reference_audio"] = voxcpm_reference_audio
    if voxcpm_prompt_audio is not None:
        generate_audio_kwargs["voxcpm_prompt_audio"] = voxcpm_prompt_audio
        generate_audio_kwargs["voxcpm_prompt_text"] = voxcpm_prompt_text
    audio_file, audio_duration, sub_maker = generate_audio(
        task_id,
        params,
        video_script,
        **generate_audio_kwargs,
    )
    if not audio_file:
        return _mark_task_failed(
            task_id,
            "audio",
            "failed to prepare narration audio",
        )

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    if stop_at == "audio":
        # Full video jobs apply gain in the final mixer. Audio-only exports have
        # no mixer, so apply it here to an owned output (never a custom input).
        audio_volume = 1.0 if params.voice_volume is None else float(params.voice_volume)
        if audio_volume != 1.0:
            descriptor, output_file = tempfile.mkstemp(
                prefix="audio-export-", suffix=".mp3", dir=utils.task_dir(task_id)
            )
            os.close(descriptor)
            export_ready = False
            try:
                export_ready = voice.apply_audio_volume(audio_file, output_file, audio_volume)
                if not export_ready:
                    return _mark_task_failed(task_id, "audio", "failed to apply audio export volume")
            finally:
                if not export_ready:
                    try:
                        os.remove(output_file)
                    except OSError as exc:
                        logger.warning(f"failed to remove incomplete audio export: {exc}")
            audio_file = output_file
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    generation_progress.stage("subtitle")
    # 4. Generate subtitle
    subtitle_path = generate_subtitle(
        task_id, params, video_script, sub_maker, audio_file
    )

    if (
        not subtitle_path and params.subtitle_enabled
        and (config.app.get("subtitle_provider") in {"auto", "whisper"}
             or getattr(sub_maker, "timing_source", "") == "estimated"
             or voice.is_gemini_voice(getattr(params, "voice_name", "") or ""))
    ):
        return _mark_task_failed(
            task_id, "subtitle", "No se pudieron sincronizar los subtítulos con el audio. Revisa Whisper e inténtalo de nuevo."
        )

    if stop_at == "subtitle":
        if not subtitle_path:
            return _mark_task_failed(
                task_id,
                "subtitle",
                "failed to generate subtitles; verify the subtitle provider and timeline",
            )
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    generation_progress.stage("materials")
    # 5. Get video materials
    downloaded_videos = get_video_materials(
        task_id,
        params,
        video_terms,
        audio_duration,
        loomloom_video_request=loomloom_video_request,
        **({"codex_image_context": codex_image_context}
           if codex_image_context is not None else {}),
        **({"jalapeno_video_endpoint": jalapeno_video_endpoint}
           if jalapeno_video_endpoint is not None else {}),
    )
    if not downloaded_videos:
        return _mark_task_failed(
            task_id,
            "materials",
            "failed to prepare video materials",
        )

    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    # Solo el flujo completo de generación de video necesita tratar el modo de concatenación de video;
    # así se evita que peticiones como /subtitle y /audio accedan a campos inexistentes.
    if type(params.video_concat_mode) is str:
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)

    generation_progress.stage("rendering")
    # 6. Generate final videos
    partial_results = {}
    try:
        final_video_paths, combined_video_paths, generation_warnings = generate_final_videos(
            task_id,
            params,
            downloaded_videos,
            audio_file,
            subtitle_path,
            audio_duration,
            partial_results=partial_results,
        )
    except Exception as exc:
        logger.exception(f"video batch failed, task_id: {task_id}, error: {exc}")
        # Warnings for the failing clip do not describe any usable output.
        completed_count = len(partial_results.get("videos", []))
        partial_results["warnings"] = [
            warning for warning in partial_results.get("warnings", [])
            if warning.get("video_index", 0) <= completed_count
        ] or None
        return _mark_task_failed(
            task_id, "video", f"{type(exc).__name__}: {exc}", details=partial_results,
        )

    if not final_video_paths:
        return _mark_task_failed(
            task_id,
            "video",
            "failed to generate final video",
        )

    logger.success(
        f"task {task_id} finished, generated {len(final_video_paths)} videos."
    )

    # 7. Primero se completa la tarea de generación de video y luego, si hace falta, se envía la publicación multiplataforma. La subida a terceros puede tardar
    # varios minutos y no debe bloquear la devolución del resultado del video ni afectar al video final ya generado.
    cross_post_enabled = (
        upload_post.upload_post_service.is_configured()
        and upload_post.upload_post_service.auto_upload
    )
    platforms = (
        list(upload_post.upload_post_service.platforms) if cross_post_enabled else []
    )
    should_cross_post = cross_post_enabled and bool(platforms)
    if cross_post_enabled and not platforms:
        logger.warning(
            f"skip cross-post because no platforms are configured, task_id: {task_id}"
        )
    cross_post_state = const.CROSS_POST_STATE_PENDING if should_cross_post else None

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "script": video_script,
        "terms": video_terms,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "materials": downloaded_videos,
        "cross_post_state": cross_post_state,
        "cross_post_results": None,
        "cross_post_error": None,
        "cross_post_owner": _cross_post_process_owner if should_cross_post else None,
        "warnings": generation_warnings or None,
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )

    if should_cross_post:
        scheduling_error = _schedule_cross_post(
            task_id=task_id,
            video_paths=final_video_paths,
            params=params,
            video_script=video_script,
            platforms=platforms,
            youtube_privacy_status=(
                upload_post.upload_post_service.youtube_privacy_status
            ),
            # Fija la selección de audiencia en el momento de encolar; cambiar después la WebUI no debe alterar la declaración de los videos ya encolados.
            youtube_made_for_kids=(
                upload_post.upload_post_service.youtube_made_for_kids
            ),
        )
        # Una cola llena o un pool de hilos cerrado son fallos de planificación conocibles de forma síncrona. La función de planificación ya actualizó el estado de la tarea
        # , y aquí se corrige de forma síncrona la instantánea devuelta, para que el llamador no reciba un pending incoherente con consultas posteriores.
        if scheduling_error:
            kwargs["cross_post_state"] = const.CROSS_POST_STATE_FAILED
            kwargs["cross_post_error"] = scheduling_error
            kwargs["cross_post_owner"] = None

    return kwargs


def start(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    voice_preview: dict | None = None,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
    allow_server_file_input: bool = False,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
    jalapeno_video_endpoint: volcengine_seedance.SeedanceEndpoint | None = None,
    codex_image_context: codex_image.CodexImageContext | None = None,
):
    """
    Ejecuta el pipeline de la tarea, garantizando que las excepciones imprevistas también se conviertan en un estado de fallo consultable.

    ``allow_server_file_input`` solo para el CLI local. La API HTTP y la WebUI deben mantener
    el valor por defecto, para que el audio personalizado esté siempre limitado al directorio de la tarea actual.
    """
    with generation_progress.task_scope(task_id):
        try:
            return _run_pipeline(
                task_id,
                params,
                stop_at=stop_at,
                voice_preview=voice_preview,
                loomloom_video_request=loomloom_video_request,
                allow_server_file_input=allow_server_file_input,
                voxcpm_reference_audio=voxcpm_reference_audio,
                voxcpm_prompt_audio=voxcpm_prompt_audio,
                voxcpm_prompt_text=voxcpm_prompt_text,
                **({"codex_image_context": codex_image_context}
                   if codex_image_context is not None else {}),
                **({"jalapeno_video_endpoint": jalapeno_video_endpoint}
                   if jalapeno_video_endpoint is not None else {}),
            )
        except Exception as exc:
            logger.exception(
                f"unexpected task pipeline failure, task_id: {task_id}, error: {exc}"
            )
            return _mark_task_failed(
                task_id,
                "pipeline",
                f"{type(exc).__name__}: {exc}",
            )


if __name__ == "__main__":
    task_id = "task_id"
    params = VideoParams(
        video_subject="La función del dinero",
        voice_name="zh-CN-XiaoyiNeural-Female",
        voice_rate=1.0,
    )
    start(task_id, params, stop_at="video")
