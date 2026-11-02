import copy
import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import asdict

from loguru import logger

from app.config import config
from app.controllers.manager.memory_manager import InMemoryTaskManager
from app.models import const
from app.models.schema import VideoParams
from app.services import state as sm
from app.services import task as tm
from app.services import codex_image, jalapeno_video, volcengine_seedance
from app.services.loomloom import LoomLoomConfirmedVideoRequest
from app.utils.logging_utils import (
    format_log_record,
    log_scope_thread_id,
    runtime_log_filter,
)


# WebUI la configuración se guarda en un diccionario global a nivel de proceso. La implementación síncrona original mantenía durante toda la generación
# runtime_config_lock, así que en la práctica distintas sesiones del navegador también se ejecutaban en serie. Aquí la concurrencia se fija
# en 1, manteniendo la coherencia de configuración original y evitando que varios hilos solo esperen inútilmente fuera del lock de configuración.
_task_manager = InMemoryTaskManager(
    max_concurrent_tasks=1,
    max_queued_tasks=max(1, int(config.app.get("max_queued_tasks", 100))),
)
_task_logs: dict[str, deque[str]] = {}
_task_logs_lock = threading.RLock()
_MAX_LOG_TASKS = 20
_MAX_LOG_RECORDS_PER_TASK = 1000
_submission_lock = threading.RLock()
_active_submissions: dict[str, str] = {}
# Streamlit no puede recibir actualizaciones de componentes enviadas directamente por hilos en segundo plano, solo mediante sondeo con Fragment. 0.5 segundos
# bastan para que el log de la WebUI se acerque a la salida en tiempo real del terminal, sin consumir recursos del navegador como un refresco de alta frecuencia.
TASK_LOG_REFRESH_INTERVAL_SECONDS = 0.5


def is_generation_active(task_id: str) -> bool:
    """se basa en el cupo reservado en segundo plano, cubriendo la ventana entre encolar y publicar el estado."""
    with _submission_lock:
        return task_id in _active_submissions.values()


def _submission_signature(
    params, references, request, endpoint, preview, codex_image_context=None
):
    # Solo se guarda un resumen, sin credenciales ni contenido subido. Distintas Keys/modelos/entradas son peticiones distintas,
    # y el mismo contenido se puede volver a generar al terminar; solo se unen los envíos duplicados que siguen en cola o en ejecución.
    request_inputs = asdict(request) if request is not None else None
    if request_inputs is not None:
        # Dos pestañas generan cada una un ID idempotente para la misma entrada de cotización; eso no es una diferencia de contenido.
        # Se conservan la cuenta, el modelo, las líneas de entrada y la versión de la cotización, y se continúa con la primera petición confirmada.
        request_inputs.pop("client_request_id", None)
    payload = {
        "params": params.model_dump(mode="json", warnings=False),
        "settings": {
            name: config.snapshot_config_with_pending(getattr(config, name))
            for name in ("app", "azure", "elevenlabs", "siliconflow", "chatterbox")
        },
        "references": [hashlib.sha256(value or b"").hexdigest() for value in references],
        "request": request_inputs,
        "endpoint": asdict(endpoint) if endpoint is not None else None,
        # El origen de la voz cambia la salida real; la ruta temporal cambia con el ID de tarea local y no es una diferencia de entrada.
        "voice_preview": {
            key: preview[key]
            for key in ("script", "voice_name", "voice_rate", "voice_volume", "duration", "params", "audio_digest")
            if key in preview
        } if preview else None,
    }
    if codex_image_context is not None:
        payload["codex_image"] = asdict(codex_image_context)
    if preview and not preview.get("audio_digest"):
        # Cuando un llamador antiguo no tiene resumen de contenido, no se puede deducir que dos archivos son iguales solo por tener la misma duración.
        # No se leen rutas arbitrarias antes de validar; es preferible mantener peticiones independientes para archivos distintos.
        payload["voice_preview"]["audio_file"] = preview.get("audio_file")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _append_task_log(task_id: str, message: str) -> None:
    """Guarda un número limitado de logs por tarea, para que el Fragment de Streamlit los sondee de forma segura."""
    with _task_logs_lock:
        records = _task_logs.get(task_id)
        if records is None:
            # Solo se conservan los logs de las tareas recientes, para que el servicio WebUI no siga ocupando memoria tras mucho tiempo en marcha.
            # dict Mantiene el orden de inserción; los logs de tareas solo sirven para diagnóstico en la interfaz, y descartar los registros más antiguos no afecta a las tareas.
            if len(_task_logs) >= _MAX_LOG_TASKS:
                oldest_task_id = next(iter(_task_logs))
                _task_logs.pop(oldest_task_id, None)
            records = deque(maxlen=_MAX_LOG_RECORDS_PER_TASK)
            _task_logs[task_id] = records
        records.append(message.rstrip())


def get_task_logs(task_id: str) -> list[str]:
    """Devuelve una instantánea de los logs, para no mantener durante el renderizado de la página el lock que usa el hilo en segundo plano."""
    with _task_logs_lock:
        return list(_task_logs.get(task_id, ()))


def _run_generation(
    task_id: str,
    params: VideoParams,
    capture_logs: bool,
    voice_preview: dict | None = None,
    loomloom_video_request: LoomLoomConfirmedVideoRequest | None = None,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
    jalapeno_video_endpoint: volcengine_seedance.SeedanceEndpoint | None = None,
    codex_image_context: codex_image.CodexImageContext | None = None,
) -> dict:
    """
    Ejecuta el pipeline de video existente en un hilo en segundo plano.

    Loguru El sink de es un recurso a nivel de proceso, así que hay que filtrar por el hilo de trabajo actual. De lo contrario, los logs de
    API tareas que se ejecutan a la vez o de otras páginas se mezclarían con la tarea actual. Los hilos auxiliares que la tarea arranca para descargas, codificación de fragmentos y latidos
    se asignan al hilo de trabajo mediante bind_log_scope, y sus logs también deben recogerse; de lo contrario
    esas fases largas no mostrarían ninguna salida en la WebUI. La página solo lee una instantánea de lista normal, sin acceder desde el hilo
    en segundo plano al session_state de Streamlit, evitando de raíz el desorden de rutas delta al refrescar.
    """
    log_handler_id = None
    worker_thread_id = threading.get_ident()
    try:
        if capture_logs:
            log_handler_id = logger.add(
                lambda message: _append_task_log(task_id, str(message)),
                level=0,
                format=format_log_record,
                colorize=False,
                filter=lambda record: (
                    log_scope_thread_id(record["thread"].id) == worker_thread_id
                    and runtime_log_filter(record)
                ),
            )

        # La tarea completa sigue usando el lock de configuración original, para impedir que otra sesión de la WebUI modifique a mitad de la generación
        # Provider, claves u otra configuración a nivel de proceso, haciendo que un mismo video use ajustes distintos de principio a fin.
        with config.runtime_config_lock():
            return tm.start(
                task_id=task_id,
                params=params,
                voice_preview=voice_preview,
                loomloom_video_request=loomloom_video_request,
                voxcpm_reference_audio=voxcpm_reference_audio,
                voxcpm_prompt_audio=voxcpm_prompt_audio,
                voxcpm_prompt_text=voxcpm_prompt_text,
                **({"codex_image_context": codex_image_context}
                   if codex_image_context is not None else {}),
                **({"jalapeno_video_endpoint": jalapeno_video_endpoint}
                   if jalapeno_video_endpoint is not None else {}),
            )
    except Exception as exc:
        # tm.start ya se encarga de convertir las excepciones del pipeline en estado de fallo; aquí se protegen además el sink de logs,
        # el lock de configuración y demás capas envolventes de la WebUI. Cualquier excepción del hilo en segundo plano debe dejar un estado final, sin que el gestor
        # de tareas siga mostrando para siempre «Generando» después de que salga el hilo de trabajo.
        error = f"{type(exc).__name__}: {exc}"
        failure = {
            "task_id": task_id,
            "state": const.TASK_STATE_FAILED,
            "progress": 0,
            "failed_stage": "webui_worker",
            "error": error,
        }
        sm.state.update_task(
            task_id,
            state=failure["state"],
            progress=failure["progress"],
            failed_stage=failure["failed_stage"],
            error=failure["error"],
        )
        logger.exception(
            f"unexpected WebUI generation worker failure, "
            f"task_id={task_id}, error={exc}"
        )
        return failure
    finally:
        with _submission_lock:
            for signature, active_id in list(_active_submissions.items()):
                if active_id == task_id:
                    del _active_submissions[signature]
        if log_handler_id is not None:
            try:
                logger.remove(log_handler_id)
            except ValueError:
                logger.debug(
                    f"WebUI task log handler already removed: task_id={task_id}"
                )


def submit_generation(
    task_id: str,
    params: VideoParams,
    capture_logs: bool = True,
    voice_preview: dict | None = None,
    loomloom_video_request: LoomLoomConfirmedVideoRequest | None = None,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
    jalapeno_video_endpoint: volcengine_seedance.SeedanceEndpoint | None = None,
) -> str:
    """
    Registra y envía la tarea de generación de video de la WebUI, y vuelve de inmediato tras la llamada.

    El estado de la tarea debe escribirse antes de arrancar el hilo. Así, al terminar esta ejecución del script de la página ya se puede consultar la tarea,
    y refrescar el navegador o reconectar el WebSocket no depende de marcadores en la memoria de la página antigua.
    """
    task_params = params.model_copy(deep=True)
    # The subtitle timeline and nested preview parameters remain mutable in
    # Streamlit session state. Snapshot them before this job enters the queue.
    voice_preview_snapshot = copy.deepcopy(voice_preview) if voice_preview else None
    # Reference audio belongs only to this queued request. It is deliberately
    # separate from VideoParams so it cannot reach task history, presets, state
    # persistence, or logs. ``bytes`` is immutable; make an explicit snapshot
    # before the background worker starts so later WebUI reruns cannot share
    # mutable upload state with this task.
    voxcpm_reference_audio_snapshot = (
        bytes(voxcpm_reference_audio) if voxcpm_reference_audio else None
    )
    voxcpm_prompt_audio_snapshot = (
        bytes(voxcpm_prompt_audio) if voxcpm_prompt_audio else None
    )
    voxcpm_prompt_text_snapshot = str(voxcpm_prompt_text or "")
    # Una petición confirmada es un objeto de datos congelado que solo se transmite dentro del proceso actual. La API Key no entra en
    # VideoParams, el estado de la tarea, los logs ni el historial en disco, y tampoco le afectan los reruns posteriores de la página.
    loomloom_request_snapshot = loomloom_video_request
    # Durante la cola se puede preparar el siguiente contenido, pero la configuración de pago confirmada no puede cambiar con el formulario global.
    # La clave de la instantánea no participa en repr; solo viaja con los parámetros de la tarea dentro del proceso y no se escribe en VideoParams.
    if task_params.video_source == "jalapeno" and jalapeno_video_endpoint is None:
        jalapeno_video_endpoint = jalapeno_video.endpoint(
            config.snapshot_config_with_pending(config.app)
        )
    codex_image_context = (
        codex_image.capture_context(config.snapshot_config_with_pending(config.app))
        if task_params.video_source == "codex_image"
        else None
    )
    signature = _submission_signature(
        task_params,
        (
            voxcpm_reference_audio_snapshot,
            voxcpm_prompt_audio_snapshot,
            voxcpm_prompt_text_snapshot.encode(),
        ),
        loomloom_request_snapshot,
        jalapeno_video_endpoint,
        voice_preview_snapshot,
        codex_image_context,
    )
    # La comprobación de duplicados y la reserva se hacen dentro del mismo lock, así que aunque dos pestañas hagan clic a la vez solo se encola una vez.
    with _submission_lock:
        existing = _active_submissions.get(signature)
        if existing:
            return existing
        if task_id in _active_submissions.values():
            return task_id
        _active_submissions[signature] = task_id
    try:
        _enqueue_generation(
            task_id,
            task_params,
            capture_logs,
            voice_preview_snapshot,
            loomloom_request_snapshot,
            voxcpm_reference_audio_snapshot,
            voxcpm_prompt_audio_snapshot,
            voxcpm_prompt_text_snapshot,
            jalapeno_video_endpoint,
            codex_image_context,
        )
    except Exception:
        with _submission_lock:
            if _active_submissions.get(signature) == task_id:
                del _active_submissions[signature]
        raise
    return task_id


def _enqueue_generation(
    task_id,
    task_params,
    capture_logs,
    voice_preview_snapshot,
    loomloom_request_snapshot,
    voxcpm_reference_audio_snapshot,
    voxcpm_prompt_audio_snapshot,
    voxcpm_prompt_text_snapshot,
    jalapeno_video_endpoint,
    codex_image_context=None,
):
    submitted_at = time.time()
    sm.state.update_task(
        task_id,
        state=const.TASK_STATE_PROCESSING,
        progress=0,
        video_subject=task_params.video_subject or task_params.video_script or task_id,
        generation_submitted_at=submitted_at,
        generation_activity={"phase": "queued", "started_at": submitted_at,
                             "updated_at": submitted_at},
    )
    try:
        _task_manager.add_task(
            _run_generation,
            task_id=task_id,
            params=task_params,
            capture_logs=capture_logs,
            voice_preview=voice_preview_snapshot,
            loomloom_video_request=loomloom_request_snapshot,
            voxcpm_reference_audio=voxcpm_reference_audio_snapshot,
            voxcpm_prompt_audio=voxcpm_prompt_audio_snapshot,
            voxcpm_prompt_text=voxcpm_prompt_text_snapshot,
            **({"codex_image_context": codex_image_context}
               if codex_image_context is not None else {}),
            **({"jalapeno_video_endpoint": jalapeno_video_endpoint}
               if jalapeno_video_endpoint is not None else {}),
        )
    except Exception as exc:
        # Un fallo de planificación, igual que un fallo del pipeline, debe convertirse en un estado consultable, para que el gestor de tareas no muestre para siempre
        # «Generando». Se conserva el tipo de excepción para localizar rápidamente problemas de cola en los logs de Docker o locales.
        error = f"{type(exc).__name__}: {exc}"
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_FAILED,
            progress=0,
            failed_stage="scheduling",
            error=error,
        )
        logger.exception(
            f"failed to submit WebUI generation task, task_id={task_id}, error={exc}"
        )
        raise
