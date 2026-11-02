import inspect
import json
import threading
import time
from typing import Dict
from uuid import uuid4

import redis
from loguru import logger
from pydantic import ValidationError

from app.controllers.manager.base_manager import TaskManager, TaskQueueFullError
from app.models import const
from app.models.schema import AudioRequest, SubtitleRequest, VideoParams
from app.services import state as sm
from app.services import task as tm

FUNC_MAP = {
    "start": tm.start,
    # 'start_test': tm.start_test
}

_ADMIT_QUEUED_TASK_SCRIPT = """
if redis.call("LLEN", KEYS[1]) >= tonumber(ARGV[1]) then
    return 0
end
redis.call("RPUSH", KEYS[1], ARGV[2])
return 1
"""

# Running slots are stored separately from the persisted queue. A sorted set
# gives every slot a lease deadline, so a crashed worker cannot consume global
# capacity forever. Active workers renew their own member periodically.
_ADMIT_TASK_SCRIPT = """
redis.call("ZREMRANGEBYSCORE", KEYS[2], "-inf", ARGV[1])
local running = redis.call("ZCARD", KEYS[2])
local queued = redis.call("LLEN", KEYS[1])
if running < tonumber(ARGV[2]) and queued == 0 then
    redis.call("ZADD", KEYS[2], ARGV[3], ARGV[4])
    return 1
end
if queued >= tonumber(ARGV[5]) then
    return -1
end
redis.call("RPUSH", KEYS[1], ARGV[6])
return 0
"""

_CLAIM_QUEUED_TASK_SCRIPT = """
redis.call("ZREMRANGEBYSCORE", KEYS[2], "-inf", ARGV[1])
if redis.call("ZCARD", KEYS[2]) >= tonumber(ARGV[2]) then
    return nil
end
local task = redis.call("LPOP", KEYS[1])
if not task then
    return nil
end
redis.call("ZADD", KEYS[2], ARGV[3], ARGV[4])
return task
"""

_RENEW_RUNNING_SLOT_SCRIPT = """
if redis.call("ZSCORE", KEYS[1], ARGV[1]) then
    redis.call("ZADD", KEYS[1], "XX", ARGV[2], ARGV[1])
    return 1
end
return 0
"""

_RUNNING_SLOT_LEASE_SECONDS = 90.0
_RUNNING_SLOT_HEARTBEAT_SECONDS = 30.0
_QUEUE_DISPATCH_RETRY_SECONDS = 5.0


class RedisTaskManager(TaskManager):
    def __init__(
        self,
        max_concurrent_tasks: int,
        redis_url: str,
        max_queued_tasks: int = 100,
    ):
        self.redis_client = redis.Redis.from_url(redis_url)
        self._dispatch_context = threading.local()
        self._worker_context = threading.local()
        self._dispatcher_stop = threading.Event()
        self._dispatcher_thread = None
        self._dispatcher_lock = threading.Lock()
        super().__init__(max_concurrent_tasks, max_queued_tasks=max_queued_tasks)

    def create_queue(self):
        return "task_queue"

    @property
    def running_slots_key(self) -> str:
        # Tests and deployments can override ``queue`` after construction; the
        # semaphore must follow the selected queue namespace automatically.
        return f"{self.queue}:running"

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _lease_deadline_ms(self) -> int:
        return self._now_ms() + int(_RUNNING_SLOT_LEASE_SECONDS * 1000)

    def add_task(self, func, *args, **kwargs):
        # Drain older persisted work first. Besides preserving FIFO order, this
        # lets a request that arrives after a crashed lease expires free one
        # queue position before its own atomic admission check.
        if self.max_concurrent_tasks > 0:
            self.check_queue()
        task = {"func": func, "args": args, "kwargs": kwargs}
        token = uuid4().hex
        admitted = self.redis_client.eval(
            _ADMIT_TASK_SCRIPT,
            2,
            self.queue,
            self.running_slots_key,
            self._now_ms(),
            self.max_concurrent_tasks,
            self._lease_deadline_ms(),
            token,
            self.max_queued_tasks,
            self._serialize_task(task),
        )
        if admitted == -1:
            raise TaskQueueFullError("task queue is full, please try again later")
        if admitted == 1:
            logger.info(f"add task: {func.__name__}, global slot: {token}")
            self._start_reserved_task(token, task, restore_on_failure=False)
            return

        logger.info(f"enqueue task: {func.__name__}, queue_size: {self.queue_size()}")
        # If a crashed worker's lease was pruned during admission, immediately
        # give the oldest queued item the newly available global slot.
        self.check_queue()

    def resume_queued_tasks(self):
        """Dispatch persisted queue entries when a new API process starts.

        Queued Redis items survive a restart, but no worker calls check_queue
        until an active task finishes. Fill each available local worker slot so
        an idle process does not leave old requests stuck in processing.
        """
        for _ in range(max(0, self.max_concurrent_tasks)):
            self.check_queue()

    def start_queue_dispatcher(self):
        """Keep persisted work moving after a process restart.

        A replacement process may start before a crashed worker's Redis lease
        expires. The one-shot startup recovery cannot claim the queued task in
        that case, so keep a lifecycle-owned dispatcher waiting for the next
        lease deadline. Normal completions and new requests still dispatch
        immediately through task_done() and add_task().
        """
        with self._dispatcher_lock:
            if (
                self._dispatcher_thread is not None
                and self._dispatcher_thread.is_alive()
            ):
                return
            self._dispatcher_stop.clear()
            self._dispatcher_thread = threading.Thread(
                target=self._dispatch_queued_until_stopped,
                name="redis-queue-dispatcher",
                daemon=True,
            )
            self._dispatcher_thread.start()

    def stop_queue_dispatcher(self):
        """Stop and join the lifecycle-owned queue dispatcher."""
        with self._dispatcher_lock:
            thread = self._dispatcher_thread
            self._dispatcher_stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_QUEUE_DISPATCH_RETRY_SECONDS)
        with self._dispatcher_lock:
            if self._dispatcher_thread is thread:
                self._dispatcher_thread = None

    def _dispatch_queued_until_stopped(self):
        while not self._dispatcher_stop.is_set():
            try:
                self.check_queue()
                delay = self._next_queue_dispatch_delay()
            except redis.RedisError as exc:
                logger.error(f"Redis queue dispatcher failed: {exc}")
                delay = _QUEUE_DISPATCH_RETRY_SECONDS
            self._dispatcher_stop.wait(delay)

    def _next_queue_dispatch_delay(self) -> float:
        """Return when recovery should retry, preferring the next lease expiry."""
        if self.queue_size() == 0:
            return _QUEUE_DISPATCH_RETRY_SECONDS
        earliest = self.redis_client.zrange(
            self.running_slots_key, 0, 0, withscores=True
        )
        if not earliest:
            # A queue entry appeared after check_queue() returned. Retry soon
            # instead of waiting for unrelated traffic.
            return 0.01
        deadline_ms = earliest[0][1]
        return max(0.01, (deadline_ms - self._now_ms()) / 1000)

    def check_queue(self):
        while True:
            token = uuid4().hex
            payload = self.redis_client.eval(
                _CLAIM_QUEUED_TASK_SCRIPT,
                2,
                self.queue,
                self.running_slots_key,
                self._now_ms(),
                self.max_concurrent_tasks,
                self._lease_deadline_ms(),
                token,
            )
            if payload is None:
                return

            task = self._deserialize_task(payload)
            if task is None:
                self._release_slot(token)
                # A stale entry must not prevent the next valid item from
                # running. Iterate instead of recursing through an unbounded
                # queue of entries left by an older deployment.
                continue
            self._start_reserved_task(token, task, restore_on_failure=True)
            return

    def _start_reserved_task(self, token: str, task: Dict, restore_on_failure: bool):
        func = task["func"]
        args = task.get("args", ())
        kwargs = task.get("kwargs", {})
        with self.lock:
            self.current_tasks += 1
        self._dispatch_context.slot_token = token
        try:
            self.execute_task(func, *args, **kwargs)
        except Exception:
            with self.lock:
                self.current_tasks -= 1
            self._release_slot(token)
            if restore_on_failure:
                self.enqueue(task)
            raise
        finally:
            self._dispatch_context.slot_token = None

    def execute_task(self, func, *args, **kwargs):
        token = getattr(self._dispatch_context, "slot_token", None)
        if token is None:
            raise RuntimeError("Redis task execution requires a reserved slot")
        thread = threading.Thread(
            target=self._run_reserved_task,
            args=(token, func, *args),
            kwargs=kwargs,
        )
        thread.start()

    def _run_reserved_task(self, token: str, func, *args, **kwargs):
        stop_heartbeat = threading.Event()
        heartbeat = threading.Thread(
            target=self._renew_slot_until_done,
            args=(token, stop_heartbeat),
            daemon=True,
        )
        heartbeat.start()
        self._worker_context.slot_token = token
        try:
            func(*args, **kwargs)
        finally:
            stop_heartbeat.set()
            self.task_done()
            self._worker_context.slot_token = None

    def _renew_slot_until_done(self, token: str, stop: threading.Event):
        while not stop.wait(_RUNNING_SLOT_HEARTBEAT_SECONDS):
            try:
                renewed = self.redis_client.eval(
                    _RENEW_RUNNING_SLOT_SCRIPT,
                    1,
                    self.running_slots_key,
                    token,
                    self._lease_deadline_ms(),
                )
                if not renewed:
                    logger.error(f"lost Redis running slot lease: {token}")
                    return
            except redis.RedisError as exc:
                # A temporary Redis outage must not kill the user task. The
                # next heartbeat can still renew before the lease expires.
                logger.error(f"failed to renew Redis running slot {token}: {exc}")

    def _release_slot(self, token: str):
        try:
            self.redis_client.zrem(self.running_slots_key, token)
        except redis.RedisError as exc:
            # The lease will expire even if Redis is temporarily unavailable.
            logger.error(f"failed to release Redis running slot {token}: {exc}")

    def task_done(self):
        token = getattr(self._worker_context, "slot_token", None)
        if token is not None:
            self._release_slot(token)
        with self.lock:
            self.current_tasks = max(0, self.current_tasks - 1)
        self.check_queue()

    def enqueue(self, task: Dict):
        self.redis_client.rpush(self.queue, self._serialize_task(task))

    def enqueue_new_task(self, task: Dict):
        # Manager locks are process-local. Keep capacity validation and admission
        # in one Redis operation so other API workers cannot claim the same slot.
        admitted = self.redis_client.eval(
            _ADMIT_QUEUED_TASK_SCRIPT,
            1,
            self.queue,
            self.max_queued_tasks,
            self._serialize_task(task),
        )
        if not admitted:
            raise TaskQueueFullError("task queue is full, please try again later")

    @staticmethod
    def _serialize_task(task: Dict) -> str:
        task_with_serializable_params = task.copy()
        # task.copy() solo copia el diccionario exterior; si se modificaran directamente los kwargs anidados, el VideoParams
        # que tiene el llamador se reemplazaría por un dict. Los logs o reintentos posteriores podrían seguir leyendo la tarea original,
        # así que aquí se copian los kwargs por separado para que la serialización no tenga efectos secundarios inesperados.
        task_kwargs = task.get("kwargs", {})
        task_with_serializable_params["kwargs"] = task_kwargs.copy()

        if "params" in task_kwargs and isinstance(
            task_kwargs["params"], (VideoParams, AudioRequest, SubtitleRequest)
        ):
            task_with_serializable_params["kwargs"]["params"] = task_kwargs[
                "params"
            ].model_dump(warnings=False)

        # Convierte el objeto función en su nombre
        task_with_serializable_params["func"] = task["func"].__name__
        return json.dumps(task_with_serializable_params)

    def dequeue(self):
        # Un bucle en lugar de un único pop: una tarea podía cumplir las reglas de validación al encolarse, pero las reglas y
        # FUNC_MAP los miembros cambian con cada despliegue (por ejemplo VideoParams añade una restricción ge=1, o se elimina alguna
        # función de entrada), así que en la cola pueden quedar entradas escritas con el schema antiguo o que ya no se pueden analizar.
        # lpop es una operación destructiva; una vez hecho el pop no se puede devolver a su sitio; la tarea ya salió de la cola de forma permanente
        # y no se puede fingir que sigue ahí. En lugar de dejar que la excepción suba desde aquí (check_queue llama
        # a este método con el lock, y la excepción derribaría el hilo de trabajo a través del finally de task_done → run_task; después
        # no habría ninguna tarea corriendo, nadie volvería a llamar a check_queue y las tareas siguientes de la cola quedarían para siempre
        # en processing), es mejor descartarla en el sitio y probar con la siguiente, manteniendo el contrato de "obtener una tarea utilizable
        # o que la cola esté realmente vacía".
        while True:
            task_json = self.redis_client.lpop(self.queue)
            # Solo cuando lpop no devuelve nada significa que la cola está vacía. Una cadena vacía (o bytes vacíos) también
            # es una entrada inutilizable y puede tener tareas válidas detrás, así que debe seguir la ruta de descarte
            # de abajo en lugar de devolver directamente como "fin de la cola".
            if task_json is None:
                return None

            task_info = self._deserialize_task(task_json)
            if task_info is not None:
                return task_info

    def _deserialize_task(self, task_json):
        """Restore one popped queue payload, or discard it as stale."""
        task_info = None
        try:
            task_info = json.loads(task_json)
            # Convierte el nombre de función de vuelta en objeto función. Si el nombre falta o ya no está en FUNC_MAP no se puede
            # indexar directamente; de lo contrario un KeyError esquivaría la estrategia de descarte para params de abajo.
            task_info["func"] = FUNC_MAP[task_info["func"]]
            task_kwargs = task_info["kwargs"]
            if not isinstance(task_kwargs, dict):
                raise ValueError("queued task has no keyword argument mapping")
            # args Si falta por completo se usa el valor por defecto de check_queue; si es null u otra forma que no sea
            # un array, check_queue lanzaría TypeError al expandir `*args`, y allí
            # la entrada se volvería a encolar y la excepción escaparía del hilo de trabajo, así que hay que interceptarlo aquí.
            if not isinstance(task_info.get("args", []), list):
                raise ValueError("queued task positional arguments are not a list")
            # A persisted request can outlive a callable's signature. Fail
            # it before dispatch: Python argument-binding errors happen
            # before start() enters its pipeline failure handler.
            inspect.signature(task_info["func"]).bind(
                *task_info.get("args", []), **task_kwargs
            )
        except (TypeError, ValueError, KeyError) as e:
            logger.error(f"dropping unusable queued task: {e}")
            # Igual que la ruta de fallo de validación de params de abajo: si se puede leer un task_id válido,
            # esta tarea que ya salió de la cola para siempre se marca como fallida; de lo contrario la API/WebUI la mostraría siempre
            # en processing. Si el payload no se puede analizar o task_id no es una cadena
            # (por ejemplo un array JSON), no hay registro que actualizar y solo se puede descartar; pasar algo que no sea cadena
            # a patch_task haría que redis lanzara DataError, interrumpiendo a su vez el bucle de descarte.
            stale_kwargs = (
                task_info.get("kwargs") if isinstance(task_info, dict) else None
            )
            task_id = (
                stale_kwargs.get("task_id")
                if isinstance(stale_kwargs, dict)
                else None
            )
            if isinstance(task_id, str) and task_id:
                sm.state.patch_task(
                    task_id,
                    state=const.TASK_STATE_FAILED,
                    failed_stage="dequeue",
                    error=f"discarded stale queued task: {e}",
                )
            return None

        if "params" in task_kwargs and isinstance(task_kwargs["params"], dict):
            try:
                params_model = VideoParams
                # /audio and /subtitle use short-form request models without
                # video_subject. Preserve those schemas at the queue boundary;
                # full VideoParams can also stop at either of these stages.
                if "video_subject" not in task_kwargs["params"]:
                    if task_kwargs.get("stop_at") == "audio":
                        params_model = AudioRequest
                    elif task_kwargs.get("stop_at") == "subtitle":
                        params_model = SubtitleRequest
                task_kwargs["params"] = params_model(**task_kwargs["params"])
            except ValidationError as e:
                logger.error(
                    "dropping queued task with params that fail current "
                    f"request model validation (queued under an older, more "
                    f"permissive schema, or corrupted): {e}"
                )
                # El registro de estado de la tarea se crea antes de encolar y por defecto es processing; si solo se
                # descartara la entrada de la cola sin tocar el registro, la API/WebUI mostraría siempre la tarea
                # en ejecución y nunca pasaría a fallida. Se usa patch_task en lugar de update_task
                # para no volver a crear la tarea si el usuario ya la eliminó.
                # task_id Si no es una cadena, igual que antes: no hay registro que actualizar y se omite la actualización de estado.
                task_id = task_kwargs.get("task_id")
                if isinstance(task_id, str) and task_id:
                    sm.state.patch_task(
                        task_id,
                        state=const.TASK_STATE_FAILED,
                        failed_stage="dequeue",
                        error=f"discarded stale queued task: {e}",
                    )
                return None

        return task_info

    def is_queue_empty(self):
        return self.redis_client.llen(self.queue) == 0

    def queue_size(self):
        return self.redis_client.llen(self.queue)
