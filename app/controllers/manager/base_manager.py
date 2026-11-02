import math
import threading
from typing import Any, Callable, Dict

from loguru import logger


class TaskQueueFullError(ValueError):
    pass


def _coerce_task_limit(value: Any, name: str) -> int:
    """
    Convierte a enteros los límites de concurrencia / cola de la configuración.

    TOML El valor puede escribirse como `max_concurrent_tasks = 5` o como `"5"`. Una cadena
    llegaría hasta la comparación de `add_task` y se convertiría en un TypeError que no indica el nombre de la clave, y además
    `max_queued_tasks` el error solo aparecería cuando se agotaran los cupos de concurrencia.`app/services/webui_task.py`
    ya aplica la misma normalización a la clave `max_queued_tasks`; aquí se lleva al único punto de construcción.

    Cualquier valor que no se pueda interpretar como entero informa el nombre de la clave: `0.5` se truncaría en silencio a 0 con `int()` (las tareas solo se encolan,
    ningún worker las ejecuta), y `inf` / `-inf` / `nan`, permitidos por TOML, tampoco son límites válidos.
    0 y los negativos se mantienen tal cual: lo primero significa no ejecutar tareas por ahora; lo segundo hace que las peticiones vayan directamente a la cola.
    """
    if isinstance(value, bool):
        # bool es una subclase de int, pero `true` claramente no es el límite que quiere el usuario.
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, float):
        # Un valor no finito no es un límite, y un decimal degeneraría tras el truncado en la semántica «0: solo encolar sin ejecutar».
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite integer, got {value!r}")
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None


class TaskManager:
    def __init__(self, max_concurrent_tasks: int, max_queued_tasks: int = 100):
        self.max_concurrent_tasks = _coerce_task_limit(
            max_concurrent_tasks, "max_concurrent_tasks"
        )
        self.max_queued_tasks = _coerce_task_limit(max_queued_tasks, "max_queued_tasks")
        self.current_tasks = 0
        self.lock = threading.Lock()
        self.queue = self.create_queue()

    def create_queue(self):
        raise NotImplementedError()

    def add_task(self, func: Callable, *args: Any, **kwargs: Any):
        with self.lock:
            if self.current_tasks < self.max_concurrent_tasks:
                logger.info(
                    f"add task: {func.__name__}, current_tasks: {self.current_tasks}"
                )
                # Reserva el cupo de concurrencia antes de arrancar el hilo. La implementación original incrementaba dentro del hilo, y peticiones consecutivas
                # podían ver todas current_tasks=0 antes de que el hilo hijo obtuviera el lock, superando así el límite
                # de concurrencia. Si el arranque falla se devuelve el cupo para que las siguientes peticiones se puedan planificar con normalidad.
                self.current_tasks += 1
                try:
                    self.execute_task(func, *args, **kwargs)
                except Exception:
                    self.current_tasks -= 1
                    raise
            else:
                queue_size = self.queue_size()
                # Solo se encola cuando la concurrencia está llena. La cola debe tener un límite; de lo contrario una interfaz anónima podría seguir
                # acumulando objetos de tarea y parámetros de petición, agotando la memoria o disparando los costes de APIs de terceros.
                if queue_size >= self.max_queued_tasks:
                    logger.warning(
                        f"reject task: {func.__name__}, queue_size: {queue_size}, "
                        f"max_queued_tasks: {self.max_queued_tasks}"
                    )
                    raise TaskQueueFullError("task queue is full, please try again later")

                logger.info(
                    f"enqueue task: {func.__name__}, current_tasks: {self.current_tasks}, "
                    f"queue_size: {queue_size}"
                )
                self.enqueue_new_task({"func": func, "args": args, "kwargs": kwargs})

    def execute_task(self, func: Callable, *args: Any, **kwargs: Any):
        thread = threading.Thread(
            target=self.run_task, args=(func, *args), kwargs=kwargs
        )
        thread.start()

    def run_task(self, func: Callable, *args: Any, **kwargs: Any):
        try:
            func(*args, **kwargs)  # call the function here, passing *args and **kwargs.
        finally:
            self.task_done()

    def check_queue(self):
        with self.lock:
            self._check_queue_locked()

    def _check_queue_locked(self):
        """Dispatch one queued task while the caller holds ``self.lock``."""
        if (
            self.current_tasks < self.max_concurrent_tasks
            and not self.is_queue_empty()
        ):
            task_info = self.dequeue()
            if task_info is None:
                # dequeue() may skip and discard queue entries that no longer
                # pass current validation (see RedisTaskManager.dequeue) and
                # return None once nothing usable is left, even though
                # is_queue_empty() was False a moment earlier.
                return
            func = task_info["func"]
            args = task_info.get("args", ())
            kwargs = task_info.get("kwargs", {})
            # Reserve the freed slot before another add_task can claim it.
            self.current_tasks += 1
            try:
                self.execute_task(func, *args, **kwargs)
            except Exception:
                self.current_tasks -= 1
                self.enqueue(task_info)
                raise

    def task_done(self):
        with self.lock:
            self.current_tasks -= 1
            self._check_queue_locked()

    def enqueue(self, task: Dict):
        raise NotImplementedError()

    def enqueue_new_task(self, task: Dict):
        """Admit new work; shared queues may need their own atomic limit check."""
        self.enqueue(task)

    def dequeue(self):
        raise NotImplementedError()

    def is_queue_empty(self):
        raise NotImplementedError()

    def queue_size(self):
        raise NotImplementedError()
