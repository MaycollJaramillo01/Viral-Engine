"""Pasa el estado de las tareas largas de distintos proveedores al almacenamiento de tareas compartido, sin cambiar las peticiones facturadas ni la estrategia de reintentos."""

import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

from loguru import logger

from app.services import state as sm


_current = ContextVar("generation_progress", default=None)
_SUCCESS = {"succeeded", "completed", "success", "successful"}
_QUEUED = {"queued", "pending", "waiting", "created"}
_RUNNING = {"running", "processing", "in_progress", "executing", "active"}
_FAILED = {"failed", "error", "cancelled", "canceled", "expired"}
_IDENTIFIER = re.compile(r"^[\w.:-]{1,200}$")


class _Reporter:
    def __init__(self, task_id):
        self.task_id = task_id
        self.activity = {}
        self.lock = threading.RLock()
        self.last_log_at = 0

    def publish(self, phase, *, reset=False, **fields):
        # Mostrar el estado es una función auxiliar. Que Redis no esté disponible momentáneamente no puede interrumpir una tarea ya pagada, ni
        # provocar una nueva generación por reintentos ante excepciones. Aquí solo se escriben campos de lista blanca, sin aceptar respuestas ni URLs de descarga.
        try:
            with self.lock:
                now = time.time()
                previous = self.activity.get("phase")
                if (phase == previous == "downloading"
                        and now - self.activity.get("updated_at", 0) < 1):
                    return
                self.activity = {} if reset else dict(self.activity)
                self.activity.setdefault("started_at", now)
                self.activity.update(phase=phase, updated_at=now, **fields)
                sm.state.patch_task(self.task_id, generation_activity=self.activity)
                if previous != phase or now - self.last_log_at >= 30:
                    self.last_log_at = now
                    logger.info(
                        "generation activity: phase={}, provider={}, remote_id={}, elapsed={}s",
                        phase, self.activity.get("provider", ""),
                        self.activity.get("remote_id", ""),
                        int(max(0, now - self.activity["started_at"])),
                    )
        except Exception as exc:
            logger.debug("could not publish generation activity: {}", type(exc).__name__)


@contextmanager
def task_scope(task_id):
    token = _current.set(_Reporter(task_id))
    try:
        yield
    finally:
        _current.reset(token)


def stage(phase):
    reporter = _current.get()
    if reporter:
        reporter.publish(phase, reset=True)


def submitting(provider):
    reporter = _current.get()
    if reporter:
        reporter.publish("submitting", reset=True, provider=provider)


def remote(provider, task_id, status="", *, retrying=False):
    reporter = _current.get()
    if not reporter:
        return
    # El ID de tarea solo sirve para diagnóstico. Se rechaza que un gateway anómalo meta una URL/credenciales completas en este campo.
    task_id = str(task_id or "")
    task_id = task_id if _IDENTIFIER.fullmatch(task_id) else ""
    status = str(status or "").strip().lower()
    phase = (
        "checking" if retrying or not status else
        "generated" if status in _SUCCESS else
        "remote_queued" if status in _QUEUED else
        "generating" if status in _RUNNING else "checking"
    )
    fields = {"provider": provider or reporter.activity.get("provider", ""),
              "remote_id": task_id, "retrying": retrying}
    if not retrying and status:
        fields["confirmed_at"] = time.time()
        # Solo se conservan categorías de estado fijas, sin volcar texto arbitrario del gateway en la interfaz ni en los logs.
        fields["remote_status"] = (
            "succeeded" if status in _SUCCESS else
            "queued" if status in _QUEUED else
            "running" if status in _RUNNING else
            "failed" if status in _FAILED else "unknown"
        )
    reporter.publish(phase, **fields)


def downloading(downloaded=0, total=0):
    reporter = _current.get()
    if reporter:
        reporter.publish("downloading", downloaded_bytes=downloaded, total_bytes=total)


def verifying():
    reporter = _current.get()
    if reporter:
        reporter.publish("verifying")
