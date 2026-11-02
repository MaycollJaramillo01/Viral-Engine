"""Lectura y escritura seguras de archivos persistidos en el directorio de la tarea."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

from loguru import logger

from app.utils import utils


_SCRIPT_LOCKS = tuple(threading.RLock() for _ in range(256))


def _script_lock(task_id: str):
    return _SCRIPT_LOCKS[hash(task_id) % len(_SCRIPT_LOCKS)]


def _script_file(task_id: str) -> Path:
    """Devuelve la ruta del listado del guion de la tarea, reutilizando la lógica unificada de creación del directorio de la tarea."""
    return Path(utils.task_dir(task_id)) / "script.json"


def _write_json_atomic(target: Path, payload: Mapping[str, Any]) -> None:
    """
    Escribe JSON de forma atómica dentro del directorio de destino, para que una interrupción del proceso no deje medio archivo.

    El archivo temporal y el de destino deben estar en el mismo directorio para que ``os.replace`` mantenga la semántica de sustitución atómica en los
    sistemas de archivos locales habituales y en directorios montados por Docker. Hasta que la escritura tiene éxito no se modifica
    el archivo existente; ante una excepción solo se limpia el archivo temporal creado esta vez, y el error se pasa al llamador para que decida si
    afecta al flujo principal.
    """
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                payload,
                temp_file,
                ensure_ascii=False,
                indent=4,
                default=lambda value: value.__dict__,
            )
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_script_data(task_id: str, payload: Mapping[str, Any]) -> None:
    """Crea o sustituye por completo el listado ``script.json`` de la tarea."""
    with _script_lock(task_id):
        _write_json_atomic(_script_file(task_id), payload)


def patch_script_data(task_id: str, **updates: Any) -> bool:
    """
    Completa el listado de la tarea conservando los campos existentes; si falla devuelve ``False``.

    El origen del material es información de diagnóstico auxiliar y no puede bloquear la generación del video por permisos de archivo, anomalías momentáneas del disco o archivos históricos dañados
    . Por eso esta entrada registra la excepción completa y se degrada; la primera creación del listado de la tarea sigue usando
    ``write_script_data``, y el flujo principal decide qué hacer si falla la escritura de los datos básicos de la tarea.
    """
    try:
        with _script_lock(task_id):
            target = _script_file(task_id)
            with target.open("r", encoding="utf-8") as script_file:
                payload = json.load(script_file)
            if not isinstance(payload, dict):
                raise ValueError("task script data must be a JSON object")

            payload.update(updates)
            _write_json_atomic(target, payload)
            return True
    except FileNotFoundError:
        # ``download_videos`` también puede llamarse de forma independiente desde pruebas, scripts o código de terceros, y en ese caso no tener
        # listado de tarea es un escenario normal, que no debe generar advertencias ni crear archivos incompletos solo para el registro auxiliar.
        logger.debug(
            f"skip task script update because script.json does not exist: "
            f"task_id={task_id}"
        )
        return False
    except Exception as exc:
        logger.warning(
            "failed to update task script data: "
            f"task_id={task_id}, fields={sorted(updates)}, "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return False
