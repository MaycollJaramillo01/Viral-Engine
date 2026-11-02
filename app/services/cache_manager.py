"""Servicio de estadísticas, vista previa y limpieza de la caché de material de video."""

from __future__ import annotations

import os
import re
import stat
import time
from dataclasses import dataclass
from typing import Iterator

from loguru import logger

from app.utils import utils


# El material en línea usa el MD5 de la URL como nombre de archivo estable. La gestión de caché solo acepta ese formato de nombre, para no
# borrar como caché videos, archivos de notas u otros archivos de negocio que el usuario haya puesto por error en el directorio.
_VIDEO_CACHE_FILE_PATTERN = re.compile(r"^vid-[0-9a-f]{32}\.mp4$")
# save_video writes to this same-directory name before atomically publishing a
# validated clip. A forced process exit can leave one behind, so include only
# aged temporary files in cache cleanup and never touch an active download.
_VIDEO_CACHE_TEMP_FILE_PATTERN = re.compile(
    r"^\.vid-[0-9a-f]{32}-[a-z0-9_]+\.mp4$"
)
_SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class VideoCacheStats:
    """Resultado estadístico ligero del directorio de caché; solo contiene metadatos del sistema de archivos."""

    file_count: int = 0
    total_size: int = 0
    oldest_mtime: float | None = None
    newest_mtime: float | None = None


@dataclass(frozen=True)
class VideoCacheCleanupResult:
    """Resultado de una limpieza; se permite que falle el borrado de algunos archivos."""

    deleted_count: int = 0
    deleted_size: int = 0
    failed_count: int = 0


@dataclass(frozen=True)
class _VideoCacheEntry:
    """Información mínima de archivo guardada en la fase de escaneo, para no abrir ni analizar videos durante la limpieza."""

    path: str
    name: str
    size: int
    mtime: float
    device: int
    inode: int
    mtime_ns: int


def video_cache_dir() -> str:
    """Devuelve el directorio de caché de video por defecto gestionado por el proyecto."""

    return os.path.realpath(utils.storage_dir("cache_videos"))


def _iter_video_cache_entries(include_temp: bool = False) -> Iterator[_VideoCacheEntry]:
    """
    Escanea secuencialmente el primer nivel del directorio de caché por defecto.

    Se usa ``os.scandir`` para reutilizar los metadatos que devuelve el recorrido del directorio cuando la caché llega a decenas de miles de archivos,
    evitando volver a consultar el tipo de archivo tras ``Path.iterdir``. Aquí no se recorre recursivamente, no se abren videos ni se llama a
    FFmpeg, así que el tiempo depende sobre todo linealmente del número de archivos, no del tamaño total de los videos.
    """

    cache_dir = video_cache_dir()
    try:
        entries = os.scandir(cache_dir)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning(
            f"failed to scan video cache directory: path={cache_dir}, error={exc}"
        )
        return

    with entries:
        for entry in entries:
            if not _VIDEO_CACHE_FILE_PATTERN.fullmatch(entry.name) and not (
                include_temp and _VIDEO_CACHE_TEMP_FILE_PATTERN.fullmatch(entry.name)
            ):
                continue

            try:
                # No se siguen enlaces simbólicos, para que la lógica de limpieza no salga de los límites del directorio de caché por defecto.
                if not entry.is_file(follow_symlinks=False):
                    continue
                # DirEntry.stat() reports st_dev/st_ino as 0 on Windows, which
                # would never match the os.stat() identity check before deletion.
                stat_result = os.stat(entry.path, follow_symlinks=False)
            except OSError as exc:
                logger.warning(
                    f"failed to inspect video cache file: file={entry.name}, error={exc}"
                )
                continue

            yield _VideoCacheEntry(
                path=entry.path,
                name=entry.name,
                size=stat_result.st_size,
                mtime=stat_result.st_mtime,
                device=stat_result.st_dev,
                inode=stat_result.st_ino,
                mtime_ns=stat_result.st_mtime_ns,
            )


def _is_cleanup_candidate(
    entry: _VideoCacheEntry,
    max_age_days: int | None,
    now: float,
) -> bool:
    if (
        _VIDEO_CACHE_TEMP_FILE_PATTERN.fullmatch(entry.name)
        and entry.mtime >= now - _SECONDS_PER_DAY
    ):
        return False
    if max_age_days is None:
        return True
    return entry.mtime < now - max_age_days * _SECONDS_PER_DAY


def _validate_max_age_days(max_age_days: int | None) -> None:
    """Aunque el directorio de caché esté vacío, se deben rechazar de forma estable los parámetros de limpieza inválidos."""
    if max_age_days is None:
        return
    if (
        isinstance(max_age_days, bool)
        or not isinstance(max_age_days, int)
        or max_age_days <= 0
    ):
        raise ValueError("max_age_days must be a positive integer or None")


def get_video_cache_stats(max_age_days: int | None = None) -> VideoCacheStats:
    """
    Calcula estadísticas de toda la caché o previsualiza la caché limpiable con fecha de modificación anterior a los días indicados.

    ``max_age_days=None`` significa toda la caché. El cálculo solo lee el tamaño y la fecha de modificación de las entradas del directorio,
    sin leer el contenido de los videos, así que aunque la caché sea muy grande no genera E/S proporcional a su tamaño.
    """

    _validate_max_age_days(max_age_days)
    now = time.time()
    file_count = 0
    total_size = 0
    oldest_mtime = None
    newest_mtime = None

    for entry in _iter_video_cache_entries(include_temp=True):
        if not _is_cleanup_candidate(entry, max_age_days, now):
            continue
        file_count += 1
        total_size += entry.size
        oldest_mtime = (
            entry.mtime if oldest_mtime is None else min(oldest_mtime, entry.mtime)
        )
        newest_mtime = (
            entry.mtime if newest_mtime is None else max(newest_mtime, entry.mtime)
        )

    return VideoCacheStats(
        file_count=file_count,
        total_size=total_size,
        oldest_mtime=oldest_mtime,
        newest_mtime=newest_mtime,
    )


def clean_video_cache(max_age_days: int | None = None) -> VideoCacheCleanupResult:
    """
    Limpia la caché de video por defecto y devuelve un resumen que se puede mostrar al usuario.

    Entre la vista previa en la página y el clic real en limpiar puede pasar mucho tiempo, así que al ejecutar hay que volver a escanear y evaluar,
    sin reutilizar la lista de candidatos antigua. El borrado tolera fallos archivo a archivo: si un archivo está en uso o sin permisos se registra
    una advertencia y se continúa, para que un archivo problemático entre cientos no haga fallar toda la limpieza.
    """

    _validate_max_age_days(max_age_days)
    now = time.time()
    logger.info(
        f"start cleaning video cache: max_age_days={max_age_days}"
    )

    candidate_count = 0
    candidate_size = 0
    deleted_count = 0
    deleted_size = 0
    failed_count = 0
    cache_dir = video_cache_dir()

    # Se borra a medida que se escanea, sin mantener en memoria la lista completa de candidatos. Aunque el directorio crezca a cientos de miles de archivos,
    # la memoria adicional de la limpieza se mantiene constante; se usa un now único al ejecutar, para que en limpiezas largas
    # el instante de corte no se desplace y produzca un rango de candidatos impredecible.
    for entry in _iter_video_cache_entries(include_temp=True):
        if not _is_cleanup_candidate(entry, max_age_days, now):
            continue
        candidate_count += 1
        candidate_size += entry.size
        try:
            # entry.path viene del scandir del primer nivel del directorio por defecto; antes de borrar se vuelven a validar el directorio padre y
            # el nombre de archivo, para evitar ampliar por accidente el rango borrable si en el futuro cambia la lógica de escaneo.
            if (
                os.path.realpath(os.path.dirname(entry.path)) != cache_dir
                or not (
                    _VIDEO_CACHE_FILE_PATTERN.fullmatch(entry.name)
                    or _VIDEO_CACHE_TEMP_FILE_PATTERN.fullmatch(entry.name)
                )
                or os.path.islink(entry.path)
            ):
                raise ValueError("cache file is outside the managed directory")
            try:
                current = os.stat(entry.path, follow_symlinks=False)
            except FileNotFoundError:
                # Another cleanup already removed the scanned candidate.
                continue
            if not stat.S_ISREG(current.st_mode) or (
                current.st_dev,
                current.st_ino,
                current.st_mtime_ns,
                current.st_size,
            ) != (entry.device, entry.inode, entry.mtime_ns, entry.size):
                # A downloader may atomically publish a new file at this name
                # after the scan. Do not delete the fresh replacement.
                continue
            os.unlink(entry.path)
            deleted_count += 1
            deleted_size += entry.size
        except (OSError, ValueError) as exc:
            failed_count += 1
            logger.warning(
                f"failed to delete video cache file: file={entry.name}, error={exc}"
            )

    logger.info(
        "finished cleaning video cache: "
        f"candidates={candidate_count}, candidate_bytes={candidate_size}, "
        f"deleted={deleted_count}, deleted_bytes={deleted_size}, failed={failed_count}"
    )
    return VideoCacheCleanupResult(
        deleted_count=deleted_count,
        deleted_size=deleted_size,
        failed_count=failed_count,
    )
