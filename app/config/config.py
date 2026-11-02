import copy
import errno
import json
import os
import shutil
import socket
import tempfile
import threading
from contextlib import contextmanager

import toml
from loguru import logger

from app import __version__, paths

root_dir = str(paths.runtime_root())
config_file = f"{root_dir}/config.toml"
_CONTAINER_CGROUP_MARKERS = ("docker", "containerd", "kubepods", "libpod", "podman")
_DOCKER_HOST_GATEWAY_NAME = "host.docker.internal"
_config_save_lock = threading.RLock()
_diagnostics_update_lock = threading.Lock()
_pending_config_lock = threading.RLock()
_pending_config_updates = {}
_pending_config_save_requested = False
_pending_config_flush_scheduled = False
_MISSING = object()
_DELETE = object()
_UTF8_BOM = "\ufeff"


class _SynchronizedConfig(dict):
    """Mantiene el uso como dict sin cambios, pero hace que las escrituras de configuración en tiempo de ejecución respeten el mismo lock."""

    def __setitem__(self, key, value):
        # Streamlit Cada rerun completo de la página vuelve a escribir los valores actuales de los controles en la configuración. Mientras la tarea de video tiene
        # runtime_config_lock , si el valor no ha cambiado esta escritura no tiene ningún efecto secundario, y
        # no debe dejar la página refrescada atascada a mitad del formulario. Las escrituras que realmente cambian la configuración siguen entrando en el lock de abajo,
        # así que no se puede cambiar de Provider, clave u otros ajustes globales a mitad de la generación de un video.
        current = super().get(key, _MISSING)
        if current is not _MISSING and current == value:
            return
        with _config_save_lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with _config_save_lock:
            super().__delitem__(key)

    def clear(self):
        if not self:
            return
        with _config_save_lock:
            super().clear()

    def pop(self, key, default=_MISSING):
        # ``pop(key, default)`` tampoco cambia la configuración cuando la key no existe. La WebUI usa
        # esta forma para expresar «usar la estrategia por defecto», y al refrescar debe poder completarse directamente.
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        with _config_save_lock:
            if default is _MISSING:
                return super().pop(key)
            return super().pop(key, default)

    def setdefault(self, key, default=None):
        # Igual que __setitem__, setdefault sobre una key existente es una operación de solo lectura. Devolver antes
        # permite que el refresco de páginas que solo leen la configuración por defecto no se vea afectado por el lock de tareas largas.
        current = super().get(key, _MISSING)
        if current is not _MISSING:
            return current
        with _config_save_lock:
            return super().setdefault(key, default)

    def update(self, *args, **kwargs):
        changes = dict(*args, **kwargs)
        if all(
            (current := dict.get(self, key, _MISSING)) is not _MISSING
            and current == value
            for key, value in changes.items()
        ):
            return
        with _config_save_lock:
            super().update(changes)


def _pending_update_key(config_section, key):
    """Genera la clave pendiente de actualización para una partición de configuración fija del proceso."""
    return id(config_section), key


def update_config_nonblocking(config_section, key, value):
    """
    Actualiza sin bloquear la configuración en tiempo de ejecución de la WebUI.

    La generación de video mantiene ``runtime_config_lock`` para asegurar que una misma tarea no cambie a mitad de ejecución
    Provider, claves o configuración de voz. Cuando cambia un control de Streamlit no se puede esperar a este lock de tarea larga,
    o el navegador parecerá congelado. Si el lock está libre se actualiza de inmediato; si está ocupado solo se guarda el último valor
    de cada elemento de configuración y se aplica todo cuando la tarea actual libera el lock.

    Devolver True significa que el valor ya está en vigor; False significa que entró en la cola de actualizaciones pendientes.
    """
    return update_config_values_nonblocking(config_section, {key: value})


def update_config_values_nonblocking(config_section, changes):
    """Encola juntos los campos relacionados de una misma operación, para no separarlos al aplicarse de forma concurrente."""
    # Todas las actualizaciones entran primero en la misma cola y luego se intenta obtener el lock de configuración. Así, cuando varias páginas modifican a la vez el mismo
    # elemento, el orden de escritura en la cola es el orden final, y no puede ocurrir que un hilo anterior, tras obtener el lock,
    # borre por error un valor más reciente que otro hilo ya había encolado.
    with _pending_config_lock:
        for key, value in changes.items():
            _pending_config_updates[_pending_update_key(config_section, key)] = (
                config_section,
                key,
                copy.deepcopy(value),
            )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        # El llamador normalmente pide guardar al final de este rerun de Streamlit, pero no se puede depender de que este paso
        # se ejecute siempre. Por ejemplo, si la página falla a mitad o la actualización ocurre justo en la fase de guardado al salir de la tarea, sigue siendo necesario
        # que un hilo de refresco en segundo plano garantice que los valores encolados acaben aplicándose.
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return all(
            config_section.get(key, _MISSING) == value
            for key, value in changes.items()
        )
    finally:
        _config_save_lock.release()


def update_logging_options(level, log_requests):
    """Publica en serie la instantánea de ejecución y la intención de persistencia, sin esperar al lock largo de configuración de la tarea de generación."""
    from app.utils.logging_utils import get_logging_options, set_logging_options

    with _diagnostics_update_lock:
        set_logging_options(level, log_requests)
        level, log_requests = get_logging_options()
        return update_config_values_nonblocking(
            diagnostics, {"log_level": level, "log_requests": log_requests}
        )


def delete_config_nonblocking(config_section, key):
    """
    Elimina sin bloquear un elemento de configuración de la WebUI.

    «Usar el valor por defecto» requiere eliminar realmente el elemento, no escribir una cadena vacía. Cuando la tarea de video ocupa el lock
    de configuración, la intención de borrado sobrescribe las actualizaciones encoladas antes para ese elemento y se ejecuta al terminar la tarea.
    """
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            _DELETE,
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return key not in config_section
    finally:
        _config_save_lock.release()


def _apply_pending_config_updates_locked():
    """Aplica los últimos valores de configuración pendientes de la WebUI mientras se mantiene el lock de escritura de configuración."""
    with _pending_config_lock:
        updates = list(_pending_config_updates.values())
        _pending_config_updates.clear()
        # Mientras se aplica la configuración se sigue manteniendo el lock de pendientes. Así, los hilos que leen la instantánea «valor actual + valor pendiente»
        # solo pueden ver el estado completo de antes o después de aplicar, nunca un conjunto de configuración actualizado a medias.
        for config_section, key, value in updates:
            if value is _DELETE:
                config_section.pop(key, None)
            else:
                config_section[key] = value
    return bool(updates)


def snapshot_config_with_pending(config_section):
    """
    Devuelve la instantánea efectiva de una partición de configuración, combinando las actualizaciones de la WebUI aún no aplicadas.

    Mientras la tarea de video mantiene el lock no se puede reescribir la configuración global, pero el usuario puede preparar el siguiente contenido. Las peticiones LLM
    usan esta instantánea, así que el Provider, modelo y clave recién elegidos en la interfaz se usan en la nueva petición, sin
    cambiar la tarea de video en curso.
    """
    with _pending_config_lock:
        snapshot = dict(config_section)
        section_id = id(config_section)
        for (pending_section_id, key), (_, _, value) in _pending_config_updates.items():
            if pending_section_id != section_id:
                continue
            if value is _DELETE:
                snapshot.pop(key, None)
            else:
                snapshot[key] = copy.deepcopy(value)
    return snapshot


def _flush_pending_config_locked(*, suppress_save_errors):
    """Aplica y guarda todas las configuraciones pendientes mientras se mantiene el lock de escritura de configuración."""
    global _pending_config_save_requested

    updates_applied = _apply_pending_config_updates_locked()
    with _pending_config_lock:
        save_requested = _pending_config_save_requested
        _pending_config_save_requested = False

    if not updates_applied and not save_requested:
        return True

    try:
        save_config()
        return True
    except Exception as exc:
        # La configuración en memoria ya se aplicó correctamente; si el guardado falla solo se conserva la marca de pendiente de guardar. La tarea de video no debe
        # marcarse como fallida porque el archivo de configuración no sea escribible temporalmente; la siguiente interacción con la página volverá a guardar.
        with _pending_config_lock:
            _pending_config_save_requested = True
        if not suppress_save_errors:
            raise
        logger.exception(f"failed to save deferred runtime config: {exc}")
        return False


def _run_deferred_config_flush():
    """Espera a que la tarea larga libere el lock de configuración y vacía de forma fiable las actualizaciones acumuladas mientras tanto."""
    global _pending_config_flush_scheduled

    while True:
        with _config_save_lock:
            flush_succeeded = _flush_pending_config_locked(
                suppress_save_errors=True
            )

        with _pending_config_lock:
            has_pending_work = bool(
                _pending_config_updates or _pending_config_save_requested
            )
            if not flush_succeeded or not has_pending_work:
                _pending_config_flush_scheduled = False
                return


def _schedule_deferred_config_flush():
    """Garantiza que como máximo un hilo en segundo plano espere para refrescar la configuración a la vez."""
    global _pending_config_flush_scheduled

    with _pending_config_lock:
        if _pending_config_flush_scheduled:
            return
        _pending_config_flush_scheduled = True

    try:
        threading.Thread(
            target=_run_deferred_config_flush,
            name="mpt-config-flush",
            daemon=True,
        ).start()
    except Exception:
        # No worker owns this reservation when construction/start fails.
        # Leave queued updates and the save request intact so a later call
        # can schedule another flush rather than silently waiting forever.
        with _pending_config_lock:
            _pending_config_flush_scheduled = False
        raise


def try_save_config():
    """
    Guarda sin bloquear la configuración de la WebUI; si el lock está ocupado, se guarda cuando termine la tarea larga actual.

    La API normal, el CLI y los scripts de mantenimiento pueden seguir llamando a ``save_config`` con la semántica de escritura bloqueante original;
    solo el rerun de Streamlit usa esta función, para evitar que la página deje de responder mucho tiempo esperando la tarea de video.
    """
    global _pending_config_save_requested

    with _pending_config_lock:
        _pending_config_save_requested = True

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        return _flush_pending_config_locked(suppress_save_errors=False)
    finally:
        _config_save_lock.release()


@contextmanager
def runtime_config_lock():
    """
    Impide que otras sesiones de la WebUI reescriban la configuración durante una operación completa que depende de la configuración global.

    El proyecto se enlaza por defecto a la dirección de loopback local y la configuración sigue siendo global de un solo usuario. Este lock ligero sirve principalmente para
    proteger operaciones largas como generar o previsualizar, evitando que otra pestaña cambie de Provider o clave a mitad de la operación.
    """
    with _config_save_lock:
        # Si al liberar el lock una operación corta el hilo de refresco en segundo plano aún no se ha planificado, la nueva tarea debe, antes de leer
        # Provider, claves y demás configuración global, aplicar primero la cola, y no seguir ejecutando todo el pipeline con la configuración antigua.
        _flush_pending_config_locked(suppress_save_errors=True)
        try:
            yield
        finally:
            _flush_pending_config_locked(suppress_save_errors=True)


@contextmanager
def try_runtime_config_lock():
    """
    Intenta obtener el lock de configuración de ejecución y devuelve inmediatamente si lo consiguió.

    WebUI La previsualización de audio es una operación corta iniciada por el usuario y no debe esperar varios minutos mientras una tarea de video en segundo plano tiene el lock.
    Si no se obtiene el lock, el llamador puede avisar al usuario para que reintente más tarde; si se obtiene, se garantiza que durante la previsualización
    Provider, claves y configuración del modelo no serán modificadas por otras sesiones.
    """
    acquired = _config_save_lock.acquire(blocking=False)
    try:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
        yield acquired
    finally:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
            _config_save_lock.release()


def is_running_in_container(
    dockerenv_path: str = "/.dockerenv",
    containerenv_path: str = "/run/.containerenv",
    cgroup_path: str = "/proc/1/cgroup",
) -> bool:
    """
    Determina si el proceso actual se ejecuta dentro de un contenedor.

    Esta comprobación se usa sobre todo para elegir la dirección por defecto de Ollama:
    - en una ejecución local normal, `localhost` apunta a la máquina del usuario;
    - Docker dentro de un contenedor, `localhost` apunta al propio contenedor, y para acceder al host Ollama
      normalmente hay que usar `host.docker.internal`.

    No basta con comprobar si existe `/proc/1/cgroup`, porque un Linux normal también tiene ese archivo.
    Aquí solo se devuelve True cuando se detecta un marcador claro de contenedor, para no afectar a usuarios de Linux sin Docker.
    El parámetro se deja como ruta inyectable para que las pruebas unitarias cubran distintos entornos de ejecución.
    """
    if os.path.isfile(dockerenv_path) or os.path.isfile(containerenv_path):
        return True

    try:
        with open(cgroup_path, mode="r", encoding="utf-8") as fp:
            cgroup_content = fp.read().lower()
    except OSError:
        return False

    return any(marker in cgroup_content for marker in _CONTAINER_CGROUP_MARKERS)


def _can_resolve_hostname(hostname: str) -> bool:
    try:
        socket.gethostbyname(hostname)
    except OSError:
        return False
    return True


def _decode_linux_route_gateway(hex_gateway: str) -> str:
    # /proc/net/route El Gateway en él está en hexadecimal little-endian; por ejemplo, 010011AC significa
    # 172.17.0.1. Se analiza aparte para que, cuando Docker nativo en Linux no tenga el registro
    # host.docker.internal DNS , todavía se pueda intentar acceder al host a través de la puerta de enlace por defecto del contenedor.
    if len(hex_gateway) != 8:
        raise ValueError("invalid gateway length")

    octets = [
        str(int(hex_gateway[index : index + 2], 16)) for index in range(6, -1, -2)
    ]
    return ".".join(octets)


def get_container_default_gateway_ip(route_path: str = "/proc/net/route") -> str:
    """
    Lee la IP de la puerta de enlace por defecto dentro de un contenedor Linux.

    Docker Desktop normalmente proporciona `host.docker.internal`, pero el nativo Linux Docker
    no siempre proporciona ese nombre DNS por defecto. La puerta de enlace por defecto suele servir para acceder a servicios del host
    como dirección de respaldo; si el Ollama del usuario solo escucha en 127.0.0.1, el usuario tendrá que hacer que
    Ollama escuche en la interfaz de red del host o configurar manualmente `ollama_base_url`.
    """
    try:
        with open(route_path, mode="r", encoding="utf-8") as fp:
            route_lines = fp.readlines()
    except OSError:
        return ""

    for line in route_lines[1:]:
        fields = line.strip().split()
        if len(fields) < 8:
            continue

        destination = fields[1]
        gateway = fields[2]
        if (
            destination != "00000000"
            or fields[7] != "00000000"
            or gateway == "00000000"
        ):
            continue

        try:
            return _decode_linux_route_gateway(gateway)
        except ValueError:
            logger.warning(f"invalid container gateway route entry: {line.strip()}")
            return ""

    return ""


def get_default_ollama_base_url() -> str:
    """
    Devuelve el base_url OpenAI-compatible por defecto de Ollama.

    Cuando el usuario configura explícitamente `ollama_base_url` no se pasa por aquí; esto solo resuelve «el mejor valor por defecto
    cuando no está configurado«. Dentro de un contenedor apunta al host por defecto; en una ejecución local normal apunta a localhost.
    """
    if not is_running_in_container():
        return "http://localhost:11434/v1"

    if _can_resolve_hostname(_DOCKER_HOST_GATEWAY_NAME):
        return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"

    gateway_ip = get_container_default_gateway_ip()
    if gateway_ip:
        logger.info(
            "host.docker.internal is not resolvable, fallback to container "
            f"default gateway for Ollama: {gateway_ip}"
        )
        return f"http://{gateway_ip}:11434/v1"

    logger.warning(
        "failed to resolve host.docker.internal and container default gateway; "
        "fallback to host.docker.internal for Ollama"
    )
    return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"


def _load_toml_config(config_path: str):
    """
    Carga TOML, tolerando los BOM UTF-8 duplicados que pueden escribir los editores de Windows.

    ``utf-8-sig`` solo elimina un BOM al inicio del archivo. Algunos editores de Windows o
    procesos de descompresión y guardado pueden volver a escribir el BOM, haciendo que un segundo carácter invisible entre en TOML
    el analizador y falle en la primera línea. Aquí solo se normaliza en modo lectura una vez que falla el análisis estándar,
    sin reescribir el archivo original, para no sobrescribir por accidente las API Keys que el usuario ya rellenó.
    """
    try:
        return toml.load(config_path)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "load config failed, retry with UTF-8 BOM compatibility: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )

    try:
        with open(config_path, mode="r", encoding="utf-8-sig") as fp:
            config_content = fp.read()

        normalized_content = config_content.lstrip(_UTF8_BOM)
        removed_bom_count = len(config_content) - len(normalized_content)
        if removed_bom_count:
            logger.warning(
                "removed repeated UTF-8 BOM characters while loading config: "
                f"path={config_path}, count={removed_bom_count}"
            )
        return toml.loads(normalized_content)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.error(
            "config file is not valid TOML after UTF-8 BOM normalization: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )
        raise



def _initialize_config_from_example(example_file):
    """Publish a complete first-boot config without replacing another writer."""
    fd, temp_path = tempfile.mkstemp(
        prefix=".config-init-", suffix=".toml.tmp",
        dir=os.path.dirname(os.path.abspath(config_file)),
    )
    try:
        with os.fdopen(fd, "wb") as destination:
            with open(example_file, "rb") as source:
                shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            if os.name == "nt":
                # Windows rename refuses an existing destination. POSIX rename
                # overwrites it, so use an exclusive hard-link publication there.
                os.rename(temp_path, config_file)
            else:
                os.link(temp_path, config_file)
        except FileExistsError:
            return False
        except OSError as exc:
            raise OSError(
                exc.errno,
                "cannot safely initialize config.toml; copy config.example.toml "
                "to config.toml before starting the application",
            ) from exc
        return True
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            # A successful Windows rename consumed the temporary path.
            pass


def load_config():
    os.makedirs(root_dir, mode=0o700, exist_ok=True)
    # Docker may create an empty directory when a missing config.toml is bind-mounted.
    # Only remove that empty placeholder; a nonempty directory may contain user data.
    if os.path.isdir(config_file):
        try:
            os.rmdir(config_file)
        except FileNotFoundError:
            # Another initializer already removed this empty bind-mount stub.
            pass
        except OSError as exc:
            if exc.errno == errno.EBUSY:
                # The directory is the bind mount itself, so it cannot be
                # removed from inside the container; it must be fixed on the host.
                raise IsADirectoryError(
                    f"{config_file} is a directory because config.toml did not exist "
                    "on the host when Docker mounted it. Stop the containers, delete "
                    "the empty config.toml folder next to the docker-compose file, "
                    "copy config.example.toml to config.toml there, then start again"
                ) from exc
            # Another initializer may already have replaced the empty Docker
            # stub with its complete regular file. Accept only that race, not
            # permission failures or a still-present/nonempty directory.
            if not (
                isinstance(exc, NotADirectoryError) and os.path.isfile(config_file)
            ):
                raise IsADirectoryError(
                    f"{config_file} is a directory and cannot be used as the config file; "
                    "move or rename it, then restart"
                ) from exc

    if not os.path.isfile(config_file):
        example_file = f"{root_dir}/config.example.toml"
        if not os.path.isfile(example_file):
            example_file = str(paths.config_template())
        if os.path.isfile(example_file):
            if _initialize_config_from_example(example_file):
                logger.info("copy config.example.toml to config.toml")

    logger.info(f"load config from file: {config_file}")

    return _load_toml_config(config_file)


class _ConfigTomlEncoder(toml.TomlEncoder):
    """
    Write strings as JSON strings, which are also valid TOML basic strings.

    toml 0.10.2's own string encoder (uiri/toml#404) drops the escaping of
    every backslash up to the last one followed by "x", so a Windows path
    with a folder such as "xiaoming" or "x64" is written as invalid TOML.
    It also writes a non-breaking space as the text "xa0".
    """

    def __init__(self):
        super().__init__()
        self.dump_funcs[str] = self._dump_str

    @staticmethod
    def _dump_str(value):
        # TOML, unlike JSON, also requires DEL to be escaped.
        return json.dumps(str(value), ensure_ascii=False).replace("\x7f", "\\u007f")


def save_config():
    """
    Guarda de forma atómica la configuración en tiempo de ejecución.

    Streamlit Distintas sesiones de pueden disparar el guardado de configuración casi al mismo tiempo. Si se sobrescribe config.toml directamente,
    otro hilo podría leer un TOML escrito solo a medias. Aquí se usa un lock reentrante del proceso para serializar
    el guardado, escribiendo primero un archivo temporal en el mismo directorio y sustituyendo después el destino de forma atómica con os.replace.

    Docker Desktop Un bind mount de un solo archivo convierte el propio config.toml en punto de montaje,
    Linux y el kernel no permite sustituir un punto de montaje con rename/replace, por lo que devuelve EBUSY.
    En ese caso solo se puede sobrescribir el archivo en el sitio dentro del lock; las demás excepciones se siguen lanzando para no ocultar errores de permisos, disco
    o ruta.

    Esto mantiene la semántica actual del proyecto de configuración global de un solo usuario, sin introducir un sistema complejo de configuración multiusuario;
    principalmente para evitar corromper el archivo de configuración con varias pestañas o reruns rápidos.
    """
    with _config_save_lock:
        config_to_save = dict(_cfg)
        config_to_save["app"] = dict(app)
        config_to_save["azure"] = dict(azure)
        config_to_save["siliconflow"] = dict(siliconflow)
        config_to_save["minimax_tts"] = dict(minimax_tts)
        config_to_save["elevenlabs"] = dict(elevenlabs)
        config_to_save["chatterbox"] = dict(chatterbox)
        config_to_save["kokoro"] = dict(kokoro)
        config_to_save["fish_audio"] = dict(fish_audio)
        config_to_save["voxcpm"] = dict(voxcpm)
        config_to_save["ui"] = dict(ui)
        config_to_save["diagnostics"] = dict(diagnostics)
        serialized_config = toml.dumps(config_to_save, encoder=_ConfigTomlEncoder())

        # WebUI Al terminar un rerun completo se llama al guardado. Si el contenido no cambió se devuelve directamente, para evitar que cada
        # clic en un control normal provoque una escritura en disco y un fsync.
        try:
            with open(config_file, mode="r", encoding="utf-8") as f:
                if f.read() == serialized_config:
                    _cfg.clear()
                    _cfg.update(config_to_save)
                    return
        except (OSError, UnicodeError):
            pass

        temp_path = ""
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=".config-",
                suffix=".toml.tmp",
                dir=root_dir,
            )
            with os.fdopen(fd, mode="w", encoding="utf-8") as f:
                f.write(serialized_config)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(temp_path, config_file)
            except OSError as exc:
                if exc.errno != errno.EBUSY:
                    raise

                logger.warning(
                    "atomic config replacement is unavailable for the mounted "
                    f"file, fallback to in-place write: {config_file}"
                )
                with open(config_file, mode="w", encoding="utf-8") as f:
                    f.write(serialized_config)
                    f.flush()
                    os.fsync(f.fileno())
            _cfg.clear()
            _cfg.update(config_to_save)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


_cfg = load_config()
diagnostics = _SynchronizedConfig(_cfg.get("diagnostics", {}))
app = _SynchronizedConfig(_cfg.get("app", {}))
whisper = _cfg.get("whisper", {})
proxy = _cfg.get("proxy", {})
azure = _SynchronizedConfig(_cfg.get("azure", {}))
siliconflow = _SynchronizedConfig(_cfg.get("siliconflow", {}))
minimax_tts = _SynchronizedConfig(_cfg.get("minimax_tts", {}))
elevenlabs = _SynchronizedConfig(_cfg.get("elevenlabs", {}))
chatterbox = _SynchronizedConfig(_cfg.get("chatterbox", {}))
kokoro = _SynchronizedConfig(_cfg.get("kokoro", {}))
fish_audio = _SynchronizedConfig(_cfg.get("fish_audio", {}))
voxcpm = _SynchronizedConfig(_cfg.get("voxcpm", {}))
ui = _SynchronizedConfig(
    _cfg.get(
        "ui",
        {
            "hide_log": False,
        },
    )
)

hostname = socket.gethostname()

log_level = _cfg.get("log_level", "DEBUG")
listen_host = _cfg.get("listen_host", "0.0.0.0")
listen_port = _cfg.get("listen_port", 8080)
project_name = _cfg.get("project_name", "MoneyPrinterTurbo")
project_description = _cfg.get(
    "project_description",
    "<a href='https://github.com/harry0703/MoneyPrinterTurbo'>https://github.com/harry0703/MoneyPrinterTurbo</a>",
)
project_version = _cfg.get("project_version", __version__)
reload_debug = False

app["redis_host"] = os.getenv(
    "MPT_APP_REDIS_HOST",
    os.getenv("REDIS_HOST", app.get("redis_host", "localhost")),
)

ffmpeg_path = app.get("ffmpeg_path", "")
if ffmpeg_path:
    # config.toml También se permiten las formas "~" / "${VAR}": se expanden primero y luego se comprueba si existen;
    # de lo contrario isfile() nunca se cumpliría y la configuración se ignoraría en silencio.
    ffmpeg_path = os.path.expanduser(os.path.expandvars(ffmpeg_path))
    if os.path.isfile(ffmpeg_path):
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

# El IMAGEIO_FFMPEG_EXE que el usuario define directamente también puede usar las formas "~" / "${VAR}"
# (issue #1662). MoviePy 2.x lee esta variable a través de imageio al importarse
# y la fija como moviepy.config.FFMPEG_BINARY, sin volver a analizarla después; por eso
# la expansión debe hacerse en este módulo (la dependencia común más temprana de todos los puntos de entrada, importada antes que moviepy),
# para que la fase de codificación obtenga la ruta real.
_env_ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE")
if _env_ffmpeg:
    os.environ["IMAGEIO_FFMPEG_EXE"] = os.path.expanduser(
        os.path.expandvars(_env_ffmpeg)
    )

logger.info(f"{project_name} v{project_version}")
