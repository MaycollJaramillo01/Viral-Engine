import functools
import os
import re
import sys
import threading

from loguru import logger


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
_PROJECT_ROOT_IN_MESSAGE_RE = re.compile(re.escape(PROJECT_ROOT) + r"(?![\w.\-])")
LOG_RECORD_FORMAT = (
    "<green>{time:%Y-%m-%d %H:%M:%S}</> | "
    "<level>{level}</> | "
    '"{file.path}:{line}":<blue> {function}</> '
    "- <level>{message}</>\n"
)
# Loguru Al arrancar, el ID del handler de terminal por defecto es 0. Cuando la WebUI se recarga solo puede sustituir esta
# salida de terminal básica, y no puede llamar a logger.remove() para vaciar todos los handlers; de lo contrario también se borraría el sink temporal
# que usan las tareas en ejecución para recoger los logs de la WebUI.
_terminal_handler_id: int | None = 0
_terminal_handler_lock = threading.RLock()
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
# Solo sustituye instantáneas inmutables; el hilo de logs no obtiene el lock de configuración que mantiene la tarea de generación.
_runtime_logging = ("DEBUG", False)


def get_logging_options() -> tuple[str, bool]:
    return _runtime_logging


def set_logging_options(level: str, log_requests: bool = False) -> None:
    """Ajusta al instante los logs de este proceso, sin reconstruir handlers ni interrumpir las tareas en ejecución."""
    global _runtime_logging
    level = str(level).strip().upper()
    if level not in LOG_LEVELS:
        raise ValueError(f"unsupported log level: {level}")
    if not isinstance(log_requests, bool):
        raise ValueError("log_requests must be a boolean")
    _runtime_logging = (level, log_requests)


def runtime_log_filter(record) -> bool:
    level, log_requests = _runtime_logging
    # El trazado de peticiones es un interruptor independiente; incluso con nivel ERROR se permite que el usuario active explícitamente el diagnóstico.
    if record["extra"].get("request_trace"):
        return log_requests
    return record["level"].no >= logger.level(level).no


# WebUI Filtra los logs por el ID del hilo de trabajo de la tarea. Los hilos auxiliares que la tarea arranca para descargas en paralelo, codificación de fragmentos y latidos
# tienen sus propios IDs de hilo, y los logs que escriben se descartarían por completo, con el resultado de que las fases más largas
# no mostrarían ninguna salida en la WebUI. Aquí se registra el mapeo «hilo auxiliar -> hilo que lo lanzó»,
# para que el filtro de logs pueda devolver los registros de los hilos auxiliares a su tarea.
_log_scope_roots: dict[int, int] = {}
_log_scope_lock = threading.Lock()


def log_scope_thread_id(thread_id: int | None = None) -> int:
    """Devuelve el ID del hilo raíz del ámbito de log al que pertenece el hilo; un hilo no vinculado es él mismo."""
    if thread_id is None:
        thread_id = threading.get_ident()
    with _log_scope_lock:
        return _log_scope_roots.get(thread_id, thread_id)


def bind_log_scope(func):
    """
    Envuelve una función que se ejecutará en otro hilo, para que sus logs pertenezcan al hilo que llama a esta función.

    Debe llamarse en el hilo que envía la tarea: el ámbito se fija al envolver, y los hilos que arranquen los hilos auxiliares
    también se asignan a la tarea original. Al terminar se desvincula, porque el sistema reutiliza los IDs de hilo
    y un mapeo residual contaría los logs de hilos no relacionados posteriores en una tarea ya terminada.
    """
    root_thread_id = log_scope_thread_id()

    @functools.wraps(func)
    def run_in_log_scope(*args, **kwargs):
        thread_id = threading.get_ident()
        if thread_id == root_thread_id:
            return func(*args, **kwargs)

        with _log_scope_lock:
            previous_root = _log_scope_roots.get(thread_id)
            _log_scope_roots[thread_id] = root_thread_id
        try:
            return func(*args, **kwargs)
        finally:
            with _log_scope_lock:
                # Los pools de hilos reutilizan hilos: se restaura la asignación previa en lugar de borrarla siempre, para evitar
                # que envoltorios anidados borren antes de tiempo un vínculo que la capa exterior aún usa.
                if previous_root is None:
                    _log_scope_roots.pop(thread_id, None)
                else:
                    _log_scope_roots[thread_id] = previous_root

    return run_in_log_scope


def _project_relative_path(file_path):
    """
    Acorta la ruta absoluta a una ruta relativa al proyecto que empieza por ``./`` y usa siempre barras normales.

    Windows En el proyecto puede arrancarse desde una unidad de red mapeada o una unidad ``subst``. En ese caso las rutas de la pila de llamadas
    siguen siendo ``X:\\MoneyPrinterTurbo\\...``, mientras que ``PROJECT_ROOT``, tras ``realpath``
    resolverse, queda en ``C:\\...``, y ``os.path.relpath`` lanza directamente ``ValueError``.
    Si la función de formato lanza un error, loguru lo captura y descarta todo el registro, y el terminal y el panel de logs de la WebUI
    quedan vacíos a la vez, así que aquí hay que devolver la ruta original como respaldo. Lo mismo para archivos fuera del directorio del proyecto: pegar
    ``./`` a una ruta de retroceso ``..`` solo daría un resultado más difícil de leer.
    """
    try:
        relative_path = os.path.relpath(file_path, PROJECT_ROOT)
    except ValueError:
        return file_path
    if relative_path == os.pardir or relative_path.startswith(os.pardir + os.sep):
        return file_path
    # Windows El relpath de devuelve rutas separadas por barras invertidas, y concatenarlas directamente daría ``./app\\utils``
    # una salida con separadores mezclados, incoherente con los logs de otras plataformas.
    return f"./{relative_path.replace(os.sep, '/')}"


def format_log_record(record):
    """
    Formatea de forma unificada los logs del terminal y de la WebUI.

    Loguru entrega el mismo registro a varios sinks. El primer sink puede haber convertido ya la ruta absoluta
    en ruta relativa al proyecto, así que aquí se admiten tanto rutas absolutas como rutas ya formateadas que empiezan por ``./``.
    WebUI sink desactiva los colores, pero la hora, el nivel, la ubicación de la llamada y el contenido del mensaje coinciden con el terminal.
    """
    file_path = record["file"].path
    if os.path.isabs(file_path):
        record["file"].path = _project_relative_path(file_path)

    # Los mensajes de log a veces incluyen rutas absolutas de archivos de tareas. Acortarlas siempre a rutas relativas al proyecto
    # evita que la WebUI y el terminal muestren dos contenidos distintos por haberse inicializado desde entradas diferentes.
    # Solo se sustituye la ruta raíz completa del proyecto; los directorios hermanos como `<root>-backup` o `<root>2` solo son
    # el mismo prefijo, y reemplazar por subcadena los convertiría en rutas erróneas como `.-backup` o `.2`.
    record["message"] = _PROJECT_ROOT_IN_MESSAGE_RE.sub(".", record["message"])
    return LOG_RECORD_FORMAT


def force_utf8_console() -> None:
    """
    Switch stdout and stderr to UTF-8 before anything is logged or printed.

    When they go to a file or a pipe on Windows, as with
    ``webui.bat > webui.log 2>&1``, Python encodes them in the ANSI code page,
    such as cp1252 on English and cp936 on Chinese Windows. A log record with
    a character outside it (the circled digits of generate_video(), Chinese
    on cp1252, emoji on cp936) raises UnicodeEncodeError in the terminal sink,
    and Loguru drops the record and prints "Logging error in Loguru Handler"
    in its place. The CLI's final print() failed the same way after the video
    was written. Streams that cannot be switched, such as None under pythonw,
    are left as they are.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            continue


def configure_terminal_logger(sink, level: str, colorize: bool = True) -> int:
    """
    Sustituye de forma segura el handler de logs de terminal a nivel de proceso, conservando los handlers específicos de tareas.

    Streamlit puede volver a ejecutar la inicialización de logs con la recarga en caliente del código o al invalidarse la caché. Aquí solo se elimina con precisión la salida de terminal antigua
    según el ID de handler registrado, así que no se interrumpen los logs que están escribiendo WebUI
    las tareas en segundo plano. El lock protege la actualización del ID cuando varias sesiones del navegador inicializan a la vez.
    """
    global _terminal_handler_id

    set_logging_options(level, get_logging_options()[1])

    with _terminal_handler_lock:
        if _terminal_handler_id is not None:
            try:
                logger.remove(_terminal_handler_id)
            except ValueError:
                # Las pruebas o entradas externas pueden haber eliminado ya ese handler. Se sigue creando la nueva salida de terminal,
                # sin afectar a los demás sinks de logs que siguen siendo válidos.
                pass

        _terminal_handler_id = logger.add(
            sink,
            level=0,
            filter=runtime_log_filter,
            format=format_log_record,
            colorize=colorize,
        )
        return _terminal_handler_id
