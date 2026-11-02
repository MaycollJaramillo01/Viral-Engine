import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from loguru import logger

from app.utils import file_security, utils


# Streamlit permite por defecto archivos subidos bastante grandes, pero la música de fondo suele ocupar solo unos MB. Aquí se fija un límite
# explícito en el servidor, para que la API o la WebUI no escriban archivos enormes completos en disco y afecten a las tareas de video del mismo proceso.
MAX_BGM_UPLOAD_BYTES = 30 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_INTERNAL_UPLOAD_PREFIX = ".bgm-upload-"
_WINDOWS_INVALID_FILENAME_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_FILENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)}
    # Win32 también reconoce los superíndices Latin-1 ¹, ², ³ como números de dispositivo, así que se tratan igual que los nombres reservados con dígitos normales,
    # para que COM¹.mp3 no esquive la protección. La lista coincide con las reglas de nombres de descarga de webui/Main.py.
    | {f"{prefix}{number}" for prefix in ("COM", "LPT") for number in ("¹", "²", "³")}
)
# El nombre de archivo llega tal cual a los logs, las respuestas de la API y la interfaz WebUI, así que además de los separadores de ruta y los nombres reservados de Win32,
# hay que rechazar todos los caracteres de control y los que alteran la forma de visualización. Antes solo se bloqueaba ord < 32 (controles C0),
# y se escapaba la otra mitad del mismo problema:
# * C1 caracteres de control U+007F-U+009F: algunos visores de logs renderizan U+0085 como salto de línea, y un solo nombre de archivo podría
#   falsificar líneas de log adicionales que parecen independientes.
# * controles de texto bidireccional U+200E/U+200F/U+202A-U+202E/U+2066-U+2069: los caracteres tras U+202E se
#   renderizan al revés, y "photo‮gnp.mp3" parece tener otra extensión en el explorador de archivos y en los logs.
# * U+2028/U+2029(separadores Unicode de línea/párrafo) también pueden crear saltos de línea visuales dentro de una línea de log.
# Ninguno de estos caracteres se puede teclear en el campo de nombre de archivo, así que las subidas normales no se ven afectadas; el criterio coincide con
# app/controllers/base.py normalize_task_id de (que allí usa directamente isprintable).
_UNSAFE_FILENAME_CHARACTERS = frozenset(
    chr(code)
    for code in (
        *range(0x00, 0x20),
        *range(0x7F, 0xA0),
        0x200E,
        0x200F,
        0x2028,
        0x2029,
        *range(0x202A, 0x202F),
        *range(0x2066, 0x206A),
    )
)
# MoviePy La música de fondo se decodifica finalmente con FFmpeg, así que no hace falta limitarla artificialmente a MP3. Aquí solo se habilitan
# extensiones de audio comunes y de significado claro, para evitar subir como música de fondo contenedores con video como MP4.
# La tupla es además la única fuente de datos del control de subida de la WebUI, así que al añadir o quitar formatos no habrá incoherencias entre frontend y backend.
SUPPORTED_BGM_EXTENSIONS = (
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
)


class BgmUploadError(ValueError):
    """Indica que el archivo subido no cumple los requisitos de seguridad o formato para música de fondo."""


class BgmServiceError(RuntimeError):
    """Indica un fallo de ejecución del servidor, como FFmpeg o el sistema de archivos no disponibles."""


def should_use_bgm(bgm_type: str | None, bgm_volume: float | None) -> bool:
    """
    Determina de forma unificada si la tarea actual necesita procesar alguna música de fondo.

    La regla no depende del origen concreto: si no se eligió origen, el volumen no es válido o no es mayor que 0, aleatorio,
    personalizado, Sonilo y cualquier proveedor futuro deben omitir el análisis del archivo, la generación externa y la mezcla final.
    Ponerlo en el servicio BGM común evita copiar la comprobación de volumen 0 cada vez que se añade un proveedor.
    """
    if not str(bgm_type or "").strip():
        return False
    try:
        normalized_volume = float(bgm_volume or 0)
    except (TypeError, ValueError):
        return False
    return math.isfinite(normalized_volume) and normalized_volume > 0


def uploaded_bgm_dir(create: bool = True) -> str:
    """
    Devuelve el directorio persistente de la música de fondo del usuario.

    Las canciones integradas son recursos del código y siguen en resource/songs; el contenido subido por el usuario son datos de ejecución
    y debe ir en storage, que Docker ya monta, para que se conserve al recrear el contenedor y no ensucie el área de trabajo de Git.
    """
    return utils.storage_dir("bgm", create=create)


def _remove_staged_file(file_path: str) -> None:
    """Limpia en lo posible los archivos temporales de subida, sin sobrescribir la excepción original que está manejando el llamador."""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        # Los archivos temporales usan un prefijo reservado y no entran en la lista de BGM; un fallo de limpieza no debe ocultar excepciones originales más precisas
        # como «audio inválido», pero debe dejar la ruta y el error del sistema para que operaciones lo localice.
        logger.warning(
            f"failed to remove staged background music: path={file_path}, "
            f"error={str(exc)}"
        )


def sanitize_upload_filename(filename: str) -> str:
    """Extrae un nombre de archivo de audio que se pueda mostrar en todas las plataformas, rechazando nombres ilegales y extensiones no soportadas."""
    safe_name = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if (
        not safe_name
        or safe_name in {".", ".."}
        or len(safe_name) > 255
        or any(character in _UNSAFE_FILENAME_CHARACTERS for character in safe_name)
        or any(character in _WINDOWS_INVALID_FILENAME_CHARS for character in safe_name)
        or safe_name.lower().startswith(_INTERNAL_UPLOAD_PREFIX)
    ):
        raise BgmUploadError("invalid background music filename")

    # Windows reconoce el primer segmento antes de la extensión como nombre de dispositivo; por ejemplo, CON.mp3 o LPT1.wav
    # no se pueden crear como archivos normales. Aunque el servidor use finalmente un UUID, rechazar antes estos nombres
    # garantiza que la API se comporte igual ante la entrada en todas las plataformas.
    windows_basename = safe_name.split(".", 1)[0].rstrip(" .").upper()
    if windows_basename in _WINDOWS_RESERVED_FILENAMES:
        raise BgmUploadError("invalid background music filename")
    if Path(safe_name).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS:
        supported_formats = ", ".join(
            extension.removeprefix(".").upper()
            for extension in SUPPORTED_BGM_EXTENSIONS
        )
        raise BgmUploadError(
            f"unsupported background music format; supported formats: {supported_formats}"
        )
    return safe_name


def _validate_audio(file_path: str, timeout_seconds: int = 30) -> None:
    """
    Usa solo el FFmpeg configurado actualmente en el proyecto para verificar que el archivo contiene una pista de audio decodificable por completo.

    El proyecto permite que imageio-ffmpeg proporcione un FFmpeg portable, y esa forma de instalación no garantiza que exista también
    FFprobe, así que no se puede añadir una dependencia binaria independiente. `-map 0:a:0` falla cuando no hay pista de audio,
    `-xerror` eleva los errores de decodificación a fallo; la decodificación completa también intercepta archivos cifrados o datos aleatorios que por casualidad
    coincidan con una cabecera de trama de audio. El archivo puede contener pistas adicionales como la carátula del álbum, pero solo se valida la primera pista de audio.
    """
    try:
        decoded = subprocess.run(
            [
                utils.get_ffmpeg_binary(),
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                # Uploaded files must be self-contained media, not playlists
                # that cause validation/rendering to read adjacent files or URLs.
                "-protocol_whitelist",
                "file,pipe",
                "-format_whitelist",
                "mp3,mov,aac,wav,flac,ogg,asf",
                "-i",
                file_path,
                "-map",
                "0:a:0",
                # Normalize timestamps from decoded samples. A header-only
                # WAV can exit successfully without producing any audio.
                "-af",
                "asetpts=N/SR/TB",
                "-progress",
                "pipe:1",
                "-nostats",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BgmServiceError("FFmpeg background music validation timed out") from exc
    except OSError as exc:
        raise BgmServiceError("failed to run FFmpeg for background music validation") from exc
    if decoded.returncode != 0:
        raise BgmUploadError("uploaded file must contain a decodable audio stream")
    decoded_audio = False
    for line in decoded.stdout.splitlines():
        key, separator, value = line.partition(b"=")
        if separator and key == b"out_time_us":
            try:
                decoded_audio = decoded_audio or int(value) > 0
            except ValueError:
                continue
    if not decoded_audio:
        raise BgmUploadError("uploaded file must contain nonempty decodable audio")


def validate_audio_file(file_path: str, timeout_seconds: int = 120) -> None:
    """
    Verifica que el archivo de audio en disco se pueda decodificar por completo con el FFmpeg del proyecto.

    La comprobación previa de subida suele necesitar solo 30 segundos; la música generada por Sonilo puede durar hasta 6 minutos, así que se ofrece
    una entrada reutilizable con tiempo de espera ajustable. El servicio solo depende de FFmpeg y no requiere instalar FFprobe en el sistema.
    """
    if not os.path.isfile(file_path) or os.path.getsize(file_path) <= 0:
        raise BgmUploadError("background music file is empty or missing")
    _validate_audio(file_path, timeout_seconds=timeout_seconds)


def _stage_bgm_upload(filename: str, source: BinaryIO) -> tuple[str, str, int]:
    """
    Escribe el flujo subido en un archivo temporal del mismo directorio y devuelve el nombre de archivo seguro, la ruta temporal y el número de bytes.

    WebUI La comprobación previa de subida y la persistencia final deben usar exactamente la misma lectura por bloques, límite de tamaño y reglas
    de nombre de archivo; de lo contrario la interfaz podría mostrar que está disponible y luego el servidor rechazarlo al hacer clic en generar.
    El llamador elimina o sustituye atómicamente el archivo temporal tras analizar el audio.
    """
    safe_name = sanitize_upload_filename(filename)
    try:
        target_dir = uploaded_bgm_dir(create=True)
    except OSError as exc:
        raise BgmServiceError("failed to prepare background music storage") from exc
    temp_path = ""
    total_bytes = 0

    try:
        try:
            source.seek(0)
        except (AttributeError, OSError) as exc:
            raise BgmUploadError("background music upload is not seekable") from exc

        # Se conserva la extensión original para que FFmpeg elija el formato correcto en formatos sin cabecera de contenedor como AAC
        # demuxer; el archivo temporal sigue en el directorio de destino para que el os.replace final sea atómico.
        descriptor, temp_path = tempfile.mkstemp(
            prefix=_INTERNAL_UPLOAD_PREFIX,
            suffix=Path(safe_name).suffix.lower(),
            dir=target_dir,
        )
        with os.fdopen(descriptor, "wb") as output:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise BgmUploadError("background music upload must be binary")
                total_bytes += len(chunk)
                if total_bytes > MAX_BGM_UPLOAD_BYTES:
                    raise BgmUploadError("background music file exceeds the 30 MB limit")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        if total_bytes == 0:
            raise BgmUploadError("background music file is empty")
        return safe_name, temp_path, total_bytes
    except Exception as exc:
        _remove_staged_file(temp_path)
        if isinstance(exc, BgmUploadError):
            raise
        if isinstance(exc, OSError):
            raise BgmServiceError("failed to stage background music upload") from exc
        raise
    finally:
        # Streamlit también necesita el mismo UploadedFile para la previsualización en el navegador; restaurar el puntero del archivo
        # evita que, tras la validación, el reproductor o el guardado final lean contenido vacío.
        try:
            source.seek(0)
        except (AttributeError, OSError):
            pass


def validate_bgm_upload(filename: str, source: BinaryIO) -> str:
    """Valida por completo el audio subido sin persistirlo, para la comprobación previa de la WebUI antes de mostrar «Listo»."""
    safe_name, temp_path, total_bytes = _stage_bgm_upload(filename, source)
    try:
        _validate_audio(temp_path)
        logger.debug(
            f"background music upload validated: name={safe_name}, "
            f"size={total_bytes} bytes"
        )
        return safe_name
    finally:
        _remove_staged_file(temp_path)


def save_bgm_upload(filename: str, source: BinaryIO) -> str:
    """
    Guarda la música de fondo del usuario por bloques, con límite de tamaño y sustitución atómica.

    Se usa con UploadFile de FastAPI y con UploadedFile de Streamlit; ambos ofrecen una interfaz de archivo
    binaria. Primero se escribe un archivo temporal en el mismo directorio y se valida, y luego se guarda de forma atómica con os.replace; así se evita
    que subidas concurrentes o la interrupción del proceso dejen medio archivo de audio, y las subidas con el mismo nombre obtienen claves UUID distintas,
    de modo que las tareas encoladas o en ejecución siempre referencian el archivo inmutable original.
    """
    safe_name, temp_path, total_bytes = _stage_bgm_upload(filename, source)
    stored_name = f"{uuid4().hex}{Path(safe_name).suffix.lower()}"
    target_path = os.path.join(os.path.dirname(temp_path), stored_name)

    try:
        _validate_audio(temp_path)
        try:
            os.replace(temp_path, target_path)
        except OSError as exc:
            raise BgmServiceError("failed to persist background music upload") from exc
        temp_path = ""
        logger.info(
            f"background music uploaded: original_name={safe_name}, "
            f"stored_name={stored_name}, size={total_bytes} bytes"
        )
        return stored_name
    finally:
        _remove_staged_file(temp_path)


def _list_bgm_files(directories: tuple[str, ...]) -> list[str]:
    """Enumera los archivos de música de fondo seguros y soportados según la prioridad de directorios."""
    files_by_name: dict[str, str] = {}
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory), key=str.lower):
            # La comprobación previa de subida y el guardado final crean brevemente archivos en el mismo directorio. Aunque los archivos temporales tengan una extensión
            # de audio válida, aún no se han validado y no pueden ser elegidos antes de tiempo por la lista aleatoria de BGM.
            if name.startswith(_INTERNAL_UPLOAD_PREFIX):
                continue
            if Path(name).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS:
                continue
            file_path = os.path.join(directory, name)
            try:
                # El resultado de la enumeración también necesita validar la ruta real. De lo contrario, un atacante podría colocar en un directorio permitido
                # un enlace simbólico de audio que apunte a un archivo externo y pasarlo a MoviePy mediante la ruta de BGM aleatoria.
                resolved_path = file_security.resolve_path_within_directory(
                    directory, file_path
                )
            except ValueError as exc:
                logger.warning(
                    f"skip unsafe background music file: name={name}, error={str(exc)}"
                )
                continue
            files_by_name[name] = resolved_path
    return [files_by_name[name] for name in sorted(files_by_name, key=str.lower)]


def list_builtin_bgm_files() -> list[str]:
    """
    Lista la música de fondo integrada que se distribuye con el proyecto.

    WebUI Las «canciones predefinidas» y la importación/exportación de ajustes predefinidos solo usan esta lista, para garantizar que los nombres de archivo guardados
    se puedan restaurar en otro equipo con la misma versión; los archivos subidos por el usuario siguen gestionándose como música personalizada.
    """
    return _list_bgm_files((utils.song_dir(),))


def list_bgm_files() -> list[str]:
    """Lista la música de fondo disponible, subida por el usuario e integrada; ante nombres repetidos se prefiere el archivo subido."""
    return _list_bgm_files((utils.song_dir(), uploaded_bgm_dir(create=True)))


def resolve_builtin_bgm_file(unsafe_path: str) -> str:
    """Resuelve la música de fondo integrada por nombre de archivo, rechazando rutas, archivos desconocidos y archivos subidos por el usuario."""
    if not unsafe_path:
        raise ValueError("background music filename is required")

    filename = str(unsafe_path)
    if filename != os.path.basename(filename):
        raise ValueError("preset background music must use a filename")

    files_by_name = {
        os.path.basename(file_path): file_path for file_path in list_builtin_bgm_files()
    }
    if filename not in files_by_name:
        raise ValueError("preset background music is not available")
    return files_by_name[filename]


def resolve_bgm_file(unsafe_path: str) -> str:
    """
    Resuelve la BGM en el directorio de subidas del usuario y en el de canciones integradas, rechazando rutas fuera de ambas listas blancas y archivos temporales
    que haya dejado una subida interrumpida.

    Los nombres de archivo buscan primero en el directorio del usuario, manteniendo los usos antiguos como `output000.mp3`, rutas absolutas de la lista blanca y
    `./resource/songs/output000.mp3` . Los archivos nuevos usan UUID, así que normalmente
    no coincidirán en nombre con canciones integradas ni con subidas anteriores.
    """
    if (
        not unsafe_path
        or Path(unsafe_path).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS
    ):
        raise ValueError("unsupported background music path")
    if os.path.basename(str(unsafe_path)).lower().startswith(_INTERNAL_UPLOAD_PREFIX):
        # La comprobación previa de subida y el guardado final crean brevemente en el mismo directorio archivos intermedios con el prefijo ``.bgm-upload-``
        # . Tienen una extensión de audio válida pero aún no se han validado;``_list_bgm_files``
        # ya los excluye de la BGM aleatoria; la entrada de resolución debe llegar a la misma conclusión, de lo contrario
        # API/CLI un nombre recibido podría apuntar a un archivo intermedio interrumpido e incompleto.
        # Los nombres recibidos del usuario se comparan sin distinguir mayúsculas, igual que los sistemas de archivos de Windows/macOS.
        raise ValueError("background music upload staging files are not selectable")

    candidates = [unsafe_path]
    if not os.path.isabs(unsafe_path):
        candidates.append(os.path.join(utils.root_dir(), unsafe_path))

    last_error = ValueError("background music file does not exist")
    for directory in (uploaded_bgm_dir(create=True), utils.song_dir()):
        for candidate in candidates:
            try:
                return file_security.resolve_path_within_directory(directory, candidate)
            except ValueError as exc:
                last_error = exc
    raise ValueError(str(last_error)) from last_error
