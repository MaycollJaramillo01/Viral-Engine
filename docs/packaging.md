# pip Paquete de instalación y publicación en PyPI

El nombre del paquete de distribución es `mpt`, el nombre del proyecto sigue siendo MoneyPrinterTurbo, y la versión pendiente de publicar es `1.3.9`.
`1.3.8` se publicó el 2026-10-16 con `harry0703` en
[PyPI](https://pypi.org/project/mpt/1.3.8/), ofreciendo wheel y sdist.
Los archivos publicados no se pueden sobrescribir; los cambios de código posteriores requieren subir la versión y volver a compilar y verificar.

## Compilación e instalación

Requiere Python 3.11 o superior. Las dependencias se resuelven e instalan con el wheel; FFmpeg usa preferentemente la versión del sistema
, y también puede usar el binario que proporciona imageio-ffmpeg.

Instalar desde PyPI y arrancar:

```sh
python -m pip install mpt
mpt webui
mpt cli --help
```

Compilación y verificación local:

```sh
uv build --out-dir storage/package/mpt-dist
python -m pip install ./storage/package/mpt-dist/mpt-1.3.9-py3-none-any.whl
mpt --help
mpt webui
mpt cli --help
```

Antes de la subida formal hay que confirmar la cuenta de PyPI, la versión objetivo y los hashes de los paquetes, evitando subir por lotes desde un directorio mezclado con paquetes antiguos
. Cada publicación requiere autorización explícita del usuario y no se publica automáticamente con commits o pushes de código.

## Comprobaciones antes de publicar

```sh
uvx twine check --strict \
  storage/package/mpt-dist/mpt-1.3.9-py3-none-any.whl \
  storage/package/mpt-dist/mpt-1.3.9.tar.gz
```

Revisa la página de descripción de PyPI, los metadatos del paquete y que las dependencias se puedan instalar, y completa una generación de video en un entorno virtual nuevo.
Solo se publica formalmente tras una autorización explícita de subida; las contraseñas de la cuenta, códigos de verificación y API Tokens no se escriben en el repositorio,
ejemplos de comandos ni logs. Los README de cada idioma ofrecen la entrada de instalación para usuarios, y la descripción de PyPI se mantiene aparte en
`docs/pypi-description.md`; actualizar la documentación del repositorio no modifica automáticamente la página de presentación de PyPI ya publicada.

`mpt webui` Por defecto solo escucha en `127.0.0.1`, elige un puerto libre a partir del 8501 y abre el navegador automáticamente.
Se puede usar `--host`, `--port` para indicar la dirección, y `--headless` para no abrir el navegador automáticamente:

```sh
mpt webui --port 8503 --headless
```

CLI Reutiliza los parámetros, el pipeline de tareas, los códigos de salida y la salida JSON de `python cli.py`. Ejemplo completo sin conexión:

```sh
mpt cli --video-script "A short video from local materials." \
  --video-source local --video-materials /absolute/path/clip.mp4 \
  --custom-audio-file /absolute/path/narration.wav \
  --no-subtitle-enabled --bgm-type none
```

Las funciones en línea de texto, material, TTS, etc. siguen necesitando la configuración, las claves y la conexión de red de los servicios correspondientes. Los modelos Whisper se cargan bajo demanda
y no se incluyen en el wheel. Los logs del CLI se escriben en stderr, y en caso de éxito stdout muestra un JSON con la ruta del video.

## Configuración y datos

La ejecución desde el código fuente sigue usando el directorio raíz del proyecto. La versión instalada no depende del directorio de trabajo actual; el directorio de ejecución por defecto es:

| Plataforma | Directorio de ejecución |
| --- | --- |
| macOS | `~/Library/Application Support/MoneyPrinterTurbo` |
| Windows | `%LOCALAPPDATA%\MoneyPrinterTurbo` |
| Linux | `$XDG_DATA_HOME/MoneyPrinterTurbo`, por defecto `~/.local/share/MoneyPrinterTurbo` |

Se puede indicar explícitamente el directorio de ejecución con `MPT_HOME` (también en la versión de código fuente). En él:

- `config.toml`: en la primera ejecución se inicializa a partir de una plantilla sin claves, y los arranques posteriores no sobrescriben la configuración existente.
- `storage/tasks`: tareas y videos generados; las demás cachés y archivos subidos están bajo `storage`.
- `resource/fonts`, `resource/songs`: en el primer uso se copian los recursos integrados, y puedes añadir tus propios archivos.
- `models`: se pueden colocar modelos Whisper descargados manualmente.

Los recursos del usuario solo completan los archivos que faltan, y las actualizaciones no sobrescriben los existentes. El paquete de instalación incluye dos fuentes chinas por defecto,
fuentes vietnamitas y música de fondo integrada; las alternativas `STHeitiLight.ttc` y`MicrosoftYaHeiNormal.ttc`
no se incluyen en el paquete para controlar su tamaño. Si necesitas esos pesos, colócalos en `resource/fonts` del directorio de ejecución.
Los recursos y el aspecto originales de la versión de código fuente no cambian.

## Requisitos de verificación local

Reconstruye el wheel desde el sdist, instálalo en un entorno virtual nuevo que no herede paquetes del sistema, sal del directorio del código fuente y elimina
`PYTHONPATH` y luego verifica `mpt cli --help`, la página real de la WebUI y la generación de video. El video de salida debe
FFmpeg decodificarse por completo, y hay que confirmar que la configuración y los artefactos no se escriben en site-packages. El paquete no debe incluir
el `config.toml` local, claves, cachés, artefactos de tareas ni instrucciones de colaboración locales.
