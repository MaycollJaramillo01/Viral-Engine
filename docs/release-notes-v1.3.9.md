[English](#en-v1.3.9) | [Español](#cn-v1.3.9)

<h2 id="en-v1.3.9">Highlights</h2>

- Added official **`mpt` pip packaging**. Start the WebUI with `mpt webui` or generate videos with `mpt cli`.
- Added **static and slow zoom image effects** for local and AI-generated images. Slow zoom remains the default.
- Added Kimi Code subscription access with browser authorization, alongside improvements to the sign-in dialog, status updates, and cancellation.
- Added an optional **local SDXL image server** for the OpenAI-compatible image material workflow, with idle GPU offloading.
- Added Opper, onomeo, Muse Spark, and EmpirioLabs AI LLM providers, and completed Brazilian Portuguese WebUI translations.
- Added **diagnostic logging controls** in the WebUI, including runtime log-level changes and opt-in API request tracing with redaction and size limits.

## Fixes and Improvements

- Fixed grayscale image brightness, transparent image backgrounds, zoom mask alignment, and subtitle background colors. Improved transitions on short clips, background music fade-out, and special-character handling in concatenation paths.
- Isolated intermediate image files across concurrent jobs, improved image cache identity, and validated cached downloads. Completed videos are retained when another video in the same batch fails.
- Fixed speech speed, volume, and audio container handling across several providers. Improved remote audio validation and preserved existing audio until replacement output is ready.
- Improved subtitle timestamp rounding, long-word wrapping, Unicode handling, and pause-tag recognition. Symbol-only lines and separators no longer distort estimated subtitle alignment.
- Fixed Chinese and Japanese fallback social titles using the entire script. Improved JSON extraction, reasoning removal, Markdown cleanup, and generated search-term validation.
- Expanded supported local upload formats, including multi-picture camera JPEGs. Improved task history checks, deletion state, subtitle previews, and model connection error messages in the WebUI.
- Added HEAD support for generated artifacts and improved byte-range downloads, validation errors, and Redis task updates and deletion safeguards.
- Reduced unsafe speech resubmissions after lost responses or redirects, preserved Upload-Post tracking handles, and improved WaveSpeed submission error detection.
- Updated Edge TTS to address 403 failures, constrained PyAV for Whisper compatibility, and fixed Windows cache cleanup and UTF-8 BOM handling in locale files.
- Improved the agent helper's configuration handling and preflight checks, including workflows with prepared narration and visuals that do not need an LLM.

## Usage and Upgrade Notes

- The official PyPI package is **`mpt`**, maintained by this project, and requires **Python 3.11 or later**. Install or upgrade with `python -m pip install --upgrade mpt`; a virtual environment is recommended. The separately named `MoneyPrinterTurbo` package is a third-party distribution.
- The pip installation stores configuration and generated files in your user directory by default; set `MPT_HOME` to choose another location. Source installations continue to use the project directory. When switching installation methods, migrate your configuration and required resources to the chosen runtime directory.
- Image motion defaults to **slow zoom** for existing users. For a still image without transition effects, select **Static** for image motion and **None** for transitions. The setting is available in the WebUI, CLI, and API.
- Kimi Code requires browser sign-in and uses your plan's usage limits. The other new LLM providers require their corresponding API credentials and configuration.
- The local SDXL server requires separate dependencies, model downloads, and a running GPU service; it is not started automatically with MPT. Its default setup unloads Ollama models before image generation, so avoid concurrent LLM and image work on the same GPU. See the [setup instructions](https://github.com/harry0703/MoneyPrinterTurbo/blob/v1.3.9/docs/local-image-server/README.md).
- API request tracing is **off by default**. Prompts and business parameters may appear in logs when enabled; turn tracing off after troubleshooting and check for private content before sharing logs. See the [diagnostic logging guide](https://github.com/harry0703/MoneyPrinterTurbo/blob/v1.3.9/docs/diagnostic-logging.md).

## What's Changed

- Official pip package with WebUI and CLI launchers — [c3634de](https://github.com/harry0703/MoneyPrinterTurbo/commit/c3634de097ea4951935d22e94580ff0ad2665503).
- Static and slow zoom image options — [90f7701](https://github.com/harry0703/MoneyPrinterTurbo/commit/90f77011d7e3b46d382e6d0180dabfb6725ffee4).
- Kimi Code subscription provider — @ysntony in [#1599](https://github.com/harry0703/MoneyPrinterTurbo/pull/1599), followed by [sign-in flow improvements](https://github.com/harry0703/MoneyPrinterTurbo/commit/4204d0f).
- Local SDXL image server — @imkashif in [#1668](https://github.com/harry0703/MoneyPrinterTurbo/pull/1668).
- Opper LLM provider — @Felixkw12 in [#1589](https://github.com/harry0703/MoneyPrinterTurbo/pull/1589).
- onomeo LLM provider — @Owaid0723666 in [#1600](https://github.com/harry0703/MoneyPrinterTurbo/pull/1600).
- Muse Spark provider and Brazilian Portuguese translations — @humbertocortezia in [#1657](https://github.com/harry0703/MoneyPrinterTurbo/pull/1657).
- EmpirioLabs AI provider — @Adam-Dalloul in [#1694](https://github.com/harry0703/MoneyPrinterTurbo/pull/1694).
- Runtime diagnostic logging and request tracing — [6940611](https://github.com/harry0703/MoneyPrinterTurbo/commit/6940611cc2791d1ca13b3af7ee893ccd9544b874).
- Chinese and Japanese fallback social titles — [e25326e](https://github.com/harry0703/MoneyPrinterTurbo/commit/e25326e).
- Retained completed videos after batch failures — @hdu-r3drum in [#1597](https://github.com/harry0703/MoneyPrinterTurbo/pull/1597).
- Edge TTS 403 fix — @HEOJUNFO in [#1652](https://github.com/harry0703/MoneyPrinterTurbo/pull/1652).
- Whisper/PyAV compatibility — @cristianchiriac in [#1582](https://github.com/harry0703/MoneyPrinterTurbo/pull/1582).
- Windows cache cleanup — @shan3520 in [#1654](https://github.com/harry0703/MoneyPrinterTurbo/pull/1654).
- UTF-8 BOM locale support — @ketpatil77 in [#1658](https://github.com/harry0703/MoneyPrinterTurbo/pull/1658).
- Task protection for unparseable states — @RerankerGuo in [#1663](https://github.com/harry0703/MoneyPrinterTurbo/pull/1663).
- Clearer model connection errors — @fengqiaoqu in [#1645](https://github.com/harry0703/MoneyPrinterTurbo/pull/1645).
- Local material preflight checks — @drakeo338 in [#1585](https://github.com/harry0703/MoneyPrinterTurbo/pull/1585); preserved Redis task IDs — @ac1982 in [#1588](https://github.com/harry0703/MoneyPrinterTurbo/pull/1588).
- Grayscale rendering, artifact HEAD requests, and prepared-media agent workflows — @rudycelekli in [#1620](https://github.com/harry0703/MoneyPrinterTurbo/pull/1620), [#1636](https://github.com/harry0703/MoneyPrinterTurbo/pull/1636), and [#1608](https://github.com/harry0703/MoneyPrinterTurbo/pull/1608), alongside many subtitle, speech, material, API, and task reliability fixes.

Thanks to all contributors and everyone who reported issues and helped test these changes!

**Full Changelog**: [v1.3.8...v1.3.9](https://github.com/harry0703/MoneyPrinterTurbo/compare/v1.3.8...v1.3.9)

---

<h2 id="cn-v1.3.9">Novedades principales</h2>

- Instalación mediante pip del **paquete oficial `mpt`**: usa `mpt webui` para iniciar la interfaz o `mpt cli` para generar videos.
- Nueva **opción de efecto dinámico de imagen**: tanto las imágenes locales como las generadas por IA pueden elegir «estática» o «zoom lento», con zoom lento por defecto.
- Integración de la suscripción Kimi Code, con inicio de sesión autorizado en el navegador, y mejoras en el diseño de la ventana de inicio de sesión, la actualización de estado y la cancelación.
- Nuevo **servicio local opcional de generación de imágenes SDXL**, conectado a la generación de material de imagen mediante una interfaz compatible con OpenAI, con liberación de memoria de GPU en reposo.
- Nuevos servicios de modelos grandes Opper, onomeo, Muse Spark y EmpirioLabs AI, y mejoras en la traducción de la interfaz al portugués de Brasil.
- Nuevos **ajustes de logs de diagnóstico**, que permiten ajustar al instante el nivel de log en la WebUI y registrar bajo demanda los parámetros de peticiones API anonimizados y con límite de longitud.

## Correcciones y mejoras

- Corregidos problemas de brillo de imágenes en escala de grises, fondo de imágenes transparentes, alineación de máscaras de zoom y color de fondo de subtítulos; mejoras en las transiciones de clips cortos, el fundido de la música de fondo y el tratamiento de caracteres especiales en las rutas de concatenación.
- Aislamiento de los archivos intermedios de imagen entre tareas concurrentes, mejores identificadores de caché de imágenes y validación de la caché de descargas; si una generación por lotes falla parcialmente, se conservan los videos ya completados.
- Corregido el tratamiento de velocidad, volumen y formato de audio en algunos servicios de voz, y mejorada la validación de audio remoto; se conservan los archivos existentes hasta que el audio de reemplazo esté listo.
- Mejoras en el redondeo de marcas de tiempo de subtítulos, el salto de línea de palabras largas, los caracteres Unicode y el reconocimiento de etiquetas de pausa, evitando que líneas de solo símbolos y separadores alteren la estimación de la línea de tiempo de subtítulos.
- Corregido que los títulos sociales de respaldo en chino y japonés usaran todo el guion; mejoras en la extracción de JSON, el filtrado de contenido de razonamiento, la limpieza de textos Markdown y la validación de términos de búsqueda de material.
- Completados los formatos de subida local admitidos, incluidos los JPEG multi-imagen de cámara; mejoras en la revisión de videos históricos de la WebUI, el estado de borrado de tareas, la vista previa de subtítulos y los mensajes de error de conexión del modelo.
- Soporte de peticiones HEAD para archivos generados, mejoras en la descarga por rangos, los mensajes de error de parámetros, y la actualización y protección de borrado del estado de tareas en Redis.
- Menor riesgo de reenvío duplicado de peticiones de voz tras perder la respuesta o una redirección, conservación de la información de trazado de peticiones de Upload-Post y mejor detección de fallos de envío de WaveSpeed.
- Actualizado Edge TTS, corrigiendo problemas 403 relacionados; limitada la versión de PyAV por compatibilidad con Whisper, y corregidas la limpieza de caché en Windows y la lectura de archivos de traducción con BOM UTF-8.
- Mejoras en la gestión de configuración y las comprobaciones de arranque de los scripts auxiliares de Agent; las tareas que ya tienen voz e imagen preparadas ya no requieren además un servicio de modelo grande.

## Uso e instrucciones de actualización

- El paquete oficial de PyPI que mantiene este proyecto se llama **`mpt`** y requiere **Python 3.11 o superior**. Usa `python -m pip install --upgrade mpt` para instalar o actualizar; se recomienda un entorno virtual. Otro paquete en PyPI llamado `MoneyPrinterTurbo` es una distribución de terceros.
- pip La versión instalada guarda por defecto la configuración y los archivos generados en el directorio del usuario, y se puede indicar la ubicación con `MPT_HOME`. La versión de código fuente sigue usando el directorio del proyecto original; al cambiar de forma de instalación, migra la configuración y los recursos necesarios al directorio de ejecución elegido.
- El efecto dinámico de imagen sigue siendo **zoom lento** por defecto, compatible con el uso existente. Si quieres que la imagen quede estática, elige a la vez **estática** y **sin transición**; la WebUI, el CLI y la API admiten este ajuste.
- Kimi Code requiere iniciar sesión en el navegador y usa la cuota de la suscripción correspondiente; los demás servicios de modelos grandes nuevos requieren configurar sus propias credenciales de API y parámetros de servicio.
- El servicio local SDXL requiere instalar dependencias por separado, descargar el modelo y arrancar el servicio de GPU; no se ejecuta automáticamente con el programa principal. La configuración por defecto descarga los modelos de Ollama antes de generar imágenes; evita ejecutar a la vez modelos grandes y tareas de generación de imágenes en la misma GPU. El método de despliegue está en la [documentación del servicio local de generación de imágenes](https://github.com/harry0703/MoneyPrinterTurbo/blob/v1.3.9/docs/local-image-server/README.md).
- API El trazado de peticiones está **desactivado por defecto**. Al activarlo, los prompts y parámetros de negocio pueden aparecer en los logs; desactívalo al terminar de investigar y revisa el contenido privado antes de compartir logs. Más detalles en la [documentación de logs de diagnóstico](https://github.com/harry0703/MoneyPrinterTurbo/blob/v1.3.9/docs/diagnostic-logging.md).

## Cambios principales

- [c3634de](https://github.com/harry0703/MoneyPrinterTurbo/commit/c3634de097ea4951935d22e94580ff0ad2665503): paquete de instalación oficial de pip y comandos de arranque de WebUI y CLI.
- [90f7701](https://github.com/harry0703/MoneyPrinterTurbo/commit/90f77011d7e3b46d382e6d0180dabfb6725ffee4): opciones de imagen estática y zoom lento.
- [#1599](https://github.com/harry0703/MoneyPrinterTurbo/pull/1599): integración de la suscripción Kimi Code, gracias a @ysntony; después se añadieron [mejoras del flujo de inicio de sesión](https://github.com/harry0703/MoneyPrinterTurbo/commit/4204d0f).
- [#1668](https://github.com/harry0703/MoneyPrinterTurbo/pull/1668): servicio local de generación de imágenes SDXL, gracias a @imkashif.
- [#1589](https://github.com/harry0703/MoneyPrinterTurbo/pull/1589): servicio de modelo grande Opper, gracias a @Felixkw12.
- [#1600](https://github.com/harry0703/MoneyPrinterTurbo/pull/1600): servicio de modelo grande onomeo, gracias a @Owaid0723666.
- [#1657](https://github.com/harry0703/MoneyPrinterTurbo/pull/1657): servicio de modelo grande Muse Spark y traducción al portugués de Brasil, gracias a @humbertocortezia.
- [#1694](https://github.com/harry0703/MoneyPrinterTurbo/pull/1694): servicio de modelo grande EmpirioLabs AI, gracias a @Adam-Dalloul.
- [6940611](https://github.com/harry0703/MoneyPrinterTurbo/commit/6940611cc2791d1ca13b3af7ee893ccd9544b874): ajustes de logs de diagnóstico en tiempo de ejecución y trazado de peticiones.
- [e25326e](https://github.com/harry0703/MoneyPrinterTurbo/commit/e25326e): corrección de los títulos sociales de respaldo en chino y japonés.
- [#1597](https://github.com/harry0703/MoneyPrinterTurbo/pull/1597): conservar los videos completados tras un fallo en la generación por lotes, gracias a @hdu-r3drum.
- [#1652](https://github.com/harry0703/MoneyPrinterTurbo/pull/1652): corrección del 403 de Edge TTS, gracias a @HEOJUNFO.
- [#1582](https://github.com/harry0703/MoneyPrinterTurbo/pull/1582): corrección de compatibilidad entre Whisper y PyAV, gracias a @cristianchiriac.
- [#1654](https://github.com/harry0703/MoneyPrinterTurbo/pull/1654): corrección de la limpieza de caché en Windows, gracias a @shan3520.
- [#1658](https://github.com/harry0703/MoneyPrinterTurbo/pull/1658): soporte de archivos de traducción con BOM UTF-8, gracias a @ketpatil77.
- [#1663](https://github.com/harry0703/MoneyPrinterTurbo/pull/1663): protección de tareas cuando el estado no se puede analizar, gracias a @RerankerGuo.
- [#1645](https://github.com/harry0703/MoneyPrinterTurbo/pull/1645): mensajes de error de conexión del modelo más claros, gracias a @fengqiaoqu.
- [#1585](https://github.com/harry0703/MoneyPrinterTurbo/pull/1585): comprobación de material local al arrancar, gracias a @drakeo338; [#1588](https://github.com/harry0703/MoneyPrinterTurbo/pull/1588): conservar la cadena original del ID de tarea de Redis, gracias a @ac1982.
- [#1620](https://github.com/harry0703/MoneyPrinterTurbo/pull/1620), [#1636](https://github.com/harry0703/MoneyPrinterTurbo/pull/1636), [#1608](https://github.com/harry0703/MoneyPrinterTurbo/pull/1608): renderizado de imágenes en escala de grises, peticiones HEAD de archivos generados y flujo de trabajo de Agent con material preparado, además de varias correcciones de estabilidad de subtítulos, voz, material, API y tareas, gracias a @rudycelekli.

¡Gracias a todos los que contribuyeron código, informaron de problemas y participaron en las pruebas!

**Registro completo de cambios**:[v1.3.8...v1.3.9](https://github.com/harry0703/MoneyPrinterTurbo/compare/v1.3.8...v1.3.9)
