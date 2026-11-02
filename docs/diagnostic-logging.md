# Logs de diagnóstico

En la WebUI, en **Ajustes → Ajustes de interfaz → Logs de diagnóstico**, elige el nivel de log, activa si hace falta
**Registrar parámetros de peticiones API** y haz clic en **Aplicar ajustes de logs**. El ajuste se aplica de inmediato al servicio actual,
incluidos los logs posteriores de las tareas en ejecución; todas las sesiones del navegador del mismo servicio comparten este ajuste.
Activar «Ocultar logs» solo oculta los logs de la interfaz, sin desactivar la salida del terminal.

API, el CLI y la WebUI usan la misma configuración:

```toml
[diagnostics]
log_level = "INFO"
log_requests = false
```

`log_level` Opciones: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL.
Si se omite se usa el `log_level` de nivel superior original, DEBUG por defecto. Tras modificar el archivo a mano hay que reiniciar;
WebUI el «Aplicar» de solo actualiza al instante el proceso actual y no modifica de forma remota otros procesos de API o CLI.

El trazado de peticiones es un interruptor independiente y se puede activar incluso con nivel ERROR. Está desactivado por defecto y registra:

- ID de petición, método HTTP, dirección anonimizada y parámetros de la petición con límite de longitud.
- estado de la respuesta (cuando se puede obtener), duración, o el tipo de excepción del fallo.
- LLM SDK parámetros de llamada. Esto es la entrada del SDK, no el JSON en bruto que el SDK envía finalmente;
  SDK los reintentos automáticos internos no se muestran uno a uno. La duración de las peticiones en streaming solo cuenta hasta obtener la respuesta o el objeto de flujo,
  sin incluir el tiempo de descarga posterior.

Cubre las peticiones que el proyecto hace explícitamente con `requests` para material, video, voz, publicación e inicio de sesión, así como
OpenAI las llamadas de texto compatibles, Azure OpenAI, Qwen, Gemini y LiteLLM.
No captura el tráfico de red propio del navegador, los cuerpos de la API entrante, los subprocesos del CLI,Edge TTS WebSocket
ni otros transportes internos de SDKs. Los logs normales de la aplicación y el nivel de log propio de los SDKs de terceros tampoco son el mismo interruptor.

No registra cabeceras de petición, cuerpos de respuesta ni contenido de archivos subidos. Las claves, Tokens, Cookies, contraseñas y campos
de firma se ocultan, los valores de los parámetros de consulta de las URL se ocultan siempre, y el contenido de audio/video/base64 se omite. Los cuerpos de petición
no estructurados en bruto no se registran. **Los prompts y otros parámetros de negocio pueden aparecer en los logs**; al terminar de investigar
desactiva el trazado, y antes de compartir logs revisa si contienen contenido privado. El diagnóstico no envía logs al exterior,
ni lanza, reintenta o repite peticiones adicionales.

CLI Los logs siguen saliendo por stderr, y stdout se mantiene como JSON legible por máquina. Los logs de tareas de la WebUI siguen usando
el filtrado por pertenencia de hilos y el límite de líneas existentes, y cambiar de nivel no elimina el manejador de logs de las tareas en ejecución.
