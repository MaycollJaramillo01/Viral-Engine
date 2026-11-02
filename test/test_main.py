import runpy
from pathlib import Path
from unittest.mock import patch

from app.config import config


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_main_starts_uvicorn_with_runtime_config():
    """
    El punto de entrada de arranque del servicio solo se encarga de pasar la configuración de ejecución a Uvicorn. Aquí se simula con mock el arranque real del servidor,
    evitando que la prueba ocupe puertos y confirmando que la dirección de escucha, el puerto y la configuración de recarga en caliente no se pierden en la capa de entrada.
    """
    with (
        patch.object(config, "listen_host", "127.0.0.1"),
        patch.object(config, "listen_port", 8765),
        patch.object(config, "reload_debug", True),
        patch("uvicorn.run") as run_server,
    ):
        runpy.run_path(str(ROOT_DIR / "main.py"), run_name="__main__")

    run_server.assert_called_once_with(
        app="app.asgi:app",
        host="127.0.0.1",
        port=8765,
        reload=True,
        log_level="warning",
    )
