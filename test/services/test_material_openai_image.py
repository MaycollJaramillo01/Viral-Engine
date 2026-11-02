# -*- coding: utf-8 -*-
import base64
import io
import os
import shutil
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import requests
from PIL import Image

from app.config import config
from app.services import material


def _png_bytes(width=64, height=96, color=(120, 40, 200)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _image_response(payload, status_code=200):
    return SimpleNamespace(json=lambda: payload, status_code=status_code)


def _download_response(content, status_code=200):
    return SimpleNamespace(
        status_code=status_code,
        content=content,
        headers={"Content-Length": str(len(content))},
        iter_content=lambda chunk_size: iter((content,)),
        close=lambda: None,
    )


class TestOpenAIImageProvider(unittest.TestCase):
    """
    OpenAI Fuente de material de texto a imagen compatible. Igual que las pruebas de otras fuentes de material, todo usa unittest.mock
    para sustituir requests y time.sleep; la CI no depende de la red real, de API keys reales ni de cobros reales.
    """

    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_proxy_config = dict(config.proxy)
        # Las aserciones deben hacerse después de que vuelva la llamada de generación; el directorio temporal no puede destruirse antes con el bloque with,
        # por eso se gestiona su ciclo de vida con mkdtemp + addCleanup.
        self.save_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.save_dir, ignore_errors=True)
        config.app["openai_image_base_url"] = "https://img.example.com/v1"
        config.app["openai_image_api_keys"] = ["sk-test-key"]
        config.app["openai_image_model"] = "test-image-model"
        # La plantilla de prompt y el tamaño personalizado están desactivados por defecto; los casos que necesiten cubrirlos los configuran ellos mismos, para evitar que
        # los ajustes del config.toml local del desarrollador afecten a las aserciones de los escenarios de comportamiento por defecto.
        config.app.pop("openai_image_prompt_template", None)
        config.app.pop("openai_image_size", None)
        config.app.pop("tls_verify", None)
        config.proxy.clear()

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        config.proxy.clear()
        config.proxy.update(self.original_proxy_config)

    @staticmethod
    def _generated_item(term, image_path, duration=5):
        item = material.MaterialInfo()
        item.provider = "openai_image"
        item.url = image_path
        item.duration = duration
        item.source_info = {
            "provider": "openai_image",
            "search_term": term,
            "rendition": {"id": None, "width": 736, "height": 1312},
        }
        return item

    # ------------------------------------------------------------------
    # Ruta de éxito
    # ------------------------------------------------------------------

    def test_generate_images_openai_with_b64_json_response(self):
        """
        b64_json la respuesta debe decodificarse y guardarse en disco como un PNG válido, escribiendo el tamaño real de la imagen rendition
        (compatible con servicios intermediarios que devuelven un tamaño distinto al pedido), y duration registra la duración objetivo del fragmento.
        """
        image_data = _png_bytes(width=736, height=1312)
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(image_data).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            results = material.generate_images_openai(
                "sunrise over mountains",
                minimum_duration=5,
                video_aspect=material.VideoAspect.portrait,
                save_dir=self.save_dir,
            )

        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item.provider, "openai_image")
        self.assertEqual(item.duration, 5)
        # El size de la petición toma el tamaño compatible oficial de OpenAI según la proporción, sin usar directamente la resolución del video
        self.assertEqual(
            post.call_args.args[0],
            "https://img.example.com/v1/images/generations",
        )
        self.assertEqual(
            post.call_args.kwargs["json"],
            {
                "model": "test-image-model",
                "prompt": "sunrise over mountains",
                "n": 1,
                "size": "1024x1536",
            },
        )
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer sk-test-key",
        )
        # el archivo guardado es decodificable PNG
        self.assertTrue(item.url.endswith(".png"))
        self.assertTrue(os.path.isfile(item.url))
        with Image.open(item.url) as saved:
            self.assertEqual(saved.size, (736, 1312))
        # rendition registra el tamaño real de la imagen, sin depender de los parámetros de la petición
        self.assertEqual(
            item.source_info["rendition"],
            {"id": None, "width": 736, "height": 1312},
        )
        self.assertEqual(item.source_info["search_term"], "sunrise over mountains")

    def test_generate_images_openai_with_url_response(self):
        """url la respuesta debe descargar de inmediato la dirección temporal y guardarla en disco."""
        response = _image_response(
            {"data": [{"url": "https://cdn.example.com/generated/abc.png?sig=1"}]}
        )
        download = _download_response(_png_bytes(width=200, height=300))

        with (
            patch("app.services.material.requests.post", return_value=response),
            patch("app.services.material.requests.get", return_value=download) as get,
        ):
            results = material.generate_images_openai(
                "city at night", minimum_duration=3, save_dir=self.save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertTrue(os.path.isfile(results[0].url))
        # La URL temporal se descarga tal cual; los parámetros de consulta firmados no pueden eliminarse
        self.assertEqual(
            get.call_args.args[0],
            "https://cdn.example.com/generated/abc.png?sig=1",
        )

    def test_generate_images_openai_skips_b64_json_with_invalid_image(self):
        """
        La capa compatible devuelve 200 pero el body no es una imagen decodificable (por ejemplo, una página de error HTML disfrazada de JSON)
        ; en ese caso, según el contrato de la fuente de material, debe devolverse una lista vacía para que la capa superior omita esa palabra clave, en lugar de que la excepción de
        decodificación interrumpa toda la tarea.
        """
        fake_content = b"<html><body>gateway degraded</body></html>"
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(fake_content).decode("ascii")}]}
        )

        with patch("app.services.material.requests.post", return_value=response):
            results = material.generate_images_openai(
                "sunrise over mountains", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(os.listdir(self.save_dir), [])

    def test_generate_images_openai_skips_url_download_with_invalid_content(self):
        """Si la URL temporal descarga un contenido 200 que no es imagen, también se sigue la ruta de omisión."""
        response = _image_response(
            {"data": [{"url": "https://cdn.example.com/generated/abc.png?sig=1"}]}
        )
        download = _download_response(b"\x89PNG\r\n\x1a\nnot-really-a-png")

        with (
            patch("app.services.material.requests.post", return_value=response),
            patch("app.services.material.requests.get", return_value=download),
        ):
            results = material.generate_images_openai(
                "city at night", minimum_duration=3, save_dir=self.save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(os.listdir(self.save_dir), [])

    def test_generate_images_openai_propagates_image_write_failure(self):
        """
        Si la imagen ya se decodificó bien pero falla la escritura del PNG, debe interrumpirse la tarea. Este tipo de fallo suele persistir
        y afectar a las palabras clave siguientes; si se toma por un problema de una sola imagen y se continúa, se generarían peticiones de pago imposibles de guardar.
        """
        response = _image_response(
            {
                "data": [
                    {"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}
                ]
            }
        )

        with (
            patch("app.services.material.requests.post", return_value=response),
            patch.object(
                Image.Image,
                "save",
                side_effect=OSError("no space left on device"),
            ),
            self.assertRaisesRegex(OSError, "no space left on device"),
        ):
            material.generate_images_openai(
                "city at night", minimum_duration=3, save_dir=self.save_dir
            )

        self.assertEqual(os.listdir(self.save_dir), [])

    # ------------------------------------------------------------------
    # Reintento con espera y rotación de key
    # ------------------------------------------------------------------

    def test_generate_images_openai_retries_429_with_backoff(self):
        """429 es una limitación temporal de tasa; debe reintentarse con espera en lugar de dar la tarea por fallida."""
        image_data = _png_bytes()
        responses = [
            _image_response({"error": {"message": "rate limited"}}, status_code=429),
            _image_response(
                {"data": [{"b64_json": base64.b64encode(image_data).decode("ascii")}]}
            ),
        ]

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", side_effect=responses) as post,
            patch("app.services.material.time.sleep") as sleep,
        ):
            results = material.generate_images_openai(
                "ocean waves", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(post.call_count, 2)
        # Antes del primer reintento debe esperarse con espera lineal, sin saturar de inmediato la interfaz remota
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(
            sleep.call_args.args[0],
            material.OPENAI_IMAGE_RETRY_BACKOFF_SECONDS[0],
        )

    def test_generate_images_openai_rotates_key_on_401(self):
        """
        401 indica que la key actual fue rechazada. Con varias keys configuradas, el reintento debe usar get_api_key
        el mecanismo de rotación para pasar a la siguiente key, en lugar de repetir con la misma key rechazada.
        """
        config.app["openai_image_api_keys"] = ["sk-bad-key", "sk-good-key"]
        image_data = _png_bytes()

        responses = [
            _image_response({"error": {"message": "unauthorized"}}, status_code=401),
            _image_response(
                {"data": [{"b64_json": base64.b64encode(image_data).decode("ascii")}]}
            ),
        ]

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", side_effect=responses) as post,
            patch("app.services.material.time.sleep"),
        ):
            results = material.generate_images_openai(
                "forest fog", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(post.call_count, 2)
        used_keys = [
            call.kwargs["headers"]["Authorization"] for call in post.call_args_list
        ]
        # Dos peticiones consecutivas deben usar keys distintas, ambas de la lista configurada
        self.assertNotEqual(used_keys[0], used_keys[1])
        for auth in used_keys:
            self.assertIn(auth.replace("Bearer ", ""), ["sk-bad-key", "sk-good-key"])

    def test_generate_images_openai_fails_fast_on_401_with_single_key(self):
        """Con una sola key, reintentar ante 401 no tiene sentido; debe fallar rápido y devolver un resultado vacío."""
        response = _image_response(
            {"error": {"message": "unauthorized"}}, status_code=401
        )

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", return_value=response) as post,
            patch("app.services.material.time.sleep") as sleep,
        ):
            results = material.generate_images_openai(
                "desert dunes", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_generate_images_openai_returns_empty_after_retries_exhausted(self):
        """
        Agotados todos los reintentos, según el contrato de la fuente de material se devuelve una lista vacía para que la capa superior omita esa palabra clave, sin guardar
        ningún archivo residual.
        """
        response = _image_response(
            {"error": {"message": "rate limited"}}, status_code=429
        )

        with (
            patch("app.services.material.requests.post", return_value=response) as post,
            patch("app.services.material.time.sleep") as sleep,
        ):
            results = material.generate_images_openai(
                "storm clouds", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(post.call_count, material.OPENAI_IMAGE_MAX_ATTEMPTS)
        self.assertEqual(sleep.call_count, material.OPENAI_IMAGE_MAX_ATTEMPTS - 1)
        # no se generó ningún archivo residual
        self.assertEqual(os.listdir(self.save_dir), [])

    def test_generate_images_openai_redacts_api_key_in_failure_detail(self):
        """Los detalles del fallo no pueden escribir la API key en claro en el log."""
        config.app["openai_image_api_keys"] = ["sk-secret-123"]
        response = _image_response(
            {"error": {"message": "invalid key sk-secret-123 provided"}},
            status_code=401,
        )

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", return_value=response),
            patch("app.services.material.logger") as logger,
        ):
            results = material.generate_images_openai(
                "redacted term", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(results, [])
        logged = [str(call) for call in logger.error.call_args_list]
        self.assertTrue(logged)
        for message in logged:
            self.assertNotIn("sk-secret-123", message)

    def test_generate_images_openai_retries_generated_image_download(self):
        """
        La imagen ya se cobró por unidad; ante fallos intermitentes de descarga debe reintentarse la misma URL, sin volver a generar
        la misma imagen y cobrarla por duplicado.
        """
        response = _image_response(
            {"data": [{"url": "https://cdn.example.com/generated/x.png"}]}
        )
        downloads = [
            _download_response(b"", status_code=502),
            _download_response(_png_bytes()),
        ]

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", return_value=response) as post,
            patch("app.services.material.requests.get", side_effect=downloads) as get,
            patch("app.services.material.time.sleep"),
        ):
            results = material.generate_images_openai(
                "aurora", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(len(results), 1)
        # los reintentos de descarga van a la misma dirección y no se lanzó una segunda generación de pago
        self.assertEqual(post.call_count, 1)
        self.assertEqual(get.call_count, 2)
        for call in get.call_args_list:
            self.assertEqual(call.args[0], "https://cdn.example.com/generated/x.png")

    def test_generated_image_download_rejects_oversize_body(self):
        """A generated image URL must not buffer an unbounded provider body."""
        response = _download_response(b"too-large")
        response.headers = {}
        closed = []
        response.close = lambda: closed.append(True)
        with (
            patch("app.services.material.requests.get", return_value=response) as get,
            patch("app.services.material.OPENAI_IMAGE_MAX_BYTES", 5, create=True),
            patch("app.services.material.time.sleep"),
        ):
            image_bytes, error = material._openai_image_download_bytes(
                "https://cdn.example.com/generated/x.png", "test-key"
            )

        self.assertIsNone(image_bytes)
        self.assertIn("limit", error)
        self.assertTrue(all(call.kwargs.get("stream") is True for call in get.call_args_list))
        self.assertEqual(closed, [True])

    def test_generated_image_download_rejects_declared_oversize_body(self):
        """An advertised oversize image should be rejected before reading bytes."""
        response = _download_response(b"small")
        response.headers = {"Content-Length": "100"}
        response.iter_content = lambda chunk_size: self.fail("body should not be read")
        with (
            patch("app.services.material.requests.get", return_value=response),
            patch("app.services.material.OPENAI_IMAGE_MAX_BYTES", 5),
        ):
            image_bytes, error = material._openai_image_download_bytes(
                "https://cdn.example.com/generated/x.png", "test-key"
            )

        self.assertIsNone(image_bytes)
        self.assertIn("limit", error)

    def test_generated_image_response_rejects_oversize_base64(self):
        """Do not decode an unbounded inline image from a provider response."""
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(b"too-large").decode()}]}
        )
        with patch("app.services.material.OPENAI_IMAGE_MAX_BYTES", 5):
            image_bytes, error = material._parse_openai_image_response(response, "test-key")

        self.assertIsNone(image_bytes)
        self.assertIn("limit", error)

    def test_generated_image_decode_rejects_pillow_bomb_warning(self):
        """Small compressed bytes can still expand to too many image pixels."""
        image_bytes = _png_bytes(width=12, height=12)
        with (
            patch.object(Image, "MAX_IMAGE_PIXELS", 100),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with self.assertRaises(material._OpenAIImageDecodeError):
                material._save_openai_image_file(image_bytes, self.save_dir)

        self.assertEqual(os.listdir(self.save_dir), [])

    def test_generated_image_download_failure_does_not_buy_another_image(self):
        """A confirmed paid result remains a task failure if its URL cannot download."""
        signed_url = "https://cdn.example.com/generated/x.png?token=private"
        response = _image_response(
            {"data": [{"url": signed_url}]}
        )
        with (
            patch("app.services.material.requests.post", return_value=response) as post,
            patch(
                "app.services.material.requests.get",
                side_effect=requests.exceptions.ConnectionError(
                    f"download failed for {signed_url}"
                ),
            ) as get,
            patch("app.services.material.time.sleep"),
            patch("app.services.material.logger") as logger,
        ):
            with self.assertRaisesRegex(RuntimeError, "could not be downloaded"):
                material.download_videos(
                    task_id="test-openai-image-download-failure",
                    search_terms=["first", "second"],
                    source="openai_image",
                    audio_duration=10,
                    max_clip_duration=5,
                )

        self.assertEqual(post.call_count, 1)
        self.assertEqual(get.call_count, material.OPENAI_IMAGE_MAX_DOWNLOAD_ATTEMPTS)
        self.assertNotIn("token=private", str(logger.warning.call_args_list))

    def test_generate_images_openai_returns_empty_on_rejected_request(self):
        """Un rechazo de negocio (por ejemplo, política de contenido) devuelve un resultado vacío, sin reintento con espera."""
        response = _image_response(
            {"error": {"message": "content policy violation"}}, status_code=400
        )

        with (
            tempfile.TemporaryDirectory() as save_dir,
            patch("app.services.material.requests.post", return_value=response) as post,
            patch("app.services.material.time.sleep") as sleep,
        ):
            results = material.generate_images_openai(
                "blocked term", minimum_duration=5, save_dir=save_dir
            )

        self.assertEqual(results, [])
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    # ------------------------------------------------------------------
    # Interruptor de configuración
    # ------------------------------------------------------------------

    def test_is_openai_image_enabled_requires_full_configuration(self):
        """
        base_url y model son ambos imprescindibles; la API key puede estar vacía: los gateways
        ComfyUI/SD totalmente locales normalmente no requieren autenticación.
        """
        self.assertTrue(material.is_openai_image_enabled())

        config.app["openai_image_base_url"] = ""
        self.assertFalse(material.is_openai_image_enabled())
        config.app["openai_image_base_url"] = "https://img.example.com/v1"

        # gateway local sin autenticación: aunque no haya key cuenta como activado
        config.app["openai_image_api_keys"] = []
        self.assertTrue(material.is_openai_image_enabled())
        config.app["openai_image_api_keys"] = ["sk-test-key"]

        config.app["openai_image_model"] = ""
        self.assertFalse(material.is_openai_image_enabled())

    def test_generate_images_openai_sends_no_authorization_without_key(self):
        """
        Sin API key configurada debe generarse con normalidad, y la petición no lleva cabecera Authorization,
        para gateways locales de ComfyUI/SD sin autenticación.
        """
        config.app["openai_image_api_keys"] = []
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with (
            patch("app.services.material.requests.post", return_value=response) as post,
        ):
            results = material.generate_images_openai(
                "local gateway", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertNotIn("Authorization", post.call_args.kwargs["headers"])

    def test_generate_images_openai_retries_connect_timeout(self):
        """
        Un tiempo de espera en la fase de conexión indica que la petición no llegó al servidor, así que no pudo crearse una tarea de pago,
        se permite reintentar con espera.
        """
        image_data = _png_bytes()
        responses = [
            requests.exceptions.ConnectTimeout("connect timed out"),
            _image_response(
                {"data": [{"b64_json": base64.b64encode(image_data).decode("ascii")}]}
            ),
        ]

        with (
            patch("app.services.material.requests.post", side_effect=responses) as post,
            patch("app.services.material.time.sleep"),
        ):
            results = material.generate_images_openai(
                "connect timeout term", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(post.call_count, 2)

    def test_generate_images_openai_does_not_retry_unconfirmed_errors(self):
        """
        Un tiempo de espera de lectura o una conexión interrumpida es un estado "no confirmado": el servidor puede haber generado y cobrado ya, solo que
        la respuesta no volvió. Reenviar automáticamente generaría y cobraría por duplicado, así que debe
        fallar directamente y terminar la tarea, sin seguir enviando peticiones de pago para las palabras clave siguientes.
        """
        for error in (
            requests.exceptions.ReadTimeout("read timed out"),
            requests.exceptions.ConnectionError("connection dropped"),
        ):
            with self.subTest(error=type(error).__name__):
                with (
                    patch(
                        "app.services.material.requests.post", side_effect=error
                    ) as post,
                    patch("app.services.material.time.sleep") as sleep,
                ):
                    with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
                        material.generate_images_openai(
                            "unconfirmed term",
                            minimum_duration=5,
                            save_dir=self.save_dir,
                        )

                self.assertEqual(post.call_count, 1)
                sleep.assert_not_called()

    def test_download_videos_openai_image_stops_after_unconfirmed_paid_request(self):
        """Earlier images must not hide a later ambiguous paid submission."""
        image_response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )
        config.app["material_directory"] = self.save_dir

        with (
            patch(
                "app.services.material.requests.post",
                side_effect=[
                    image_response,
                    requests.exceptions.ReadTimeout("response lost"),
                    image_response,
                ],
            ) as post,
            patch(
                "app.services.material._render_openai_image_video",
                return_value="/tmp/rendered.mp4",
            ),
            patch("app.services.material._persist_material_sources"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
                material.download_videos(
                    task_id="test-openai-image-unconfirmed",
                    search_terms=["first", "uncertain", "third"],
                    source="openai_image",
                    audio_duration=20,
                    max_clip_duration=5,
                )

        self.assertEqual(post.call_count, 2)

    def test_generate_images_openai_size_defaults_and_override(self):
        """
        Por defecto se toma el tamaño compatible oficial de OpenAI según la proporción(portrait 1024x1536 /
        landscape 1536x1024);al configurar openai_image_size se sobrescribe por completo,
        para gateways locales que admiten cualquier resolución.
        """
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as default_post:
            material.generate_images_openai(
                "landscape term",
                minimum_duration=5,
                video_aspect=material.VideoAspect.landscape,
                save_dir=self.save_dir,
            )
        # En horizontal se toma por defecto el tamaño compatible oficial de OpenAI
        self.assertEqual(default_post.call_args.kwargs["json"]["size"], "1536x1024")

        config.app["openai_image_size"] = "1080x1920"
        self.addCleanup(config.app.pop, "openai_image_size", None)

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            material.generate_images_openai(
                "custom size term", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(
            post.call_args.kwargs["json"]["size"],
            "1080x1920",
        )

    def test_generate_images_openai_raises_without_base_url(self):
        """Al llamar directamente sin base_url configurado, debe lanzarse un error con indicaciones de configuración."""
        config.app["openai_image_base_url"] = ""
        with self.assertRaises(ValueError):
            material.generate_images_openai("term", minimum_duration=5)

    # ------------------------------------------------------------------
    # Plantilla de prompt
    # ------------------------------------------------------------------

    def test_generate_images_openai_applies_prompt_template(self):
        """
        Con una plantilla configurada que contiene el marcador {term}, el prompt de la petición debe ser el resultado de sustituir la plantilla,
        Se añade siempre un modificador de estilo para mejorar la coincidencia entre imagen y texto.
        """
        config.app["openai_image_prompt_template"] = (
            "cinematic photo of {term}, photorealistic, high detail"
        )
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            material.generate_images_openai(
                "un vaso de cristal a la luz del amanecer", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(
            post.call_args.kwargs["json"]["prompt"],
            "cinematic photo of un vaso de cristal a la luz del amanecer, photorealistic, high detail",
        )

    def test_generate_images_openai_sends_raw_term_without_template(self):
        """Sin plantilla configurada, el prompt debe ser la palabra clave original, con el mismo comportamiento que la versión anterior."""
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            material.generate_images_openai(
                "raw term", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(post.call_args.kwargs["json"]["prompt"], "raw term")

    def test_generate_images_openai_falls_back_when_template_lacks_placeholder(self):
        """Si la plantilla no contiene el marcador {term} no se puede inyectar la palabra clave y debe usarse el texto original."""
        config.app["openai_image_prompt_template"] = "no placeholder here"
        response = _image_response(
            {"data": [{"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}]}
        )

        with patch(
            "app.services.material.requests.post", return_value=response
        ) as post:
            material.generate_images_openai(
                "fallback term", minimum_duration=5, save_dir=self.save_dir
            )

        self.assertEqual(post.call_args.kwargs["json"]["prompt"], "fallback term")

    # ------------------------------------------------------------------
    # download_videos Distribución y generación bajo demanda
    # ------------------------------------------------------------------

    def test_download_videos_openai_image_generates_on_demand_and_stops(self):
        """
        El texto a imagen se cobra por imagen; no se puede generar para todas las palabras clave y luego elegir. El material debe generarse imagen a imagen según se necesite
        ; cuando la duración válida acumulada (limitada por la duración del fragmento) alcanza la duración de voz necesaria, las palabras clave siguientes
        ya no lanzan ninguna petición de pago.
        """
        generated = {
            "term-1": [self._generated_item("term-1", "/tmp/img-1.png")],
            "term-2": [self._generated_item("term-2", "/tmp/img-2.png")],
            "term-3": [self._generated_item("term-3", "/tmp/img-3.png")],
        }

        def fake_generate(search_term, minimum_duration, video_aspect, save_dir=""):
            return generated[search_term]

        def fake_render(image_path, clip_duration, image_motion="zoom"):
            return f"{image_path}.mp4"

        with (
            patch(
                "app.services.material.generate_images_openai",
                side_effect=fake_generate,
            ) as generate,
            patch(
                "app.services.material._render_openai_image_video",
                side_effect=fake_render,
            ) as render,
        ):
            result = material.download_videos(
                task_id="test-openai-image-lazy",
                search_terms=["term-1", "term-2", "term-3"],
                source="openai_image",
                audio_duration=8,
                max_clip_duration=5,
            )

        # 5s + 5s > 8s,La tercera palabra clave no puede generar más peticiones de generación de pago
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(
            [call.kwargs["search_term"] for call in generate.call_args_list],
            ["term-1", "term-2"],
        )
        # cada imagen solo cuenta en la duración tras renderizarse como fragmento mp4
        self.assertEqual(render.call_count, 2)
        self.assertEqual(result, ["/tmp/img-1.png.mp4", "/tmp/img-2.png.mp4"])

    def test_download_videos_openai_image_continues_after_invalid_image(self):
        """
        Si la respuesta de la primera interfaz compatible no se puede decodificar, solo se omite la palabra clave correspondiente, y una imagen válida posterior aún puede
        guardarse y renderizarse, verificando que la corrección cubre la cadena real de generación bajo demanda y no solo una función.
        """
        invalid_response = _image_response(
            {
                "data": [
                    {
                        "b64_json": base64.b64encode(
                            b"<html>gateway degraded</html>"
                        ).decode("ascii")
                    }
                ]
            }
        )
        valid_response = _image_response(
            {
                "data": [
                    {"b64_json": base64.b64encode(_png_bytes()).decode("ascii")}
                ]
            }
        )
        config.app["material_directory"] = self.save_dir

        with (
            patch(
                "app.services.material.requests.post",
                side_effect=[invalid_response, valid_response],
            ) as post,
            patch(
                "app.services.material._render_openai_image_video",
                return_value="/tmp/rendered-openai-image.mp4",
            ) as render,
            patch("app.services.material._persist_material_sources"),
        ):
            result = material.download_videos(
                task_id="test-openai-image-invalid-then-valid",
                search_terms=["invalid term", "valid term"],
                source="openai_image",
                audio_duration=5,
                max_clip_duration=5,
            )

        self.assertEqual(post.call_count, 2)
        self.assertEqual(render.call_count, 1)
        self.assertEqual(result, ["/tmp/rendered-openai-image.mp4"])

    def test_download_videos_openai_image_stops_when_duration_exactly_covered(self):
        """Regresión de límite: alcanzar exactamente la duración necesaria ya es suficiente; la condición de parada debe ser >= y no >."""
        generated = {
            "term-1": [self._generated_item("term-1", "/tmp/img-1.png")],
            "term-2": [self._generated_item("term-2", "/tmp/img-2.png")],
            "term-3": [self._generated_item("term-3", "/tmp/img-3.png")],
        }

        def fake_generate(search_term, minimum_duration, video_aspect, save_dir=""):
            return generated[search_term]

        with (
            patch(
                "app.services.material.generate_images_openai",
                side_effect=fake_generate,
            ) as generate,
            patch(
                "app.services.material._render_openai_image_video",
                return_value="/tmp/rendered.mp4",
            ),
        ):
            result = material.download_videos(
                task_id="test-openai-image-exact",
                search_terms=["term-1", "term-2", "term-3"],
                source="openai_image",
                audio_duration=10,
                max_clip_duration=5,
            )

        # 5s + 5s == 10s,cubre exactamente; el 3.er fragmento nunca debe generarse
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(len(result), 2)

    def test_download_videos_openai_image_bypasses_search_cache(self):
        """
        El resultado generado es un archivo de imagen de un solo uso y no participa en la caché de búsqueda de 24 horas: la caché haría que distintas
        tareas obtuvieran repetidamente la misma imagen. download_videos debe ir directamente a la rama de generación bajo demanda.
        """
        with (
            patch(
                "app.services.material.generate_images_openai",
                return_value=[self._generated_item("sunrise", "/tmp/img-1.png")],
            ) as generate,
            patch("app.services.material._search_videos_with_cache") as cached_search,
            patch(
                "app.services.material._render_openai_image_video",
                return_value="/tmp/img-1.png.mp4",
            ),
        ):
            result = material.download_videos(
                task_id="test-openai-image-cache-bypass",
                search_terms=["sunrise"],
                source="openai_image",
                audio_duration=5,
                max_clip_duration=5,
            )

        self.assertEqual(generate.call_count, 1)
        cached_search.assert_not_called()
        self.assertEqual(result, ["/tmp/img-1.png.mp4"])

    def test_download_videos_openai_image_skips_rejected_segment(self):
        """
        Si un resultado generado rechazado explícitamente está vacío, se omite esa palabra clave y se sigue con el siguiente fragmento.
        """
        generated = {
            "term-1": [],  # fallo de generación
            "term-2": [self._generated_item("term-2", "/tmp/img-2.png")],
            "term-3": [self._generated_item("term-3", "/tmp/img-3.png")],
        }

        def fake_generate(search_term, minimum_duration, video_aspect, save_dir=""):
            return generated[search_term]

        def fake_render(image_path, clip_duration, image_motion="zoom"):
            return f"{image_path}.mp4"

        with (
            patch(
                "app.services.material.generate_images_openai",
                side_effect=fake_generate,
            ) as generate,
            patch(
                "app.services.material._render_openai_image_video",
                side_effect=fake_render,
            ),
        ):
            result = material.download_videos(
                task_id="test-openai-image-skip",
                search_terms=["term-1", "term-2", "term-3"],
                source="openai_image",
                audio_duration=5,
                max_clip_duration=5,
            )

        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result, ["/tmp/img-2.png.mp4"])

    def test_download_videos_openai_image_stops_after_render_failure(self):
        """A paid image saved locally should not trigger another purchase if render fails."""
        generated_item = self._generated_item("first", "/tmp/img-1.png")
        with (
            patch(
                "app.services.material.generate_images_openai",
                return_value=[generated_item],
            ) as generate,
            patch("app.services.material._render_openai_image_video", return_value=""),
        ):
            with self.assertRaisesRegex(RuntimeError, "could not be rendered"):
                material.download_videos(
                    task_id="test-openai-image-render-failure",
                    search_terms=["first", "second"],
                    source="openai_image",
                    audio_duration=10,
                    max_clip_duration=5,
                )

        self.assertEqual(generate.call_count, 1)

    def test_download_videos_openai_image_skips_generation_without_audio(self):
        """Si la duración de la voz no es positiva se devuelve vacío directamente, sin pagar por imagen en una tarea que nunca podría completarse."""
        with patch("app.services.material.generate_images_openai") as generate:
            result = material.download_videos(
                task_id="test-openai-image-no-audio",
                search_terms=["term-1"],
                source="openai_image",
                audio_duration=0,
                max_clip_duration=5,
            )

        generate.assert_not_called()
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
