"""Fluxion AI Regresión de integración: usa la codificación/decodificación real del SDK, y la capa HTTP no accede a Internet ni consume cuota."""

import json
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI

from app.config import config
from app.models.llm_provider import get_llm_provider
from app.services import llm


@pytest.mark.parametrize(
    "base_url,model,expected_url,expected_model",
    [
        ("", "", "https://fluxionai.space/v1/chat/completions", "gpt-5.5"),
        ("  ", "  ", "https://fluxionai.space/v1/chat/completions", "gpt-5.5"),
        (
            "https://gateway.example.com/custom/v1/",
            "custom-group-model",
            "https://gateway.example.com/custom/v1/chat/completions",
            "custom-group-model",
        ),
    ],
)
def test_fluxionai_defaults_and_group_overrides(
    base_url, model, expected_url, expected_model
):
    """Una configuración vacía usa los valores por defecto, los valores personalizados del grupo no se sobrescriben, y la configuración de otros Providers se mantiene igual."""
    snapshot = dict(config.app)
    requests = []

    def respond(request):
        requests.append(request)
        assert str(request.url) == expected_url
        assert request.headers["Authorization"] == "Bearer test-fluxion-key"
        assert json.loads(request.content) == {
            "model": expected_model,
            "messages": [{"role": "user", "content": "Generar texto en español"}],
        }
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": expected_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "¡Hola, mundo!"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    # Se conserva la concatenación de URL y el análisis de respuestas del SDK real, sustituyendo solo la capa de transporte HTTP, para que los mocks no oculten problemas de protocolo.
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        with OpenAI(
            api_key="test-fluxion-key",
            base_url=base_url.strip() or "https://fluxionai.space/v1",
            http_client=http_client,
            max_retries=0,
        ) as client:
            with patch.object(llm, "OpenAI", return_value=client) as factory:
                result = llm._generate_response(
                    "Generar texto en español",
                    app_config={
                        "llm_provider": "fluxionai",
                        "fluxionai_api_key": "test-fluxion-key",
                        "fluxionai_base_url": base_url,
                        "fluxionai_model_name": model,
                    },
                )
                factory.assert_called_once_with(
                    api_key="test-fluxion-key",
                    base_url=base_url.strip() or "https://fluxionai.space/v1",
                )
    assert result == "¡Hola, mundo!"
    assert len(requests) == 1
    assert config.app == snapshot


def test_fluxionai_missing_key_fails_before_network():
    """Si falta la clave se usa el mensaje de error unificado, sin crear el cliente ni lanzar peticiones de pago."""
    with patch.object(llm, "OpenAI") as factory:
        result = llm._generate_response(
            "test", app_config={"llm_provider": "fluxionai"}
        )
    assert result.startswith("Error:")
    assert "fluxionai: api_key is not set" in result
    factory.assert_not_called()


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_fluxionai_http_errors_are_not_successful_text(status):
    """Los errores de autenticación, permisos de grupo, límites de tasa y del servidor deben ir por la ruta de error, sin tratarse como texto normal."""

    def respond(request):
        return httpx.Response(
            status,
            json={"error": {"message": "test upstream failure", "type": "api_error"}},
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        with OpenAI(
            api_key="test-key",
            base_url="https://fluxionai.space/v1",
            http_client=http_client,
            max_retries=0,
        ) as client:
            with patch.object(llm, "OpenAI", return_value=client):
                result = llm._generate_response(
                    "test",
                    app_config={
                        "llm_provider": "fluxionai",
                        "fluxionai_api_key": "test-key",
                    },
                )
    assert result.startswith("Error:")
    assert str(status) in result


def test_fluxionai_registry_metadata():
    """Fija el modelo por defecto, el protocolo, los parámetros promocionales y los enlaces de modelos, para que la información no sea incoherente entre entradas."""
    provider = get_llm_provider("fluxionai")
    assert provider.default_model == "gpt-5.5"
    assert provider.default_base_url == "https://fluxionai.space/v1"
    assert provider.adapter == "openai_compatible"
    assert provider.requires_api_key
    assert (
        provider.api_key_url
        == "https://fluxionai.space/register?source=github&campaign=moneyprinterturbo&promo=MONEYPRINTERTURBO"
    )
    assert provider.model_docs_url == "https://fluxionai.space/model-plaza"
