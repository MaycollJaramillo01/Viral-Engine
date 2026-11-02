from dataclasses import dataclass


DEFAULT_LLM_PROVIDER_ID = "moonshot"


@dataclass(frozen=True, slots=True)
class LLMProviderField:
    """Describe los campos de configuración adicionales de un Provider aparte de la API Key, la Base URL y el nombre del modelo."""

    config_suffix: str
    label_key: str
    required: bool = False
    secret: bool = False
    default_value: str = ""


@dataclass(frozen=True, slots=True)
class LLMProviderEndpoint:
    """Describe los accesos y direcciones de API asociados que un mismo Provider usa en distintas regiones de servicio."""

    endpoint_id: str
    default_label: str
    base_url: str
    api_key_url: str
    model_docs_url: str = ""


@dataclass(frozen=True, slots=True)
class LLMProviderSpec:
    """
    LLM Provider declaración centralizada.

    Aquí se guardan de forma centralizada los metadatos estables que usan la WebUI, la carga de configuración y las llamadas a servicios, incluidos el nombre
    mostrado por defecto y la locale key, pero no se guardan los textos traducidos ni se implementan las peticiones a la API. Así,
    Provider el «qué es» lo mantiene el Registry y el «cómo se llama» sigue a cargo de los adaptadores de la capa de servicio.
    """

    provider_id: str
    default_label: str
    adapter: str = "openai_compatible"
    api_key_url: str = ""
    default_model: str = ""
    default_base_url: str = ""
    model_docs_url: str = ""
    requires_api_key: bool = True
    requires_model_name: bool = True
    requires_base_url: bool = True
    show_api_key: bool = True
    show_base_url: bool = True
    deprecated_models: tuple[str, ...] = ()
    deprecated_base_urls: tuple[str, ...] = ()
    extra_fields: tuple[LLMProviderField, ...] = ()
    service_endpoints: tuple[LLMProviderEndpoint, ...] = ()
    default_service_endpoint_id: str = ""
    international_service_endpoint_id: str = ""

    @property
    def label_key(self) -> str:
        return f"llm_provider_label.{self.provider_id}"

    @property
    def tips_key(self) -> str:
        return f"llm_provider_tips.{self.provider_id}"

    @property
    def endpoint_selector_label_key(self) -> str:
        return f"llm_provider_endpoint_selector.{self.provider_id}"

    @property
    def endpoint_selector_help_key(self) -> str:
        return f"llm_provider_endpoint_selector_help.{self.provider_id}"

    @property
    def authentication_error_key(self) -> str:
        return f"llm_provider_authentication_error.{self.provider_id}"

    def endpoint_label_key(self, endpoint_id: str) -> str:
        return f"llm_provider_endpoint.{self.provider_id}.{endpoint_id}"

    def config_key(self, suffix: str) -> str:
        return f"{self.provider_id}_{suffix}"

    def resolve_model_name(self, configured_model: str | None) -> str:
        """Resuelve los valores vacíos o los valores por defecto históricos obsoletos al modelo por defecto actual."""
        model_name = (configured_model or "").strip()
        if not model_name or model_name in self.deprecated_models:
            return self.default_model
        return model_name

    def resolve_base_url(self, configured_base_url: str | None) -> str:
        """Resuelve la Base URL y migra las direcciones históricas ya dadas de baja al valor por defecto actual."""
        base_url = (configured_base_url or "").strip()
        deprecated_urls = {url.rstrip("/") for url in self.deprecated_base_urls}
        if not base_url or base_url.rstrip("/") in deprecated_urls:
            return self.effective_default_base_url
        return base_url

    def get_service_endpoint(self, endpoint_id: str) -> LLMProviderEndpoint | None:
        """Obtiene la región de servicio por un ID estable, para que la lógica de negocio no dependa de enlaces promocionales cambiantes."""
        return next(
            (
                endpoint
                for endpoint in self.service_endpoints
                if endpoint.endpoint_id == endpoint_id
            ),
            None,
        )

    @property
    def default_service_endpoint(self) -> LLMProviderEndpoint | None:
        """Devuelve la región de servicio por defecto declarada por el Provider."""
        return self.get_service_endpoint(self.default_service_endpoint_id)

    @property
    def international_service_endpoint(self) -> LLMProviderEndpoint | None:
        """Devuelve la región de servicio internacional declarada por el Provider."""
        return self.get_service_endpoint(self.international_service_endpoint_id)

    @property
    def effective_default_base_url(self) -> str:
        """Lee primero la Base URL de la región de servicio por defecto; los Providers normales siguen usando el campo original."""
        endpoint = self.default_service_endpoint
        return endpoint.base_url if endpoint else self.default_base_url

    def preferred_service_endpoint(
        self, *, prefer_international: bool
    ) -> LLMProviderEndpoint | None:
        """Devuelve el acceso preferido según la región de la interfaz; si falta el acceso internacional, recurre con seguridad al acceso por defecto."""
        if prefer_international and self.international_service_endpoint:
            return self.international_service_endpoint
        return self.default_service_endpoint

    def effective_api_key_url(self, *, prefer_international: bool = False) -> str:
        """Resuelve de forma unificada el acceso para solicitar la API Key, para que los Endpoint Providers no mantengan enlaces duplicados."""
        endpoint = self.preferred_service_endpoint(
            prefer_international=prefer_international
        )
        return endpoint.api_key_url if endpoint else self.api_key_url

    def effective_model_docs_url(self, *, prefer_international: bool = False) -> str:
        """Resuelve de forma unificada los accesos a la lista de modelos y la documentación, para que los Endpoint Providers no mantengan enlaces duplicados."""
        endpoint = self.preferred_service_endpoint(
            prefer_international=prefer_international
        )
        return endpoint.model_docs_url if endpoint and endpoint.model_docs_url else self.model_docs_url

    def find_service_endpoint(
        self, configured_base_url: str | None
    ) -> LLMProviderEndpoint | None:
        """Identifica la región de servicio estándar del Provider a partir de la Base URL guardada."""
        normalized_url = (configured_base_url or "").strip().rstrip("/")
        if not normalized_url:
            return None
        return next(
            (
                endpoint
                for endpoint in self.service_endpoints
                if endpoint.base_url.rstrip("/") == normalized_url
            ),
            None,
        )

    def select_service_endpoint(
        self,
        configured_base_url: str | None,
        *,
        has_api_key: bool,
        prefer_international: bool,
    ) -> LLMProviderEndpoint | None:
        """
        Elige la región de servicio estándar que debe mostrar la WebUI.

        Tiene prioridad la dirección estándar guardada explícitamente; las direcciones desconocidas se mantienen como personalizadas. Las configuraciones históricas pueden tener solo
        API Key sin Base URL; esos usuarios siguen usando la región por defecto del Registry, para evitar
        que tras actualizar se cambie de servicio por usar otro idioma de interfaz. Solo las configuraciones nuevas eligen según el idioma de la interfaz
        el acceso internacional.
        """
        configured_url = (configured_base_url or "").strip()
        if configured_url:
            return self.find_service_endpoint(configured_url)

        default_endpoint = self.default_service_endpoint
        if has_api_key or not prefer_international:
            return default_endpoint

        return self.preferred_service_endpoint(
            prefer_international=prefer_international
        )


# El orden de la tupla es el orden del desplegable de la WebUI. Para añadir un Provider OpenAI-compatible normal,
# normalmente basta con añadir un elemento aquí y completar la locale; solo los Providers con un protocolo distinto necesitan
# app/services/llm.py añadir la implementación del adapter correspondiente.
LLM_PROVIDER_REGISTRY = (
    # Recomendado Provider
    LLMProviderSpec(
        "moonshot",
        "Kimi / Moonshot AI",
        default_model="kimi-k3",
        service_endpoints=(
            LLMProviderEndpoint(
                endpoint_id="china",
                default_label="China",
                base_url="https://api.moonshot.cn/v1",
                api_key_url=(
                    "https://platform.kimi.com?"
                    "track_id=track-6eec1e56a4494e52adcaebbcbbefce59&"
                    "aff=moneyprinterturbo"
                ),
                model_docs_url=(
                    "https://platform.kimi.com/docs/models?"
                    "track_id=track-6eec1e56a4494e52adcaebbcbbefce59&"
                    "aff=moneyprinterturbo"
                ),
            ),
            LLMProviderEndpoint(
                endpoint_id="global",
                default_label="Global",
                base_url="https://api.moonshot.ai/v1",
                api_key_url=(
                    "https://platform.kimi.ai?"
                    "track_id=track-9e3b711aa2594e378f6fe5b8de718a76&"
                    "aff=moneyprinterturbo"
                ),
                model_docs_url=(
                    "https://platform.kimi.ai/docs/models?"
                    "track_id=track-9e3b711aa2594e378f6fe5b8de718a76&"
                    "aff=moneyprinterturbo"
                ),
            ),
        ),
        default_service_endpoint_id="china",
        international_service_endpoint_id="global",
    ),
    # Kimi La suscripción (beneficio Kimi Code de los planes Plus o superiores) usa la autorización de dispositivo OAuth:
    # tras confirmar con un clic en el navegador, la capa de servicio llama directamente con el Bearer token al endpoint
    # OpenAI compatible oficial, sin necesidad de solicitar una API Key de pago por uso. El sitio de China y Global usan cada uno
    # su propio sistema de cuentas y autorización; el endpoint sincroniza la página de autorización, la dirección de la API y el enlace de suscripción.
    LLMProviderSpec(
        "kimi_code",
        "Kimi Code (Kimi subscription)",
        adapter="kimi_code",
        requires_api_key=False,
        show_api_key=False,
        requires_base_url=False,
        show_base_url=False,
        requires_model_name=False,
        default_model="kimi-for-coding",
        extra_fields=(
            LLMProviderField("timeout", "Timeout (seconds)", default_value="300"),
        ),
        service_endpoints=(
            LLMProviderEndpoint(
                endpoint_id="china",
                default_label="China",
                base_url="https://api.kimi.com/coding/v1",
                api_key_url="https://www.kimi.com/code?aff=moneyprinterturbo",
                model_docs_url="https://platform.kimi.com/docs/models?aff=moneyprinterturbo",
            ),
            LLMProviderEndpoint(
                endpoint_id="global",
                default_label="Global",
                base_url="https://api.kimi.ai/coding/v1",
                api_key_url="https://www.kimi.ai/code?aff=moneyprinterturbo",
                model_docs_url="https://platform.kimi.ai/docs/models?aff=moneyprinterturbo",
            ),
        ),
        default_service_endpoint_id="china",
        international_service_endpoint_id="global",
    ),
    # Fabricantes de modelos y proveedores cloud principales
    LLMProviderSpec(
        "openai",
        "OpenAI",
        api_key_url="https://platform.openai.com/api-keys",
        default_model="gpt-5.5",
        default_base_url="https://api.openai.com/v1",
    ),
    LLMProviderSpec(
        "openai_codex",
        "OpenAI - Codex",
        adapter="openai_codex",
        requires_api_key=False,
        show_api_key=False,
        requires_base_url=False,
        show_base_url=False,
        extra_fields=(
            LLMProviderField("timeout", "Timeout (seconds)", default_value="300"),
        ),
    ),
    LLMProviderSpec(
        "openai_chatgpt",
        "OpenAI - ChatGPT (Native OAuth)",
        adapter="openai_chatgpt",
        requires_api_key=False,
        show_api_key=False,
        requires_base_url=False,
        show_base_url=False,
        extra_fields=(
            LLMProviderField("timeout", "Timeout (seconds)", default_value="300"),
        ),
    ),
    LLMProviderSpec(
        "anthropic",
        "Anthropic Claude",
        api_key_url="https://platform.claude.com/settings/keys",
        default_model="claude-sonnet-5",
        default_base_url="https://api.anthropic.com/v1/",
    ),
    LLMProviderSpec(
        "gemini",
        "Google Gemini",
        adapter="gemini",
        api_key_url="https://aistudio.google.com/app/apikey",
        default_model="gemini-3.1-pro-preview",
        requires_base_url=False,
        show_base_url=False,
        deprecated_models=("gemini-pro", "gemini-1.0-pro"),
    ),
    LLMProviderSpec(
        "muse_spark",
        "Muse Spark",
        api_key_url="https://dev.meta.ai/docs/authentication",
        default_model="muse-spark-1.3",
        default_base_url="https://api.meta.ai/v1",
        model_docs_url="https://dev.meta.ai/docs/models",
    ),
    LLMProviderSpec(
        "deepseek",
        "DeepSeek",
        api_key_url="https://platform.deepseek.com/api_keys",
        default_model="deepseek-v4-pro",
        default_base_url="https://api.deepseek.com",
    ),
    LLMProviderSpec(
        "qwen",
        "Alibaba Cloud Qwen",
        adapter="qwen",
        api_key_url="https://dashscope.console.aliyun.com/apiKey",
        default_model="qwen-max",
        requires_base_url=False,
        show_base_url=False,
    ),
    LLMProviderSpec(
        "azure",
        "Microsoft Azure OpenAI",
        adapter="azure",
        api_key_url=(
            "https://portal.azure.com/#view/"
            "Microsoft_Azure_ProjectOxford/CognitiveServicesHub/~/OpenAI"
        ),
        default_model="gpt-35-turbo",
    ),
    LLMProviderSpec(
        "volcengine",
        "ByteDance VolcEngine Ark",
        api_key_url=(
            "https://www.volcengine.com/activity/ai618?utm_campaign=hw&"
            "utm_content=hw&utm_medium=devrel_tool_web&utm_source=OWO&"
            "utm_term=MoneyPrinterTurbo"
        ),
        default_model="doubao-seed-2-1-turbo-260628",
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
    ),
    LLMProviderSpec(
        "grok",
        "xAI Grok",
        api_key_url="https://console.x.ai/",
        default_model="grok-4.3",
        default_base_url="https://api.x.ai/v1",
    ),
    LLMProviderSpec(
        "minimax",
        "MiniMax",
        api_key_url="https://platform.minimax.io/",
        default_model="MiniMax-M3",
        default_base_url="https://api.minimax.io/v1",
    ),
    LLMProviderSpec(
        "mimo",
        "Xiaomi MiMo",
        api_key_url=(
            "https://platform.xiaomimimo.com/docs/zh-CN/quick-start/first-api-call"
        ),
        default_model="mimo-v2.5-pro",
        default_base_url="https://api.xiaomimimo.com/v1",
    ),
    # El pago por uso y el Token Plan de iFlytek Xingchen MaaS usan direcciones OpenAI-compatibles distintas,
    # y las API Keys de ambos no son intercambiables. Aquí el valor por defecto es pago por uso; los usuarios de Token Plan Base URL
    # solo tienen que cambiar a la dirección de maas-token-api en
    LLMProviderSpec(
        "iflytek",
        "iFlytek Spark (Astron MaaS)",
        api_key_url="https://maas.xfyun.cn/",
        default_model="spark-x2.5",
        default_base_url="https://maas-api.cn-huabei-1.xf-yun.com/v2",
        model_docs_url="https://maas.xfyun.cn/modelSquare",
    ),
    # Plataformas de agregación y acceso unificado
    LLMProviderSpec(
        "jalapeno",
        "Jalapeno Cloud",
        api_key_url="https://www.jalapeno-cloud.ai/MoneyPrinterTurbo",
        default_model="DeepSeek-V4.1-Flash",
        default_base_url="https://api.jalapeno-cloud.ai/v1",
        model_docs_url="https://www.jalapeno-cloud.ai/models",
    ),
    LLMProviderSpec(
        "infistar",
        "Infistar",
        api_key_url="https://infistar.cc/keys",
        default_model="gpt-6-luna",
        default_base_url="https://infistar.cc/v1",
        model_docs_url="https://infistar.cc/pricing",
    ),
    LLMProviderSpec(
        "shengsuanyun",
        "Shengsuan Cloud",
        api_key_url="https://www.shengsuanyun.com/?from=CH_XUQ4OTSK",
        default_model="deepseek/deepseek-v4-flash",
        default_base_url="https://router.shengsuanyun.com/api/v1",
    ),
    # APIMart ofrece a la vez la interfaz de negocio `/api/v1` y la interfaz compatible con OpenAI `/v1`.
    # La capa de servicio LLM actual depende de que el SDK de OpenAI lea directamente choices, así que se debe usar el acceso `/v1` sin
    # code/data envoltorio exterior, y no copiar la dirección de la interfaz de negocio asíncrona.
    LLMProviderSpec(
        "apimart",
        "APIMart",
        api_key_url="https://go.apimart.ai/gh-moneyprinterturbo",
        default_model="gpt-5.6-terra",
        default_base_url="https://api.apimart.ai/v1",
    ),
    LLMProviderSpec(
        "cloudflare",
        "Cloudflare AI Gateway",
        adapter="cloudflare_ai_gateway",
        api_key_url="https://dash.cloudflare.com/",
        default_model="openai/gpt-4.1-mini",
        requires_base_url=False,
        show_base_url=False,
        deprecated_models=("@cf/meta/llama-3.1-8b-instruct",),
        extra_fields=(
            LLMProviderField("account_id", "Account ID", required=True),
            LLMProviderField(
                "gateway_id",
                "Gateway ID",
                default_value="default",
            ),
        ),
    ),
    LLMProviderSpec(
        "modelscope",
        "Alibaba ModelScope",
        adapter="modelscope",
        api_key_url=("https://modelscope.cn/docs/model-service/API-Inference/intro"),
        default_model="ZhipuAI/GLM-5.2",
        default_base_url="https://api-inference.modelscope.cn/v1/",
    ),
    LLMProviderSpec(
        "aihubmix",
        "AIHubMix",
        api_key_url="https://aihubmix.com/",
        default_model="gpt-5.4-mini",
        default_base_url="https://aihubmix.com/v1",
    ),
    LLMProviderSpec(
        "aimlapi",
        "AIML API",
        api_key_url="https://aimlapi.com/app/keys",
        default_model="openai/gpt-5-5",
        default_base_url="https://api.aimlapi.com/v1",
    ),
    LLMProviderSpec(
        "evolink",
        "EvoLink",
        api_key_url="https://evolink.ai/dashboard/keys",
        default_model="gpt-5.5",
        default_base_url="https://direct.evolink.ai/v1",
    ),
    LLMProviderSpec(
        "openrouter",
        "OpenRouter",
        api_key_url="https://openrouter.ai/settings/keys",
        default_model="minimax/minimax-m3:free",
        default_base_url="https://openrouter.ai/api/v1",
    ),
    LLMProviderSpec(
        "api_route",
        "API Route",
        api_key_url="https://www.api-route.com",
        default_model="gpt-5.4-mini",
        default_base_url="https://www.api-route.com/v1",
        model_docs_url="https://www.api-route.com/pricing",
    ),
    # Fluxion el grupo OpenAI usa la interfaz /v1 y reutiliza el adaptador Chat Completions existente.
    # El grupo determina los modelos y direcciones disponibles, así que aquí solo se dan valores por defecto, manteniendo la posibilidad de que el usuario los sobrescriba;
    # Anthropic/Gemini los grupos nativos no pueden usar este acceso directamente, para evitar incompatibilidades de protocolo.
    LLMProviderSpec(
        "fluxionai",
        "Fluxion AI",
        api_key_url=(
            "https://fluxionai.space/register?source=github"
            "&campaign=moneyprinterturbo&promo=MONEYPRINTERTURBO"
        ),
        default_model="gpt-5.5",
        default_base_url="https://fluxionai.space/v1",
        model_docs_url="https://fluxionai.space/model-plaza",
    ),
    LLMProviderSpec(
        "cheaperinference",
        "Cheaper Inference",
        api_key_url="https://cheaperinference.com/signup",
        default_model="gpt-5.4-mini",
        default_base_url="https://api.cheaperinference.com/v1",
        model_docs_url="https://cheaperinference.com/#models",
    ),
    LLMProviderSpec(
        "requesty",
        "Requesty",
        api_key_url="https://app.requesty.ai/api-keys",
        default_model="openai/gpt-5.4-mini",
        default_base_url="https://router.requesty.ai/v1",
        model_docs_url="https://www.requesty.ai/models",
    ),
    LLMProviderSpec(
        "futureinfra",
        "FutureInfra",
        api_key_url="https://futureinfra.ai/console/?screen=ai-router",
        default_model="openai/gpt-4o-mini",
        default_base_url="https://futureinfra.ai/v1/ai",
        model_docs_url="https://futureinfra.ai/ai/",
    ),
    LLMProviderSpec(
        "yapi",
        "Y-API",
        api_key_url="https://y-api.bestvirtualgoods.com/app/keys",
        default_model="deepseek/deepseek-v4-flash",
        default_base_url="https://api.y-api.bestvirtualgoods.com/v1",
        model_docs_url="https://y-api.bestvirtualgoods.com/models",
    ),
    LLMProviderSpec(
        "opper",
        "Opper",
        api_key_url="https://platform.opper.ai",
        default_model="gpt-5.4-mini",
        default_base_url="https://api.opper.ai/v3/compat",
        model_docs_url="https://opper.ai/models",
    ),
    LLMProviderSpec(
        "onomeo",
        "onomeo",
        api_key_url="https://onomeo.com/dashboard",
        default_model="auto",
        default_base_url="https://onomeo.com/v1",
        model_docs_url="https://onomeo.com/models",
    ),
    LLMProviderSpec(
        "empiriolabs",
        "EmpirioLabs AI",
        api_key_url="https://platform.empiriolabs.ai/dashboard/api-keys",
        default_model="qwen3-8-flash",
        default_base_url="https://api.empiriolabs.ai/v1",
        model_docs_url="https://docs.empiriolabs.ai/models-pricing",
    ),
    LLMProviderSpec(
        "runcrate",
        "Runcrate",
        api_key_url="https://www.runcrate.ai/dashboard/api-keys?utm_source=MoneyPrinterTurbo&utm_medium=directory&utm_campaign=gpu_pricing",
        default_model="deepseek-ai/DeepSeek-V4.1-Flash",
        default_base_url="https://api.runcrate.ai/v1",
        model_docs_url="https://www.runcrate.ai/inference/pricing?utm_source=MoneyPrinterTurbo&utm_medium=directory&utm_campaign=gpu_pricing",
    ),
    LLMProviderSpec(
        "cloudgpu",
        "CloudGPU",
        api_key_url="https://cloudgpu.app",
        default_model="deepseek-v4-flash",
        default_base_url="https://cloudgpu.app/v1",
        model_docs_url="https://cloudgpu.app/v1/models",
    ),
    # Despliegue local y gateways genéricos
    LLMProviderSpec(
        "ollama",
        "Ollama",
        requires_api_key=False,
        show_api_key=False,
    ),
    # Claude Las suscripciones (Pro / Max / Team) no emiten API Keys; las credenciales solo las puede usar Claude Code
    # el cliente oficial, por eso este Provider no usa una interfaz HTTP sino que llama al claude CLI
    # ya autenticado en la máquina. Si el nombre del modelo se deja vacío se usa el modelo por defecto actual del CLI.
    LLMProviderSpec(
        "claude_code",
        "Claude Code (Claude subscription)",
        adapter="claude_code",
        requires_api_key=False,
        show_api_key=False,
        requires_base_url=False,
        show_base_url=False,
        requires_model_name=False,
        extra_fields=(
            LLMProviderField("cli_path", "Claude CLI Path"),
            LLMProviderField("timeout", "Timeout (seconds)", default_value="300"),
        ),
    ),
    LLMProviderSpec(
        "oneapi",
        "OneAPI",
        api_key_url="https://github.com/songquanpeng/one-api",
    ),
    LLMProviderSpec(
        "litellm",
        "LiteLLM",
        adapter="litellm",
        default_model="openai/gpt-4o-mini",
        requires_api_key=False,
        requires_base_url=False,
        show_api_key=False,
        show_base_url=False,
    ),
    # Otros servicios de inferencia y públicos
    LLMProviderSpec(
        "groq",
        "Groq",
        api_key_url="https://console.groq.com/keys",
        default_model="openai/gpt-oss-120b",
        default_base_url="https://api.groq.com/openai/v1",
    ),
    LLMProviderSpec(
        "pollinations",
        "Pollinations AI",
        api_key_url="https://enter.pollinations.ai/",
        default_model="openai-fast",
        default_base_url="https://gen.pollinations.ai/v1",
        deprecated_models=("default",),
        deprecated_base_urls=("https://text.pollinations.ai/openai",),
    ),
)

LLM_PROVIDERS = {provider.provider_id: provider for provider in LLM_PROVIDER_REGISTRY}

if len(LLM_PROVIDERS) != len(LLM_PROVIDER_REGISTRY):
    raise RuntimeError("duplicate LLM provider id in registry")


def get_llm_provider(provider_id: str) -> LLMProviderSpec | None:
    return LLM_PROVIDERS.get((provider_id or "").lower())


def normalize_provider_override(value: str | None, default_value: str | None) -> str:
    """
    Solo conserva los valores sobrescritos por el usuario que difieren de los valores por defecto del Registry.

    WebUI Hay que mostrar el valor por defecto en el campo de entrada, pero eso no debe fijar el valor por defecto en config.toml;
    de lo contrario, al actualizar el modelo o la dirección por defecto del Registry, la configuración antigua seguiría sobrescribiendo el nuevo valor por defecto.
    """
    normalized_value = (value or "").strip()
    normalized_default = (default_value or "").strip()
    if normalized_value == normalized_default:
        return ""
    return normalized_value
