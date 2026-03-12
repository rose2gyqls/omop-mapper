"""
LLM Client Module

Provides a unified LangChain-based interface for multiple LLM routes:
- openai: OpenAI API route (model configurable via env/CLI/API)
- together: Together AI serverless route
"""

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load .env defaults without overriding real process environment variables.
load_dotenv(override=False)

logger = logging.getLogger(__name__)

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    logger.warning("LangChain OpenAI packages not installed. LLM features unavailable.")


class LLMProvider:
    """Available LLM routes."""

    OPENAI = "openai"
    TOGETHER = "together"


PROVIDER_ALIASES = {
    LLMProvider.OPENAI: LLMProvider.OPENAI,
    "openai_api": LLMProvider.OPENAI,
    "openai-open": LLMProvider.OPENAI,
    "openai_open": LLMProvider.OPENAI,
    LLMProvider.TOGETHER: LLMProvider.TOGETHER,
    "together_ai": LLMProvider.TOGETHER,
    "together-ai": LLMProvider.TOGETHER,
}


TOGETHER_MODEL_ALIASES = {
    "mistral_small_24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    "gpt_oss_20b": "openai/gpt-oss-20b",
    "llama4_maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
}


DEFAULT_CONFIGS = {
    LLMProvider.OPENAI: {
        "model": "gpt-5-mini-2025-08-07",
        "base_url": None,
        "supports_json_mode": True,
    },
    LLMProvider.TOGETHER: {
        "model": "openai/gpt-oss-20b",
        "base_url": "https://api.together.xyz/v1",
        "supports_json_mode": False,
    },
}


def normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases to canonical keys."""

    raw = (provider or LLMProvider.OPENAI).strip().lower()
    return PROVIDER_ALIASES.get(raw, LLMProvider.OPENAI)


def _provider_env_prefix(provider: str) -> str:
    return {
        LLMProvider.OPENAI: "OPENAI",
        LLMProvider.TOGETHER: "TOGETHER",
    }[provider]


def _clean_optional_env(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _escape_prompt_content(text: str) -> str:
    """Escape braces so ChatPromptTemplate treats content as literal text."""

    return text.replace("{", "{{").replace("}", "}}")


def _normalize_prompt_role(role: str) -> str:
    role = role.strip().lower()
    if role == "user":
        return "human"
    if role == "assistant":
        return "ai"
    return role


def _get_env_value(prefix: str, field: str, fallback: Optional[str] = None) -> Optional[str]:
    value = os.getenv(f"{prefix}_{field}")
    if value is None and fallback:
        value = os.getenv(fallback)
    return _clean_optional_env(value)


def _get_env_float(prefix: str, field: str, fallback_key: str, default: float) -> float:
    raw = os.getenv(f"{prefix}_{field}") or os.getenv(fallback_key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s value '%s'; using default %.3f", fallback_key, raw, default)
        return default


def _get_env_int(prefix: str, field: str, fallback_key: str, default: int) -> int:
    raw = os.getenv(f"{prefix}_{field}") or os.getenv(fallback_key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s value '%s'; using default %d", fallback_key, raw, default)
        return default


def _default_max_tokens_for_model(model: str) -> int:
    """
    Return a safe default output-token cap for well-known models.

    GPT-5 family currently supports up to 128,000 output tokens. For unknown
    models, keep a conservative fallback unless overridden by env/CLI/API.
    """

    model_name = (model or "").strip().lower()
    if model_name.startswith("gpt-5"):
        return 128000
    return 2048


def normalize_model_name(provider: str, model: Optional[str]) -> Optional[str]:
    """Normalize provider-specific model aliases to canonical model IDs."""

    if model is None:
        return None

    model_name = model.strip()
    if not model_name:
        return None

    if provider == LLMProvider.TOGETHER:
        return TOGETHER_MODEL_ALIASES.get(model_name.lower(), model_name)

    return model_name


def get_env_config(provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve LLM configuration from environment variables.

    Precedence:
    1. Provider-specific env vars for the resolved provider
    2. Shared LLM_* env vars
    3. Provider defaults
    """

    load_dotenv(override=False)

    active_provider = normalize_provider(os.getenv("LLM_PROVIDER", LLMProvider.OPENAI))
    resolved_provider = normalize_provider(provider or active_provider)
    defaults = DEFAULT_CONFIGS.get(resolved_provider, DEFAULT_CONFIGS[LLMProvider.OPENAI])
    prefix = _provider_env_prefix(resolved_provider)
    use_shared_provider_overrides = provider is None or resolved_provider == active_provider

    api_key = _get_env_value(prefix, "API_KEY")

    shared_model = _clean_optional_env(os.getenv("LLM_MODEL")) if use_shared_provider_overrides else None
    resolved_model = normalize_model_name(
        resolved_provider,
        (
            _get_env_value(prefix, "MODEL")
            or shared_model
            or defaults["model"]
        ),
    )
    default_max_tokens = _default_max_tokens_for_model(resolved_model)

    shared_base_url = _clean_optional_env(os.getenv("LLM_BASE_URL")) if use_shared_provider_overrides else None
    return {
        "provider": resolved_provider,
        "model": resolved_model,
        "base_url": (
            _get_env_value(prefix, "BASE_URL")
            or shared_base_url
            or defaults["base_url"]
        ),
        "api_key": api_key,
        "temperature": _get_env_float(prefix, "TEMPERATURE", "LLM_TEMPERATURE", 0.3),
        "top_p": _get_env_float(prefix, "TOP_P", "LLM_TOP_P", 1.0),
        "max_tokens": _get_env_int(prefix, "MAX_TOKENS", "LLM_MAX_TOKENS", default_max_tokens),
        "supports_json_mode": defaults["supports_json_mode"],
    }


class LLMClient:
    """
    LangChain-based LLM client using ChatOpenAI for OpenAI and OpenAI-compatible
    backends.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_env_config: bool = True,
    ):
        env_config = get_env_config(provider=provider) if use_env_config else {
            "provider": normalize_provider(provider),
            "model": None,
            "base_url": None,
            "api_key": None,
            "temperature": 0.3,
            "top_p": 1.0,
            "max_tokens": 2048,
            "supports_json_mode": DEFAULT_CONFIGS[normalize_provider(provider)]["supports_json_mode"],
        }

        self.provider = normalize_provider(provider or env_config["provider"])
        defaults = DEFAULT_CONFIGS[self.provider]
        self.model = normalize_model_name(
            self.provider,
            model or env_config["model"] or defaults["model"],
        )
        self.base_url = _clean_optional_env(base_url) if base_url is not None else env_config["base_url"]
        self.api_key = _clean_optional_env(api_key) if api_key is not None else env_config["api_key"]
        self.temperature = temperature if temperature is not None else env_config["temperature"]
        self.top_p = top_p if top_p is not None else env_config["top_p"]
        self.max_tokens = max_tokens if max_tokens is not None else env_config["max_tokens"]
        self.supports_json_mode = env_config["supports_json_mode"]
        self.use_env_config = use_env_config

        self.client = None
        self._initialize_client()

    def _resolved_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.provider == LLMProvider.OPENAI:
            return None
        return "dummy"

    def _build_chat_model(
        self,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ):
        if not HAS_LANGCHAIN:
            return None

        api_key = self._resolved_api_key()
        if self.provider == LLMProvider.OPENAI and not api_key:
            logger.error("API key required for OpenAI provider")
            return None

        model_kwargs: Dict[str, Any] = {}
        effective_top_p = self.top_p if top_p is None else top_p
        if json_mode and self.supports_json_mode:
            model_kwargs["response_format"] = {"type": "json_object"}

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "api_key": api_key,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": effective_top_p,
            "max_retries": 2,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        kwargs["max_tokens"] = self.max_tokens if max_tokens is None else max_tokens
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs

        return ChatOpenAI(**kwargs)

    def _initialize_client(self):
        """Initialize the default chat model."""

        if not HAS_LANGCHAIN:
            logger.error("LangChain OpenAI packages not installed")
            self.client = None
            return

        try:
            self.client = self._build_chat_model()
            if self.client is None:
                return

            logger.info(
                "LLM Client initialized: provider=%s, model=%s, base_url=%s",
                self.provider,
                self.model,
                self.base_url or "default",
            )
        except Exception as e:
            logger.error("LLM Client initialization failed: %s", e)
            self.client = None

    def reload_config(self):
        """Reload configuration from environment variables."""

        env_config = get_env_config()
        self.provider = env_config["provider"]
        self.model = env_config["model"]
        self.base_url = env_config["base_url"]
        self.api_key = env_config["api_key"]
        self.temperature = env_config["temperature"]
        self.top_p = env_config["top_p"]
        self.max_tokens = env_config["max_tokens"]
        self.supports_json_mode = env_config["supports_json_mode"]
        self._initialize_client()

    @property
    def is_initialized(self) -> bool:
        return self.client is not None

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> Optional[str]:
        """
        Send a chat completion request through a LangChain prompt/model/parser
        chain and return the raw text response.
        """

        if not self.is_initialized:
            logger.error("LLM client not initialized")
            return None

        try:
            prompt_messages = [
                (_normalize_prompt_role(msg["role"]), _escape_prompt_content(msg["content"]))
                for msg in messages
            ]
            prompt = ChatPromptTemplate.from_messages(prompt_messages)
            model = self._build_chat_model(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
            if model is None:
                return None

            chain = prompt | model | StrOutputParser()
            return chain.invoke({})
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            return None

    def get_info(self) -> Dict[str, Any]:
        """Get current LLM configuration info."""

        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "supports_json_mode": self.supports_json_mode,
            "is_initialized": self.is_initialized,
            "backend": "langchain_openai" if HAS_LANGCHAIN else "unavailable",
        }


_default_client: Optional[LLMClient] = None


def get_llm_client(force_reload: bool = False, **overrides) -> LLMClient:
    """
    Get the default client or create a configured client when overrides are
    provided.
    """

    global _default_client

    if overrides:
        return LLMClient(**overrides)

    if _default_client is None:
        _default_client = LLMClient()
    elif force_reload:
        _default_client.reload_config()

    return _default_client


def create_llm_client(**kwargs) -> LLMClient:
    """Create a new LLM client with custom configuration."""

    return LLMClient(**kwargs)
