"""
LLM Client Module

Provides a unified LangChain-based interface for multiple LLM routes:
- openai: OpenAI API route (model configurable via env/CLI/API)
- together: Together AI serverless route
"""

import logging
import os
import time
from dataclasses import dataclass, field
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


@dataclass
class LLMCallRecord:
    """Single LLM API call measurement."""

    tag: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    model: str
    provider: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class LLMMetricsSummary:
    """Aggregated metrics over multiple LLM calls."""

    call_count: int
    success_count: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    mean_cost_usd_per_call: float
    by_tag: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "".join(parts)
    return str(content or "")


def _extract_token_usage(message: Any) -> Dict[str, int]:
    usage_meta = getattr(message, "usage_metadata", None) or {}
    if usage_meta:
        input_tokens = int(usage_meta.get("input_tokens") or 0)
        output_tokens = int(usage_meta.get("output_tokens") or 0)
        total_tokens = int(usage_meta.get("total_tokens") or (input_tokens + output_tokens))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    response_meta = getattr(message, "response_metadata", None) or {}
    token_usage = response_meta.get("token_usage") or response_meta.get("usage") or {}
    input_tokens = int(
        token_usage.get("input_tokens")
        or token_usage.get("prompt_tokens")
        or 0
    )
    output_tokens = int(
        token_usage.get("output_tokens")
        or token_usage.get("completion_tokens")
        or 0
    )
    total_tokens = int(token_usage.get("total_tokens") or (input_tokens + output_tokens))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def get_default_price_per_1m(provider: str, model: str) -> tuple[float, float]:
    """
    Return (input_usd_per_1m, output_usd_per_1m) for cost estimation.

    Override with LLM_INPUT_USD_PER_1M / LLM_OUTPUT_USD_PER_1M in .env.
    """

    input_override = os.getenv("LLM_INPUT_USD_PER_1M")
    output_override = os.getenv("LLM_OUTPUT_USD_PER_1M")
    if input_override and output_override:
        try:
            return float(input_override), float(output_override)
        except ValueError:
            pass

    model_lower = (model or "").lower()
    if normalize_provider(provider) == LLMProvider.OPENAI:
        if model_lower.startswith("gpt-5-mini"):
            return 0.25, 2.00
        if model_lower.startswith("gpt-5"):
            return 1.25, 10.00
        if model_lower.startswith("gpt-4o-mini"):
            return 0.15, 0.60
        if model_lower.startswith("gpt-4o"):
            return 2.50, 10.00
    return 0.0, 0.0


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    *,
    provider: str,
    model: str,
    input_price_per_1m: Optional[float] = None,
    output_price_per_1m: Optional[float] = None,
) -> float:
    """Estimate USD cost from token counts and per-1M prices."""

    if input_price_per_1m is None or output_price_per_1m is None:
        default_in, default_out = get_default_price_per_1m(provider, model)
        input_price_per_1m = default_in if input_price_per_1m is None else input_price_per_1m
        output_price_per_1m = default_out if output_price_per_1m is None else output_price_per_1m

    return (
        (input_tokens / 1_000_000.0) * input_price_per_1m
        + (output_tokens / 1_000_000.0) * output_price_per_1m
    )


def summarize_llm_metrics(
    records: List[LLMCallRecord],
    *,
    input_price_per_1m: Optional[float] = None,
    output_price_per_1m: Optional[float] = None,
) -> LLMMetricsSummary:
    """Aggregate call records into paper-friendly summary statistics."""

    if not records:
        return LLMMetricsSummary(
            call_count=0,
            success_count=0,
            mean_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tokens=0,
            total_cost_usd=0.0,
            mean_cost_usd_per_call=0.0,
            by_tag={},
        )

    latencies = [r.latency_ms for r in records if r.success]
    success_records = [r for r in records if r.success]
    total_cost = sum(r.cost_usd for r in success_records)

    by_tag: Dict[str, Dict[str, Any]] = {}
    for record in records:
        bucket = by_tag.setdefault(
            record.tag,
            {
                "call_count": 0,
                "success_count": 0,
                "mean_latency_ms": 0.0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "_latencies": [],
            },
        )
        bucket["call_count"] += 1
        if record.success:
            bucket["success_count"] += 1
            bucket["total_tokens"] += record.total_tokens
            bucket["total_cost_usd"] += record.cost_usd
            bucket["_latencies"].append(record.latency_ms)

    for bucket in by_tag.values():
        tag_latencies = bucket.pop("_latencies")
        bucket["mean_latency_ms"] = (
            sum(tag_latencies) / len(tag_latencies) if tag_latencies else 0.0
        )

    return LLMMetricsSummary(
        call_count=len(records),
        success_count=len(success_records),
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        total_input_tokens=sum(r.input_tokens for r in success_records),
        total_output_tokens=sum(r.output_tokens for r in success_records),
        total_tokens=sum(r.total_tokens for r in success_records),
        total_cost_usd=total_cost,
        mean_cost_usd_per_call=total_cost / len(success_records) if success_records else 0.0,
        by_tag=by_tag,
    )


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
        enable_metrics: bool = False,
        input_price_per_1m: Optional[float] = None,
        output_price_per_1m: Optional[float] = None,
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
        self.enable_metrics = enable_metrics
        self.input_price_per_1m = input_price_per_1m
        self.output_price_per_1m = output_price_per_1m
        self.metrics_records: List[LLMCallRecord] = []

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

    def reset_metrics(self) -> None:
        """Clear accumulated call metrics."""

        self.metrics_records.clear()

    def get_metrics_records(self) -> List[LLMCallRecord]:
        """Return a shallow copy of accumulated metrics records."""

        return list(self.metrics_records)

    def summarize_metrics(self) -> LLMMetricsSummary:
        """Summarize metrics collected when enable_metrics=True."""

        return summarize_llm_metrics(
            self.metrics_records,
            input_price_per_1m=self.input_price_per_1m,
            output_price_per_1m=self.output_price_per_1m,
        )

    def _record_call_metric(
        self,
        *,
        tag: str,
        latency_ms: float,
        usage: Dict[str, int],
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        if not self.enable_metrics:
            return

        cost_usd = estimate_cost_usd(
            usage["input_tokens"],
            usage["output_tokens"],
            provider=self.provider,
            model=self.model,
            input_price_per_1m=self.input_price_per_1m,
            output_price_per_1m=self.output_price_per_1m,
        )
        self.metrics_records.append(
            LLMCallRecord(
                tag=tag,
                latency_ms=latency_ms,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
                cost_usd=cost_usd,
                model=self.model,
                provider=self.provider,
                success=success,
                error=error,
            )
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        metrics_tag: str = "chat_completion",
    ) -> Optional[str]:
        """
        Send a chat completion request and return the raw text response.

        When enable_metrics=True, each call records latency and token usage
        (from provider response metadata) for cost/latency benchmarking.
        """

        if not self.is_initialized:
            logger.error("LLM client not initialized")
            return None

        started = time.perf_counter()
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

            if self.enable_metrics:
                ai_message = (prompt | model).invoke({})
                latency_ms = (time.perf_counter() - started) * 1000.0
                usage = _extract_token_usage(ai_message)
                self._record_call_metric(
                    tag=metrics_tag,
                    latency_ms=latency_ms,
                    usage=usage,
                    success=True,
                )
                return _extract_message_text(ai_message)

            chain = prompt | model | StrOutputParser()
            return chain.invoke({})
        except Exception as e:
            latency_ms = (time.perf_counter() - started) * 1000.0
            if self.enable_metrics:
                self._record_call_metric(
                    tag=metrics_tag,
                    latency_ms=latency_ms,
                    usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    success=False,
                    error=str(e),
                )
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
