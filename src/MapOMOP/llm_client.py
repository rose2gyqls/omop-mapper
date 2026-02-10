"""
LLM Client Module

Provides a unified interface for multiple LLM providers:
- OpenAI (gpt-4o-mini, etc.)
- SNUH Hari (snuh/hari-q3-14b via vLLM)
- Google Gemma (google/gemma-3-12b-it via vLLM)

Configuration via environment variables for container deployment flexibility.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Reload environment variables at import time
# This allows runtime configuration changes in containerized deployments
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Optional dependency
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("OpenAI library not installed. LLM features unavailable.")


class LLMProvider:
    """Available LLM providers."""
    OPENAI = "openai"
    HARI = "hari"      # snuh/hari-q3-14b
    GEMMA = "gemma"    # google/gemma-3-12b-it


# Default configurations for each provider
DEFAULT_CONFIGS = {
    LLMProvider.OPENAI: {
        "model": "gpt-4o-mini",
        "base_url": None,  # Use default OpenAI endpoint
        "supports_json_mode": True,
    },
    LLMProvider.HARI: {
        "model": "snuh/hari-q3-14b",
        "base_url": "http://localhost:8000/v1",  # Default vLLM endpoint
        "supports_json_mode": False,  # vLLM may not support json_object mode
    },
    LLMProvider.GEMMA: {
        "model": "google/gemma-3-12b-it",
        "base_url": "http://localhost:8001/v1",  # Default vLLM endpoint
        "supports_json_mode": False,
    },
}


def get_env_config() -> Dict[str, Any]:
    """
    Get LLM configuration from environment variables.
    
    Reloads .env file to support runtime configuration changes.
    
    Environment variables:
        LLM_PROVIDER: Provider name (openai, hari, gemma)
        LLM_MODEL: Model name (overrides default)
        LLM_BASE_URL: API base URL (for vLLM servers)
        LLM_API_KEY: API key (for OpenAI or secured vLLM)
        LLM_TEMPERATURE: Temperature (0.0-2.0)
        LLM_TOP_P: Top-p / nucleus sampling (0.0-1.0)
        
    Returns:
        Configuration dictionary
    """
    # Reload environment to get latest values
    load_dotenv(override=True)
    
    provider = os.getenv("LLM_PROVIDER", LLMProvider.OPENAI).lower()
    
    # Get default config for provider
    default = DEFAULT_CONFIGS.get(provider, DEFAULT_CONFIGS[LLMProvider.OPENAI])
    
    # Override with environment variables
    config = {
        "provider": provider,
        "model": os.getenv("LLM_MODEL", default["model"]),
        "base_url": os.getenv("LLM_BASE_URL", default["base_url"]),
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
        "top_p": float(os.getenv("LLM_TOP_P", "1.0")),
        "supports_json_mode": default["supports_json_mode"],
    }
    
    return config


class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    
    Uses OpenAI-compatible API for all providers (OpenAI, vLLM).
    Configuration can be loaded from environment or passed directly.
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        use_env_config: bool = True
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider (openai, hari, gemma)
            model: Model name
            base_url: API base URL (for vLLM)
            api_key: API key
            temperature: Temperature (0.0-2.0)
            top_p: Top-p (0.0-1.0)
            use_env_config: Load configuration from environment variables
        """
        if not HAS_OPENAI:
            logger.error("OpenAI library not installed")
            self.client = None
            return
        
        # Load config from environment if enabled
        if use_env_config:
            env_config = get_env_config()
            self.provider = provider or env_config["provider"]
            self.model = model or env_config["model"]
            self.base_url = base_url or env_config["base_url"]
            self.api_key = api_key or env_config["api_key"]
            self.temperature = temperature if temperature is not None else env_config["temperature"]
            self.top_p = top_p if top_p is not None else env_config["top_p"]
            self.supports_json_mode = env_config["supports_json_mode"]
        else:
            self.provider = provider or LLMProvider.OPENAI
            self.model = model or DEFAULT_CONFIGS[self.provider]["model"]
            self.base_url = base_url or DEFAULT_CONFIGS[self.provider]["base_url"]
            self.api_key = api_key
            self.temperature = temperature or 0.3
            self.top_p = top_p or 1.0
            self.supports_json_mode = DEFAULT_CONFIGS.get(
                self.provider, DEFAULT_CONFIGS[LLMProvider.OPENAI]
            )["supports_json_mode"]
        
        # Initialize OpenAI-compatible client
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI-compatible client."""
        try:
            client_kwargs = {}
            
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            elif self.provider == LLMProvider.OPENAI:
                logger.error("API key required for OpenAI provider")
                return
            else:
                # vLLM servers may not require API key
                client_kwargs["api_key"] = "dummy"  # OpenAI client requires some value
            
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            self.client = OpenAI(**client_kwargs)
            
            logger.info(
                f"LLM Client initialized: provider={self.provider}, "
                f"model={self.model}, base_url={self.base_url or 'default'}"
            )
            
        except Exception as e:
            logger.error(f"LLM Client initialization failed: {e}")
            self.client = None
    
    def reload_config(self):
        """
        Reload configuration from environment variables.
        
        Useful for runtime configuration changes in containerized deployments.
        """
        env_config = get_env_config()
        
        old_provider = self.provider
        old_model = self.model
        old_base_url = self.base_url
        
        self.provider = env_config["provider"]
        self.model = env_config["model"]
        self.base_url = env_config["base_url"]
        self.api_key = env_config["api_key"]
        self.temperature = env_config["temperature"]
        self.top_p = env_config["top_p"]
        self.supports_json_mode = env_config["supports_json_mode"]
        
        # Reinitialize client if configuration changed
        if (self.provider != old_provider or 
            self.model != old_model or 
            self.base_url != old_base_url):
            logger.info("LLM configuration changed, reinitializing client...")
            self._initialize_client()
    
    @property
    def is_initialized(self) -> bool:
        """Check if client is properly initialized."""
        return self.client is not None
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: int = 2048,
        json_mode: bool = False
    ) -> Optional[str]:
        """
        Send chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            top_p: Override default top_p
            max_tokens: Maximum tokens in response
            json_mode: Request JSON formatted response
            
        Returns:
            Response content string or None on error
        """
        if not self.client:
            logger.error("LLM client not initialized")
            return None
        
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "top_p": top_p if top_p is not None else self.top_p,
                "max_tokens": max_tokens,
            }
            
            # Only add response_format if provider supports it
            if json_mode and self.supports_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get current LLM configuration info."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "supports_json_mode": self.supports_json_mode,
            "is_initialized": self.is_initialized,
        }


# Singleton instance for shared use
_default_client: Optional[LLMClient] = None


def get_llm_client(force_reload: bool = False) -> LLMClient:
    """
    Get the default LLM client instance.
    
    Creates a singleton client or reloads configuration if requested.
    
    Args:
        force_reload: Force reload configuration from environment
        
    Returns:
        LLMClient instance
    """
    global _default_client
    
    if _default_client is None:
        _default_client = LLMClient()
    elif force_reload:
        _default_client.reload_config()
    
    return _default_client


def create_llm_client(**kwargs) -> LLMClient:
    """
    Create a new LLM client with custom configuration.
    
    Args:
        **kwargs: Arguments passed to LLMClient constructor
        
    Returns:
        New LLMClient instance
    """
    return LLMClient(**kwargs)
