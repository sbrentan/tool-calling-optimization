"""
Client factory for creating LLM clients.

Provides a unified interface to create clients for different providers
based on model name or explicit provider selection.
"""
from typing import Optional

from loguru import logger

from .base import BaseLLMClient
from .gemini_client import GeminiClient
from .cerebras_client import CerebrasClient
from .openai_client import OpenAIClient
from .ollama_client import OllamaClient


# Registry of available providers and their clients
PROVIDER_REGISTRY: dict[str, type[BaseLLMClient]] = {
    "gemini": GeminiClient,
    "cerebras": CerebrasClient,
    "openai": OpenAIClient,
    "ollama": OllamaClient,
}

# Model name to provider mapping for auto-detection
MODEL_TO_PROVIDER: dict[str, str] = {
    # Gemini models
    "gemini-2.0-flash": "gemini",
    "gemini-2.0-flash-lite": "gemini",
    "gemini-1.5-flash": "gemini",
    "gemini-1.5-flash-8b": "gemini",
    "gemini-1.5-pro": "gemini",
    
    # Cerebras models
    "llama-3.3-70b": "cerebras",
    "llama3.1-8b": "cerebras",
    "qwen-3-32b": "cerebras",
    
    # OpenAI models
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4-turbo": "openai",
    "gpt-4": "openai",
    "gpt-3.5-turbo": "openai",
    
    # Ollama local models
    "gpt-oss:20b": "ollama",
    "llama3:8b": "ollama",
    "llama3:70b": "ollama",
    "mistral:7b": "ollama",
    "mixtral:8x7b": "ollama",
    "codellama:7b": "ollama",
    "phi3:mini": "ollama",
    "qwen2:7b": "ollama",
}


def get_available_providers() -> list[str]:
    """Get list of available provider names."""
    return list(PROVIDER_REGISTRY.keys())


def get_available_models(provider: Optional[str] = None) -> dict[str, list[str]]:
    """
    Get available models, optionally filtered by provider.
    
    Args:
        provider: Optional provider name to filter by
        
    Returns:
        Dict mapping provider names to their available models
    """
    if provider:
        if provider not in PROVIDER_REGISTRY:
            raise ValueError(f"Unknown provider: {provider}")
        client_class = PROVIDER_REGISTRY[provider]
        return {provider: client_class.AVAILABLE_MODELS}
    
    return {
        name: client_class.AVAILABLE_MODELS
        for name, client_class in PROVIDER_REGISTRY.items()
    }


def detect_provider_from_model(model: str) -> Optional[str]:
    """
    Detect the provider from a model name.
    
    Args:
        model: Model name
        
    Returns:
        Provider name if detected, None otherwise
    """
    # Direct lookup
    if model in MODEL_TO_PROVIDER:
        return MODEL_TO_PROVIDER[model]
    
    # Pattern matching
    model_lower = model.lower()
    
    if "gemini" in model_lower:
        return "gemini"
    elif "gpt-4" in model_lower or "gpt-3" in model_lower:
        # Explicit OpenAI GPT models
        return "openai"
    elif ":" in model_lower:
        # Ollama models typically have format "model:tag" (e.g., llama3:8b, gpt-oss:20b)
        return "ollama"
    elif "llama" in model_lower or "qwen" in model_lower:
        # Could be Cerebras or other providers
        return "cerebras"
    
    return None


def create_client(
    model: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
) -> BaseLLMClient:
    """
    Create an LLM client for the specified model/provider.
    
    Args:
        model: Model name to use
        provider: Provider name (auto-detected from model if not specified)
        api_key: API key (uses environment variable if not specified)
        temperature: Sampling temperature
        **kwargs: Additional provider-specific arguments
        
    Returns:
        Initialized LLM client
        
    Raises:
        ValueError: If provider cannot be determined or is unknown
    """
    # Detect provider if not specified
    if provider is None:
        provider = detect_provider_from_model(model)
        if provider is None:
            raise ValueError(
                f"Cannot auto-detect provider for model '{model}'. "
                f"Please specify provider explicitly. "
                f"Available providers: {get_available_providers()}"
            )
        logger.info(f"Auto-detected provider '{provider}' for model '{model}'")
    
    # Validate provider
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available providers: {get_available_providers()}"
        )
    
    # Create client
    client_class = PROVIDER_REGISTRY[provider]
    
    try:
        client = client_class(
            api_key=api_key,
            model=model,
            temperature=temperature,
            **kwargs
        )
        return client
    except ImportError as e:
        raise ImportError(
            f"Failed to create {provider} client. "
            f"Make sure the required package is installed. "
            f"Error: {e}"
        )


def list_all_models() -> None:
    """Print all available models organized by provider."""
    print("\nAvailable Models by Provider:")
    print("=" * 50)
    
    for provider, models in get_available_models().items():
        print(f"\n{provider.upper()}:")
        for model in models:
            print(f"  - {model}")
    
    print("\n" + "=" * 50)
