"""
Base client interface for LLM providers.

All provider-specific clients inherit from this base class
to ensure consistent API for tool calling experiments.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class CallResult:
    """Result of a tool calling API request."""
    success: bool
    called_tool: Optional[str] = None
    called_args: Optional[dict[str, Any]] = None
    all_calls: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None
    raw_response: Optional[Any] = None
    model: str = ""
    provider: str = ""
    
    # Token usage tracking (Phase 3)
    tokens_input: Optional[int] = None   # Input/prompt tokens used
    tokens_output: Optional[int] = None  # Output/completion tokens used
    tokens_total: Optional[int] = None   # Total tokens (input + output)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "called_tool": self.called_tool,
            "called_args": self.called_args,
            "all_calls": self.all_calls,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "model": self.model,
            "provider": self.provider,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_total": self.tokens_total,
        }


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM provider clients.
    
    All provider implementations must implement these methods
    to ensure consistent behavior in experiments.
    """
    
    # Provider name for identification
    PROVIDER_NAME: str = "base"
    
    # Available models for this provider
    AVAILABLE_MODELS: list[str] = []
    
    @abstractmethod
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "",
        temperature: float = 0.0
    ):
        """
        Initialize the client.
        
        Args:
            api_key: API key for the provider
            model: Model to use
            temperature: Sampling temperature
        """
        pass
    
    @abstractmethod
    def set_model(self, model: str) -> None:
        """Change the model being used."""
        pass
    
    @abstractmethod
    def call_with_tools(
        self,
        prompt: str,
        tools: list,
        system_instruction: Optional[str] = None
    ) -> CallResult:
        """
        Send a prompt with available tools and get the response.
        
        Args:
            prompt: User prompt to send
            tools: List of Tool objects available for calling
            system_instruction: Optional system instruction
            
        Returns:
            CallResult with the tool call information
        """
        pass
    
    def batch_call_with_tools(
        self,
        prompts: list[str],
        tools: list,
        system_instruction: Optional[str] = None
    ) -> list[CallResult]:
        """
        Send multiple prompts with the same tools.
        
        Default implementation calls sequentially.
        Override for batch API support.
        """
        from loguru import logger
        results = []
        for i, prompt in enumerate(prompts):
            logger.debug(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.call_with_tools(prompt, tools, system_instruction)
            results.append(result)
        return results
    
    @classmethod
    def get_provider_name(cls) -> str:
        """Get the provider name."""
        return cls.PROVIDER_NAME
    
    @classmethod
    def get_available_models(cls) -> list[str]:
        """Get list of available models for this provider."""
        return cls.AVAILABLE_MODELS
