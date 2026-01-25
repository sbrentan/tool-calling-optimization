"""
OpenAI-compatible API client for tool calling experiments.

This client works with:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Azure OpenAI
- Any OpenAI-compatible API (LocalAI, Ollama, etc.)
"""
import os
import time
import json
from typing import Any, Optional

from loguru import logger
import httpx

from .base import BaseLLMClient, CallResult
from .rate_limit_handler import handle_api_error_with_retry, UserAbortError
from .timeout_utils import get_default_client_timeouts, is_interrupted


class OpenAIClient(BaseLLMClient):
    """
    Client for OpenAI and OpenAI-compatible APIs.
    
    Supports GPT-4, GPT-3.5, and other OpenAI models.
    Can also work with Azure OpenAI and compatible APIs.
    """
    
    PROVIDER_NAME = "openai"
    
    # Available models on OpenAI
    AVAILABLE_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        base_url: Optional[str] = None
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use for generation
            temperature: Sampling temperature (0.0 = deterministic)
            base_url: Optional custom base URL for compatible APIs
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided and not found in environment")
        
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        
        # Import and initialize OpenAI client with HTTP timeouts
        try:
            from openai import OpenAI
            
            # Get timeout configuration
            connect_timeout, read_timeout = get_default_client_timeouts()
            http_client = httpx.Client(
                timeout=httpx.Timeout(connect_timeout, read=read_timeout, write=read_timeout, pool=connect_timeout)
            )
            
            client_kwargs = {"api_key": self.api_key, "http_client": http_client}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = OpenAI(**client_kwargs)
            self._http_client = http_client  # Keep reference for cleanup
        except ImportError:
            raise ImportError(
                "openai not installed. "
                "Install with: pip install openai"
            )
        
        logger.info(f"Initialized OpenAI client with model: {model}")
    
    def set_model(self, model: str) -> None:
        """Change the model being used."""
        if model not in self.AVAILABLE_MODELS:
            logger.warning(f"Model {model} not in known models list, using anyway")
        self.model = model
        logger.info(f"Switched to model: {model}")
    
    def set_api_key(self, api_key: str) -> None:
        """Update the API key and reinitialize the client."""
        self.api_key = api_key
        try:
            from openai import OpenAI
            
            # Get timeout configuration
            connect_timeout, read_timeout = get_default_client_timeouts()
            http_client = httpx.Client(
                timeout=httpx.Timeout(connect_timeout, read=read_timeout, write=read_timeout, pool=connect_timeout)
            )
            
            client_kwargs = {"api_key": self.api_key, "http_client": http_client}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = OpenAI(**client_kwargs)
            self._http_client = http_client
            logger.debug(f"Updated OpenAI client with new API key")
        except Exception as e:
            logger.error(f"Failed to update OpenAI client with new API key: {e}")
            raise
    
    def _convert_tools_to_openai_format(self, tools: list) -> list[dict]:
        """
        Convert Tool objects to OpenAI function format.
        """
        openai_tools = []
        
        for tool in tools:
            # Build properties and required list
            properties = {}
            required = []
            
            for param in tool.parameters:
                prop = {
                    "type": param.type if param.type in ["string", "integer", "number", "boolean", "array"] else "string",
                    "description": param.description
                }
                if param.enum:
                    prop["enum"] = param.enum
                    
                properties[param.name] = prop
                
                if param.required:
                    required.append(param.name)
            
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        
        return openai_tools
    
    def call_with_tools(
        self,
        prompt: str,
        tools: list,
        system_instruction: Optional[str] = None
    ) -> CallResult:
        """
        Send a prompt to OpenAI with available tools.
        
        Args:
            prompt: User prompt to send
            tools: List of Tool objects available for calling
            system_instruction: Optional system instruction
            
        Returns:
            CallResult with the tool call information
        """
        # Convert tools to OpenAI format (only once, outside retry loop)
        openai_tools = self._convert_tools_to_openai_format(tools)
        
        # Build messages (only once, outside retry loop)
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        # Retry loop for rate limit handling
        while True:
            start_time = time.time()
            
            try:
                # Make API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract token usage from response
                tokens_input = None
                tokens_output = None
                tokens_total = None
                if hasattr(response, 'usage') and response.usage:
                    tokens_input = getattr(response.usage, 'prompt_tokens', None)
                    tokens_output = getattr(response.usage, 'completion_tokens', None)
                    tokens_total = getattr(response.usage, 'total_tokens', None)
                
                # Extract function calls
                message = response.choices[0].message
                
                if message.tool_calls:
                    all_calls = []
                    for tc in message.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        except json.JSONDecodeError:
                            args = {}
                        
                        all_calls.append({
                            "name": tc.function.name,
                            "args": args
                        })
                    
                    return CallResult(
                        success=True,
                        called_tool=all_calls[0]["name"] if all_calls else None,
                        called_args=all_calls[0]["args"] if all_calls else None,
                        all_calls=all_calls,
                        latency_ms=latency_ms,
                        raw_response=response,
                        model=self.model,
                        provider=self.PROVIDER_NAME,
                        tokens_input=tokens_input,
                        tokens_output=tokens_output,
                        tokens_total=tokens_total
                    )
                else:
                    # Model didn't call any tool
                    return CallResult(
                        success=True,
                        called_tool=None,
                        latency_ms=latency_ms,
                        raw_response=response,
                        model=self.model,
                        provider=self.PROVIDER_NAME,
                        error="Model did not call any tool",
                        tokens_input=tokens_input,
                        tokens_output=tokens_output,
                        tokens_total=tokens_total
                    )
                    
            except UserAbortError:
                # User chose to abort - re-raise to stop the experiment
                raise
                    
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                logger.error(f"OpenAI API call failed: {e}")
                
                # Check if this is a rate limit error and use key rotation
                if handle_api_error_with_retry(
                    e, 
                    context="calling OpenAI API",
                    provider=self.PROVIDER_NAME,
                    on_key_rotated=self.set_api_key
                ):
                    # Retry with potentially rotated key
                    logger.info("Retrying OpenAI API call...")
                    continue
                
                return CallResult(
                    success=False,
                    error=str(e),
                    latency_ms=latency_ms,
                    model=self.model,
                    provider=self.PROVIDER_NAME
                )
