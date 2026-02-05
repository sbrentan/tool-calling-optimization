"""
Cerebras API client for tool calling experiments.

Cerebras uses an OpenAI-compatible API format.
Free tier offers 1M tokens/day - great for testing.
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


class CerebrasClient(BaseLLMClient):
    """
    Client for Cerebras Inference API.
    
    Cerebras offers extremely fast inference with free tier access.
    Uses OpenAI-compatible API format for tool calling.
    
    Get API key at: https://cloud.cerebras.ai/
    """
    
    PROVIDER_NAME = "cerebras"
    
    # Available models on Cerebras
    AVAILABLE_MODELS = [
        "llama-3.3-70b",
        "llama3.1-8b",
        "qwen-3-32b",
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b",
        temperature: float = 0.0
    ):
        """
        Initialize the Cerebras client.
        
        Args:
            api_key: Cerebras API key (or set CEREBRAS_API_KEY env var)
            model: Model to use for generation
            temperature: Sampling temperature (0.0 = deterministic)
        """
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY not provided and not found in environment")
        
        self.model = model
        self.temperature = temperature
        
        # Import and initialize Cerebras client with HTTP timeouts
        try:
            from cerebras.cloud.sdk import Cerebras
            
            # Get timeout configuration
            connect_timeout, read_timeout = get_default_client_timeouts()
            http_client = httpx.Client(
                timeout=httpx.Timeout(connect_timeout, read=read_timeout, write=read_timeout, pool=connect_timeout)
            )
            
            self.client = Cerebras(api_key=self.api_key, http_client=http_client)
            self._http_client = http_client  # Keep reference for cleanup
        except ImportError:
            raise ImportError(
                "cerebras-cloud-sdk not installed. "
                "Install with: pip install cerebras-cloud-sdk"
            )
        
        logger.info(f"Initialized Cerebras client with model: {model}")
    
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
            from cerebras.cloud.sdk import Cerebras
            
            # Get timeout configuration
            connect_timeout, read_timeout = get_default_client_timeouts()
            http_client = httpx.Client(
                timeout=httpx.Timeout(connect_timeout, read=read_timeout, write=read_timeout, pool=connect_timeout)
            )
            
            self.client = Cerebras(api_key=self.api_key, http_client=http_client)
            self._http_client = http_client
            logger.debug(f"Updated Cerebras client with new API key")
        except Exception as e:
            logger.error(f"Failed to update Cerebras client with new API key: {e}")
            raise
    
    def _convert_tools_to_cerebras_format(self, tools: list) -> list[dict]:
        """
        Convert Tool objects to Cerebras/OpenAI function format.
        
        Cerebras uses OpenAI-compatible tool format with strict mode.
        """
        cerebras_tools = []
        
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
            
            cerebras_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "strict": True,  # Required by Cerebras
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        
        return cerebras_tools
    
    def call_with_tools(
        self,
        prompt: str,
        tools: list,
        system_instruction: Optional[str] = None
    ) -> CallResult:
        """
        Send a prompt to Cerebras with available tools.
        
        Args:
            prompt: User prompt to send
            tools: List of Tool objects available for calling
            system_instruction: Optional system instruction
            
        Returns:
            CallResult with the tool call information
        """
        # Convert tools to Cerebras format (only once, outside retry loop)
        cerebras_tools = self._convert_tools_to_cerebras_format(tools)
        
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
                    tools=cerebras_tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                )
                
                latency_ms = (time.time() - start_time) * 1000

                logger.debug(f"Cerebras response: {response}")
                
                # Extract function calls
                message = response.choices[0].message

                logger.debug(f"Cerebras response message: {message}")
                
                # Extract token usage from response
                tokens_input = None
                tokens_output = None
                tokens_total = None
                if hasattr(response, 'usage') and response.usage:
                    tokens_input = getattr(response.usage, 'prompt_tokens', None)
                    tokens_output = getattr(response.usage, 'completion_tokens', None)
                    tokens_total = getattr(response.usage, 'total_tokens', None)
                
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
                logger.error(f"Cerebras API call failed: {e}")
                
                # Check if this is a rate limit error and use key rotation
                if handle_api_error_with_retry(
                    e, 
                    context="calling Cerebras API",
                    provider=self.PROVIDER_NAME,
                    on_key_rotated=self.set_api_key
                ):
                    # Retry with potentially rotated key
                    logger.info("Retrying Cerebras API call...")
                    continue
                
                return CallResult(
                    success=False,
                    error=str(e),
                    latency_ms=latency_ms,
                    model=self.model,
                    provider=self.PROVIDER_NAME
                )
