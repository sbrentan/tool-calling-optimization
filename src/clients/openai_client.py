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

from .base import BaseLLMClient, CallResult


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
        
        # Import and initialize OpenAI client
        try:
            from openai import OpenAI
            client_kwargs = {"api_key": self.api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = OpenAI(**client_kwargs)
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
        start_time = time.time()
        
        try:
            # Convert tools to OpenAI format
            openai_tools = self._convert_tools_to_openai_format(tools)
            
            # Build messages
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                temperature=self.temperature,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
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
                    provider=self.PROVIDER_NAME
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
                    error="Model did not call any tool"
                )
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"OpenAI API call failed: {e}")
            return CallResult(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
                model=self.model,
                provider=self.PROVIDER_NAME
            )
