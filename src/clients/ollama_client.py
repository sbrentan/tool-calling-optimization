"""
Ollama client for local LLM tool calling experiments.

This client works with locally running Ollama models,
providing a way to test tool calling without cloud API costs.
"""
import os
import time
import json
from typing import Any, Optional

from loguru import logger

from .base import BaseLLMClient, CallResult
from .rate_limit_handler import handle_api_error_with_retry, UserAbortError
from .timeout_utils import get_default_client_timeouts, is_interrupted


class OllamaClient(BaseLLMClient):
    """
    Client for Ollama local LLM server.
    
    Requires Ollama to be running locally (default: http://localhost:11434).
    Supports any model available in your local Ollama installation.
    """
    
    PROVIDER_NAME = "ollama"
    
    # Common Ollama models - this is not exhaustive, any installed model works
    AVAILABLE_MODELS = [
        "gpt-oss:20b",
        "llama3:8b",
        "llama3:70b",
        "mistral:7b",
        "mixtral:8x7b",
        "codellama:7b",
        "phi3:mini",
        "qwen2:7b",
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,  # Not used, kept for interface compatibility
        model: str = "gpt-oss:20b",
        temperature: float = 0.0,
        base_url: Optional[str] = None
    ):
        """
        Initialize the Ollama client.
        
        Args:
            api_key: Not used for Ollama (local), kept for interface compatibility
            model: Model to use (must be installed via `ollama pull <model>`)
            temperature: Sampling temperature (0.0 = deterministic)
            base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.model = model
        self.temperature = temperature
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Import and initialize Ollama client via langchain with timeout
        try:
            from langchain_ollama import ChatOllama
            
            # Get timeout configuration
            connect_timeout, read_timeout = get_default_client_timeouts()
            
            self.client = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                # Langchain's ChatOllama supports timeout parameter
                timeout=read_timeout,
            )
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed. "
                "Install with: pip install langchain-ollama"
            )
        
        logger.info(f"Initialized Ollama client with model: {model} at {self.base_url}")
    
    def set_model(self, model: str) -> None:
        """Change the model being used."""
        self.model = model
        # Reinitialize client with new model
        try:
            from langchain_ollama import ChatOllama
            
            # Get timeout configuration
            connect_timeout, read_timeout = get_default_client_timeouts()
            
            self.client = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                timeout=read_timeout,
            )
            logger.info(f"Switched to model: {model}")
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            raise
    
    def set_api_key(self, api_key: str) -> None:
        """
        Not used for Ollama (local server), kept for interface compatibility.
        """
        logger.debug("set_api_key called on Ollama client (no-op for local server)")
    
    def _convert_tools_to_ollama_format(self, tools: list) -> list[dict]:
        """
        Convert Tool objects to Ollama/OpenAI-compatible function format.
        
        Ollama uses the same tool format as OpenAI.
        """
        ollama_tools = []
        
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
            
            ollama_tools.append({
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
        
        return ollama_tools
    
    def call_with_tools(
        self,
        prompt: str,
        tools: list,
        system_instruction: Optional[str] = None
    ) -> CallResult:
        """
        Send a prompt to Ollama with available tools.
        
        Args:
            prompt: User prompt to send
            tools: List of Tool objects available for calling
            system_instruction: Optional system instruction
            
        Returns:
            CallResult with the tool call information
        """
        # Convert tools to Ollama format (only once, outside retry loop)
        ollama_tools = self._convert_tools_to_ollama_format(tools)
        
        # Build messages
        from langchain_core.messages import HumanMessage, SystemMessage
        
        messages = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=prompt))
        
        # Bind tools to the client
        client_with_tools = self.client.bind_tools(ollama_tools)
        
        # Retry loop for error handling
        while True:
            start_time = time.time()
            
            try:
                # Make API call
                response = client_with_tools.invoke(messages)
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract token usage from response metadata if available
                tokens_input = None
                tokens_output = None
                tokens_total = None
                
                if hasattr(response, 'response_metadata'):
                    metadata = response.response_metadata
                    # Ollama may provide token usage in different formats
                    if 'prompt_eval_count' in metadata:
                        tokens_input = metadata.get('prompt_eval_count')
                    if 'eval_count' in metadata:
                        tokens_output = metadata.get('eval_count')
                    if tokens_input and tokens_output:
                        tokens_total = tokens_input + tokens_output
                
                # Extract tool calls from response
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    all_calls = []
                    for tc in response.tool_calls:
                        # LangChain tool_calls are dicts with 'name' and 'args'
                        tool_name = tc.get('name', '') if isinstance(tc, dict) else getattr(tc, 'name', '')
                        tool_args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                        
                        all_calls.append({
                            "name": tool_name,
                            "args": tool_args
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
                logger.error(f"Ollama API call failed: {e}")
                
                # Check if this is a connection error or other recoverable error
                error_str = str(e).lower()
                if "connection" in error_str or "timeout" in error_str:
                    # Check if Ollama server is running
                    logger.warning(
                        f"Connection issue with Ollama server at {self.base_url}. "
                        "Make sure Ollama is running with: ollama serve"
                    )
                
                # Use the standard retry handler
                if handle_api_error_with_retry(
                    e, 
                    context="calling Ollama API",
                    provider=self.PROVIDER_NAME,
                    on_key_rotated=self.set_api_key
                ):
                    # Retry
                    logger.info("Retrying Ollama API call...")
                    continue
                
                return CallResult(
                    success=False,
                    error=str(e),
                    latency_ms=latency_ms,
                    model=self.model,
                    provider=self.PROVIDER_NAME
                )
