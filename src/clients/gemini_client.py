"""
Gemini API client wrapper for tool calling experiments.

This client handles communication with the Gemini API, including:
- Sending prompts with tool declarations
- Extracting function call responses
- Managing API configuration
"""
import os
import time
from typing import Any, Optional

from loguru import logger

from .base import BaseLLMClient, CallResult
from .rate_limit_handler import handle_api_error_with_retry, UserAbortError
from src.tools.base import Tool
from src.adapters.gemini_adapter import GeminiAdapter


class GeminiClient(BaseLLMClient):
    """
    Client for interacting with the Gemini API for tool calling experiments.
    
    Supports:
    - Multiple Gemini models (flash, pro, etc.)
    - Tool/function calling
    - Response parsing and extraction
    """
    
    PROVIDER_NAME = "gemini"
    
    # Available models for testing
    AVAILABLE_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.0
    ):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            model: Model to use for generation
            temperature: Sampling temperature (0.0 = deterministic)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not provided and not found in environment")
        
        self.model = model
        self.temperature = temperature
        
        # Import and initialize client
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "google-genai not installed. "
                "Install with: pip install google-genai"
            )
        
        logger.info(f"Initialized Gemini client with model: {model}")
    
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
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            logger.debug(f"Updated Gemini client with new API key")
        except Exception as e:
            logger.error(f"Failed to update Gemini client with new API key: {e}")
            raise
    
    def call_with_tools(
        self,
        prompt: str,
        tools: list[Tool],
        system_instruction: Optional[str] = None
    ) -> CallResult:
        """
        Send a prompt to Gemini with available tools and get the response.
        
        Args:
            prompt: User prompt to send
            tools: List of Tool objects available for calling
            system_instruction: Optional system instruction
            
        Returns:
            CallResult with the tool call information
        """
        from google.genai import types
        
        # Convert tools to Gemini format (only once, outside retry loop)
        gemini_tools = GeminiAdapter.tools_to_gemini_tools(tools)
        
        # Build config (only once, outside retry loop)
        config = types.GenerateContentConfig(
            tools=[gemini_tools],
            temperature=self.temperature,
        )
        
        if system_instruction:
            config.system_instruction = system_instruction
        
        # Retry loop for rate limit handling
        while True:
            start_time = time.time()
            
            try:
                # Make API call
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract token usage from response
                tokens_input = None
                tokens_output = None
                tokens_total = None
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    tokens_input = getattr(response.usage_metadata, 'prompt_token_count', None)
                    tokens_output = getattr(response.usage_metadata, 'candidates_token_count', None)
                    tokens_total = getattr(response.usage_metadata, 'total_token_count', None)
                
                # Extract function call
                function_call = GeminiAdapter.extract_function_call(response)
                all_calls = GeminiAdapter.extract_all_function_calls(response)
                
                if function_call:
                    return CallResult(
                        success=True,
                        called_tool=function_call["name"],
                        called_args=function_call["args"],
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
                logger.error(f"Gemini API call failed: {e}")
                
                # Check if this is a rate limit error and use key rotation
                if handle_api_error_with_retry(
                    e, 
                    context="calling Gemini API",
                    provider=self.PROVIDER_NAME,
                    on_key_rotated=self.set_api_key
                ):
                    # Retry with potentially rotated key
                    logger.info("Retrying Gemini API call...")
                    continue
                
                return CallResult(
                    success=False,
                    error=str(e),
                    latency_ms=latency_ms,
                    model=self.model,
                    provider=self.PROVIDER_NAME
                )
