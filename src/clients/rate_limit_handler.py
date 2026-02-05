"""
Rate limit handling utilities for LLM API clients.

Provides interactive retry mechanism when rate limits (429) are hit,
with automatic API key rotation before prompting the user.

Key rotation flow:
1. Rate limit error detected
2. Try rotating to next available API key
3. If rotation succeeds, retry with new key
4. If all keys exhausted, prompt user for manual retry
5. If user chooses retry, reset rotation and try all keys again
"""
import sys
from typing import Optional, Callable
from loguru import logger


class RateLimitError(Exception):
    """Exception raised when a rate limit (429) error is detected."""
    def __init__(self, message: str, original_exception: Exception):
        super().__init__(message)
        self.original_exception = original_exception


class UserAbortError(Exception):
    """Exception raised when user chooses to abort after rate limit."""
    pass


class KeyRotationResult:
    """Result of a key rotation attempt."""
    def __init__(self, rotated: bool, new_key: Optional[str] = None, all_exhausted: bool = False):
        self.rotated = rotated
        self.new_key = new_key
        self.all_exhausted = all_exhausted


def is_rate_limit_error(exception: Exception) -> bool:
    """
    Check if an exception is a rate limit (429) error.
    
    Works with various API client libraries that may format errors differently.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if this is a rate limit error, False otherwise
    """
    error_str = str(exception).lower()
    
    # Check for common rate limit indicators
    rate_limit_indicators = [
        "429",
        "rate limit",
        "rate_limit",
        "too many requests",
        "too_many_requests",
        "quota exceeded",
        "quota_exceeded",
        "high traffic",
        "queue_exceeded",
    ]
    
    return any(indicator in error_str for indicator in rate_limit_indicators)


def prompt_user_for_retry(error: Exception, context: str = "", provider: str = None, keys_exhausted: bool = False) -> bool:
    """
    Prompt the user to decide whether to retry after a rate limit error.
    
    Args:
        error: The rate limit exception
        context: Optional context about what was being attempted
        provider: Optional provider name for display
        keys_exhausted: Whether all API keys have been exhausted
        
    Returns:
        True if user wants to retry, False otherwise
        
    Raises:
        UserAbortError: If user chooses to abort the entire run
    """
    context_msg = f" while {context}" if context else ""
    provider_msg = f" ({provider})" if provider else ""
    
    print("\n" + "=" * 60)
    print("⚠️  RATE LIMIT ERROR DETECTED" + provider_msg)
    print("=" * 60)
    print(f"Error{context_msg}:")
    print(f"  {error}")
    print()
    
    if keys_exhausted:
        print("⚠️  All available API keys have been rate limited!")
        print()
    
    print("Options:")
    print("  [R] Retry the request (wait a bit then press R)")
    print("  [A] Abort the entire experiment run")
    print("=" * 60)
    
    while True:
        try:
            choice = input("Your choice (R/A): ").strip().upper()
            
            if choice == 'R':
                logger.info("User chose to retry the request")
                return True
            elif choice == 'A':
                logger.info("User chose to abort the experiment")
                raise UserAbortError("User aborted after rate limit error")
            else:
                print("Invalid choice. Please enter 'R' to retry or 'A' to abort.")
        except EOFError:
            # Non-interactive mode, abort
            logger.warning("Non-interactive mode detected, aborting")
            raise UserAbortError("Non-interactive mode, cannot prompt for retry")
        except KeyboardInterrupt:
            print("\n")
            logger.info("User interrupted with Ctrl+C, aborting")
            raise UserAbortError("User interrupted with Ctrl+C")


def handle_rate_limit_with_rotation(
    error: Exception,
    provider: str,
    context: str = "",
    interactive: bool = True,
    on_key_rotated: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Handle a rate limit error with automatic API key rotation.
    
    This is the new preferred method for handling rate limits. It will:
    1. Try to rotate to the next available API key
    2. If rotation succeeds, return True to retry with the new key
    3. If all keys are exhausted, prompt the user
    4. If user chooses to retry, reset rotation and try again
    
    Args:
        error: The exception that occurred
        provider: Provider name (cerebras, openai, gemini)
        context: Optional context about what was being attempted
        interactive: Whether to prompt user when keys exhausted
        on_key_rotated: Optional callback when key is rotated (receives new key)
        
    Returns:
        True if should retry, False if should not retry
        
    Raises:
        UserAbortError: If user chooses to abort
    """
    if not is_rate_limit_error(error):
        return False
    
    # Import here to avoid circular imports
    from .api_key_manager import get_api_key_manager
    
    manager = get_api_key_manager()
    pool = manager.get_pool(provider)
    
    if not pool or pool.num_keys <= 1:
        # No key rotation available, fall back to user prompt
        logger.warning(f"No additional API keys available for {provider}, prompting user")
        if interactive:
            return prompt_user_for_retry(error, context, provider, keys_exhausted=True)
        else:
            logger.warning(f"Rate limit hit in non-interactive mode: {error}")
            return False
    
    # Try to rotate to next key
    if manager.rotate_key(provider):
        new_key = pool.current_key
        logger.info(f"Rotated to API key {pool.current_index + 1}/{pool.num_keys} for {provider}")
        
        # Notify callback of new key
        if on_key_rotated and new_key:
            on_key_rotated(new_key)
        
        return True  # Retry with new key
    
    # All keys exhausted
    logger.warning(f"All {pool.num_keys} API keys exhausted for {provider}")
    
    if interactive:
        should_retry = prompt_user_for_retry(error, context, provider, keys_exhausted=True)
        if should_retry:
            # Reset rotation and try again
            manager.reset_rotation(provider)
            # Notify callback of current key (in case it changed)
            if on_key_rotated and pool.current_key:
                on_key_rotated(pool.current_key)
        return should_retry
    else:
        logger.warning(f"Rate limit hit in non-interactive mode with all keys exhausted: {error}")
        return False


def handle_api_error_with_retry(
    error: Exception,
    context: str = "",
    interactive: bool = True,
    provider: str = None,
    on_key_rotated: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Handle an API error, with key rotation and prompting for retry if rate limited.
    
    Args:
        error: The exception that occurred
        context: Optional context about what was being attempted
        interactive: Whether to prompt user (if False, just returns False)
        provider: Optional provider name for key rotation
        on_key_rotated: Optional callback when key is rotated
        
    Returns:
        True if should retry, False if should not retry
        
    Raises:
        UserAbortError: If user chooses to abort
    """
    if is_rate_limit_error(error):
        # Use key rotation if provider is specified
        if provider:
            return handle_rate_limit_with_rotation(
                error, provider, context, interactive, on_key_rotated
            )
        
        # Legacy behavior: just prompt user
        if interactive:
            return prompt_user_for_retry(error, context)
        else:
            logger.warning(f"Rate limit hit in non-interactive mode: {error}")
            return False
    else:
        # Not a rate limit error, don't retry
        return False
        return False
