"""
Rate limit handling utilities for LLM API clients.

Provides interactive retry mechanism when rate limits (429) are hit.
"""
import sys
from loguru import logger


class RateLimitError(Exception):
    """Exception raised when a rate limit (429) error is detected."""
    def __init__(self, message: str, original_exception: Exception):
        super().__init__(message)
        self.original_exception = original_exception


class UserAbortError(Exception):
    """Exception raised when user chooses to abort after rate limit."""
    pass


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


def prompt_user_for_retry(error: Exception, context: str = "") -> bool:
    """
    Prompt the user to decide whether to retry after a rate limit error.
    
    Args:
        error: The rate limit exception
        context: Optional context about what was being attempted
        
    Returns:
        True if user wants to retry, False otherwise
        
    Raises:
        UserAbortError: If user chooses to abort the entire run
    """
    context_msg = f" while {context}" if context else ""
    
    print("\n" + "=" * 60)
    print("⚠️  RATE LIMIT ERROR DETECTED")
    print("=" * 60)
    print(f"Error{context_msg}:")
    print(f"  {error}")
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


def handle_api_error_with_retry(
    error: Exception,
    context: str = "",
    interactive: bool = True
) -> bool:
    """
    Handle an API error, prompting for retry if it's a rate limit error.
    
    Args:
        error: The exception that occurred
        context: Optional context about what was being attempted
        interactive: Whether to prompt user (if False, just returns False)
        
    Returns:
        True if should retry, False if should not retry
        
    Raises:
        UserAbortError: If user chooses to abort
    """
    if is_rate_limit_error(error):
        if interactive:
            return prompt_user_for_retry(error, context)
        else:
            # Non-interactive: don't retry rate limits automatically
            logger.warning(f"Rate limit hit in non-interactive mode: {error}")
            return False
    else:
        # Not a rate limit error, don't retry
        return False
