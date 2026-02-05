"""
Timeout and interrupt handling utilities for cross-platform support.

This module provides robust timeout mechanisms that work on both Windows and macOS/Linux.

Key features:
1. Interruptible timeouts that respond to Ctrl+C
2. Proper thread cleanup using daemon threads
3. Socket-level timeout configuration for API clients
4. Cross-platform signal handling

The main problem with ThreadPoolExecutor.result(timeout=...) is that:
1. The underlying thread continues running even after timeout
2. On Windows, signals can't interrupt the main thread when blocked
3. HTTP requests in threads can hang indefinitely

Our solution:
1. Use daemon threads that die when main program exits
2. Periodically check for interrupts using short timeout intervals
3. Store interrupt flag for detecting Ctrl+C from worker threads
4. Configure HTTP client timeouts at the socket level
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FuturesTimeoutError
from typing import TypeVar, Callable, Any, Optional
from loguru import logger

T = TypeVar('T')


class InterruptFlag:
    """
    Thread-safe flag for detecting and propagating interrupts.
    
    This is used to signal worker threads that they should stop,
    and to detect if an interrupt occurred during execution.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._interrupted = threading.Event()
        return cls._instance
    
    def set(self):
        """Set the interrupt flag."""
        self._interrupted.set()
    
    def clear(self):
        """Clear the interrupt flag."""
        self._interrupted.clear()
    
    def is_set(self) -> bool:
        """Check if interrupt flag is set."""
        return self._interrupted.is_set()
    
    def wait(self, timeout: float = None) -> bool:
        """Wait for interrupt flag to be set."""
        return self._interrupted.wait(timeout=timeout)


# Global interrupt flag instance
interrupt_flag = InterruptFlag()


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


class InterruptedError(Exception):
    """Raised when an operation is interrupted by the user."""
    pass


def run_with_interruptible_timeout(
    func: Callable[..., T],
    timeout_seconds: float,
    *args,
    check_interval: float = 0.5,
    **kwargs
) -> T:
    """
    Run a function with a timeout that can be interrupted by Ctrl+C.
    
    Unlike ThreadPoolExecutor.result(timeout=...), this function:
    1. Can be interrupted by Ctrl+C on Windows
    2. Uses daemon threads that die when program exits
    3. Periodically checks for interrupts
    
    Args:
        func: Function to run
        timeout_seconds: Maximum time to wait in seconds
        *args: Positional arguments for func
        check_interval: How often to check for interrupts (default 0.5s)
        **kwargs: Keyword arguments for func
        
    Returns:
        Result of func(*args, **kwargs)
        
    Raises:
        TimeoutError: If the function times out
        InterruptedError: If interrupted by Ctrl+C
        Exception: Any exception raised by func
    """
    result_holder = {'result': None, 'exception': None, 'done': False}
    
    def worker():
        try:
            result_holder['result'] = func(*args, **kwargs)
        except Exception as e:
            result_holder['exception'] = e
        finally:
            result_holder['done'] = True
    
    # Use daemon thread so it dies when main program exits
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    
    start_time = time.time()
    
    while True:
        # Check for interrupt flag
        if interrupt_flag.is_set():
            logger.debug("Interrupt flag detected in timeout wrapper")
            raise InterruptedError("Operation interrupted by user")
        
        # Check if done
        if result_holder['done']:
            if result_holder['exception'] is not None:
                raise result_holder['exception']
            return result_holder['result']
        
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            logger.warning(f"Operation timed out after {timeout_seconds}s")
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        
        # Wait a short interval before checking again
        # This allows the main thread to process signals
        remaining = timeout_seconds - elapsed
        wait_time = min(check_interval, remaining)
        thread.join(timeout=wait_time)


def configure_client_timeouts(connect_timeout: float = 30.0, read_timeout: float = 120.0) -> dict:
    """
    Get timeout configuration for HTTP clients.
    
    These timeouts should be passed to the HTTP client library to ensure
    that requests don't hang indefinitely at the socket level.
    
    Args:
        connect_timeout: Timeout for establishing connection (seconds)
        read_timeout: Timeout for reading response (seconds)
        
    Returns:
        Dictionary with timeout settings for common HTTP libraries
    """
    return {
        'connect_timeout': connect_timeout,
        'read_timeout': read_timeout,
        # For requests library
        'requests_timeout': (connect_timeout, read_timeout),
        # For httpx/httpcore (used by some clients)
        'httpx_timeout': {
            'connect': connect_timeout,
            'read': read_timeout,
            'write': read_timeout,
            'pool': connect_timeout,
        }
    }


def get_default_client_timeouts() -> tuple[float, float]:
    """
    Get default client timeouts from environment or defaults.
    
    Environment variables:
        EXPERIMENT_CLIENT_CONNECT_TIMEOUT: Connection timeout (default: 30)
        EXPERIMENT_CLIENT_READ_TIMEOUT: Read timeout (default: 120)
    
    Returns:
        Tuple of (connect_timeout, read_timeout)
    """
    connect_timeout = float(os.environ.get('EXPERIMENT_CLIENT_CONNECT_TIMEOUT', '30'))
    read_timeout = float(os.environ.get('EXPERIMENT_CLIENT_READ_TIMEOUT', '120'))
    return connect_timeout, read_timeout


class TimeoutExecutor:
    """
    Executor that runs functions with interruptible timeouts.
    
    This class provides a cleaner interface for running functions with timeouts
    and handles cleanup properly.
    
    Usage:
        with TimeoutExecutor() as executor:
            result = executor.run(func, args, timeout=60)
    """
    
    def __init__(self, default_timeout: float = 60.0, check_interval: float = 0.5):
        """
        Initialize the executor.
        
        Args:
            default_timeout: Default timeout in seconds
            check_interval: How often to check for interrupts
        """
        self.default_timeout = default_timeout
        self.check_interval = check_interval
        self._thread_pool = None
    
    def __enter__(self):
        # We use daemon threads directly, not ThreadPoolExecutor
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Daemon threads clean up automatically
        pass
    
    def run(
        self,
        func: Callable[..., T],
        *args,
        timeout: float = None,
        **kwargs
    ) -> T:
        """
        Run a function with timeout.
        
        Args:
            func: Function to run
            *args: Positional arguments
            timeout: Timeout in seconds (uses default if not specified)
            **kwargs: Keyword arguments
            
        Returns:
            Result of func
            
        Raises:
            TimeoutError: If timeout exceeded
            InterruptedError: If interrupted
        """
        timeout = timeout if timeout is not None else self.default_timeout
        return run_with_interruptible_timeout(
            func, timeout, *args, 
            check_interval=self.check_interval, 
            **kwargs
        )


def set_interrupt():
    """Set the global interrupt flag (called from signal handler)."""
    interrupt_flag.set()


def clear_interrupt():
    """Clear the global interrupt flag."""
    interrupt_flag.clear()


def is_interrupted() -> bool:
    """Check if the global interrupt flag is set."""
    return interrupt_flag.is_set()
