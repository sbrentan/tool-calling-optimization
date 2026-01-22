"""
API Key management and rotation for LLM providers.

Supports rotating through multiple API keys when rate limits are hit,
with per-provider key pools loaded from environment variables.

Environment variable pattern:
    {PROVIDER}_API_KEY        - Primary key
    {PROVIDER}_API_KEY_2      - Secondary key
    {PROVIDER}_API_KEY_3      - Tertiary key
    ... and so on

Example:
    CEREBRAS_API_KEY=key1
    CEREBRAS_API_KEY_2=key2
    CEREBRAS_API_KEY_3=key3
    
    OPENAI_API_KEY=key1
    OPENAI_API_KEY_2=key2
"""
import os
import re
from typing import Optional, Callable
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class ApiKeyPool:
    """
    Pool of API keys for a single provider with rotation support.
    
    Tracks current key index and rotates through available keys
    when rate limits are hit.
    """
    provider: str
    keys: list[str] = field(default_factory=list)
    current_index: int = 0
    _exhausted_in_rotation: set = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize exhausted set if not provided."""
        if not hasattr(self, '_exhausted_in_rotation') or self._exhausted_in_rotation is None:
            self._exhausted_in_rotation = set()
    
    @property
    def current_key(self) -> Optional[str]:
        """Get the current API key."""
        if not self.keys:
            return None
        return self.keys[self.current_index]
    
    @property
    def has_keys(self) -> bool:
        """Check if any keys are available."""
        return len(self.keys) > 0
    
    @property
    def num_keys(self) -> int:
        """Get total number of keys."""
        return len(self.keys)
    
    def rotate(self) -> bool:
        """
        Rotate to the next API key.
        
        Marks current key as exhausted in this rotation cycle and moves
        to the next available key.
        
        Returns:
            True if rotated to a new key, False if all keys exhausted
        """
        if not self.keys:
            return False
        
        # Mark current key as exhausted in this rotation
        self._exhausted_in_rotation.add(self.current_index)
        
        # Check if all keys have been exhausted
        if len(self._exhausted_in_rotation) >= len(self.keys):
            logger.warning(f"All {len(self.keys)} API keys for {self.provider} have been rate limited")
            return False
        
        # Find next non-exhausted key
        next_index = (self.current_index + 1) % len(self.keys)
        while next_index in self._exhausted_in_rotation:
            next_index = (next_index + 1) % len(self.keys)
        
        old_index = self.current_index
        self.current_index = next_index
        
        logger.info(
            f"Rotated {self.provider} API key: key_{old_index + 1} -> key_{self.current_index + 1} "
            f"({len(self._exhausted_in_rotation)}/{len(self.keys)} exhausted)"
        )
        
        return True
    
    def reset_rotation(self) -> None:
        """
        Reset the rotation cycle.
        
        Called when user chooses to retry after all keys exhausted,
        allowing all keys to be tried again.
        """
        self._exhausted_in_rotation.clear()
        self.current_index = 0  # Also reset to first key
        logger.info(f"Reset rotation cycle for {self.provider} - all {len(self.keys)} keys available again, starting from key 1")
    
    def key_succeeded(self) -> None:
        """
        Mark that the current key succeeded.
        
        Clears the exhausted set since the current key is working.
        """
        if self._exhausted_in_rotation:
            logger.debug(f"Key {self.current_index + 1} succeeded, clearing exhausted markers")
            self._exhausted_in_rotation.clear()


class ApiKeyManager:
    """
    Manages API key pools for multiple providers with rotation support.
    
    Loads keys from environment variables following the pattern:
        {PROVIDER}_API_KEY        - Primary key
        {PROVIDER}_API_KEY_2      - Secondary key
        {PROVIDER}_API_KEY_N      - Nth key
    
    Provides:
        - Automatic key rotation on rate limit errors
        - Per-provider key pools
        - Callbacks for clients to update their keys
    """
    
    # Known providers and their env var prefixes
    PROVIDER_ENV_PREFIXES = {
        "cerebras": "CEREBRAS",
        "openai": "OPENAI",
        "gemini": "GEMINI",
    }
    
    _instance: Optional["ApiKeyManager"] = None
    
    def __init__(self):
        """Initialize the key manager and load keys from environment."""
        self._pools: dict[str, ApiKeyPool] = {}
        self._key_update_callbacks: dict[str, list[Callable[[str], None]]] = {}
        self._load_all_keys()
    
    @classmethod
    def get_instance(cls) -> "ApiKeyManager":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
    
    def _load_all_keys(self) -> None:
        """Load API keys for all providers from environment variables."""
        for provider, prefix in self.PROVIDER_ENV_PREFIXES.items():
            keys = self._load_provider_keys(prefix)
            if keys:
                self._pools[provider] = ApiKeyPool(provider=provider, keys=keys)
                logger.info(f"Loaded {len(keys)} API key(s) for {provider}")
    
    def _load_provider_keys(self, prefix: str) -> list[str]:
        """
        Load all API keys for a provider from environment variables.
        
        Looks for:
            {PREFIX}_API_KEY
            {PREFIX}_API_KEY_2
            {PREFIX}_API_KEY_3
            ... etc
        
        Args:
            prefix: Environment variable prefix (e.g., "CEREBRAS")
            
        Returns:
            List of API keys in order
        """
        keys = []
        
        # Primary key (no suffix)
        primary = os.getenv(f"{prefix}_API_KEY")
        if primary:
            keys.append(primary)
        
        # Secondary and beyond (with suffix)
        suffix_num = 2
        while True:
            key = os.getenv(f"{prefix}_API_KEY_{suffix_num}")
            if key:
                keys.append(key)
                suffix_num += 1
            else:
                break
        
        return keys
    
    def get_pool(self, provider: str) -> Optional[ApiKeyPool]:
        """Get the key pool for a provider."""
        return self._pools.get(provider)
    
    def get_current_key(self, provider: str) -> Optional[str]:
        """Get the current API key for a provider."""
        pool = self._pools.get(provider)
        return pool.current_key if pool else None
    
    def register_key_update_callback(self, provider: str, callback: Callable[[str], None]) -> None:
        """
        Register a callback to be called when a provider's key is rotated.
        
        Args:
            provider: Provider name
            callback: Function that takes the new API key as argument
        """
        if provider not in self._key_update_callbacks:
            self._key_update_callbacks[provider] = []
        self._key_update_callbacks[provider].append(callback)
    
    def unregister_key_update_callback(self, provider: str, callback: Callable[[str], None]) -> None:
        """Unregister a key update callback."""
        if provider in self._key_update_callbacks:
            try:
                self._key_update_callbacks[provider].remove(callback)
            except ValueError:
                pass
    
    def _notify_key_update(self, provider: str, new_key: str) -> None:
        """Notify all registered callbacks of a key update."""
        for callback in self._key_update_callbacks.get(provider, []):
            try:
                callback(new_key)
            except Exception as e:
                logger.error(f"Error in key update callback for {provider}: {e}")
    
    def rotate_key(self, provider: str) -> bool:
        """
        Rotate to the next API key for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            True if rotated successfully, False if no more keys available
        """
        pool = self._pools.get(provider)
        if not pool:
            logger.warning(f"No API key pool for provider: {provider}")
            return False
        
        if pool.rotate():
            # Notify callbacks of new key
            self._notify_key_update(provider, pool.current_key)
            return True
        
        return False
    
    def reset_rotation(self, provider: str) -> None:
        """Reset the rotation cycle for a provider."""
        pool = self._pools.get(provider)
        if pool:
            pool.reset_rotation()
    
    def reset_all_rotations(self) -> None:
        """
        Reset rotation cycles for all providers.
        
        Should be called between experiment configs to ensure clean state.
        """
        for provider, pool in self._pools.items():
            pool.reset_rotation()
        logger.info("Reset rotation cycles for all providers")
    
    def mark_key_succeeded(self, provider: str) -> None:
        """Mark that the current key for a provider succeeded."""
        pool = self._pools.get(provider)
        if pool:
            pool.key_succeeded()
    
    def get_status(self) -> dict[str, dict]:
        """
        Get status of all API key pools.
        
        Returns:
            Dict mapping provider to status info
        """
        status = {}
        for provider, pool in self._pools.items():
            status[provider] = {
                "total_keys": pool.num_keys,
                "current_index": pool.current_index + 1,  # 1-indexed for display
                "exhausted_count": len(pool._exhausted_in_rotation),
            }
        return status


# Convenience functions for module-level access
def get_api_key_manager() -> ApiKeyManager:
    """Get the global API key manager instance."""
    return ApiKeyManager.get_instance()


def get_current_key(provider: str) -> Optional[str]:
    """Get the current API key for a provider."""
    return get_api_key_manager().get_current_key(provider)


def rotate_key(provider: str) -> bool:
    """Rotate to the next API key for a provider."""
    return get_api_key_manager().rotate_key(provider)


def reset_rotation(provider: str) -> None:
    """Reset the rotation cycle for a provider."""
    get_api_key_manager().reset_rotation(provider)


def reset_all_rotations() -> None:
    """Reset rotation cycles for all providers."""
    get_api_key_manager().reset_all_rotations()


def mark_key_succeeded(provider: str) -> None:
    """Mark that the current key for a provider succeeded."""
    get_api_key_manager().mark_key_succeeded(provider)
