"""
Methodology factory for creating methodology instances from configuration.
"""
from typing import Any, Optional, Type

from .base import BaseMethodology
from .mcp import MCPMethodology
from .clustering import ClusteringMethodology


# Registry of available methodologies
METHODOLOGY_REGISTRY: dict[str, Type[BaseMethodology]] = {
    "mcp": MCPMethodology,
    "clustering": ClusteringMethodology,
}


class MethodologyFactory:
    """
    Factory for creating methodology instances.
    
    Supports registration of custom methodologies and
    configuration-based instantiation.
    """
    
    _registry: dict[str, Type[BaseMethodology]] = METHODOLOGY_REGISTRY.copy()
    
    @classmethod
    def register(cls, name: str, methodology_class: Type[BaseMethodology]) -> None:
        """
        Register a new methodology.
        
        Args:
            name: Name to register under
            methodology_class: Methodology class to register
        """
        cls._registry[name] = methodology_class
    
    @classmethod
    def get_available(cls) -> list[str]:
        """Get list of available methodology names."""
        return list(cls._registry.keys())
    
    @classmethod
    def create(
        cls,
        name: str,
        **kwargs: Any,
    ) -> BaseMethodology:
        """
        Create a methodology instance.
        
        Args:
            name: Methodology name (mcp, clustering, etc.)
            **kwargs: Configuration options passed to methodology constructor
            
        Returns:
            Configured methodology instance
            
        Raises:
            ValueError: If methodology name is not registered
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown methodology '{name}'. Available: {available}"
            )
        
        methodology_class = cls._registry[name]
        return methodology_class(**kwargs)


def create_methodology(
    name: str,
    max_steps: Optional[int] = None,
    allow_backtrack: bool = True,
    allow_decline: bool = False,
    **kwargs: Any,
) -> BaseMethodology:
    """
    Convenience function to create a methodology.
    
    Args:
        name: Methodology name (mcp, clustering, etc.)
        max_steps: Maximum steps for multi-step methodologies
        allow_backtrack: Allow backtracking (for step-based methodologies)
        allow_decline: Allow declining to call any tool
        **kwargs: Additional methodology-specific options
        
    Returns:
        Configured methodology instance
    """
    # Build kwargs based on methodology type
    if name == "mcp":
        return MethodologyFactory.create(
            name,
            allow_no_tool_call=allow_decline,
        )
    elif name == "clustering":
        return MethodologyFactory.create(
            name,
            max_steps=max_steps,
            allow_backtrack=allow_backtrack,
            allow_decline=allow_decline,
            **kwargs,
        )
    else:
        # Generic creation
        return MethodologyFactory.create(name, **kwargs)
