"""
Methodologies for tool calling optimization experiments.

This module provides different approaches for how tools are presented
to LLMs and how the selection process works.

Available methodologies:
- mcp: Model Context Protocol - all tools passed at once
- clustering: Two-step selection - first select cluster, then tool
- rag: Retrieval-Augmented Generation - semantic search to retrieve relevant tools
"""

from .base import BaseMethodology, StepBasedMethodology, MethodologyResult, StepInfo
from .factory import MethodologyFactory, create_methodology
from .mcp import MCPMethodology
from .clustering import ClusteringMethodology
from .rag import RAGMethodology
from .hybrid import HybridMethodology
from .adaptive_rag import AdaptiveRAGMethodology
from .confidence import ConfidenceMethodology

__all__ = [
    "BaseMethodology",
    "StepBasedMethodology",
    "MethodologyResult",
    "StepInfo",
    "MethodologyFactory",
    "create_methodology",
    "MCPMethodology",
    "ClusteringMethodology",
    "RAGMethodology",
    "HybridMethodology",
    "AdaptiveRAGMethodology",
    "ConfidenceMethodology",
]
