"""
Confidence-based Fallback methodology for tool calling experiments.

Implements a fallback chain that tries methodologies in order of
speed/cost, falling back to more expensive ones when confidence is low:
1. Try RAG first (fastest, uses embeddings)
2. If confidence low, try Clustering (uses LLM for category selection)
3. If still low, fall back to MCP (full context, most accurate)

Confidence is estimated from RAG similarity scores and clustering
step outcomes.
"""
import time
from typing import Any, Optional
import numpy as np
from loguru import logger

from src.tools.base import Tool
from src.clients.base import BaseLLMClient
from .base import BaseMethodology, MethodologyResult, StepInfo, StepType
from .rag import RAGMethodology
from .clustering import ClusteringMethodology
from .mcp import MCPMethodology


class ConfidenceMethodology(BaseMethodology):
    """
    Confidence-based Fallback methodology.
    
    Tries methodologies in order of cost/latency, using confidence
    estimation to decide when to fall back to more expensive options.
    
    Fallback chain: RAG → Clustering → MCP
    
    Confidence estimation strategies:
    - RAG: Based on top similarity score and gap to second score
    - Clustering: Based on successful category/tool selection
    """
    
    NAME: str = "confidence"
    
    def __init__(
        self,
        rag_confidence_threshold: float = 0.7,
        clustering_confidence_threshold: float = 0.6,
        rag_config: Optional[dict] = None,
        clustering_config: Optional[dict] = None,
        allow_no_tool_call: bool = False,
        allow_clarification: bool = False,
    ):
        """
        Initialize Confidence methodology.
        
        Args:
            rag_confidence_threshold: Min confidence to accept RAG result (0-1)
            clustering_confidence_threshold: Min confidence to accept Clustering result (0-1)
            rag_config: Configuration dict for RAG methodology
            clustering_config: Configuration dict for Clustering methodology
            allow_no_tool_call: If True, allow declining to call any tool
            allow_clarification: If True, allow requesting clarification
        """
        self.rag_confidence_threshold = rag_confidence_threshold
        self.clustering_confidence_threshold = clustering_confidence_threshold
        self.allow_no_tool_call = allow_no_tool_call
        self.allow_clarification = allow_clarification
        
        # Initialize sub-methodologies
        rag_kwargs = rag_config or {}
        rag_kwargs.setdefault("allow_no_tool_call", allow_no_tool_call)
        rag_kwargs.setdefault("allow_clarification", allow_clarification)
        self.rag = RAGMethodology(**rag_kwargs)
        
        clustering_kwargs = clustering_config or {}
        clustering_kwargs.setdefault("allow_decline", allow_no_tool_call)
        clustering_kwargs.setdefault("allow_clarification", allow_clarification)
        self.clustering = ClusteringMethodology(**clustering_kwargs)
        
        self.mcp = MCPMethodology(
            allow_no_tool_call=allow_no_tool_call,
            allow_clarification=allow_clarification,
        )
        
        logger.debug(
            f"[Confidence] Initialized with RAG threshold={rag_confidence_threshold}, "
            f"Clustering threshold={clustering_confidence_threshold}, "
            f"allow_clarification={allow_clarification}"
        )
    
    def _compute_rag_confidence(self, result: MethodologyResult) -> float:
        """
        Compute confidence score from RAG result.
        
        Uses two signals:
        1. Top similarity score (higher = more confident)
        2. Gap between top and second score (larger gap = more confident)
        
        Args:
            result: MethodologyResult from RAG
            
        Returns:
            Confidence score between 0 and 1
        """
        if not hasattr(result, '_rag_metadata') or result._rag_metadata is None:
            return 0.5  # Default uncertainty
        
        scores = result._rag_metadata.get("similarity_scores", [])
        if not scores:
            return 0.5
        
        top_score = scores[0]
        
        # Compute gap to second score
        if len(scores) > 1:
            score_gap = scores[0] - scores[1]
        else:
            score_gap = 0.5  # Only one tool, assume moderate gap
        
        # Combine signals: 60% weight on top score, 40% on gap
        # Scores are typically in [0, 1] range for normalized embeddings
        # Gap is typically in [0, 0.5] range, so we scale it
        normalized_gap = min(1.0, score_gap * 2)  # Scale gap to [0, 1]
        
        confidence = (top_score * 0.6) + (normalized_gap * 0.4)
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        logger.debug(
            f"[Confidence] RAG confidence: {confidence:.3f} "
            f"(top_score={top_score:.3f}, gap={score_gap:.3f})"
        )
        
        return confidence
    
    def _compute_clustering_confidence(self, result: MethodologyResult) -> float:
        """
        Compute confidence score from Clustering result.
        
        Uses signals:
        1. Whether a tool was successfully selected
        2. Number of backtracks (more backtracks = less confident)
        3. Number of steps (more steps may indicate difficulty)
        
        Args:
            result: MethodologyResult from Clustering
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence: did we get a tool?
        if result.called_tool is None and not result.declined_tool_call:
            return 0.0  # No selection made
        
        if result.declined_tool_call:
            return 0.7  # Explicit decline is somewhat confident
        
        # Start with high confidence
        confidence = 0.9
        
        # Reduce confidence based on backtracks
        # Each backtrack reduces confidence by 0.1
        confidence -= result.backtrack_count * 0.1
        
        # Reduce confidence if too many steps (expected: 2 steps)
        if len(result.steps) > 3:
            confidence -= (len(result.steps) - 3) * 0.05
        
        # Check for errors in steps
        for step in result.steps:
            if step.error:
                confidence -= 0.2
                break
        
        confidence = max(0.0, min(1.0, confidence))
        
        logger.debug(
            f"[Confidence] Clustering confidence: {confidence:.3f} "
            f"(backtracks={result.backtrack_count}, steps={len(result.steps)})"
        )
        
        return confidence
    
    def run_single(
        self,
        prompt: str,
        tools: list[Tool],
        client: BaseLLMClient,
        system_instruction: Optional[str] = None,
    ) -> MethodologyResult:
        """
        Run Confidence methodology with fallback chain.
        
        Tries methodologies in order, falling back when confidence is low:
        1. RAG (fastest)
        2. Clustering (moderate)
        3. MCP (slowest but most reliable)
        
        Args:
            prompt: User prompt to process
            tools: All available tools
            client: LLM client to use
            system_instruction: Optional system instruction
            
        Returns:
            MethodologyResult with tool selection and fallback metadata
        """
        logger.debug(f"[Confidence] ===== run_single START =====")
        logger.debug(f"[Confidence] Prompt: {prompt[:150]}...")
        logger.debug(f"[Confidence] Thresholds: RAG={self.rag_confidence_threshold}, Clustering={self.clustering_confidence_threshold}")
        
        total_start = time.time()
        fallback_info = {
            "methods_tried": [],
            "confidences": {},
            "final_method": None,
            "all_results": [],  # Store all results for token aggregation
        }
        
        # Step 1: Try RAG first
        logger.debug("[Confidence] Trying RAG methodology...")
        rag_result = self.rag.run_single(prompt, tools, client, system_instruction)
        rag_confidence = self._compute_rag_confidence(rag_result)
        
        fallback_info["methods_tried"].append("rag")
        fallback_info["confidences"]["rag"] = rag_confidence
        fallback_info["all_results"].append(rag_result)
        
        if rag_confidence >= self.rag_confidence_threshold:
            logger.debug(f"[Confidence] RAG confidence {rag_confidence:.3f} >= {self.rag_confidence_threshold}, accepting")
            fallback_info["final_method"] = "rag"
            return self._wrap_result(rag_result, fallback_info, total_start)
        
        logger.debug(f"[Confidence] RAG confidence {rag_confidence:.3f} < {self.rag_confidence_threshold}, falling back to Clustering")
        
        # Step 2: Try Clustering
        logger.debug("[Confidence] Trying Clustering methodology...")
        clustering_result = self.clustering.run_single(prompt, tools, client, system_instruction)
        clustering_confidence = self._compute_clustering_confidence(clustering_result)
        
        fallback_info["methods_tried"].append("clustering")
        fallback_info["confidences"]["clustering"] = clustering_confidence
        fallback_info["all_results"].append(clustering_result)
        
        if clustering_confidence >= self.clustering_confidence_threshold:
            logger.debug(f"[Confidence] Clustering confidence {clustering_confidence:.3f} >= {self.clustering_confidence_threshold}, accepting")
            fallback_info["final_method"] = "clustering"
            return self._wrap_result(clustering_result, fallback_info, total_start)
        
        logger.debug(f"[Confidence] Clustering confidence {clustering_confidence:.3f} < {self.clustering_confidence_threshold}, falling back to MCP")
        
        # Step 3: Fall back to MCP
        logger.debug("[Confidence] Trying MCP methodology (final fallback)...")
        mcp_result = self.mcp.run_single(prompt, tools, client, system_instruction)
        
        fallback_info["methods_tried"].append("mcp")
        fallback_info["confidences"]["mcp"] = 1.0  # MCP is the ground truth
        fallback_info["final_method"] = "mcp"
        fallback_info["all_results"].append(mcp_result)
        
        return self._wrap_result(mcp_result, fallback_info, total_start)
    
    def _wrap_result(
        self,
        result: MethodologyResult,
        fallback_info: dict[str, Any],
        start_time: float,
    ) -> MethodologyResult:
        """
        Wrap a methodology result with confidence metadata.
        
        Args:
            result: Original MethodologyResult
            fallback_info: Dict with fallback chain information
            start_time: Start time for total latency calculation
            
        Returns:
            MethodologyResult with confidence metadata attached
        """
        total_latency = (time.time() - start_time) * 1000
        
        # Aggregate token usage from all methods tried
        total_tokens_input = 0
        total_tokens_output = 0
        for r in fallback_info.get("all_results", []):
            if r.tokens_input is not None:
                total_tokens_input += r.tokens_input
            if r.tokens_output is not None:
                total_tokens_output += r.tokens_output
        
        # Create a new result with our methodology name
        wrapped = MethodologyResult(
            success=result.success,
            called_tool=result.called_tool,
            called_args=result.called_args,
            all_calls=result.all_calls,
            latency_ms=total_latency,
            error=result.error,
            raw_response=result.raw_response,
            model=result.model,
            provider=result.provider,
            methodology=self.NAME,
            steps=result.steps,
            categories_selected=result.categories_selected,
            backtrack_count=result.backtrack_count,
            declined_tool_call=result.declined_tool_call,
            final_category=result.final_category,
            # Aggregated token usage across all fallback attempts
            tokens_input=total_tokens_input if total_tokens_input > 0 else None,
            tokens_output=total_tokens_output if total_tokens_output > 0 else None,
            tokens_total=(total_tokens_input + total_tokens_output) if (total_tokens_input > 0 or total_tokens_output > 0) else None,
        )
        
        # Attach confidence-specific metadata
        wrapped._confidence_metadata = {
            "methods_tried": fallback_info["methods_tried"],
            "confidences": fallback_info["confidences"],
            "final_method": fallback_info["final_method"],
            "num_fallbacks": len(fallback_info["methods_tried"]) - 1,
            "rag_threshold": self.rag_confidence_threshold,
            "clustering_threshold": self.clustering_confidence_threshold,
        }
        
        # Preserve original metadata if available
        if hasattr(result, '_rag_metadata'):
            wrapped._rag_metadata = result._rag_metadata
        if hasattr(result, '_adaptive_metadata'):
            wrapped._adaptive_metadata = result._adaptive_metadata
        
        logger.debug(f"[Confidence] ===== run_single END =====")
        logger.debug(
            f"[Confidence] Total latency: {total_latency:.1f}ms, "
            f"Final method: {fallback_info['final_method']}, "
            f"Fallbacks: {len(fallback_info['methods_tried']) - 1}"
        )
        
        return wrapped
