"""
Adaptive RAG methodology for tool calling experiments.

Extends the standard RAG methodology with dynamic top-K selection:
1. Instead of fixed top-K, analyzes similarity score distribution
2. Uses elbow detection to find optimal cutoff point
3. Bounded by min_k and max_k parameters

This provides better tool retrieval by adapting to query complexity -
simple queries get fewer tools, complex queries get more.
"""
import time
from typing import Any, Optional
import numpy as np
from loguru import logger

from src.tools.base import Tool
from src.clients.base import BaseLLMClient
from .base import MethodologyResult, StepInfo, StepType
from .rag import RAGMethodology


class AdaptiveRAGMethodology(RAGMethodology):
    """
    Adaptive RAG methodology with dynamic top-K selection.
    
    Extends RAGMethodology to dynamically determine the number of
    tools to retrieve based on similarity score distribution.
    
    Uses multiple strategies:
    1. Elbow detection: Find significant drops in similarity scores
    2. Threshold-based: Include all tools above minimum similarity
    3. Combined: Use the more conservative of the two
    """
    
    NAME: str = "adaptive_rag"
    
    def __init__(
        self,
        embedding_model: str = RAGMethodology.DEFAULT_EMBEDDING_MODEL,
        min_k: int = 3,
        max_k: int = 20,
        similarity_drop_threshold: float = 0.1,
        min_similarity: float = 0.3,
        allow_no_tool_call: bool = False,
        cache_embeddings: bool = True,
        include_params_in_embedding: bool = False,
    ):
        """
        Initialize Adaptive RAG methodology.
        
        Args:
            embedding_model: Name of sentence-transformers model to use
            min_k: Minimum number of tools to retrieve
            max_k: Maximum number of tools to retrieve
            similarity_drop_threshold: Minimum drop in similarity to trigger elbow
            min_similarity: Minimum similarity score for inclusion
            allow_no_tool_call: If True, add a decline option
            cache_embeddings: If True, cache tool embeddings between calls
            include_params_in_embedding: If True, include parameter info in embeddings
        """
        # Initialize parent with max_k as top_k (we'll override retrieval)
        super().__init__(
            embedding_model=embedding_model,
            top_k=max_k,
            similarity_threshold=0.0,  # We handle threshold ourselves
            allow_no_tool_call=allow_no_tool_call,
            cache_embeddings=cache_embeddings,
            include_params_in_embedding=include_params_in_embedding,
        )
        
        self.min_k = min_k
        self.max_k = max_k
        self.similarity_drop_threshold = similarity_drop_threshold
        self.min_similarity = min_similarity
        
        logger.debug(
            f"[AdaptiveRAG] Initialized with min_k={min_k}, max_k={max_k}, "
            f"drop_threshold={similarity_drop_threshold}, min_sim={min_similarity}"
        )
    
    def _determine_adaptive_k(
        self,
        similarities: np.ndarray,
    ) -> tuple[int, dict[str, Any]]:
        """
        Determine optimal k based on similarity distribution.
        
        Uses multiple strategies and returns the most appropriate k:
        1. Elbow detection: Find first significant drop in similarity
        2. Threshold-based: Count tools above minimum similarity
        3. Return bounded result with metadata about decision
        
        Args:
            similarities: Array of similarity scores (unsorted)
            
        Returns:
            Tuple of (optimal_k, metadata_dict)
        """
        sorted_sims = np.sort(similarities)[::-1]  # Descending order
        n_tools = len(sorted_sims)
        
        metadata = {
            "top_5_similarities": sorted_sims[:5].tolist() if len(sorted_sims) >= 5 else sorted_sims.tolist(),
            "similarity_mean": float(np.mean(similarities)),
            "similarity_std": float(np.std(similarities)),
        }
        
        # Strategy 1: Elbow detection - find first significant drop
        elbow_k = n_tools  # Default to all if no elbow found
        if len(sorted_sims) > 1:
            gradients = np.abs(np.diff(sorted_sims))
            for i, grad in enumerate(gradients):
                # Check if this is a significant drop AND we've passed min_k
                if i >= self.min_k - 1 and grad >= self.similarity_drop_threshold:
                    elbow_k = i + 1  # +1 because gradient is between points
                    break
        
        metadata["elbow_k"] = elbow_k
        
        # Strategy 2: Count tools above minimum similarity threshold
        above_threshold = int(np.sum(sorted_sims >= self.min_similarity))
        metadata["above_threshold_k"] = above_threshold
        
        # Strategy 3: Combine - use the more restrictive (smaller) value
        # but ensure we have at least min_k
        combined_k = min(elbow_k, above_threshold) if above_threshold > 0 else elbow_k
        
        # Apply bounds
        final_k = max(self.min_k, min(combined_k, self.max_k, n_tools))
        
        metadata["strategy_used"] = "elbow" if final_k == elbow_k else (
            "threshold" if final_k == above_threshold else "bounded"
        )
        metadata["final_k"] = final_k
        
        logger.debug(
            f"[AdaptiveRAG] Adaptive k: elbow={elbow_k}, threshold={above_threshold}, "
            f"final={final_k} (strategy: {metadata['strategy_used']})"
        )
        
        return final_k, metadata
    
    def _retrieve_tools_adaptive(
        self,
        query: str,
        tools: list[Tool],
    ) -> tuple[list[Tool], list[float], float, dict[str, Any]]:
        """
        Retrieve tools with adaptive k selection.
        
        Args:
            query: User query
            tools: All available tools
            
        Returns:
            Tuple of (retrieved_tools, similarity_scores, retrieval_latency_ms, adaptive_metadata)
        """
        start_time = time.time()
        
        # Embed tools and query
        tool_names, tool_embeddings = self._embed_tools(tools)
        query_embedding = self._embed_query(query)
        
        # Compute all similarities
        similarities = np.dot(tool_embeddings, query_embedding)
        
        # Determine adaptive k
        adaptive_k, adaptive_metadata = self._determine_adaptive_k(similarities)
        
        # Get top-k indices
        if adaptive_k >= len(tools):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -adaptive_k)[-adaptive_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        # Build results
        retrieved_tools = []
        scores = []
        name_to_tool = {tool.name: tool for tool in tools}
        
        for idx in top_indices:
            tool_name = tool_names[idx]
            score = float(similarities[idx])
            if tool_name in name_to_tool:
                retrieved_tools.append(name_to_tool[tool_name])
                scores.append(score)
        
        retrieval_latency = (time.time() - start_time) * 1000
        
        logger.debug(
            f"[AdaptiveRAG] Retrieved {len(retrieved_tools)} tools "
            f"(adaptive k={adaptive_k}) in {retrieval_latency:.1f}ms"
        )
        
        return retrieved_tools, scores, retrieval_latency, adaptive_metadata
    
    def run_single(
        self,
        prompt: str,
        tools: list[Tool],
        client: BaseLLMClient,
        system_instruction: Optional[str] = None,
    ) -> MethodologyResult:
        """
        Run Adaptive RAG methodology for a single prompt.
        
        Similar to RAG but uses adaptive k selection based on
        similarity score distribution.
        
        Args:
            prompt: User prompt to process
            tools: All available tools
            client: LLM client to use
            system_instruction: Optional system instruction
            
        Returns:
            MethodologyResult with tool selection and adaptive metadata
        """
        logger.debug(f"[AdaptiveRAG] ===== run_single START =====")
        logger.debug(f"[AdaptiveRAG] Prompt: {prompt[:150]}...")
        logger.debug(f"[AdaptiveRAG] Total tools: {len(tools)}, k range: [{self.min_k}, {self.max_k}]")
        
        # Step 1: Retrieve tools with adaptive k
        retrieved_tools, similarity_scores, retrieval_latency, adaptive_metadata = \
            self._retrieve_tools_adaptive(query=prompt, tools=tools)
        
        if not retrieved_tools:
            logger.warning("[AdaptiveRAG] No tools retrieved!")
            return MethodologyResult(
                success=False,
                called_tool=None,
                called_args=None,
                all_calls=[],
                latency_ms=retrieval_latency,
                error="No tools retrieved",
                raw_response=None,
                model="",
                provider="",
                methodology=self.NAME,
                steps=[],
                categories_selected=[],
                backtrack_count=0,
                declined_tool_call=False,
                final_category=None,
            )
        
        # Optionally add decline pseudo-tool
        tools_to_use = list(retrieved_tools)
        if self.allow_no_tool_call:
            decline_tool = Tool(
                name=self.DECLINE_TOOL,
                description="Indicate that no tool call is needed for this request.",
                category="system",
                parameters=[],
                tags=["system"],
                complexity="simple",
            )
            tools_to_use.append(decline_tool)
        
        # Step 2: Call LLM with retrieved tools
        logger.debug(f"[AdaptiveRAG] Calling LLM with {len(tools_to_use)} tools...")
        
        call_result = client.call_with_tools(
            prompt=prompt,
            tools=tools_to_use,
            system_instruction=system_instruction,
        )
        
        total_latency = retrieval_latency + call_result.latency_ms
        
        # Check for decline
        declined = False
        called_tool = call_result.called_tool
        if called_tool == self.DECLINE_TOOL:
            declined = True
            called_tool = None
        
        # Create step info
        retrieval_step = StepInfo(
            step_number=1,
            step_type=StepType.SELECT_CATEGORY,
            selection=f"retrieved_{len(retrieved_tools)}_tools_k{adaptive_metadata['final_k']}",
            latency_ms=retrieval_latency,
            raw_response=None,
            error=None,
        )
        
        selection_step = StepInfo(
            step_number=2,
            step_type=StepType.DECLINE if declined else StepType.SELECT_TOOL,
            selection=call_result.called_tool,
            latency_ms=call_result.latency_ms,
            raw_response=call_result.raw_response,
            error=call_result.error,
        )
        
        # Determine final category
        final_category = None
        if called_tool:
            for tool in tools:
                if tool.name == called_tool:
                    final_category = tool.category
                    break
        
        # Get categories of retrieved tools
        retrieved_categories = list(set(t.category for t in retrieved_tools))
        
        result = MethodologyResult(
            success=call_result.success,
            called_tool=called_tool,
            called_args=call_result.called_args,
            all_calls=call_result.all_calls,
            latency_ms=total_latency,
            error=call_result.error,
            raw_response=call_result.raw_response,
            model=call_result.model,
            provider=call_result.provider,
            methodology=self.NAME,
            steps=[retrieval_step, selection_step],
            categories_selected=retrieved_categories,
            backtrack_count=0,
            declined_tool_call=declined,
            final_category=final_category,
        )
        
        # Store adaptive RAG-specific metadata
        result._rag_metadata = {
            "retrieved_tools": [t.name for t in retrieved_tools],
            "similarity_scores": similarity_scores,
            "retrieval_latency_ms": retrieval_latency,
            "top_k_requested": self.max_k,
            "tools_retrieved": len(retrieved_tools),
            "similarity_threshold": self.min_similarity,
        }
        
        result._adaptive_metadata = {
            "min_k": self.min_k,
            "max_k": self.max_k,
            "adaptive_k_used": adaptive_metadata["final_k"],
            "strategy_used": adaptive_metadata["strategy_used"],
            "elbow_k": adaptive_metadata["elbow_k"],
            "above_threshold_k": adaptive_metadata["above_threshold_k"],
            "similarity_drop_threshold": self.similarity_drop_threshold,
            "top_similarities": adaptive_metadata["top_5_similarities"],
        }
        
        logger.debug(f"[AdaptiveRAG] ===== run_single END =====")
        logger.debug(
            f"[AdaptiveRAG] Total latency: {total_latency:.1f}ms, "
            f"Adaptive k: {adaptive_metadata['final_k']}, Result: {called_tool}"
        )
        
        return result
