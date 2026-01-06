"""
RAG (Retrieval-Augmented Generation) methodology for tool calling experiments.

A retrieval-based approach where:
1. Tool descriptions are embedded into a vector space
2. User query is embedded and top-K similar tools are retrieved
3. LLM selects from only the retrieved subset of tools

This reduces context size while maintaining semantic relevance.
"""
import time
import hashlib
from typing import Any, Optional
import numpy as np
from loguru import logger

from src.tools.base import Tool
from src.clients.base import BaseLLMClient
from .base import BaseMethodology, MethodologyResult, StepInfo, StepType, StepBasedMethodology


class RAGMethodology(BaseMethodology):
    """
    RAG methodology for retrieval-based tool selection.
    
    Uses semantic embeddings to retrieve the most relevant tools
    for a given query, then passes only those tools to the LLM.
    
    This reduces context size compared to MCP while maintaining
    high relevance through semantic similarity.
    """
    
    NAME: str = "rag"
    DECLINE_TOOL: str = StepBasedMethodology.DECLINE_TOOL
    CLARIFICATION_TOOL: str = StepBasedMethodology.CLARIFICATION_TOOL
    
    # Default embedding model (small, fast, good quality)
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        allow_no_tool_call: bool = False,
        allow_clarification: bool = False,
        cache_embeddings: bool = True,
        include_params_in_embedding: bool = False,
    ):
        """
        Initialize RAG methodology.
        
        Args:
            embedding_model: Name of sentence-transformers model to use
            top_k: Number of tools to retrieve for each query
            similarity_threshold: Minimum similarity score for retrieval (0.0 = no threshold)
            allow_no_tool_call: If True, add a decline option for cases
                               where no tool call is needed
            allow_clarification: If True, add a clarification option for
                                ambiguous requests
            cache_embeddings: If True, cache tool embeddings between calls
            include_params_in_embedding: If True, include parameter info in tool text for embedding
        """
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.allow_no_tool_call = allow_no_tool_call
        self.allow_clarification = allow_clarification
        self.cache_embeddings = cache_embeddings
        self.include_params_in_embedding = include_params_in_embedding
        
        # Lazy load the embedding model
        self._embedder = None
        self._embedder_loaded = False
        
        # Cache for tool embeddings: hash(tools) -> (tool_names, embeddings)
        self._embeddings_cache: dict[str, tuple[list[str], np.ndarray]] = {}
        
        logger.debug(
            f"[RAG] Initialized with model={embedding_model}, top_k={top_k}, "
            f"threshold={similarity_threshold}, cache={cache_embeddings}, "
            f"allow_clarification={allow_clarification}"
        )
    
    def _load_embedder(self):
        """Lazy load the sentence transformer model."""
        if not self._embedder_loaded:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"[RAG] Loading embedding model: {self.embedding_model_name}")
                self._embedder = SentenceTransformer(self.embedding_model_name)
                self._embedder_loaded = True
                logger.info(f"[RAG] Embedding model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for RAG methodology. "
                    "Install it with: pip install sentence-transformers"
                )
    
    def _get_tool_text(self, tool: Tool) -> str:
        """
        Create searchable text representation of a tool.
        
        This text will be embedded for similarity search.
        
        Args:
            tool: Tool to convert to text
            
        Returns:
            Text representation for embedding
        """
        text_parts = [
            f"Tool: {tool.name}",
            f"Description: {tool.description}",
            f"Category: {tool.category}",
        ]
        
        if tool.tags:
            text_parts.append(f"Tags: {', '.join(tool.tags)}")
        
        if self.include_params_in_embedding and tool.parameters:
            param_texts = []
            for param in tool.parameters:
                param_texts.append(f"{param.name}: {param.description}")
            text_parts.append(f"Parameters: {'; '.join(param_texts)}")
        
        return " | ".join(text_parts)
    
    def _compute_tools_hash(self, tools: list[Tool]) -> str:
        """Compute a hash for a list of tools to use as cache key."""
        tool_strs = sorted([f"{t.name}:{t.description[:50]}" for t in tools])
        combined = "|".join(tool_strs)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _embed_tools(self, tools: list[Tool]) -> tuple[list[str], np.ndarray]:
        """
        Embed all tools, using cache if available.
        
        Args:
            tools: List of tools to embed
            
        Returns:
            Tuple of (tool_names, embeddings_matrix)
        """
        self._load_embedder()
        
        # Check cache
        if self.cache_embeddings:
            cache_key = self._compute_tools_hash(tools)
            if cache_key in self._embeddings_cache:
                logger.debug(f"[RAG] Using cached embeddings for {len(tools)} tools")
                return self._embeddings_cache[cache_key]
        
        # Compute embeddings
        logger.debug(f"[RAG] Computing embeddings for {len(tools)} tools")
        tool_texts = [self._get_tool_text(tool) for tool in tools]
        tool_names = [tool.name for tool in tools]
        
        embeddings = self._embedder.encode(
            tool_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # For cosine similarity via dot product
        )
        
        # Cache result
        if self.cache_embeddings:
            self._embeddings_cache[cache_key] = (tool_names, embeddings)
        
        return tool_names, embeddings
    
    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.
        
        Args:
            query: User query to embed
            
        Returns:
            Query embedding as numpy array
        """
        self._load_embedder()
        
        embedding = self._embedder.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding
    
    def _retrieve_tools(
        self,
        query: str,
        tools: list[Tool],
        top_k: int,
    ) -> tuple[list[Tool], list[float], float]:
        """
        Retrieve top-k most similar tools for a query.
        
        Args:
            query: User query
            tools: All available tools
            top_k: Number of tools to retrieve
            
        Returns:
            Tuple of (retrieved_tools, similarity_scores, retrieval_latency_ms)
        """
        start_time = time.time()
        
        # Embed tools and query
        tool_names, tool_embeddings = self._embed_tools(tools)
        query_embedding = self._embed_query(query)
        
        # Compute similarities (dot product of normalized vectors = cosine similarity)
        similarities = np.dot(tool_embeddings, query_embedding)
        
        # Get top-k indices
        if top_k >= len(tools):
            # Return all tools sorted by similarity
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Get top-k
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            # Sort top-k by similarity (highest first)
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        # Apply similarity threshold
        if self.similarity_threshold > 0:
            top_indices = [
                idx for idx in top_indices 
                if similarities[idx] >= self.similarity_threshold
            ]
        
        # Build results
        retrieved_tools = []
        scores = []
        
        # Create name -> tool mapping
        name_to_tool = {tool.name: tool for tool in tools}
        
        for idx in top_indices:
            tool_name = tool_names[idx]
            score = float(similarities[idx])
            
            if tool_name in name_to_tool:
                retrieved_tools.append(name_to_tool[tool_name])
                scores.append(score)
        
        retrieval_latency = (time.time() - start_time) * 1000
        
        logger.debug(
            f"[RAG] Retrieved {len(retrieved_tools)} tools in {retrieval_latency:.1f}ms"
        )
        for i, (tool, score) in enumerate(zip(retrieved_tools[:5], scores[:5])):
            logger.debug(f"[RAG]   {i+1}. {tool.name} (score: {score:.3f})")
        
        return retrieved_tools, scores, retrieval_latency
    
    def run_single(
        self,
        prompt: str,
        tools: list[Tool],
        client: BaseLLMClient,
        system_instruction: Optional[str] = None,
    ) -> MethodologyResult:
        """
        Run RAG methodology for a single prompt.
        
        Steps:
        1. Embed query and retrieve top-K similar tools
        2. Pass only retrieved tools to LLM
        3. LLM selects from the subset
        
        Args:
            prompt: User prompt to process
            tools: All available tools
            client: LLM client to use
            system_instruction: Optional system instruction
            
        Returns:
            MethodologyResult with tool selection and retrieval metadata
        """
        logger.debug(f"[RAG] ===== run_single START =====")
        logger.debug(f"[RAG] Prompt: {prompt[:150]}...")
        logger.debug(f"[RAG] Total tools available: {len(tools)}")
        logger.debug(f"[RAG] Top-K: {self.top_k}")
        
        # Step 1: Retrieve relevant tools
        retrieved_tools, similarity_scores, retrieval_latency = self._retrieve_tools(
            query=prompt,
            tools=tools,
            top_k=self.top_k,
        )
        
        logger.debug(f"[RAG] Retrieved {len(retrieved_tools)} tools")
        
        if not retrieved_tools:
            logger.warning("[RAG] No tools retrieved! Check similarity threshold.")
            return MethodologyResult(
                success=False,
                called_tool=None,
                called_args=None,
                all_calls=[],
                latency_ms=retrieval_latency,
                error="No tools retrieved above similarity threshold",
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
            logger.debug(f"[RAG] Added decline tool, total tools: {len(tools_to_use)}")
        
        # Optionally add clarification pseudo-tool
        if self.allow_clarification:
            from src.tools.base import ToolParameter
            clarification_tool = Tool(
                name=self.CLARIFICATION_TOOL,
                description="Request clarification when the user's request is ambiguous "
                           "and could match multiple tools. Use this when you cannot "
                           "determine which tool the user wants with high confidence.",
                category="system",
                parameters=[
                    ToolParameter(
                        name="question",
                        type="string",
                        description="The clarifying question to ask the user",
                        required=True,
                    ),
                    ToolParameter(
                        name="candidate_tools",
                        type="array",
                        description="List of tool names that could potentially match the request",
                        required=True,
                    ),
                ],
                tags=["system"],
                complexity="simple",
            )
            tools_to_use.append(clarification_tool)
            logger.debug(f"[RAG] Added clarification tool, total tools: {len(tools_to_use)}")
        
        # Step 2: Call LLM with retrieved tools
        logger.debug(f"[RAG] Calling LLM with {len(tools_to_use)} tools...")
        
        call_result = client.call_with_tools(
            prompt=prompt,
            tools=tools_to_use,
            system_instruction=system_instruction,
        )
        
        logger.debug(f"[RAG] LLM call completed in {call_result.latency_ms:.1f}ms")
        logger.debug(f"[RAG] Success: {call_result.success}")
        logger.debug(f"[RAG] Called tool: {call_result.called_tool}")
        
        # Total latency = retrieval + LLM call
        total_latency = retrieval_latency + call_result.latency_ms
        
        # Check for decline or clarification
        declined = False
        clarification_requested = False
        clarification_question = None
        candidate_tools = []
        called_tool = call_result.called_tool
        step_type = StepType.SELECT_TOOL
        
        if called_tool == self.DECLINE_TOOL:
            declined = True
            called_tool = None
            step_type = StepType.DECLINE
            logger.debug(f"[RAG] LLM declined to call any tool")
        elif called_tool == self.CLARIFICATION_TOOL:
            clarification_requested = True
            called_tool = None
            step_type = StepType.CLARIFICATION
            # Extract clarification details from args
            args = call_result.called_args or {}
            clarification_question = args.get("question", "")
            candidate_tools = args.get("candidate_tools", [])
            if isinstance(candidate_tools, str):
                candidate_tools = [candidate_tools]
            logger.debug(f"[RAG] LLM requested clarification: {clarification_question}")
            logger.debug(f"[RAG] Candidate tools: {candidate_tools}")
        
        # Create step info for retrieval
        retrieval_step = StepInfo(
            step_number=1,
            step_type=StepType.SELECT_CATEGORY,  # Using this for "retrieval" phase
            selection=f"retrieved_{len(retrieved_tools)}_tools",
            latency_ms=retrieval_latency,
            raw_response=None,
            error=None,
        )
        
        # Create step info for LLM selection
        selection_step = StepInfo(
            step_number=2,
            step_type=step_type,
            selection=call_result.called_tool,
            latency_ms=call_result.latency_ms,
            raw_response=call_result.raw_response,
            error=call_result.error,
        )
        
        # Determine the category of the called tool (if any)
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
            clarification_requested=clarification_requested,
            clarification_question=clarification_question,
            candidate_tools=candidate_tools,
        )
        
        # Store additional RAG-specific metadata in raw_response or separate field
        # This can be accessed for detailed analysis
        result._rag_metadata = {
            "retrieved_tools": [t.name for t in retrieved_tools],
            "similarity_scores": similarity_scores,
            "retrieval_latency_ms": retrieval_latency,
            "top_k_requested": self.top_k,
            "tools_retrieved": len(retrieved_tools),
            "similarity_threshold": self.similarity_threshold,
        }
        
        logger.debug(f"[RAG] ===== run_single END =====")
        logger.debug(f"[RAG] Total latency: {total_latency:.1f}ms")
        logger.debug(f"[RAG] Result: {called_tool}, success={result.success}")
        
        return result
    
    def get_retrieval_stats(self, result: MethodologyResult) -> Optional[dict[str, Any]]:
        """
        Get RAG-specific retrieval statistics from a result.
        
        Args:
            result: MethodologyResult from run_single
            
        Returns:
            Dictionary with retrieval stats, or None if not available
        """
        if hasattr(result, '_rag_metadata'):
            return result._rag_metadata
        return None
