"""
Hybrid RAG + Clustering methodology for tool calling experiments.

Combines the strengths of both approaches:
1. Use semantic embeddings to select top-K categories (not individual tools)
2. Retrieve all tools from selected categories
3. LLM selects specific tool from narrowed set

This provides more robust category selection than pure clustering
(semantic similarity vs. name matching) while reducing context
compared to full MCP.
"""
import time
import hashlib
from typing import Any, Optional
from collections import defaultdict
import numpy as np
from loguru import logger

from src.tools.base import Tool
from src.clients.base import BaseLLMClient
from .base import BaseMethodology, MethodologyResult, StepInfo, StepType, StepBasedMethodology


TOOLS_FOLDER = "tools_xlam"

class HybridMethodology(BaseMethodology):
    """
    Hybrid RAG + Clustering methodology.
    
    Uses semantic embeddings to select categories, then passes
    all tools from those categories to the LLM for final selection.
    
    This combines the contextual understanding of embeddings with
    the category-based organization of clustering.
    """
    
    NAME: str = "hybrid"
    DECLINE_TOOL: str = StepBasedMethodology.DECLINE_TOOL
    CLARIFICATION_TOOL: str = StepBasedMethodology.CLARIFICATION_TOOL
    
    # Default embedding model
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_k_categories: int = 3,
        allow_no_tool_call: bool = False,
        allow_clarification: bool = False,
        cache_embeddings: bool = True,
        category_embedding_strategy: str = "mean",  # "mean" or "description"
    ):
        """
        Initialize Hybrid methodology.
        
        Args:
            embedding_model: Name of sentence-transformers model to use
            top_k_categories: Number of categories to retrieve
            allow_no_tool_call: If True, add a decline option
            allow_clarification: If True, add a clarification option
            cache_embeddings: If True, cache embeddings between calls
            category_embedding_strategy: How to compute category embeddings:
                - "mean": Average of tool embeddings in category
                - "description": Embed category description directly
        """
        self.embedding_model_name = embedding_model
        self.top_k_categories = top_k_categories
        self.allow_no_tool_call = allow_no_tool_call
        self.allow_clarification = allow_clarification
        self.cache_embeddings = cache_embeddings
        self.category_embedding_strategy = category_embedding_strategy
        
        # Lazy load the embedding model
        self._embedder = None
        self._embedder_loaded = False
        
        # Cache for embeddings
        self._tool_embeddings_cache: dict[str, tuple[list[str], np.ndarray]] = {}
        self._category_embeddings_cache: dict[str, dict[str, np.ndarray]] = {}
        
        # Category descriptions (loaded lazily)
        self._category_descriptions: Optional[dict[str, str]] = None
        
        logger.debug(
            f"[Hybrid] Initialized with model={embedding_model}, "
            f"top_k_categories={top_k_categories}, strategy={category_embedding_strategy}"
        )
    
    def _load_embedder(self):
        """Lazy load the sentence transformer model."""
        if not self._embedder_loaded:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"[Hybrid] Loading embedding model: {self.embedding_model_name}")
                self._embedder = SentenceTransformer(self.embedding_model_name)
                self._embedder_loaded = True
                logger.info(f"[Hybrid] Embedding model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for Hybrid methodology. "
                    "Install it with: pip install sentence-transformers"
                )
    
    def _load_category_descriptions(self) -> dict[str, str]:
        """Load category descriptions from categories.yaml."""
        if self._category_descriptions is not None:
            return self._category_descriptions
        
        from pathlib import Path
        import yaml
        
        project_root = Path(__file__).parent.parent.parent
        categories_file = project_root / TOOLS_FOLDER / "categories.yaml"
        
        descriptions = {}
        if categories_file.exists():
            try:
                with open(categories_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data and "categories" in data:
                        descriptions = {cat["name"]: cat["description"] for cat in data["categories"]}
            except Exception as e:
                logger.warning(f"[Hybrid] Failed to load categories file: {e}")
        
        self._category_descriptions = descriptions
        return descriptions
    
    def _get_tool_text(self, tool: Tool) -> str:
        """Create searchable text representation of a tool."""
        text_parts = [
            f"Tool: {tool.name}",
            f"Description: {tool.description}",
            f"Category: {tool.category}",
        ]
        if tool.tags:
            text_parts.append(f"Tags: {', '.join(tool.tags)}")
        return " | ".join(text_parts)
    
    def _compute_tools_hash(self, tools: list[Tool]) -> str:
        """Compute a hash for a list of tools to use as cache key."""
        tool_strs = sorted([f"{t.name}:{t.description[:50]}" for t in tools])
        combined = "|".join(tool_strs)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string."""
        self._load_embedder()
        embedding = self._embedder.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding
    
    def _embed_tools(self, tools: list[Tool]) -> tuple[list[str], np.ndarray]:
        """Embed all tools, using cache if available."""
        self._load_embedder()
        
        if self.cache_embeddings:
            cache_key = self._compute_tools_hash(tools)
            if cache_key in self._tool_embeddings_cache:
                return self._tool_embeddings_cache[cache_key]
        
        tool_texts = [self._get_tool_text(tool) for tool in tools]
        tool_names = [tool.name for tool in tools]
        
        embeddings = self._embedder.encode(
            tool_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        
        if self.cache_embeddings:
            self._tool_embeddings_cache[cache_key] = (tool_names, embeddings)
        
        return tool_names, embeddings
    
    def _compute_category_embeddings(
        self,
        tools: list[Tool],
    ) -> dict[str, np.ndarray]:
        """
        Compute embeddings for each category.
        
        Strategy depends on category_embedding_strategy:
        - "mean": Average embedding of all tools in the category
        - "description": Embed the category description directly
        
        Args:
            tools: List of all tools
            
        Returns:
            Dict mapping category name to embedding vector
        """
        self._load_embedder()
        
        # Check cache
        if self.cache_embeddings:
            cache_key = self._compute_tools_hash(tools)
            if cache_key in self._category_embeddings_cache:
                return self._category_embeddings_cache[cache_key]
        
        category_embeddings = {}
        
        if self.category_embedding_strategy == "description":
            # Embed category descriptions directly
            descriptions = self._load_category_descriptions()
            categories = set(tool.category for tool in tools)
            
            for category in categories:
                desc = descriptions.get(category, f"Tools for {category.replace('_', ' ')}")
                embedding = self._embedder.encode(
                    desc,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                category_embeddings[category] = embedding
        else:
            # "mean" strategy: average tool embeddings per category
            tool_names, tool_embeddings = self._embed_tools(tools)
            name_to_idx = {name: idx for idx, name in enumerate(tool_names)}
            
            # Group tools by category
            category_tools: dict[str, list[Tool]] = defaultdict(list)
            for tool in tools:
                category_tools[tool.category].append(tool)
            
            # Compute mean embedding per category
            for category, cat_tools in category_tools.items():
                indices = [name_to_idx[t.name] for t in cat_tools if t.name in name_to_idx]
                if indices:
                    cat_embeddings = tool_embeddings[indices]
                    mean_embedding = np.mean(cat_embeddings, axis=0)
                    # Normalize the mean
                    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
                    category_embeddings[category] = mean_embedding
        
        # Cache result
        if self.cache_embeddings:
            self._category_embeddings_cache[cache_key] = category_embeddings
        
        logger.debug(f"[Hybrid] Computed embeddings for {len(category_embeddings)} categories")
        return category_embeddings
    
    def _retrieve_categories(
        self,
        query: str,
        category_embeddings: dict[str, np.ndarray],
        top_k: int,
    ) -> tuple[list[str], list[float], float]:
        """
        Retrieve top-k most similar categories for a query.
        
        Args:
            query: User query
            category_embeddings: Dict of category name -> embedding
            top_k: Number of categories to retrieve
            
        Returns:
            Tuple of (category_names, similarity_scores, retrieval_latency_ms)
        """
        start_time = time.time()
        
        query_embedding = self._embed_query(query)
        
        # Compute similarities
        categories = list(category_embeddings.keys())
        embeddings = np.array([category_embeddings[cat] for cat in categories])
        similarities = np.dot(embeddings, query_embedding)
        
        # Get top-k
        if top_k >= len(categories):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        top_categories = [categories[idx] for idx in top_indices]
        top_scores = [float(similarities[idx]) for idx in top_indices]
        
        retrieval_latency = (time.time() - start_time) * 1000
        
        logger.debug(f"[Hybrid] Retrieved {len(top_categories)} categories in {retrieval_latency:.1f}ms")
        for cat, score in zip(top_categories[:5], top_scores[:5]):
            logger.debug(f"[Hybrid]   {cat}: {score:.3f}")
        
        return top_categories, top_scores, retrieval_latency
    
    def run_single(
        self,
        prompt: str,
        tools: list[Tool],
        client: BaseLLMClient,
        system_instruction: Optional[str] = None,
    ) -> MethodologyResult:
        """
        Run Hybrid methodology for a single prompt.
        
        Steps:
        1. Compute category embeddings (cached)
        2. Retrieve top-K categories by semantic similarity
        3. Get all tools from selected categories
        4. LLM selects from narrowed tool set
        
        Args:
            prompt: User prompt to process
            tools: All available tools
            client: LLM client to use
            system_instruction: Optional system instruction
            
        Returns:
            MethodologyResult with tool selection and metadata
        """
        logger.debug(f"[Hybrid] ===== run_single START =====")
        logger.debug(f"[Hybrid] Prompt: {prompt[:150]}...")
        logger.debug(f"[Hybrid] Total tools: {len(tools)}, Top-K categories: {self.top_k_categories}")
        
        # Step 1: Compute category embeddings
        category_embeddings = self._compute_category_embeddings(tools)
        
        # Step 2: Retrieve top-K categories
        top_categories, category_scores, retrieval_latency = self._retrieve_categories(
            query=prompt,
            category_embeddings=category_embeddings,
            top_k=self.top_k_categories,
        )
        
        # Step 3: Get tools from selected categories
        relevant_tools = [t for t in tools if t.category in top_categories]
        logger.debug(f"[Hybrid] Selected {len(relevant_tools)} tools from {len(top_categories)} categories")
        
        if not relevant_tools:
            logger.warning("[Hybrid] No tools found in selected categories!")
            return MethodologyResult(
                success=False,
                called_tool=None,
                called_args=None,
                all_calls=[],
                latency_ms=retrieval_latency,
                error="No tools found in selected categories",
                raw_response=None,
                model="",
                provider="",
                methodology=self.NAME,
                steps=[],
                categories_selected=top_categories,
                backtrack_count=0,
                declined_tool_call=False,
                final_category=None,
                # No tokens used since no LLM call was made
                tokens_input=None,
                tokens_output=None,
                tokens_total=None,
            )
        
        # Optionally add decline pseudo-tool
        tools_to_use = list(relevant_tools)
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
        
        # Step 4: Call LLM with relevant tools
        logger.debug(f"[Hybrid] Calling LLM with {len(tools_to_use)} tools...")
        
        call_result = client.call_with_tools(
            prompt=prompt,
            tools=tools_to_use,
            system_instruction=system_instruction,
        )
        
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
        elif called_tool == self.CLARIFICATION_TOOL:
            clarification_requested = True
            called_tool = None
            step_type = StepType.CLARIFICATION
            args = call_result.called_args or {}
            clarification_question = args.get("question", "")
            candidate_tools = args.get("candidate_tools", [])
            if isinstance(candidate_tools, str):
                candidate_tools = [candidate_tools]
        
        # Create step info
        retrieval_step = StepInfo(
            step_number=1,
            step_type=StepType.SELECT_CATEGORY,
            selection=f"categories:{','.join(top_categories)}",
            latency_ms=retrieval_latency,
            raw_response=None,
            error=None,
        )
        
        selection_step = StepInfo(
            step_number=2,
            step_type=step_type,
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
            categories_selected=top_categories,
            backtrack_count=0,
            declined_tool_call=declined,
            final_category=final_category,
            clarification_requested=clarification_requested,
            clarification_question=clarification_question,
            candidate_tools=candidate_tools,
            # Token usage from the LLM call
            tokens_input=call_result.tokens_input,
            tokens_output=call_result.tokens_output,
            tokens_total=call_result.tokens_total,
        )
        
        # Store hybrid-specific metadata
        result._hybrid_metadata = {
            "top_categories": top_categories,
            "category_scores": category_scores,
            "tools_in_context": len(relevant_tools),
            "retrieval_latency_ms": retrieval_latency,
            "category_embedding_strategy": self.category_embedding_strategy,
        }
        
        logger.debug(f"[Hybrid] ===== run_single END =====")
        logger.debug(f"[Hybrid] Total latency: {total_latency:.1f}ms, Result: {called_tool}")
        
        return result
