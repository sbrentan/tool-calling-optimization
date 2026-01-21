"""
Clustering methodology for tool calling experiments.

A multi-step approach where:
1. LLM first selects a category/cluster from available categories
2. LLM then sees tools within that category and selects one
3. LLM can backtrack to select a different category if needed
4. LLM can decline to call any tool
"""
import time
from pathlib import Path
from typing import Any, Optional
import yaml
from loguru import logger

from src.tools.base import Tool
from src.clients.base import BaseLLMClient
from .base import (
    StepBasedMethodology,
    MethodologyResult,
    StepInfo,
    StepType,
)


# Default category descriptions if categories.yaml doesn't exist
DEFAULT_CATEGORY_DESCRIPTIONS = {
    "ai_ml_operations": "AI and machine learning operations including text generation, classification, embeddings, and model inference",
    "analytics_operations": "Data analytics, statistics, reporting, and business intelligence operations",
    "audio_operations": "Audio processing, transcription, text-to-speech, and sound manipulation",
    "authentication_operations": "User authentication, login, logout, password management, and identity verification",
    "backup_operations": "Data backup, restore, snapshot creation, and disaster recovery",
    "caching_operations": "Cache management, storing and retrieving cached data, cache invalidation",
    "calendar_operations": "Calendar management, event scheduling, reminders, and availability checking",
    "cloud_storage_operations": "Cloud storage operations like upload, download, and file management in cloud services",
    "compliance_operations": "Regulatory compliance, audit logging, policy enforcement, and compliance reporting",
    "container_operations": "Container management, Docker operations, Kubernetes, and orchestration",
    "data_operations": "Data transformation, validation, parsing, and format conversion",
    "database_operations": "Database queries, CRUD operations, schema management, and data retrieval",
    "devops_operations": "CI/CD pipelines, deployment, infrastructure management, and automation",
    "document_operations": "Document creation, editing, conversion, and management",
    "ecommerce_operations": "E-commerce operations including orders, inventory, products, and shopping carts",
    "email_operations": "Email sending, receiving, management, and email-related utilities",
    "file_operations": "File system operations like read, write, copy, move, and delete files",
    "healthcare_operations": "Healthcare-related operations, patient records, medical data management",
    "image_operations": "Image processing, manipulation, resizing, and format conversion",
    "iot_operations": "Internet of Things device management, sensor data, and device control",
    "location_operations": "Geolocation, maps, geocoding, and location-based services",
    "logging_operations": "Application logging, log management, and log analysis",
    "math_operations": "Mathematical calculations, statistics, and numerical operations",
    "messaging_operations": "Instant messaging, chat, and real-time communication",
    "monitoring_operations": "System monitoring, health checks, metrics collection, and alerting",
    "networking_operations": "Network operations, DNS, HTTP requests, and connectivity",
    "notification_operations": "Push notifications, alerts, and notification management",
    "payment_operations": "Payment processing, transactions, billing, and financial operations",
    "queue_operations": "Message queue management, job queuing, and async task handling",
    "rate_limiting_operations": "Rate limiting, throttling, and request quota management",
    "scheduling_operations": "Job scheduling, cron tasks, and timed execution",
    "search_operations": "Search functionality, indexing, and query operations",
    "secrets_operations": "Secret management, encryption keys, and sensitive data handling",
    "security_operations": "Security operations, encryption, hashing, and access control",
    "social_media_operations": "Social media integration, posting, and social platform APIs",
    "system_operations": "System-level operations, process management, and OS interactions",
    "testing_operations": "Testing utilities, test execution, and quality assurance",
    "text_operations": "Text processing, string manipulation, and text analysis",
    "video_operations": "Video processing, transcoding, and video manipulation",
    "web_operations": "Web scraping, HTTP operations, and web-related utilities",
    "workflow_operations": "Workflow management, state machines, and process orchestration",
}


class ClusteringMethodology(StepBasedMethodology):
    """
    Clustering methodology for hierarchical tool selection.
    
    The LLM first selects a category/cluster, then selects a specific
    tool within that category. It can backtrack if needed.
    
    This reduces initial context by only showing category names first,
    then showing full tool details for the selected category.
    """
    
    NAME: str = "clustering"
    
    # Pseudo-tool prefix for category selection
    CATEGORY_PREFIX: str = "select_category_"
    
    def __init__(
        self,
        max_steps: Optional[int] = None,
        allow_backtrack: bool = True,
        allow_decline: bool = False,
        allow_clarification: bool = False,
        categories_file: Optional[str] = None,
    ):
        """
        Initialize clustering methodology.
        
        Args:
            max_steps: Maximum steps before forcing termination
            allow_backtrack: Whether to allow backtracking to category selection
            allow_decline: Whether to allow declining to call any tool
            allow_clarification: Whether to allow requesting clarification
            categories_file: Path to YAML file with category descriptions
        """
        super().__init__(max_steps=max_steps)
        self.allow_backtrack = allow_backtrack
        self.allow_decline = allow_decline
        self.allow_clarification = allow_clarification
        self.category_descriptions = self._load_category_descriptions(categories_file)
        logger.debug(f"[Clustering] Initialized with max_steps={max_steps}, allow_backtrack={allow_backtrack}, allow_decline={allow_decline}, allow_clarification={allow_clarification}")
        logger.debug(f"[Clustering] Loaded {len(self.category_descriptions)} category descriptions")
    
    def _load_category_descriptions(self, categories_file: Optional[str]) -> dict[str, str]:
        """Load category descriptions from file or use defaults."""
        if categories_file is None:
            # Try default location
            project_root = Path(__file__).parent.parent.parent
            categories_file = project_root / "tools" / "categories.yaml"
        else:
            categories_file = Path(categories_file)
        
        if categories_file.exists():
            try:
                with open(categories_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data and "categories" in data:
                        return {cat["name"]: cat["description"] for cat in data["categories"]}
            except Exception as e:
                logger.warning(f"Failed to load categories file: {e}")
        
        return DEFAULT_CATEGORY_DESCRIPTIONS.copy()
    
    def _get_category_tool_name(self, category: str) -> str:
        """Get the pseudo-tool name for a category."""
        return f"{self.CATEGORY_PREFIX}{category}"
    
    def _extract_category_from_tool_name(self, tool_name: str) -> Optional[str]:
        """Extract category name from pseudo-tool name."""
        if tool_name.startswith(self.CATEGORY_PREFIX):
            return tool_name[len(self.CATEGORY_PREFIX):]
        return None
    
    def get_initial_options(self, tools: list[Tool]) -> tuple[list[Tool], str]:
        """
        Get category selection pseudo-tools.
        
        Args:
            tools: All available tools
            
        Returns:
            Tuple of (category pseudo-tools, system instruction)
        """
        # Group tools by category
        categories = set(tool.category for tool in tools)
        logger.debug(f"[Clustering] get_initial_options: Found {len(categories)} categories from {len(tools)} tools")
        
        # Create pseudo-tools for each category
        category_tools = []
        for category in sorted(categories):
            description = self.category_descriptions.get(
                category,
                f"Tools for {category.replace('_', ' ')}"
            )
            # Count tools in category
            tool_count = sum(1 for t in tools if t.category == category)
            full_description = f"{description} ({tool_count} tools available)"
            
            pseudo_tool = self.create_pseudo_tool(
                name=self._get_category_tool_name(category),
                description=full_description,
                category="category_selection",
            )
            category_tools.append(pseudo_tool)
            logger.debug(f"[Clustering]   Category '{category}': {tool_count} tools")
        
        # Add decline option if enabled
        if self.allow_decline:
            category_tools.append(self.get_decline_tool())
            logger.debug(f"[Clustering]   Added decline option")
        
        # Add clarification option if enabled
        if self.allow_clarification:
            category_tools.append(self.get_clarification_tool())
            logger.debug(f"[Clustering]   Added clarification option")
        
        system_instruction = (
            "You are selecting from tool categories. Choose the category that best matches "
            "the user's request. After selecting a category, you will see the specific tools "
            "available in that category."
        )
        
        logger.debug(f"[Clustering] Presenting {len(category_tools)} category options")
        return category_tools, system_instruction
    
    def get_category_tools(
        self,
        category: str,
        tools: list[Tool],
    ) -> tuple[list[Tool], str]:
        """
        Get tools for a specific category plus control options.
        
        Args:
            category: Selected category
            tools: All available tools
            
        Returns:
            Tuple of (tools in category + control tools, system instruction)
        """
        # Filter tools by category
        category_tools = [t for t in tools if t.category == category]
        logger.debug(f"[Clustering] get_category_tools: Found {len(category_tools)} tools in category '{category}'")
        logger.debug(f"[Clustering]   Tool names: {[t.name for t in category_tools]}")
        
        # Add control pseudo-tools
        result_tools = list(category_tools)
        
        if self.allow_backtrack:
            result_tools.append(self.get_backtrack_tool())
        
        if self.allow_decline:
            result_tools.append(self.get_decline_tool())
        
        if self.allow_clarification:
            result_tools.append(self.get_clarification_tool())
        
        system_instruction = (
            f"You are now viewing tools in the '{category}' category. "
            f"Select the tool that best matches the user's request."
        )
        if self.allow_backtrack:
            system_instruction += (
                f" If none of these tools match, use '{self.BACKTRACK_TOOL}' "
                "to go back and select a different category."
            )
        
        return result_tools, system_instruction
    
    def process_selection(
        self,
        selection: str,
        tools: list[Tool],
        current_state: dict[str, Any],
    ) -> tuple[Optional[list[Tool]], StepType, dict[str, Any]]:
        """
        Process a selection and determine next step.
        
        Args:
            selection: Name of selected tool/option
            tools: All available tools
            current_state: Current methodology state
            
        Returns:
            Tuple of (next options or None if done, step type, updated state)
        """
        logger.debug(f"[Clustering] process_selection: selection='{selection}', phase={current_state.get('phase')}")
        
        # Check for backtrack
        if selection == self.BACKTRACK_TOOL:
            logger.debug(f"[Clustering]   -> BACKTRACK: Going back to category selection")
            # Go back to category selection
            category_tools, _ = self.get_initial_options(tools)
            new_state = {
                **current_state,
                "phase": "category_selection",
                "current_category": None,
            }
            return category_tools, StepType.BACKTRACK, new_state
        
        # Check for decline
        if selection == self.DECLINE_TOOL:
            logger.debug(f"[Clustering]   -> DECLINE: LLM declined to call any tool")
            return None, StepType.DECLINE, current_state
        
        # Check for clarification
        if selection == self.CLARIFICATION_TOOL:
            logger.debug(f"[Clustering]   -> CLARIFICATION: LLM requested clarification")
            return None, StepType.CLARIFICATION, current_state
        
        # Check if this is a category selection
        category = self._extract_category_from_tool_name(selection)
        if category is not None:
            logger.debug(f"[Clustering]   -> SELECT_CATEGORY: Selected category '{category}'")
            # Move to tool selection within this category
            category_tools, _ = self.get_category_tools(category, tools)
            new_state = {
                **current_state,
                "phase": "tool_selection",
                "current_category": category,
            }
            return category_tools, StepType.SELECT_CATEGORY, new_state
        
        # This is a tool selection - we're done
        logger.debug(f"[Clustering]   -> SELECT_TOOL: Selected tool '{selection}'")
        return None, StepType.SELECT_TOOL, current_state
    
    def run_single(
        self,
        prompt: str,
        tools: list[Tool],
        client: BaseLLMClient,
        system_instruction: Optional[str] = None,
    ) -> MethodologyResult:
        """
        Run clustering methodology for a single prompt.
        
        Multi-step process:
        1. Present category options
        2. On category selection, present tools in that category
        3. Allow backtracking or tool selection
        4. Repeat until tool selected, declined, or max steps reached
        
        Args:
            prompt: User prompt to process
            tools: All available tools
            client: LLM client to use
            system_instruction: Optional base system instruction
            
        Returns:
            MethodologyResult with tool selection and step details
        """
        logger.debug(f"[Clustering] ===== run_single START =====")
        logger.debug(f"[Clustering] Prompt: {prompt[:150]}...")
        logger.debug(f"[Clustering] Total tools available: {len(tools)}")
        logger.debug(f"[Clustering] Max steps: {self.max_steps}")
        
        steps: list[StepInfo] = []
        categories_selected: list[str] = []
        backtrack_count = 0
        total_latency = 0.0
        
        # Token tracking across all steps
        total_tokens_input = 0
        total_tokens_output = 0
        
        # Initialize state
        state = {
            "phase": "category_selection",
            "current_category": None,
        }
        
        # Get initial options (categories)
        current_options, step_system = self.get_initial_options(tools)
        logger.debug(f"[Clustering] Initial phase: category_selection with {len(current_options)} options")
        
        # Combine system instructions
        if system_instruction:
            full_system = f"{system_instruction}\n\n{step_system}"
        else:
            full_system = step_system
        logger.debug(f"[Clustering] System instruction: {full_system[:200]}...")
        
        # Iterative selection loop
        for step_num in range(1, self.max_steps + 1):
            logger.debug(f"[Clustering] ----- Step {step_num}/{self.max_steps} -----")
            logger.debug(f"[Clustering] Phase: {state['phase']}, Current category: {state['current_category']}")
            logger.debug(f"[Clustering] Options presented: {[t.name for t in current_options]}")
            
            start_time = time.time()
            
            # Make API call
            logger.debug(f"[Clustering] Making API call to {client.PROVIDER_NAME}...")
            call_result = client.call_with_tools(
                prompt=prompt,
                tools=current_options,
                system_instruction=full_system,
            )
            
            step_latency = (time.time() - start_time) * 1000
            total_latency += call_result.latency_ms or step_latency
            
            # Aggregate token usage across all steps
            if call_result.tokens_input is not None:
                total_tokens_input += call_result.tokens_input
            if call_result.tokens_output is not None:
                total_tokens_output += call_result.tokens_output
            
            logger.debug(f"[Clustering] API call completed in {call_result.latency_ms:.1f}ms")
            logger.debug(f"[Clustering] Success: {call_result.success}")
            logger.debug(f"[Clustering] Called tool: {call_result.called_tool}")
            logger.debug(f"[Clustering] Called args: {call_result.called_args}")
            if call_result.error:
                logger.debug(f"[Clustering] Error: {call_result.error}")
            if call_result.all_calls:
                logger.debug(f"[Clustering] All calls: {call_result.all_calls}")
            
            # Get selection
            selection = call_result.called_tool
            
            if selection is None:
                logger.debug(f"[Clustering] ERROR: No tool selected by LLM")
                # No tool called - treat as error or no selection
                step = StepInfo(
                    step_number=step_num,
                    step_type=StepType.ERROR,
                    selection=None,
                    latency_ms=call_result.latency_ms,
                    raw_response=call_result.raw_response,
                    error=call_result.error or "No tool selected",
                )
                steps.append(step)
                
                logger.debug(f"[Clustering] ===== run_single END (error: no selection) =====")
                return MethodologyResult(
                    success=False,
                    called_tool=None,
                    called_args=None,
                    all_calls=[],
                    latency_ms=total_latency,
                    error="No tool selected in step",
                    raw_response=call_result.raw_response,
                    model=call_result.model,
                    provider=call_result.provider,
                    methodology=self.NAME,
                    steps=steps,
                    categories_selected=categories_selected,
                    backtrack_count=backtrack_count,
                    declined_tool_call=False,
                    final_category=state.get("current_category"),
                    # Token usage aggregated across all steps
                    tokens_input=total_tokens_input if total_tokens_input > 0 else None,
                    tokens_output=total_tokens_output if total_tokens_output > 0 else None,
                    tokens_total=(total_tokens_input + total_tokens_output) if (total_tokens_input > 0 or total_tokens_output > 0) else None,
                )
            
            # Process the selection
            next_options, step_type, state = self.process_selection(
                selection, tools, state
            )
            
            # Track category selections
            category = self._extract_category_from_tool_name(selection)
            if category is not None:
                categories_selected.append(category)
            
            # Track backtracks
            if step_type == StepType.BACKTRACK:
                backtrack_count += 1
            
            # Record step
            step = StepInfo(
                step_number=step_num,
                step_type=step_type,
                selection=selection,
                latency_ms=call_result.latency_ms,
                raw_response=call_result.raw_response,
                error=call_result.error,
            )
            steps.append(step)
            logger.debug(f"[Clustering] Step {step_num} recorded: type={step_type.value}, selection={selection}")
            
            # Check if we're done
            if next_options is None:
                # Done - either tool selected, declined, or clarification requested
                if step_type == StepType.DECLINE:
                    logger.debug(f"[Clustering] ===== run_single END (declined) =====")
                    return MethodologyResult(
                        success=True,
                        called_tool=None,
                        called_args=None,
                        all_calls=[],
                        latency_ms=total_latency,
                        error=None,
                        raw_response=call_result.raw_response,
                        model=call_result.model,
                        provider=call_result.provider,
                        methodology=self.NAME,
                        steps=steps,
                        categories_selected=categories_selected,
                        backtrack_count=backtrack_count,
                        declined_tool_call=True,
                        final_category=state.get("current_category"),
                        # Token usage aggregated across all steps
                        tokens_input=total_tokens_input if total_tokens_input > 0 else None,
                        tokens_output=total_tokens_output if total_tokens_output > 0 else None,
                        tokens_total=(total_tokens_input + total_tokens_output) if (total_tokens_input > 0 or total_tokens_output > 0) else None,
                    )
                elif step_type == StepType.CLARIFICATION:
                    # Extract clarification details from args
                    args = call_result.called_args or {}
                    clarification_question = args.get("question", "")
                    candidate_tools = args.get("candidate_tools", [])
                    if isinstance(candidate_tools, str):
                        candidate_tools = [candidate_tools]
                    logger.debug(f"[Clustering] ===== run_single END (clarification requested) =====")
                    logger.debug(f"[Clustering] Question: {clarification_question}")
                    logger.debug(f"[Clustering] Candidates: {candidate_tools}")
                    return MethodologyResult(
                        success=True,
                        called_tool=None,
                        called_args=None,
                        all_calls=[],
                        latency_ms=total_latency,
                        error=None,
                        raw_response=call_result.raw_response,
                        model=call_result.model,
                        provider=call_result.provider,
                        methodology=self.NAME,
                        steps=steps,
                        categories_selected=categories_selected,
                        backtrack_count=backtrack_count,
                        declined_tool_call=False,
                        final_category=state.get("current_category"),
                        clarification_requested=True,
                        clarification_question=clarification_question,
                        candidate_tools=candidate_tools,
                        # Token usage aggregated across all steps
                        tokens_input=total_tokens_input if total_tokens_input > 0 else None,
                        tokens_output=total_tokens_output if total_tokens_output > 0 else None,
                        tokens_total=(total_tokens_input + total_tokens_output) if (total_tokens_input > 0 or total_tokens_output > 0) else None,
                    )
                else:
                    # Tool selected
                    logger.debug(f"[Clustering] ===== run_single END (tool selected: {selection}) =====")
                    logger.debug(f"[Clustering] Final category: {state.get('current_category')}")
                    logger.debug(f"[Clustering] Categories visited: {categories_selected}")
                    logger.debug(f"[Clustering] Backtracks: {backtrack_count}")
                    logger.debug(f"[Clustering] Total steps: {len(steps)}")
                    logger.debug(f"[Clustering] Total latency: {total_latency:.1f}ms")
                    return MethodologyResult(
                        success=True,
                        called_tool=selection,
                        called_args=call_result.called_args,
                        all_calls=call_result.all_calls,
                        latency_ms=total_latency,
                        error=None,
                        raw_response=call_result.raw_response,
                        model=call_result.model,
                        provider=call_result.provider,
                        methodology=self.NAME,
                        steps=steps,
                        categories_selected=categories_selected,
                        backtrack_count=backtrack_count,
                        declined_tool_call=False,
                        final_category=state.get("current_category"),
                        # Token usage aggregated across all steps
                        tokens_input=total_tokens_input if total_tokens_input > 0 else None,
                        tokens_output=total_tokens_output if total_tokens_output > 0 else None,
                        tokens_total=(total_tokens_input + total_tokens_output) if (total_tokens_input > 0 or total_tokens_output > 0) else None,
                    )
            
            # Continue with next options
            logger.debug(f"[Clustering] Continuing to next step with {len(next_options)} options")
            current_options = next_options
            _, step_system = (
                self.get_initial_options(tools)
                if state["phase"] == "category_selection"
                else self.get_category_tools(state["current_category"], tools)
            )
            full_system = f"{system_instruction}\n\n{step_system}" if system_instruction else step_system
        
        # Max steps reached
        logger.debug(f"[Clustering] ===== run_single END (max steps reached) =====")
        logger.debug(f"[Clustering] Total steps: {len(steps)}, Categories visited: {categories_selected}")
        return MethodologyResult(
            success=False,
            called_tool=None,
            called_args=None,
            all_calls=[],
            latency_ms=total_latency,
            error=f"Max steps ({self.max_steps}) reached without selection",
            raw_response=None,
            model=client.PROVIDER_NAME,
            provider=client.PROVIDER_NAME,
            methodology=self.NAME,
            steps=steps,
            categories_selected=categories_selected,
            backtrack_count=backtrack_count,
            declined_tool_call=False,
            final_category=state.get("current_category"),
            # Token usage aggregated across all steps
            tokens_input=total_tokens_input if total_tokens_input > 0 else None,
            tokens_output=total_tokens_output if total_tokens_output > 0 else None,
            tokens_total=(total_tokens_input + total_tokens_output) if (total_tokens_input > 0 or total_tokens_output > 0) else None,
        )
