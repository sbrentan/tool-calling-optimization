"""
Base methodology classes for tool calling experiments.

Provides abstract interfaces for different tool selection strategies:
- BaseMethodology: Simple single-step methodologies
- StepBasedMethodology: Multi-step methodologies with iterative LLM calls
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

from src.tools.base import Tool, TestCase
from src.clients.base import BaseLLMClient, CallResult


class StepType(str, Enum):
    """Types of steps in a multi-step methodology."""
    SELECT_CATEGORY = "select_category"  # LLM selects a category/cluster
    SELECT_TOOL = "select_tool"          # LLM selects a specific tool
    BACKTRACK = "backtrack"              # LLM decides to go back
    DECLINE = "decline"                  # LLM decides not to call any tool
    ERROR = "error"                      # An error occurred


@dataclass
class StepInfo:
    """Information about a single step in a multi-step methodology."""
    step_number: int
    step_type: StepType
    selection: Optional[str] = None  # What was selected (category name, tool name, etc.)
    latency_ms: float = 0.0
    raw_response: Optional[Any] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "selection": self.selection,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class MethodologyResult:
    """
    Result of running a methodology for a single test case.
    
    Extends CallResult with methodology-specific information like
    steps taken, categories selected, and backtrack counts.
    """
    # Core result (compatible with CallResult)
    success: bool
    called_tool: Optional[str] = None
    called_args: Optional[dict[str, Any]] = None
    all_calls: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0  # Total latency across all steps
    error: Optional[str] = None
    raw_response: Optional[Any] = None
    model: str = ""
    provider: str = ""
    
    # Methodology-specific fields
    methodology: str = "unknown"
    steps: list[StepInfo] = field(default_factory=list)
    categories_selected: list[str] = field(default_factory=list)
    backtrack_count: int = 0
    declined_tool_call: bool = False  # True if LLM explicitly declined
    final_category: Optional[str] = None  # Category where tool was found
    
    def to_call_result(self) -> CallResult:
        """Convert to standard CallResult for compatibility."""
        return CallResult(
            success=self.success,
            called_tool=self.called_tool,
            called_args=self.called_args,
            all_calls=self.all_calls,
            latency_ms=self.latency_ms,
            error=self.error,
            raw_response=self.raw_response,
            model=self.model,
            provider=self.provider,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "called_tool": self.called_tool,
            "called_args": self.called_args,
            "all_calls": self.all_calls,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "model": self.model,
            "provider": self.provider,
            "methodology": self.methodology,
            "steps": [s.to_dict() for s in self.steps],
            "categories_selected": self.categories_selected,
            "backtrack_count": self.backtrack_count,
            "declined_tool_call": self.declined_tool_call,
            "final_category": self.final_category,
        }


class BaseMethodology(ABC):
    """
    Abstract base class for tool calling methodologies.
    
    A methodology defines how tools are presented to the LLM
    and how the selection process works.
    """
    
    # Methodology name for identification
    NAME: str = "base"
    
    @abstractmethod
    def run_single(
        self,
        prompt: str,
        tools: list[Tool],
        client: BaseLLMClient,
        system_instruction: Optional[str] = None,
    ) -> MethodologyResult:
        """
        Run the methodology for a single prompt.
        
        Args:
            prompt: User prompt to process
            tools: All available tools
            client: LLM client to use
            system_instruction: Optional system instruction
            
        Returns:
            MethodologyResult with tool selection and metadata
        """
        pass
    
    def run_batch(
        self,
        test_cases: list[TestCase],
        tools: list[Tool],
        client: BaseLLMClient,
        system_instruction: Optional[str] = None,
    ) -> list[MethodologyResult]:
        """
        Run the methodology for a batch of test cases.
        
        Default implementation runs sequentially.
        Override for parallel execution if supported.
        
        Args:
            test_cases: List of test cases to run
            tools: All available tools
            client: LLM client to use
            system_instruction: Optional system instruction
            
        Returns:
            List of MethodologyResult objects
        """
        results = []
        for test_case in test_cases:
            result = self.run_single(
                prompt=test_case.prompt,
                tools=tools,
                client=client,
                system_instruction=system_instruction,
            )
            results.append(result)
        return results


class StepBasedMethodology(BaseMethodology):
    """
    Base class for multi-step methodologies.
    
    Supports iterative LLM calls where the model can:
    - Select from options (categories, tools)
    - Backtrack to previous steps
    - Decline to call any tool
    
    Subclasses define the specific steps and transitions.
    """
    
    # Maximum steps before forcing termination
    DEFAULT_MAX_STEPS: int = 10
    
    # Pseudo-tool names for special actions
    BACKTRACK_TOOL: str = "__backtrack__"
    DECLINE_TOOL: str = "__decline_tool_call__"
    
    def __init__(self, max_steps: Optional[int] = None):
        """
        Initialize the step-based methodology.
        
        Args:
            max_steps: Maximum steps allowed (None = use default)
        """
        self.max_steps = max_steps or self.DEFAULT_MAX_STEPS
    
    @abstractmethod
    def get_initial_options(self, tools: list[Tool]) -> tuple[list[Tool], str]:
        """
        Get the initial options to present to the LLM.
        
        Args:
            tools: All available tools
            
        Returns:
            Tuple of (pseudo-tools for selection, system instruction addition)
        """
        pass
    
    @abstractmethod
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
        pass
    
    def create_pseudo_tool(
        self,
        name: str,
        description: str,
        category: str = "system",
    ) -> Tool:
        """Create a pseudo-tool for system actions like backtrack/decline."""
        return Tool(
            name=name,
            description=description,
            category=category,
            parameters=[],
            tags=["system", "control"],
            complexity="simple",
        )
    
    def get_backtrack_tool(self) -> Tool:
        """Get the pseudo-tool for backtracking."""
        return self.create_pseudo_tool(
            name=self.BACKTRACK_TOOL,
            description="Go back to the previous step and select a different option. "
                       "Use this if the current options don't contain what you need.",
        )
    
    def get_decline_tool(self) -> Tool:
        """Get the pseudo-tool for declining to call any tool."""
        return self.create_pseudo_tool(
            name=self.DECLINE_TOOL,
            description="Indicate that no tool call is needed for this request. "
                       "Use this if the user's request doesn't require any tool.",
        )
