"""
Evaluation metrics for tool calling accuracy.

Measures:
- Tool selection accuracy (correct tool called)
- Parameter accuracy (correct parameters extracted)
- Latency statistics
- Methodology-specific metrics (steps, backtracks, category accuracy)
- No-tool test case evaluation (false positive rate)
- Multi-tool test case evaluation (sequence/set matching)
"""
from typing import Any, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd
from loguru import logger

from src.tools.base import TestCase, MultiToolTestCase, AmbiguousTestCase
from src.clients.base import CallResult

if TYPE_CHECKING:
    from src.methodologies.base import MethodologyResult

# Type alias for any test case
AnyTestCase = Union[TestCase, MultiToolTestCase, AmbiguousTestCase]


@dataclass
class TestResult:
    """Result of a single test case evaluation."""
    test_case: AnyTestCase
    call_result: CallResult
    tool_correct: bool
    params_correct: Optional[bool] = None
    
    # Methodology-specific fields
    methodology: str = "mcp"
    steps_count: int = 1
    backtrack_count: int = 0
    declined_tool_call: bool = False
    category_correct: Optional[bool] = None  # True if correct category was selected
    final_category: Optional[str] = None
    categories_visited: list[str] = field(default_factory=list)
    
    # No-tool test case fields
    is_no_tool_test: bool = False  # True if this was a no-tool expected test
    false_positive: bool = False   # True if tool was called when none expected
    
    # Multi-tool test case fields
    is_multi_tool_test: bool = False  # True if this was a multi-tool test
    tools_called: list[str] = field(default_factory=list)  # All tools called
    completion_rate: Optional[float] = None  # % of expected tools called
    sequence_correct: Optional[bool] = None  # True if tools called in correct order
    extra_calls: int = 0  # Number of unexpected tool calls
    
    # Per-parameter accuracy
    param_details: dict[str, bool] = field(default_factory=dict)  # param_name -> correct
    
    # Phase 2 methodology fields
    confidence_score: Optional[float] = None  # Confidence methodology score
    fallback_method_used: Optional[str] = None  # Which method was used in confidence fallback
    num_fallbacks: int = 0  # Number of fallbacks in confidence methodology
    adaptive_k_used: Optional[int] = None  # Adaptive RAG: k value used
    adaptive_strategy: Optional[str] = None  # Adaptive RAG: strategy used (elbow/threshold)
    
    # Phase 3: Token tracking
    tokens_input: Optional[int] = None   # Input tokens used
    tokens_output: Optional[int] = None  # Output tokens used
    tokens_total: Optional[int] = None   # Total tokens
    
    # Phase 3: Retrieval metrics (for RAG-based methodologies)
    retrieval_recall: Optional[bool] = None  # Was correct tool in retrieved set?
    retrieved_tools: list[str] = field(default_factory=list)  # Tools that were retrieved
    retrieval_rank: Optional[int] = None  # Rank of correct tool in retrieved set (1-indexed)
    
    # Phase 4: Clarification metrics
    is_ambiguous_test: bool = False  # True if this was an ambiguous/clarification test
    clarification_requested: bool = False  # True if LLM requested clarification
    clarification_correct: Optional[bool] = None  # True if correct tool was in candidates
    clarification_score: Optional[float] = None  # Score based on candidate set size
    clarification_question: Optional[str] = None  # The clarifying question asked
    candidate_tools: list[str] = field(default_factory=list)  # Tools LLM suggested as candidates
    expected_candidate_tools: list[str] = field(default_factory=list)  # Expected candidate tools for ambiguous tests
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for analysis."""
        # Handle both TestCase and MultiToolTestCase
        if isinstance(self.test_case, MultiToolTestCase):
            expected_tool = ",".join(self.test_case.expected_tools)
        else:
            expected_tool = self.test_case.expected_tool
        
        return {
            "prompt": self.test_case.prompt,
            "expected_tool": expected_tool,
            "called_tool": self.call_result.called_tool,
            "tool_correct": self.tool_correct,
            "params_correct": self.params_correct,
            "latency_ms": self.call_result.latency_ms,
            "category": self.test_case.category,
            "difficulty": self.test_case.difficulty,
            "model": self.call_result.model,
            "error": self.call_result.error,
            # Methodology-specific
            "methodology": self.methodology,
            "steps_count": self.steps_count,
            "backtrack_count": self.backtrack_count,
            "declined_tool_call": self.declined_tool_call,
            "category_correct": self.category_correct,
            "final_category": self.final_category,
            "categories_visited": self.categories_visited,
            # No-tool metrics
            "is_no_tool_test": self.is_no_tool_test,
            "false_positive": self.false_positive,
            # Multi-tool metrics
            "is_multi_tool_test": self.is_multi_tool_test,
            "tools_called": self.tools_called,
            "completion_rate": self.completion_rate,
            "sequence_correct": self.sequence_correct,
            "extra_calls": self.extra_calls,
            # Per-parameter accuracy
            "param_details": self.param_details,
            # Phase 2 methodology metrics
            "confidence_score": self.confidence_score,
            "fallback_method_used": self.fallback_method_used,
            "num_fallbacks": self.num_fallbacks,
            "adaptive_k_used": self.adaptive_k_used,
            "adaptive_strategy": self.adaptive_strategy,
            # Phase 3: Token tracking
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_total": self.tokens_total,
            # Phase 3: Retrieval metrics
            "retrieval_recall": self.retrieval_recall,
            "retrieved_tools": self.retrieved_tools,
            "retrieval_rank": self.retrieval_rank,
            # Phase 4: Clarification metrics
            "is_ambiguous_test": self.is_ambiguous_test,
            "clarification_requested": self.clarification_requested,
            "clarification_correct": self.clarification_correct,
            "clarification_score": self.clarification_score,
            "clarification_question": self.clarification_question,
            "candidate_tools": self.candidate_tools,
            "expected_candidate_tools": self.expected_candidate_tools,
        }


@dataclass
class EvaluationResult:
    """Aggregated evaluation results for an experiment."""
    total_tests: int = 0
    tool_correct: int = 0
    tool_incorrect: int = 0
    no_tool_called: int = 0
    errors: int = 0
    
    # Detailed results
    test_results: list[TestResult] = field(default_factory=list)
    
    # Latency stats
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # Per-category accuracy
    category_accuracy: dict[str, float] = field(default_factory=dict)
    
    # Experiment metadata
    experiment_config: dict[str, Any] = field(default_factory=dict)
    
    # Methodology-specific metrics
    methodology: str = "mcp"
    avg_steps_per_call: float = 1.0
    total_backtracks: int = 0
    avg_backtracks_per_call: float = 0.0
    declined_tool_calls: int = 0
    category_selection_accuracy: float = 0.0  # For clustering: correct category rate
    
    # No-tool test metrics
    no_tool_tests: int = 0  # Number of no-tool test cases
    no_tool_correct: int = 0  # Correctly didn't call a tool
    false_positive_count: int = 0  # Called a tool when none expected
    false_positive_rate: float = 0.0  # FP / no_tool_tests
    
    # Multi-tool test metrics
    multi_tool_tests: int = 0  # Number of multi-tool test cases
    multi_tool_correct: int = 0  # All expected tools called correctly
    avg_completion_rate: float = 0.0  # Average % of expected tools called
    avg_sequence_accuracy: float = 0.0  # Average sequence correctness (for require_sequence tests)
    avg_extra_calls: float = 0.0  # Average unnecessary tool calls per multi-tool test
    
    # Parameter accuracy metrics
    params_tested: int = 0  # Number of tests with expected params
    params_correct: int = 0  # Number with all params correct
    params_accuracy: float = 0.0  # params_correct / params_tested
    
    # Phase 2 methodology metrics
    # Confidence methodology
    fallback_rate: float = 0.0  # Rate at which fallback was needed
    method_used_distribution: dict[str, int] = field(default_factory=dict)  # Count per method used
    avg_num_fallbacks: float = 0.0  # Average fallbacks per call
    avg_confidence_score: float = 0.0  # Average confidence when accepted
    
    # Adaptive RAG methodology
    adaptive_k_stats: dict[str, float] = field(default_factory=dict)  # min/max/avg k used
    adaptive_strategy_distribution: dict[str, int] = field(default_factory=dict)  # Count per strategy
    
    # Hybrid methodology (uses category_selection_accuracy already defined)
    
    # Phase 3: Token usage metrics
    total_tokens_input: int = 0    # Total input tokens across all tests
    total_tokens_output: int = 0   # Total output tokens across all tests
    total_tokens: int = 0          # Total tokens (input + output)
    avg_tokens_input: float = 0.0  # Average input tokens per test
    avg_tokens_output: float = 0.0 # Average output tokens per test
    avg_tokens_total: float = 0.0  # Average total tokens per test
    
    # Phase 3: Retrieval metrics (for RAG-based methodologies)
    retrieval_tests: int = 0       # Number of tests with retrieval data
    retrieval_recall_count: int = 0  # Times correct tool was in retrieved set
    retrieval_recall_rate: float = 0.0  # retrieval_recall_count / retrieval_tests
    avg_retrieval_rank: float = 0.0  # Average rank of correct tool when retrieved
    
    # Phase 4: Clarification metrics
    ambiguous_tests: int = 0  # Number of ambiguous test cases
    ambiguous_correct: int = 0  # Correctly asked for clarification with correct tool in candidates
    clarification_accuracy: float = 0.0  # ambiguous_correct / ambiguous_tests
    avg_clarification_score: float = 0.0  # Average clarification score
    false_clarification_count: int = 0  # Asked for clarification when not needed
    false_clarification_rate: float = 0.0  # Rate of unnecessary clarification requests
    clarification_requests: int = 0  # Total clarification requests across all tests
    
    @property
    def accuracy(self) -> float:
        """Overall tool selection accuracy."""
        if self.total_tests == 0:
            return 0.0
        return self.tool_correct / self.total_tests
    
    @property
    def call_rate(self) -> float:
        """Rate at which the model calls any tool."""
        if self.total_tests == 0:
            return 0.0
        return (self.tool_correct + self.tool_incorrect) / self.total_tests
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "total_tests": self.total_tests,
            "tool_correct": self.tool_correct,
            "tool_incorrect": self.tool_incorrect,
            "no_tool_called": self.no_tool_called,
            "errors": self.errors,
            "accuracy": self.accuracy,
            "call_rate": self.call_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "category_accuracy": self.category_accuracy,
            "experiment_config": self.experiment_config,
            # Methodology-specific
            "methodology": self.methodology,
            "avg_steps_per_call": self.avg_steps_per_call,
            "total_backtracks": self.total_backtracks,
            "avg_backtracks_per_call": self.avg_backtracks_per_call,
            "declined_tool_calls": self.declined_tool_calls,
            "category_selection_accuracy": self.category_selection_accuracy,
            # No-tool metrics
            "no_tool_tests": self.no_tool_tests,
            "no_tool_correct": self.no_tool_correct,
            "false_positive_count": self.false_positive_count,
            "false_positive_rate": self.false_positive_rate,
            # Multi-tool metrics
            "multi_tool_tests": self.multi_tool_tests,
            "multi_tool_correct": self.multi_tool_correct,
            "avg_completion_rate": self.avg_completion_rate,
            "avg_sequence_accuracy": self.avg_sequence_accuracy,
            "avg_extra_calls": self.avg_extra_calls,
            # Parameter accuracy
            "params_tested": self.params_tested,
            "params_correct": self.params_correct,
            "params_accuracy": self.params_accuracy,
            # Phase 2 methodology metrics
            "fallback_rate": self.fallback_rate,
            "method_used_distribution": self.method_used_distribution,
            "avg_num_fallbacks": self.avg_num_fallbacks,
            "avg_confidence_score": self.avg_confidence_score,
            "adaptive_k_stats": self.adaptive_k_stats,
            "adaptive_strategy_distribution": self.adaptive_strategy_distribution,
            # Phase 3: Token usage metrics
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "total_tokens": self.total_tokens,
            "avg_tokens_input": self.avg_tokens_input,
            "avg_tokens_output": self.avg_tokens_output,
            "avg_tokens_total": self.avg_tokens_total,
            # Phase 3: Retrieval metrics
            "retrieval_tests": self.retrieval_tests,
            "retrieval_recall_count": self.retrieval_recall_count,
            "retrieval_recall_rate": self.retrieval_recall_rate,
            "avg_retrieval_rank": self.avg_retrieval_rank,
            # Phase 4: Clarification metrics
            "ambiguous_tests": self.ambiguous_tests,
            "ambiguous_correct": self.ambiguous_correct,
            "clarification_accuracy": self.clarification_accuracy,
            "avg_clarification_score": self.avg_clarification_score,
            "false_clarification_count": self.false_clarification_count,
            "false_clarification_rate": self.false_clarification_rate,
            "clarification_requests": self.clarification_requests,
        }
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert detailed results to pandas DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.test_results])
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 50,
            "EVALUATION RESULTS",
            "=" * 50,
            f"Methodology: {self.methodology}",
            f"Total Tests: {self.total_tests}",
            f"Tool Selection Accuracy: {self.accuracy:.2%}",
            f"Tool Call Rate: {self.call_rate:.2%}",
            "",
            f"Correct: {self.tool_correct}",
            f"Incorrect: {self.tool_incorrect}",
            f"No Tool Called: {self.no_tool_called}",
            f"Declined Tool Calls: {self.declined_tool_calls}",
            f"Errors: {self.errors}",
            "",
            f"Avg Latency: {self.avg_latency_ms:.1f}ms",
            f"Min Latency: {self.min_latency_ms:.1f}ms",
            f"Max Latency: {self.max_latency_ms:.1f}ms",
        ]
        
        # Add no-tool test stats
        if self.no_tool_tests > 0:
            lines.append("")
            lines.append("No-Tool Test Results:")
            lines.append(f"  No-Tool Tests: {self.no_tool_tests}")
            lines.append(f"  Correct (no tool called): {self.no_tool_correct}")
            lines.append(f"  False Positives: {self.false_positive_count}")
            lines.append(f"  False Positive Rate: {self.false_positive_rate:.2%}")
        
        # Add multi-tool test stats
        if self.multi_tool_tests > 0:
            lines.append("")
            lines.append("Multi-Tool Test Results:")
            lines.append(f"  Multi-Tool Tests: {self.multi_tool_tests}")
            lines.append(f"  Fully Correct: {self.multi_tool_correct}")
            lines.append(f"  Avg Completion Rate: {self.avg_completion_rate:.2%}")
            lines.append(f"  Avg Sequence Accuracy: {self.avg_sequence_accuracy:.2%}")
            lines.append(f"  Avg Extra Calls: {self.avg_extra_calls:.2f}")
        
        # Add parameter accuracy stats
        if self.params_tested > 0:
            lines.append("")
            lines.append("Parameter Accuracy:")
            lines.append(f"  Tests with Expected Params: {self.params_tested}")
            lines.append(f"  Fully Correct Params: {self.params_correct}")
            lines.append(f"  Parameter Accuracy: {self.params_accuracy:.2%}")
        
        # Add methodology-specific stats
        if self.methodology != "mcp":
            lines.append("")
            lines.append("Methodology Stats:")
            lines.append(f"  Avg Steps per Call: {self.avg_steps_per_call:.2f}")
            lines.append(f"  Total Backtracks: {self.total_backtracks}")
            lines.append(f"  Avg Backtracks per Call: {self.avg_backtracks_per_call:.2f}")
            if self.methodology in ("clustering", "hybrid"):
                lines.append(f"  Category Selection Accuracy: {self.category_selection_accuracy:.2%}")
        
        # Add Phase 2 methodology stats
        if self.methodology == "confidence" and self.method_used_distribution:
            lines.append("")
            lines.append("Confidence Methodology Stats:")
            lines.append(f"  Fallback Rate: {self.fallback_rate:.2%}")
            lines.append(f"  Avg Fallbacks per Call: {self.avg_num_fallbacks:.2f}")
            lines.append(f"  Avg Confidence Score: {self.avg_confidence_score:.3f}")
            lines.append("  Method Distribution:")
            for method, count in sorted(self.method_used_distribution.items()):
                lines.append(f"    {method}: {count}")
        
        if self.methodology == "adaptive_rag" and self.adaptive_k_stats:
            lines.append("")
            lines.append("Adaptive RAG Stats:")
            lines.append(f"  Min K Used: {self.adaptive_k_stats.get('min_k', 0):.0f}")
            lines.append(f"  Max K Used: {self.adaptive_k_stats.get('max_k', 0):.0f}")
            lines.append(f"  Avg K Used: {self.adaptive_k_stats.get('avg_k', 0):.1f}")
            if self.adaptive_strategy_distribution:
                lines.append("  Strategy Distribution:")
                for strategy, count in sorted(self.adaptive_strategy_distribution.items()):
                    lines.append(f"    {strategy}: {count}")
        
        # Phase 3: Token usage stats
        if self.total_tokens > 0:
            lines.append("")
            lines.append("Token Usage:")
            lines.append(f"  Total Input Tokens: {self.total_tokens_input:,}")
            lines.append(f"  Total Output Tokens: {self.total_tokens_output:,}")
            lines.append(f"  Total Tokens: {self.total_tokens:,}")
            lines.append(f"  Avg Input Tokens/Test: {self.avg_tokens_input:.1f}")
            lines.append(f"  Avg Output Tokens/Test: {self.avg_tokens_output:.1f}")
            lines.append(f"  Avg Total Tokens/Test: {self.avg_tokens_total:.1f}")
        
        # Phase 3: Retrieval metrics (for RAG-based methodologies)
        if self.retrieval_tests > 0:
            lines.append("")
            lines.append("Retrieval Metrics:")
            lines.append(f"  Tests with Retrieval: {self.retrieval_tests}")
            lines.append(f"  Retrieval Recall: {self.retrieval_recall_rate:.2%}")
            lines.append(f"  Avg Rank of Correct Tool: {self.avg_retrieval_rank:.1f}")
        
        # Phase 4: Clarification metrics
        if self.ambiguous_tests > 0 or self.clarification_requests > 0:
            lines.append("")
            lines.append("Clarification Metrics:")
            if self.ambiguous_tests > 0:
                lines.append(f"  Ambiguous Tests: {self.ambiguous_tests}")
                lines.append(f"  Correctly Handled: {self.ambiguous_correct}")
                lines.append(f"  Clarification Accuracy: {self.clarification_accuracy:.2%}")
                lines.append(f"  Avg Clarification Score: {self.avg_clarification_score:.3f}")
            if self.clarification_requests > 0:
                lines.append(f"  Total Clarification Requests: {self.clarification_requests}")
            if self.false_clarification_count > 0:
                lines.append(f"  False Clarifications: {self.false_clarification_count}")
                lines.append(f"  False Clarification Rate: {self.false_clarification_rate:.2%}")
        
        if self.category_accuracy:
            lines.append("")
            lines.append("Per-Category Accuracy:")
            for cat, acc in sorted(self.category_accuracy.items()):
                lines.append(f"  {cat}: {acc:.2%}")
        
        if self.experiment_config:
            lines.append("")
            lines.append("Experiment Config:")
            for key, value in self.experiment_config.items():
                lines.append(f"  {key}: {value}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


class ToolCallEvaluator:
    """
    Evaluator for tool calling accuracy.
    
    Compares expected tool calls against actual model responses
    and computes various metrics. Supports:
    - Single-tool test cases
    - No-tool test cases (expecting no tool call)
    - Multi-tool test cases (expecting multiple tools)
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results: list[TestResult] = []
    
    def evaluate_single(
        self,
        test_case: AnyTestCase,
        call_result: CallResult,
        methodology_result: Optional["MethodologyResult"] = None,
        max_clarification_candidates: int = 3,
    ) -> TestResult:
        """
        Evaluate a single test case (single-tool, no-tool, or ambiguous).
        
        Args:
            test_case: Expected test case (TestCase, MultiToolTestCase, or AmbiguousTestCase)
            call_result: Actual result from API call
            methodology_result: Optional methodology-specific result
            max_clarification_candidates: Max candidates for full clarification score
            
        Returns:
            TestResult with evaluation details
        """
        # Handle multi-tool test cases separately
        if isinstance(test_case, MultiToolTestCase):
            return self.evaluate_multi_tool(test_case, call_result, methodology_result)
        
        # Handle ambiguous test cases separately
        if isinstance(test_case, AmbiguousTestCase):
            return self.evaluate_ambiguous(test_case, call_result, methodology_result, max_clarification_candidates)
        
        # Determine if this is a no-tool test case
        is_no_tool_test = test_case.expected_tool is None
        
        # Extract clarification info from methodology result
        clarification_requested = False
        clarification_question = None
        candidate_tools = []
        if methodology_result is not None:
            clarification_requested = methodology_result.clarification_requested
            clarification_question = methodology_result.clarification_question
            candidate_tools = methodology_result.candidate_tools
        
        # Check if correct tool was called
        if is_no_tool_test:
            # For no-tool tests: correct if no tool was called (clarification is also acceptable)
            tool_correct = call_result.called_tool is None
            false_positive = call_result.called_tool is not None and not clarification_requested
        else:
            # For regular tests: correct if expected tool was called
            # Clarification on a non-ambiguous test is a false clarification
            tool_correct = (
                call_result.called_tool is not None and
                call_result.called_tool == test_case.expected_tool
            )
            false_positive = False
        
        # Check parameters if expected (with improved checking)
        params_correct = None
        param_details = {}
        if tool_correct and test_case.expected_params is not None:
            params_correct, param_details = self._check_params_detailed(
                test_case.expected_params,
                call_result.called_args or {}
            )
        
        # Extract methodology-specific info
        methodology = "mcp"
        steps_count = 1
        backtrack_count = 0
        declined_tool_call = False
        category_correct = None
        final_category = None
        categories_visited = []
        
        if methodology_result is not None:
            methodology = methodology_result.methodology
            steps_count = len(methodology_result.steps)
            backtrack_count = methodology_result.backtrack_count
            declined_tool_call = methodology_result.declined_tool_call
            final_category = methodology_result.final_category
            categories_visited = methodology_result.categories_selected
            
            # Check if correct category was selected (for clustering/hybrid)
            if methodology in ("clustering", "hybrid") and final_category is not None:
                category_correct = (final_category == test_case.category)
        
        # Extract Phase 2 methodology-specific metrics
        confidence_score = None
        fallback_method_used = None
        num_fallbacks = 0
        adaptive_k_used = None
        adaptive_strategy = None
        
        # Phase 3: Retrieval metrics (for RAG-based methodologies)
        retrieved_tools = []
        retrieval_recall = None
        retrieval_rank = None
        
        if methodology_result is not None:
            # Confidence methodology metrics
            if hasattr(methodology_result, '_confidence_metadata') and methodology_result._confidence_metadata:
                conf_meta = methodology_result._confidence_metadata
                fallback_method_used = conf_meta.get('final_method')
                num_fallbacks = conf_meta.get('num_fallbacks', 0)
                # Get confidence of the method that was accepted
                confidences = conf_meta.get('confidences', {})
                if fallback_method_used and fallback_method_used in confidences:
                    confidence_score = confidences[fallback_method_used]
            
            # Adaptive RAG methodology metrics
            if hasattr(methodology_result, '_adaptive_metadata') and methodology_result._adaptive_metadata:
                adapt_meta = methodology_result._adaptive_metadata
                adaptive_k_used = adapt_meta.get('adaptive_k_used')
                adaptive_strategy = adapt_meta.get('strategy_used')
            
            # RAG retrieval metrics (from _rag_metadata)
            if hasattr(methodology_result, '_rag_metadata') and methodology_result._rag_metadata:
                rag_meta = methodology_result._rag_metadata
                retrieved_tools = rag_meta.get('retrieved_tools', [])
                
                # Check if expected tool was in the retrieved set
                if not is_no_tool_test and test_case.expected_tool:
                    retrieval_recall = test_case.expected_tool in retrieved_tools
                    if retrieval_recall:
                        # Find rank (1-indexed)
                        try:
                            retrieval_rank = retrieved_tools.index(test_case.expected_tool) + 1
                        except ValueError:
                            retrieval_rank = None
        
        result = TestResult(
            test_case=test_case,
            call_result=call_result,
            tool_correct=tool_correct,
            params_correct=params_correct,
            methodology=methodology,
            steps_count=steps_count,
            backtrack_count=backtrack_count,
            declined_tool_call=declined_tool_call,
            category_correct=category_correct,
            final_category=final_category,
            categories_visited=categories_visited,
            # No-tool fields
            is_no_tool_test=is_no_tool_test,
            false_positive=false_positive,
            # Per-parameter accuracy
            param_details=param_details,
            # Phase 2 methodology fields
            confidence_score=confidence_score,
            fallback_method_used=fallback_method_used,
            num_fallbacks=num_fallbacks,
            adaptive_k_used=adaptive_k_used,
            adaptive_strategy=adaptive_strategy,
            # Phase 3: Retrieval metrics
            retrieval_recall=retrieval_recall,
            retrieved_tools=retrieved_tools,
            retrieval_rank=retrieval_rank,
            # Phase 3: Token tracking
            tokens_input=call_result.tokens_input,
            tokens_output=call_result.tokens_output,
            tokens_total=call_result.tokens_total,
            # Phase 4: Clarification fields
            clarification_requested=clarification_requested,
            clarification_question=clarification_question,
            candidate_tools=candidate_tools,
        )
        
        self.results.append(result)
        return result
    
    def evaluate_multi_tool(
        self,
        test_case: MultiToolTestCase,
        call_result: CallResult,
        methodology_result: Optional["MethodologyResult"] = None,
    ) -> TestResult:
        """
        Evaluate a multi-tool test case.
        
        Args:
            test_case: Multi-tool test case with expected_tools list
            call_result: Actual result from API call
            methodology_result: Optional methodology-specific result
            
        Returns:
            TestResult with multi-tool evaluation details
        """
        # Get all tools called from methodology result if available
        tools_called = []
        if methodology_result is not None and methodology_result.all_calls:
            tools_called = [
                call.get("tool") or call.get("name", "")
                for call in methodology_result.all_calls
                if call.get("tool") or call.get("name")
            ]
        elif call_result.called_tool:
            tools_called = [call_result.called_tool]
        
        expected_tools = test_case.expected_tools
        expected_set = set(expected_tools)
        called_set = set(tools_called)
        
        # Calculate metrics
        matched_tools = expected_set & called_set
        completion_rate = len(matched_tools) / len(expected_tools) if expected_tools else 0.0
        extra_calls = len(called_set - expected_set)
        
        # Check sequence if required
        sequence_correct = None
        if test_case.require_sequence and tools_called:
            # Filter called_tools to only include expected tools, preserving order
            filtered_calls = [t for t in tools_called if t in expected_set]
            # Check if the order matches the expected order
            sequence_correct = filtered_calls == expected_tools[:len(filtered_calls)]
        
        # All expected tools were called (order may or may not matter)
        if test_case.require_sequence:
            tool_correct = (tools_called == expected_tools)
        else:
            tool_correct = (expected_set == matched_tools and len(matched_tools) == len(expected_tools))
        
        # Extract methodology-specific info
        methodology = "mcp"
        steps_count = 1
        backtrack_count = 0
        declined_tool_call = False
        final_category = None
        categories_visited = []
        
        if methodology_result is not None:
            methodology = methodology_result.methodology
            steps_count = len(methodology_result.steps)
            backtrack_count = methodology_result.backtrack_count
            declined_tool_call = methodology_result.declined_tool_call
            final_category = methodology_result.final_category
            categories_visited = methodology_result.categories_selected
        
        result = TestResult(
            test_case=test_case,
            call_result=call_result,
            tool_correct=tool_correct,
            params_correct=None,  # TODO: Multi-tool param checking
            methodology=methodology,
            steps_count=steps_count,
            backtrack_count=backtrack_count,
            declined_tool_call=declined_tool_call,
            category_correct=None,
            final_category=final_category,
            categories_visited=categories_visited,
            # Multi-tool fields
            is_multi_tool_test=True,
            tools_called=tools_called,
            completion_rate=completion_rate,
            sequence_correct=sequence_correct,
            extra_calls=extra_calls,
            # Phase 3: Token tracking
            tokens_input=call_result.tokens_input,
            tokens_output=call_result.tokens_output,
            tokens_total=call_result.tokens_total,
        )
        
        self.results.append(result)
        return result
    
    def evaluate_ambiguous(
        self,
        test_case: AmbiguousTestCase,
        call_result: CallResult,
        methodology_result: Optional["MethodologyResult"] = None,
        max_clarification_candidates: int = 3,
    ) -> TestResult:
        """
        Evaluate an ambiguous test case where clarification is expected.
        
        Scoring formula:
        - If clarification requested and correct tool in candidates:
          - Score = 1.0 if len(candidates) <= max_clarification_candidates
          - Score = 1.0 / len(candidates) otherwise (penalized for too many candidates)
        - If clarification not requested: Score = 0
        - If correct tool not in candidates: Score = 0
        
        Args:
            test_case: Ambiguous test case with expected_candidate_tools
            call_result: Actual result from API call
            methodology_result: Optional methodology-specific result
            max_clarification_candidates: Max candidates for full score
            
        Returns:
            TestResult with clarification evaluation details
        """
        # Extract clarification info from methodology result
        clarification_requested = False
        clarification_question = None
        candidate_tools = []
        
        if methodology_result is not None:
            clarification_requested = methodology_result.clarification_requested
            clarification_question = methodology_result.clarification_question
            candidate_tools = methodology_result.candidate_tools
        
        # Check if any expected candidate tool is in the LLM's candidates
        # The "correct" behavior is to include at least one of the expected candidates
        expected_candidates = set(test_case.expected_candidate_tools)
        actual_candidates = set(candidate_tools)
        
        # Check if the correct tool (if specified) is among candidates
        correct_tool = test_case.correct_tool
        has_correct_tool = False
        if correct_tool:
            has_correct_tool = correct_tool in actual_candidates
        else:
            # If no specific correct tool, check if any expected candidate is present
            has_correct_tool = bool(expected_candidates & actual_candidates)
        
        # Compute clarification score
        clarification_score = 0.0
        clarification_correct = False
        
        if clarification_requested and has_correct_tool:
            clarification_correct = True
            num_candidates = len(candidate_tools)
            if num_candidates <= max_clarification_candidates:
                clarification_score = 1.0
            else:
                clarification_score = 1.0 / num_candidates
        elif clarification_requested:
            # Clarification requested but correct tool not in candidates
            clarification_correct = False
            clarification_score = 0.0
        else:
            # No clarification requested when it should have been
            clarification_correct = False
            clarification_score = 0.0
        
        # For ambiguous tests, tool_correct means handled correctly
        tool_correct = clarification_correct
        
        # Extract methodology-specific info
        methodology = "mcp"
        steps_count = 1
        backtrack_count = 0
        declined_tool_call = False
        final_category = None
        categories_visited = []
        
        if methodology_result is not None:
            methodology = methodology_result.methodology
            steps_count = len(methodology_result.steps)
            backtrack_count = methodology_result.backtrack_count
            declined_tool_call = methodology_result.declined_tool_call
            final_category = methodology_result.final_category
            categories_visited = methodology_result.categories_selected
        
        result = TestResult(
            test_case=test_case,
            call_result=call_result,
            tool_correct=tool_correct,
            params_correct=None,
            methodology=methodology,
            steps_count=steps_count,
            backtrack_count=backtrack_count,
            declined_tool_call=declined_tool_call,
            category_correct=None,
            final_category=final_category,
            categories_visited=categories_visited,
            # Ambiguous test fields
            is_ambiguous_test=True,
            clarification_requested=clarification_requested,
            clarification_correct=clarification_correct,
            clarification_score=clarification_score,
            clarification_question=clarification_question,
            candidate_tools=candidate_tools,
            expected_candidate_tools=list(expected_candidates),
            # Phase 3: Token tracking
            tokens_input=call_result.tokens_input,
            tokens_output=call_result.tokens_output,
            tokens_total=call_result.tokens_total,
        )
        
        self.results.append(result)
        return result
    
    def evaluate_batch(
        self,
        test_cases: list[AnyTestCase],
        call_results: list[CallResult]
    ) -> list[TestResult]:
        """
        Evaluate a batch of test cases.
        
        Args:
            test_cases: List of expected test cases (TestCase or MultiToolTestCase)
            call_results: List of actual results from API calls
            
        Returns:
            List of TestResult objects
        """
        if len(test_cases) != len(call_results):
            raise ValueError("Number of test cases must match number of results")
        
        results = []
        for test_case, call_result in zip(test_cases, call_results):
            result = self.evaluate_single(test_case, call_result)
            results.append(result)
        
        return results
    
    def compute_metrics(
        self,
        experiment_config: Optional[dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Compute aggregate metrics from all evaluated results.
        
        Args:
            experiment_config: Optional config metadata to include
            
        Returns:
            EvaluationResult with aggregate metrics
        """
        if not self.results:
            return EvaluationResult(experiment_config=experiment_config or {})
        
        # Count results
        total = len(self.results)
        correct = sum(1 for r in self.results if r.tool_correct)
        incorrect = sum(
            1 for r in self.results
            if not r.tool_correct and r.call_result.called_tool is not None
        )
        no_call = sum(
            1 for r in self.results
            if r.call_result.called_tool is None and r.call_result.success
        )
        errors = sum(1 for r in self.results if not r.call_result.success)
        
        # Latency stats
        latencies = [r.call_result.latency_ms for r in self.results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        
        # Per-category accuracy
        category_counts = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in self.results:
            cat = r.test_case.category
            category_counts[cat]["total"] += 1
            if r.tool_correct:
                category_counts[cat]["correct"] += 1
        
        category_accuracy = {
            cat: counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            for cat, counts in category_counts.items()
        }
        
        # Methodology-specific metrics
        methodology = self.results[0].methodology if self.results else "mcp"
        
        # Steps and backtracks
        total_steps = sum(r.steps_count for r in self.results)
        avg_steps = total_steps / total if total > 0 else 1.0
        
        total_backtracks = sum(r.backtrack_count for r in self.results)
        avg_backtracks = total_backtracks / total if total > 0 else 0.0
        
        # Declined calls
        declined_count = sum(1 for r in self.results if r.declined_tool_call)
        
        # Category selection accuracy (for clustering and hybrid)
        category_selection_accuracy = 0.0
        if methodology in ("clustering", "hybrid"):
            category_correct_count = sum(
                1 for r in self.results 
                if r.category_correct is True
            )
            category_total = sum(
                1 for r in self.results 
                if r.category_correct is not None
            )
            if category_total > 0:
                category_selection_accuracy = category_correct_count / category_total
        
        # No-tool test metrics
        no_tool_results = [r for r in self.results if r.is_no_tool_test]
        no_tool_tests = len(no_tool_results)
        no_tool_correct = sum(1 for r in no_tool_results if r.tool_correct)
        false_positive_count = sum(1 for r in no_tool_results if r.false_positive)
        false_positive_rate = false_positive_count / no_tool_tests if no_tool_tests > 0 else 0.0
        
        # Multi-tool test metrics
        multi_tool_results = [r for r in self.results if r.is_multi_tool_test]
        multi_tool_tests = len(multi_tool_results)
        multi_tool_correct = sum(1 for r in multi_tool_results if r.tool_correct)
        
        avg_completion_rate = 0.0
        avg_sequence_accuracy = 0.0
        avg_extra_calls = 0.0
        if multi_tool_tests > 0:
            completion_rates = [r.completion_rate for r in multi_tool_results if r.completion_rate is not None]
            avg_completion_rate = sum(completion_rates) / len(completion_rates) if completion_rates else 0.0
            
            sequence_results = [r.sequence_correct for r in multi_tool_results if r.sequence_correct is not None]
            avg_sequence_accuracy = sum(1 for s in sequence_results if s) / len(sequence_results) if sequence_results else 0.0
            
            extra_calls_list = [r.extra_calls for r in multi_tool_results]
            avg_extra_calls = sum(extra_calls_list) / len(extra_calls_list) if extra_calls_list else 0.0
        
        # Parameter accuracy metrics
        params_results = [r for r in self.results if r.params_correct is not None]
        params_tested = len(params_results)
        params_correct_count = sum(1 for r in params_results if r.params_correct)
        params_accuracy = params_correct_count / params_tested if params_tested > 0 else 0.0
        
        # Phase 2 methodology metrics
        # Confidence methodology metrics
        fallback_rate = 0.0
        method_used_distribution: dict[str, int] = {}
        avg_num_fallbacks = 0.0
        avg_confidence_score = 0.0
        
        confidence_results = [r for r in self.results if r.fallback_method_used is not None]
        if confidence_results:
            # Count fallbacks (when more than one method was tried)
            fallback_count = sum(1 for r in confidence_results if r.num_fallbacks > 0)
            fallback_rate = fallback_count / len(confidence_results)
            
            # Distribution of final methods used
            for r in confidence_results:
                method = r.fallback_method_used
                method_used_distribution[method] = method_used_distribution.get(method, 0) + 1
            
            # Average number of fallbacks
            total_fallbacks = sum(r.num_fallbacks for r in confidence_results)
            avg_num_fallbacks = total_fallbacks / len(confidence_results)
            
            # Average confidence score
            confidence_scores = [r.confidence_score for r in confidence_results if r.confidence_score is not None]
            if confidence_scores:
                avg_confidence_score = sum(confidence_scores) / len(confidence_scores)
        
        # Adaptive RAG methodology metrics
        adaptive_k_stats: dict[str, float] = {}
        adaptive_strategy_distribution: dict[str, int] = {}
        
        adaptive_results = [r for r in self.results if r.adaptive_k_used is not None]
        if adaptive_results:
            k_values = [r.adaptive_k_used for r in adaptive_results]
            adaptive_k_stats = {
                "min_k": float(min(k_values)),
                "max_k": float(max(k_values)),
                "avg_k": sum(k_values) / len(k_values),
            }
            
            # Distribution of strategies used
            for r in adaptive_results:
                if r.adaptive_strategy:
                    strategy = r.adaptive_strategy
                    adaptive_strategy_distribution[strategy] = adaptive_strategy_distribution.get(strategy, 0) + 1
        
        # Phase 3: Token usage metrics
        total_tokens_input = 0
        total_tokens_output = 0
        total_tokens_count = 0
        token_results = [r for r in self.results if r.tokens_input is not None or r.tokens_output is not None]
        for r in token_results:
            if r.tokens_input is not None:
                total_tokens_input += r.tokens_input
            if r.tokens_output is not None:
                total_tokens_output += r.tokens_output
        total_tokens_count = total_tokens_input + total_tokens_output
        
        avg_tokens_input = total_tokens_input / len(token_results) if token_results else 0.0
        avg_tokens_output = total_tokens_output / len(token_results) if token_results else 0.0
        avg_tokens_total = total_tokens_count / len(token_results) if token_results else 0.0
        
        # Phase 3: Retrieval metrics (for RAG-based methodologies)
        retrieval_results = [r for r in self.results if r.retrieval_recall is not None]
        retrieval_tests = len(retrieval_results)
        retrieval_recall_count = sum(1 for r in retrieval_results if r.retrieval_recall)
        retrieval_recall_rate = retrieval_recall_count / retrieval_tests if retrieval_tests > 0 else 0.0
        
        # Average rank of correct tool when it was retrieved
        rank_results = [r.retrieval_rank for r in retrieval_results if r.retrieval_rank is not None]
        avg_retrieval_rank = sum(rank_results) / len(rank_results) if rank_results else 0.0
        
        # Phase 4: Clarification metrics
        ambiguous_results = [r for r in self.results if r.is_ambiguous_test]
        ambiguous_tests = len(ambiguous_results)
        ambiguous_correct = sum(1 for r in ambiguous_results if r.clarification_correct)
        clarification_accuracy = ambiguous_correct / ambiguous_tests if ambiguous_tests > 0 else 0.0
        
        # Average clarification score for ambiguous tests
        clarification_scores = [r.clarification_score for r in ambiguous_results if r.clarification_score is not None]
        avg_clarification_score = sum(clarification_scores) / len(clarification_scores) if clarification_scores else 0.0
        
        # Total clarification requests across all tests
        clarification_requests = sum(1 for r in self.results if r.clarification_requested)
        
        # False clarifications (clarification on non-ambiguous tests)
        non_ambiguous_results = [r for r in self.results if not r.is_ambiguous_test]
        false_clarification_count = sum(1 for r in non_ambiguous_results if r.clarification_requested)
        false_clarification_rate = false_clarification_count / len(non_ambiguous_results) if non_ambiguous_results else 0.0
        
        return EvaluationResult(
            total_tests=total,
            tool_correct=correct,
            tool_incorrect=incorrect,
            no_tool_called=no_call,
            errors=errors,
            test_results=self.results.copy(),
            avg_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            category_accuracy=category_accuracy,
            experiment_config=experiment_config or {},
            # Methodology-specific
            methodology=methodology,
            avg_steps_per_call=avg_steps,
            total_backtracks=total_backtracks,
            avg_backtracks_per_call=avg_backtracks,
            declined_tool_calls=declined_count,
            category_selection_accuracy=category_selection_accuracy,
            # No-tool metrics
            no_tool_tests=no_tool_tests,
            no_tool_correct=no_tool_correct,
            false_positive_count=false_positive_count,
            false_positive_rate=false_positive_rate,
            # Multi-tool metrics
            multi_tool_tests=multi_tool_tests,
            multi_tool_correct=multi_tool_correct,
            avg_completion_rate=avg_completion_rate,
            avg_sequence_accuracy=avg_sequence_accuracy,
            avg_extra_calls=avg_extra_calls,
            # Parameter accuracy
            params_tested=params_tested,
            params_correct=params_correct_count,
            params_accuracy=params_accuracy,
            # Phase 2 methodology metrics
            fallback_rate=fallback_rate,
            method_used_distribution=method_used_distribution,
            avg_num_fallbacks=avg_num_fallbacks,
            avg_confidence_score=avg_confidence_score,
            adaptive_k_stats=adaptive_k_stats,
            adaptive_strategy_distribution=adaptive_strategy_distribution,
            # Phase 3: Token usage metrics
            total_tokens_input=total_tokens_input,
            total_tokens_output=total_tokens_output,
            total_tokens=total_tokens_count,
            avg_tokens_input=avg_tokens_input,
            avg_tokens_output=avg_tokens_output,
            avg_tokens_total=avg_tokens_total,
            # Phase 3: Retrieval metrics
            retrieval_tests=retrieval_tests,
            retrieval_recall_count=retrieval_recall_count,
            retrieval_recall_rate=retrieval_recall_rate,
            avg_retrieval_rank=avg_retrieval_rank,
            # Phase 4: Clarification metrics
            ambiguous_tests=ambiguous_tests,
            ambiguous_correct=ambiguous_correct,
            clarification_accuracy=clarification_accuracy,
            avg_clarification_score=avg_clarification_score,
            false_clarification_count=false_clarification_count,
            false_clarification_rate=false_clarification_rate,
            clarification_requests=clarification_requests,
        )
    
    def reset(self) -> None:
        """Clear all stored results."""
        self.results = []
    
    def _check_params(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any]
    ) -> bool:
        """Check if actual parameters match expected (simple equality check)."""
        for key, value in expected.items():
            if key not in actual:
                return False
            if actual[key] != value:
                return False
        return True
    
    def _check_params_detailed(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any]
    ) -> tuple[bool, dict[str, bool]]:
        """
        Check if actual parameters match expected with detailed per-parameter results.
        
        Supports:
        - Exact matching
        - Type coercion (string "123" matches int 123)
        - Case-insensitive string matching
        
        Args:
            expected: Expected parameter values
            actual: Actual parameter values from model
            
        Returns:
            Tuple of (all_correct, {param_name: correct})
        """
        param_details = {}
        all_correct = True
        
        for key, expected_value in expected.items():
            if key not in actual:
                param_details[key] = False
                all_correct = False
                continue
            
            actual_value = actual[key]
            
            # Try exact match first
            if actual_value == expected_value:
                param_details[key] = True
                continue
            
            # Try type coercion for common cases
            matched = False
            
            # String to number coercion
            if isinstance(expected_value, (int, float)) and isinstance(actual_value, str):
                try:
                    if isinstance(expected_value, int):
                        matched = int(actual_value) == expected_value
                    else:
                        matched = float(actual_value) == expected_value
                except (ValueError, TypeError):
                    pass
            
            # Number to string coercion
            elif isinstance(expected_value, str) and isinstance(actual_value, (int, float)):
                matched = str(actual_value) == expected_value
            
            # Case-insensitive string matching
            elif isinstance(expected_value, str) and isinstance(actual_value, str):
                matched = expected_value.lower() == actual_value.lower()
            
            # Boolean coercion
            elif isinstance(expected_value, bool):
                if isinstance(actual_value, str):
                    matched = actual_value.lower() in ("true", "1", "yes") if expected_value else actual_value.lower() in ("false", "0", "no")
                elif isinstance(actual_value, (int, float)):
                    matched = bool(actual_value) == expected_value
            
            param_details[key] = matched
            if not matched:
                all_correct = False
        
        return all_correct, param_details
