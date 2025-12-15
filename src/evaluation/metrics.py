"""
Evaluation metrics for tool calling accuracy.

Measures:
- Tool selection accuracy (correct tool called)
- Parameter accuracy (correct parameters extracted)
- Latency statistics
"""
from typing import Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd
from loguru import logger

from src.tools.base import TestCase
from src.clients.gemini_client import CallResult


@dataclass
class TestResult:
    """Result of a single test case evaluation."""
    test_case: TestCase
    call_result: CallResult
    tool_correct: bool
    params_correct: Optional[bool] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            "prompt": self.test_case.prompt,
            "expected_tool": self.test_case.expected_tool,
            "called_tool": self.call_result.called_tool,
            "tool_correct": self.tool_correct,
            "params_correct": self.params_correct,
            "latency_ms": self.call_result.latency_ms,
            "category": self.test_case.category,
            "difficulty": self.test_case.difficulty,
            "model": self.call_result.model,
            "error": self.call_result.error
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
        return {
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
            "experiment_config": self.experiment_config
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert detailed results to pandas DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.test_results])
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 50,
            "EVALUATION RESULTS",
            "=" * 50,
            f"Total Tests: {self.total_tests}",
            f"Tool Selection Accuracy: {self.accuracy:.2%}",
            f"Tool Call Rate: {self.call_rate:.2%}",
            "",
            f"Correct: {self.tool_correct}",
            f"Incorrect: {self.tool_incorrect}",
            f"No Tool Called: {self.no_tool_called}",
            f"Errors: {self.errors}",
            "",
            f"Avg Latency: {self.avg_latency_ms:.1f}ms",
            f"Min Latency: {self.min_latency_ms:.1f}ms",
            f"Max Latency: {self.max_latency_ms:.1f}ms",
        ]
        
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
    and computes various metrics.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results: list[TestResult] = []
    
    def evaluate_single(
        self,
        test_case: TestCase,
        call_result: CallResult
    ) -> TestResult:
        """
        Evaluate a single test case.
        
        Args:
            test_case: Expected test case
            call_result: Actual result from API call
            
        Returns:
            TestResult with evaluation details
        """
        # Check if correct tool was called
        tool_correct = (
            call_result.called_tool is not None and
            call_result.called_tool == test_case.expected_tool
        )
        
        # Check parameters if expected
        params_correct = None
        if tool_correct and test_case.expected_params is not None:
            params_correct = self._check_params(
                test_case.expected_params,
                call_result.called_args or {}
            )
        
        result = TestResult(
            test_case=test_case,
            call_result=call_result,
            tool_correct=tool_correct,
            params_correct=params_correct
        )
        
        self.results.append(result)
        return result
    
    def evaluate_batch(
        self,
        test_cases: list[TestCase],
        call_results: list[CallResult]
    ) -> list[TestResult]:
        """
        Evaluate a batch of test cases.
        
        Args:
            test_cases: List of expected test cases
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
            experiment_config=experiment_config or {}
        )
    
    def reset(self) -> None:
        """Clear all stored results."""
        self.results = []
    
    def _check_params(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any]
    ) -> bool:
        """Check if actual parameters match expected."""
        for key, value in expected.items():
            if key not in actual:
                return False
            if actual[key] != value:
                return False
        return True
