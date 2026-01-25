"""
Experiment configuration dataclass.

Defines all configurable parameters for tool calling experiments.

Environment Variables:
    EXPERIMENT_NUM_SAMPLES: Override num_test_samples (use "__env__" in YAML)
    EXPERIMENT_MODEL: Override model (use "__env__" in YAML)
    EXPERIMENT_SEED: Override seed (use "__env__" in YAML)
"""
from dataclasses import dataclass, field
from typing import Any, Optional
import os
import yaml
from pathlib import Path


# Environment variable mappings for configurable parameters
ENV_VAR_MAPPINGS = {
    "num_test_samples": ("EXPERIMENT_NUM_SAMPLES", lambda x: int(x) if x else None),
    "model": ("EXPERIMENT_MODEL", lambda x: x if x else "llama-3.3-70b"),
    "seed": ("EXPERIMENT_SEED", lambda x: int(x) if x else 42),
}


def _resolve_env_vars(data: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve environment variable placeholders in config data.
    
    Values set to "__env__" will be replaced with the corresponding
    environment variable value, or the default if not set.
    """
    resolved = data.copy()
    
    for field_name, (env_var, converter) in ENV_VAR_MAPPINGS.items():
        if field_name in resolved and resolved[field_name] == "__env__":
            env_value = os.getenv(env_var)
            resolved[field_name] = converter(env_value)
    
    return resolved


@dataclass
class ExperimentConfig:
    """
    Configuration for a tool calling experiment.
    
    Attributes:
        name: Experiment name for identification
        
        # Tool configuration
        num_tools: Number of tools to make available
        doc_length: Documentation verbosity level
        num_similar_tools: Number of similar/distractor tools
        categories: Tool categories to include
        
        # Model configuration
        model: Model to use (provider auto-detected from name)
        provider: Explicit provider (gemini/cerebras/openai)
        temperature: Sampling temperature
        
        # Test configuration
        num_test_samples: Number of test cases to run
        seed: Random seed for reproducibility
        include_multi_tool: Include multi-tool test cases
        include_no_tool: Include no-tool test cases
        
        # Output configuration
        output_dir: Directory for results
        save_raw_responses: Whether to save raw API responses
    """
    # Experiment identification
    name: str = "baseline"
    description: str = ""
    
    # Tool configuration
    num_tools: int = 10
    doc_length: str = "medium"  # minimal, short, medium, long, verbose
    prompt_type: str = "concise"  # concise or clear
    num_similar_tools: int = 0
    categories: Optional[list[str]] = None
    
    # Model configuration
    model: str = "llama-3.3-70b"  # Default to Cerebras (free tier)
    provider: Optional[str] = None  # Auto-detect from model if not specified
    temperature: float = 0.0
    
    # Test configuration
    num_test_samples: Optional[int] = None  # None = test all generated tools
    seed: int = 42
    include_multi_tool: bool = False  # Include multi-tool test scenarios
    include_no_tool: bool = False  # Include no-tool test scenarios (negative tests)
    include_ambiguous: bool = False  # Include ambiguous test scenarios (clarification tests)
    test_categories: Optional[list[str]] = None  # Limit tests to specific categories (None = all)
    validate_params: bool = True  # Enable parameter validation using prompts_mapping.json
    tools_dir: Optional[str] = None  # Custom tools directory path (None = default)
    
    # Output configuration
    output_dir: str = "experiments/results"
    save_raw_responses: bool = False
    
    # Methodology configuration
    methodology: str = "mcp"  # mcp, clustering, rag, etc.
    max_steps: int = 10  # Max steps for multi-step methodologies
    allow_backtrack: bool = True  # Allow backtracking in step-based methodologies
    allow_no_tool_call: bool = False  # Allow LLM to decline calling any tool
    allow_clarification: bool = False  # Allow LLM to request clarification
    max_clarification_candidates: int = 3  # Max candidates for full score (else penalized)
    
    # RAG methodology configuration
    rag_config: Optional[dict[str, Any]] = None  # RAG-specific settings
    
    # Phase 2 methodology configurations
    hybrid_config: Optional[dict[str, Any]] = None  # Hybrid methodology settings
    adaptive_rag_config: Optional[dict[str, Any]] = None  # Adaptive RAG settings
    confidence_config: Optional[dict[str, Any]] = None  # Confidence fallback settings
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "num_tools": self.num_tools,
            "doc_length": self.doc_length,
            "prompt_type": self.prompt_type,
            "num_similar_tools": self.num_similar_tools,
            "categories": self.categories,
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "num_test_samples": self.num_test_samples,
            "seed": self.seed,
            "include_multi_tool": self.include_multi_tool,
            "include_no_tool": self.include_no_tool,
            "include_ambiguous": self.include_ambiguous,
            "test_categories": self.test_categories,
            "validate_params": self.validate_params,
            "tools_dir": self.tools_dir,
            "output_dir": self.output_dir,
            "save_raw_responses": self.save_raw_responses,
            "methodology": self.methodology,
            "max_steps": self.max_steps,
            "allow_backtrack": self.allow_backtrack,
            "allow_no_tool_call": self.allow_no_tool_call,
            "allow_clarification": self.allow_clarification,
            "max_clarification_candidates": self.max_clarification_candidates,
            "rag_config": self.rag_config,
            "hybrid_config": self.hybrid_config,
            "adaptive_rag_config": self.adaptive_rag_config,
            "confidence_config": self.confidence_config,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary, resolving environment variable placeholders."""
        resolved_data = _resolve_env_vars(data)
        return cls(**{k: v for k, v in resolved_data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from YAML file, resolving environment variable placeholders."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
