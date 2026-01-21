"""
Plan configuration for multi-run experiments.

Allows specifying multiple runs with different seeds, models, and sample counts
to enable cross-run aggregation and statistical analysis.

Example YAML format:
    runs:
      - name: run_1
        model: llama-3.3-70b
        seed: 42
        num_samples: 10
      - name: run_2
        model: gpt-4o
        seed: 123
        num_samples: 10
      - name: run_3
        model: llama-3.3-70b
        seed: 456
        num_samples: 10
    
    # Optional: Plan-level defaults
    defaults:
        num_samples: 10
"""
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import yaml

from loguru import logger


@dataclass
class RunConfig:
    """Configuration for a single run within a plan."""
    name: str
    model: str = "llama-3.3-70b"
    seed: int = 42
    num_samples: Optional[int] = None  # None = all samples
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "model": self.model,
            "seed": self.seed,
            "num_samples": self.num_samples,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any], defaults: dict[str, Any] = None) -> "RunConfig":
        """Create from dictionary with optional defaults."""
        defaults = defaults or {}
        
        # Apply defaults for missing values
        merged = {**defaults, **data}
        
        return cls(
            name=merged.get("name", f"run_{merged.get('seed', 42)}"),
            model=merged.get("model", "llama-3.3-70b"),
            seed=merged.get("seed", 42),
            num_samples=merged.get("num_samples"),
        )


@dataclass
class PlanConfig:
    """
    Configuration for a multi-run experiment plan.
    
    Specifies multiple runs with different seeds/models/samples to enable
    cross-run aggregation and statistical analysis.
    
    The run_id_prefix allows distinguishing between different plan config executions.
    For example, if you run plan_config_A with prefix "a" and plan_config_B with prefix "b",
    the runs will be named "run_a_1", "run_a_2", "run_b_1", "run_b_2", etc.
    This allows aggregating results from multiple plan config runs.
    """
    runs: list[RunConfig] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)
    run_id_prefix: str = ""  # Optional prefix to distinguish plan config runs
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "runs": [r.to_dict() for r in self.runs],
            "defaults": self.defaults,
            "run_id_prefix": self.run_id_prefix,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanConfig":
        """Create from dictionary."""
        defaults = data.get("defaults", {})
        run_id_prefix = data.get("run_id_prefix", "")
        runs = [
            RunConfig.from_dict(r, defaults) 
            for r in data.get("runs", [])
        ]
        return cls(runs=runs, defaults=defaults, run_id_prefix=run_id_prefix)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "PlanConfig":
        """Load plan configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        if data is None:
            return cls()
        
        return cls.from_dict(data)
    
    @classmethod
    def from_cli_args(
        cls, 
        models: list[str], 
        seeds: list[int], 
        num_samples: list[int] = None,
        run_id_prefix: str = ""
    ) -> "PlanConfig":
        """
        Create plan config from CLI arguments.
        
        Args:
            models: List of model names (must match length of seeds)
            seeds: List of seeds (must match length of models)
            num_samples: Optional list of sample counts (must match length if provided)
            run_id_prefix: Optional prefix for run names to distinguish plan config runs
            
        Returns:
            PlanConfig with runs for each model/seed pair
            
        Raises:
            ValueError: If list lengths don't match
        """
        if len(models) != len(seeds):
            raise ValueError(
                f"Number of models ({len(models)}) must match number of seeds ({len(seeds)})"
            )
        
        if num_samples and len(num_samples) != len(models):
            raise ValueError(
                f"Number of num_samples ({len(num_samples)}) must match number of models ({len(models)})"
            )
        
        runs = []
        prefix = f"{run_id_prefix}_" if run_id_prefix else ""
        for i, (model, seed) in enumerate(zip(models, seeds)):
            samples = num_samples[i] if num_samples else None
            runs.append(RunConfig(
                name=f"run_{prefix}{i+1}",
                model=model,
                seed=seed,
                num_samples=samples,
            ))
        
        return cls(runs=runs, run_id_prefix=run_id_prefix)
    
    @classmethod
    def single_run(cls, model: str = "llama-3.3-70b", seed: int = 42, num_samples: int = None) -> "PlanConfig":
        """Create a plan config with a single run (default/backward compatible). No run suffix added."""
        return cls(runs=[])
    
    def save_yaml(self, path: str | Path) -> None:
        """Save plan configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def __len__(self) -> int:
        """Return number of runs."""
        return len(self.runs)
    
    def __iter__(self):
        """Iterate over runs."""
        return iter(self.runs)
    
    def __getitem__(self, index: int) -> RunConfig:
        """Get run by index."""
        return self.runs[index]


def create_example_plan_config(output_path: str | Path = None) -> PlanConfig:
    """
    Create an example plan configuration file.
    
    Args:
        output_path: Optional path to save the example config
        
    Returns:
        Example PlanConfig
    """
    config = PlanConfig(
        runs=[
            RunConfig(name="run_1_llama_seed42", model="llama-3.3-70b", seed=42, num_samples=10),
            RunConfig(name="run_2_llama_seed123", model="llama-3.3-70b", seed=123, num_samples=10),
            RunConfig(name="run_3_llama_seed456", model="llama-3.3-70b", seed=456, num_samples=10),
        ],
        defaults={"num_samples": 10}
    )
    
    if output_path:
        config.save_yaml(output_path)
        logger.info(f"Saved example plan config to: {output_path}")
    
    return config
