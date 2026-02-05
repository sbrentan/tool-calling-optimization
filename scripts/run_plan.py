#!/usr/bin/env python
"""
Batch runner for experiment plan configurations with multi-run support.

This script runs all YAML configurations in the experiments/plan folder
in alphabetical order (by numeric prefix), with user confirmation between tests.
Supports multiple runs with different seeds/models for statistical aggregation.

Usage:
    # Run all experiments from the beginning (single run with defaults)
    python scripts/run_plan.py
    
    # Start from a specific test number (e.g., resume from test 15)
    python scripts/run_plan.py --start-from 15
    
    # Run without confirmation prompts (for automated runs)
    python scripts/run_plan.py --no-confirm
    
    # List all experiments without running
    python scripts/run_plan.py --list
    
    # Dry run (show what would be executed)
    python scripts/run_plan.py --dry-run
    
    # Multi-run mode with plan config file
    python scripts/run_plan.py --plan-config experiments/plan_runs.yaml
    
    # Multi-run mode with CLI args (same length required)
    python scripts/run_plan.py --models llama-3.3-70b,gpt-4o --seeds 42,123
    
    # Custom timeout and retry settings
    python scripts/run_plan.py --timeout 120 --max-retries 5
    
    # Resume from a saved progress file
    python scripts/run_plan.py --resume tmp/progress/progress_plan_20240101_120000.json
    
    # Resume from specific config and run index
    python scripts/run_plan.py --start-from 15 --start-from-run 1

Environment Variables:
    EXPERIMENT_NUM_SAMPLES: Number of test samples (default: 10)
    EXPERIMENT_MODEL: Model to use (default: llama-3.3-70b)
    EXPERIMENT_SEED: Random seed (default: 42)
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import signal
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import partial
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import after path setup
from src.experiments.config import ExperimentConfig
from src.experiments.plan_config import PlanConfig, RunConfig
from src.clients.rate_limit_handler import UserAbortError
from src.clients.api_key_manager import reset_all_rotations
from src.clients.timeout_utils import set_interrupt, clear_interrupt, is_interrupted

app = typer.Typer(help="Batch runner for experiment plan configurations")
console = Console()

# METHODOLOGIES_TO_SKIP = ["clustering", "hybrid", "rag", "adaptive_rag", "mcp"]  # List of methodologies to skip during execution
METHODOLOGIES_TO_SKIP = []  # List of methodologies to skip during execution

# Default timeout and retry settings
DEFAULT_TIMEOUT_SECONDS = 60  # 1 minute timeout per experiment
DEFAULT_MAX_RETRIES = 3  # Maximum retries on timeout

# Progress state file location
PROGRESS_DIR = project_root / "tmp" / "progress"
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


class InterruptState:
    """
    Global state to track current execution position for Ctrl+C handling.
    This allows us to save progress when the user interrupts.
    """
    def __init__(self):
        self.plan_dir: str = "plan"
        self.current_config_idx: int = 0
        self.current_run_idx: int = 0
        self.current_test_idx: int = 0  # Track which test we're on within an experiment
        self.results: list[dict] = []
        self.plan_config: PlanConfig = None
        self.configs: list[Path] = []
        self.enabled: bool = False  # Only save on interrupt if enabled
        self.test_results: list[dict] = []  # Serialized test results from evaluator
    
    def update(
        self,
        plan_dir: str,
        current_config_idx: int,
        current_run_idx: int,
        results: list[dict],
        plan_config: PlanConfig,
        configs: list[Path],
        current_test_idx: int = 0,
        test_results: list[dict] = None,
    ):
        """Update the current state."""
        self.plan_dir = plan_dir
        self.current_config_idx = current_config_idx
        self.current_run_idx = current_run_idx
        self.current_test_idx = current_test_idx
        self.results = results
        self.plan_config = plan_config
        self.configs = configs
        self.enabled = True
        self.test_results = test_results or []
    
    def update_test_idx(self, test_idx: int):
        """Update just the current test index (called from run_experiment)."""
        self.current_test_idx = test_idx
    
    def update_test_results(self, test_results: list[dict]):
        """Update the current test results (called from run_experiment after each test)."""
        self.test_results = test_results


# Global interrupt state
_interrupt_state = InterruptState()


def sigint_handler(signum, frame):
    """Handle Ctrl+C by saving progress before exiting."""
    # Set the global interrupt flag to signal all threads to stop
    set_interrupt()
    
    console.print("\n\n[bold yellow]Interrupt received (Ctrl+C)![/bold yellow]")
    
    if _interrupt_state.enabled and _interrupt_state.configs:
        console.print("[yellow]Saving progress before exit...[/yellow]")
        try:
            save_progress_state(
                PROGRESS_DIR,
                _interrupt_state.plan_dir,
                _interrupt_state.current_config_idx,
                _interrupt_state.current_run_idx,
                _interrupt_state.current_test_idx,
                _interrupt_state.results,
                _interrupt_state.plan_config,
                _interrupt_state.configs,
                _interrupt_state.test_results,  # Include accumulated test results
            )
        except Exception as e:
            console.print(f"[red]Failed to save progress: {e}[/red]")
    else:
        console.print("[dim]No progress to save (not in experiment loop)[/dim]")
    
    console.print("[yellow]Exiting...[/yellow]")
    # Use os._exit to force immediate exit (sys.exit waits for threads)
    os._exit(130)


class ExperimentTimeoutError(Exception):
    """Raised when an experiment times out."""
    pass


def save_progress_state(
    progress_dir: Path,
    plan_dir: str,
    current_config_idx: int,
    current_run_idx: int,
    current_test_idx: int,
    results: list[dict],
    plan_config: "PlanConfig",
    configs: list[Path],
    test_results: list[dict] = None,
) -> Path:
    """
    Save current progress state to allow resuming.
    
    Args:
        test_results: Serialized test results from the evaluator (individual test outcomes)
    
    Returns:
        Path to the saved progress file.
    """
    progress_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_file = progress_dir / f"progress_{plan_dir}_{timestamp}.json"
    
    state = {
        "timestamp": datetime.now().isoformat(),
        "plan_dir": plan_dir,
        "current_config_idx": current_config_idx,  # 1-indexed config number
        "current_run_idx": current_run_idx,  # 0-indexed run within config
        "current_test_idx": current_test_idx,  # 0-indexed test within experiment
        "total_configs": len(configs),
        "config_files": [str(c) for c in configs],
        "plan_config": plan_config.to_dict() if plan_config else None,
        "results_so_far": results,
        "test_results": test_results or [],  # Individual test results for current experiment
    }
    
    with open(progress_file, "w") as f:
        json.dump(state, f, indent=2, default=str)
    
    console.print(f"\n[cyan]Progress saved to: {progress_file}[/cyan]")
    console.print(f"[cyan]To resume from config {current_config_idx}, run {current_run_idx + 1}, test {current_test_idx + 1}, use:[/cyan]")
    console.print(f"  python scripts/run_plan.py --resume {progress_file}")
    console.print(f"[cyan]Or to skip the stuck test:[/cyan]")
    console.print(f"  python scripts/run_plan.py --resume {progress_file} --skip-to-test {current_test_idx + 2}")
    console.print(f"[cyan]Or to skip to next config:[/cyan]")
    console.print(f"  python scripts/run_plan.py --plan-dir {plan_dir} --start-from {current_config_idx + 1}")
    
    return progress_file


def load_progress_state(progress_file: Path) -> dict:
    """
    Load progress state from a saved file.
    
    Returns:
        Dictionary with saved state.
    """
    if not progress_file.exists():
        raise FileNotFoundError(f"Progress file not found: {progress_file}")
    
    with open(progress_file, "r") as f:
        return json.load(f)


def run_with_timeout(func, timeout_seconds: int, *args, **kwargs):
    """
    Run a function with a timeout.
    
    Args:
        func: Function to run
        timeout_seconds: Timeout in seconds
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Function result
        
    Raises:
        ExperimentTimeoutError: If the function times out
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            raise ExperimentTimeoutError(
                f"Experiment timed out after {timeout_seconds} seconds"
            )


def get_plan_configs(plan_dir: str = "plan") -> list[Path]:
    """Get all YAML config files from experiments/plan folder, sorted by name."""
    plan_path = project_root / "experiments" / plan_dir
    if not plan_path.exists():
        console.print(f"[red]Error: Plan directory not found: {plan_path}[/red]")
        raise typer.Exit(1)
    
    configs = sorted(plan_path.glob("*.yaml"))
    # Filter out non-experiment files (like EXPERIMENTATION_PLAN.md)
    configs = [c for c in configs if c.suffix == ".yaml"]
    return configs


def load_plan_config(
    plan_config_path: Path = None, 
    models: str = None, 
    seeds: str = None, 
    num_samples: str = None,
    run_id_prefix: str = None
) -> PlanConfig:
    """
    Load plan configuration from file or CLI arguments.
    
    Args:
        plan_config_path: Path to plan config YAML file
        models: Comma-separated list of models
        seeds: Comma-separated list of seeds
        num_samples: Comma-separated list of sample counts
        run_id_prefix: Prefix for run names (to distinguish plan config runs)
        
    Returns:
        PlanConfig with runs to execute. Empty runs means single-run mode (no suffix).
    """
    if plan_config_path:
        config = PlanConfig.from_yaml(plan_config_path)
        # Apply run_id_prefix if specified via CLI (overrides YAML)
        if run_id_prefix:
            config.run_id_prefix = run_id_prefix
            # Update run names with prefix
            for run in config.runs:
                if not run.name.startswith(f"run_{run_id_prefix}"):
                    run.name = f"run_{run_id_prefix}_{run.name.replace('run_', '')}"
        return config
    
    if models and seeds:
        model_list = [m.strip() for m in models.split(",")]
        seed_list = [int(s.strip()) for s in seeds.split(",")]
        sample_list = [int(n.strip()) for n in num_samples.split(",")] if num_samples else None
        return PlanConfig.from_cli_args(model_list, seed_list, sample_list, run_id_prefix or "")
    
    # Default: single run mode (no suffix added to experiment names)
    # This maintains backward compatibility with existing results
    return PlanConfig.single_run()


def apply_run_config_to_experiment(config: ExperimentConfig, run_config: RunConfig) -> ExperimentConfig:
    """
    Apply run configuration to an experiment config.
    
    Creates a new config with model, seed, and num_samples from the run config.
    The experiment name is also updated to include run info.
    """
    # Create a copy of the config dict
    config_dict = config.to_dict()
    
    # Apply run overrides
    config_dict["model"] = run_config.model
    config_dict["seed"] = run_config.seed
    if run_config.num_samples is not None:
        config_dict["num_test_samples"] = run_config.num_samples
    
    # Update name to include run info
    original_name = config_dict["name"]
    config_dict["name"] = f"{original_name}_{run_config.name}"
    
    return ExperimentConfig.from_dict(config_dict)


def display_config_summary(config_path: Path, index: int, total: int) -> ExperimentConfig:
    """Display a summary of the configuration to be run."""
    try:
        config = ExperimentConfig.from_yaml(config_path)
        
        # Extract phase from filename
        filename = config_path.stem
        phase = filename.split("_")[0] if "_" in filename else "unknown"
        
        table = Table(title=f"Experiment {index}/{total}: {config.name}")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("File", config_path.name)
        table.add_row("Phase", phase)
        table.add_row("Description", config.description or "N/A")
        table.add_row("Methodology", config.methodology)
        table.add_row("Num Tools", str(config.num_tools))
        table.add_row("Doc Length", config.doc_length)
        table.add_row("Prompt Type", config.prompt_type)
        table.add_row("Model", config.model)
        table.add_row("Test Samples", str(config.num_test_samples) if config.num_test_samples else "all")
        
        # Show methodology-specific config
        if config.methodology == "rag" and config.rag_config:
            table.add_row("RAG top_k", str(config.rag_config.get("top_k", "N/A")))
        elif config.methodology == "adaptive_rag" and config.adaptive_rag_config:
            table.add_row("Adaptive min_k/max_k", 
                         f"{config.adaptive_rag_config.get('min_k', 'N/A')}/{config.adaptive_rag_config.get('max_k', 'N/A')}")
        elif config.methodology == "hybrid" and config.hybrid_config:
            table.add_row("Hybrid top_k_categories", str(config.hybrid_config.get("top_k_categories", "N/A")))
        elif config.methodology == "confidence" and config.confidence_config:
            table.add_row("Confidence thresholds", 
                         f"RAG={config.confidence_config.get('rag_confidence_threshold', 'N/A')}, "
                         f"Cluster={config.confidence_config.get('clustering_confidence_threshold', 'N/A')}")
        
        if config.num_similar_tools > 0:
            table.add_row("Similar Tools", str(config.num_similar_tools))
        if config.include_no_tool:
            table.add_row("Include No-Tool", "Yes")
        
        console.print(table)

        return config
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not parse config {config_path.name}: {e}[/yellow]")


def run_single_experiment(
    config_path: Path, 
    run_config: RunConfig = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    start_from_test: int = 0,
    previous_test_results: list[dict] = None,
) -> dict:
    """
    Run a single experiment and return results.
    
    Args:
        config_path: Path to experiment config YAML
        run_config: Optional run configuration overrides
        timeout_seconds: Timeout per attempt in seconds (currently unused, per-test timeout handles this)
        max_retries: Maximum number of retries on timeout (currently unused, per-test timeout handles this)
        start_from_test: Test index to start from (0-indexed) for resuming mid-experiment
        previous_test_results: Serialized test results from previous run to restore when resuming
        
    Returns:
        Dictionary with experiment results
    """
    # Import here to avoid circular imports and speed up --list
    from scripts.run_experiment import run_experiment, set_interrupt_state
    
    # Pass interrupt state to run_experiment for test-level tracking
    set_interrupt_state(_interrupt_state)
    
    config = ExperimentConfig.from_yaml(config_path)
    
    # Apply run config overrides if provided
    if run_config:
        config = apply_run_config_to_experiment(config, run_config)
    
    console.print(f"\n[bold blue]Running: {config.name}[/bold blue]")
    console.print(f"Configuration: {config.num_tools} tools, {config.methodology} methodology")
    console.print(f"Model: {config.model}, Seed: {config.seed}")
    if start_from_test > 0:
        console.print(f"[dim]Resuming from test {start_from_test + 1}[/dim]")
        if previous_test_results:
            console.print(f"[dim]Restoring {len(previous_test_results)} previous test results[/dim]")
    
    start_time = datetime.now()
    
    try:
        # Run experiment (per-test timeouts are handled inside run_experiment)
        results = run_experiment(
            config, 
            start_from_test=start_from_test,
            previous_test_results=previous_test_results,
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display quick summary
        if results:
            accuracy = results.get("accuracy", 0) * 100
            console.print(f"[green]✓ Completed in {duration:.1f}s - Accuracy: {accuracy:.1f}%[/green]")
        
        return {
            "status": "success",
            "config": config_path.name,
            "experiment_name": config.name,
            "run_config": run_config.to_dict() if run_config else None,
            "duration": duration,
            "results": results,
        }
    
    except UserAbortError as e:
        # User chose to abort after rate limit - re-raise to stop batch
        console.print(f"\n[yellow]User aborted: {e}[/yellow]")
        raise
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        console.print(f"[red]✗ Failed after {duration:.1f}s: {e}[/red]")
        return {
            "status": "error",
            "config": config_path.name,
            "duration": duration,
            "error": str(e),
        }


def save_batch_summary(results: list[dict], output_path: Path, plan_config: PlanConfig = None) -> None:
    """Save a summary of all batch results."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "total_duration": sum(r["duration"] for r in results),
        "plan_config": plan_config.to_dict() if plan_config else None,
        "experiments": results
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    console.print(f"\n[green]Batch summary saved to: {output_path}[/green]")


@app.command()
def run(
    start_from: int = typer.Option(1, "--start-from", "-s", 
                                    help="Start from this experiment number (1-indexed)"),
    start_from_run: int = typer.Option(0, "--start-from-run",
                                        help="Start from this run index within the config (0-indexed, use with --start-from)"),
    no_confirm: bool = typer.Option(False, "--no-confirm", "-y",
                                     help="Run without confirmation prompts"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n",
                                  help="Show what would be executed without running"),
    list_only: bool = typer.Option(False, "--list", "-l",
                                    help="List all experiments without running"),
    verbose: bool = typer.Option(False, "--verbose", "-v",
                                  help="Enable verbose output"),
    plan_dir: str = typer.Option("plan", "--plan-dir", "-d",
                                  help="Plan directory under experiments/ (default: plan)"),
    plan_config: Path = typer.Option(None, "--plan-config", "-p",
                                      help="Path to plan config YAML for multi-run mode"),
    models: str = typer.Option(None, "--models", "-m",
                                help="Comma-separated list of models for multi-run mode"),
    seeds: str = typer.Option(None, "--seeds",
                               help="Comma-separated list of seeds for multi-run mode"),
    num_samples: str = typer.Option(None, "--num-samples",
                                     help="Comma-separated list of sample counts for multi-run mode"),
    run_id_prefix: str = typer.Option(None, "--run-prefix",
                                       help="Prefix for run names (to distinguish multiple plan config runs)"),
    timeout: int = typer.Option(DEFAULT_TIMEOUT_SECONDS, "--timeout", "-t",
                                 help="Timeout in seconds for each experiment (default: 60)"),
    max_retries: int = typer.Option(DEFAULT_MAX_RETRIES, "--max-retries", "-r",
                                     help="Maximum retries on timeout (default: 3)"),
    resume: Path = typer.Option(None, "--resume",
                                 help="Resume from a saved progress file"),
    skip_to_test: int = typer.Option(None, "--skip-to-test",
                                      help="Skip to this test number (1-indexed) when resuming, to skip a stuck test"),
    save_progress_on_error: bool = typer.Option(True, "--save-progress/--no-save-progress",
                                                  help="Save progress state on error/timeout for later resume"),
):
    """
    Run all experiment plan configurations in sequence.
    
    Experiments are run in alphabetical order by filename (numeric prefix).
    User confirmation is requested before each experiment unless --no-confirm is set.
    Use --start-from to resume from a specific experiment number.
    
    Multi-run mode (for statistical aggregation):
        --plan-config experiments/plan_runs.yaml  # Use YAML file
        --models llama-3.3-70b,gpt-4o --seeds 42,123  # Use CLI args
        --run-prefix a  # Add prefix to distinguish from other plan runs
    
    Timeout and retry:
        --timeout 120  # 2 minute timeout per experiment (default: 60)
        --max-retries 5  # 5 retries on timeout (default: 3)
    
    Resume from saved progress:
        --resume tmp/progress/progress_plan_20240101_120000.json
        --skip-to-test 203  # Skip stuck test 202, start from test 203
        --start-from 15 --start-from-run 1  # Start from config 15, run index 1

    Environment Variables (used if no multi-run mode specified):
        EXPERIMENT_NUM_SAMPLES: Number of test samples (default: None - all samples)
        EXPERIMENT_MODEL: Model to use (default: llama-3.3-70b)
        EXPERIMENT_SEED: Random seed (default: 42)
    """
    # Setup logging
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, 
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    # Clear any previous interrupt flag
    clear_interrupt()
    
    # Register signal handler for Ctrl+C
    # On Windows, this works best when the main thread isn't blocked
    # Our new timeout mechanism uses short sleep intervals to allow signal processing
    signal.signal(signal.SIGINT, sigint_handler)
    
    # On Windows, also try to handle CTRL_C_EVENT and CTRL_BREAK_EVENT
    if sys.platform == 'win32':
        try:
            signal.signal(signal.SIGBREAK, sigint_handler)
        except (AttributeError, ValueError):
            pass  # SIGBREAK not available on all Windows configurations
    
    # Handle resume from progress file
    resume_run_idx = start_from_run
    previous_results = []
    previous_test_results = []  # Test results to restore when resuming mid-experiment
    
    if resume:
        try:
            console.print(f"[cyan]Loading progress from: {resume}[/cyan]")
            state = load_progress_state(resume)
            
            # Override settings from saved state
            plan_dir = state.get("plan_dir", plan_dir)
            start_from = state.get("current_config_idx", start_from)
            resume_run_idx = state.get("current_run_idx", 0)
            resume_test_idx = state.get("current_test_idx", 0)
            previous_results = state.get("results_so_far", [])
            previous_test_results = state.get("test_results", [])  # Load saved test results
            
            # Handle --skip-to-test override (1-indexed from CLI, convert to 0-indexed)
            if skip_to_test is not None:
                resume_test_idx = skip_to_test - 1  # Convert to 0-indexed
                console.print(f"[yellow]Skipping to test {skip_to_test} (overriding saved test index)[/yellow]")
            
            # Load plan config from state if available
            if state.get("plan_config"):
                run_plan_config = PlanConfig.from_dict(state["plan_config"])
                console.print(f"[green]Resumed from config {start_from}, run {resume_run_idx + 1}, test {resume_test_idx + 1}[/green]")
                console.print(f"[dim]Loaded {len(previous_results)} previous experiment results[/dim]")
                if previous_test_results:
                    console.print(f"[dim]Loaded {len(previous_test_results)} previous test results to restore[/dim]")
            else:
                # Load fresh plan config
                try:
                    run_plan_config = load_plan_config(plan_config, models, seeds, num_samples, run_id_prefix)
                except ValueError as e:
                    console.print(f"[red]Error loading plan config: {e}[/red]")
                    raise typer.Exit(1)
        except FileNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    else:
        resume_test_idx = 0
        # Handle --skip-to-test without resume
        if skip_to_test is not None:
            resume_test_idx = skip_to_test - 1
            console.print(f"[yellow]Starting from test {skip_to_test}[/yellow]")
        
        # Load plan configuration
        try:
            run_plan_config = load_plan_config(plan_config, models, seeds, num_samples, run_id_prefix)
        except ValueError as e:
            console.print(f"[red]Error loading plan config: {e}[/red]")
            raise typer.Exit(1)
    
    # Get all configs
    configs = get_plan_configs(plan_dir)
    total_configs = len(configs)
    
    # Determine if we're in multi-run mode or single-run mode
    is_multi_run = len(run_plan_config) > 0
    num_runs = len(run_plan_config) if is_multi_run else 1
    total_experiments = total_configs * num_runs
    
    if total_configs == 0:
        console.print(f"[red]No experiment configurations found in experiments/{plan_dir}/[/red]")
        raise typer.Exit(1)
    
    # Display configuration summary
    if is_multi_run:
        runs_info = "\n".join([f"  {r.name}: model={r.model}, seed={r.seed}, samples={r.num_samples or 'all'}" 
                               for r in run_plan_config])
        mode_info = f"Mode: Multi-run ({num_runs} runs per config)"
    else:
        runs_info = f"  Single run using environment variables/defaults"
        mode_info = "Mode: Single-run (no run suffix added)"
    
    console.print(Panel.fit(
        f"[bold]Execution Configuration[/bold]\n"
        f"Plan Directory: experiments/{plan_dir}/\n"
        f"Total Configs: {total_configs}\n"
        f"{mode_info}\n"
        f"Total Experiments: {total_experiments}\n"
        f"Timeout: {timeout}s per experiment\n"
        f"Max Retries: {max_retries}\n"
        f"Save Progress: {save_progress_on_error}\n"
        f"\n[bold]Runs:[/bold]\n" + runs_info,
        title="Configuration"
    ))
    
    # List mode
    if list_only:
        console.print(f"\n[bold]Found {total_configs} experiment configurations × {num_runs} runs = {total_experiments} total:[/bold]\n")
        
        table = Table()
        table.add_column("#", style="dim")
        table.add_column("Config File", style="cyan")
        table.add_column("Phase", style="yellow")
        table.add_column("Methodology", style="green")
        table.add_column("Tools", style="blue")
        
        for i, config_path in enumerate(configs, 1):
            try:
                config = ExperimentConfig.from_yaml(config_path)
                phase = config_path.stem.split("_")[0]
                table.add_row(
                    str(i),
                    config_path.name,
                    phase,
                    config.methodology,
                    str(config.num_tools)
                )
            except Exception:
                table.add_row(str(i), config_path.name, "?", "?", "?")
        
        console.print(table)
        
        if num_runs > 1:
            console.print(f"\n[dim]Each config will be run {num_runs} times with different seeds/models[/dim]")
        
        raise typer.Exit(0)
    
    # Validate start_from
    if start_from < 1 or start_from > total_configs:
        console.print(f"[red]Error: --start-from must be between 1 and {total_configs}[/red]")
        raise typer.Exit(1)
    
    # Display plan summary
    console.print(f"\n[bold]Experiment Plan: {total_configs} configurations × {num_runs} runs[/bold]")
    console.print(f"Starting from config: {start_from}/{total_configs}")
    
    if dry_run:
        console.print("\n[yellow]DRY RUN MODE - No experiments will be executed[/yellow]\n")
    
    # Prepare results tracking
    results = list(previous_results)  # Start with any previous results from resume
    skipped = start_from - 1
    user_aborted = False
    
    # Build run list - for single-run mode, use a placeholder
    if is_multi_run:
        run_list = list(run_plan_config)
    else:
        # Single-run mode: one run with no suffix (uses env vars/defaults)
        run_list = [None]  # Placeholder for single run
    
    # Run experiments (config × run matrix)
    experiment_num = 0
    for i, config_path in enumerate(configs, 1):
        skip_config = False
        if i < start_from:
            continue
        
        # Check for interrupt at the start of each config
        if is_interrupted():
            console.print("[yellow]Interrupt detected, stopping batch execution...[/yellow]")
            user_aborted = True
            break
        
        for run_idx, run_config in enumerate(run_list):
            # Skip runs that were already completed when resuming
            if i == start_from and run_idx < resume_run_idx:
                continue
            
            # Check for interrupt at the start of each run
            if is_interrupted():
                console.print("[yellow]Interrupt detected, stopping batch execution...[/yellow]")
                user_aborted = True
                break
                
            experiment_num += 1
            
            console.print(f"\n{'='*60}")
            if is_multi_run:
                console.print(f"[bold]Config {i}/{total_configs}, Run {run_idx + 1}/{num_runs}[/bold]")
                console.print(f"[dim]Run: {run_config.name} (model={run_config.model}, seed={run_config.seed})[/dim]")
            else:
                console.print(f"[bold]Config {i}/{total_configs}[/bold]")
                console.print(f"[dim]Single run mode (using environment variables/defaults)[/dim]")
            config = display_config_summary(config_path, i, total_configs)
            if config.methodology in METHODOLOGIES_TO_SKIP:
                console.print(f"[yellow]Warning: configs with {config.methodology} methodology are skipped[/yellow]")
                skipped += 1
                skip_config = True
                break
            
            if dry_run:
                console.print("[yellow]Would run this experiment (dry run)[/yellow]")
                continue
            
            # Confirmation prompt (only for first run of each config)
            if not no_confirm and run_idx == 0:
                proceed = typer.confirm(
                    f"\nProceed with config {i}/{total_configs} ({num_runs} run(s))?",
                    default=True
                )
                if not proceed:
                    console.print("[yellow]Skipping this config...[/yellow]")
                    for _ in range(num_runs):
                        result_dict = {
                            "status": "skipped",
                            "config": config_path.name,
                            "duration": 0
                        }
                        if run_config is not None:
                            result_dict["run_config"] = run_config.to_dict()
                        results.append(result_dict)
                    
                    # Ask if user wants to continue with remaining
                    if i < total_configs:
                        continue_remaining = typer.confirm(
                            "Continue with remaining configs?",
                            default=True
                        )
                        if not continue_remaining:
                            console.print("[yellow]Stopping batch execution.[/yellow]")
                            user_aborted = True
                    break  # Skip remaining runs for this config
            
            # Run experiment
            try:
                # Update interrupt state before running (for Ctrl+C handling)
                _interrupt_state.update(
                    plan_dir=plan_dir,
                    current_config_idx=i,
                    current_run_idx=run_idx,
                    results=results,
                    plan_config=run_plan_config,
                    configs=configs,
                )
                
                # Reset API key rotation state before each experiment to avoid desync
                reset_all_rotations()
                
                # Determine start_from_test and previous_test_results for this experiment
                # Only apply resume values to the first experiment after resume
                current_start_test = 0
                current_previous_test_results = None
                if i == start_from and run_idx == resume_run_idx and resume_test_idx > 0:
                    current_start_test = resume_test_idx
                    current_previous_test_results = previous_test_results
                    # Reset resume values after using them once
                    resume_test_idx = 0
                    previous_test_results = []
                
                result = run_single_experiment(
                    config_path, 
                    run_config,
                    timeout_seconds=timeout,
                    max_retries=max_retries,
                    start_from_test=current_start_test,
                    previous_test_results=current_previous_test_results,
                )
                results.append(result)
                
                # Clear test results from interrupt state after experiment completes
                _interrupt_state.test_results = []
                
                # Save progress on timeout if enabled
                if result.get("status") == "timeout" and save_progress_on_error:
                    save_progress_state(
                        PROGRESS_DIR,
                        plan_dir,
                        i,
                        run_idx + 1,  # Next run to attempt
                        0,  # Start from beginning of next experiment
                        results,
                        run_plan_config,
                        configs,
                    )
                    console.print("[yellow]Continuing to next experiment...[/yellow]")
                    
            except UserAbortError:
                # User chose to abort after rate limit
                console.print("\n[bold red]Batch execution aborted by user due to rate limit.[/bold red]")
                result_dict = {
                    "status": "aborted",
                    "config": config_path.name,
                    "duration": 0,
                    "error": "User aborted due to rate limit"
                }
                if run_config is not None:
                    result_dict["run_config"] = run_config.to_dict()
                results.append(result_dict)
                
                # Save progress on abort if enabled
                if save_progress_on_error:
                    save_progress_state(
                        PROGRESS_DIR,
                        plan_dir,
                        i,
                        run_idx,
                        0,  # Test index not applicable for abort
                        results,
                        run_plan_config,
                        configs,
                    )
                    
                user_aborted = True
                break
            
            # Show progress
            completed = len([r for r in results if r["status"] == "success"])
            failed = len([r for r in results if r["status"] == "error"])
            timed_out = len([r for r in results if r["status"] == "timeout"])
            remaining = total_experiments - experiment_num - (skipped * num_runs)
            
            console.print(f"\n[dim]Progress: {completed} completed, {failed} failed, {timed_out} timed out, ~{remaining} remaining[/dim]")
        
        if skip_config:
            continue

        if user_aborted:
            break
    
    # Final summary
    if not dry_run and results:
        console.print(f"\n{'='*60}")
        if user_aborted:
            console.print("[bold yellow]Batch Execution Aborted by User[/bold yellow]\n")
        else:
            console.print("[bold]Batch Execution Complete[/bold]\n")
        
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        timed_out = sum(1 for r in results if r["status"] == "timeout")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")
        aborted_count = sum(1 for r in results if r["status"] == "aborted")
        total_duration = sum(r["duration"] for r in results)
        
        summary_table = Table(title="Execution Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Experiments", str(len(results) + (skipped * num_runs)))
        summary_table.add_row("Started From Config", str(start_from))
        summary_table.add_row("Runs Per Config", str(num_runs))
        summary_table.add_row("Successful", f"[green]{successful}[/green]")
        summary_table.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else "0")
        summary_table.add_row("Timed Out", f"[yellow]{timed_out}[/yellow]" if timed_out > 0 else "0")
        summary_table.add_row("Skipped", f"[yellow]{skipped_count + (skipped * num_runs)}[/yellow]")
        if aborted_count > 0:
            summary_table.add_row("Aborted", f"[yellow]{aborted_count}[/yellow]")
        summary_table.add_row("Total Duration", f"{total_duration:.1f}s")
        
        console.print(summary_table)
        
        # Save batch summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = project_root / "experiments" / "results" / plan_dir / f"batch_summary_{timestamp}.json"
        save_batch_summary(results, summary_path, run_plan_config)
        
        # Show failed experiments if any
        if failed > 0:
            console.print("\n[red]Failed experiments:[/red]")
            for r in results:
                if r["status"] == "error":
                    run_info = f" ({r.get('run_config', {}).get('name', '')})" if r.get('run_config') else ""
                    console.print(f"  - {r['config']}{run_info}: {r.get('error', 'Unknown error')}")
        
        # Show timed out experiments if any
        if timed_out > 0:
            console.print("\n[yellow]Timed out experiments:[/yellow]")
            for r in results:
                if r["status"] == "timeout":
                    run_info = f" ({r.get('run_config', {}).get('name', '')})" if r.get('run_config') else ""
                    attempts = r.get('attempts', '?')
                    console.print(f"  - {r['config']}{run_info}: {attempts} attempts exhausted")
        
        # Show hint for resuming if aborted or check for saved progress files
        if user_aborted:
            console.print(f"\n[cyan]A progress file was saved. To resume, check:[/cyan]")
            console.print(f"  ls {PROGRESS_DIR}")
            console.print(f"\n[cyan]Then run:[/cyan]")
            console.print(f"  python scripts/run_plan.py --resume <progress_file.json>")


@app.command()
def list_experiments(
    plan_dir: str = typer.Option("plan", "--plan-dir", "-d",
                                  help="Plan directory under experiments/ (default: plan)")
):
    """List all experiment configurations in the plan folder."""
    run(list_only=True, plan_dir=plan_dir)


if __name__ == "__main__":
    app()
