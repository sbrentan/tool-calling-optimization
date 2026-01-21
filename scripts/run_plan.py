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

app = typer.Typer(help="Batch runner for experiment plan configurations")
console = Console()


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


def display_config_summary(config_path: Path, index: int, total: int) -> None:
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
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not parse config {config_path.name}: {e}[/yellow]")


def run_single_experiment(config_path: Path, run_config: RunConfig = None) -> dict:
    """Run a single experiment and return results."""
    # Import here to avoid circular imports and speed up --list
    from scripts.run_experiment import run_experiment
    
    config = ExperimentConfig.from_yaml(config_path)
    
    # Apply run config overrides if provided
    if run_config:
        config = apply_run_config_to_experiment(config, run_config)
    
    console.print(f"\n[bold blue]Running: {config.name}[/bold blue]")
    console.print(f"Configuration: {config.num_tools} tools, {config.methodology} methodology")
    console.print(f"Model: {config.model}, Seed: {config.seed}")
    
    start_time = datetime.now()
    
    try:
        results = run_experiment(config)
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
            "results": results
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
            "error": str(e)
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
    results = []
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
        if i < start_from:
            continue
        
        for run_idx, run_config in enumerate(run_list):
            experiment_num += 1
            
            console.print(f"\n{'='*60}")
            if is_multi_run:
                console.print(f"[bold]Config {i}/{total_configs}, Run {run_idx + 1}/{num_runs}[/bold]")
                console.print(f"[dim]Run: {run_config.name} (model={run_config.model}, seed={run_config.seed})[/dim]")
            else:
                console.print(f"[bold]Config {i}/{total_configs}[/bold]")
                console.print(f"[dim]Single run mode (using environment variables/defaults)[/dim]")
            display_config_summary(config_path, i, total_configs)
            
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
                result = run_single_experiment(config_path, run_config)
                results.append(result)
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
                user_aborted = True
                break
            
            # Show progress
            completed = len([r for r in results if r["status"] == "success"])
            failed = len([r for r in results if r["status"] == "error"])
            remaining = total_experiments - experiment_num - (skipped * num_runs)
            
            console.print(f"\n[dim]Progress: {completed} completed, {failed} failed, ~{remaining} remaining[/dim]")
        
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
        
        # Show hint for resuming if aborted
        if user_aborted:
            resume_from = (len([r for r in results if r["status"] == "success"]) // num_runs) + skipped + 1
            console.print(f"\n[cyan]To resume from where you left off, run:[/cyan]")
            console.print(f"  python scripts/run_plan.py --start-from {resume_from} --plan-dir {plan_dir}")


@app.command()
def list_experiments(
    plan_dir: str = typer.Option("plan", "--plan-dir", "-d",
                                  help="Plan directory under experiments/ (default: plan)")
):
    """List all experiment configurations in the plan folder."""
    run(list_only=True, plan_dir=plan_dir)


if __name__ == "__main__":
    app()
