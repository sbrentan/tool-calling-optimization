#!/usr/bin/env python
"""
Batch runner for experiment plan configurations.

This script runs all YAML configurations in the experiments/plan folder
in alphabetical order (by numeric prefix), with user confirmation between tests.

Usage:
    # Run all experiments from the beginning
    python scripts/run_plan.py
    
    # Start from a specific test number (e.g., resume from test 15)
    python scripts/run_plan.py --start-from 15
    
    # Run without confirmation prompts (for automated runs)
    python scripts/run_plan.py --no-confirm
    
    # List all experiments without running
    python scripts/run_plan.py --list
    
    # Dry run (show what would be executed)
    python scripts/run_plan.py --dry-run

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
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import after path setup
from src.experiments.config import ExperimentConfig
from src.clients.rate_limit_handler import UserAbortError

app = typer.Typer(help="Batch runner for experiment plan configurations")
console = Console()


def get_plan_configs() -> list[Path]:
    """Get all YAML config files from experiments/plan folder, sorted by name."""
    plan_dir = project_root / "experiments" / "plan"
    if not plan_dir.exists():
        console.print(f"[red]Error: Plan directory not found: {plan_dir}[/red]")
        raise typer.Exit(1)
    
    configs = sorted(plan_dir.glob("*.yaml"))
    # Filter out non-experiment files (like EXPERIMENTATION_PLAN.md)
    configs = [c for c in configs if c.suffix == ".yaml"]
    return configs


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


def run_single_experiment(config_path: Path) -> dict:
    """Run a single experiment and return results."""
    # Import here to avoid circular imports and speed up --list
    from scripts.run_experiment import run_experiment
    
    config = ExperimentConfig.from_yaml(config_path)
    
    console.print(f"\n[bold blue]Running: {config.name}[/bold blue]")
    console.print(f"Configuration: {config.num_tools} tools, {config.methodology} methodology")
    
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


def save_batch_summary(results: list[dict], output_path: Path) -> None:
    """Save a summary of all batch results."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "total_duration": sum(r["duration"] for r in results),
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
                                  help="Enable verbose output")
):
    """
    Run all experiment plan configurations in sequence.
    
    Experiments are run in alphabetical order by filename (numeric prefix).
    User confirmation is requested before each experiment unless --no-confirm is set.
    Use --start-from to resume from a specific experiment number.

    Environment Variables:
        EXPERIMENT_NUM_SAMPLES: Number of test samples (default: None - all samples)
        EXPERIMENT_MODEL: Model to use (default: llama-3.3-70b)
        EXPERIMENT_SEED: Random seed (default: 42)
    """
    # Setup logging
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, 
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    # Get all configs
    configs = get_plan_configs()
    total = len(configs)
    
    if total == 0:
        console.print("[red]No experiment configurations found in experiments/plan/[/red]")
        raise typer.Exit(1)
    
    # Display environment configuration
    console.print(Panel.fit(
        f"[bold]Environment Configuration[/bold]\n"
        f"EXPERIMENT_NUM_SAMPLES: {os.getenv('EXPERIMENT_NUM_SAMPLES', 'None (all samples)')}\n"
        f"EXPERIMENT_MODEL: {os.getenv('EXPERIMENT_MODEL', 'llama-3.3-70b (default)')}\n"
        f"EXPERIMENT_SEED: {os.getenv('EXPERIMENT_SEED', '42 (default)')}",
        title="Environment"
    ))
    
    # List mode
    if list_only:
        console.print(f"\n[bold]Found {total} experiment configurations:[/bold]\n")
        
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
        raise typer.Exit(0)
    
    # Validate start_from
    if start_from < 1 or start_from > total:
        console.print(f"[red]Error: --start-from must be between 1 and {total}[/red]")
        raise typer.Exit(1)
    
    # Display plan summary
    console.print(f"\n[bold]Experiment Plan: {total} configurations[/bold]")
    console.print(f"Starting from experiment: {start_from}/{total}")
    
    if dry_run:
        console.print("\n[yellow]DRY RUN MODE - No experiments will be executed[/yellow]\n")
    
    # Prepare results tracking
    results = []
    skipped = start_from - 1
    user_aborted = False
    
    # Run experiments
    for i, config_path in enumerate(configs, 1):
        if i < start_from:
            continue
        
        console.print(f"\n{'='*60}")
        display_config_summary(config_path, i, total)
        
        if dry_run:
            console.print("[yellow]Would run this experiment (dry run)[/yellow]")
            continue
        
        # Confirmation prompt
        if not no_confirm:
            proceed = typer.confirm(
                f"\nProceed with experiment {i}/{total}?",
                default=True
            )
            if not proceed:
                console.print("[yellow]Skipping this experiment...[/yellow]")
                results.append({
                    "status": "skipped",
                    "config": config_path.name,
                    "duration": 0
                })
                
                # Ask if user wants to continue with remaining
                if i < total:
                    continue_remaining = typer.confirm(
                        "Continue with remaining experiments?",
                        default=True
                    )
                    if not continue_remaining:
                        console.print("[yellow]Stopping batch execution.[/yellow]")
                        break
                continue
        
        # Run experiment
        try:
            result = run_single_experiment(config_path)
            results.append(result)
        except UserAbortError:
            # User chose to abort after rate limit
            console.print("\n[bold red]Batch execution aborted by user due to rate limit.[/bold red]")
            results.append({
                "status": "aborted",
                "config": config_path.name,
                "duration": 0,
                "error": "User aborted due to rate limit"
            })
            user_aborted = True
            break
        
        # Show progress
        completed = len([r for r in results if r["status"] == "success"])
        failed = len([r for r in results if r["status"] == "error"])
        remaining = total - i
        
        console.print(f"\n[dim]Progress: {completed} completed, {failed} failed, {remaining} remaining[/dim]")
    
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
        
        summary_table.add_row("Total Experiments", str(len(results) + skipped))
        summary_table.add_row("Started From", str(start_from))
        summary_table.add_row("Successful", f"[green]{successful}[/green]")
        summary_table.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else "0")
        summary_table.add_row("Skipped", f"[yellow]{skipped_count + skipped}[/yellow]")
        if aborted_count > 0:
            summary_table.add_row("Aborted", f"[yellow]{aborted_count}[/yellow]")
        summary_table.add_row("Total Duration", f"{total_duration:.1f}s")
        
        console.print(summary_table)
        
        # Save batch summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = project_root / "experiments" / "results" / "plan" / f"batch_summary_{timestamp}.json"
        save_batch_summary(results, summary_path)
        
        # Show failed experiments if any
        if failed > 0:
            console.print("\n[red]Failed experiments:[/red]")
            for r in results:
                if r["status"] == "error":
                    console.print(f"  - {r['config']}: {r.get('error', 'Unknown error')}")
        
        # Show hint for resuming if aborted
        if user_aborted:
            resume_from = len([r for r in results if r["status"] == "success"]) + skipped + 1
            console.print(f"\n[cyan]To resume from where you left off, run:[/cyan]")
            console.print(f"  python scripts/run_plan.py --start-from {resume_from}")


@app.command()
def list_experiments():
    """List all experiment configurations in the plan folder."""
    run(list_only=True)


if __name__ == "__main__":
    app()
