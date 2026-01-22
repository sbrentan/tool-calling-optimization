#!/usr/bin/env python
"""
Script to reorganize experiment configs based on completion status.

This script:
1. Identifies which experiments in plan/ have not been run (missing results)
2. Moves incomplete experiments to other_configs/
3. Merges plan_2/ configs into plan/ with appropriate numbering

Usage:
    python scripts/reorganize_configs.py --dry-run  # See what would happen
    python scripts/reorganize_configs.py            # Actually reorganize
"""
import sys
import shutil
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="Reorganize experiment configs based on completion status")
console = Console()


def get_result_experiment_names(results_dir: Path) -> set[str]:
    """Extract experiment names from result files."""
    names = set()
    for f in results_dir.glob("*_summary.json"):
        # Result files are named like: phase1_mcp_100tools_medium_20260107_182759_summary.json
        # We need to extract: phase1_mcp_100tools_medium
        name = f.stem.replace("_summary", "")
        # Remove timestamp (last two underscore-separated parts: date and time)
        parts = name.rsplit("_", 2)
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
            name = parts[0]
        names.add(name)
    return names


def get_config_experiment_name(config_path: Path) -> str:
    """Extract experiment name from config filename."""
    # Config files are named like: 01_phase1_mcp_100tools_medium.yaml
    # We need to extract: phase1_mcp_100tools_medium
    name = config_path.stem
    # Remove numeric prefix (e.g., "01_")
    parts = name.split("_", 1)
    if len(parts) >= 2 and parts[0].isdigit():
        name = parts[1]
    return name


def identify_incomplete_configs(plan_dir: Path, results_dir: Path) -> tuple[list[Path], list[Path]]:
    """
    Identify which configs have been run and which haven't.
    
    Returns:
        (complete_configs, incomplete_configs)
    """
    result_names = get_result_experiment_names(results_dir)
    
    complete = []
    incomplete = []
    
    for config in sorted(plan_dir.glob("*.yaml")):
        exp_name = get_config_experiment_name(config)
        if exp_name in result_names:
            complete.append(config)
        else:
            incomplete.append(config)
    
    return complete, incomplete


def get_next_config_number(plan_dir: Path) -> int:
    """Get the next available config number."""
    max_num = 0
    for config in plan_dir.glob("*.yaml"):
        parts = config.stem.split("_", 1)
        if len(parts) >= 1 and parts[0].isdigit():
            num = int(parts[0])
            max_num = max(max_num, num)
    return max_num + 1


@app.command()
def check(
    plan_dir: str = typer.Option("plan", "--plan-dir", "-p", help="Plan directory name"),
    results_dir: str = typer.Option("plan", "--results-dir", "-r", help="Results directory name"),
):
    """Check which experiments are incomplete without making changes."""
    plan_path = project_root / "experiments" / plan_dir
    results_path = project_root / "experiments" / "results" / results_dir
    
    if not plan_path.exists():
        console.print(f"[red]Plan directory not found: {plan_path}[/red]")
        raise typer.Exit(1)
    
    complete, incomplete = identify_incomplete_configs(plan_path, results_path)
    
    console.print(Panel.fit(
        f"[bold]Experiment Status Check[/bold]\n"
        f"Plan directory: experiments/{plan_dir}/\n"
        f"Results directory: experiments/results/{results_dir}/\n"
        f"\n[green]Complete:[/green] {len(complete)}\n"
        f"[yellow]Incomplete:[/yellow] {len(incomplete)}",
        title="Summary"
    ))
    
    if incomplete:
        console.print("\n[yellow]Incomplete experiments:[/yellow]")
        table = Table()
        table.add_column("Config File", style="cyan")
        table.add_column("Experiment Name", style="yellow")
        
        for config in incomplete:
            exp_name = get_config_experiment_name(config)
            table.add_row(config.name, exp_name)
        
        console.print(table)
    
    if complete:
        console.print(f"\n[green]{len(complete)} experiments completed successfully[/green]")


@app.command()
def reorganize(
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be done without making changes"),
    skip_confirmation: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
    move_to_other: bool = typer.Option(False, "--move-to-other/--no-move-to-other", help="Move incomplete configs to other_configs/")
):
    """
    Reorganize configs by moving incomplete experiments to other_configs/
    and merging plan_2/ configs into plan/.
    """
    plan_path = project_root / "experiments" / "plan"
    plan2_path = project_root / "experiments" / "plan_3"
    other_path = project_root / "experiments" / "other_configs"
    results_path = project_root / "experiments" / "results" / "plan"
    
    # Check existence
    if not plan_path.exists():
        console.print(f"[red]Plan directory not found: {plan_path}[/red]")
        raise typer.Exit(1)
    
    # Identify incomplete configs
    complete, incomplete = identify_incomplete_configs(plan_path, results_path)
    
    console.print(Panel.fit(
        f"[bold]Reorganization Plan[/bold]\n"
        f"\n[green]Complete experiments:[/green] {len(complete)}\n"
        f"[yellow]Incomplete experiments:[/yellow] {len(incomplete)}\n"
        f"[blue]Plan_2 configs to merge:[/blue] {len(list(plan2_path.glob('*.yaml'))) if plan2_path.exists() else 0}",
        title="Summary"
    ))
    
    if dry_run:
        console.print("\n[yellow]DRY RUN MODE - No changes will be made[/yellow]\n")
    
    # Show incomplete experiments to be moved
    if move_to_other and incomplete:
        console.print("\n[yellow]Incomplete experiments to move to other_configs/:[/yellow]")
        for config in incomplete:
            console.print(f"  {config.name}")
    
    # Show plan_2 configs to merge
    if plan2_path.exists():
        plan2_configs = list(plan2_path.glob("*.yaml"))
        if plan2_configs:
            console.print(f"\n[blue]Plan_2 configs to merge into plan/:[/blue]")
            for config in sorted(plan2_configs)[:10]:
                console.print(f"  {config.name}")
            if len(plan2_configs) > 10:
                console.print(f"  ... and {len(plan2_configs) - 10} more")
    
    if dry_run:
        console.print("\n[yellow]Run without --dry-run to apply changes[/yellow]")
        return
    
    # Confirmation
    if not skip_confirmation:
        if move_to_other and incomplete:
            proceed = typer.confirm(
                f"\nMove {len(incomplete)} incomplete configs to other_configs/?",
                default=True
            )
            if not proceed:
                console.print("[yellow]Skipping incomplete config move[/yellow]")
                incomplete = []
        
        if plan2_path.exists() and list(plan2_path.glob("*.yaml")):
            proceed = typer.confirm(
                f"\nMerge plan_2/ configs into plan/?",
                default=True
            )
            if not proceed:
                console.print("[yellow]Skipping plan_2 merge[/yellow]")
                plan2_path = None
    
    # Move incomplete configs
    if move_to_other and incomplete:
        other_path.mkdir(parents=True, exist_ok=True)
        console.print(f"\n[yellow]Moving {len(incomplete)} incomplete configs to {other_path}[/yellow]")
        
        for config in incomplete:
            dest = other_path / config.name
            shutil.move(str(config), str(dest))
            console.print(f"  Moved: {config.name}")
    
    # Merge plan_2 configs
    if plan2_path and plan2_path.exists():
        plan2_configs = sorted(plan2_path.glob("*.yaml"))
        if plan2_configs:
            console.print(f"\n[blue]Merging {len(plan2_configs)} configs from plan_2/[/blue]")
            
            next_num = get_next_config_number(plan_path)
            
            for config in plan2_configs:
                # Generate new name with sequential numbering
                exp_name = get_config_experiment_name(config)
                new_name = f"{next_num:02d}_{exp_name}.yaml"
                dest = plan_path / new_name
                
                shutil.copy(str(config), str(dest))
                console.print(f"  Copied: {config.name} -> {new_name}")
                next_num += 1
    
    console.print("\n[green]Reorganization complete![/green]")
    
    # Show summary
    final_complete, final_incomplete = identify_incomplete_configs(plan_path, results_path)
    console.print(f"\nFinal state: {len(final_complete)} complete, {len(final_incomplete)} incomplete in plan/")


if __name__ == "__main__":
    app()
