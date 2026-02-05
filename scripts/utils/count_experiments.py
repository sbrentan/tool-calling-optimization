#!/usr/bin/env python
"""
Experiment Count Summary Script

This script parses YAML config files in experiments/plan/ and generates a summary
showing how many tests exist for each (methodology, tool_count, doc_length) combination.

Usage:
    python scripts/count_experiments.py
    python scripts/count_experiments.py --plan-dir experiments/plan_3
    python scripts/count_experiments.py --output docs/EXPERIMENT_SUMMARY.md
"""
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import typer
import yaml
from loguru import logger

app = typer.Typer(help="Count experiments by methodology, tool count, and verbosity")

# Known verbosity levels in order
DOC_LENGTH_ORDER = ["minimal", "medium", "verbose"]

# Methodology display names
METHODOLOGY_DISPLAY_NAMES = {
    "mcp": "MCP (Baseline)",
    "clustering": "Clustering",
    "rag": "RAG",
    "adaptive_rag": "Adaptive RAG",
    "hybrid": "Hybrid",
}


def parse_experiment_config(filepath: Path) -> dict:
    """Parse a YAML experiment config file and extract key parameters."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return {
            "name": config.get("name", filepath.stem),
            "methodology": config.get("methodology"),
            "num_tools": config.get("num_tools"),
            "doc_length": config.get("doc_length", "medium"),  # default to medium
            "prompt_type": config.get("prompt_type", "concise"),
            "filepath": str(filepath),
        }
    except Exception as e:
        logger.warning(f"Failed to parse {filepath}: {e}")
        return None


def count_experiments(plan_dir: Path) -> dict:
    """
    Count experiments by methodology, tool_count, and doc_length.
    
    Returns a nested dict: methodology -> tool_count -> doc_length -> count
    """
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    experiment_list = []
    
    yaml_files = sorted(plan_dir.glob("*.yaml"))
    
    for filepath in yaml_files:
        if filepath.name.startswith("EXPERIMENTATION"):
            continue  # Skip plan documentation files
            
        config = parse_experiment_config(filepath)
        if config and config["methodology"] and config["num_tools"]:
            methodology = config["methodology"]
            num_tools = config["num_tools"]
            doc_length = config["doc_length"]
            
            counts[methodology][num_tools][doc_length] += 1
            experiment_list.append(config)
    
    return counts, experiment_list


def generate_markdown_summary(counts: dict, experiment_list: list, plan_dir: Path) -> str:
    """Generate a markdown summary document."""
    lines = [
        "# Experiment Summary",
        "",
        f"**Source directory:** `{plan_dir}`",
        f"**Total experiments:** {len(experiment_list)}",
        "",
        "This document summarizes the number of experiment configurations by methodology, tool count, and documentation verbosity level.",
        "",
    ]
    
    # Get all unique tool counts across all methodologies
    all_tool_counts = set()
    for meth_data in counts.values():
        all_tool_counts.update(meth_data.keys())
    all_tool_counts = sorted(all_tool_counts)
    
    # Summary table: methodology x tool_count (total tests)
    lines.append("## Overview: Tests per Methodology × Tool Count")
    lines.append("")
    
    # Header
    header = "| Methodology |" + " | ".join(str(tc) for tc in all_tool_counts) + " | Total |"
    separator = "|-------------|" + " | ".join(["---:"] * len(all_tool_counts)) + " | ---: |"
    lines.append(header)
    lines.append(separator)
    
    methodology_order = ["mcp", "clustering", "rag", "adaptive_rag", "hybrid"]
    for meth in methodology_order:
        if meth not in counts:
            continue
        display_name = METHODOLOGY_DISPLAY_NAMES.get(meth, meth)
        row = [display_name]
        total = 0
        for tc in all_tool_counts:
            cell_count = sum(counts[meth][tc].values()) if tc in counts[meth] else 0
            row.append(str(cell_count) if cell_count > 0 else "-")
            total += cell_count
        row.append(str(total))
        lines.append("| " + " | ".join(row) + " |")
    
    lines.append("")
    
    # Detailed tables per methodology
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Breakdown by Methodology")
    lines.append("")
    
    for meth in methodology_order:
        if meth not in counts:
            continue
            
        display_name = METHODOLOGY_DISPLAY_NAMES.get(meth, meth)
        lines.append(f"### {display_name}")
        lines.append("")
        
        meth_tool_counts = sorted(counts[meth].keys())
        
        # Header: Tool Count | minimal | medium | verbose | Total
        header = "| Tool Count |" + " | ".join(dl.capitalize() for dl in DOC_LENGTH_ORDER) + " | Total |"
        separator = "| ---: |" + " | ".join(["---:"] * len(DOC_LENGTH_ORDER)) + " | ---: |"
        lines.append(header)
        lines.append(separator)
        
        for tc in meth_tool_counts:
            row = [str(tc)]
            total = 0
            for dl in DOC_LENGTH_ORDER:
                cell_count = counts[meth][tc].get(dl, 0)
                row.append(str(cell_count) if cell_count > 0 else "-")
                total += cell_count
            row.append(str(total))
            lines.append("| " + " | ".join(row) + " |")
        
        lines.append("")
    
    # Verbosity coverage section
    lines.append("---")
    lines.append("")
    lines.append("## Verbosity Coverage Analysis")
    lines.append("")
    lines.append("This section shows which (methodology, tool_count) pairs have complete verbosity coverage (all 3 levels).")
    lines.append("")
    
    complete_pairs = []
    incomplete_pairs = []
    
    for meth in methodology_order:
        if meth not in counts:
            continue
        for tc in sorted(counts[meth].keys()):
            available = set(counts[meth][tc].keys())
            if available == set(DOC_LENGTH_ORDER):
                complete_pairs.append((meth, tc))
            else:
                missing = set(DOC_LENGTH_ORDER) - available
                incomplete_pairs.append((meth, tc, missing))
    
    lines.append("### ✅ Complete Coverage (all 3 verbosity levels)")
    lines.append("")
    if complete_pairs:
        for meth, tc in complete_pairs:
            display_name = METHODOLOGY_DISPLAY_NAMES.get(meth, meth)
            lines.append(f"- {display_name} @ {tc} tools")
    else:
        lines.append("*None*")
    lines.append("")
    
    lines.append("### ⚠️ Partial Coverage")
    lines.append("")
    if incomplete_pairs:
        for meth, tc, missing in incomplete_pairs:
            display_name = METHODOLOGY_DISPLAY_NAMES.get(meth, meth)
            missing_str = ", ".join(sorted(missing))
            lines.append(f"- {display_name} @ {tc} tools - missing: {missing_str}")
    else:
        lines.append("*None*")
    lines.append("")
    
    return "\n".join(lines)


@app.command()
def main(
    plan_dir: Path = typer.Option(
        Path("experiments/plan"),
        "--plan-dir", "-d",
        help="Directory containing experiment YAML configs"
    ),
    output: Optional[Path] = typer.Option(
        Path("docs/EXPERIMENT_SUMMARY.md"),
        "--output", "-o",
        help="Output file path for the markdown summary"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """Count experiments and generate summary document."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")
    
    if not plan_dir.exists():
        logger.error(f"Plan directory does not exist: {plan_dir}")
        raise typer.Exit(1)
    
    logger.info(f"Counting experiments in {plan_dir}")
    counts, experiment_list = count_experiments(plan_dir)
    
    if not experiment_list:
        logger.error("No valid experiment configs found")
        raise typer.Exit(1)
    
    logger.info(f"Found {len(experiment_list)} experiment configurations")
    
    # Print quick summary to console
    print(f"\n📊 Experiment Summary for {plan_dir}")
    print(f"   Total configs: {len(experiment_list)}")
    print(f"   Methodologies: {', '.join(counts.keys())}")
    
    # Generate markdown
    markdown = generate_markdown_summary(counts, experiment_list, plan_dir)
    
    # Write output
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w', encoding='utf-8') as f:
            f.write(markdown)
        logger.info(f"Summary written to {output}")
        print(f"\n✅ Summary saved to: {output}")
    else:
        # Print to stdout
        print("\n" + markdown)


if __name__ == "__main__":
    app()
