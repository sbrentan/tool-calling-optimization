#!/usr/bin/env python
"""
Experiment Analysis and Visualization Script

This script provides comprehensive analysis and visualization capabilities:
1. Aggregated methodology comparison (heatmaps, radar charts)
2. Per-methodology parameter impact analysis
3. Scaling analysis (accuracy vs. tool count)
4. Error and robustness analysis
5. Latency analysis with outlier filtering
6. Automated report generation

Usage:
    # Generate all charts and report
    python scripts/analyze_results.py generate-report --results-dir experiments/results/plan
    
    # Generate specific visualization types
    python scripts/analyze_results.py overview --results-dir experiments/results/plan
    python scripts/analyze_results.py methodology-analysis --methodology rag
    python scripts/analyze_results.py scaling-analysis
    
    # Compare specific experiments
    python scripts/analyze_results.py compare --experiments exp1_summary.json exp2_summary.json
"""
import sys
import json
import re
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import typer
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

app = typer.Typer(help="Analyze and visualize experiment results")

# Methodology display names and colors for consistent styling
METHODOLOGY_COLORS = {
    "mcp": "#1f77b4",         # Blue
    "clustering": "#ff7f0e",  # Orange
    "rag": "#2ca02c",         # Green
    "adaptive_rag": "#d62728", # Red
    "hybrid": "#9467bd",       # Purple
    "confidence": "#8c564b",   # Brown
}

METHODOLOGY_DISPLAY_NAMES = {
    "mcp": "MCP (Baseline)",
    "clustering": "Clustering",
    "rag": "RAG",
    "adaptive_rag": "Adaptive RAG",
    "hybrid": "Hybrid",
    "confidence": "Confidence",
}

DOC_LENGTH_ORDER = ["minimal", "medium", "clear", "verbose"]


# =============================================================================
# Data Loading and Aggregation
# =============================================================================

def load_experiment_summary(filepath: Path) -> dict:
    """Load experiment summary JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_experiment_details(filepath: Path) -> pd.DataFrame:
    """Load experiment details CSV file."""
    return pd.read_csv(filepath)


def find_experiments(results_dir: Path, pattern: str = "*_summary.json") -> list[Path]:
    """Find all experiment summary files in a directory."""
    return sorted(results_dir.glob(pattern))


def parse_experiment_name(name: str) -> dict:
    """
    Parse experiment name to extract configuration parameters.
    
    Examples:
        phase1_mcp_100tools_medium -> {phase: 1, methodology: mcp, num_tools: 100, doc_length: medium}
        phase3_rag_200tools_k10_t02 -> {phase: 3, methodology: rag, num_tools: 200, top_k: 10, threshold: 0.2}
    """
    params = {}
    
    # Extract phase
    phase_match = re.search(r'phase(\d+)', name)
    if phase_match:
        params['phase'] = int(phase_match.group(1))
    
    # Extract num_tools
    tools_match = re.search(r'(\d+)tools', name)
    if tools_match:
        params['num_tools'] = int(tools_match.group(1))
    
    # Extract doc_length
    for doc_len in ['minimal', 'medium', 'clear', 'verbose']:
        if doc_len in name:
            params['doc_length'] = doc_len
            break
    
    # Extract RAG-specific parameters
    k_match = re.search(r'_k(\d+)', name)
    if k_match:
        params['top_k'] = int(k_match.group(1))
    
    threshold_match = re.search(r'_t0?(\d+)', name)
    if threshold_match:
        params['similarity_threshold'] = float(f"0.{threshold_match.group(1)}")
    
    # Extract adaptive RAG parameters
    if 'mink' in name:
        mink_match = re.search(r'mink(\d+)', name)
        if mink_match:
            params['min_k'] = int(mink_match.group(1))
    
    if 'maxk' in name:
        maxk_match = re.search(r'maxk(\d+)', name)
        if maxk_match:
            params['max_k'] = int(maxk_match.group(1))
    
    if 'drop' in name:
        drop_match = re.search(r'drop0?(\d+)', name)
        if drop_match:
            params['drop_threshold'] = float(f"0.{drop_match.group(1)}")
    
    if 'minsim' in name:
        minsim_match = re.search(r'minsim0?(\d+)', name)
        if minsim_match:
            params['min_similarity'] = float(f"0.{minsim_match.group(1)}")
    
    # Extract clustering parameters
    params['allow_backtrack'] = 'backtrack' in name and 'nobacktrack' not in name
    
    # Extract hybrid parameters
    cat_match = re.search(r'_cat(\d+)', name)
    if cat_match:
        params['top_k_categories'] = int(cat_match.group(1))
    
    # Robustness tests
    if 'similar' in name:
        similar_match = re.search(r'similar(\d+)', name)
        if similar_match:
            params['num_similar_tools'] = int(similar_match.group(1))
    
    params['is_no_tool_test'] = 'notool' in name
    
    return params


def load_all_experiments_as_dataframe(results_dir: Path) -> pd.DataFrame:
    """
    Load all experiments and aggregate into a unified DataFrame.
    
    Columns include:
    - experiment_name, methodology, num_tools, doc_length
    - accuracy, call_rate, avg_latency_ms
    - methodology-specific parameters (top_k, allow_backtrack, etc.)
    - methodology-specific metrics (category_accuracy, adaptive_k_stats, etc.)
    - run-specific fields (model, seed, run_name) for multi-run aggregation
    """
    rows = []
    
    for summary_path in find_experiments(results_dir):
        try:
            summary = load_experiment_summary(summary_path)
        except Exception as e:
            logger.warning(f"Failed to load {summary_path}: {e}")
            continue
        
        config = summary.get("experiment_config", {})
        exp_name = config.get("name", summary_path.stem)
        
        # Parse parameters from experiment name
        parsed = parse_experiment_name(exp_name)
        
        # Extract run info if present (e.g., "experiment_name_run_1" -> run_name="run_1")
        base_exp_name = exp_name
        run_name = None
        if "_run_" in exp_name:
            parts = exp_name.rsplit("_run_", 1)
            if len(parts) == 2:
                base_exp_name = parts[0]
                run_name = f"run_{parts[1]}"
        
        row = {
            "experiment_name": exp_name,
            "base_experiment_name": base_exp_name,  # Name without run suffix
            "run_name": run_name,
            "model": config.get("model", "unknown"),
            "seed": config.get("seed", 42),
            "file_path": str(summary_path),
            "methodology": summary.get("methodology", config.get("methodology", "unknown")),
            "num_tools": config.get("num_tools", parsed.get("num_tools", 0)),
            "doc_length": config.get("doc_length", parsed.get("doc_length", "medium")),
            "prompt_type": config.get("prompt_type", "concise"),
            
            # Core metrics
            "accuracy": summary.get("accuracy", 0.0),
            "call_rate": summary.get("call_rate", 0.0),
            "total_tests": summary.get("total_tests", 0),
            "tool_correct": summary.get("tool_correct", 0),
            "tool_incorrect": summary.get("tool_incorrect", 0),
            "no_tool_called": summary.get("no_tool_called", 0),
            "errors": summary.get("errors", 0),
            
            # Latency metrics
            "avg_latency_ms": summary.get("avg_latency_ms", 0.0),
            "min_latency_ms": summary.get("min_latency_ms", 0.0),
            "max_latency_ms": summary.get("max_latency_ms", 0.0),
            
            # Methodology-specific metrics
            "category_selection_accuracy": summary.get("category_selection_accuracy", 0.0),
            "avg_steps_per_call": summary.get("avg_steps_per_call", 0.0),
            "total_backtracks": summary.get("total_backtracks", 0),
            "fallback_rate": summary.get("fallback_rate", 0.0),
            
            # Retrieval metrics (may be 0 for older results)
            "retrieval_recall_rate": summary.get("retrieval_recall_rate", 0.0),
            "avg_retrieval_rank": summary.get("avg_retrieval_rank", 0.0),
            
            # Token metrics
            "total_tokens_input": summary.get("total_tokens_input", 0),
            "total_tokens_output": summary.get("total_tokens_output", 0),
            "total_tokens": summary.get("total_tokens", 0),
            "avg_tokens_input": summary.get("avg_tokens_input", 0.0),
            "avg_tokens_output": summary.get("avg_tokens_output", 0.0),
            "avg_tokens_total": summary.get("avg_tokens_total", 0.0),
            
            # Parsed parameters
            "phase": parsed.get("phase"),
            "top_k": parsed.get("top_k"),
            "similarity_threshold": parsed.get("similarity_threshold"),
            "allow_backtrack": parsed.get("allow_backtrack"),
            "top_k_categories": parsed.get("top_k_categories"),
            "min_k": parsed.get("min_k"),
            "max_k": parsed.get("max_k"),
            "drop_threshold": parsed.get("drop_threshold"),
            "min_similarity": parsed.get("min_similarity"),
            "num_similar_tools": parsed.get("num_similar_tools", 0),
            "is_no_tool_test": parsed.get("is_no_tool_test", False),
        }
        
        # Add adaptive RAG stats
        adaptive_stats = summary.get("adaptive_k_stats", {})
        row["adaptive_k_avg"] = adaptive_stats.get("avg_k", 0.0)
        row["adaptive_k_min"] = adaptive_stats.get("min_k", 0.0)
        row["adaptive_k_max"] = adaptive_stats.get("max_k", 0.0)
        
        # Add adaptive strategy distribution
        strategy_dist = summary.get("adaptive_strategy_distribution", {})
        row["adaptive_threshold_count"] = strategy_dist.get("threshold", 0)
        row["adaptive_bounded_count"] = strategy_dist.get("bounded", 0)
        
        # Store category_accuracy dict as JSON string
        row["category_accuracy_json"] = json.dumps(summary.get("category_accuracy", {}))
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by methodology and num_tools for consistent ordering
    if not df.empty:
        df = df.sort_values(["methodology", "num_tools", "experiment_name"]).reset_index(drop=True)
    
    return df


def aggregate_across_runs(df: pd.DataFrame, treat_single_as_run: bool = True) -> pd.DataFrame:
    """
    Aggregate experiment results across multiple runs (different seeds/models).
    
    Groups by base_experiment_name and computes mean/std for numeric metrics.
    This function handles:
    - Multi-run experiments (with _run_X suffix)
    - Single-run experiments (without suffix) - treated as their own run if treat_single_as_run=True
    - Multiple plan config runs (all runs with same base name are aggregated)
    
    Args:
        df: DataFrame with experiment results (may include multiple runs)
        treat_single_as_run: If True, experiments without run suffix are included in aggregation
                            with their base name. If False, they're kept separate.
        
    Returns:
        DataFrame with aggregated results, including _mean and _std columns
    """
    if df.empty:
        return df
    
    # Ensure we have the required columns
    if "base_experiment_name" not in df.columns:
        df = df.copy()
        df["base_experiment_name"] = df["experiment_name"]
    
    if "run_name" not in df.columns:
        df = df.copy()
        df["run_name"] = None
    
    # For experiments without run_name, use base_experiment_name as-is for grouping
    # This allows single-run results to be aggregated with multi-run results
    if treat_single_as_run:
        # Experiments without run suffix: base_experiment_name equals experiment_name
        # These can be aggregated with multi-run experiments of the same base name
        pass  # The grouping by base_experiment_name will work correctly
    else:
        # Keep single-run experiments separate by using their full name as base
        mask = df["run_name"].isna()
        df.loc[mask, "base_experiment_name"] = df.loc[mask, "experiment_name"]
    
    # Numeric columns to aggregate
    numeric_cols = [
        "accuracy", "call_rate", "avg_latency_ms", "min_latency_ms", "max_latency_ms",
        "total_tests", "tool_correct", "tool_incorrect", "no_tool_called", "errors",
        "category_selection_accuracy", "avg_steps_per_call", "total_backtracks",
        "fallback_rate", "retrieval_recall_rate", "avg_retrieval_rank",
        "avg_tokens_input", "avg_tokens_output", "avg_tokens_total",
        "adaptive_k_avg", "adaptive_k_min", "adaptive_k_max",
    ]
    
    # Columns to keep first value (should be same across runs)
    group_cols = [
        "base_experiment_name", "methodology", "num_tools", "doc_length", "prompt_type",
        "phase", "top_k", "similarity_threshold", "allow_backtrack", "top_k_categories",
        "min_k", "max_k", "drop_threshold", "min_similarity", "num_similar_tools", "is_no_tool_test",
    ]
    
    # Group by base experiment name
    grouped = df.groupby("base_experiment_name")
    
    agg_rows = []
    for base_name, group in grouped:
        row = {
            "experiment_name": base_name,
            "num_runs": len(group),
            "models": list(group["model"].unique()) if "model" in group.columns else [],
            "seeds": list(group["seed"].unique()) if "seed" in group.columns else [],
            "run_names": [r for r in group["run_name"].tolist() if r is not None],
        }
        
        # Copy group columns (first value)
        for col in group_cols:
            if col in group.columns:
                row[col] = group[col].iloc[0]
        
        # Aggregate numeric columns
        for col in numeric_cols:
            if col in group.columns:
                values = group[col].dropna()
                if len(values) > 0:
                    row[f"{col}_mean"] = values.mean()
                    row[f"{col}_std"] = values.std() if len(values) > 1 else 0.0
                    row[col] = values.mean()  # Keep mean as primary value
        
        agg_rows.append(row)
    
    agg_df = pd.DataFrame(agg_rows)
    
    if not agg_df.empty:
        agg_df = agg_df.sort_values(["methodology", "num_tools", "experiment_name"]).reset_index(drop=True)
    
    # Log aggregation summary
    single_runs = len([r for r in agg_df["num_runs"] if r == 1])
    multi_runs = len(agg_df) - single_runs
    logger.info(f"Aggregated {len(df)} results into {len(agg_df)} experiments ({single_runs} single-run, {multi_runs} multi-run)")
    
    return agg_df


def load_all_details_as_dataframe(results_dir: Path) -> pd.DataFrame:
    """
    Load all experiment details CSVs and combine into single DataFrame.
    Adds experiment_name column for grouping.
    """
    all_details = []
    
    for summary_path in find_experiments(results_dir):
        details_path = summary_path.with_name(
            summary_path.name.replace("_summary.json", "_details.csv")
        )
        if not details_path.exists():
            continue
        
        try:
            details = load_experiment_details(details_path)
            # Extract experiment name from summary
            summary = load_experiment_summary(summary_path)
            config = summary.get("experiment_config", {})
            exp_name = config.get("name", summary_path.stem)
            details["experiment_name"] = exp_name
            all_details.append(details)
        except Exception as e:
            logger.warning(f"Failed to load {details_path}: {e}")
            continue
    
    if all_details:
        return pd.concat(all_details, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# Statistical Analysis
# =============================================================================

def compute_confidence_interval(data: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Returns:
        (mean, ci_lower, ci_upper)
    """
    data = np.array(data)
    mean = np.mean(data)
    
    if len(data) < 2:
        return mean, mean, mean
    
    # Bootstrap
    n_bootstrap = 1000
    rng = np.random.default_rng(seed=42)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return mean, ci_lower, ci_upper


def filter_latency_outliers(latencies: pd.Series, method: str = "iqr", factor: float = 1.5) -> pd.Series:
    """
    Filter outlier latencies using IQR or percentile method.
    
    Args:
        latencies: Series of latency values
        method: "iqr" for IQR-based, "percentile" for P5-P95
        factor: IQR multiplier (default 1.5)
        
    Returns:
        Filtered series
    """
    if method == "iqr":
        q1 = latencies.quantile(0.25)
        q3 = latencies.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        return latencies[(latencies >= lower_bound) & (latencies <= upper_bound)]
    else:  # percentile
        p5 = latencies.quantile(0.05)
        p95 = latencies.quantile(0.95)
        return latencies[(latencies >= p5) & (latencies <= p95)]


# =============================================================================
# Visualization Functions
# =============================================================================

def setup_matplotlib():
    """Configure matplotlib for consistent styling."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving figures
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 150,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })


def generate_methodology_comparison_bar(df: pd.DataFrame, output_path: Path):
    """
    Generate grouped bar chart comparing methodologies by accuracy.
    Groups experiments by methodology and shows mean accuracy with error bars.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    # Aggregate by methodology
    agg = df.groupby("methodology").agg({
        "accuracy": ["mean", "std", "count"],
        "avg_latency_ms": "mean",
    }).reset_index()
    agg.columns = ["methodology", "accuracy_mean", "accuracy_std", "count", "latency_mean"]
    
    # Sort by accuracy
    agg = agg.sort_values("accuracy_mean", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(agg))
    colors = [METHODOLOGY_COLORS.get(m, "#666666") for m in agg["methodology"]]
    
    bars = ax.bar(x, agg["accuracy_mean"] * 100, 
                  yerr=agg["accuracy_std"] * 100, 
                  capsize=5,
                  color=colors, 
                  edgecolor='white', 
                  linewidth=1.5)
    
    # Add value labels
    for bar, acc in zip(bars, agg["accuracy_mean"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([METHODOLOGY_DISPLAY_NAMES.get(m, m) for m in agg["methodology"]], rotation=30, ha='right')
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Methodology")
    ax.set_title("Average Accuracy by Methodology", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    
    # Add count annotations
    for i, (bar, count) in enumerate(zip(bars, agg["count"])):
        ax.text(bar.get_x() + bar.get_width()/2, 5,
                f'n={count}', ha='center', va='bottom', fontsize=9, color='white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_accuracy_heatmap(df: pd.DataFrame, output_path: Path):
    """
    Generate heatmap of accuracy by methodology and num_tools.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_matplotlib()
    
    # Pivot for heatmap
    pivot = df.pivot_table(
        values="accuracy",
        index="methodology",
        columns="num_tools",
        aggfunc="mean"
    )
    
    # Sort index by METHODOLOGY_DISPLAY_NAMES order
    method_order = [m for m in METHODOLOGY_COLORS.keys() if m in pivot.index]
    pivot = pivot.reindex(method_order)
    
    # Sort columns numerically
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(
        pivot * 100, 
        annot=True, 
        fmt=".0f", 
        cmap="RdYlGn",
        vmin=0, 
        vmax=100,
        ax=ax,
        cbar_kws={"label": "Accuracy (%)"},
        linewidths=0.5,
    )
    
    ax.set_yticklabels([METHODOLOGY_DISPLAY_NAMES.get(m, m) for m in pivot.index], rotation=0)
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Methodology")
    ax.set_title("Accuracy Heatmap: Methodology × Tool Count", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_scaling_curves(df: pd.DataFrame, output_path: Path):
    """
    Generate line chart showing accuracy vs. number of tools for each methodology.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Group by methodology and num_tools
    for methodology in df["methodology"].unique():
        meth_df = df[df["methodology"] == methodology]
        
        # Aggregate by num_tools
        agg = meth_df.groupby("num_tools").agg({
            "accuracy": ["mean", "std"]
        }).reset_index()
        agg.columns = ["num_tools", "accuracy_mean", "accuracy_std"]
        agg = agg.sort_values("num_tools")
        
        color = METHODOLOGY_COLORS.get(methodology, "#666666")
        label = METHODOLOGY_DISPLAY_NAMES.get(methodology, methodology)
        
        ax.plot(agg["num_tools"], agg["accuracy_mean"] * 100, 
                marker='o', linewidth=2, markersize=8,
                color=color, label=label)
        
        # Add error band if we have std
        if agg["accuracy_std"].notna().any():
            ax.fill_between(
                agg["num_tools"],
                (agg["accuracy_mean"] - agg["accuracy_std"]) * 100,
                (agg["accuracy_mean"] + agg["accuracy_std"]) * 100,
                alpha=0.2, color=color
            )
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Scaling by Methodology", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Log scale for x-axis if range is large
    if df["num_tools"].max() / df["num_tools"].min() > 10:
        ax.set_xscale('log')
        ax.set_xticks(sorted(df["num_tools"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_latency_comparison(df: pd.DataFrame, details_df: pd.DataFrame, output_path: Path):
    """
    Generate latency boxplot comparison with and without outliers.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_matplotlib()
    
    if details_df.empty or "latency_ms" not in details_df.columns:
        logger.warning("No latency data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Raw latency (with outliers)
    ax1 = axes[0]
    order = sorted(details_df["methodology"].unique(), 
                   key=lambda x: METHODOLOGY_COLORS.get(x, "zzz"))
    palette = {m: METHODOLOGY_COLORS.get(m, "#666666") for m in order}
    
    sns.boxplot(data=details_df, x="methodology", y="latency_ms", 
                ax=ax1, order=order, hue="methodology", palette=palette, 
                legend=False, showfliers=True)
    ax1.set_title("Latency Distribution (with outliers)", fontweight='bold')
    ax1.set_xlabel("Methodology")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels([METHODOLOGY_DISPLAY_NAMES.get(m, m) for m in order], rotation=30, ha='right')
    
    # Right: Filtered latency (without outliers)
    ax2 = axes[1]
    
    # Filter outliers per methodology
    filtered_data = []
    for meth in order:
        meth_data = details_df[details_df["methodology"] == meth]["latency_ms"]
        filtered = filter_latency_outliers(meth_data, method="iqr", factor=1.5)
        for val in filtered:
            filtered_data.append({"methodology": meth, "latency_ms": val})
    
    filtered_df = pd.DataFrame(filtered_data)
    
    if not filtered_df.empty:
        sns.boxplot(data=filtered_df, x="methodology", y="latency_ms", 
                    ax=ax2, order=order, hue="methodology", palette=palette, 
                    legend=False, showfliers=False)
    ax2.set_title("Latency Distribution (outliers filtered)", fontweight='bold')
    ax2.set_xlabel("Methodology")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels([METHODOLOGY_DISPLAY_NAMES.get(m, m) for m in order], rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_latency_vs_accuracy_scatter(df: pd.DataFrame, output_path: Path):
    """
    Generate scatter plot of latency vs accuracy colored by methodology.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for methodology in df["methodology"].unique():
        meth_df = df[df["methodology"] == methodology]
        color = METHODOLOGY_COLORS.get(methodology, "#666666")
        label = METHODOLOGY_DISPLAY_NAMES.get(methodology, methodology)
        
        ax.scatter(
            meth_df["avg_latency_ms"], 
            meth_df["accuracy"] * 100,
            c=color, 
            label=label,
            s=meth_df["num_tools"] / 5,  # Size by num_tools
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5
        )
    
    ax.set_xlabel("Average Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Latency Trade-off\n(point size = num_tools)", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_error_breakdown(df: pd.DataFrame, output_path: Path):
    """
    Generate stacked bar chart showing error breakdown by methodology.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    # Aggregate by methodology
    agg = df.groupby("methodology").agg({
        "tool_correct": "sum",
        "tool_incorrect": "sum",
        "no_tool_called": "sum",
        "errors": "sum",
        "total_tests": "sum",
    }).reset_index()
    
    # Sort by total tests for better visualization
    agg = agg.sort_values("total_tests", ascending=False)
    
    # Convert to percentages
    for col in ["tool_correct", "tool_incorrect", "no_tool_called", "errors"]:
        agg[f"{col}_pct"] = agg[col] / agg["total_tests"] * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(agg))
    width = 0.6
    
    # Stacked bars
    bottom = np.zeros(len(agg))
    
    colors = ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    labels = ["Correct", "Incorrect", "No Tool Called", "Errors"]
    
    for i, (col, color, label) in enumerate(zip(
        ["tool_correct_pct", "tool_incorrect_pct", "no_tool_called_pct", "errors_pct"],
        colors, labels
    )):
        values = agg[col].values
        ax.bar(x, values, width, bottom=bottom, label=label, color=color, edgecolor='white')
        bottom += values
    
    ax.set_xticks(x)
    ax.set_xticklabels([METHODOLOGY_DISPLAY_NAMES.get(m, m) for m in agg["methodology"]], rotation=30, ha='right')
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Methodology")
    ax.set_title("Result Breakdown by Methodology", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_doc_length_impact(df: pd.DataFrame, output_path: Path):
    """
    Generate grouped bar chart showing accuracy by doc_length for each methodology.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    # Filter to experiments that vary doc_length
    df_filtered = df[df["doc_length"].notna()]
    
    if df_filtered.empty or df_filtered["doc_length"].nunique() < 2:
        logger.warning("Not enough doc_length variation to plot")
        return
    
    # Pivot for grouped bars
    pivot = df_filtered.pivot_table(
        values="accuracy",
        index="methodology",
        columns="doc_length",
        aggfunc="mean"
    )
    
    # Reorder columns
    doc_order = [d for d in DOC_LENGTH_ORDER if d in pivot.columns]
    pivot = pivot[doc_order]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(pivot.index))
    width = 0.2
    
    for i, doc_len in enumerate(pivot.columns):
        offset = (i - len(pivot.columns)/2 + 0.5) * width
        bars = ax.bar(x + offset, pivot[doc_len] * 100, width, label=doc_len.capitalize())
    
    ax.set_xticks(x)
    ax.set_xticklabels([METHODOLOGY_DISPLAY_NAMES.get(m, m) for m in pivot.index], rotation=30, ha='right')
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Methodology")
    ax.set_title("Impact of Documentation Length on Accuracy", fontsize=14, fontweight='bold')
    ax.legend(title="Doc Length", loc='upper right', frameon=True)
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


# =============================================================================
# Token Usage Analysis Charts
# =============================================================================

def generate_token_usage_by_methodology(df: pd.DataFrame, output_path: Path):
    """
    Generate bar chart showing average input tokens by methodology.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    # Filter to experiments with token data
    df_tokens = df[df["avg_tokens_input"] > 0].copy()
    
    if df_tokens.empty:
        logger.warning("No token usage data available")
        return
    
    # Aggregate by methodology
    agg = df_tokens.groupby("methodology").agg({
        "avg_tokens_input": ["mean", "std"],
        "avg_tokens_output": ["mean", "std"],
        "avg_tokens_total": ["mean", "std"],
    }).reset_index()
    agg.columns = ["methodology", "input_mean", "input_std", "output_mean", "output_std", "total_mean", "total_std"]
    
    # Sort by total tokens
    agg = agg.sort_values("total_mean", ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(agg))
    width = 0.35
    
    # Input tokens
    bars1 = ax.barh(x - width/2, agg["input_mean"], width, 
                    xerr=agg["input_std"].fillna(0), 
                    label="Input Tokens", color="#3498db", capsize=3)
    
    # Output tokens
    bars2 = ax.barh(x + width/2, agg["output_mean"], width, 
                    xerr=agg["output_std"].fillna(0), 
                    label="Output Tokens", color="#e74c3c", capsize=3)
    
    ax.set_yticks(x)
    ax.set_yticklabels([METHODOLOGY_DISPLAY_NAMES.get(m, m) for m in agg["methodology"]])
    ax.set_xlabel("Average Tokens per Request")
    ax.set_title("Token Usage by Methodology", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(axis='x', alpha=0.3)
    
    # Add total values as text
    for i, (_, row) in enumerate(agg.iterrows()):
        total = row["input_mean"] + row["output_mean"]
        ax.text(total + 50, i, f'{total:.0f}', va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_token_usage_scaling(df: pd.DataFrame, output_path: Path):
    """
    Generate line chart showing how input tokens scale with number of tools.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    # Filter to experiments with token data
    df_tokens = df[df["avg_tokens_input"] > 0].copy()
    
    if df_tokens.empty:
        logger.warning("No token usage data available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Group by methodology and num_tools
    for methodology in df_tokens["methodology"].unique():
        meth_data = df_tokens[df_tokens["methodology"] == methodology]
        
        # Aggregate by num_tools
        agg = meth_data.groupby("num_tools").agg({
            "avg_tokens_input": "mean"
        }).reset_index()
        
        if len(agg) < 2:
            continue
        
        agg = agg.sort_values("num_tools")
        
        color = METHODOLOGY_COLORS.get(methodology, "#666666")
        label = METHODOLOGY_DISPLAY_NAMES.get(methodology, methodology)
        
        ax.plot(agg["num_tools"], agg["avg_tokens_input"], 
                marker='o', linewidth=2, markersize=8,
                color=color, label=label)
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Average Input Tokens")
    ax.set_title("Input Token Usage Scaling by Methodology", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Log scale for x-axis if range is large
    if df_tokens["num_tools"].max() / df_tokens["num_tools"].min() > 10:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_token_efficiency_scatter(df: pd.DataFrame, output_path: Path):
    """
    Generate scatter plot of accuracy vs. input tokens (token efficiency).
    Shows which methodologies achieve better accuracy with fewer tokens.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    # Filter to experiments with token data
    df_tokens = df[df["avg_tokens_input"] > 0].copy()
    
    if df_tokens.empty:
        logger.warning("No token usage data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for methodology in df_tokens["methodology"].unique():
        meth_data = df_tokens[df_tokens["methodology"] == methodology]
        
        color = METHODOLOGY_COLORS.get(methodology, "#666666")
        label = METHODOLOGY_DISPLAY_NAMES.get(methodology, methodology)
        
        # Size by num_tools
        sizes = (meth_data["num_tools"] / meth_data["num_tools"].max() * 200 + 50).values
        
        ax.scatter(meth_data["avg_tokens_input"], meth_data["accuracy"] * 100,
                   s=sizes, alpha=0.7, color=color, label=label, edgecolors='white')
    
    ax.set_xlabel("Average Input Tokens")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Token Efficiency: Accuracy vs. Input Token Cost\n(point size = num_tools)", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_token_heatmap(df: pd.DataFrame, output_path: Path):
    """
    Generate heatmap of average input tokens by methodology and num_tools.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_matplotlib()
    
    # Filter to experiments with token data
    df_tokens = df[df["avg_tokens_input"] > 0].copy()
    
    if df_tokens.empty:
        logger.warning("No token usage data available")
        return
    
    # Pivot for heatmap
    pivot = df_tokens.pivot_table(
        values="avg_tokens_input",
        index="methodology",
        columns="num_tools",
        aggfunc="mean"
    )
    
    # Sort index by METHODOLOGY_DISPLAY_NAMES order
    method_order = [m for m in METHODOLOGY_COLORS.keys() if m in pivot.index]
    pivot = pivot.reindex(method_order)
    
    # Sort columns numerically
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(
        pivot, 
        annot=True, 
        fmt=".0f", 
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Avg Input Tokens"},
        linewidths=0.5,
    )
    
    ax.set_yticklabels([METHODOLOGY_DISPLAY_NAMES.get(m, m) for m in pivot.index], rotation=0)
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Methodology")
    ax.set_title("Input Token Usage Heatmap: Methodology × Tool Count", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


# =============================================================================
# Methodology-Specific Analysis Charts
# =============================================================================

def generate_rag_topk_analysis(df: pd.DataFrame, output_path: Path):
    """
    Generate line chart showing RAG accuracy vs. top_k parameter.
    Averages multiple experiments with the same (num_tools, top_k) combination.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    rag_df = df[(df["methodology"] == "rag") & (df["top_k"].notna())]
    
    if rag_df.empty:
        logger.warning("No RAG experiments with top_k parameter found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by num_tools to show different lines
    for num_tools in sorted(rag_df["num_tools"].unique()):
        subset = rag_df[rag_df["num_tools"] == num_tools]
        # Average accuracy for same top_k values
        agg = subset.groupby("top_k").agg({
            "accuracy": ["mean", "std"]
        }).reset_index()
        agg.columns = ["top_k", "accuracy_mean", "accuracy_std"]
        agg = agg.sort_values("top_k")
        
        ax.plot(agg["top_k"], agg["accuracy_mean"] * 100, 
                marker='o', linewidth=2, markersize=8,
                label=f"{num_tools} tools")
        
        # Add error band if we have std
        if agg["accuracy_std"].notna().any() and (agg["accuracy_std"] > 0).any():
            ax.fill_between(
                agg["top_k"],
                (agg["accuracy_mean"] - agg["accuracy_std"]) * 100,
                (agg["accuracy_mean"] + agg["accuracy_std"]) * 100,
                alpha=0.2
            )
    
    ax.set_xlabel("Top-K Retrieved")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("RAG: Impact of Top-K on Accuracy", fontsize=14, fontweight='bold')
    ax.legend(title="Tool Count", loc='lower right', frameon=True)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_clustering_backtrack_analysis(df: pd.DataFrame, output_path: Path):
    """
    Generate grouped bar chart comparing backtrack vs. no-backtrack.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    cluster_df = df[df["methodology"] == "clustering"].copy()
    
    if cluster_df.empty:
        logger.warning("No clustering experiments found")
        return
    
    # Group by num_tools and backtrack status
    cluster_df["backtrack_label"] = cluster_df["allow_backtrack"].map({True: "Backtrack", False: "No Backtrack"})
    
    pivot = cluster_df.pivot_table(
        values="accuracy",
        index="num_tools",
        columns="backtrack_label",
        aggfunc="mean"
    )
    
    if pivot.empty or pivot.shape[1] < 2:
        # Not enough variation - just show bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        agg = cluster_df.groupby("num_tools")["accuracy"].mean().reset_index()
        ax.bar(range(len(agg)), agg["accuracy"] * 100, color=METHODOLOGY_COLORS["clustering"])
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg["num_tools"])
        ax.set_xlabel("Number of Tools")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Clustering: Accuracy by Tool Count", fontsize=14, fontweight='bold')
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(pivot.index))
        width = 0.35
        
        if "Backtrack" in pivot.columns:
            ax.bar(x - width/2, pivot["Backtrack"] * 100, width, label="Backtrack", color="#ff7f0e")
        if "No Backtrack" in pivot.columns:
            ax.bar(x + width/2, pivot["No Backtrack"] * 100, width, label="No Backtrack", color="#ffbb78")
        
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index)
        ax.set_xlabel("Number of Tools")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Clustering: Backtrack vs. No Backtrack", fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', frameon=True)
    
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_adaptive_k_distribution(df: pd.DataFrame, details_df: pd.DataFrame, output_path: Path):
    """
    Generate histogram of adaptive_k_used values for adaptive RAG.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_matplotlib()
    
    adaptive_details = details_df[
        (details_df["methodology"] == "adaptive_rag") & 
        (details_df["adaptive_k_used"].notna()) &
        (details_df["adaptive_k_used"] > 0)
    ]
    
    if adaptive_details.empty:
        logger.warning("No adaptive_k_used data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Histogram of k values
    ax1 = axes[0]
    sns.histplot(adaptive_details["adaptive_k_used"], bins=20, kde=True, 
                 ax=ax1, color=METHODOLOGY_COLORS["adaptive_rag"])
    ax1.axvline(adaptive_details["adaptive_k_used"].mean(), color='red', linestyle='--', 
                label=f'Mean: {adaptive_details["adaptive_k_used"].mean():.1f}')
    ax1.set_xlabel("Adaptive K Value")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Adaptive K Values", fontweight='bold')
    ax1.legend()
    
    # Right: Strategy distribution
    ax2 = axes[1]
    if "adaptive_strategy" in adaptive_details.columns:
        strategy_counts = adaptive_details["adaptive_strategy"].value_counts()
        colors = plt.cm.Set2(np.linspace(0, 1, len(strategy_counts)))
        ax2.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax2.set_title("Adaptive Strategy Distribution", fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "No strategy data available", ha='center', va='center')
        ax2.set_title("Adaptive Strategy Distribution", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_hybrid_category_analysis(df: pd.DataFrame, output_path: Path):
    """
    Generate chart showing hybrid accuracy vs. top_k_categories.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    hybrid_df = df[(df["methodology"] == "hybrid") & (df["top_k_categories"].notna())]
    
    if hybrid_df.empty:
        logger.warning("No hybrid experiments with top_k_categories found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by num_tools
    for num_tools in sorted(hybrid_df["num_tools"].unique()):
        subset = hybrid_df[hybrid_df["num_tools"] == num_tools]
        # Average accuracy for same top_k_categories values
        agg = subset.groupby("top_k_categories").agg({
            "accuracy": ["mean", "std"]
        }).reset_index()
        agg.columns = ["top_k_categories", "accuracy_mean", "accuracy_std"]
        agg = agg.sort_values("top_k_categories")
        
        ax.plot(agg["top_k_categories"], agg["accuracy_mean"] * 100, 
                marker='o', linewidth=2, markersize=8,
                label=f"{num_tools} tools")
        
        # Add error band if we have std
        if agg["accuracy_std"].notna().any() and (agg["accuracy_std"] > 0).any():
            ax.fill_between(
                agg["top_k_categories"],
                (agg["accuracy_mean"] - agg["accuracy_std"]) * 100,
                (agg["accuracy_mean"] + agg["accuracy_std"]) * 100,
                alpha=0.2
            )
    
    ax.set_xlabel("Top-K Categories")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Hybrid: Impact of Category Count on Accuracy", fontsize=14, fontweight='bold')
    ax.legend(title="Tool Count", loc='lower right', frameon=True)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(hybrid_df["top_k_categories"].unique()))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_category_accuracy_comparison(df: pd.DataFrame, output_path: Path):
    """
    Generate heatmap showing per-category accuracy across methodologies.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_matplotlib()
    
    # Parse category accuracy from JSON
    all_cat_acc = []
    for _, row in df.iterrows():
        try:
            cat_acc = json.loads(row.get("category_accuracy_json", "{}"))
            for cat, acc in cat_acc.items():
                all_cat_acc.append({
                    "methodology": row["methodology"],
                    "category": cat.replace("_operations", ""),
                    "accuracy": acc
                })
        except:
            continue
    
    if not all_cat_acc:
        logger.warning("No category accuracy data available")
        return
    
    cat_df = pd.DataFrame(all_cat_acc)
    
    # Pivot to heatmap format
    pivot = cat_df.pivot_table(
        values="accuracy",
        index="category",
        columns="methodology",
        aggfunc="mean"
    )
    
    # Sort by overall accuracy
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    
    # Order columns by methodology order
    method_order = [m for m in METHODOLOGY_COLORS.keys() if m in pivot.columns]
    pivot = pivot[method_order]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.5)))
    
    sns.heatmap(
        pivot * 100, 
        annot=True, 
        fmt=".0f", 
        cmap="RdYlGn",
        vmin=0, 
        vmax=100,
        ax=ax,
        cbar_kws={"label": "Accuracy (%)"},
        linewidths=0.5,
    )
    
    ax.set_xticklabels([METHODOLOGY_DISPLAY_NAMES.get(m, m) for m in pivot.columns], rotation=30, ha='right')
    ax.set_ylabel("Category")
    ax.set_xlabel("Methodology")
    ax.set_title("Per-Category Accuracy by Methodology", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_radar_chart(df: pd.DataFrame, output_path: Path, num_tools_filter: int = None):
    """
    Generate radar chart comparing methodologies on multiple metrics.
    """
    import matplotlib.pyplot as plt
    from math import pi
    setup_matplotlib()
    
    # Filter by num_tools if specified
    if num_tools_filter:
        df_filtered = df[df["num_tools"] == num_tools_filter]
    else:
        df_filtered = df
    
    # Aggregate by methodology
    agg = df_filtered.groupby("methodology").agg({
        "accuracy": "mean",
        "call_rate": "mean",
        "avg_latency_ms": "mean",
        "category_selection_accuracy": "mean",
    }).reset_index()
    
    # Normalize latency (lower is better, so invert)
    max_latency = agg["avg_latency_ms"].max()
    if max_latency > 0:
        agg["latency_score"] = 1 - (agg["avg_latency_ms"] / max_latency)
    else:
        agg["latency_score"] = 1.0
    
    # Metrics for radar
    metrics = ["accuracy", "call_rate", "category_selection_accuracy", "latency_score"]
    metric_labels = ["Accuracy", "Call Rate", "Category Accuracy", "Speed (normalized)"]
    
    # Number of variables
    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for _, row in agg.iterrows():
        methodology = row["methodology"]
        values = [row[m] for m in metrics]
        values += values[:1]  # Complete the loop
        
        color = METHODOLOGY_COLORS.get(methodology, "#666666")
        label = METHODOLOGY_DISPLAY_NAMES.get(methodology, methodology)
        
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=label)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=11)
    ax.set_ylim(0, 1)
    
    title = "Methodology Comparison Radar"
    if num_tools_filter:
        title += f" ({num_tools_filter} tools)"
    ax.set_title(title, size=14, fontweight='bold', y=1.08)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


# =============================================================================
# Report Generation
# =============================================================================

def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics table."""
    summary = df.groupby("methodology").agg({
        "accuracy": ["mean", "std", "min", "max"],
        "avg_latency_ms": ["mean", "min", "max"],
        "total_tests": "sum",
        "experiment_name": "count"
    }).round(3)
    
    summary.columns = [
        "Accuracy (Mean)", "Accuracy (Std)", "Accuracy (Min)", "Accuracy (Max)",
        "Latency Mean (ms)", "Latency Min (ms)", "Latency Max (ms)",
        "Total Tests", "Num Experiments"
    ]
    
    return summary


def generate_html_report(
    df: pd.DataFrame, 
    details_df: pd.DataFrame,
    output_path: Path, 
    figures_dir: Path
):
    """Generate comprehensive HTML analysis report."""
    summary_table = generate_summary_table(df)
    
    # Calculate key findings
    best_accuracy = df.loc[df["accuracy"].idxmax()]
    best_latency = df.loc[df["avg_latency_ms"].idxmin()]
    
    # Token usage stats
    df_with_tokens = df[df["avg_tokens_input"] > 0]
    has_token_data = len(df_with_tokens) > 0
    if has_token_data:
        avg_tokens_input = df_with_tokens["avg_tokens_input"].mean()
        avg_tokens_output = df_with_tokens["avg_tokens_output"].mean()
        most_efficient = df_with_tokens.loc[df_with_tokens["avg_tokens_input"].idxmin()]
    
    # Methodology rankings - treating missing experiments as 0% accuracy
    # Get all unique tool counts tested across all methodologies
    all_tool_counts = sorted(df["num_tools"].unique())
    all_methodologies = df["methodology"].unique()
    
    # Create a complete grid of methodology × tool count, filling missing with 0
    meth_rankings_data = {}
    missing_experiments = []
    
    for meth in all_methodologies:
        meth_df = df[df["methodology"] == meth]
        meth_tool_counts = set(meth_df["num_tools"].unique())
        
        # Calculate accuracy including 0 for missing tool counts
        total_accuracy = 0.0
        for tc in all_tool_counts:
            if tc in meth_tool_counts:
                # Get accuracy for this methodology-tool count combination
                acc = meth_df[meth_df["num_tools"] == tc]["accuracy"].mean()
                total_accuracy += acc
            else:
                # Missing experiment counts as 0%
                missing_experiments.append((meth, tc))
        
        # Average across all tool counts (including missing as 0)
        meth_rankings_data[meth] = total_accuracy / len(all_tool_counts)
    
    meth_rankings = pd.Series(meth_rankings_data).sort_values(ascending=False)
    has_missing_experiments = len(missing_experiments) > 0
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Tool Calling Experiment Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 15px; }}
        h2 {{ color: #555; border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-top: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; max-width: 1000px; margin: 20px auto; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric {{ font-size: 28px; font-weight: bold; color: #4CAF50; }}
        .card {{ background: #f9f9f9; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #4CAF50; }}
        .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .stat-card h3 {{ margin: 0 0 10px 0; color: #666; font-size: 14px; }}
        .stat-card .value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .figure {{ margin: 30px 0; text-align: center; }}
        .figure img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .figure-caption {{ margin-top: 10px; color: #666; font-style: italic; }}
        .ranking {{ display: flex; gap: 10px; flex-wrap: wrap; }}
        .rank-badge {{ padding: 8px 16px; border-radius: 20px; color: white; font-weight: bold; }}
        .rank-1 {{ background: #FFD700; color: #333; }}
        .rank-2 {{ background: #C0C0C0; color: #333; }}
        .rank-3 {{ background: #CD7F32; }}
        .rank-other {{ background: #999; }}
        .figure {{ max-width: 800px; margin: 20px auto; }}
        .figure.hidden {{ display: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 Tool Calling Experiment Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Experiments analyzed:</strong> {len(df)} experiments across {df['methodology'].nunique()} methodologies</p>
        
        <h2>📊 Executive Summary</h2>
        <div class="card-grid">
            <div class="stat-card">
                <h3>Total Experiments</h3>
                <div class="value">{len(df)}</div>
            </div>
            <div class="stat-card">
                <h3>Total Test Cases</h3>
                <div class="value">{df['total_tests'].sum():,}</div>
            </div>
            <div class="stat-card">
                <h3>Best Accuracy</h3>
                <div class="value">{best_accuracy['accuracy']*100:.1f}%</div>
                <div style="font-size: 12px; color: #666;">{best_accuracy['experiment_name']}</div>
            </div>
            <div class="stat-card">
                <h3>Fastest Avg. Latency</h3>
                <div class="value">{best_latency['avg_latency_ms']:.0f}ms</div>
                <div style="font-size: 12px; color: #666;">{best_latency['experiment_name']}</div>
            </div>
        </div>
"""
    
    # Add token usage stats if available
    if has_token_data:
        html += f"""
        <h2>🔤 Token Usage Summary</h2>
        <div class="card-grid">
            <div class="stat-card">
                <h3>Experiments with Token Data</h3>
                <div class="value">{len(df_with_tokens)}</div>
            </div>
            <div class="stat-card">
                <h3>Avg Input Tokens</h3>
                <div class="value">{avg_tokens_input:,.0f}</div>
            </div>
            <div class="stat-card">
                <h3>Avg Output Tokens</h3>
                <div class="value">{avg_tokens_output:,.0f}</div>
            </div>
            <div class="stat-card">
                <h3>Most Token-Efficient</h3>
                <div class="value">{most_efficient['avg_tokens_input']:,.0f}</div>
                <div style="font-size: 12px; color: #666;">{most_efficient['experiment_name']}</div>
            </div>
        </div>
"""
    
    html += """
        <h2>🏆 Methodology Rankings (by Average Accuracy)</h2>
        <div class="ranking">
"""
    
    for i, (meth, acc) in enumerate(meth_rankings.items()):
        rank_class = f"rank-{i+1}" if i < 3 else "rank-other"
        display_name = METHODOLOGY_DISPLAY_NAMES.get(meth, meth)
        html += f'<span class="rank-badge {rank_class}">#{i+1} {display_name}: {acc*100:.1f}%</span>\n'
    
    html += """
        </div>
"""
    
    # Add info note about missing experiments if any
    if has_missing_experiments:
        missing_by_meth = {}
        for meth, tc in missing_experiments:
            if meth not in missing_by_meth:
                missing_by_meth[meth] = []
            missing_by_meth[meth].append(tc)
        
        html += """
        <div class="card" style="border-left-color: #f39c12; background: #fef9e7;">
            <p><strong>ℹ️ Note:</strong> Rankings are calculated treating missing experiments as 0% accuracy. 
            This penalizes methodologies that could not be tested at certain tool scales (e.g., due to token limits).</p>
            <p><strong>Missing experiments:</strong></p>
            <ul>
"""
        for meth, tool_counts in missing_by_meth.items():
            display_name = METHODOLOGY_DISPLAY_NAMES.get(meth, meth)
            tc_str = ", ".join(str(tc) for tc in sorted(tool_counts))
            html += f"<li>{display_name}: {tc_str} tools</li>\n"
        
        html += """
            </ul>
        </div>
"""
    
    html += """
        <h2>📈 Summary Statistics</h2>
"""
    
    html += summary_table.to_html(classes='summary')
    
    html += """
        <h2>📉 Visualizations</h2>
"""
    
    # Add figures
    figure_descriptions = {
        "01_methodology_comparison.png": "Average accuracy comparison across methodologies with standard deviation error bars.",
        "02_accuracy_heatmap.png": "Accuracy by methodology and tool count. Green = high accuracy, red = low accuracy.",
        "03_scaling_curves.png": "How accuracy changes as the number of tools increases for each methodology.",
        "04_latency_comparison.png": "Latency distribution comparison. Left shows raw data with outliers, right shows filtered data.",
        "05_latency_vs_accuracy.png": "Trade-off between latency and accuracy. Point size indicates number of tools.",
        "06_error_breakdown.png": "Breakdown of test outcomes by methodology.",
        "07_doc_length_impact.png": "Impact of tool documentation length on accuracy.",
        "08_rag_topk_analysis.png": "RAG methodology: Effect of top-K parameter on accuracy.",
        # "09_clustering_backtrack.png": "Clustering methodology: Backtracking impact on accuracy.",
        "10_adaptive_k_distribution.png": "Adaptive RAG: Distribution of dynamically selected K values.",
        "11_hybrid_category_analysis.png": "Hybrid methodology: Effect of top-K categories on accuracy.",
        "12_category_accuracy.png": "Per-category accuracy comparison across methodologies.",
        # "13_radar_comparison.png": "Multi-dimensional comparison of methodology performance.",
        "14_token_usage_methodology.png": "Average input and output tokens per request by methodology.",
        "15_token_usage_scaling.png": "How input token usage scales with the number of tools for each methodology.",
        "16_token_efficiency.png": "Token efficiency: accuracy achieved relative to input token cost.",
        "17_token_heatmap.png": "Input token usage heatmap by methodology and tool count.",
    }
    
    for fig_name, description in figure_descriptions.items():
        fig_path = figures_dir / fig_name
        if fig_path.exists():
            rel_path = fig_path.name
            html += f"""
        <div class="figure">
            <img src="figures/{rel_path}" alt="{fig_name.replace('.png', '').replace('_', ' ').title()}">
            <p class="figure-caption">{description}</p>
        </div>
"""
    
    html += """
        <h2>🔍 Key Findings</h2>
        <div class="card">
            <ul>
"""
    
    # Add key findings
    if len(meth_rankings) > 0:
        best_meth = meth_rankings.index[0]
        html += f"<li><strong>Best performing methodology:</strong> {METHODOLOGY_DISPLAY_NAMES.get(best_meth, best_meth)} with {meth_rankings.iloc[0]*100:.1f}% average accuracy</li>\n"
    
    if "mcp" in meth_rankings.index:
        mcp_rank = list(meth_rankings.index).index("mcp") + 1
        html += f"<li><strong>MCP baseline:</strong> Ranked #{mcp_rank} with {meth_rankings['mcp']*100:.1f}% accuracy</li>\n"
    
    # Scaling insights
    large_scale = df[df["num_tools"] >= 500]
    if not large_scale.empty:
        best_large = large_scale.groupby("methodology")["accuracy"].mean().idxmax()
        html += f"<li><strong>Best at scale (500+ tools):</strong> {METHODOLOGY_DISPLAY_NAMES.get(best_large, best_large)}</li>\n"
    
    html += """
            </ul>
        </div>
        
        <h2>⚠️ Limitations & Notes</h2>
        <div class="card">
            <ul>
                <li>Results may be affected by rate limiting during experiments</li>
                <li>Latency outliers have been filtered using IQR method in some charts</li>
                <li>Category accuracy may vary based on category representation in test set</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"Generated HTML report: {output_path}")


# =============================================================================
# CLI Commands
# =============================================================================

@app.command()
def generate_report(
    results_dir: Path = typer.Option(
        Path("experiments/results/plan"),
        "--results-dir", "-d",
        help="Directory containing experiment results"
    ),
    output_dir: Path = typer.Option(
        Path("reports"),
        "--output-dir", "-o",
        help="Output directory for report and figures"
    ),
    aggregate_runs: bool = typer.Option(
        True,
        "--aggregate/--no-aggregate",
        help="Aggregate results across multiple runs (different seeds/models)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """
    Generate comprehensive analysis report with all visualizations.
    
    When --aggregate is enabled (default), results from multiple runs of the same
    experiment (with different seeds/models) are aggregated together, showing
    mean values and standard deviations.
    """
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")
    
    logger.info(f"Loading experiments from {results_dir}")
    df = load_all_experiments_as_dataframe(results_dir)
    
    if df.empty:
        logger.error("No experiments found")
        raise typer.Exit(1)
    
    logger.info(f"Loaded {len(df)} experiment results")
    
    # Aggregate across runs if enabled
    if aggregate_runs:
        # Check if there's multi-run data
        has_runs = "run_name" in df.columns and df["run_name"].notna().any()
        if has_runs:
            logger.info("Aggregating results across multiple runs...")
            df = aggregate_across_runs(df)
            logger.info(f"Aggregated to {len(df)} unique experiments")
    
    # Load details for detailed analysis
    details_df = load_all_details_as_dataframe(results_dir)
    logger.info(f"Loaded {len(details_df)} detailed test records")
    
    # Create output directories
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    logger.info("Generating visualizations...")
    
    # Overview charts
    generate_methodology_comparison_bar(df, figures_dir / "01_methodology_comparison.png")
    generate_accuracy_heatmap(df, figures_dir / "02_accuracy_heatmap.png")
    generate_scaling_curves(df, figures_dir / "03_scaling_curves.png")
    generate_latency_comparison(df, details_df, figures_dir / "04_latency_comparison.png")
    generate_latency_vs_accuracy_scatter(df, figures_dir / "05_latency_vs_accuracy.png")
    generate_error_breakdown(df, figures_dir / "06_error_breakdown.png")
    generate_doc_length_impact(df, figures_dir / "07_doc_length_impact.png")
    
    # Methodology-specific charts
    generate_rag_topk_analysis(df, figures_dir / "08_rag_topk_analysis.png")
    generate_clustering_backtrack_analysis(df, figures_dir / "09_clustering_backtrack.png")
    generate_adaptive_k_distribution(df, details_df, figures_dir / "10_adaptive_k_distribution.png")
    generate_hybrid_category_analysis(df, figures_dir / "11_hybrid_category_analysis.png")
    generate_category_accuracy_comparison(df, figures_dir / "12_category_accuracy.png")
    generate_radar_chart(df, figures_dir / "13_radar_comparison.png")
    
    # Token usage analysis charts
    generate_token_usage_by_methodology(df, figures_dir / "14_token_usage_methodology.png")
    generate_token_usage_scaling(df, figures_dir / "15_token_usage_scaling.png")
    generate_token_efficiency_scatter(df, figures_dir / "16_token_efficiency.png")
    generate_token_heatmap(df, figures_dir / "17_token_heatmap.png")
    
    # Generate HTML report
    generate_html_report(df, details_df, output_dir / "analysis_report.html", figures_dir)
    
    logger.info(f"Report generated at {output_dir / 'analysis_report.html'}")
    print(f"\n✅ Report generated: {output_dir / 'analysis_report.html'}")
    print(f"   Figures saved to: {figures_dir}")


@app.command()
def overview(
    results_dir: Path = typer.Option(
        Path("experiments/results/plan"),
        "--results-dir", "-d"
    ),
    output_dir: Path = typer.Option(
        Path("reports/figures"),
        "--output-dir", "-o"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """Generate overview comparison charts only."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")
    
    df = load_all_experiments_as_dataframe(results_dir)
    details_df = load_all_details_as_dataframe(results_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_methodology_comparison_bar(df, output_dir / "methodology_comparison.png")
    generate_accuracy_heatmap(df, output_dir / "accuracy_heatmap.png")
    generate_scaling_curves(df, output_dir / "scaling_curves.png")
    generate_latency_comparison(df, details_df, output_dir / "latency_comparison.png")
    
    logger.info(f"Overview charts saved to {output_dir}")


@app.command()
def compare(
    experiments: list[str] = typer.Argument(..., help="Summary JSON files to compare"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """Compare specific experiments with statistical analysis."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")
    
    summaries = []
    for exp_path in experiments:
        summaries.append(load_experiment_summary(Path(exp_path)))
    
    # Create comparison table
    rows = []
    for summary in summaries:
        config = summary.get("experiment_config", {})
        rows.append({
            "Experiment": config.get("name", "unknown"),
            "Methodology": summary.get("methodology"),
            "Num Tools": config.get("num_tools"),
            "Accuracy": f"{summary.get('accuracy', 0)*100:.1f}%",
            "Latency (ms)": f"{summary.get('avg_latency_ms', 0):.1f}",
            "Call Rate": f"{summary.get('call_rate', 0)*100:.1f}%",
        })
    
    comparison_df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    
    if output:
        comparison_df.to_csv(output, index=False)
        logger.info(f"Saved comparison to {output}")


@app.command()
def export_data(
    results_dir: Path = typer.Option(
        Path("experiments/results/plan"),
        "--results-dir", "-d"
    ),
    output: Path = typer.Option(
        Path("reports/experiment_data.csv"),
        "--output", "-o"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """Export aggregated experiment data to CSV for external analysis."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")
    
    df = load_all_experiments_as_dataframe(results_dir)
    
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    
    logger.info(f"Exported {len(df)} experiments to {output}")
    print(f"\n✅ Exported data to: {output}")


if __name__ == "__main__":
    app()
