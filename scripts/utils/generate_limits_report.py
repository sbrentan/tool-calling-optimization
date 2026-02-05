#!/usr/bin/env python
"""
Methodology Limits Report Generator

This script generates an HTML report showing where each methodology's limitations
are reached and how subsequent methodologies address them.

Sections:
1. MCP Limits - Shows token scaling issues with increasing tools
2. Clustering Analysis - Category selection accuracy and backtracking analysis
3. Hybrid Methodology - RAG + Clustering combination results
4. RAG Improvement - Direct tool retrieval with stable token usage
5. Adaptive RAG - Dynamic K optimization
6. Final Summary - Latency distributions and accuracy heatmaps

Usage:
    python scripts/generate_limits_report.py --results-dir experiments/results/tmp
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

app = typer.Typer(help="Generate methodology limits analysis report")

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
# Data Loading and Aggregation (copied from analyze_results.py)
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
    """Parse experiment name to extract configuration parameters."""
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
    """Load all experiments and aggregate into a unified DataFrame."""
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
        
        # Extract run info
        base_exp_name = exp_name
        run_name = None
        if "_run_" in exp_name:
            parts = exp_name.rsplit("_run_", 1)
            if len(parts) == 2:
                base_exp_name = parts[0]
                run_name = f"run_{parts[1]}"
        
        row = {
            "experiment_name": exp_name,
            "base_experiment_name": base_exp_name,
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
            "avg_backtracks_per_call": summary.get("avg_backtracks_per_call", 0.0),
            "fallback_rate": summary.get("fallback_rate", 0.0),
            
            # Retrieval metrics
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
        
        # Store category_accuracy dict as JSON string
        row["category_accuracy_json"] = json.dumps(summary.get("category_accuracy", {}))
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if not df.empty:
        df = df.sort_values(["methodology", "num_tools", "experiment_name"]).reset_index(drop=True)
    
    return df


def aggregate_across_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate experiment results across multiple runs."""
    if df.empty:
        return df
    
    if "base_experiment_name" not in df.columns:
        df = df.copy()
        df["base_experiment_name"] = df["experiment_name"]
    
    if "run_name" not in df.columns:
        df = df.copy()
        df["run_name"] = None
    
    numeric_cols = [
        "accuracy", "call_rate", "avg_latency_ms", "min_latency_ms", "max_latency_ms",
        "total_tests", "tool_correct", "tool_incorrect", "no_tool_called", "errors",
        "category_selection_accuracy", "avg_steps_per_call", "total_backtracks",
        "avg_backtracks_per_call", "fallback_rate", "retrieval_recall_rate", "avg_retrieval_rank",
        "avg_tokens_input", "avg_tokens_output", "avg_tokens_total",
        "adaptive_k_avg", "adaptive_k_min", "adaptive_k_max",
    ]
    
    group_cols = [
        "base_experiment_name", "methodology", "num_tools", "doc_length", "prompt_type",
        "phase", "top_k", "similarity_threshold", "allow_backtrack", "top_k_categories",
        "min_k", "max_k", "drop_threshold", "min_similarity", "num_similar_tools", "is_no_tool_test",
    ]
    
    grouped = df.groupby("base_experiment_name")
    
    agg_rows = []
    for base_name, group in grouped:
        row = {
            "experiment_name": base_name,
            "num_runs": len(group),
        }
        
        for col in group_cols:
            if col in group.columns:
                row[col] = group[col].iloc[0]
        
        for col in numeric_cols:
            if col in group.columns:
                values = group[col].dropna()
                if len(values) > 0:
                    row[f"{col}_mean"] = values.mean()
                    row[f"{col}_std"] = values.std() if len(values) > 1 else 0.0
                    row[col] = values.mean()
        
        if "category_accuracy_json" in group.columns:
            all_cat_acc = []
            for cat_json in group["category_accuracy_json"]:
                try:
                    cat_acc = json.loads(cat_json) if isinstance(cat_json, str) else {}
                    if cat_acc:
                        all_cat_acc.append(cat_acc)
                except:
                    continue
            
            if all_cat_acc:
                merged_cat_acc = {}
                all_categories = set()
                for cat_acc in all_cat_acc:
                    all_categories.update(cat_acc.keys())
                
                for cat in all_categories:
                    values = [ca.get(cat) for ca in all_cat_acc if cat in ca]
                    if values:
                        merged_cat_acc[cat] = sum(values) / len(values)
                
                row["category_accuracy_json"] = json.dumps(merged_cat_acc)
            else:
                row["category_accuracy_json"] = "{}"
        
        agg_rows.append(row)
    
    agg_df = pd.DataFrame(agg_rows)
    
    if not agg_df.empty:
        agg_df = agg_df.sort_values(["methodology", "num_tools", "experiment_name"]).reset_index(drop=True)
    
    return agg_df


def load_all_details_as_dataframe(results_dir: Path) -> pd.DataFrame:
    """Load all experiment details CSVs and combine into single DataFrame."""
    all_details = []
    
    for summary_path in find_experiments(results_dir):
        details_path = summary_path.with_name(
            summary_path.name.replace("_summary.json", "_details.csv")
        )
        if not details_path.exists():
            continue
        
        try:
            details = load_experiment_details(details_path)
            summary = load_experiment_summary(summary_path)
            config = summary.get("experiment_config", {})
            exp_name = config.get("name", summary_path.stem)
            details["experiment_name"] = exp_name
            details["num_tools"] = config.get("num_tools", 0)
            details["doc_length"] = config.get("doc_length", "medium")
            all_details.append(details)
        except Exception as e:
            logger.warning(f"Failed to load {details_path}: {e}")
            continue
    
    if all_details:
        return pd.concat(all_details, ignore_index=True)
    return pd.DataFrame()


def filter_latency_outliers(latencies: pd.Series, method: str = "iqr", factor: float = 1.5) -> pd.Series:
    """Filter outlier latencies using IQR or percentile method."""
    if method == "iqr":
        q1 = latencies.quantile(0.25)
        q3 = latencies.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        return latencies[(latencies >= lower_bound) & (latencies <= upper_bound)]
    else:
        p5 = latencies.quantile(0.05)
        p95 = latencies.quantile(0.95)
        return latencies[(latencies >= p5) & (latencies <= p95)]


# =============================================================================
# Visualization Setup
# =============================================================================

def setup_matplotlib():
    """Configure matplotlib for consistent styling."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
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


# =============================================================================
# Section 1: MCP Limits
# =============================================================================

def generate_mcp_accuracy_vs_tools(df: pd.DataFrame, output_path: Path):
    """
    Generate chart showing MCP accuracy degradation as tools increase.
    Shows the scaling limit where MCP fails (>=500 tools).
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    # Filter MCP experiments with medium verbosity
    mcp_df = df[(df["methodology"] == "mcp") & (df["doc_length"] == "medium")]
    
    if mcp_df.empty:
        # Try without doc_length filter
        mcp_df = df[df["methodology"] == "mcp"]
    
    if mcp_df.empty:
        logger.warning("No MCP experiments found")
        return
    
    # Aggregate by num_tools
    agg = mcp_df.groupby("num_tools").agg({
        "accuracy": ["mean", "std"],
        "errors": "sum",
        "total_tests": "sum"
    }).reset_index()
    agg.columns = ["num_tools", "accuracy_mean", "accuracy_std", "errors", "total_tests"]
    agg = agg.sort_values("num_tools")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    color = METHODOLOGY_COLORS["mcp"]
    
    # Plot accuracy line
    ax.plot(agg["num_tools"], agg["accuracy_mean"] * 100, 
            marker='o', linewidth=2, markersize=10,
            color=color, label="MCP Accuracy")
    
    # Add error band
    if agg["accuracy_std"].notna().any():
        ax.fill_between(
            agg["num_tools"],
            (agg["accuracy_mean"] - agg["accuracy_std"]) * 100,
            (agg["accuracy_mean"] + agg["accuracy_std"]) * 100,
            alpha=0.2, color=color
        )
    
    # Mark failed experiments (high error rate)
    error_threshold = 0.3  # >30% errors indicates failure
    for _, row in agg.iterrows():
        error_rate = row["errors"] / row["total_tests"] if row["total_tests"] > 0 else 0
        if error_rate > error_threshold:
            ax.scatter([row["num_tools"]], [row["accuracy_mean"] * 100], 
                      s=200, marker='x', color='red', linewidths=3, zorder=5)
            ax.annotate(f'High errors\n({error_rate*100:.0f}%)', 
                       (row["num_tools"], row["accuracy_mean"] * 100),
                       textcoords="offset points", xytext=(10, -20),
                       fontsize=9, color='red')
    
    # Add annotation for scaling limit
    max_tested = agg["num_tools"].max()
    ax.axvline(x=350, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.annotate('Practical limit\n(~350 tools)', xy=(350, 50), 
               fontsize=10, color='red', ha='center')
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MCP: Accuracy Degradation with Increasing Tools\n(Context window limits become critical at ~350+ tools)", 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Log scale for x-axis if range is large
    if agg["num_tools"].max() / agg["num_tools"].min() > 5:
        ax.set_xscale('log')
        ax.set_xticks(sorted(agg["num_tools"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_mcp_tokens_vs_tools(df: pd.DataFrame, output_path: Path):
    """
    Generate chart showing MCP token usage explosion as tools increase.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    # Filter MCP experiments with medium verbosity
    mcp_df = df[(df["methodology"] == "mcp") & (df["doc_length"] == "medium")]
    
    if mcp_df.empty:
        mcp_df = df[df["methodology"] == "mcp"]
    
    if mcp_df.empty:
        logger.warning("No MCP experiments found")
        return
    
    # Aggregate by num_tools
    agg = mcp_df.groupby("num_tools").agg({
        "avg_tokens_input": ["mean", "std"],
    }).reset_index()
    agg.columns = ["num_tools", "tokens_mean", "tokens_std"]
    agg = agg.sort_values("num_tools")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    color = METHODOLOGY_COLORS["mcp"]
    
    ax.plot(agg["num_tools"], agg["tokens_mean"], 
            marker='s', linewidth=2, markersize=10,
            color=color, label="Avg Input Tokens")
    
    if agg["tokens_std"].notna().any():
        ax.fill_between(
            agg["num_tools"],
            agg["tokens_mean"] - agg["tokens_std"],
            agg["tokens_mean"] + agg["tokens_std"],
            alpha=0.2, color=color
        )
    
    # Add typical context window limits as reference lines
    ax.axhline(y=64000, color='red', linestyle='--', alpha=0.7, linewidth=2, label='64K context limit')
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Average Input Tokens")
    ax.set_title("MCP: Token Usage Scaling with Tool Count\n(Linear growth makes large tool sets impractical)", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)
    
    if agg["num_tools"].max() / agg["num_tools"].min() > 5:
        # ax.set_xscale('log')
        ax.set_xticks(sorted(agg["num_tools"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


# =============================================================================
# Section 2: Clustering Analysis
# =============================================================================

def generate_clustering_accuracy_vs_tools(df: pd.DataFrame, output_path: Path):
    """
    Generate chart showing clustering accuracy across tool counts.
    Compare with MCP baseline where available.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for methodology in ["mcp", "clustering"]:
        meth_df = df[(df["methodology"] == methodology) & (df["doc_length"] == "medium")]
        if meth_df.empty:
            meth_df = df[df["methodology"] == methodology]
        
        if meth_df.empty:
            continue
        
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
        
        if agg["accuracy_std"].notna().any():
            ax.fill_between(
                agg["num_tools"],
                (agg["accuracy_mean"] - agg["accuracy_std"]) * 100,
                (agg["accuracy_mean"] + agg["accuracy_std"]) * 100,
                alpha=0.2, color=color
            )
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Clustering vs MCP: Accuracy by Tool Count\n(Clustering enables scaling but with lower accuracy)", 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', frameon=True)
    ax.grid(True, alpha=0.3)
    
    all_tools = df["num_tools"].unique()
    if len(all_tools) > 0 and max(all_tools) / min(all_tools) > 5:
        ax.set_xscale('log')
        ax.set_xticks(sorted(all_tools))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_clustering_tokens_vs_tools(df: pd.DataFrame, output_path: Path):
    """
    Generate chart showing clustering token usage compared to MCP.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for methodology in ["mcp", "clustering"]:
        meth_df = df[(df["methodology"] == methodology) & (df["doc_length"] == "medium")]
        if meth_df.empty:
            meth_df = df[df["methodology"] == methodology]
        
        if meth_df.empty:
            continue
        
        agg = meth_df.groupby("num_tools").agg({
            "avg_tokens_input": ["mean", "std"]
        }).reset_index()
        agg.columns = ["num_tools", "tokens_mean", "tokens_std"]
        agg = agg.sort_values("num_tools")
        
        color = METHODOLOGY_COLORS.get(methodology, "#666666")
        label = METHODOLOGY_DISPLAY_NAMES.get(methodology, methodology)
        
        ax.plot(agg["num_tools"], agg["tokens_mean"], 
                marker='s', linewidth=2, markersize=8,
                color=color, label=label)
        
        if agg["tokens_std"].notna().any():
            ax.fill_between(
                agg["num_tools"],
                agg["tokens_mean"] - agg["tokens_std"],
                agg["tokens_mean"] + agg["tokens_std"],
                alpha=0.2, color=color
            )
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Average Input Tokens")
    ax.set_title("Clustering vs MCP: Token Usage Scaling\n(Clustering maintains manageable token counts)", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)
    
    all_tools = df["num_tools"].unique()
    if len(all_tools) > 0 and max(all_tools) / min(all_tools) > 5:
        # ax.set_xscale('log')
        ax.set_xticks(sorted(all_tools))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_clustering_category_confusion_matrix(df: pd.DataFrame, details_df: pd.DataFrame, output_path: Path):
    """
    Generate confusion matrix for clustering category selection.
    Shows which categories are mistaken for others.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_matplotlib()
    
    # Filter clustering experiments
    cluster_details = details_df[details_df["methodology"] == "clustering"].copy()
    
    if cluster_details.empty:
        logger.warning("No clustering details available")
        return
    
    # Filter to rows with category info
    valid_rows = cluster_details[
        cluster_details["category"].notna() & 
        cluster_details["final_category"].notna()
    ].copy()
    
    if valid_rows.empty:
        logger.warning("No category selection data available")
        return
    
    # Clean category names
    valid_rows["expected"] = valid_rows["category"].str.replace("_operations", "")
    valid_rows["predicted"] = valid_rows["final_category"].str.replace("_operations", "")
    
    # Get top categories by frequency (for readability)
    top_categories = valid_rows["expected"].value_counts().head(15).index.tolist()
    
    # Filter to top categories
    filtered = valid_rows[
        valid_rows["expected"].isin(top_categories) & 
        valid_rows["predicted"].isin(top_categories)
    ]
    
    if filtered.empty:
        logger.warning("Not enough data for confusion matrix")
        return
    
    # Create confusion matrix
    confusion = pd.crosstab(
        filtered["expected"], 
        filtered["predicted"],
        normalize='index'  # Row-normalize to show % of expected category
    )
    
    # Reorder to put diagonal entries more visible
    category_order = sorted(confusion.index)
    confusion = confusion.reindex(index=category_order, columns=category_order, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        confusion * 100,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "% of Expected Category"},
        linewidths=0.5,
        vmin=0,
        vmax=100
    )
    
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Expected Category")
    ax.set_title("Clustering: Category Selection Confusion Matrix\n(Row-normalized, showing % of expected category predicted as each)", 
                 fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


# =============================================================================
# Section 3: Hybrid Methodology
# =============================================================================

def generate_hybrid_accuracy_vs_tools(df: pd.DataFrame, output_path: Path):
    """
    Generate chart showing hybrid accuracy compared to clustering and MCP.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for methodology in ["mcp", "clustering", "hybrid"]:
        meth_df = df[(df["methodology"] == methodology) & (df["doc_length"] == "medium")]
        if meth_df.empty:
            meth_df = df[df["methodology"] == methodology]
        
        if meth_df.empty:
            continue
        
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
        
        if agg["accuracy_std"].notna().any():
            ax.fill_between(
                agg["num_tools"],
                (agg["accuracy_mean"] - agg["accuracy_std"]) * 100,
                (agg["accuracy_mean"] + agg["accuracy_std"]) * 100,
                alpha=0.2, color=color
            )
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Hybrid: Accuracy Improvement over Clustering\n(RAG-based category selection improves accuracy)", 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', frameon=True)
    ax.grid(True, alpha=0.3)
    
    all_tools = df["num_tools"].unique()
    if len(all_tools) > 0 and max(all_tools) / min(all_tools) > 5:
        ax.set_xscale('log')
        ax.set_xticks(sorted(all_tools))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_hybrid_tokens_vs_tools(df: pd.DataFrame, output_path: Path):
    """
    Generate chart showing hybrid token usage - middle ground between MCP and clustering.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for methodology in ["mcp", "clustering", "hybrid"]:
        meth_df = df[(df["methodology"] == methodology) & (df["doc_length"] == "medium")]
        if meth_df.empty:
            meth_df = df[df["methodology"] == methodology]
        
        if meth_df.empty:
            continue
        
        agg = meth_df.groupby("num_tools").agg({
            "avg_tokens_input": ["mean", "std"]
        }).reset_index()
        agg.columns = ["num_tools", "tokens_mean", "tokens_std"]
        agg = agg.sort_values("num_tools")
        
        color = METHODOLOGY_COLORS.get(methodology, "#666666")
        label = METHODOLOGY_DISPLAY_NAMES.get(methodology, methodology)
        
        ax.plot(agg["num_tools"], agg["tokens_mean"], 
                marker='s', linewidth=2, markersize=8,
                color=color, label=label)
        
        if agg["tokens_std"].notna().any():
            ax.fill_between(
                agg["num_tools"],
                agg["tokens_mean"] - agg["tokens_std"],
                agg["tokens_mean"] + agg["tokens_std"],
                alpha=0.2, color=color
            )
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Average Input Tokens")
    ax.set_title("Hybrid: Token Usage Trade-off\n(More tokens than clustering, but scales to large tool counts)", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)
    
    all_tools = df["num_tools"].unique()
    if len(all_tools) > 0 and max(all_tools) / min(all_tools) > 5:
        # ax.set_xscale('log')
        ax.set_xticks(sorted(all_tools))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_hybrid_category_confusion_matrix(df: pd.DataFrame, details_df: pd.DataFrame, output_path: Path):
    """
    Generate confusion matrix for hybrid methodology category selection.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_matplotlib()
    
    # Filter hybrid experiments
    hybrid_details = details_df[details_df["methodology"] == "hybrid"].copy()
    
    if hybrid_details.empty:
        logger.warning("No hybrid details available")
        return
    
    valid_rows = hybrid_details[
        hybrid_details["category"].notna() & 
        hybrid_details["final_category"].notna()
    ].copy()
    
    if valid_rows.empty:
        logger.warning("No category selection data available for hybrid")
        return
    
    valid_rows["expected"] = valid_rows["category"].str.replace("_operations", "")
    valid_rows["predicted"] = valid_rows["final_category"].str.replace("_operations", "")
    
    top_categories = valid_rows["expected"].value_counts().head(15).index.tolist()
    
    filtered = valid_rows[
        valid_rows["expected"].isin(top_categories) & 
        valid_rows["predicted"].isin(top_categories)
    ]
    
    if filtered.empty:
        logger.warning("Not enough data for hybrid confusion matrix")
        return
    
    confusion = pd.crosstab(
        filtered["expected"], 
        filtered["predicted"],
        normalize='index'
    )
    
    category_order = sorted(confusion.index)
    confusion = confusion.reindex(index=category_order, columns=category_order, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        confusion * 100,
        annot=True,
        fmt=".0f",
        cmap="Purples",
        ax=ax,
        cbar_kws={"label": "% of Expected Category"},
        linewidths=0.5,
        vmin=0,
        vmax=100
    )
    
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Expected Category")
    ax.set_title("Hybrid: Category Selection Confusion Matrix\n(RAG-based selection should show better diagonal concentration)", 
                 fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_hybrid_category_count_impact(df: pd.DataFrame, output_path: Path):
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
    
    for num_tools in sorted(hybrid_df["num_tools"].unique()):
        subset = hybrid_df[hybrid_df["num_tools"] == num_tools]
        agg = subset.groupby("top_k_categories").agg({
            "accuracy": ["mean", "std"]
        }).reset_index()
        agg.columns = ["top_k_categories", "accuracy_mean", "accuracy_std"]
        agg = agg.sort_values("top_k_categories")
        
        ax.plot(agg["top_k_categories"], agg["accuracy_mean"] * 100, 
                marker='o', linewidth=2, markersize=8,
                label=f"{num_tools} tools")
        
        if agg["accuracy_std"].notna().any() and (agg["accuracy_std"] > 0).any():
            ax.fill_between(
                agg["top_k_categories"],
                (agg["accuracy_mean"] - agg["accuracy_std"]) * 100,
                (agg["accuracy_mean"] + agg["accuracy_std"]) * 100,
                alpha=0.2
            )
    
    ax.set_xlabel("Top-K Categories")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Hybrid: Impact of Category Count on Accuracy\n(More categories = higher recall but more context)", 
                 fontsize=14, fontweight='bold')
    ax.legend(title="Tool Count", loc='lower right', frameon=True)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(hybrid_df["top_k_categories"].unique()))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


# =============================================================================
# Section 4: RAG Improvement
# =============================================================================

def generate_rag_accuracy_vs_tools(df: pd.DataFrame, output_path: Path):
    """
    Generate chart showing RAG accuracy - stable across tool counts.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for methodology in ["mcp", "clustering", "hybrid", "rag"]:
        meth_df = df[(df["methodology"] == methodology) & (df["doc_length"] == "medium")]
        if meth_df.empty:
            meth_df = df[df["methodology"] == methodology]
        
        if meth_df.empty:
            continue
        
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
        
        if agg["accuracy_std"].notna().any():
            ax.fill_between(
                agg["num_tools"],
                (agg["accuracy_mean"] - agg["accuracy_std"]) * 100,
                (agg["accuracy_mean"] + agg["accuracy_std"]) * 100,
                alpha=0.2, color=color
            )
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("RAG: Accuracy Improvement via Direct Tool Retrieval\n(Bypasses category selection, maintains high accuracy)", 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', frameon=True)
    ax.grid(True, alpha=0.3)
    
    all_tools = df["num_tools"].unique()
    if len(all_tools) > 0 and max(all_tools) / min(all_tools) > 5:
        ax.set_xscale('log')
        ax.set_xticks(sorted(all_tools))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_rag_tokens_vs_tools(df: pd.DataFrame, output_path: Path):
    """
    Generate chart showing RAG token usage - relatively constant regardless of tool count.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for methodology in ["mcp", "clustering", "hybrid", "rag"]:
        meth_df = df[(df["methodology"] == methodology) & (df["doc_length"] == "medium")]
        if meth_df.empty:
            meth_df = df[df["methodology"] == methodology]
        
        if meth_df.empty:
            continue
        
        agg = meth_df.groupby("num_tools").agg({
            "avg_tokens_input": ["mean", "std"]
        }).reset_index()
        agg.columns = ["num_tools", "tokens_mean", "tokens_std"]
        agg = agg.sort_values("num_tools")
        
        color = METHODOLOGY_COLORS.get(methodology, "#666666")
        label = METHODOLOGY_DISPLAY_NAMES.get(methodology, methodology)
        
        ax.plot(agg["num_tools"], agg["tokens_mean"], 
                marker='s', linewidth=2, markersize=8,
                color=color, label=label)
        
        if agg["tokens_std"].notna().any():
            ax.fill_between(
                agg["num_tools"],
                agg["tokens_mean"] - agg["tokens_std"],
                agg["tokens_mean"] + agg["tokens_std"],
                alpha=0.2, color=color
            )
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Average Input Tokens")
    ax.set_title("RAG: Constant Token Usage Regardless of Tool Count\n(Fixed K tools retrieved = predictable context size)", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)
    
    all_tools = df["num_tools"].unique()
    if len(all_tools) > 0 and max(all_tools) / min(all_tools) > 5:
        # ax.set_xscale('log')
        ax.set_xticks(sorted(all_tools))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_rag_k_accuracy_impact(df: pd.DataFrame, output_path: Path):
    """
    Generate chart showing how varying fixed K in RAG impacts accuracy
    across different tool counts.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    # Filter for RAG experiments with top_k values
    rag_df = df[(df["methodology"] == "rag") & (df["top_k"].notna())]
    
    if rag_df.empty:
        logger.warning("No RAG experiments with varying K values found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get unique K values and tool counts
    k_values = sorted(rag_df["top_k"].unique())
    tool_counts = sorted(rag_df["num_tools"].unique())
    
    # Color palette for different K values
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(k_values)))
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    
    for i, k in enumerate(k_values):
        k_subset = rag_df[rag_df["top_k"] == k]
        
        agg = k_subset.groupby("num_tools").agg({
            "accuracy": ["mean", "std"]
        }).reset_index()
        agg.columns = ["num_tools", "accuracy_mean", "accuracy_std"]
        agg = agg.sort_values("num_tools")
        
        marker = markers[i % len(markers)]
        ax.plot(agg["num_tools"], agg["accuracy_mean"] * 100, 
                marker=marker, linewidth=2, markersize=9,
                color=colors[i], label=f"K={int(k)}")
        
        if agg["accuracy_std"].notna().any() and (agg["accuracy_std"] > 0).any():
            ax.fill_between(
                agg["num_tools"],
                (agg["accuracy_mean"] - agg["accuracy_std"]) * 100,
                (agg["accuracy_mean"] + agg["accuracy_std"]) * 100,
                alpha=0.15, color=colors[i]
            )
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("RAG: Impact of Fixed K on Accuracy Across Tool Counts\n(K higher than 15 yields diminishing returns)", 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(title="Top-K Retrieved", loc='lower left', frameon=True, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks to actual tool counts
    if len(tool_counts) > 0:
        ax.set_xticks(tool_counts)
        ax.set_xticklabels([str(t) for t in tool_counts])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_rag_k_tokens_impact(df: pd.DataFrame, output_path: Path):
    """
    Generate chart showing how varying fixed K in RAG impacts token usage
    across different tool counts.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    # Filter for RAG experiments with top_k values
    rag_df = df[(df["methodology"] == "rag") & (df["top_k"].notna())]
    
    if rag_df.empty:
        logger.warning("No RAG experiments with varying K values found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get unique K values and tool counts
    k_values = sorted(rag_df["top_k"].unique())
    tool_counts = sorted(rag_df["num_tools"].unique())
    
    # Color palette for different K values
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(k_values)))
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    
    for i, k in enumerate(k_values):
        k_subset = rag_df[rag_df["top_k"] == k]
        
        agg = k_subset.groupby("num_tools").agg({
            "avg_tokens_input": ["mean", "std"]
        }).reset_index()
        agg.columns = ["num_tools", "tokens_mean", "tokens_std"]
        agg = agg.sort_values("num_tools")
        
        marker = markers[i % len(markers)]
        ax.plot(agg["num_tools"], agg["tokens_mean"], 
                marker=marker, linewidth=2, markersize=9,
                color=colors[i], label=f"K={int(k)}")
        
        if agg["tokens_std"].notna().any() and (agg["tokens_std"] > 0).any():
            ax.fill_between(
                agg["num_tools"],
                agg["tokens_mean"] - agg["tokens_std"],
                agg["tokens_mean"] + agg["tokens_std"],
                alpha=0.15, color=colors[i]
            )
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Average Input Tokens")
    ax.set_title("RAG: Impact of Fixed K on Token Usage Across Tool Counts\n(Token usage scales with K, remains constant regardless of total tools)", 
                 fontsize=14, fontweight='bold')
    ax.legend(title="Top-K Retrieved", loc='upper left', frameon=True, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks to actual tool counts
    if len(tool_counts) > 0:
        ax.set_xticks(tool_counts)
        ax.set_xticklabels([str(t) for t in tool_counts])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_rag_recall_vs_accuracy(df: pd.DataFrame, output_path: Path):
    """
    Generate scatter plot comparing retrieval recall to final accuracy for RAG methods.
    This helps identify whether retrieval or LLM selection is the bottleneck.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    # Filter for RAG-based methodologies
    rag_methods = ["rag", "adaptive_rag", "hybrid"]
    rag_df = df[df["methodology"].isin(rag_methods) & 
                (df["retrieval_recall_rate"].notna()) & 
                (df["retrieval_recall_rate"] > 0) &
                (df["accuracy"].notna())]
    
    if rag_df.empty:
        logger.warning("No RAG experiments with both recall and accuracy data found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for methodology in rag_methods:
        meth_df = rag_df[rag_df["methodology"] == methodology]
        if meth_df.empty:
            continue
        
        color = METHODOLOGY_COLORS.get(methodology, "#666666")
        label = METHODOLOGY_DISPLAY_NAMES.get(methodology, methodology)
        
        ax.scatter(meth_df["retrieval_recall_rate"] * 100, 
                   meth_df["accuracy"] * 100,
                   c=color, label=label, s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
    
    # Add diagonal reference line (perfect retrieval -> accuracy)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Retrieval = Accuracy')
    
    ax.set_xlabel("Retrieval Recall (%)")
    ax.set_ylabel("Tool Selection Accuracy (%)")
    ax.set_title("Retrieval Recall vs. Final Accuracy\n(Gap indicates LLM selection errors from retrieved candidates)", 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('Points below diagonal:\nLLM struggling with selection', 
                xy=(85, 65), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


# =============================================================================
# Section 5: Adaptive RAG
# =============================================================================

def generate_adaptive_accuracy_vs_tools(df: pd.DataFrame, output_path: Path):
    """
    Generate chart showing Adaptive RAG accuracy compared to all other methodologies.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for methodology in ["mcp", "clustering", "hybrid", "rag", "adaptive_rag"]:
        meth_df = df[(df["methodology"] == methodology) & (df["doc_length"] == "medium")]
        if meth_df.empty:
            meth_df = df[df["methodology"] == methodology]
        
        if meth_df.empty:
            continue
        
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
        
        if agg["accuracy_std"].notna().any():
            ax.fill_between(
                agg["num_tools"],
                (agg["accuracy_mean"] - agg["accuracy_std"]) * 100,
                (agg["accuracy_mean"] + agg["accuracy_std"]) * 100,
                alpha=0.2, color=color
            )
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Adaptive RAG: Accuracy Comparison Across All Methodologies\n(Dynamic K selection maintains high accuracy at scale)", 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', frameon=True)
    ax.grid(True, alpha=0.3)
    
    all_tools = df["num_tools"].unique()
    if len(all_tools) > 0 and max(all_tools) / min(all_tools) > 5:
        ax.set_xscale('log')
        ax.set_xticks(sorted(all_tools))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_adaptive_tokens_vs_tools(df: pd.DataFrame, output_path: Path):
    """
    Generate chart showing Adaptive RAG token usage compared to all other methodologies.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for methodology in ["mcp", "clustering", "hybrid", "rag", "adaptive_rag"]:
        meth_df = df[(df["methodology"] == methodology) & (df["doc_length"] == "medium")]
        if meth_df.empty:
            meth_df = df[df["methodology"] == methodology]
        
        if meth_df.empty:
            continue
        
        agg = meth_df.groupby("num_tools").agg({
            "avg_tokens_input": ["mean", "std"]
        }).reset_index()
        agg.columns = ["num_tools", "tokens_mean", "tokens_std"]
        agg = agg.sort_values("num_tools")
        
        color = METHODOLOGY_COLORS.get(methodology, "#666666")
        label = METHODOLOGY_DISPLAY_NAMES.get(methodology, methodology)
        
        ax.plot(agg["num_tools"], agg["tokens_mean"], 
                marker='s', linewidth=2, markersize=8,
                color=color, label=label)
        
        if agg["tokens_std"].notna().any():
            ax.fill_between(
                agg["num_tools"],
                agg["tokens_mean"] - agg["tokens_std"],
                agg["tokens_mean"] + agg["tokens_std"],
                alpha=0.2, color=color
            )
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Average Input Tokens")
    ax.set_title("Adaptive RAG: Token Efficiency Across All Methodologies\n(Dynamic K provides efficient token usage while scaling)", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)
    
    all_tools = df["num_tools"].unique()
    if len(all_tools) > 0 and max(all_tools) / min(all_tools) > 5:
        ax.set_xticks(sorted(all_tools))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_adaptive_vs_rag_tokens(df: pd.DataFrame, output_path: Path):
    """
    Generate comparison of Adaptive RAG vs standard RAG token usage (verbose).
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for methodology in ["rag", "adaptive_rag"]:
        meth_df = df[(df["methodology"] == methodology) & (df["doc_length"] == "verbose")]
        if meth_df.empty:
            meth_df = df[(df["methodology"] == methodology) & (df["doc_length"] == "medium")]
        if meth_df.empty:
            meth_df = df[df["methodology"] == methodology]
        
        if meth_df.empty:
            continue
        
        agg = meth_df.groupby("num_tools").agg({
            "avg_tokens_input": ["mean", "std"]
        }).reset_index()
        agg.columns = ["num_tools", "tokens_mean", "tokens_std"]
        agg = agg.sort_values("num_tools")
        
        color = METHODOLOGY_COLORS.get(methodology, "#666666")
        label = METHODOLOGY_DISPLAY_NAMES.get(methodology, methodology)
        
        ax.plot(agg["num_tools"], agg["tokens_mean"], 
                marker='s', linewidth=2, markersize=8,
                color=color, label=label)
        
        if agg["tokens_std"].notna().any():
            ax.fill_between(
                agg["num_tools"],
                agg["tokens_mean"] - agg["tokens_std"],
                agg["tokens_mean"] + agg["tokens_std"],
                alpha=0.2, color=color
            )
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Average Input Tokens")
    ax.set_title("Adaptive RAG vs RAG: Token Usage Comparison (Verbose)\n(Dynamic K reduces context when fewer tools needed)", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)
    
    all_tools = df["num_tools"].unique()
    if len(all_tools) > 0 and max(all_tools) / min(all_tools) > 5:
        # ax.set_xscale('log')
        ax.set_xticks(sorted(all_tools))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_adaptive_prompt_clarity_comparison(df: pd.DataFrame, output_path: Path):
    """
    Generate comparison of Adaptive RAG accuracy with concise vs clear prompts.
    Shows how clearer prompts improve accuracy across tool counts.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter for adaptive_rag methodology
    adaptive_df = df[df["methodology"] == "adaptive_rag"]
    
    if adaptive_df.empty:
        logger.warning("No adaptive_rag data found")
        fig.text(0.5, 0.5, "No Adaptive RAG data available", ha='center', va='center', fontsize=14)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    prompt_styles = {
        "concise": {"color": "#FF6B6B", "marker": "o", "label": "Concise Prompts"},
        "clear": {"color": "#4ECDC4", "marker": "s", "label": "Clear Prompts"}
    }
    
    for prompt_type, style in prompt_styles.items():
        prompt_df = adaptive_df[adaptive_df["prompt_type"] == prompt_type]
        
        if prompt_df.empty:
            continue
        
        agg = prompt_df.groupby("num_tools").agg({
            "accuracy": ["mean", "std"]
        }).reset_index()
        agg.columns = ["num_tools", "accuracy_mean", "accuracy_std"]
        agg = agg.sort_values("num_tools")
        
        ax.plot(agg["num_tools"], agg["accuracy_mean"] * 100, 
                marker=style["marker"], linewidth=2.5, markersize=8,
                color=style["color"], label=style["label"])
        
        if agg["accuracy_std"].notna().any():
            ax.fill_between(
                agg["num_tools"],
                (agg["accuracy_mean"] - agg["accuracy_std"]) * 100,
                (agg["accuracy_mean"] + agg["accuracy_std"]) * 100,
                alpha=0.2, color=style["color"]
            )
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Adaptive RAG: Prompt Clarity Impact on Accuracy\n(Clear prompts provide better context for tool selection)", 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', frameon=True)
    ax.grid(True, alpha=0.3)
    
    all_tools = adaptive_df["num_tools"].unique()
    if len(all_tools) > 0 and max(all_tools) / min(all_tools) > 5:
        ax.set_xscale('log')
        ax.set_xticks(sorted(all_tools))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_adaptive_k_distribution(df: pd.DataFrame, details_df: pd.DataFrame, output_path: Path):
    """
    Generate histogram of adaptive_k_used values for adaptive RAG.
    Shows distribution of dynamically selected K values and strategy distribution.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_matplotlib()
    
    # Check if adaptive_k_used column exists
    if "adaptive_k_used" not in details_df.columns:
        logger.warning("No adaptive_k_used column in details - skipping adaptive K distribution chart")
        # Create a placeholder figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No adaptive K data available", ha='center', va='center', fontsize=14)
        ax.set_title("Distribution of Adaptive K Values", fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    adaptive_details = details_df[
        (details_df["methodology"] == "adaptive_rag") & 
        (details_df["adaptive_k_used"].notna()) &
        (details_df["adaptive_k_used"] > 0)
    ]
    
    if adaptive_details.empty:
        logger.warning("No adaptive_k_used data available")
        # Create a placeholder figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No adaptive K data available", ha='center', va='center', fontsize=14)
        ax.set_title("Distribution of Adaptive K Values", fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
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
    if "adaptive_strategy" in adaptive_details.columns and adaptive_details["adaptive_strategy"].notna().any():
        strategy_counts = adaptive_details["adaptive_strategy"].value_counts()
        colors = plt.cm.Set2(np.linspace(0, 1, len(strategy_counts)))
        ax2.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax2.set_title("Adaptive Strategy Distribution", fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "No strategy data available", ha='center', va='center', fontsize=12)
        ax2.set_title("Adaptive Strategy Distribution", fontweight='bold')
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_clustering_category_accuracy(df: pd.DataFrame, output_path: Path):
    """
    Generate line chart showing clustering methodology category selection accuracy
    as the number of tools increases.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    # Filter to clustering methodology only
    cluster_df = df[df["methodology"] == "clustering"].copy()
    
    if cluster_df.empty:
        logger.warning("No clustering experiments found")
        return
    
    # Check if we have category_selection_accuracy
    if "category_selection_accuracy" not in cluster_df.columns or cluster_df["category_selection_accuracy"].isna().all():
        logger.warning("No category_selection_accuracy data available for clustering")
        return
    
    # Group by num_tools
    agg = cluster_df.groupby("num_tools").agg({
        "category_selection_accuracy": ["mean", "std"],
        "accuracy": ["mean", "std"]
    }).reset_index()
    agg.columns = ["num_tools", "cat_acc_mean", "cat_acc_std", "tool_acc_mean", "tool_acc_std"]
    agg = agg.sort_values("num_tools")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot category selection accuracy
    ax.plot(agg["num_tools"], agg["cat_acc_mean"] * 100, 
            marker='o', linewidth=2, markersize=8,
            color=METHODOLOGY_COLORS["clustering"], label="Category Selection Accuracy")
    
    # Add error band
    if agg["cat_acc_std"].notna().any():
        ax.fill_between(
            agg["num_tools"],
            (agg["cat_acc_mean"] - agg["cat_acc_std"]) * 100,
            (agg["cat_acc_mean"] + agg["cat_acc_std"]) * 100,
            alpha=0.2, color=METHODOLOGY_COLORS["clustering"]
        )
    
    # Also plot tool accuracy for comparison
    ax.plot(agg["num_tools"], agg["tool_acc_mean"] * 100, 
            marker='s', linewidth=2, markersize=8, linestyle='--',
            color="#999999", label="Final Tool Accuracy")
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Clustering: Category Selection Accuracy vs Tool Count", fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', frameon=True)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    # Log scale if range is large
    if len(agg) > 1 and agg["num_tools"].max() / agg["num_tools"].min() > 10:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.set_xticks(agg["num_tools"].values)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_doc_length_impact_filtered(df: pd.DataFrame, output_path: Path):
    """
    Generate grouped bar chart showing accuracy by doc_length for each methodology.
    
    FILTERING: For each methodology, finds a (methodology, num_tools) pair where
    ALL 3 verbosity levels (minimal, medium, verbose) are present. Different
    methodologies may use different tool counts - this is the fairest comparison
    possible given the available data.
    """
    import matplotlib.pyplot as plt
    setup_matplotlib()
    
    # Filter to experiments that have doc_length
    df_filtered = df[df["doc_length"].notna()].copy()
    
    if df_filtered.empty or df_filtered["doc_length"].nunique() < 2:
        logger.warning("Not enough doc_length variation to plot")
        return
    
    # Required verbosity levels (3 levels, not 4 - "clear" is not used in experiments)
    required_verbosities = {"minimal", "medium", "verbose"}
    
    # For each methodology, find a tool count where all 3 verbosity levels are present
    # Prefer larger tool counts if multiple options exist
    methodology_best_pair = {}
    
    for (meth, num_tools), group in df_filtered.groupby(["methodology", "num_tools"]):
        available_verbosities = set(group["doc_length"].unique())
        if required_verbosities.issubset(available_verbosities):
            # This pair has all required verbosity levels
            if meth not in methodology_best_pair or num_tools > methodology_best_pair[meth][1]:
                methodology_best_pair[meth] = (meth, num_tools)
    
    if not methodology_best_pair:
        logger.warning("No methodology has complete verbosity coverage - cannot create fair comparison")
        # Create placeholder chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No data available with all verbosity levels\nfor fair comparison", 
                ha='center', va='center', fontsize=14)
        ax.set_title("Impact of Documentation Length on Accuracy", fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    valid_pairs = list(methodology_best_pair.values())
    valid_pairs_set = set(valid_pairs)
    
    # Filter to only valid pairs
    df_fair = df_filtered[
        df_filtered.apply(lambda row: (row["methodology"], row["num_tools"]) in valid_pairs_set, axis=1)
    ]
    
    # Also filter to only the 3 required verbosity levels
    df_fair = df_fair[df_fair["doc_length"].isin(required_verbosities)]
    
    logger.info(f"Doc length chart: using {len(valid_pairs)} methodology pairs: {valid_pairs}")
    
    # Pivot for grouped bars
    pivot = df_fair.pivot_table(
        values="accuracy",
        index="methodology",
        columns="doc_length",
        aggfunc="mean"
    )
    
    # Reorder columns: minimal, medium, verbose
    verbosity_order = ["minimal", "medium", "verbose"]
    doc_order = [d for d in verbosity_order if d in pivot.columns]
    pivot = pivot[doc_order]
    
    # Reorder rows by methodology order
    methodology_order = ["mcp", "clustering", "rag", "adaptive_rag", "hybrid"]
    pivot = pivot.reindex([m for m in methodology_order if m in pivot.index])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(pivot.index))
    width = 0.25  # Slightly wider bars for 3 groups
    
    # Colors for verbosity levels
    verbosity_colors = {"minimal": "#3498db", "medium": "#2ecc71", "verbose": "#e74c3c"}
    
    for i, doc_len in enumerate(pivot.columns):
        offset = (i - len(pivot.columns)/2 + 0.5) * width
        bars = ax.bar(x + offset, pivot[doc_len] * 100, width, 
                      label=doc_len.capitalize(), color=verbosity_colors.get(doc_len, None))
    
    # Create x-axis labels with tool count info
    x_labels = []
    for meth in pivot.index:
        display_name = METHODOLOGY_DISPLAY_NAMES.get(meth, meth)
        tool_count = methodology_best_pair[meth][1]
        x_labels.append(f"{display_name}\n({tool_count} tools)")
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha='center')
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Methodology")
    ax.set_title("Impact of Documentation Length on Accuracy\n(Each methodology at tool count with complete verbosity data)", 
                 fontsize=14, fontweight='bold')
    ax.legend(title="Doc Length", loc='upper right', frameon=True)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


# =============================================================================
# Section 6: Final Summary
# =============================================================================

def generate_latency_distribution_filtered(df: pd.DataFrame, details_df: pd.DataFrame, output_path: Path):
    """
    Generate latency boxplot distribution with outliers filtered.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_matplotlib()
    
    if details_df.empty or "latency_ms" not in details_df.columns:
        logger.warning("No latency data available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    order = [m for m in METHODOLOGY_COLORS.keys() if m in details_df["methodology"].unique()]
    palette = {m: METHODOLOGY_COLORS.get(m, "#666666") for m in order}
    
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
                    ax=ax, order=order, hue="methodology", palette=palette, 
                    legend=False, showfliers=False)
    
    ax.set_title("Latency Distribution by Methodology (Outliers Filtered)", fontweight='bold', fontsize=14)
    ax.set_xlabel("Methodology")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([METHODOLOGY_DISPLAY_NAMES.get(m, m) for m in order], rotation=30, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def generate_accuracy_heatmap_summary(df: pd.DataFrame, output_path: Path):
    """
    Generate heatmap of accuracy by methodology and num_tools.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    setup_matplotlib()

    # Exclude from mcp data where tool counts is 300 or 350
    df = df[~((df["methodology"] == "mcp") & (df["num_tools"].isin([300, 350])))]
    
    # Pivot for heatmap
    pivot = df.pivot_table(
        values="accuracy",
        index="methodology",
        columns="num_tools",
        aggfunc="mean"
    )
    
    # Sort index by METHODOLOGY_COLORS order
    method_order = [m for m in METHODOLOGY_COLORS.keys() if m in pivot.index]
    pivot = pivot.reindex(method_order)
    
    # Sort columns numerically
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
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
    ax.set_title("Accuracy Heatmap: Methodology × Tool Count\n(Comprehensive view of methodology performance)", 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


# =============================================================================
# HTML Report Generation
# =============================================================================

def generate_limits_html_report(
    df: pd.DataFrame,
    details_df: pd.DataFrame,
    output_path: Path,
    figures_dir: Path
):
    """Generate comprehensive HTML report for methodology limits analysis."""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Methodology Limits Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #e74c3c; padding-bottom: 15px; }}
        h2 {{ color: #555; border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-top: 40px; }}
        h3 {{ color: #666; margin-top: 30px; }}
        .section {{ margin: 30px 0; padding: 20px; background: #fafafa; border-radius: 8px; border-left: 4px solid #3498db; }}
        .section.mcp {{ border-left-color: {METHODOLOGY_COLORS['mcp']}; }}
        .section.clustering {{ border-left-color: {METHODOLOGY_COLORS['clustering']}; }}
        .section.hybrid {{ border-left-color: {METHODOLOGY_COLORS['hybrid']}; }}
        .section.rag {{ border-left-color: {METHODOLOGY_COLORS['rag']}; }}
        .section.adaptive {{ border-left-color: {METHODOLOGY_COLORS['adaptive_rag']}; }}
        .section.summary {{ border-left-color: #2ecc71; }}
        .figure {{ margin: 20px 0; text-align: center; }}
        .figure img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .figure-caption {{ margin-top: 10px; color: #666; font-style: italic; }}
        .key-points {{ background: #e8f6ff; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .key-points ul {{ margin: 0; padding-left: 20px; }}
        .key-points li {{ margin: 8px 0; }}
        .limit-badge {{ display: inline-block; background: #e74c3c; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; margin: 10px 5px; }}
        .solution-badge {{ display: inline-block; background: #2ecc71; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; margin: 10px 5px; }}
        .tradeoff-badge {{ display: inline-block; background: #f39c12; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; margin: 10px 5px; }}
        .nav {{ position: fixed; top: 20px; right: 20px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 200px; }}
        .nav a {{ display: block; padding: 5px 0; color: #3498db; text-decoration: none; }}
        .nav a:hover {{ color: #2980b9; }}
        @media (max-width: 1600px) {{ .nav {{ display: none; }} }}
    </style>
</head>
<body>
    <nav class="nav">
        <strong>Navigation</strong>
        <a href="#section-1">1. MCP Limits</a>
        <a href="#section-2">2. Clustering</a>
        <a href="#section-3">3. Hybrid</a>
        <a href="#section-4">4. RAG</a>
        <a href="#section-5">5. Adaptive RAG</a>
        <a href="#section-6">6. Summary</a>
    </nav>
    
    <div class="container">
        <h1>🔧 Methodology Limits Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Experiments analyzed:</strong> {len(df)} experiments across {df['methodology'].nunique()} methodologies</p>
        <p>This report shows where each tool-calling methodology reaches its limits and how subsequent methodologies address those limitations.</p>
        
        <!-- Section 1: MCP Limits -->
        <div id="section-1" class="section mcp">
            <h2>📊 Section 1: MCP (Baseline) - Scaling Limits</h2>
            <p>The Model Context Protocol (MCP) approach passes <strong>all available tools</strong> to the LLM in a single call. 
            While this achieves high accuracy with small tool sets, it fails to scale due to context window limitations.</p>
            
            <div class="key-points">
                <strong>Key Limitations:</strong>
                <ul>
                    <li><span class="limit-badge">Context Window</span> Token count grows linearly with tool count</li>
                    <li><span class="limit-badge">Practical Limit</span> Cannot reliably test with ≥500 tools</li>
                    <li><span class="limit-badge">Cost</span> High token usage = higher API costs</li>
                </ul>
            </div>
            
            <div class="figure">
                <img src="limits_figures/01_mcp_accuracy_vs_tools.png" alt="MCP Accuracy vs Tools">
                <p class="figure-caption">MCP accuracy degradation as tool count increases. Red markers indicate experiments with high error rates.</p>
            </div>
            
            <div class="figure">
                <img src="limits_figures/02_mcp_tokens_vs_tools.png" alt="MCP Tokens vs Tools">
                <p class="figure-caption">Token usage scales linearly with tool count, quickly exceeding context limits.</p>
            </div>
        </div>
        
        <!-- Section 2: Clustering -->
        <div id="section-2" class="section clustering">
            <h2>📊 Section 2: Clustering - Hierarchical Selection</h2>
            <p>Clustering introduces a <strong>two-step hierarchical approach</strong>: first select a category, then select a tool within that category.
            This enables scaling to large tool sets but introduces category selection errors.</p>
            
            <div class="key-points">
                <strong>Trade-offs:</strong>
                <ul>
                    <li><span class="solution-badge">Scalability</span> Can handle 500+ tools with manageable context</li>
                    <li><span class="limit-badge">Accuracy Loss</span> Category selection errors cascade to tool selection</li>
                    <li><span class="tradeoff-badge">Backtracking</span> Allows recovery from wrong category selection at cost of more steps</li>
                </ul>
            </div>
            
            <div class="figure">
                <img src="limits_figures/03_clustering_accuracy_vs_tools.png" alt="Clustering Accuracy vs Tools">
                <p class="figure-caption">Clustering accuracy compared to MCP baseline across tool counts.</p>
            </div>
            
            <div class="figure">
                <img src="limits_figures/04_clustering_tokens_vs_tools.png" alt="Clustering Tokens vs Tools">
                <p class="figure-caption">Clustering maintains lower token usage even with large tool sets.</p>
            </div>
            
            <div class="figure">
                <img src="limits_figures/05_clustering_category_confusion.png" alt="Clustering Category Confusion">
                <p class="figure-caption">Category confusion matrix showing which categories are mistaken for others.</p>
            </div>
            
            <h3>Category Selection Accuracy vs Tool Count</h3>
            
            <div class="figure">
                <img src="limits_figures/07a_clustering_category_accuracy.png" alt="Clustering Category Accuracy">
                <p class="figure-caption">Category selection accuracy compared to final tool accuracy as tool count increases.</p>
            </div>
        </div>
        
        <!-- Section 3: Hybrid -->
        <div id="section-3" class="section hybrid">
            <h2>📊 Section 3: Hybrid - RAG-Enhanced Category Selection</h2>
            <p>The Hybrid methodology uses <strong>RAG to retrieve relevant categories</strong> instead of presenting all categories to the LLM.
            This improves category selection accuracy while maintaining the ability to scale.</p>
            
            <div class="key-points">
                <strong>Improvements:</strong>
                <ul>
                    <li><span class="solution-badge">Better Category Selection</span> RAG filters to most relevant categories</li>
                    <li><span class="tradeoff-badge">Token Trade-off</span> More tokens than pure clustering, less than MCP</li>
                    <li><span class="solution-badge">Scales to 500-985 tools</span> Successfully tested at large scale</li>
                </ul>
            </div>
            
            <div class="figure">
                <img src="limits_figures/08_hybrid_accuracy_vs_tools.png" alt="Hybrid Accuracy vs Tools">
                <p class="figure-caption">Hybrid accuracy improvement over clustering.</p>
            </div>
            
            <div class="figure">
                <img src="limits_figures/09_hybrid_tokens_vs_tools.png" alt="Hybrid Tokens vs Tools">
                <p class="figure-caption">Hybrid token usage - a middle ground between MCP and clustering.</p>
            </div>
            
            <div class="figure">
                <img src="limits_figures/10_hybrid_category_confusion.png" alt="Hybrid Category Confusion">
                <p class="figure-caption">Hybrid category selection confusion matrix - should show better diagonal concentration.</p>
            </div>
            
            <div class="figure">
                <img src="limits_figures/11_hybrid_category_count.png" alt="Hybrid Category Count Impact">
                <p class="figure-caption">Impact of top-K categories on hybrid accuracy.</p>
            </div>
        </div>
        
        <!-- Section 4: RAG -->
        <div id="section-4" class="section rag">
            <h2>📊 Section 4: RAG - Direct Tool Retrieval</h2>
            <p>RAG bypasses category selection entirely, using <strong>vector similarity to retrieve the most relevant tools directly</strong>.
            This provides high accuracy with constant token usage regardless of total tool count.</p>
            
            <div class="key-points">
                <strong>Key Advantages:</strong>
                <ul>
                    <li><span class="solution-badge">No Category Errors</span> Direct tool retrieval eliminates category selection failures</li>
                    <li><span class="solution-badge">Constant Context</span> Fixed K tools = predictable token usage</li>
                    <li><span class="solution-badge">High Accuracy</span> Often matches or exceeds MCP baseline</li>
                </ul>
            </div>
            
            <div class="figure">
                <img src="limits_figures/12_rag_accuracy_vs_tools.png" alt="RAG Accuracy vs Tools">
                <p class="figure-caption">RAG accuracy remains stable regardless of total tool count.</p>
            </div>
            
            <div class="figure">
                <img src="limits_figures/13_rag_tokens_vs_tools.png" alt="RAG Tokens vs Tools">
                <p class="figure-caption">RAG token usage is constant - independent of total tools.</p>
            </div>
            
            <h3>Impact of K (Retrieved Tools) on Performance</h3>
            <p>The fixed K parameter determines how many tools are retrieved by the embedding similarity search. 
            Analyzing how different K values affect accuracy and token usage helps find the optimal balance.</p>
            
            <div class="figure">
                <img src="limits_figures/14a_rag_k_accuracy_impact.png" alt="RAG K Accuracy Impact">
                <p class="figure-caption">Impact of fixed K on accuracy across different tool counts. K higher than 15 yields diminishing returns.</p>
            </div>
            
            <div class="figure">
                <img src="limits_figures/14b_rag_k_tokens_impact.png" alt="RAG K Tokens Impact">
                <p class="figure-caption">Impact of fixed K on token usage. Token usage scales linearly with K but remains constant regardless of total tool count.</p>
            </div>
            
            <h3>Retrieval Recall Analysis</h3>
            <p>Retrieval recall measures whether the correct tool was present in the retrieved candidate set. 
            High retrieval recall with lower final accuracy indicates the LLM is struggling with selection from candidates; 
            low retrieval recall indicates the embedding-based retrieval is the bottleneck.</p>
            
            <div class="figure">
                <img src="limits_figures/14d_rag_recall_vs_accuracy.png" alt="Recall vs Accuracy">
                <p class="figure-caption">Retrieval recall vs final accuracy. Points below the diagonal indicate LLM selection errors from retrieved candidates.</p>
            </div>
        </div>
        
        <!-- Section 5: Adaptive RAG -->
        <div id="section-5" class="section adaptive">
            <h2>📊 Section 5: Adaptive RAG - Dynamic K Selection</h2>
            <p>Adaptive RAG extends RAG by <strong>dynamically selecting K based on similarity score patterns</strong>.
            When a query clearly matches one tool, fewer tools are retrieved; when ambiguous, more are retrieved.</p>
            
            <div class="key-points">
                <strong>Optimization:</strong>
                <ul>
                    <li><span class="solution-badge">Context Efficiency</span> Reduces tokens when fewer tools are needed</li>
                    <li><span class="solution-badge">Maintained Accuracy</span> Dynamic K doesn't sacrifice accuracy</li>
                    <li><span class="tradeoff-badge">Complexity</span> Requires tuning of similarity thresholds</li>
                </ul>
            </div>
            
            <h3>Comparison Across All Methodologies</h3>
            
            <div class="figure">
                <img src="limits_figures/15_adaptive_accuracy_vs_tools.png" alt="Adaptive RAG Accuracy vs All">
                <p class="figure-caption">Adaptive RAG accuracy compared to all other methodologies across tool counts.</p>
            </div>
            
            <div class="figure">
                <img src="limits_figures/15a_adaptive_tokens_vs_tools.png" alt="Adaptive RAG Tokens vs All">
                <p class="figure-caption">Adaptive RAG token efficiency compared to all other methodologies.</p>
            </div>
            
            <h3>Adaptive RAG vs Standard RAG</h3>
            
            <div class="figure">
                <img src="limits_figures/17_adaptive_vs_rag_tokens.png" alt="Adaptive vs RAG Tokens">
                <p class="figure-caption">Adaptive RAG token savings compared to fixed-K RAG.</p>
            </div>
            
            <h3>Adaptive RAG Prompt Clarity Impact</h3>
            
            <div class="figure">
                <img src="limits_figures/18_adaptive_prompt_clarity.png" alt="Adaptive Prompt Clarity">
                <p class="figure-caption">Impact of prompt clarity on Adaptive RAG accuracy - clear prompts vs concise prompts.</p>
            </div>
            
            <h3>Adaptive K Value Distribution</h3>
            
            <div class="figure">
                <img src="limits_figures/18a_adaptive_k_distribution.png" alt="Adaptive K Distribution">
                <p class="figure-caption">Distribution of dynamically selected K values and strategy distribution. Shows how adaptive RAG adjusts context size based on query clarity.</p>
            </div>
        </div>
        
        <!-- Section 6: Summary -->
        <div id="section-6" class="section summary">
            <h2>📊 Section 6: Final Summary</h2>
            <p>Comprehensive comparison of all methodologies across accuracy and latency dimensions.</p>
            
            <div class="figure">
                <img src="limits_figures/19_latency_distribution.png" alt="Latency Distribution">
                <p class="figure-caption">Latency distribution across all methodologies (outliers filtered).</p>
            </div>
            
            <div class="figure">
                <img src="limits_figures/20_accuracy_heatmap.png" alt="Accuracy Heatmap">
                <p class="figure-caption">Complete accuracy heatmap: methodology × tool count.</p>
            </div>
            
            <h3>Documentation Length Impact</h3>
            
            <div class="figure">
                <img src="limits_figures/21_doc_length_impact.png" alt="Doc Length Impact">
                <p class="figure-caption">Impact of documentation length on accuracy. Only includes (methodology, tool_count) pairs where all verbosity levels were tested for fair comparison.</p>
            </div>
        </div>
        
        <h2>🔍 Key Takeaways</h2>
        <div class="key-points">
            <ul>
                <li><strong>MCP</strong> works well for small tool sets (&lt;200) but doesn't scale</li>
                <li><strong>Clustering</strong> enables scale but loses accuracy due to category selection errors</li>
                <li><strong>Hybrid</strong> improves on clustering by using RAG for category selection</li>
                <li><strong>RAG</strong> provides the best balance: high accuracy with constant token usage</li>
                <li><strong>Adaptive RAG</strong> optimizes RAG further by dynamically adjusting context size</li>
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

@app.command(name="generate-report")
def generate_report(
    results_dir: Path = typer.Option(
        Path("experiments/results/tmp"),
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
        help="Aggregate results across multiple runs"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """Generate methodology limits analysis report with all visualizations."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")
    
    logger.info(f"Loading experiments from {results_dir}")
    df = load_all_experiments_as_dataframe(results_dir)
    
    if df.empty:
        logger.error("No experiments found")
        raise typer.Exit(1)
    
    logger.info(f"Loaded {len(df)} experiment results")
    
    if aggregate_runs:
        has_runs = "run_name" in df.columns and df["run_name"].notna().any()
        if has_runs:
            logger.info("Aggregating results across multiple runs...")
            df = aggregate_across_runs(df)
            logger.info(f"Aggregated to {len(df)} unique experiments")
    
    details_df = load_all_details_as_dataframe(results_dir)
    logger.info(f"Loaded {len(details_df)} detailed test records")
    
    # Create output directories
    figures_dir = output_dir / "limits_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating visualizations...")
    
    # Section 1: MCP Limits
    logger.info("Generating Section 1: MCP Limits...")
    generate_mcp_accuracy_vs_tools(df, figures_dir / "01_mcp_accuracy_vs_tools.png")
    generate_mcp_tokens_vs_tools(df, figures_dir / "02_mcp_tokens_vs_tools.png")
    
    # Section 2: Clustering Analysis
    logger.info("Generating Section 2: Clustering Analysis...")
    generate_clustering_accuracy_vs_tools(df, figures_dir / "03_clustering_accuracy_vs_tools.png")
    generate_clustering_tokens_vs_tools(df, figures_dir / "04_clustering_tokens_vs_tools.png")
    generate_clustering_category_confusion_matrix(df, details_df, figures_dir / "05_clustering_category_confusion.png")
    generate_clustering_category_accuracy(df, figures_dir / "07a_clustering_category_accuracy.png")
    
    # Section 3: Hybrid Methodology
    logger.info("Generating Section 3: Hybrid Methodology...")
    generate_hybrid_accuracy_vs_tools(df, figures_dir / "08_hybrid_accuracy_vs_tools.png")
    generate_hybrid_tokens_vs_tools(df, figures_dir / "09_hybrid_tokens_vs_tools.png")
    generate_hybrid_category_confusion_matrix(df, details_df, figures_dir / "10_hybrid_category_confusion.png")
    generate_hybrid_category_count_impact(df, figures_dir / "11_hybrid_category_count.png")
    
    # Section 4: RAG Improvement
    logger.info("Generating Section 4: RAG Improvement...")
    generate_rag_accuracy_vs_tools(df, figures_dir / "12_rag_accuracy_vs_tools.png")
    generate_rag_tokens_vs_tools(df, figures_dir / "13_rag_tokens_vs_tools.png")
    generate_rag_k_accuracy_impact(df, figures_dir / "14a_rag_k_accuracy_impact.png")
    generate_rag_k_tokens_impact(df, figures_dir / "14b_rag_k_tokens_impact.png")
    generate_rag_recall_vs_accuracy(df, figures_dir / "14d_rag_recall_vs_accuracy.png")
    
    # Section 5: Adaptive RAG
    logger.info("Generating Section 5: Adaptive RAG...")
    generate_adaptive_accuracy_vs_tools(df, figures_dir / "15_adaptive_accuracy_vs_tools.png")
    generate_adaptive_tokens_vs_tools(df, figures_dir / "15a_adaptive_tokens_vs_tools.png")
    generate_adaptive_vs_rag_tokens(df, figures_dir / "17_adaptive_vs_rag_tokens.png")
    generate_adaptive_prompt_clarity_comparison(df, figures_dir / "18_adaptive_prompt_clarity.png")
    generate_adaptive_k_distribution(df, details_df, figures_dir / "18a_adaptive_k_distribution.png")
    
    # Section 6: Final Summary
    logger.info("Generating Section 6: Final Summary...")
    generate_latency_distribution_filtered(df, details_df, figures_dir / "19_latency_distribution.png")
    generate_accuracy_heatmap_summary(df, figures_dir / "20_accuracy_heatmap.png")
    generate_doc_length_impact_filtered(df, figures_dir / "21_doc_length_impact.png")
    
    # Generate HTML report
    generate_limits_html_report(df, details_df, output_dir / "limits_report.html", figures_dir)
    
    logger.info(f"Report generated at {output_dir / 'limits_report.html'}")
    print(f"\n✅ Limits report generated: {output_dir / 'limits_report.html'}")
    print(f"   Figures saved to: {figures_dir}")


if __name__ == "__main__":
    app()
