#!/usr/bin/env python
"""
Phase 3: Analysis and Comparison Script

This script provides comprehensive analysis capabilities for experiment results:
1. Side-by-side methodology comparison
2. Statistical significance testing (paired t-test, bootstrap CI)
3. Confusion matrix generation (category-level)
4. Visualization generation (accuracy curves, latency distributions)

Usage:
    # Compare all experiments in results directory
    python scripts/analyze_results.py compare --results-dir experiments/results
    
    # Compare specific experiments
    python scripts/analyze_results.py compare --experiments exp1_summary.json exp2_summary.json
    
    # Generate visualizations
    python scripts/analyze_results.py visualize --results-dir experiments/results --output-dir reports/figures
    
    # Generate confusion matrix
    python scripts/analyze_results.py confusion --details-csv experiments/results/clustering_*.csv
    
    # Full analysis report
    python scripts/analyze_results.py report --results-dir experiments/results --output reports/analysis.html
"""
import sys
import json
import glob
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

app = typer.Typer(help="Analyze and compare experiment results")


# =============================================================================
# Data Loading
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


def load_all_experiments(results_dir: Path) -> tuple[list[dict], list[pd.DataFrame]]:
    """Load all experiments from a results directory."""
    summaries = []
    details = []
    
    for summary_path in find_experiments(results_dir):
        summary = load_experiment_summary(summary_path)
        summaries.append(summary)
        
        # Load corresponding details CSV
        details_path = summary_path.with_name(
            summary_path.name.replace("_summary.json", "_details.csv")
        )
        if details_path.exists():
            details.append(load_experiment_details(details_path))
        else:
            logger.warning(f"Details file not found: {details_path}")
            details.append(None)
    
    return summaries, details


# =============================================================================
# Statistical Analysis
# =============================================================================

def paired_t_test(accuracies_a: list[float], accuracies_b: list[float]) -> dict:
    """
    Perform paired t-test between two sets of per-test accuracies.
    
    Returns:
        dict with t-statistic, p-value, and significance at various levels
    """
    if len(accuracies_a) != len(accuracies_b):
        raise ValueError("Sample sizes must match for paired t-test")
    
    # Check for zero variance (identical data) - t-test is undefined
    diff = np.array(accuracies_a) - np.array(accuracies_b)
    if np.var(diff) == 0:
        # No difference between methods (or both perfect/identical)
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,  # No significant difference
            "significant_0.05": False,
            "significant_0.01": False,
            "significant_0.001": False,
            "n_samples": len(accuracies_a),
            "note": "Zero variance in differences (identical results)"
        }
    
    # Suppress expected warnings for near-identical data
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Precision loss.*')
        warnings.filterwarnings('ignore', message='.*divide by zero.*')
        warnings.filterwarnings('ignore', message='.*invalid value.*')
        t_stat, p_value = stats.ttest_rel(accuracies_a, accuracies_b)
    
    # Handle NaN p-values
    if np.isnan(p_value):
        p_value = 1.0
        t_stat = 0.0 if np.isnan(t_stat) else t_stat
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_0.05": p_value < 0.05,
        "significant_0.01": p_value < 0.01,
        "significant_0.001": p_value < 0.001,
        "n_samples": len(accuracies_a),
    }


def bootstrap_confidence_interval(
    data: list[float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    statistic: str = "mean"
) -> dict:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        statistic: 'mean' or 'median'
    
    Returns:
        dict with point estimate, CI bounds, and standard error
    """
    data = np.array(data)
    n = len(data)
    
    # Compute point estimate
    if statistic == "mean":
        point_estimate = np.mean(data)
        stat_func = np.mean
    else:
        point_estimate = np.median(data)
        stat_func = np.median
    
    # Bootstrap resampling
    bootstrap_stats = []
    rng = np.random.default_rng(seed=42)
    
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute confidence interval (percentile method)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return {
        "point_estimate": float(point_estimate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence_level": confidence_level,
        "standard_error": float(np.std(bootstrap_stats)),
        "n_bootstrap": n_bootstrap,
    }


def compare_methodologies_statistical(
    details_a: pd.DataFrame,
    details_b: pd.DataFrame,
    methodology_a: str,
    methodology_b: str
) -> dict:
    """
    Comprehensive statistical comparison between two methodologies.
    
    Args:
        details_a, details_b: DataFrames with per-test results
        methodology_a, methodology_b: Names of methodologies
    
    Returns:
        dict with t-test results, bootstrap CIs, and effect size
    """
    # Extract accuracy per test (tool_correct is boolean)
    acc_a = details_a["tool_correct"].astype(float).tolist()
    acc_b = details_b["tool_correct"].astype(float).tolist()
    
    # Paired t-test (if same tests)
    if len(acc_a) == len(acc_b):
        t_test = paired_t_test(acc_a, acc_b)
    else:
        # Use independent t-test if different number of tests
        # Check for zero variance edge case
        if np.var(acc_a) == 0 and np.var(acc_b) == 0:
            t_stat, p_value = 0.0, 1.0
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*Precision loss.*')
                warnings.filterwarnings('ignore', message='.*divide by zero.*')
                t_stat, p_value = stats.ttest_ind(acc_a, acc_b)
            if np.isnan(p_value):
                p_value = 1.0
                t_stat = 0.0 if np.isnan(t_stat) else t_stat
        
        t_test = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_0.05": p_value < 0.05,
            "note": "Independent t-test (different sample sizes)"
        }
    
    # Bootstrap CIs for each
    ci_a = bootstrap_confidence_interval(acc_a)
    ci_b = bootstrap_confidence_interval(acc_b)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(acc_a) + np.var(acc_b)) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(acc_a) - np.mean(acc_b)) / pooled_std
    else:
        cohens_d = 0.0
    
    return {
        "methodology_a": methodology_a,
        "methodology_b": methodology_b,
        "accuracy_a": float(np.mean(acc_a)),
        "accuracy_b": float(np.mean(acc_b)),
        "difference": float(np.mean(acc_a) - np.mean(acc_b)),
        "t_test": t_test,
        "bootstrap_ci_a": ci_a,
        "bootstrap_ci_b": ci_b,
        "cohens_d": float(cohens_d),
        "effect_size_interpretation": interpret_cohens_d(cohens_d),
    }


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


# =============================================================================
# Confusion Matrix Analysis
# =============================================================================

def build_category_confusion_matrix(details_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build confusion matrix for category selection (clustering/hybrid methodologies).
    
    Args:
        details_df: DataFrame with 'category' (expected) and 'final_category' (selected) columns
    
    Returns:
        DataFrame representing the confusion matrix
    """
    # Filter to rows that have category information
    df = details_df.dropna(subset=["category", "final_category"])
    
    if len(df) == 0:
        logger.warning("No category data available for confusion matrix")
        return pd.DataFrame()
    
    # Get unique categories
    all_categories = sorted(set(df["category"].unique()) | set(df["final_category"].unique()))
    
    # Build confusion matrix
    matrix = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        expected = row["category"]
        actual = row["final_category"]
        matrix[expected][actual] += 1
    
    # Convert to DataFrame
    confusion_df = pd.DataFrame(0, index=all_categories, columns=all_categories)
    for expected in matrix:
        for actual in matrix[expected]:
            confusion_df.loc[expected, actual] = matrix[expected][actual]
    
    return confusion_df


def analyze_confusion_matrix(confusion_df: pd.DataFrame) -> dict:
    """
    Analyze confusion matrix to identify problematic category pairs.
    
    Returns:
        dict with per-category accuracy, most confused pairs, etc.
    """
    if confusion_df.empty:
        return {"error": "Empty confusion matrix"}
    
    # Per-category accuracy (diagonal / row sum)
    category_accuracy = {}
    for cat in confusion_df.index:
        row_sum = confusion_df.loc[cat].sum()
        if row_sum > 0:
            category_accuracy[cat] = confusion_df.loc[cat, cat] / row_sum
        else:
            category_accuracy[cat] = 0.0
    
    # Find most confused pairs (off-diagonal elements)
    confused_pairs = []
    for expected in confusion_df.index:
        for actual in confusion_df.columns:
            if expected != actual and confusion_df.loc[expected, actual] > 0:
                confused_pairs.append({
                    "expected": expected,
                    "selected": actual,
                    "count": int(confusion_df.loc[expected, actual]),
                })
    
    # Sort by count descending
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)
    
    # Overall accuracy
    total_correct = sum(confusion_df.loc[c, c] for c in confusion_df.index)
    total = confusion_df.values.sum()
    overall_accuracy = total_correct / total if total > 0 else 0.0
    
    return {
        "overall_accuracy": float(overall_accuracy),
        "category_accuracy": category_accuracy,
        "most_confused_pairs": confused_pairs[:10],  # Top 10
        "total_samples": int(total),
    }


# =============================================================================
# Visualization
# =============================================================================

def generate_accuracy_comparison_chart(
    summaries: list[dict],
    output_path: Optional[Path] = None
):
    """
    Generate bar chart comparing accuracy across methodologies.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib and seaborn required for visualization")
        return
    
    # Extract data
    methodologies = []
    accuracies = []
    
    for summary in summaries:
        methodology = summary.get("methodology", "unknown")
        accuracy = summary.get("accuracy", 0.0)
        methodologies.append(methodology)
        accuracies.append(accuracy * 100)  # Convert to percentage
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", len(methodologies))
    bars = ax.bar(methodologies, accuracies, color=colors)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Methodology")
    ax.set_title("Tool Selection Accuracy by Methodology")
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved accuracy chart to {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_latency_boxplot(
    details_list: list[pd.DataFrame],
    methodology_names: list[str],
    output_path: Optional[Path] = None
):
    """
    Generate boxplot comparing latency distributions across methodologies.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib and seaborn required for visualization")
        return
    
    # Combine data
    all_data = []
    for details, name in zip(details_list, methodology_names):
        if details is not None and "latency_ms" in details.columns:
            df = details[["latency_ms"]].copy()
            df["methodology"] = name
            all_data.append(df)
    
    if not all_data:
        logger.warning("No latency data available")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Create boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=combined, x="methodology", y="latency_ms", ax=ax, palette="husl")
    
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Methodology")
    ax.set_title("Latency Distribution by Methodology")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved latency boxplot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_accuracy_vs_tools_chart(
    summaries: list[dict],
    output_path: Optional[Path] = None
):
    """
    Generate line chart showing accuracy vs number of tools.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib and seaborn required for visualization")
        return
    
    # Group by methodology
    methodology_data = defaultdict(list)
    
    for summary in summaries:
        methodology = summary.get("methodology", "unknown")
        num_tools = summary.get("experiment_config", {}).get("num_tools", 0)
        accuracy = summary.get("accuracy", 0.0)
        methodology_data[methodology].append((num_tools, accuracy * 100))
    
    # Sort by num_tools within each methodology
    for methodology in methodology_data:
        methodology_data[methodology].sort(key=lambda x: x[0])
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", len(methodology_data))
    
    for (methodology, data), color in zip(methodology_data.items(), colors):
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        ax.plot(x, y, marker='o', label=methodology, color=color, linewidth=2)
    
    ax.set_xlabel("Number of Tools")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Tool Selection Accuracy vs. Tool Set Size")
    ax.legend(loc='lower left')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved accuracy vs tools chart to {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_confidence_histogram(
    details_df: pd.DataFrame,
    methodology_name: str = "confidence",
    output_path: Optional[Path] = None
):
    """
    Generate histogram of confidence scores for confidence-based methodology.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib and seaborn required for visualization")
        return
    
    if "confidence_score" not in details_df.columns:
        logger.warning("No confidence_score column found")
        return
    
    scores = details_df["confidence_score"].dropna()
    if len(scores) == 0:
        logger.warning("No confidence scores available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(scores, bins=20, kde=True, ax=ax, color="steelblue")
    
    ax.axvline(scores.mean(), color='red', linestyle='--', label=f'Mean: {scores.mean():.3f}')
    ax.axvline(scores.median(), color='green', linestyle='--', label=f'Median: {scores.median():.3f}')
    
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence Score Distribution ({methodology_name})")
    ax.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confidence histogram to {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_fallback_breakdown(
    details_df: pd.DataFrame,
    output_path: Optional[Path] = None
):
    """
    Generate pie chart showing fallback method usage distribution.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for visualization")
        return
    
    if "fallback_method_used" not in details_df.columns:
        logger.warning("No fallback_method_used column found")
        return
    
    method_counts = details_df["fallback_method_used"].value_counts()
    if len(method_counts) == 0:
        logger.warning("No fallback data available")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(method_counts)))
    
    wedges, texts, autotexts = ax.pie(
        method_counts.values,
        labels=method_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    
    ax.set_title("Fallback Method Usage Distribution")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved fallback breakdown to {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_confusion_heatmap(
    confusion_df: pd.DataFrame,
    output_path: Optional[Path] = None
):
    """
    Generate heatmap visualization of confusion matrix.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib and seaborn required for visualization")
        return
    
    if confusion_df.empty:
        logger.warning("Empty confusion matrix")
        return
    
    # Normalize by row (expected category)
    row_sums = confusion_df.sum(axis=1)
    normalized = confusion_df.div(row_sums, axis=0).fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Expected Category")
    ax.set_title("Category Confusion Matrix (Normalized)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion heatmap to {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# Comparison Report Generation
# =============================================================================

def generate_comparison_table(summaries: list[dict]) -> pd.DataFrame:
    """
    Generate comparison table across all experiments.
    """
    rows = []
    for summary in summaries:
        config = summary.get("experiment_config", {})
        row = {
            "Experiment": config.get("name", "unknown"),
            "Methodology": summary.get("methodology", "unknown"),
            "Num Tools": config.get("num_tools", 0),
            "Accuracy": f"{summary.get('accuracy', 0) * 100:.1f}%",
            "Avg Latency (ms)": f"{summary.get('avg_latency_ms', 0):.1f}",
            "Call Rate": f"{summary.get('call_rate', 0) * 100:.1f}%",
            "Total Tests": summary.get("total_tests", 0),
            "Errors": summary.get("errors", 0),
        }
        
        # Add methodology-specific metrics
        methodology = summary.get("methodology", "")
        if methodology == "clustering" or methodology == "hybrid":
            row["Category Accuracy"] = f"{summary.get('category_selection_accuracy', 0) * 100:.1f}%"
        if methodology == "adaptive_rag":
            k_stats = summary.get("adaptive_k_stats", {})
            row["Avg K"] = f"{k_stats.get('avg_k', 0):.1f}"
        if methodology == "confidence":
            row["Fallback Rate"] = f"{summary.get('fallback_rate', 0) * 100:.1f}%"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_html_report(
    summaries: list[dict],
    details_list: list[pd.DataFrame],
    output_path: Path,
    figures_dir: Optional[Path] = None
):
    """
    Generate comprehensive HTML report.
    """
    comparison_table = generate_comparison_table(summaries)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Tool Calling Experiment Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .card {{ background: #f9f9f9; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .figure {{ margin: 20px 0; text-align: center; }}
        .figure img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>Tool Calling Experiment Analysis Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Executive Summary</h2>
    <div class="card">
        <p>Total experiments analyzed: <span class="metric">{len(summaries)}</span></p>
        <p>Methodologies tested: <span class="metric">{len(set(s.get('methodology', '') for s in summaries))}</span></p>
    </div>
    
    <h2>Comparison Table</h2>
    {comparison_table.to_html(index=False, classes='comparison')}
    
    <h2>Key Findings</h2>
    <div class="card">
        <ul>
"""
    
    # Add key findings
    if summaries:
        best_accuracy = max(summaries, key=lambda s: s.get("accuracy", 0))
        fastest = min(summaries, key=lambda s: s.get("avg_latency_ms", float('inf')))
        
        html += f"""
            <li><strong>Highest Accuracy:</strong> {best_accuracy.get('methodology')} with {best_accuracy.get('accuracy', 0) * 100:.1f}%</li>
            <li><strong>Fastest:</strong> {fastest.get('methodology')} with {fastest.get('avg_latency_ms', 0):.1f}ms average latency</li>
"""
    
    html += """
        </ul>
    </div>
"""
    
    # Add figures if available
    if figures_dir and figures_dir.exists():
        html += "<h2>Visualizations</h2>"
        for fig_path in sorted(figures_dir.glob("*.png")):
            rel_path = fig_path.relative_to(output_path.parent)
            html += f"""
    <div class="figure">
        <img src="{rel_path}" alt="{fig_path.stem}">
        <p>{fig_path.stem.replace('_', ' ').title()}</p>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Generated HTML report: {output_path}")


# =============================================================================
# CLI Commands
# =============================================================================

@app.command()
def compare(
    results_dir: Path = typer.Option(
        Path("experiments/results"),
        "--results-dir", "-d",
        help="Directory containing experiment results"
    ),
    experiments: Optional[list[str]] = typer.Option(
        None,
        "--experiments", "-e",
        help="Specific experiment summary files to compare"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for comparison results (JSON)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """
    Compare multiple experiment results with statistical analysis.
    """
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")
    
    if experiments:
        summaries = [load_experiment_summary(Path(e)) for e in experiments]
        details = []
        for exp in experiments:
            detail_path = Path(exp).with_name(
                Path(exp).name.replace("_summary.json", "_details.csv")
            )
            details.append(load_experiment_details(detail_path) if detail_path.exists() else None)
    else:
        summaries, details = load_all_experiments(results_dir)
    
    if len(summaries) == 0:
        logger.error("No experiments found")
        raise typer.Exit(1)
    
    logger.info(f"Loaded {len(summaries)} experiments")
    
    # Generate comparison table
    comparison_table = generate_comparison_table(summaries)
    print("\n" + "=" * 80)
    print("METHODOLOGY COMPARISON")
    print("=" * 80)
    print(comparison_table.to_string(index=False))
    print()
    
    # Statistical comparisons between pairs
    if len(summaries) >= 2 and all(d is not None for d in details):
        print("\n" + "=" * 80)
        print("STATISTICAL COMPARISONS")
        print("=" * 80)
        
        comparisons = []
        for i in range(len(summaries)):
            for j in range(i + 1, len(summaries)):
                if details[i] is not None and details[j] is not None:
                    comparison = compare_methodologies_statistical(
                        details[i], details[j],
                        summaries[i].get("methodology", "A"),
                        summaries[j].get("methodology", "B")
                    )
                    comparisons.append(comparison)
                    
                    print(f"\n{comparison['methodology_a']} vs {comparison['methodology_b']}:")
                    print(f"  Accuracy: {comparison['accuracy_a']:.3f} vs {comparison['accuracy_b']:.3f}")
                    print(f"  Difference: {comparison['difference']:+.3f}")
                    print(f"  T-test p-value: {comparison['t_test']['p_value']:.4f}")
                    print(f"  Effect size (Cohen's d): {comparison['cohens_d']:.3f} ({comparison['effect_size_interpretation']})")
                    if comparison['t_test'].get('significant_0.05'):
                        print(f"  *** Statistically significant at p<0.05 ***")
    
    # Save results
    if output:
        results = {
            "comparison_table": comparison_table.to_dict(orient="records"),
            "timestamp": datetime.now().isoformat(),
        }
        if 'comparisons' in dir():
            results["statistical_comparisons"] = comparisons
        
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved comparison results to {output}")


@app.command()
def visualize(
    results_dir: Path = typer.Option(
        Path("experiments/results"),
        "--results-dir", "-d",
        help="Directory containing experiment results"
    ),
    output_dir: Path = typer.Option(
        Path("reports/figures"),
        "--output-dir", "-o",
        help="Output directory for figures"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """
    Generate visualization charts from experiment results.
    """
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")
    
    summaries, details = load_all_experiments(results_dir)
    
    if len(summaries) == 0:
        logger.error("No experiments found")
        raise typer.Exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating visualizations for {len(summaries)} experiments")
    
    # Generate all charts
    generate_accuracy_comparison_chart(summaries, output_dir / "accuracy_comparison.png")
    
    methodology_names = [s.get("methodology", "unknown") for s in summaries]
    generate_latency_boxplot(details, methodology_names, output_dir / "latency_distribution.png")
    
    generate_accuracy_vs_tools_chart(summaries, output_dir / "accuracy_vs_tools.png")
    
    # Generate methodology-specific charts
    for i, (summary, detail) in enumerate(zip(summaries, details)):
        if detail is None:
            continue
        
        methodology = summary.get("methodology", "unknown")
        
        if methodology == "confidence" and "confidence_score" in detail.columns:
            generate_confidence_histogram(
                detail, methodology,
                output_dir / f"confidence_histogram_{i}.png"
            )
            generate_fallback_breakdown(
                detail,
                output_dir / f"fallback_breakdown_{i}.png"
            )
        
        if methodology in ["clustering", "hybrid"] and "final_category" in detail.columns:
            confusion = build_category_confusion_matrix(detail)
            if not confusion.empty:
                generate_confusion_heatmap(
                    confusion,
                    output_dir / f"confusion_matrix_{methodology}_{i}.png"
                )
    
    logger.info(f"All visualizations saved to {output_dir}")


@app.command()
def confusion(
    details_csv: Path = typer.Argument(..., help="Details CSV file to analyze"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output file for confusion matrix analysis (JSON)"
    ),
    visualize_matrix: bool = typer.Option(
        True, "--visualize/--no-visualize",
        help="Generate heatmap visualization"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """
    Generate and analyze category confusion matrix.
    """
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")
    
    details = load_experiment_details(details_csv)
    confusion_matrix = build_category_confusion_matrix(details)
    
    if confusion_matrix.empty:
        logger.error("Could not build confusion matrix (no category data)")
        raise typer.Exit(1)
    
    analysis = analyze_confusion_matrix(confusion_matrix)
    
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 80)
    print(f"\nOverall Category Accuracy: {analysis['overall_accuracy']:.2%}")
    print(f"Total Samples: {analysis['total_samples']}")
    
    print("\nPer-Category Accuracy:")
    for cat, acc in sorted(analysis['category_accuracy'].items(), key=lambda x: x[1]):
        print(f"  {cat}: {acc:.2%}")
    
    print("\nMost Confused Pairs (Expected → Selected):")
    for pair in analysis['most_confused_pairs'][:5]:
        print(f"  {pair['expected']} → {pair['selected']}: {pair['count']} times")
    
    if output:
        with open(output, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Saved analysis to {output}")
    
    if visualize_matrix:
        fig_path = details_csv.with_name(details_csv.stem + "_confusion.png")
        generate_confusion_heatmap(confusion_matrix, fig_path)


@app.command()
def report(
    results_dir: Path = typer.Option(
        Path("experiments/results"),
        "--results-dir", "-d",
        help="Directory containing experiment results"
    ),
    output: Path = typer.Option(
        Path("reports/analysis_report.html"),
        "--output", "-o",
        help="Output HTML report file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v")
):
    """
    Generate comprehensive HTML analysis report.
    """
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")
    
    summaries, details = load_all_experiments(results_dir)
    
    if len(summaries) == 0:
        logger.error("No experiments found")
        raise typer.Exit(1)
    
    # Generate figures first
    figures_dir = output.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating visualizations...")
    generate_accuracy_comparison_chart(summaries, figures_dir / "accuracy_comparison.png")
    
    methodology_names = [s.get("methodology", "unknown") for s in summaries]
    generate_latency_boxplot(details, methodology_names, figures_dir / "latency_distribution.png")
    
    logger.info("Generating HTML report...")
    generate_html_report(summaries, details, output, figures_dir)
    
    print(f"\nReport generated: {output}")


if __name__ == "__main__":
    app()
