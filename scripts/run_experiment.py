#!/usr/bin/env python
"""
Main experiment runner script.

This script orchestrates tool calling experiments:
1. Loads configuration (from file or CLI args)
2. Generates tools and test cases
3. Runs tests against the Gemini API
4. Evaluates results and saves metrics

Usage:
    python scripts/run_experiment.py --config experiments/configs/baseline.yaml
    python scripts/run_experiment.py --num-tools 10 --doc-length medium --model gemini-2.0-flash
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
from dotenv import load_dotenv
from loguru import logger

from src.tools.generator import ToolGenerator
from src.clients.gemini_client import GeminiClient
from src.evaluation.metrics import ToolCallEvaluator
from src.experiments.config import ExperimentConfig

# Load environment variables
load_dotenv()

app = typer.Typer(help="Run tool calling experiments")


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")


def run_experiment(config: ExperimentConfig) -> dict:
    """
    Run a single experiment with the given configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with experiment results
    """
    logger.info(f"Starting experiment: {config.name}")
    logger.info(f"Configuration: {config.num_tools} tools, {config.doc_length} docs, model={config.model}")
    
    # Initialize components
    generator = ToolGenerator(seed=config.seed)
    client = GeminiClient(model=config.model, temperature=config.temperature)
    evaluator = ToolCallEvaluator()
    
    # Generate tools
    logger.info(f"Generating {config.num_tools} tools...")
    tools = generator.generate_tools(
        num_tools=config.num_tools,
        doc_length=config.doc_length,
        include_similar=config.num_similar_tools,
        categories=config.categories
    )
    logger.info(f"Generated {len(tools)} tools across categories")
    
    # Generate test cases
    test_cases = generator.generate_test_cases(tools)
    if config.num_test_samples is not None:
        test_cases = test_cases[:config.num_test_samples]
    logger.info(f"Running {len(test_cases)} test cases...")
    
    # Run tests
    for i, test_case in enumerate(test_cases):
        logger.debug(f"Test {i+1}/{len(test_cases)}: {test_case.prompt[:50]}...")
        
        # Make API call
        result = client.call_with_tools(
            prompt=test_case.prompt,
            tools=tools
        )
        
        # Evaluate
        test_result = evaluator.evaluate_single(test_case, result)
        
        status = "✓" if test_result.tool_correct else "✗"
        logger.info(f"  [{status}] Expected: {test_case.expected_tool}, Got: {result.called_tool}")
    
    # Compute metrics
    metrics = evaluator.compute_metrics(experiment_config=config.to_dict())
    
    # Print summary
    print("\n" + metrics.summary())
    
    # Save results
    save_results(config, metrics)
    
    return metrics.to_dict()


def save_results(config: ExperimentConfig, metrics):
    """Save experiment results to files."""
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{config.name}_{timestamp}"
    
    # Save summary JSON
    summary_path = output_dir / f"{base_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2, default=str)
    logger.info(f"Saved summary to {summary_path}")
    
    # Save detailed results CSV
    df = metrics.to_dataframe()
    csv_path = output_dir / f"{base_name}_details.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved details to {csv_path}")


@app.command()
def main(
    config: str = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    num_tools: int = typer.Option(10, "--num-tools", "-n", help="Number of tools"),
    doc_length: str = typer.Option("medium", "--doc-length", "-d", help="Documentation length"),
    model: str = typer.Option("gemini-2.0-flash", "--model", "-m", help="Gemini model to use"),
    num_similar: int = typer.Option(0, "--num-similar", "-s", help="Number of similar tools"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    name: str = typer.Option("experiment", "--name", help="Experiment name"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Run a tool calling experiment."""
    setup_logging(verbose)
    
    # Load or create config
    if config:
        logger.info(f"Loading config from {config}")
        exp_config = ExperimentConfig.from_yaml(config)
    else:
        exp_config = ExperimentConfig(
            name=name,
            num_tools=num_tools,
            doc_length=doc_length,
            model=model,
            num_similar_tools=num_similar,
            seed=seed
        )
    
    # Validate API key
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY not set. Please set it in .env file or environment.")
        raise typer.Exit(1)
    
    # Run experiment
    try:
        run_experiment(exp_config)
    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        raise typer.Exit(1)


@app.command()
def sweep(
    output_dir: str = typer.Option("experiments/results/sweep", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Run a parameter sweep across multiple configurations."""
    setup_logging(verbose)
    
    # Define sweep parameters
    tool_counts = [5, 10, 15, 20, 25]
    doc_lengths = ["minimal", "short", "medium", "long"]
    models = ["gemini-2.0-flash"]
    
    results = []
    
    for num_tools in tool_counts:
        for doc_length in doc_lengths:
            for model in models:
                config = ExperimentConfig(
                    name=f"sweep_t{num_tools}_d{doc_length}_{model}",
                    num_tools=num_tools,
                    doc_length=doc_length,
                    model=model,
                    output_dir=output_dir,
                    seed=42
                )
                
                try:
                    result = run_experiment(config)
                    results.append({
                        "num_tools": num_tools,
                        "doc_length": doc_length,
                        "model": model,
                        "accuracy": result["accuracy"],
                        "avg_latency_ms": result["avg_latency_ms"]
                    })
                except Exception as e:
                    logger.error(f"Failed: {e}")
                    continue
    
    # Save sweep summary
    sweep_path = Path(output_dir) / "sweep_summary.json"
    with open(sweep_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Sweep complete. Results saved to {sweep_path}")


if __name__ == "__main__":
    app()
