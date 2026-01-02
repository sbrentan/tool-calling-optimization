#!/usr/bin/env python
"""
Main experiment runner script.

This script orchestrates tool calling experiments:
1. Loads configuration (from file or CLI args)
2. Generates tools and test cases
3. Runs tests against LLM APIs (Gemini, Cerebras, OpenAI)
4. Evaluates results and saves metrics

Usage:
    # Run with config file
    python scripts/run_experiment.py run --config experiments/configs/baseline.yaml
    
    # Run with CLI options
    python scripts/run_experiment.py run --num-tools 10 --doc-length medium --model llama-3.3-70b
    
    # Run parameter sweep
    python scripts/run_experiment.py sweep
    
    # List available models
    python scripts/run_experiment.py list-models
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import typer
from loguru import logger
from dotenv import load_dotenv

from src.tools.generator import ToolGenerator
from src.clients import create_client, get_available_models, get_available_providers
from src.evaluation.metrics import ToolCallEvaluator
from src.experiments.config import ExperimentConfig
from src.methodologies import create_methodology, MethodologyFactory

# Load environment variables
load_dotenv()

app = typer.Typer(help="Tool Calling Optimization Experiments")


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
    logger.info(f"Methodology: {config.methodology}")
    
    # Initialize components
    generator = ToolGenerator(seed=config.seed)
    
    # Create client using factory (auto-detects provider from model)
    try:
        client = create_client(
            model=config.model,
            provider=config.provider if hasattr(config, 'provider') else None,
            temperature=config.temperature
        )
    except ValueError as e:
        logger.error(f"Failed to create client: {e}")
        raise
    
    # Create methodology
    try:
        methodology = create_methodology(
            name=config.methodology,
            max_steps=config.max_steps,
            allow_backtrack=config.allow_backtrack,
            allow_decline=config.allow_no_tool_call,
        )
        logger.info(f"Using methodology: {methodology.NAME}")
    except ValueError as e:
        logger.error(f"Failed to create methodology: {e}")
        raise
    
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
    
    # Log tool categories distribution
    categories_count = {}
    for tool in tools:
        categories_count[tool.category] = categories_count.get(tool.category, 0) + 1
    logger.debug(f"Tools by category: {categories_count}")
    
    # Generate test cases
    prompt_type = getattr(config, 'prompt_type', 'concise')
    logger.info(f"Using prompt type: {prompt_type}")
    test_cases = generator.generate_test_cases(tools, prompt_type=prompt_type)
    if config.num_test_samples is not None:
        test_cases = test_cases[:config.num_test_samples]
    logger.info(f"Running {len(test_cases)} test cases...")
    
    # Run tests using methodology
    for i, test_case in enumerate(test_cases):
        logger.info(f"")
        logger.info(f"======== Test {i+1}/{len(test_cases)} ========")
        logger.debug(f"Prompt: {test_case.prompt}")
        logger.debug(f"Expected tool: {test_case.expected_tool}")
        logger.debug(f"Expected category: {test_case.category}")
        
        # Use methodology to run the test
        methodology_result = methodology.run_single(
            prompt=test_case.prompt,
            tools=tools,
            client=client,
        )
        
        # Convert to CallResult for evaluation
        result = methodology_result.to_call_result()
        
        # Evaluate
        test_result = evaluator.evaluate_single(test_case, result, methodology_result)
        
        status = "✓" if test_result.tool_correct else "✗"
        extra_info = ""
        if hasattr(methodology_result, 'steps') and len(methodology_result.steps) > 1:
            extra_info = f" (steps: {len(methodology_result.steps)}"
            if methodology_result.backtrack_count > 0:
                extra_info += f", backtracks: {methodology_result.backtrack_count}"
            extra_info += ")"
        
        # Log detailed result info
        logger.debug(f"Result details:")
        logger.debug(f"  Called tool: {result.called_tool}")
        logger.debug(f"  Called args: {result.called_args}")
        logger.debug(f"  Latency: {result.latency_ms:.1f}ms")
        if methodology_result.methodology != "mcp":
            logger.debug(f"  Categories selected: {methodology_result.categories_selected}")
            logger.debug(f"  Final category: {methodology_result.final_category}")
            logger.debug(f"  Steps: {[{'type': s.step_type.value, 'selection': s.selection} for s in methodology_result.steps]}")
        
        logger.info(f"  [{status}] Expected: {test_case.expected_tool}, Got: {result.called_tool}{extra_info}")
    
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
def run(
    config: str = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    num_tools: int = typer.Option(10, "--num-tools", "-n", help="Number of tools"),
    doc_length: str = typer.Option("medium", "--doc-length", "-d", help="Documentation length (minimal/short/medium/long/verbose)"),
    prompt_type: str = typer.Option("concise", "--prompt-type", "-t", help="Prompt type (concise/clear)"),
    model: str = typer.Option("llama-3.3-70b", "--model", "-m", help="Model to use (auto-detects provider)"),
    provider: str = typer.Option(None, "--provider", "-p", help="Provider (gemini/cerebras/openai). Auto-detected if not specified."),
    num_similar: int = typer.Option(0, "--num-similar", "-s", help="Number of similar tools"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    name: str = typer.Option("experiment", "--name", help="Experiment name"),
    methodology: str = typer.Option("mcp", "--methodology", help="Methodology to use (mcp/clustering)"),
    max_steps: int = typer.Option(10, "--max-steps", help="Max steps for multi-step methodologies"),
    allow_backtrack: bool = typer.Option(True, "--allow-backtrack/--no-backtrack", help="Allow backtracking in step-based methodologies"),
    allow_decline: bool = typer.Option(False, "--allow-decline/--no-decline", help="Allow LLM to decline calling any tool"),
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
            prompt_type=prompt_type,
            model=model,
            num_similar_tools=num_similar,
            seed=seed,
            methodology=methodology,
            max_steps=max_steps,
            allow_backtrack=allow_backtrack,
            allow_no_tool_call=allow_decline,
        )
        # Store provider if explicitly specified
        if provider:
            exp_config.provider = provider
    
    # Run experiment
    try:
        run_experiment(exp_config)
    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        raise typer.Exit(1)


@app.command()
def sweep(
    output_dir: str = typer.Option("experiments/results/sweep", help="Output directory"),
    provider: str = typer.Option("cerebras", "--provider", "-p", help="Provider to use for sweep"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Run a parameter sweep across multiple configurations."""
    setup_logging(verbose)
    
    # Get default model for provider
    models_by_provider = get_available_models(provider)
    default_model = models_by_provider[provider][0] if models_by_provider[provider] else "llama-3.3-70b"
    
    # Define sweep parameters
    tool_counts = [5, 10, 15, 20, 25]
    doc_lengths = ["minimal", "short", "medium", "long"]
    
    results = []
    
    for num_tools in tool_counts:
        for doc_length in doc_lengths:
            config = ExperimentConfig(
                name=f"sweep_t{num_tools}_d{doc_length}_{default_model}",
                num_tools=num_tools,
                doc_length=doc_length,
                model=default_model,
                output_dir=output_dir,
                seed=42
            )
            
            try:
                result = run_experiment(config)
                results.append({
                    "num_tools": num_tools,
                    "doc_length": doc_length,
                    "model": default_model,
                    "provider": provider,
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


@app.command("list-models")
def list_models():
    """List all available models by provider."""
    print("\n" + "=" * 60)
    print("AVAILABLE MODELS BY PROVIDER")
    print("=" * 60)
    
    all_models = get_available_models()
    
    for provider, models in all_models.items():
        print(f"\n{provider.upper()}:")
        for model in models:
            print(f"  - {model}")
    
    print("\n" + "=" * 60)
    print("\nRequired Environment Variables:")
    print("  CEREBRAS_API_KEY  - For Cerebras models (free tier: 1M tokens/day)")
    print("  GEMINI_API_KEY    - For Gemini models")
    print("  OPENAI_API_KEY    - For OpenAI models")
    print("=" * 60 + "\n")


@app.command("list-tools")
def list_tools(
    category: str = typer.Option(None, "--category", "-c", help="Filter by category"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed info"),
):
    """List all available tools from YAML definitions."""
    generator = ToolGenerator()
    
    print("\n" + "=" * 60)
    print("AVAILABLE TOOLS FROM YAML DEFINITIONS")
    print("=" * 60)
    
    categories = generator.get_categories()
    total_tools = generator.get_tool_count()
    
    print(f"\nTotal: {total_tools} tools across {len(categories)} categories")
    print(f"Categories: {', '.join(categories)}")
    print()
    
    all_tools = generator.list_all_tools()
    
    # Filter by category if specified
    if category:
        all_tools = [t for t in all_tools if t["category"] == category]
    
    # Group by category for display
    by_category = {}
    for tool in all_tools:
        cat = tool["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(tool)
    
    for cat, tools in sorted(by_category.items()):
        print(f"\n{cat.upper().replace('_', ' ')} ({len(tools)} tools):")
        for tool in tools:
            if verbose:
                tags = ", ".join(tool["tags"][:5])
                multi = " [multi-tool]" if tool["has_multi_tool_tests"] else ""
                print(f"  - {tool['name']} ({tool['param_count']} params) [{tags}]{multi}")
            else:
                print(f"  - {tool['name']}")
    
    print("\n" + "=" * 60 + "\n")


@app.command("list-methodologies")
def list_methodologies():
    """List all available methodologies."""
    print("\n" + "=" * 60)
    print("AVAILABLE METHODOLOGIES")
    print("=" * 60)
    
    available = MethodologyFactory.get_available()
    
    descriptions = {
        "mcp": "Model Context Protocol - all tools passed to LLM at once (baseline)",
        "clustering": "Two-step selection - first select category, then tool within category",
    }
    
    for name in available:
        desc = descriptions.get(name, "No description available")
        print(f"\n  {name}:")
        print(f"    {desc}")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    app()
