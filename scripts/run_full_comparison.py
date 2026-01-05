#!/usr/bin/env python
"""
Run full comparison across all methodologies.

This script automates running experiments for all 6 methodologies
on the same test configuration for fair comparison.

Usage:
    python scripts/run_full_comparison.py
    python scripts/run_full_comparison.py --num-tools 100 --num-samples 100
    python scripts/run_full_comparison.py --methodologies mcp,rag,adaptive_rag
"""
import sys
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import typer
import yaml
from loguru import logger

app = typer.Typer(help="Run full methodology comparison experiments")

# All available methodologies
ALL_METHODOLOGIES = ["mcp", "clustering", "rag", "hybrid", "adaptive_rag", "confidence"]


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr, 
        level=level, 
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )


def generate_config(
    methodology: str,
    experiment_name: str,
    num_tools: int,
    num_samples: int,
    model: str,
    provider: str,
    include_multi_tool: bool,
    include_no_tool: bool,
    output_dir: Path,
) -> dict:
    """Generate experiment configuration dictionary."""
    config = {
        "name": experiment_name,
        "description": f"{methodology} methodology comparison experiment",
        "num_tools": num_tools,
        "doc_length": "medium",
        "prompt_type": "clear",
        "num_similar_tools": 0,
        "categories": None,
        "model": model,
        "provider": provider,
        "temperature": 0.0,
        "num_test_samples": num_samples,
        "seed": 42,
        "include_multi_tool": include_multi_tool,
        "include_no_tool": include_no_tool,
        "output_dir": str(output_dir),
        "save_raw_responses": False,
        "methodology": methodology,
        "max_steps": 10,
        "allow_backtrack": True,
        "allow_no_tool_call": True,
    }
    
    # Add methodology-specific configs
    if methodology == "rag":
        config["rag_config"] = {
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 10,
            "similarity_threshold": 0.0,
            "cache_embeddings": True,
            "include_params_in_embedding": False,
        }
    elif methodology == "hybrid":
        config["hybrid_config"] = {
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k_categories": 3,
            "cache_embeddings": True,
        }
    elif methodology == "adaptive_rag":
        config["adaptive_rag_config"] = {
            "embedding_model": "all-MiniLM-L6-v2",
            "min_k": 3,
            "max_k": 20,
            "similarity_drop_threshold": 0.1,
            "min_similarity": 0.3,
            "cache_embeddings": True,
            "include_params_in_embedding": False,
        }
    elif methodology == "confidence":
        config["confidence_config"] = {
            "rag_confidence_threshold": 0.7,
            "clustering_confidence_threshold": 0.6,
            "rag_config": {
                "embedding_model": "all-MiniLM-L6-v2",
                "top_k": 10,
                "similarity_threshold": 0.0,
                "cache_embeddings": True,
            },
            "clustering_config": {
                "allow_backtrack": True,
            },
        }
    
    return config


@app.command()
def run(
    methodologies: str = typer.Option(
        ",".join(ALL_METHODOLOGIES),
        "--methodologies", "-m",
        help="Comma-separated list of methodologies to run"
    ),
    num_tools: int = typer.Option(50, "--num-tools", "-t", help="Number of tools to generate"),
    num_samples: int = typer.Option(50, "--num-samples", "-n", help="Number of test samples"),
    model: str = typer.Option("llama-3.3-70b", "--model", help="Model to use"),
    provider: str = typer.Option("cerebras", "--provider", help="Provider to use"),
    include_multi_tool: bool = typer.Option(True, "--multi-tool/--no-multi-tool", help="Include multi-tool tests"),
    include_no_tool: bool = typer.Option(True, "--no-tool/--no-no-tool", help="Include no-tool tests"),
    output_dir: Path = typer.Option(
        Path("experiments/results"),
        "--output-dir", "-o",
        help="Output directory for results"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without running"),
):
    """
    Run experiments for all specified methodologies with the same configuration.
    """
    setup_logging(verbose)
    
    methodology_list = [m.strip() for m in methodologies.split(",")]
    
    # Validate methodologies
    for m in methodology_list:
        if m not in ALL_METHODOLOGIES:
            logger.error(f"Unknown methodology: {m}")
            logger.info(f"Available: {ALL_METHODOLOGIES}")
            raise typer.Exit(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Running comparison experiment")
    logger.info(f"  Methodologies: {methodology_list}")
    logger.info(f"  Num tools: {num_tools}")
    logger.info(f"  Num samples: {num_samples}")
    logger.info(f"  Model: {model}")
    logger.info(f"  Provider: {provider}")
    logger.info("")
    
    results = []
    
    for i, methodology in enumerate(methodology_list, 1):
        experiment_name = f"comparison_{methodology}_{num_tools}tools_{timestamp}"
        
        logger.info(f"[{i}/{len(methodology_list)}] Running {methodology}...")
        
        # Generate config
        config = generate_config(
            methodology=methodology,
            experiment_name=experiment_name,
            num_tools=num_tools,
            num_samples=num_samples,
            model=model,
            provider=provider,
            include_multi_tool=include_multi_tool,
            include_no_tool=include_no_tool,
            output_dir=output_dir,
        )
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would run {methodology} with config:")
            logger.info(f"    num_tools: {num_tools}, num_samples: {num_samples}")
            continue
        
        # Write temporary config file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False, dir=project_root
        ) as f:
            yaml.dump(config, f, default_flow_style=False)
            config_path = f.name
        
        try:
            # Build command using config file
            cmd = [
                sys.executable, "scripts/run_experiment.py", "run",
                "--config", config_path,
            ]
            
            if verbose:
                cmd.append("--verbose")
            
            logger.debug(f"Command: {' '.join(cmd)}")
            
            # Run experiment
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=not verbose,
                text=True,
            )
            
            if result.returncode == 0:
                logger.info(f"  ✓ {methodology} completed successfully")
                results.append((methodology, "success", experiment_name))
            else:
                logger.error(f"  ✗ {methodology} failed with code {result.returncode}")
                if result.stderr:
                    logger.error(f"    Error: {result.stderr[:500]}")
                results.append((methodology, "failed", experiment_name))
        
        except Exception as e:
            logger.error(f"  ✗ {methodology} failed with exception: {e}")
            results.append((methodology, "error", str(e)))
        
        finally:
            # Clean up temp config file
            try:
                Path(config_path).unlink()
            except Exception:
                pass
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPARISON EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    
    successful = [r for r in results if r[1] == "success"]
    failed = [r for r in results if r[1] != "success"]
    
    logger.info(f"Successful: {len(successful)}/{len(results)}")
    for methodology, status, name in successful:
        logger.info(f"  ✓ {methodology}: {name}")
    
    if failed:
        logger.info(f"Failed: {len(failed)}/{len(results)}")
        for methodology, status, info in failed:
            logger.info(f"  ✗ {methodology}: {info}")
    
    if successful and not dry_run:
        logger.info("")
        logger.info("To analyze results:")
        logger.info(f"  python scripts/analyze_results.py compare --results-dir {output_dir}")
        logger.info(f"  python scripts/analyze_results.py visualize --results-dir {output_dir}")
        logger.info(f"  python scripts/analyze_results.py report --results-dir {output_dir}")


@app.command()
def scaling(
    methodology: str = typer.Option(
        "rag",
        "--methodology", "-m",
        help="Methodology to test"
    ),
    tool_counts: str = typer.Option(
        "10,25,50,100,150,200",
        "--tool-counts", "-t",
        help="Comma-separated list of tool counts to test"
    ),
    num_samples: int = typer.Option(30, "--num-samples", "-n", help="Number of test samples per run"),
    model: str = typer.Option("llama-3.3-70b", "--model", help="Model to use"),
    provider: str = typer.Option("cerebras", "--provider", help="Provider to use"),
    output_dir: Path = typer.Option(
        Path("experiments/results"),
        "--output-dir", "-o",
        help="Output directory for results"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without running"),
):
    """
    Run scaling experiment: test one methodology at different tool counts.
    
    This generates data for accuracy vs. tool count analysis.
    """
    setup_logging(verbose)
    
    if methodology not in ALL_METHODOLOGIES:
        logger.error(f"Unknown methodology: {methodology}")
        logger.info(f"Available: {ALL_METHODOLOGIES}")
        raise typer.Exit(1)
    
    counts = [int(c.strip()) for c in tool_counts.split(",")]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Running scaling experiment for {methodology}")
    logger.info(f"  Tool counts: {counts}")
    logger.info(f"  Samples per run: {num_samples}")
    logger.info("")
    
    results = []
    
    for i, num_tools in enumerate(counts, 1):
        experiment_name = f"scaling_{methodology}_{num_tools}tools_{timestamp}"
        
        logger.info(f"[{i}/{len(counts)}] Testing with {num_tools} tools...")
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would run with {num_tools} tools")
            continue
        
        # Generate config
        config = generate_config(
            methodology=methodology,
            experiment_name=experiment_name,
            num_tools=num_tools,
            num_samples=num_samples,
            model=model,
            provider=provider,
            include_multi_tool=False,
            include_no_tool=False,
            output_dir=output_dir,
        )
        
        # Write temporary config file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False, dir=project_root
        ) as f:
            yaml.dump(config, f, default_flow_style=False)
            config_path = f.name
        
        try:
            cmd = [
                sys.executable, "scripts/run_experiment.py", "run",
                "--config", config_path,
            ]
            
            if verbose:
                cmd.append("--verbose")
            
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=not verbose,
                text=True,
            )
            
            if result.returncode == 0:
                logger.info(f"  ✓ {num_tools} tools completed")
                results.append((num_tools, "success"))
            else:
                logger.error(f"  ✗ {num_tools} tools failed")
                results.append((num_tools, "failed"))
        
        except Exception as e:
            logger.error(f"  ✗ {num_tools} tools failed: {e}")
            results.append((num_tools, "error"))
        
        finally:
            # Clean up temp config file
            try:
                Path(config_path).unlink()
            except Exception:
                pass
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SCALING EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Methodology: {methodology}")
    
    successful = [r for r in results if r[1] == "success"]
    logger.info(f"Successful: {len(successful)}/{len(results)}")
    
    if successful and not dry_run:
        logger.info("")
        logger.info("To visualize scaling results:")
        logger.info(f"  python scripts/analyze_results.py visualize --results-dir {output_dir}")


if __name__ == "__main__":
    app()
