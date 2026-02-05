# Plan Configuration Guide

This guide explains how to use the multi-run plan configuration system to run experiments with different seeds, models, and sample sizes for statistical validity.

## Overview

The plan executor (`scripts/run_plan.py`) supports two modes:

1. **Single-run mode** (default): Run each experiment config once using environment variables or defaults
2. **Multi-run mode**: Run each experiment config multiple times with different configurations (seeds, models, samples)

Multi-run mode is useful for:
- Running experiments with different random seeds for statistical validity
- Comparing different models across the same experiment configurations
- Running subsets of samples for quick validation

## Quick Start

### Single-run mode (default)

```bash
# Run all experiments once with default settings
python scripts/run_plan.py run

# Override model/seed via environment variables
EXPERIMENT_MODEL=gpt-4o EXPERIMENT_SEED=42 python scripts/run_plan.py run
```

### Multi-run mode

```bash
# Run with 3 different seeds (quick CLI)
python scripts/run_plan.py run --seeds "42,123,456"

# Run with multiple models
python scripts/run_plan.py run --models "llama-4-scout-17b-16e-instruct,qwen-3-32b"

# Use a YAML plan config file
python scripts/run_plan.py run --plan-config experiments/plan_runs.yaml
```

## Plan Configuration YAML

For complex multi-run configurations, use a YAML file:

```yaml
# experiments/plan_runs.yaml
defaults:
  model: "llama-4-scout-17b-16e-instruct"
  seed: 42
  num_samples: null  # null = use all samples

runs:
  - name: "run_1"
    seed: 42
  - name: "run_2" 
    seed: 123
  - name: "run_3"
    seed: 456
```

### Configuration Fields

| Field | Type | Description |
|-------|------|-------------|
| `defaults` | object | Default values for runs that don't specify them |
| `defaults.model` | string | Default LLM model to use |
| `defaults.seed` | int | Default random seed |
| `defaults.num_samples` | int/null | Default number of test samples (null = all) |
| `runs` | list | List of run configurations |
| `runs[].name` | string | Unique name for this run (used in result filenames) |
| `runs[].model` | string | Override model for this run |
| `runs[].seed` | int | Override seed for this run |
| `runs[].num_samples` | int/null | Override sample count for this run |

### Example: Multiple Seeds

```yaml
defaults:
  model: "llama-4-scout-17b-16e-instruct"
  
runs:
  - name: "seed_42"
    seed: 42
  - name: "seed_123"
    seed: 123
  - name: "seed_456"
    seed: 456
```

### Example: Model Comparison

```yaml
defaults:
  seed: 42

runs:
  - name: "llama4"
    model: "llama-4-scout-17b-16e-instruct"
  - name: "qwen3"
    model: "qwen-3-32b"
  - name: "deepseek"
    model: "deepseek-r1-0528"
```

### Example: Quick Validation with Fewer Samples

```yaml
defaults:
  model: "llama-4-scout-17b-16e-instruct"
  num_samples: 10  # Only run 10 samples for quick testing

runs:
  - name: "quick_test"
    seed: 42
```

## CLI Options

### Multi-run Options

| Option | Description |
|--------|-------------|
| `--plan-config` | Path to a YAML plan configuration file |
| `--seeds` | Comma-separated list of seeds (e.g., "42,123,456") |
| `--models` | Comma-separated list of models |
| `--num-samples` | Number of samples per run (overrides config) |
| `--run-prefix` | Prefix for run names (see below) |

### Combining Multiple Plan Configs

When running the same experiments with multiple plan configs (e.g., different model sets), use `--run-prefix` to distinguish results:

```bash
# First plan config run
python scripts/run_plan.py run --plan-config configs_a.yaml --run-prefix "a"
# Results: experiment_a_run_1, experiment_a_run_2, ...

# Second plan config run  
python scripts/run_plan.py run --plan-config configs_b.yaml --run-prefix "b"
# Results: experiment_b_run_1, experiment_b_run_2, ...
```

This allows you to aggregate all results together later.

## Result Naming Convention

Results are saved with a suffix indicating the run:

| Mode | Result Filename Pattern |
|------|------------------------|
| Single-run | `experiment_name.json` |
| Multi-run | `experiment_name_run_1.json`, `experiment_name_run_2.json`, ... |
| With prefix | `experiment_name_a_run_1.json`, `experiment_name_b_run_1.json`, ... |

## Aggregating Results

After running experiments, use the report generator to aggregate and visualize results:

```bash
python scripts/utils/generate_limits_report.py --results-dir experiments/<experiment>/results --output-dir experiments/<experiment>/report
```

The aggregation system:

1. **Groups by base experiment name**: Strips the `_run_*` suffix to identify related runs
2. **Calculates statistics**: Mean, standard deviation, min, max across runs
3. **Includes single-run results**: Existing results without `_run_` suffix are included in aggregation

### Aggregation Example

Given results:
- `mcp_50tools_medium.json` (existing single run)
- `mcp_50tools_medium_run_1.json` (new multi-run)
- `mcp_50tools_medium_run_2.json` (new multi-run)

All three are grouped under `mcp_50tools_medium` and aggregated together.

### Aggregation Output

The aggregated report includes:
- **Per-experiment statistics**: Mean accuracy, std dev, number of runs
- **Confidence intervals**: For experiments with multiple runs
- **Comparison tables**: Side-by-side comparison of methodologies

## API Key Rotation

The system automatically rotates API keys when rate limits are hit:

1. Set multiple API keys via environment variables:
   ```bash
   export CEREBRAS_API_KEY="key1"
   export CEREBRAS_API_KEY_2="key2"
   export CEREBRAS_API_KEY_3="key3"
   ```

2. On rate limit, the system:
   - Tries the next available key
   - Continues through all keys before prompting user
   - Resets rotation when a key succeeds

## Best Practices

### For Statistical Validity

1. **Use at least 3-5 runs** with different seeds
2. **Use the same model** across runs for fair comparison
3. **Include all samples** (don't use `num_samples` limit) for final results

### For Quick Iteration

1. **Use fewer samples** (`num_samples: 10`) during development
2. **Use single-run mode** for debugging
3. **Use `--dry-run`** to preview what will run

### For Large-Scale Experiments

1. **Use `--no-confirm`** to skip prompts
2. **Use `--start-from`** to resume interrupted runs
3. **Save batch summaries** for progress tracking

## Troubleshooting

### "No experiments found"

Make sure experiment configs exist in the plan directory:
```bash
ls experiments/<experiment>/plan/
```

### "Rate limit exceeded"

1. The system will automatically try rotating API keys
2. If all keys are exhausted, you'll be prompted to wait or abort
3. Add more API keys to environment variables

### "Results not aggregating correctly"

1. Check that base experiment names match
2. Verify runs have the `_run_*` suffix pattern
3. Use `--run-prefix` when running multiple plan configs

## Example Workflow

```bash
# 1. Preview what will run
python scripts/run_plan.py list-experiments --plan-dir full_cloud/plan

# 2. Do a quick test run with fewer samples
python scripts/run_plan.py run --plan-dir full_cloud/plan --num-samples 5 --no-confirm

# 3. Run full experiments with multiple seeds
python scripts/run_plan.py run --plan-dir full_cloud/plan --plan-config experiments/full_cloud/plan_runs.yaml --no-confirm

# 4. Generate analysis report
python scripts/utils/generate_limits_report.py --results-dir experiments/full_cloud/results --output-dir experiments/full_cloud/report
```
