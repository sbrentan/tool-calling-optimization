# Tool Calling Optimization

A testing framework for measuring and analyzing LLM tool-calling accuracy across different retrieval methodologies, tool set sizes, and documentation verbosity levels. This project was developed for the course "Designing Large Scale AI Systems" of University of Trento. The course project aims to evaluate and compare different approaches for helping LLMs select the correct tool from large tool catalogs.

## Overview

When LLMs need to call tools/functions, their accuracy can degrade as the number of available tools increases. This project explores **different methodologies** to optimize tool calling:

| Methodology | Description |
|-------------|-------------|
| **MCP** (Baseline) | All tools passed directly to the LLM context |
| **Clustering** | Two-step selection: first select category, then tool within category |
| **RAG** | Semantic retrieval of top-K most relevant tools using embeddings |
| **Adaptive RAG** | Dynamic K selection based on similarity score distribution |
| **Hybrid** | RAG-based category retrieval + LLM tool selection |

## Quick Start

### 1. Setup

```bash
# Clone and enter the repository
git clone <repository-url>
cd tool-calling-optimization

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your API keys
```

**Disclaimer:** Only the Cerebras API key and local Ollama models have been tested. Gemini and OpenAI may have some issues.

### 3. Run Your First Experiment

**Option A: Using Ollama (Local, Free)**

```bash
# Install Ollama from https://ollama.ai
# Pull a model
ollama pull llama3.2:3b

# Run experiment
python scripts/run_experiment.py run \
    --model llama3.2:3b \
    --num-tools 10 \
    --methodology mcp \
    --name my_first_test
```

**Option B: Using Cerebras API (Cloud, Free Tier)**

```bash
# Get free API key from https://cloud.cerebras.ai/
# Add to .env: CEREBRAS_API_KEY=your_key_here

python scripts/run_experiment.py run \
    --model llama-3.3-70b \
    --num-tools 10 \
    --methodology rag \
    --name my_cloud_test
```

Results are saved to `experiments/results/`.

---

## Supported Providers

| Provider | Models | API Key Required |
|----------|--------|------------------|
| **Ollama** (Local) | Any model installed via `ollama pull` (e.g., `llama3.2:3b`, `qwen2:7b`) | No (runs locally) |
| **Cerebras** | `llama-3.3-70b`, `llama3.1-8b`, `qwen-3-32b` | Yes (free tier: 1M tokens/day) |
| **Gemini** | `gemini-2.0-flash`, `gemini-1.5-pro` | Yes |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo` | Yes (paid) |

**Note:** Provider is auto-detected from model name. Models with `:` format (e.g., `llama3.2:3b`) use Ollama. Use `--provider` to override.

Get API keys:
- **Cerebras** (recommended, free): https://cloud.cerebras.ai/
- **Gemini**: https://aistudio.google.com/
- **OpenAI**: https://platform.openai.com/

---

## Running Experiments

### Single Experiment with `run_experiment.py`

The main script for running individual experiments:

```bash
python scripts/run_experiment.py run [OPTIONS]
```

**Common Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--config`, `-c` | Path to YAML config file | - |
| `--model`, `-m` | Model name (auto-detects provider) | `llama-3.3-70b` |
| `--provider`, `-p` | Force provider: `cerebras`, `gemini`, `openai`, `ollama` | Auto |
| `--num-tools`, `-n` | Number of tools to include | `10` |
| `--doc-length`, `-d` | Tool doc verbosity: `minimal`, `short`, `medium`, `long`, `verbose` | `medium` |
| `--methodology` | Method: `mcp`, `clustering`, `rag`, `adaptive_rag`, `hybrid` | `mcp` |
| `--seed` | Random seed for reproducibility | `42` |
| `--name` | Experiment name (used in output files) | `experiment` |
| `-v`, `--verbose` | Enable debug logging | `false` |

**Examples:**

```bash
# Simple test with RAG methodology
python scripts/run_experiment.py run --model llama3.2:3b --num-tools 50 --methodology rag

# Using a config file
python scripts/run_experiment.py run --config experiments/full_cloud/plan/01_phase1_mcp_10tools_medium.yaml

# Verbose documentation with clustering
python scripts/run_experiment.py run --model llama-3.3-70b --num-tools 100 --doc-length verbose --methodology clustering
```

### Batch Experiments with `run_plan.py`

For running multiple experiment configurations (as defined in a folder of YAML configs):

```bash
python scripts/run_plan.py run --plan-dir <experiment>/plan [OPTIONS]
```

**Key Options:**

| Option | Description |
|--------|-------------|
| `--plan-dir`, `-d` | Folder containing experiment configs (default: `plan`) |
| `--plan-config`, `-p` | YAML file defining multiple runs with different seeds/models |
| `--start-from N` | Resume from experiment number N |
| `--list`, `-l` | List all experiments without running (can also use `list-experiments` command) |
| `--dry-run`, `-n` | Preview what would be executed |
| `--no-confirm`, `-y` | Run without confirmation prompts |
| `--seeds` | Run each config multiple times with different seeds |

**Example: Run all configs in an experiment folder**

```bash
# List what would run
python scripts/run_plan.py list-experiments --plan-dir full_cloud/plan

# Run all with the plan configuration (3 different seeds)
python scripts/run_plan.py run --plan-dir full_cloud/plan --plan-config experiments/full_cloud/plan_runs.yaml

# Run without confirmation prompts
python scripts/run_plan.py run --plan-dir full_cloud/plan --no-confirm
```

### Multi-Key Support for Rate Limiting

When running many experiments, you may hit API rate limits. The system supports multiple API keys that rotate automatically:

```bash
# In .env file
CEREBRAS_API_KEY=key1
CEREBRAS_API_KEY_2=key2
CEREBRAS_API_KEY_3=key3
```

When a rate limit is hit, the system automatically tries the next key.

---

## Project Structure

```
├── scripts/                    # Main entry points
│   ├── run_experiment.py       # Run single experiments
│   ├── run_plan.py             # Run batch experiment plans
│   └── utils/                  # Helper utilities
│       └── generate_limits_report.py  # Generate analysis reports
│
├── src/                        # Core library
│   ├── methodologies/          # Tool selection strategies (MCP, RAG, Clustering, etc.)
│   ├── clients/                # LLM API clients (Cerebras, Gemini, OpenAI, Ollama)
│   ├── tools/                  # Tool generation and test case creation
│   ├── evaluation/             # Accuracy metrics and scoring
│   └── experiments/            # Configuration handling
│
├── tools/                      # Tool definitions (YAML)
│   ├── file_operations.yaml
│   ├── database_operations.yaml
│   ├── email_operations.yaml
│   └── ... (35+ categories, 900+ tools)
│
├── experiments/                # Experiment configurations and results
│   ├── full_cloud/             # Cloud experiments (Cerebras)
│   │   ├── plan/               # YAML configs for each test
│   │   ├── results/            # Raw experiment outputs
│   │   ├── report/             # Generated analysis reports
│   │   ├── plan_runs.yaml      # Multi-run configuration
│   │   └── EXPERIMENT_SUMMARY.md
│   ├── fair_local/             # Local experiments (Ollama)
│   └── ...
│
└── docs/                       # Documentation
    ├── CONFIG_REFERENCE.md     # All configuration parameters
    ├── METHODOLOGY_GUIDE.md    # How each methodology works
    └── PLAN_CONFIGURATION.md   # Multi-run setup guide
```

### Experiment Folder Structure

Each experiment folder (e.g., `experiments/full_cloud/`) contains:

| Folder/File | Description |
|-------------|-------------|
| `plan/` | YAML config files for each test scenario |
| `results/` | Raw JSON/CSV outputs from running experiments |
| `report/` | Generated analysis reports and visualizations |
| `tools/` | (Optional) Custom tool definitions if different from root `tools/` |
| `plan_runs.yaml` | Configuration for multi-seed runs |
| `EXPERIMENT_SUMMARY.md` | Summary of test coverage |

---

## Experiment Types

The `experiments/` folder contains different experiment sets, each with specific characteristics:

### Full vs Fair Experiments

| Type | Description |
|------|-------------|
| **full** | All experiment configs are used. May have uneven coverage across methodologies, tool counts, and verbosity levels. More comprehensive but potentially biased in aggregate statistics. |
| **fair** | A curated subset of configs designed for balanced comparison. Equal representation across methodologies and tool counts for fair statistical analysis, but may have reduced chart coverage. |

### Cloud vs Local Experiments

| Type | Model | Samples per Run | Runs | Notes |
|------|-------|-----------------|------|-------|
| **cloud** | Cerebras `llama-3.3-70b` | 10 samples | 3 (seeds: 42, 123, 456) | Rate-limited by API quotas. Uses multiple API keys for rotation. |
| **local** | Ollama `llama3.2:3b` | All samples | 3 (seeds: 42, 123, 456) or 1 when `full`(seed: ) | No rate limits. Runs all test cases per config. Requires local GPU/CPU. |

The `plan_runs.yaml` file in each experiment folder defines these run configurations.

### xLAM Experiments (Validation Dataset)

The `*_xlam` experiments use a different toolset extracted from the public [xLAM Function Calling 60k dataset](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) for validation purposes:

- Tools are extracted using `scripts/utils/convert_xlam_tools.py`
- Located in `experiments/<experiment>/tools/` folder
- Contains ~985 unique tools with their test prompts
- Each config already has `tools_dir` set to point to the correct tools folder

**Running xLAM experiments:**

```bash
# The configs already include tools_dir, just run normally
python scripts/run_plan.py run \
    --plan-dir fair_cloud_xlam/plan \
    --plan-config experiments/fair_cloud_xlam/plan_runs.yaml
```

**To create your own xLAM toolset:**

```bash
# Download xlam_function_calling_60k.json from HuggingFace
# Then convert to project format
python scripts/utils/convert_xlam_tools.py \
    --input xlam_function_calling_60k.json \
    --output-dir experiments/my_experiment/tools \
    --num-tools 985
```

---

## Plan Runs Configuration

The `plan_runs.yaml` file defines how experiments should be executed multiple times with different parameters for statistical validity.

**Example `plan_runs.yaml` (cloud experiments):**

```yaml
runs:
  - name: run_1
    model: llama-3.3-70b
    seed: 42
    num_samples: 10   # Limited due to API rate limits
    
  - name: run_2
    model: llama-3.3-70b
    seed: 123
    num_samples: 10
    
  - name: run_3
    model: llama-3.3-70b
    seed: 456
    num_samples: 10

defaults:
  num_samples: 10
```

**Example `plan_runs.yaml` (local experiments):**

```yaml
runs:
  - name: run_1
    model: llama3.2:3b
    seed: 42
    num_samples: null  # null = run all samples
    
  - name: run_2
    model: llama3.2:3b
    seed: 123
    num_samples: null
    
  - name: run_3
    model: llama3.2:3b
    seed: 456
    num_samples: null
```

**Usage with `--plan-config`:**

```bash
# Run all configs in fair_local/plan with the multi-run configuration
python scripts/run_plan.py run \
    --plan-dir fair_local/plan \
    --plan-config experiments/fair_local/plan_runs.yaml \
    --no-confirm

# Each config is executed 3 times (once per run in plan_runs.yaml)
# Results are saved with run suffix: experiment_name_run_1.json, experiment_name_run_2.json, etc.
```

The report generator aggregates results across runs to compute mean, std, and confidence intervals.

---

## Configuration

Experiments are configured via YAML files. See [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) for full reference.

**Example config:**

```yaml
name: rag_100tools_test
description: "Test RAG with 100 tools"

# Model settings
model: llama-3.3-70b
temperature: 0.0

# Tool settings  
num_tools: 100
doc_length: medium
prompt_type: concise

# Methodology
methodology: rag
rag_config:
  top_k: 15
  embedding_model: all-MiniLM-L6-v2

# Output
output_dir: experiments/results
```

---

## Output Files

Each experiment generates:

| File | Description |
|------|-------------|
| `{name}_{timestamp}_summary.json` | Aggregated metrics (accuracy, latency, steps, etc.) |
| `{name}_{timestamp}_details.csv` | Per-test-case results for analysis |

**Key Metrics:**

- **Accuracy**: % of correct tool selections
- **Latency**: Average response time (ms)
- **Steps**: For multi-step methodologies, number of LLM calls required
- **Retrieval Recall**: For RAG methods, whether correct tool was in retrieved set

---

## Tool Catalog

The `tools/` folder contains 35+ categories with 900+ tool definitions. See [tools/README.md](tools/README.md) for details.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) | Complete configuration parameter reference |
| [docs/METHODOLOGY_GUIDE.md](docs/METHODOLOGY_GUIDE.md) | Detailed explanation of each methodology |
| [docs/PLAN_CONFIGURATION.md](docs/PLAN_CONFIGURATION.md) | Multi-run experiment setup guide |
| [GRADING.md](GRADING.md) | University grading rubric |

---

## License

This project was developed for educational purposes as part of an AI Design university course.
