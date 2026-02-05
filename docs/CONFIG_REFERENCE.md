# Configuration Reference

This document provides a comprehensive reference for all configurable parameters in the Tool-Calling Optimization Framework.

---

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Base Experiment Settings](#base-experiment-settings)
3. [Tool Configuration](#tool-configuration)
4. [Model Configuration](#model-configuration)
5. [Test Configuration](#test-configuration)
6. [Methodology Configuration](#methodology-configuration)
7. [RAG Configuration](#rag-configuration)
8. [Adaptive RAG Configuration](#adaptive-rag-configuration)
9. [Hybrid Configuration](#hybrid-configuration)
10. [Confidence Configuration](#confidence-configuration)
11. [Output Configuration](#output-configuration)
12. [Example Configurations](#example-configurations)

---

## Configuration Overview

Configurations can be specified via:
- **YAML files** in experiment configs
- **Command-line arguments** when running experiments
- **Python code** using the `ExperimentConfig` dataclass

YAML is recommended for reproducibility and documentation.

### Basic YAML Structure

```yaml
name: my_experiment
description: "Description of the experiment"
methodology: rag

# Tool settings
num_tools: 50
doc_length: medium

# Model settings
model: llama-3.3-70b
provider: cerebras
temperature: 0.0

# Methodology-specific settings
rag_config:
  top_k: 10
  similarity_threshold: 0.3
```

---

## Base Experiment Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"baseline"` | Unique experiment identifier. Used in output filenames. |
| `description` | `str` | `""` | Human-readable description for documentation purposes. |

---

## Tool Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_tools` | `int` | `10` | Number of tools to include in the experiment. Higher values increase difficulty. Recommended: 10-200. |
| `doc_length` | `str` | `"medium"` | Verbosity level of tool documentation. Options: `"minimal"`, `"short"`, `"medium"`, `"long"`, `"verbose"`. |
| `prompt_type` | `str` | `"concise"` | Test prompt style. Options: `"concise"` (natural, shorter) or `"clear"` (explicit, detailed). |
| `num_similar_tools` | `int` | `0` | Number of similar/distractor tools to add. Increases difficulty through semantic overlap. |
| `categories` | `list[str]` or `null` | `null` | Tool categories to include. If `null`, all categories are used. Example: `["file_operations", "database_operations"]`. |

### Doc Length Examples

| Level | Description |
|-------|-------------|
| `minimal` | Tool name and one-line description only |
| `short` | Brief description with basic parameter info |
| `medium` | Standard description with parameter details and types |
| `long` | Extended description with examples |
| `verbose` | Full documentation with usage examples and edge cases |

---

## Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"llama-3.3-70b"` | Model identifier. Determines provider if not explicitly set. |
| `provider` | `str` or `null` | `null` | Explicit provider selection. Options: `"cerebras"`, `"gemini"`, `"openai"`, `"ollama"`. Auto-detected from model name if `null`. |
| `temperature` | `float` | `0.0` | Sampling temperature. Lower values (0.0) are more deterministic. Range: 0.0-2.0. |

### Provider Auto-Detection

| Model Pattern | Detected Provider |
|---------------|-------------------|
| `llama-*`, `llama3*` | `cerebras` |
| `gemini-*` | `gemini` |
| `gpt-4*`, `gpt-3*` | `openai` |
| `*:*` (e.g., `model:tag`) | `ollama` |

### Ollama Provider (Local LLM)

Ollama allows running models locally without cloud API costs. Requires [Ollama](https://ollama.ai) to be installed and running.

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

#### Supported Models

Any model installed via `ollama pull <model>` can be used. Common models:

| Model | Description |
|-------|-------------|
| `gpt-oss:20b` | GPT-OSS 20B parameter model |
| `llama3:8b` | Llama 3 8B |
| `llama3:70b` | Llama 3 70B |
| `mistral:7b` | Mistral 7B |
| `mixtral:8x7b` | Mixtral 8x7B |
| `codellama:7b` | Code Llama 7B |
| `phi3:mini` | Phi-3 Mini |
| `qwen2:7b` | Qwen 2 7B |

#### Example Configuration

```yaml
name: ollama_local_test
model: gpt-oss:20b
provider: ollama  # Optional if model has ":" format
temperature: 0.0

# Ollama-specific (optional)
# base_url: http://localhost:11434

methodology: rag
num_tools: 100
```

#### Usage

1. Install Ollama: https://ollama.ai
2. Pull your model: `ollama pull gpt-oss:20b`
3. Start Ollama server: `ollama serve`
4. Run experiment with `provider: ollama`

---

## Test Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_test_samples` | `int` or `null` | `null` | Number of test cases to run. If `null`, all generated test cases are used. |
| `seed` | `int` | `42` | Random seed for reproducibility. Affects test case sampling and ordering. |
| `include_multi_tool` | `bool` | `false` | Include multi-tool test scenarios requiring sequential tool calls. |
| `include_no_tool` | `bool` | `false` | Include negative test cases where no tool should be called. |
| `include_ambiguous` | `bool` | `false` | Include ambiguous test cases to evaluate clarification capability. |

### Test Case Types

| Type | Description | Enabled By |
|------|-------------|-----------|
| Single-tool | Standard test with one correct tool | Always enabled |
| Multi-tool | Requires multiple tools in sequence | `include_multi_tool: true` |
| No-tool | Request that shouldn't trigger any tool | `include_no_tool: true` |
| Ambiguous | Vague request matching multiple tools | `include_ambiguous: true` |

---

## Methodology Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `methodology` | `str` | `"mcp"` | Methodology to use. Options: `"mcp"`, `"clustering"`, `"rag"`, `"hybrid"`, `"adaptive_rag"`, `"confidence"`. |
| `max_steps` | `int` | `10` | Maximum steps for multi-step methodologies (clustering, hybrid). Prevents infinite loops. |
| `allow_backtrack` | `bool` | `true` | Allow step-based methodologies to backtrack and retry with different selections. |
| `allow_no_tool_call` | `bool` | `false` | Allow LLM to explicitly decline calling any tool when no tool matches the request. |
| `allow_clarification` | `bool` | `false` | Allow LLM to request clarification when multiple tools could match an ambiguous request. |
| `max_clarification_candidates` | `int` | `3` | Maximum number of candidate tools for full clarification score (1.0). If more candidates, score is penalized as 1/len(candidates). |

### Methodology Comparison

| Methodology | Description | Best For | Key Config |
|-------------|-------------|----------|------------|
| `mcp` | All tools in context | <20 tools | None |
| `clustering` | Category → tool selection | 20-100 tools | Category descriptions |
| `rag` | Semantic retrieval | 50-200 tools | `rag_config.top_k` |
| `hybrid` | RAG categories + tools | 50-200 tools | `hybrid_config.top_k_categories` |
| `adaptive_rag` | Dynamic K retrieval | 50-200 tools | `adaptive_rag_config.min_k/max_k` |
| `confidence` | Fallback chain | Any | Threshold tuning |

---

## RAG Configuration

RAG (Retrieval-Augmented Generation) uses semantic embeddings to retrieve relevant tools.

**YAML key:** `rag_config`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_model` | `str` | `"all-MiniLM-L6-v2"` | Sentence-transformer model for embeddings. Options: `"all-MiniLM-L6-v2"` (fast), `"all-mpnet-base-v2"` (better quality). |
| `top_k` | `int` | `10` | Number of tools to retrieve. Higher values increase recall but add noise. Recommended: 5-20. |
| `similarity_threshold` | `float` | `0.0` | Minimum similarity score to include a tool. Range: 0.0-1.0. Set to 0.0 to rely only on top_k. |
| `cache_embeddings` | `bool` | `true` | Cache tool embeddings between runs. Significantly speeds up repeated experiments. |
| `include_params_in_embedding` | `bool` | `false` | Include parameter names in embedding text. Can improve retrieval for parameter-specific queries. |

### RAG Tuning Guidelines

| Symptom | Solution |
|---------|----------|
| Correct tool not retrieved | Increase `top_k` or lower `similarity_threshold` |
| Too many irrelevant tools | Decrease `top_k` or raise `similarity_threshold` |
| Poor semantic matching | Try `all-mpnet-base-v2` or improve tool descriptions |
| Slow embedding | Enable `cache_embeddings: true` |

---

## Adaptive RAG Configuration

Adaptive RAG dynamically adjusts K based on similarity score distribution.

**YAML key:** `adaptive_rag_config`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_model` | `str` | `"all-MiniLM-L6-v2"` | Sentence-transformer model for embeddings. |
| `min_k` | `int` | `3` | Minimum number of tools to retrieve. |
| `max_k` | `int` | `20` | Maximum number of tools to retrieve. |
| `similarity_drop_threshold` | `float` | `0.1` | Threshold for detecting "elbow" in similarity scores. When consecutive similarity drops exceed this, stop retrieving. |
| `min_similarity` | `float` | `0.3` | Minimum absolute similarity score. Tools below this are excluded regardless of drop threshold. |
| `cache_embeddings` | `bool` | `true` | Cache tool embeddings between runs. |
| `include_params_in_embedding` | `bool` | `false` | Include parameter names in embedding text. |

### Adaptive RAG Tuning Guidelines

| Symptom | Solution |
|---------|----------|
| K always at min_k | Raise `similarity_drop_threshold` to 0.15-0.2 |
| K always at max_k | Lower `similarity_drop_threshold` to 0.05 |
| Poor tool quality | Raise `min_similarity` to 0.4-0.5 |
| Missing correct tool | Lower `min_similarity` or raise `max_k` |

---

## Hybrid Configuration

Hybrid methodology combines RAG-based category retrieval with LLM tool selection.

**YAML key:** `hybrid_config`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_model` | `str` | `"all-MiniLM-L6-v2"` | Sentence-transformer model for category embeddings. |
| `top_k_categories` | `int` | `3` | Number of categories to retrieve. Higher values increase recall but add context. Recommended: 2-5. |
| `cache_embeddings` | `bool` | `true` | Cache category embeddings between runs. |

### Hybrid Tuning Guidelines

| Symptom | Solution |
|---------|----------|
| Correct category not retrieved | Increase `top_k_categories` |
| Too many categories | Decrease `top_k_categories` |
| Category confusion | Improve category descriptions in `tools/categories.yaml` |

---

## Confidence Configuration

Confidence methodology uses a fallback chain: RAG → Hybrid/Clustering → MCP.

**YAML key:** `confidence_config`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rag_threshold` | `float` | `0.8` | Minimum similarity score to accept RAG result without fallback. Range: 0.0-1.0. |
| `cluster_threshold` | `float` | `0.7` | Minimum confidence to accept Hybrid/Clustering result. Range: 0.0-1.0. |
| `rag_config` | `dict` | See RAG section | RAG configuration for first stage. |
| `hybrid_config` | `dict` | See Hybrid section | Hybrid configuration for second stage. |

### Confidence Tuning Guidelines

| Symptom | Solution |
|---------|----------|
| Too many fallbacks (slow) | Lower thresholds: `rag_threshold: 0.7`, `cluster_threshold: 0.6` |
| Wrong tool selected | Raise thresholds: `rag_threshold: 0.85`, `cluster_threshold: 0.8` |
| RAG dominates (>90%) | Raise `rag_threshold` to use fallback more |
| MCP dominates (>50%) | Lower thresholds or improve RAG/Hybrid quality |

---

## Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | `"experiments/results"` | Directory for experiment results. Created if doesn't exist. |
| `save_raw_responses` | `bool` | `false` | Save raw LLM API responses. Useful for debugging but increases disk usage. |

### Output Files

For each experiment, the following files are generated:
- `{name}_{timestamp}_summary.json` - Aggregated metrics
- `{name}_{timestamp}_details.csv` - Per-test-case results

---

## Example Configurations

### Minimal Configuration

```yaml
name: quick_test
methodology: mcp
num_tools: 10
model: llama-3.3-70b
```

### RAG with Tuned Retrieval

```yaml
name: rag_optimized
methodology: rag
num_tools: 100
doc_length: medium
model: llama-3.3-70b
temperature: 0.0

rag_config:
  embedding_model: all-MiniLM-L6-v2
  top_k: 15
  similarity_threshold: 0.2
  cache_embeddings: true
```

### Full Comparison Setup

```yaml
name: full_comparison
num_tools: 50
doc_length: medium
prompt_type: clear
model: llama-3.3-70b
temperature: 0.0
num_test_samples: 50
seed: 42
include_multi_tool: true
include_no_tool: true
allow_no_tool_call: true
max_steps: 10
allow_backtrack: true
```

### Clarification Testing

```yaml
name: clarification_test
num_tools: 50
methodology: rag
include_ambiguous: true
allow_clarification: true
max_clarification_candidates: 3

rag_config:
  top_k: 10
```

### Confidence with Custom Thresholds

```yaml
name: confidence_tuned
methodology: confidence
num_tools: 100

confidence_config:
  rag_threshold: 0.75
  cluster_threshold: 0.65
  rag_config:
    top_k: 12
    similarity_threshold: 0.1
  hybrid_config:
    top_k_categories: 4
```

### Adaptive RAG for Large Tool Sets

```yaml
name: adaptive_large
methodology: adaptive_rag
num_tools: 200

adaptive_rag_config:
  embedding_model: all-mpnet-base-v2
  min_k: 5
  max_k: 25
  similarity_drop_threshold: 0.08
  min_similarity: 0.25
  cache_embeddings: true
```

---

## Command-Line Override

Any YAML parameter can be overridden via command line:

```bash
python scripts/run_experiment.py run \
  --config experiments/full_cloud/plan/03_phase3_rag_100tools_k10.yaml \
  --num-tools 100 \
  --methodology adaptive_rag \
  --name override_test
```

See `python scripts/run_experiment.py run --help` for all options.

---

## See Also

- [METHODOLOGY_GUIDE.md](METHODOLOGY_GUIDE.md) - Methodology explanations and tuning guidelines
- [PLAN_CONFIGURATION.md](PLAN_CONFIGURATION.md) - Multi-run experiment setup
