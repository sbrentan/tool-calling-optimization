# Tool Calling Optimization

A testing framework for measuring LLM tool-calling accuracy and finding breaking points across multiple providers.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Supported Providers

**Disclaimer:** Only Cerebras has been tested for cloud experiments. 

| Provider | Models |
|----------|--------|
| **Cerebras** | llama-3.3-70b, llama3.1-8b, qwen-3-32b |
| **Gemini** | gemini-2.0-flash, gemini-1.5-pro, etc. |
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-3.5-turbo |
| **Local** | Local Llama 3.2 3B, Local Qwen 3.2 7B |

All models not matching Cerebras, Gemini, or OpenAI naming conventions are treated as local models with Ollama API - e.g. `llama3.2:3b` (check docs/CONFIG_REFERENCE.md for details).

Get free API keys:
- **Cerebras**: https://cloud.cerebras.ai/
- **Gemini**: https://aistudio.google.com/
- **OpenAI**: https://platform.openai.com/

## Running Experiments

### List Available Models
```bash
python scripts/run_experiment.py list-models
```

### Run a Single Experiment

With Cerebras (free, recommended):
```bash
python scripts/run_experiment.py run --model llama-3.3-70b --num-tools 10 --doc-length medium
```

With config file:
```bash
python scripts/run_experiment.py run --config experiments/configs/baseline.yaml
```

With explicit provider:
```bash
python scripts/run_experiment.py run --model llama-3.3-70b --provider cerebras --num-tools 15
```

### CLI Options
```
Options:
  -c, --config PATH       Path to YAML config file
  -n, --num-tools INT     Number of tools (default: 10)
  -d, --doc-length TEXT   Documentation length: minimal/short/medium/long/verbose
  -m, --model TEXT        Model name (auto-detects provider)
  -p, --provider TEXT     Provider: gemini/cerebras/openai (optional)
  -s, --num-similar INT   Number of similar/distractor tools
  --seed INT              Random seed for reproducibility
  --name TEXT             Experiment name
  -v, --verbose           Enable debug logging
```

### Run Parameter Sweep
```bash
python scripts/run_experiment.py sweep --provider cerebras
```

## Project Structure

```
src/
├── tools/          # Tool generation and schemas
├── adapters/       # Format adapters (MCP → provider formats)
├── clients/        # API clients (Cerebras, Gemini, OpenAI)
├── evaluation/     # Metrics and scoring
└── experiments/    # Experiment configuration

scripts/            # Entry point scripts
experiments/        # Experiment configs and results
```

## Experiment Parameters

- `num_tools`: Number of available tools (5, 10, 25, 50, 100)
- `doc_length`: Documentation verbosity (minimal, short, medium, long, verbose)
- `model`: Model to use (auto-detects provider from model name)
- `provider`: Explicit provider selection (optional)
- `num_similar_tools`: Number of semantically similar distractor tools

## Output

Results are saved to `experiments/results/`:
- `*_summary.json` - Aggregated metrics (accuracy, latency, etc.)
- `*_details.csv` - Per-test-case results for analysis

## Tool Categories

You can get the list of available tool categories in `tools/README.md`.

Additionally, you can list the amount of available tools per category by running this PowerShell command in the root directory (for Windows):
```powershell
Get-ChildItem -Path ".\tools\*.yaml" | ForEach-Object { $file = $_.Name; $count = (Select-String -Path $_.FullName -Pattern "^  - name:" | Measure-Object).Count; "$file`: $count tools" } | Sort-Object
```

Or for Linux/Mac:
```bash
for file in tools/*.yaml; do count=$(grep -c "^  - name:" "$file"); echo "$(basename "$file"): $count tools"; done | sort
```

## Run Multiple Experiments

You can run multiple experiments with different configurations by executing `scripts/run_plan.py`. Check `scripts/PLAN_CONFIGURATION.md` for details on how to set up a plan configuration file.
