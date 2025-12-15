# Tool Calling Optimization

A testing framework for measuring LLM tool-calling accuracy and finding breaking points.

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
# Edit .env and add your GEMINI_API_KEY
```

## Running Experiments

Run a baseline experiment:
```bash
python scripts/run_experiment.py --config experiments/configs/baseline.yaml
```

Or with command-line options:
```bash
python scripts/run_experiment.py --num-tools 10 --doc-length medium --model gemini-2.0-flash
```

## Project Structure

```
src/
├── tools/          # Tool generation and schemas
├── adapters/       # Format adapters (MCP → Gemini)
├── clients/        # API clients
├── evaluation/     # Metrics and scoring
└── experiments/    # Experiment configuration

scripts/            # Entry point scripts
experiments/        # Experiment configs and results
```

## Experiment Parameters

- `num_tools`: Number of available tools (5, 10, 25, 50, 100)
- `doc_length`: Documentation verbosity (minimal, short, medium, long, verbose)
- `model`: Gemini model (gemini-2.0-flash, gemini-2.0-pro, etc.)
- `num_similar_tools`: Number of semantically similar distractor tools
- `required_calls`: Single or multiple tool calls expected