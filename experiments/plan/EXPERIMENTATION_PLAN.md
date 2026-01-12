# Tool-Calling Optimization: Comprehensive Experimentation Plan

## Overview

This document outlines a systematic experimentation strategy for evaluating and optimizing LLM tool-calling capabilities across 985 tools. The plan is designed to address the university grading rubric dimensions:

1. **Design & Prototypes** (0-10): Testing meaningful design choices and alternatives
2. **Experiments & Analysis** (0-10): Structured experiments with clear metrics and honest analysis
3. **Communication & Structure** (0-10): Clear documentation of motivations, choices, and lessons learned

## Environment Configuration

The following environment variables can be set to override experiment parameters:

| Variable | Description | Default |
|----------|-------------|---------|
| `EXPERIMENT_NUM_SAMPLES` | Number of test samples per experiment | None (all samples) |
| `EXPERIMENT_MODEL` | Model to use for experiments | `llama-3.3-70b` |
| `EXPERIMENT_SEED` | Random seed for reproducibility | 42 |

Set these in your `.env` file or export them before running experiments:
```bash
export EXPERIMENT_NUM_SAMPLES=10
export EXPERIMENT_MODEL=llama-3.3-70b
export EXPERIMENT_SEED=42
```

## Methodology Progression Logic

The experimentation follows a **progressive complexity** approach:

```
MCP (Baseline) → Clustering → RAG → Adaptive RAG → Hybrid → Confidence
     ↓               ↓          ↓         ↓            ↓          ↓
  "Perfect"     Hierarchical  Semantic  Dynamic K   Combined   Fallback
  but scales    selection     retrieval  adjustment  approach   chain
  poorly
```

Each phase identifies the **limitations** of the current methodology, justifying the need for the next.

---

## Phase 1: MCP Baseline & Breaking Point Analysis

### Objective
Establish accuracy ceiling and identify context limit breaking points for the simplest approach (all tools in context).

### Hypothesis
- MCP will achieve near-perfect accuracy with small tool sets (<50 tools)
- Performance will degrade as context size increases (>100 tools)
- Documentation verbosity will significantly impact token usage

### Rubric Alignment
- **Design**: Testing the baseline design with no optimization
- **Experiments**: Establishing metrics baseline for comparison
- **Communication**: Documenting the need for optimization

### Configuration Files

| Config | Tools | Doc Length | Prompt Type | Purpose |
|--------|-------|------------|-------------|---------|
| `01_phase1_mcp_10tools_medium.yaml` | 10 | medium | concise | Baseline with minimal tools |
| `01_phase1_mcp_25tools_medium.yaml` | 25 | medium | concise | Small scale test |
| `01_phase1_mcp_50tools_medium.yaml` | 50 | medium | concise | Medium scale test |
| `01_phase1_mcp_100tools_medium.yaml` | 100 | medium | concise | Context pressure begins |
| `01_phase1_mcp_200tools_medium.yaml` | 200 | medium | concise | Significant context load |
| `01_phase1_mcp_500tools_medium.yaml` | 500 | medium | concise | Near breaking point |
| `01_phase1_mcp_985tools_medium.yaml` | 985 | medium | concise | Full tool set (likely fails) |
| `01_phase1_mcp_50tools_minimal.yaml` | 50 | minimal | concise | Token reduction test |
| `01_phase1_mcp_50tools_verbose.yaml` | 50 | verbose | concise | Token increase impact |
| `01_phase1_mcp_50tools_clear.yaml` | 50 | medium | clear | Prompt clarity impact |

### Expected Outcomes
- Identify the tool count at which MCP accuracy drops below 90%
- Quantify token usage growth with tool count
- Establish baseline latency metrics

### Limitation Discovery
> **Expected Finding**: MCP becomes impractical above ~100-200 tools due to context limits and cost.
> **Next Step**: Test Clustering to reduce per-call context size.

---

## Phase 2: Clustering Methodology Exploration

### Objective
Evaluate hierarchical selection (category → tool) as a context-reduction strategy.

### Hypothesis
- Clustering will maintain accuracy while reducing context per call
- Backtracking will improve recovery from category selection errors
- Performance depends heavily on category organization quality

### Rubric Alignment
- **Design**: Testing alternative architecture (hierarchical vs flat)
- **Experiments**: Comparing with/without backtracking
- **Communication**: Analyzing when hierarchy helps vs hurts

### Configuration Files

| Config | Tools | Backtrack | Doc Length | Prompt Type | Purpose |
|--------|-------|-----------|------------|-------------|---------|
| `02_phase2_clustering_50tools_backtrack.yaml` | 50 | true | medium | concise | Small scale with recovery |
| `02_phase2_clustering_50tools_nobacktrack.yaml` | 50 | false | medium | concise | Impact of no recovery |
| `02_phase2_clustering_100tools_backtrack.yaml` | 100 | true | medium | concise | Medium scale |
| `02_phase2_clustering_200tools_backtrack.yaml` | 200 | true | medium | concise | Larger scale |
| `02_phase2_clustering_500tools_backtrack.yaml` | 500 | true | medium | concise | Large scale |
| `02_phase2_clustering_985tools_backtrack.yaml` | 985 | true | medium | concise | Full scale |
| `02_phase2_clustering_985tools_nobacktrack.yaml` | 985 | false | medium | concise | Full scale, no recovery |
| `02_phase2_clustering_200tools_minimal.yaml` | 200 | true | minimal | concise | Token reduction |
| `02_phase2_clustering_200tools_clear.yaml` | 200 | true | medium | clear | Prompt clarity |

### Expected Outcomes
- Measure category selection accuracy
- Quantify backtracking frequency and its impact
- Compare token usage vs MCP at same tool counts

### Limitation Discovery
> **Expected Finding**: Clustering struggles when queries don't clearly map to a single category.
> **Next Step**: Test RAG for semantic-based tool retrieval.

---

## Phase 3: RAG Parameter Sweep

### Objective
Explore semantic retrieval as an alternative to hierarchical selection, optimizing key parameters.

### Hypothesis
- RAG will handle ambiguous queries better than Clustering
- `top_k` is the most critical parameter for accuracy/efficiency tradeoff
- Higher `similarity_threshold` reduces noise but risks missing correct tools

### Rubric Alignment
- **Design**: Testing semantic vs categorical organization
- **Experiments**: Systematic parameter exploration
- **Communication**: Documenting optimal parameter ranges

### Configuration Files

| Config | Tools | top_k | sim_threshold | Doc Length | Purpose |
|--------|-------|-------|---------------|------------|---------|
| `03_phase3_rag_100tools_k5.yaml` | 100 | 5 | 0.0 | medium | Minimal retrieval |
| `03_phase3_rag_100tools_k10.yaml` | 100 | 10 | 0.0 | medium | Default retrieval |
| `03_phase3_rag_100tools_k15.yaml` | 100 | 15 | 0.0 | medium | Increased retrieval |
| `03_phase3_rag_100tools_k20.yaml` | 100 | 20 | 0.0 | medium | High retrieval |
| `03_phase3_rag_200tools_k10.yaml` | 200 | 10 | 0.0 | medium | Scale test |
| `03_phase3_rag_200tools_k10_t01.yaml` | 200 | 10 | 0.1 | medium | Threshold filtering |
| `03_phase3_rag_200tools_k10_t02.yaml` | 200 | 10 | 0.2 | medium | Higher threshold |
| `03_phase3_rag_500tools_k10.yaml` | 500 | 10 | 0.0 | medium | Large scale |
| `03_phase3_rag_500tools_k15.yaml` | 500 | 15 | 0.0 | medium | Large scale, more k |
| `03_phase3_rag_500tools_k20.yaml` | 500 | 20 | 0.0 | medium | Large scale, high k |
| `03_phase3_rag_985tools_k10.yaml` | 985 | 10 | 0.0 | medium | Full scale |
| `03_phase3_rag_985tools_k15.yaml` | 985 | 15 | 0.0 | medium | Full scale, more k |
| `03_phase3_rag_985tools_k20.yaml` | 985 | 20 | 0.0 | medium | Full scale, high k |
| `03_phase3_rag_985tools_k30.yaml` | 985 | 30 | 0.0 | medium | Full scale, very high k |
| `03_phase3_rag_500tools_k10_minimal.yaml` | 500 | 10 | 0.0 | minimal | Token reduction |
| `03_phase3_rag_500tools_k10_clear.yaml` | 500 | 10 | 0.0 | medium | Prompt clarity (clear) |

### Expected Outcomes
- Identify optimal `top_k` for accuracy vs context size tradeoff
- Measure retrieval recall (correct tool in top-k)
- Compare with Clustering at same tool counts

### Limitation Discovery
> **Expected Finding**: Fixed `top_k` is suboptimal—simple queries need fewer tools, complex queries need more.
> **Next Step**: Test Adaptive RAG with dynamic K selection.

---

## Phase 4: Adaptive RAG Tuning

### Objective
Optimize dynamic K selection based on query complexity and similarity distribution.

### Hypothesis
- Adaptive K will improve efficiency for simple queries while maintaining accuracy for complex ones
- `similarity_drop_threshold` controls the sensitivity of K adjustment
- `min_similarity` prevents inclusion of irrelevant tools

### Rubric Alignment
- **Design**: Testing dynamic vs static parameters
- **Experiments**: Tuning multiple interacting parameters
- **Communication**: Understanding elbow detection behavior

### Configuration Files

| Config | Tools | min_k | max_k | drop_threshold | min_sim | Purpose |
|--------|-------|-------|-------|----------------|---------|---------|
| `04_phase4_adaptive_200tools_default.yaml` | 200 | 3 | 20 | 0.1 | 0.3 | Default settings |
| `04_phase4_adaptive_200tools_drop005.yaml` | 200 | 3 | 20 | 0.05 | 0.3 | More sensitive elbow |
| `04_phase4_adaptive_200tools_drop015.yaml` | 200 | 3 | 20 | 0.15 | 0.3 | Less sensitive |
| `04_phase4_adaptive_200tools_drop020.yaml` | 200 | 3 | 20 | 0.20 | 0.3 | Aggressive cutoff |
| `04_phase4_adaptive_500tools_default.yaml` | 500 | 3 | 20 | 0.1 | 0.3 | Scale test |
| `04_phase4_adaptive_500tools_mink5.yaml` | 500 | 5 | 20 | 0.1 | 0.3 | Higher minimum |
| `04_phase4_adaptive_500tools_maxk30.yaml` | 500 | 3 | 30 | 0.1 | 0.3 | Higher maximum |
| `04_phase4_adaptive_500tools_minsim02.yaml` | 500 | 3 | 20 | 0.1 | 0.2 | Lower min similarity |
| `04_phase4_adaptive_500tools_minsim04.yaml` | 500 | 3 | 20 | 0.1 | 0.4 | Higher min similarity |
| `04_phase4_adaptive_985tools_default.yaml` | 985 | 3 | 20 | 0.1 | 0.3 | Full scale |
| `04_phase4_adaptive_985tools_optimized.yaml` | 985 | 5 | 25 | 0.12 | 0.25 | Tuned settings |
| `04_phase4_adaptive_985tools_minimal.yaml` | 985 | 3 | 20 | 0.1 | 0.3 | Token reduction (minimal) |
| `04_phase4_adaptive_985tools_clear.yaml` | 985 | 3 | 20 | 0.1 | 0.3 | Prompt clarity (clear) |

### Expected Outcomes
- Measure K distribution across queries
- Compare average context size vs fixed-K RAG
- Identify optimal parameter combination

### Limitation Discovery
> **Expected Finding**: Semantic similarity alone may miss structurally organized tools.
> **Next Step**: Test Hybrid approach combining semantic + category structure.

---

## Phase 5: Hybrid Methodology

### Objective
Evaluate combining semantic retrieval with category structure for best of both worlds.

### Hypothesis
- Hybrid will leverage category organization for structured queries
- Category embedding strategy affects retrieval quality
- Works best with well-organized tool categories

### Rubric Alignment
- **Design**: Testing combined approach vs individual components
- **Experiments**: Comparing embedding strategies
- **Communication**: When to use hybrid vs pure semantic

### Configuration Files

| Config | Tools | top_k_cat | strategy | Doc Length | Purpose |
|--------|-------|-----------|----------|------------|---------|
| `05_phase5_hybrid_200tools_cat2.yaml` | 200 | 2 | mean | medium | Narrow category selection |
| `05_phase5_hybrid_200tools_cat3.yaml` | 200 | 3 | mean | medium | Default category count |
| `05_phase5_hybrid_200tools_cat5.yaml` | 200 | 5 | mean | medium | Wide category selection |
| `05_phase5_hybrid_200tools_cat3_desc.yaml` | 200 | 3 | description | medium | Description-based embedding |
| `05_phase5_hybrid_500tools_cat3.yaml` | 500 | 3 | mean | medium | Scale test |
| `05_phase5_hybrid_500tools_cat5.yaml` | 500 | 5 | mean | medium | More categories at scale |
| `05_phase5_hybrid_985tools_cat3.yaml` | 985 | 3 | mean | medium | Full scale |
| `05_phase5_hybrid_985tools_cat5.yaml` | 985 | 5 | mean | medium | Full scale, wider |
| `05_phase5_hybrid_985tools_cat7.yaml` | 985 | 7 | mean | medium | Full scale, very wide |
| `05_phase5_hybrid_500tools_minimal.yaml` | 500 | 3 | mean | minimal | Token reduction |
| `05_phase5_hybrid_500tools_clear.yaml` | 500 | 3 | mean | medium | Prompt clarity (clear) |

### Expected Outcomes
- Compare with pure RAG and pure Clustering
- Measure category retrieval accuracy
- Identify optimal category count

### Limitation Discovery
> **Expected Finding**: All methods have failure cases; need fallback mechanism for production reliability.
> **Next Step**: Test Confidence-based fallback chain.

---

## Phase 6: Confidence Fallback Chain

### Objective
Implement and tune a production-ready fallback system: RAG → Clustering → MCP.

### Hypothesis
- Fallback chain will achieve highest reliability
- Threshold tuning balances efficiency vs accuracy
- MCP fallback should be rare but available for edge cases

### Rubric Alignment
- **Design**: Testing production reliability architecture
- **Experiments**: Threshold optimization
- **Communication**: Understanding fallback trigger patterns

### Configuration Files

| Config | rag_thresh | cluster_thresh | top_k | Purpose |
|--------|------------|----------------|-------|---------|
| `06_phase6_confidence_985tools_high.yaml` | 0.9 | 0.8 | 10 | High confidence requirements |
| `06_phase6_confidence_985tools_med.yaml` | 0.8 | 0.7 | 10 | Medium confidence |
| `06_phase6_confidence_985tools_low.yaml` | 0.7 | 0.6 | 10 | Lower confidence |
| `06_phase6_confidence_985tools_vlow.yaml` | 0.6 | 0.5 | 10 | Very low confidence |
| `06_phase6_confidence_985tools_ragonly.yaml` | 0.5 | 0.4 | 10 | Rarely fallback |
| `06_phase6_confidence_985tools_k15_med.yaml` | 0.8 | 0.7 | 15 | More tools per RAG |
| `06_phase6_confidence_985tools_k20_med.yaml` | 0.8 | 0.7 | 20 | Even more tools |
| `06_phase6_confidence_500tools_med.yaml` | 0.8 | 0.7 | 10 | Scale comparison |
| `06_phase6_confidence_985tools_minimal.yaml` | 0.8 | 0.7 | 10 | Token reduction (minimal) |
| `06_phase6_confidence_985tools_clear.yaml` | 0.8 | 0.7 | 10 | Prompt clarity (clear) |

### Expected Outcomes
- Measure fallback frequency per method
- Compare reliability vs pure methods
- Identify optimal threshold combination

---

## Phase 7: Robustness & Edge Cases

### Objective
Test methodology robustness with similar/distractor tools and no-tool scenarios.

### Hypothesis
- Similar tools will decrease accuracy for all methods
- RAG-based methods may be more vulnerable to semantic distractors
- No-tool handling is critical for production use

### Rubric Alignment
- **Design**: Testing edge case handling
- **Experiments**: Stress testing with adversarial conditions
- **Communication**: Honest analysis of failure modes

### Configuration Files

| Config | Methodology | Similar Tools | No-Tool | Purpose |
|--------|-------------|---------------|---------|---------|
| `07_phase7_robust_mcp_similar5.yaml` | mcp | 5 | false | MCP with distractors |
| `07_phase7_robust_mcp_similar10.yaml` | mcp | 10 | false | More distractors |
| `07_phase7_robust_rag_similar5.yaml` | rag | 5 | false | RAG robustness |
| `07_phase7_robust_rag_similar10.yaml` | rag | 10 | false | RAG with more distractors |
| `07_phase7_robust_adaptive_similar10.yaml` | adaptive_rag | 10 | false | Adaptive robustness |
| `07_phase7_robust_confidence_similar10.yaml` | confidence | 10 | false | Confidence robustness |
| `07_phase7_robust_mcp_notool.yaml` | mcp | 0 | true | MCP no-tool handling |
| `07_phase7_robust_rag_notool.yaml` | rag | 0 | true | RAG no-tool handling |
| `07_phase7_robust_adaptive_notool.yaml` | adaptive_rag | 0 | true | Adaptive no-tool |
| `07_phase7_robust_confidence_notool.yaml` | confidence | 0 | true | Confidence no-tool |

### Expected Outcomes
- Quantify accuracy degradation with distractors
- Measure false positive rates for no-tool scenarios
- Identify most robust methodology

---

## Phase 8: Model Comparison

### Objective
Test winning methodology across different LLM models.

### Hypothesis
- Larger models will perform better but cost more
- Model choice may interact with methodology effectiveness
- Free-tier models (Cerebras) may be sufficient for optimized approaches

### Rubric Alignment
- **Design**: Testing model alternatives
- **Experiments**: Cross-model comparison
- **Communication**: Cost-accuracy tradeoffs

### Configuration Files

| Config | Model | Methodology | Tools | Purpose |
|--------|-------|-------------|-------|---------|
| `08_phase8_model_llama70b_adaptive.yaml` | llama-3.3-70b | adaptive_rag | 985 | Default model |
| `08_phase8_model_llama8b_adaptive.yaml` | llama3.1-8b | adaptive_rag | 985 | Smaller model |
| `08_phase8_model_gemini_adaptive.yaml` | gemini-2.0-flash | adaptive_rag | 985 | Gemini provider |
| `08_phase8_model_gpt4omini_adaptive.yaml` | gpt-4o-mini | adaptive_rag | 985 | OpenAI provider |
| `08_phase8_model_llama70b_confidence.yaml` | llama-3.3-70b | confidence | 985 | Confidence + default |
| `08_phase8_model_llama8b_confidence.yaml` | llama3.1-8b | confidence | 985 | Confidence + small |
| `08_phase8_model_gemini_confidence.yaml` | gemini-2.0-flash | confidence | 985 | Confidence + Gemini |
| `08_phase8_model_gpt4omini_confidence.yaml` | gpt-4o-mini | confidence | 985 | Confidence + OpenAI |

### Expected Outcomes
- Compare accuracy across models
- Measure latency and cost differences
- Identify best model-methodology combination

---

## Execution Instructions

### Running All Experiments

Use the batch runner script:

```bash
# Run all experiments from the beginning
python scripts/run_plan.py

# Start from a specific test (e.g., if resuming after stopping)
python scripts/run_plan.py --start-from 15

# Run without confirmation prompts (for automated runs)
python scripts/run_plan.py --no-confirm
```

### Environment Setup

```bash
# Required environment variables
export CEREBRAS_API_KEY=your_key_here
export EXPERIMENT_NUM_SAMPLES=10
export EXPERIMENT_MODEL=llama-3.3-70b
export EXPERIMENT_SEED=42

# Optional: for model comparison phase
export GEMINI_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
```

---

## Results Analysis Framework

After running experiments, analyze results using:

```bash
python scripts/analyze_results.py experiments/results --output reports/plan_analysis.html
```

### Key Metrics to Compare

1. **Accuracy**: Correct tool selection rate
2. **Retrieval Recall**: Correct tool in retrieved set (RAG methods)
3. **Token Usage**: Input/output tokens per call
4. **Latency**: Average response time
5. **Fallback Rate**: Frequency of fallback usage (Confidence method)
6. **Distractor Resistance**: Accuracy with similar tools

### Expected Report Structure

1. **Phase-by-Phase Analysis**: Results and insights per phase
2. **Cross-Phase Comparison**: Best methodology identification
3. **Parameter Sensitivity**: Impact of key parameters
4. **Failure Analysis**: When and why methods fail
5. **Recommendations**: Production deployment guidance

---

## Summary: Experiment Count

| Phase | Description | Configs |
|-------|-------------|---------|
| Phase 1 | MCP Baseline | 10 |
| Phase 2 | Clustering | 9 |
| Phase 3 | RAG | 16 |
| Phase 4 | Adaptive RAG | 13 |
| Phase 5 | Hybrid | 11 |
| Phase 6 | Confidence | 10 |
| Phase 7 | Robustness | 10 |
| Phase 8 | Model Comparison | 8 |
| **Total** | | **87** |

With 10 samples per experiment (configurable via `EXPERIMENT_NUM_SAMPLES`), total test runs: **870 test cases**

---

## Quick Start

```bash
# Set environment variables
export EXPERIMENT_NUM_SAMPLES=10
export EXPERIMENT_MODEL=llama-3.3-70b
export EXPERIMENT_SEED=42

# List all experiments
python scripts/run_plan.py --list

# Run all experiments (with confirmation between each)
python scripts/run_plan.py

# Run without confirmation (automated)
python scripts/run_plan.py --no-confirm

# Resume from experiment 25
python scripts/run_plan.py --start-from 25

# Dry run (see what would execute)
python scripts/run_plan.py --dry-run
```
