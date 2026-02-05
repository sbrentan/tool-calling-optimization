# Experiment Summary

**Model:** Local Llama 3.2 3B
**Total experiments:** 28

This document summarizes the number of experiment configurations by methodology, tool count, and documentation verbosity level.

## Overview: Tests per Methodology × Tool Count

| Methodology |10 | 25 | 50 | 100 | 200 | 300 | 350 | 500 | 985 | Total |
|-------------|---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MCP (Baseline) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | - | 7 |
| RAG | 1 | 1 | 1 | 1 | 1 | - | - | 1 | 1 | 7 |
| Adaptive RAG | 1 | 1 | 1 | 1 | 1 | - | - | 1 | 1 | 7 |
| Hybrid | 1 | 1 | 1 | 1 | 1 | - | - | 1 | 1 | 7 |

---

## Detailed Breakdown by Methodology

### MCP (Baseline)

| Tool Count |Minimal | Medium | Verbose | Total |
| ---: |---: | ---: | ---: | ---: |
| 10 | - | 1 | - | 1 |
| 25 | - | 1 | - | 1 |
| 50 | - | 1 | - | 1 |
| 100 | - | 1 | - | 1 |
| 200 | - | 1 | - | 1 |
| 300 | - | 1 | - | 1 |
| 350 | - | 1 | - | 1 |

### RAG

| Tool Count |Minimal | Medium | Verbose | Total |
| ---: |---: | ---: | ---: | ---: |
| 10 | - | 1 | - | 1 |
| 25 | - | 1 | - | 1 |
| 50 | - | 1 | - | 1 |
| 100 | - | 1 | - | 1 |
| 200 | - | 1 | - | 1 |
| 500 | - | 1 | - | 1 |
| 985 | - | 1 | - | 1 |

### Adaptive RAG

| Tool Count |Minimal | Medium | Verbose | Total |
| ---: |---: | ---: | ---: | ---: |
| 10 | - | 1 | - | 1 |
| 25 | - | 1 | - | 1 |
| 50 | - | 1 | - | 1 |
| 100 | - | 1 | - | 1 |
| 200 | - | 1 | - | 1 |
| 500 | - | 1 | - | 1 |
| 985 | - | 1 | - | 1 |

### Hybrid

| Tool Count |Minimal | Medium | Verbose | Total |
| ---: |---: | ---: | ---: | ---: |
| 10 | - | 1 | - | 1 |
| 25 | - | 1 | - | 1 |
| 50 | - | 1 | - | 1 |
| 100 | - | 1 | - | 1 |
| 200 | - | 1 | - | 1 |
| 500 | - | 1 | - | 1 |
| 985 | - | 1 | - | 1 |

---

## Verbosity Coverage Analysis

This section shows which (methodology, tool_count) pairs have complete verbosity coverage (all 3 levels).

### ✅ Complete Coverage (all 3 verbosity levels)

*None*

### ⚠️ Partial Coverage

- MCP (Baseline) @ 10 tools - missing: minimal, verbose
- MCP (Baseline) @ 25 tools - missing: minimal, verbose
- MCP (Baseline) @ 50 tools - missing: minimal, verbose
- MCP (Baseline) @ 100 tools - missing: minimal, verbose
- MCP (Baseline) @ 200 tools - missing: minimal, verbose
- MCP (Baseline) @ 300 tools - missing: minimal, verbose
- MCP (Baseline) @ 350 tools - missing: minimal, verbose
- RAG @ 10 tools - missing: minimal, verbose
- RAG @ 25 tools - missing: minimal, verbose
- RAG @ 50 tools - missing: minimal, verbose
- RAG @ 100 tools - missing: minimal, verbose
- RAG @ 200 tools - missing: minimal, verbose
- RAG @ 500 tools - missing: minimal, verbose
- RAG @ 985 tools - missing: minimal, verbose
- Adaptive RAG @ 10 tools - missing: minimal, verbose
- Adaptive RAG @ 25 tools - missing: minimal, verbose
- Adaptive RAG @ 50 tools - missing: minimal, verbose
- Adaptive RAG @ 100 tools - missing: minimal, verbose
- Adaptive RAG @ 200 tools - missing: minimal, verbose
- Adaptive RAG @ 500 tools - missing: minimal, verbose
- Adaptive RAG @ 985 tools - missing: minimal, verbose
- Hybrid @ 10 tools - missing: minimal, verbose
- Hybrid @ 25 tools - missing: minimal, verbose
- Hybrid @ 50 tools - missing: minimal, verbose
- Hybrid @ 100 tools - missing: minimal, verbose
- Hybrid @ 200 tools - missing: minimal, verbose
- Hybrid @ 500 tools - missing: minimal, verbose
- Hybrid @ 985 tools - missing: minimal, verbose
