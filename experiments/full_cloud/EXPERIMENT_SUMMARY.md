# Experiment Summary

**Model:** Cloud Llama 3.3 70B
**Total experiments:** 139

This document summarizes the number of experiment configurations by methodology, tool count, and documentation verbosity level.

## Overview: Tests per Methodology × Tool Count

| Methodology |10 | 25 | 50 | 100 | 200 | 300 | 350 | 500 | 985 | Total |
|-------------|---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MCP (Baseline) | 1 | 1 | 4 | 4 | 2 | 1 | 1 | - | - | 14 |
| Clustering | 1 | 1 | 3 | 3 | 6 | - | - | 1 | 2 | 17 |
| RAG | 1 | 1 | 2 | 12 | 14 | - | - | 9 | 7 | 46 |
| Adaptive RAG | 1 | 1 | 3 | 5 | 10 | - | - | 9 | 6 | 35 |
| Hybrid | 1 | 1 | 2 | 6 | 6 | - | - | 7 | 4 | 27 |

---

## Detailed Breakdown by Methodology

### MCP (Baseline)

| Tool Count |Minimal | Medium | Verbose | Total |
| ---: |---: | ---: | ---: | ---: |
| 10 | - | 1 | - | 1 |
| 25 | - | 1 | - | 1 |
| 50 | 1 | 2 | 1 | 4 |
| 100 | 1 | 2 | 1 | 4 |
| 200 | - | 2 | - | 2 |
| 300 | - | 1 | - | 1 |
| 350 | - | 1 | - | 1 |

### Clustering

| Tool Count |Minimal | Medium | Verbose | Total |
| ---: |---: | ---: | ---: | ---: |
| 10 | - | 1 | - | 1 |
| 25 | - | 1 | - | 1 |
| 50 | - | 2 | 1 | 3 |
| 100 | - | 3 | - | 3 |
| 200 | 2 | 3 | 1 | 6 |
| 500 | - | 1 | - | 1 |
| 985 | - | 2 | - | 2 |

### RAG

| Tool Count |Minimal | Medium | Verbose | Total |
| ---: |---: | ---: | ---: | ---: |
| 10 | - | 1 | - | 1 |
| 25 | - | 1 | - | 1 |
| 50 | - | 1 | 1 | 2 |
| 100 | - | 11 | 1 | 12 |
| 200 | 2 | 10 | 2 | 14 |
| 500 | 1 | 7 | 1 | 9 |
| 985 | - | 6 | 1 | 7 |

### Adaptive RAG

| Tool Count |Minimal | Medium | Verbose | Total |
| ---: |---: | ---: | ---: | ---: |
| 10 | - | 1 | - | 1 |
| 25 | - | 1 | - | 1 |
| 50 | - | 2 | 1 | 3 |
| 100 | - | 4 | 1 | 5 |
| 200 | 2 | 6 | 2 | 10 |
| 500 | - | 8 | 1 | 9 |
| 985 | 1 | 4 | 1 | 6 |

### Hybrid

| Tool Count |Minimal | Medium | Verbose | Total |
| ---: |---: | ---: | ---: | ---: |
| 10 | - | 1 | - | 1 |
| 25 | - | 1 | - | 1 |
| 50 | - | 1 | 1 | 2 |
| 100 | - | 6 | - | 6 |
| 200 | - | 6 | - | 6 |
| 500 | 1 | 5 | 1 | 7 |
| 985 | - | 4 | - | 4 |

---

## Verbosity Coverage Analysis

This section shows which (methodology, tool_count) pairs have complete verbosity coverage (all 3 levels).

### ✅ Complete Coverage (all 3 verbosity levels)

- MCP (Baseline) @ 50 tools
- MCP (Baseline) @ 100 tools
- Clustering @ 200 tools
- RAG @ 200 tools
- RAG @ 500 tools
- Adaptive RAG @ 200 tools
- Adaptive RAG @ 985 tools
- Hybrid @ 500 tools

### ⚠️ Partial Coverage

- MCP (Baseline) @ 10 tools - missing: minimal, verbose
- MCP (Baseline) @ 25 tools - missing: minimal, verbose
- MCP (Baseline) @ 200 tools - missing: minimal, verbose
- MCP (Baseline) @ 300 tools - missing: minimal, verbose
- MCP (Baseline) @ 350 tools - missing: minimal, verbose
- Clustering @ 10 tools - missing: minimal, verbose
- Clustering @ 25 tools - missing: minimal, verbose
- Clustering @ 50 tools - missing: minimal
- Clustering @ 100 tools - missing: minimal, verbose
- Clustering @ 500 tools - missing: minimal, verbose
- Clustering @ 985 tools - missing: minimal, verbose
- RAG @ 10 tools - missing: minimal, verbose
- RAG @ 25 tools - missing: minimal, verbose
- RAG @ 50 tools - missing: minimal
- RAG @ 100 tools - missing: minimal
- RAG @ 985 tools - missing: minimal
- Adaptive RAG @ 10 tools - missing: minimal, verbose
- Adaptive RAG @ 25 tools - missing: minimal, verbose
- Adaptive RAG @ 50 tools - missing: minimal
- Adaptive RAG @ 100 tools - missing: minimal
- Adaptive RAG @ 500 tools - missing: minimal
- Hybrid @ 10 tools - missing: minimal, verbose
- Hybrid @ 25 tools - missing: minimal, verbose
- Hybrid @ 50 tools - missing: minimal
- Hybrid @ 100 tools - missing: minimal, verbose
- Hybrid @ 200 tools - missing: minimal, verbose
- Hybrid @ 985 tools - missing: minimal, verbose
