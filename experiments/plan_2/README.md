# ============================================================
# SUPPLEMENTARY EXPERIMENTS PLAN (Updated)
# ============================================================
# These configs fill gaps for better comparative analysis.
# Priority order for running when rate limits allow.
# ============================================================

## Priority 1: Fill Heatmap Gaps (by methodology)
# Missing tool counts to complete the methodology × num_tools matrix

### Clustering (missing 10, 25)
09_fill_clustering_10tools.yaml
10_fill_clustering_25tools.yaml

### RAG (missing 10, 25, 50)
11_fill_rag_10tools.yaml
12_fill_rag_25tools.yaml
13_fill_rag_50tools.yaml

### Adaptive RAG (missing 10, 25, 50, 100)
14_fill_adaptive_10tools.yaml
15_fill_adaptive_25tools.yaml
16_fill_adaptive_50tools.yaml
17_fill_adaptive_100tools.yaml

### Hybrid (missing 10, 25, 50, 100)
18_fill_hybrid_10tools.yaml
19_fill_hybrid_25tools.yaml
20_fill_hybrid_50tools.yaml
21_fill_hybrid_100tools.yaml

## Priority 2: Verbose Documentation Tests
# Only MCP has verbose tests - add for other methodologies

22_verbose_clustering_50tools.yaml
23_verbose_rag_50tools.yaml
24_verbose_adaptive_50tools.yaml
25_verbose_hybrid_50tools.yaml

## Priority 3: Cross-methodology comparison at 200 tools
01_cross_mcp_200tools.yaml
02_cross_adaptive_200tools.yaml
03_cross_hybrid_200tools.yaml
08_cross_clustering_200tools.yaml

## Priority 4: Doc length impact for advanced methods
04_doclength_adaptive_200tools_minimal.yaml
05_doclength_adaptive_200tools_verbose.yaml
06_doclength_rag_200tools_minimal.yaml
07_doclength_rag_200tools_verbose.yaml
