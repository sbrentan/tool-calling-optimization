# Tool Calling Methodologies Guide

This guide documents all methodologies implemented in the tool-calling optimization project, including their trade-offs, tuning parameters, and recommendations for when to use each approach.

---

## Table of Contents

1. [Overview](#overview)
2. [MCP (Model Context Protocol)](#mcp-model-context-protocol)
3. [Clustering](#clustering)
4. [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
5. [Hybrid (RAG + Clustering)](#hybrid-rag--clustering)
6. [Adaptive RAG](#adaptive-rag)
7. [Comparison Summary](#comparison-summary)
8. [Tuning Guidelines](#tuning-guidelines)

---

## Overview

Each methodology addresses the **context-accuracy tradeoff** in LLM tool calling:

| Challenge | Description |
|-----------|-------------|
| **Context Limits** | LLMs have limited context windows; can't fit all tools |
| **Selection Accuracy** | More tools in context → higher chance of correct selection |
| **Latency** | More context → slower inference |
| **Cost** | More tokens → higher API costs |

The methodologies fall into three categories:

1. **Full Context**: MCP (all tools in context)
2. **Reduced Context**: Clustering, RAG (subset of tools)
3. **Adaptive**: Hybrid, Adaptive RAG, Confidence (dynamic selection)

---

## MCP (Model Context Protocol)

### How It Works

The simplest approach: include **all tools** in the LLM context and let the model select the appropriate one.

```
User Query → [LLM with ALL tools in context] → Selected Tool + Arguments
```

### Implementation

Located in `src/methodologies/mcp.py`

```python
class MCPMethodology(BaseMethodology):
    NAME = "mcp"
    
    def run_single(self, prompt, tools, client):
        # Format all tools for context
        tool_descriptions = self._format_tools(tools)
        
        # Single LLM call with all tools
        response = client.tool_call(prompt, tools)
        return response
```

### Trade-offs

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Accuracy** | ⭐⭐⭐⭐⭐ | Highest - LLM sees all options |
| **Context Usage** | ⭐ | Scales linearly with tool count |
| **Latency** | ⭐⭐ | Slower with more tools |
| **Cost** | ⭐⭐ | Higher token usage |
| **Scalability** | ⭐ | Fails at 100+ tools |

### When to Use

✅ **Use MCP when:**
- Tool set is small (<50 tools)
- Accuracy is critical
- Context window is large (e.g., GPT-4 128k)
- Cost is not a concern

❌ **Avoid MCP when:**
- 100+ tools in the system
- Latency-sensitive applications
- Cost optimization required

### Configuration

```yaml
methodology: mcp
# No additional configuration needed
```

---

## Clustering

### How It Works

Two-step hierarchical selection:
1. **Category Selection**: LLM picks the most relevant category
2. **Tool Selection**: LLM picks the tool from that category

```
User Query → [LLM: Select Category] → [LLM: Select Tool from Category] → Result
```

### Implementation

Located in `src/methodologies/clustering.py`

```python
class ClusteringMethodology(StepBasedMethodology):
    NAME = "clustering"
    
    def run_single(self, prompt, tools, client):
        # Step 1: Select category
        categories = self._get_categories(tools)
        selected_category = client.select_category(prompt, categories)
        
        # Step 2: Select tool from category
        category_tools = [t for t in tools if t.category == selected_category]
        selected_tool = client.tool_call(prompt, category_tools)
        
        return selected_tool
```

### Trade-offs

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Accuracy** | ⭐⭐⭐ | Can fail on category selection |
| **Context Usage** | ⭐⭐⭐⭐⭐ | Very low - only category tools |
| **Latency** | ⭐⭐⭐ | 2 LLM calls required |
| **Cost** | ⭐⭐⭐⭐ | Low per-call tokens |
| **Scalability** | ⭐⭐⭐⭐ | Good for many tools |

### Common Failure Modes

1. **Category Confusion**: Tools that could belong to multiple categories
   - Example: `send_digest_notification` → analytics vs. notification?
   
2. **Vague Queries**: User intent doesn't clearly map to a category
   - Example: "Process the data" → data_operations vs. analytics?

3. **No Backtrack**: If wrong category selected, can't recover (unless `allow_backtrack=true`)

### When to Use

✅ **Use Clustering when:**
- Tools are well-organized into distinct categories
- Categories have clear, non-overlapping purposes
- Cost optimization is important
- You can tune category descriptions

❌ **Avoid Clustering when:**
- Categories are ambiguous or overlapping
- Tools don't fit neatly into categories
- User queries are often vague

### Configuration

```yaml
methodology: clustering
max_steps: 10          # Max LLM calls before giving up
allow_backtrack: true  # Try different category on failure
```

### Tuning Tips

1. **Improve Category Descriptions** in `tools/categories.yaml`:
   ```yaml
   - name: notification_operations
     description: >
       Push notifications, alerts, digests, and activity summaries.
       Use for: sending alerts, weekly digests, notification preferences.
       NOT for: emails (use email_operations), SMS (use messaging_operations).
   ```

2. **Add Negative Examples**: Tell the LLM what each category is NOT for.

3. **Cross-Reference Confusing Categories**: Add disambiguation hints.

---

## RAG (Retrieval-Augmented Generation)

### How It Works

Uses semantic embeddings to retrieve the most relevant tools:
1. **Embed Tools**: Create vector embeddings of tool descriptions
2. **Embed Query**: Embed the user's query
3. **Retrieve Top-K**: Find K most similar tools by cosine similarity
4. **LLM Selection**: LLM selects from retrieved tools only

```
User Query → [Embed] → [Vector Search: Top-K tools] → [LLM: Select Tool] → Result
```

### Implementation

Located in `src/methodologies/rag.py`

```python
class RAGMethodology(BaseMethodology):
    NAME = "rag"
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2", top_k=10):
        self.embedder = SentenceTransformer(embedding_model)
        self.top_k = top_k
    
    def run_single(self, prompt, tools, client):
        # Embed query
        query_embedding = self.embedder.encode(prompt)
        
        # Retrieve top-k similar tools
        similarities = self._compute_similarities(query_embedding, tools)
        top_tools = self._get_top_k(tools, similarities)
        
        # LLM selects from retrieved tools
        return client.tool_call(prompt, top_tools)
```

### Trade-offs

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Accuracy** | ⭐⭐⭐⭐ | Depends on embedding quality |
| **Context Usage** | ⭐⭐⭐⭐ | Fixed at K tools |
| **Latency** | ⭐⭐⭐⭐ | Fast embedding + 1 LLM call |
| **Cost** | ⭐⭐⭐⭐ | Low token usage |
| **Scalability** | ⭐⭐⭐⭐⭐ | Excellent for large tool sets |

### Common Failure Modes

1. **Semantic Gaps**: Query vocabulary differs from tool descriptions
   - Example: "resize my pic" doesn't match "image_resize: Adjust image dimensions"

2. **Fixed K**: Too low K misses correct tool; too high K adds noise

3. **Embedding Model Limits**: MiniLM may miss domain-specific relationships

### When to Use

✅ **Use RAG when:**
- Large tool sets (50-500+ tools)
- Tool descriptions are semantically rich
- Queries match tool vocabulary reasonably well
- You can tune the embedding model or K value

❌ **Avoid RAG when:**
- Very small tool sets (MCP is simpler)
- Tool descriptions are terse or technical jargon
- Queries use informal/slang language not in descriptions

### Configuration

```yaml
methodology: rag
rag_config:
  embedding_model: "all-MiniLM-L6-v2"  # Or "all-mpnet-base-v2" for better quality
  top_k: 10                             # Number of tools to retrieve
  similarity_threshold: 0.3             # Minimum similarity to include
  cache_embeddings: true                # Cache tool embeddings
  include_params_in_embedding: false    # Include parameter info
```

### Tuning Tips

1. **Tune K Based on Experiments**:
   - Start with K=10
   - If accuracy is low, increase K
   - Monitor retrieval recall (was correct tool in top-K?)

2. **Try Better Embedding Models**:
   - `all-mpnet-base-v2`: Higher quality, slower
   - Domain-specific models if available

3. **Enhance Tool Descriptions**:
   ```python
   # Add examples to make tools more retrievable
   tool.description = f"{tool.name}: {tool.description}. Example: 'resize image to 800x600'"
   ```

4. **Query Expansion**:
   ```python
   # Expand query before embedding
   expanded_query = f"{query}. Keywords: image, resize, dimensions"
   ```

---

## Hybrid (RAG + Clustering)

### How It Works

Combines RAG and Clustering strengths:
1. **Embed Categories**: Create embeddings for category descriptions
2. **Retrieve Top-K Categories**: Find most relevant categories semantically
3. **Get Category Tools**: Collect all tools from selected categories
4. **LLM Selection**: Select from combined tool set

```
Query → [Embed] → [Retrieve Top-K Categories] → [Get Tools from Categories] → [LLM] → Result
```

### Implementation

Located in `src/methodologies/hybrid.py`

```python
class HybridMethodology(BaseMethodology):
    NAME = "hybrid"
    
    def run_single(self, prompt, tools, client):
        # Retrieve top-k categories by semantic similarity
        query_embedding = self.embedder.encode(prompt)
        top_categories = self._get_top_categories(query_embedding, k=3)
        
        # Get all tools from those categories
        relevant_tools = [t for t in tools if t.category in top_categories]
        
        # LLM selects from relevant tools
        return client.tool_call(prompt, relevant_tools)
```

### Trade-offs

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Accuracy** | ⭐⭐⭐⭐ | More robust than pure clustering |
| **Context Usage** | ⭐⭐⭐⭐ | Moderate (category tools) |
| **Latency** | ⭐⭐⭐⭐ | Single LLM call |
| **Cost** | ⭐⭐⭐⭐ | Moderate tokens |
| **Scalability** | ⭐⭐⭐⭐ | Good for organized tool sets |

### When to Use

✅ **Use Hybrid when:**
- Tools are organized into meaningful categories
- Pure clustering fails due to category name matching issues
- You want semantic category selection

❌ **Avoid Hybrid when:**
- Categories are poorly defined
- Very small tool sets
- No clear category structure

### Configuration

```yaml
methodology: hybrid
hybrid_config:
  embedding_model: "all-MiniLM-L6-v2"
  top_k_categories: 3    # Number of categories to select
  cache_embeddings: true
```

---

## Adaptive RAG

### How It Works

Dynamically adjusts the number of retrieved tools (K) based on query characteristics:
1. **Compute Similarities**: Get similarity scores for all tools
2. **Determine Optimal K**: Use elbow detection or threshold-based selection
3. **Retrieve Dynamic K Tools**: Include tools above threshold
4. **LLM Selection**: Select from retrieved tools

```
Query → [Compute Similarities] → [Elbow Detection] → [Dynamic K Tools] → [LLM] → Result
```

### Implementation

Located in `src/methodologies/adaptive_rag.py`

```python
class AdaptiveRAGMethodology(RAGMethodology):
    NAME = "adaptive_rag"
    
    def _determine_k(self, similarities):
        # Sort similarities descending
        sorted_sims = np.sort(similarities)[::-1]
        
        # Elbow detection: find where similarity drops sharply
        gradients = np.diff(sorted_sims)
        elbow = np.argmax(gradients < -self.drop_threshold) + 1
        
        # Clamp to min/max bounds
        return max(self.min_k, min(elbow, self.max_k))
```

### Trade-offs

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Accuracy** | ⭐⭐⭐⭐⭐ | Better than fixed-K RAG |
| **Context Usage** | ⭐⭐⭐⭐ | Adapts to query needs |
| **Latency** | ⭐⭐⭐⭐ | Similar to RAG |
| **Cost** | ⭐⭐⭐⭐ | Variable, often lower than fixed-K |
| **Scalability** | ⭐⭐⭐⭐⭐ | Excellent |

### When to Use

✅ **Use Adaptive RAG when:**
- Query complexity varies significantly
- Some queries are simple (need few tools), others complex (need many)
- You want to optimize context usage automatically

❌ **Avoid Adaptive RAG when:**
- All queries are similar complexity
- You prefer predictable context sizes
- Elbow detection doesn't work well for your similarity distribution

### Configuration

```yaml
methodology: adaptive_rag
adaptive_rag_config:
  embedding_model: "all-MiniLM-L6-v2"
  min_k: 3                       # Minimum tools to retrieve
  max_k: 20                      # Maximum tools to retrieve
  similarity_drop_threshold: 0.1 # Gradient threshold for elbow detection
  min_similarity: 0.3            # Absolute minimum similarity to include
  cache_embeddings: true
```

### Tuning Tips

1. **Monitor K Distribution**: Track `adaptive_k_stats` in results
   - If always hitting max_k, threshold too tight
   - If always at min_k, threshold too loose

2. **Adjust Thresholds Based on Results**:
   - Low accuracy? Increase max_k or lower drop_threshold
   - High context usage? Lower max_k or raise drop_threshold

---

## Comparison Summary

| Methodology | Best For | Context | Accuracy | Speed | Cost |
|-------------|----------|---------|----------|-------|------|
| **MCP** | Small tool sets (<50) | High | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Clustering** | Well-organized categories | Low | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **RAG** | Large tool sets | Medium | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Hybrid** | Semantic category selection | Medium | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Adaptive RAG** | Variable complexity queries | Adaptive | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Tuning Guidelines

### General Principles

1. **Start Simple**: Begin with MCP for baseline, then add complexity
2. **Measure Everything**: Track accuracy, latency, cost, and failure modes
3. **Iterate on Failures**: Analyze incorrect selections to improve descriptions
4. **A/B Test**: Compare methodologies on the same test set

### Key Metrics to Monitor

| Metric | What It Tells You |
|--------|-------------------|
| **Accuracy** | Overall correctness |
| **Retrieval Recall@K** | Was correct tool in retrieved set? |
| **Category Accuracy** | Is category selection working? |
| **Latency Distribution** | Are there outliers? |

### Improving Accuracy

1. **Enhance Tool Descriptions**:
   - Add examples: "Use this to resize images, e.g., 'resize image.png to 800x600'"
   - Add keywords: "Keywords: image, photo, picture, resize, scale, dimensions"
   - Add negative examples: "NOT for cropping (use crop_image)"

2. **Tune Retrieval Parameters**:
   - Increase K if correct tool not retrieved
   - Lower similarity threshold if too few tools retrieved
   - Try different embedding models

3. **Improve Category Definitions**:
   - Add clear descriptions with examples
   - Add disambiguation between similar categories
   - Consider merging overlapping categories

### Reducing Latency

1. **Cache Embeddings**: Enable `cache_embeddings: true`
2. **Use Smaller Models**: Try smaller embedding models if accuracy allows
3. **Reduce K**: Lower K means less context for LLM
4. **Parallelize**: Pre-compute embeddings at startup

### Reducing Cost

1. **Use Efficient Methodologies**: RAG < Hybrid < Clustering < MCP
2. **Lower K Values**: Fewer tokens per request
3. **Use Smaller LLMs**: For simple queries, smaller models suffice

---

## Appendix: File Locations

| Component | Location |
|-----------|----------|
| MCP Methodology | `src/methodologies/mcp.py` |
| Clustering Methodology | `src/methodologies/clustering.py` |
| RAG Methodology | `src/methodologies/rag.py` |
| Hybrid Methodology | `src/methodologies/hybrid.py` |
| Adaptive RAG Methodology | `src/methodologies/adaptive_rag.py` |
| Tool Categories | `tools/categories.yaml` |
| Experiment Configs | `experiments/<experiment>/plan/*.yaml` |
| Report Generator | `scripts/utils/generate_limits_report.py` |
