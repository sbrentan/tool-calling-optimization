# Tuning Guide for Tool-Calling Methodologies

This guide provides practical recommendations for tuning each methodology to achieve optimal performance.

---

## Quick Reference: Key Parameters

| Methodology | Key Parameters | Default | Tune When |
|-------------|----------------|---------|-----------|
| **RAG** | `top_k` | 10 | Low retrieval recall |
| **RAG** | `similarity_threshold` | 0.0 | Too many irrelevant tools retrieved |
| **Adaptive RAG** | `min_k` / `max_k` | 3 / 20 | K always at bounds |
| **Adaptive RAG** | `similarity_drop_threshold` | 0.1 | Poor elbow detection |
| **Clustering** | Category descriptions | - | Category selection errors |
| **Hybrid** | `top_k_categories` | 3 | Missing correct category |
| **Confidence** | `rag_threshold` | 0.8 | Too many/few fallbacks |
| **Confidence** | `cluster_threshold` | 0.7 | Too many fallbacks to MCP |

---

## RAG Methodology Tuning

### Problem: Low Accuracy

**Symptom**: Correct tool not being selected

**Check Retrieval Recall First**:
```bash
# Look at retrieval_recall_rate in results
python scripts/analyze_results.py compare --results-dir experiments/results
```

If `retrieval_recall_rate` is low (< 90%):
1. **Increase `top_k`**: Try 15, 20, or 25
2. **Lower `similarity_threshold`**: Set to 0.0 to include more tools
3. **Improve tool descriptions**: Add keywords and examples

If `retrieval_recall_rate` is high but accuracy is low:
1. **Tool descriptions are ambiguous**: Clarify differences
2. **LLM is confused**: Check for similar tool names

### Problem: High Latency

**Cause**: Too many tools in context

**Solutions**:
1. **Decrease `top_k`**: Try 5 or 7
2. **Increase `similarity_threshold`**: Set to 0.3 or 0.4
3. **Use smaller embedding model**: `all-MiniLM-L6-v2` (default) is fast

### Recommended Starting Configuration

```yaml
rag_config:
  embedding_model: all-MiniLM-L6-v2
  top_k: 10
  similarity_threshold: 0.0
  cache_embeddings: true
```

---

## Adaptive RAG Tuning

### Problem: K Always at min_k

**Symptom**: `adaptive_k_stats.avg_k` ≈ `min_k`

**Cause**: Similarity drop threshold too aggressive

**Solutions**:
1. **Raise `similarity_drop_threshold`**: Try 0.15 or 0.2
2. **Lower `min_similarity`**: Try 0.2
3. **Check embedding model**: May need better semantic matching

### Problem: K Always at max_k

**Symptom**: `adaptive_k_stats.avg_k` ≈ `max_k`

**Cause**: Similarity scores don't drop enough

**Solutions**:
1. **Lower `similarity_drop_threshold`**: Try 0.05
2. **Improve tool descriptions**: More distinct descriptions
3. **Try different embedding model**: `all-mpnet-base-v2`

### Recommended Starting Configuration

```yaml
adaptive_rag_config:
  embedding_model: all-MiniLM-L6-v2
  min_k: 3
  max_k: 15
  similarity_drop_threshold: 0.1
  min_similarity: 0.3
  cache_embeddings: true
```

---

## Clustering Tuning

### Problem: Wrong Category Selected

**Symptom**: `category_selection_accuracy` is low

**Root Causes**:
1. **Ambiguous categories**: Tools could belong to multiple categories
2. **Poor category descriptions**: LLM can't distinguish categories
3. **Vague user queries**: Query doesn't clearly indicate category

### Solutions

**1. Improve Category Descriptions** (`tools/categories.yaml`):

```yaml
# Before (vague)
- name: notification_operations
  description: "Operations for notifications"

# After (clear)
- name: notification_operations
  description: >
    Push notifications, in-app alerts, digest summaries, and notification preferences.
    Use for: sending push notifications, weekly digest emails, alert settings.
    NOT for: SMS (use messaging_operations), emails (use email_operations).
```

**2. Add Disambiguation Hints**:

For commonly confused category pairs, add explicit distinctions:

```yaml
- name: data_operations
  description: >
    Data transformation, validation, parsing, and format conversion.
    Use for: transforming CSV to JSON, validating schemas, parsing text.
    NOT for: database queries (use database_operations), 
    analytics/reporting (use analytics_operations).
```

**3. Consider Merging Overlapping Categories**:

If two categories are consistently confused, merge them.

### Recommended Approach

1. Run experiment and check `category_accuracy` per category
2. Identify worst-performing categories
3. Update their descriptions with clear boundaries
4. Re-run and compare

---

## Hybrid Methodology Tuning

### Problem: Correct Category Not Retrieved

**Symptom**: Category selection accuracy lower than pure RAG

**Solutions**:
1. **Increase `top_k_categories`**: Try 4 or 5
2. **Improve category embedding text**: Add more descriptive category summaries
3. **Check category distribution**: Ensure categories are balanced

### Recommended Starting Configuration

```yaml
hybrid_config:
  embedding_model: all-MiniLM-L6-v2
  top_k_categories: 3
  cache_embeddings: true
```

---

## Confidence Methodology Tuning

### Problem: Too Many Fallbacks (High Latency/Cost)

**Symptom**: `fallback_rate` > 50%

**Cause**: Thresholds too high for RAG/Clustering

**Solutions**:
1. **Lower `rag_threshold`**: Try 0.7 or 0.6
2. **Lower `cluster_threshold`**: Try 0.6 or 0.5
3. **Improve underlying methodology**: Better RAG retrieval or category descriptions

### Problem: Wrong Tool Selected (Low Accuracy)

**Symptom**: `fallback_rate` is low but accuracy is also low

**Cause**: Accepting poor results with false confidence

**Solutions**:
1. **Raise `rag_threshold`**: Try 0.85 or 0.9
2. **Raise `cluster_threshold`**: Try 0.8
3. **Check confidence score distribution**: May need recalibration

### Problem: Unbalanced Method Distribution

**Symptom**: One method dominates (e.g., 90% RAG)

**Analysis**: Check `method_used_distribution` in results

If RAG dominates:
- RAG is working well, consider using RAG directly
- Or raise `rag_threshold` to use fallback more

If MCP dominates:
- Lower thresholds to accept faster methods more often
- Improve RAG/Clustering quality

### Recommended Starting Configuration

```yaml
confidence_config:
  rag_threshold: 0.8
  cluster_threshold: 0.7
  rag_config:
    top_k: 10
  hybrid_config:
    top_k_categories: 3
```

---

## Embedding Model Selection

| Model | Quality | Speed | Memory | Use When |
|-------|---------|-------|--------|----------|
| `all-MiniLM-L6-v2` | Good | Fast | Low | Default, production |
| `all-mpnet-base-v2` | Better | Medium | Medium | Higher quality needed |
| Domain-specific | Best | Varies | Varies | Specialized vocabulary |

### When to Change Models

1. **Low retrieval recall with proper K**: Try `all-mpnet-base-v2`
2. **Domain-specific jargon**: Fine-tune or use domain model
3. **Memory constraints**: Stick with MiniLM

---

## Iterative Tuning Process

### Step 1: Establish Baseline

```bash
python scripts/run_experiment.py run \
  --methodology rag \
  --num-tools 50 \
  --num-test-samples 50 \
  --name baseline_rag
```

### Step 2: Analyze Results

```bash
python scripts/analyze_results.py compare --results-dir experiments/results
```

Check:
- `accuracy`
- `retrieval_recall_rate` (for RAG)
- `category_selection_accuracy` (for Clustering/Hybrid)
- `fallback_rate` (for Confidence)

### Step 3: Identify Bottleneck

| If This Is Low | Tune This |
|----------------|-----------|
| Retrieval recall | Increase K or lower threshold |
| Accuracy (with good recall) | Improve tool descriptions |
| Category accuracy | Improve category descriptions |
| Confidence accepted too easily | Raise thresholds |

### Step 4: Make One Change

Change only one parameter at a time to isolate its effect.

### Step 5: Re-run and Compare

```bash
python scripts/run_experiment.py run \
  --methodology rag \
  --rag-top-k 15 \
  --name tuned_rag_k15
```

### Step 6: Repeat

Continue until metrics stop improving.

---

## Common Pitfalls

### 1. Over-tuning to Test Set

**Problem**: Tuning parameters too specifically to test cases

**Solution**: Use separate validation set or cross-validation

### 2. Ignoring Latency

**Problem**: Achieving high accuracy but unacceptable latency

**Solution**: Track `avg_latency_ms` and set acceptable bounds

### 3. Not Checking Token Usage

**Problem**: High accuracy but excessive token usage

**Solution**: Monitor `avg_tokens_total` and optimize context size

### 4. Assuming One-Size-Fits-All

**Problem**: Using same parameters for different domains

**Solution**: Tune per use case; parameters depend on tool characteristics

---

## Quick Tuning Recipes

### Recipe: High Accuracy Priority

```yaml
methodology: confidence
confidence_config:
  rag_threshold: 0.9
  cluster_threshold: 0.85
  rag_config:
    top_k: 15
    embedding_model: all-mpnet-base-v2
```

### Recipe: Low Latency Priority

```yaml
methodology: rag
rag_config:
  top_k: 5
  similarity_threshold: 0.4
  embedding_model: all-MiniLM-L6-v2
  cache_embeddings: true
```

### Recipe: Large Tool Sets (500+)

```yaml
methodology: adaptive_rag
adaptive_rag_config:
  min_k: 5
  max_k: 25
  similarity_drop_threshold: 0.08
  embedding_model: all-MiniLM-L6-v2
  cache_embeddings: true
```

### Recipe: Well-Organized Categories

```yaml
methodology: hybrid
hybrid_config:
  top_k_categories: 3
  embedding_model: all-MiniLM-L6-v2
  cache_embeddings: true
```
