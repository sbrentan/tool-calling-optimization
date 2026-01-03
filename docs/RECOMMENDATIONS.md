# Tool Calling Optimization - Recommendations and Future Work

**Date:** January 3, 2026  
**Author:** Analysis based on current codebase review

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Testing Improvements](#testing-improvements)
4. [New Methodology Proposals](#new-methodology-proposals)
5. [Infrastructure Improvements](#infrastructure-improvements)
6. [Priority Roadmap](#priority-roadmap)

---

## Executive Summary

Your project provides a solid foundation for analyzing LLM tool-calling performance across three methodologies: Simple MCP, Clustering, and RAG. Based on my analysis, the main opportunities for improvement fall into three categories:

1. **Testing Gaps**: Current tests focus on single-tool calls; need multi-tool, no-tool, and adversarial scenarios
2. **Methodology Extensions**: Several promising new approaches can address the accuracy-context tradeoff differently
3. **Evaluation Depth**: Metrics could be enriched to better capture real-world performance characteristics

---

## Current State Analysis

### Methodology Comparison

| Methodology | Context Usage | Accuracy Profile | Main Challenge |
|-------------|---------------|------------------|----------------|
| **Simple MCP** | High (all tools in context) | High for small tool sets | Context window limits at scale |
| **Clustering** | Low (categories → tools) | Variable | Category disambiguation; multi-step latency |
| **RAG** | Medium (top-k tools) | Depends on embedding quality | Semantic gaps; requires tuning k |

### Identified Weaknesses

1. **Clustering Issues** (from experiment results):
   - Category selection failures (e.g., `send_digest_notification` → `analytics_operations` instead of `notification_operations`)
   - Ambiguous category boundaries (data ops vs. database ops vs. analytics ops)
   - "No tool selected in step" errors when LLM doesn't match category prompts

2. **RAG Limitations**:
   - Fixed top-k may miss relevant tools or include too many irrelevant ones
   - Embedding model (MiniLM) may not capture domain-specific semantics well
   - No fallback when retrieved tools don't match intent

3. **Testing Coverage**:
   - Only single-tool test cases currently validated
   - No "negative" tests (when no tool should be called)
   - No parameter extraction accuracy evaluation in experiments

---

## Testing Improvements

### 1. Multi-Tool Test Cases

**Problem**: Real-world requests often require multiple tools in sequence or parallel.

**Implementation Approach**:

```python
@dataclass
class MultiToolTestCase:
    prompt: str
    expected_tools: list[str]  # Ordered list of expected tools
    expected_sequence: bool = True  # False = parallel execution acceptable
    category: str = "multi_tool"
    
# Example test cases:
# "Download the image from URL and resize it to 800x600"
#   → [download_file, resize_image]
# "Search for users named John and export the results to CSV"
#   → [search_users, export_csv]
```

**Key Metrics to Track**:
- **Sequence Accuracy**: Did the LLM call tools in the correct order?
- **Completion Rate**: Were all required tools called?
- **Extra Calls**: Did the LLM call unnecessary tools?

### 2. No-Tool-Required Test Cases (Negative Tests)

**Problem**: LLMs sometimes force tool calls when none is needed.

**Types of Negative Tests**:

| Type | Example Prompt | Expected Behavior |
|------|----------------|-------------------|
| **Informational** | "What does the send_email function do?" | No tool call; explain the tool |
| **Clarification** | "I want to do something with files" | No tool call; ask for clarification |
| **Out-of-Scope** | "What's the weather in Paris?" | No tool call; indicate inability |
| **Conversational** | "Thanks for your help!" | No tool call; respond naturally |

**Implementation**:

```python
@dataclass  
class NoToolTestCase:
    prompt: str
    expected_tool: None = None  # Explicitly no tool expected
    category: str = "no_tool"
    reason: str = ""  # Why no tool should be called
```

**Metrics**:
- **False Positive Rate**: How often does LLM call a tool when it shouldn't?
- **Appropriate Response Rate**: Did LLM respond sensibly without calling a tool?

### 3. Adversarial/Edge Case Tests

**Purpose**: Stress-test tool selection under difficult conditions.

| Test Type | Description | Example |
|-----------|-------------|---------|
| **Ambiguous Intent** | Prompt matches multiple tools equally well | "Process the data" (transform? validate? parse?) |
| **Similar Tools** | Nearly identical tools in context | `send_email` vs `send_notification_email` |
| **Missing Tool** | Request requires a tool not in context | "Fax the document" when no fax tool exists |
| **Partial Match** | Prompt partially matches a tool's purpose | "Almost like encryption but not quite" |
| **Typos/Informal** | Realistic user input quality | "plz resize my pic" |

### 4. Parameter Extraction Testing

**Current Gap**: `params_correct` is tracked but not actively tested.

**Recommendations**:
- Include expected parameters in test cases
- Test parameter type coercion (string → int, etc.)
- Test optional vs required parameter handling
- Test parameter validation (enums, ranges)

```python
TestCase(
    prompt="Send an email to john@example.com with subject 'Hello'",
    expected_tool="send_email",
    expected_params={
        "to": "john@example.com",
        "subject": "Hello"
    }
)
```

---

## New Methodology Proposals

### 1. Hybrid RAG + Clustering

**Concept**: Combine the best of both approaches.

**How It Works**:
1. Use embeddings to select top-k **categories** (not individual tools)
2. Retrieve all tools from selected categories
3. LLM selects specific tool from narrowed set

**Advantages**:
- More robust category selection (semantic similarity vs. name matching)
- Reduced context vs. full MCP
- Avoids the "wrong category" problem in pure clustering

**Implementation Sketch**:

```python
class HybridMethodology(BaseMethodology):
    def run_single(self, prompt, tools, client):
        # 1. Embed categories (aggregate tool embeddings per category)
        category_embeddings = self._compute_category_embeddings(tools)
        
        # 2. Find top-k categories by semantic similarity
        query_embedding = self._embed_query(prompt)
        top_categories = self._find_top_categories(query_embedding, k=3)
        
        # 3. Get all tools from those categories
        relevant_tools = [t for t in tools if t.category in top_categories]
        
        # 4. LLM selects from relevant tools
        return self._llm_select_tool(prompt, relevant_tools, client)
```

### 2. Adaptive RAG (Dynamic Top-K)

**Problem**: Fixed top-k is suboptimal — simple queries need fewer tools, complex ones need more.

**Solution**: Dynamically adjust k based on:
- Query complexity (number of concepts mentioned)
- Embedding similarity distribution (if scores cluster, use fewer)
- Historical accuracy for similar queries

**Implementation Approach**:

```python
class AdaptiveRAGMethodology(RAGMethodology):
    def _determine_k(self, query: str, similarities: np.ndarray) -> int:
        # Strategy 1: Elbow method on similarity scores
        sorted_sims = np.sort(similarities)[::-1]
        gradient = np.diff(sorted_sims)
        elbow_point = np.argmax(gradient < self.threshold) + 1
        
        # Strategy 2: Include all above threshold
        above_threshold = np.sum(similarities > self.min_similarity)
        
        # Strategy 3: Query complexity heuristic
        word_count = len(query.split())
        complexity_k = min(max(3, word_count // 3), 15)
        
        return min(elbow_point, above_threshold, complexity_k)
```

### 3. Two-Stage LLM Selection

**Concept**: Use a fast, cheap LLM for filtering, then accurate LLM for selection.

**Architecture**:
```
Query → [Fast LLM: "Which 5 tools are relevant?"] → [Accurate LLM: "Select & call tool"]
```

**Advantages**:
- Better than pure embedding (LLM understands context)
- Cheaper than full MCP with expensive model
- Can use different model sizes for each stage

**Trade-offs**:
- Two LLM calls = higher latency
- Fast LLM might filter out correct tool

### 4. Tool Description Rewriting

**Problem**: Tool descriptions may not match user query vocabulary.

**Solution**: Pre-process tools to create multiple description variants:

```python
class RewrittenToolMethodology:
    def prepare_tools(self, tools: list[Tool]) -> list[Tool]:
        rewritten = []
        for tool in tools:
            # Use LLM to generate alternative descriptions
            variants = self._generate_description_variants(tool)
            # Create searchable index with all variants
            tool.searchable_descriptions = [tool.description] + variants
        return tools
```

**Variants to Generate**:
- Task-oriented: "Use this when you need to..."
- Example-based: "For example, if a user asks to..."  
- Keyword-rich: "Keywords: send, email, message, notify..."

### 5. Confidence-Based Fallback

**Problem**: All methodologies can fail silently with a wrong tool selection.

**Solution**: Implement confidence scoring with fallback chains.

```python
class ConfidenceMethodology(BaseMethodology):
    def run_single(self, prompt, tools, client):
        # Try RAG first (fastest)
        rag_result = self.rag.run_single(prompt, tools, client)
        if rag_result.confidence > 0.8:
            return rag_result
        
        # Fall back to clustering
        cluster_result = self.clustering.run_single(prompt, tools, client)
        if cluster_result.confidence > 0.7:
            return cluster_result
        
        # Fall back to full MCP (most accurate but expensive)
        return self.mcp.run_single(prompt, tools, client)
```

### 6. Tool Summarization for Large Tool Sets

**Problem**: With 100+ tools, even category descriptions become too long.

**Solution**: Generate compressed tool index.

```python
class SummarizedMCPMethodology:
    def prepare_context(self, tools: list[Tool]) -> str:
        # Create compact index: one line per tool
        index = "Available tools:\n"
        for tool in tools:
            index += f"- {tool.name}: {tool.description[:50]}...\n"
        
        # Add instruction for two-step process
        return f"""
        {index}
        
        First, identify which tool(s) might be relevant.
        Then I'll provide full details for those tools.
        """
```

---

## Infrastructure Improvements

### 1. Enhanced Metrics

**New Metrics to Track**:

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Tokens Used** | Input + output tokens per call | Cost optimization |
| **Confidence Score** | LLM's certainty in selection | Quality indicator |
| **Reasoning Quality** | Did LLM explain choice correctly? | Debugging insight |
| **Category Confusion Matrix** | Which categories get confused? | Clustering improvement |
| **Retrieval Recall@K** | Was correct tool in top-k? | RAG tuning |

### 2. Experiment Framework Enhancements

**A. Comparative Mode**:
Run same test cases across all methodologies simultaneously:

```python
python scripts/run_experiment.py compare \
    --methodologies mcp,clustering,rag,hybrid \
    --num-tools 50 \
    --output comparison_report.html
```

**B. Ablation Studies**:
Systematically vary one parameter while holding others constant:

```yaml
# experiments/configs/ablation_rag_k.yaml
ablation:
  methodology: rag
  vary_parameter: top_k
  values: [3, 5, 10, 15, 20]
  hold_constant:
    num_tools: 50
    embedding_model: all-MiniLM-L6-v2
```

**C. Statistical Significance Testing**:
Add confidence intervals and significance tests to comparisons.

### 3. Visualization Dashboard

Create automated visualizations:
- Accuracy vs. context size curves
- Latency distributions
- Category confusion matrices
- Tool similarity heatmaps
- Parameter extraction accuracy breakdown

### 4. Test Case Management

**Improvements**:
- Separate test case definitions from tool definitions
- Create a test case database with tagging
- Support for test case versioning
- Easy addition of manually curated test cases

```yaml
# tests/cases/multi_tool.yaml
test_cases:
  - id: multi_001
    prompt: "Download image.png and resize it to 800x600"
    expected_tools: [download_file, resize_image]
    tags: [multi_tool, file_ops, image_ops]
    difficulty: medium
    added_date: 2026-01-03
```

---

## Priority Roadmap

### Phase 1: Testing Foundation (1-2 weeks)

1. **Implement No-Tool Test Cases**
   - Add `NoToolTestCase` class
   - Create 20+ no-tool test scenarios
   - Update evaluator to handle no-tool expectation

2. **Enable Multi-Tool Testing**
   - Implement `MultiToolTestCase` evaluation
   - Use existing `include_multi_tool` parameter properly
   - Track sequence and completeness metrics

3. **Add Parameter Accuracy Tests**
   - Include expected_params in test cases
   - Implement parameter comparison logic
   - Report parameter-level metrics

### Phase 2: Methodology Extensions (2-3 weeks)

4. **Implement Hybrid RAG + Clustering**
   - Build category embedding index
   - Create `HybridMethodology` class
   - Benchmark against pure approaches

5. **Add Adaptive Top-K to RAG**
   - Implement dynamic k selection
   - Compare fixed vs adaptive performance
   - Tune thresholds

6. **Implement Confidence Scoring**
   - Extract confidence from LLM responses
   - Create fallback chain methodology
   - Measure when fallback improves accuracy

### Phase 3: Analysis & Polish (1-2 weeks)

7. **Enhance Metrics Collection**
   - Track token usage
   - Calculate retrieval recall
   - Generate confusion matrices

8. **Create Comparison Framework**
   - Side-by-side methodology comparison
   - Statistical significance testing
   - Automated report generation

9. **Documentation & Examples**
   - Document all methodologies
   - Provide tuning guidelines
   - Create example notebooks

---

## Appendix: Implementation Notes

### Quick Win: Fix Category Prompts

Based on experiment results showing category selection failures, consider:

1. **Better Category Descriptions** in `categories.yaml`:
   ```yaml
   - name: notification_operations
     description: >
       Push notifications, alerts, digests, and activity summaries.
       Use for: sending alerts, weekly digests, notification preferences.
       NOT for: emails (use email_operations), SMS (use messaging_operations).
   ```

2. **Negative Examples in Prompts**:
   Tell the LLM what each category is NOT for.

3. **Cross-Reference Table**:
   For commonly confused categories, add explicit disambiguation.

### Quick Win: Improve RAG Embeddings

1. **Use Domain-Specific Model**:
   Try `sentence-transformers/all-mpnet-base-v2` (better quality)
   Or fine-tune on tool-query pairs

2. **Query Expansion**:
   Before embedding the query, expand it:
   ```python
   expanded_query = llm.generate(f"Rephrase this request: {query}")
   ```

3. **Include Examples in Tool Text**:
   ```python
   tool_text = f"{tool.name}: {tool.description}. Example uses: {examples}"
   ```

---

## Conclusion

The project has a solid foundation. The main recommendations are:

1. **Immediate**: Fix testing coverage (no-tool, multi-tool, parameters)
2. **Short-term**: Implement Hybrid methodology to combine RAG and Clustering strengths
3. **Medium-term**: Add confidence-based fallback for production robustness

These improvements will strengthen both the "Experiments & Analysis" and "Design & Prototypes" dimensions of the grading rubric, demonstrating thoughtful iteration on design choices with supporting evidence.
