# Optimizing Tool Calling at Scale for LLM-based Agents

## Abstract

This project investigates strategies to optimize tool calling in large-scale LLM-based agents.
As the number of available tools increases, naive approaches that load all tool definitions into the model context become infeasible due to token budget constraints, latency, and cost.
We explore various tool selection and loading strategies to minimize these issues while maintaining or improving the correctness and reliability of tool usage.
Our experiments demonstrate that intelligent tool selection mechanisms can significantly reduce context load and improve performance, enabling scalable deployment of LLM agents in real-world applications.
In particular, implemented tool selection strategies include naive context loading, clustering-based two-step selection, hybrid rag-based category selection, rag-based tool selection and its variant adaptive rag-based selection.


## Introduction

Large Language Models (LLMs) have revolutionized the capabilities of AI agents by enabling them to perform complex tasks through tool usage, such as calling APIs, functions, or plugins.
However, as the ecosystem of available tools expands to hundreds or thousands, traditional methods that preload all tool definitions into the model's context become impractical.
This project aims to address the challenges associated with scaling tool calling in LLM-based agents by exploring and evaluating various tool selection and loading strategies.

### Problem Statement

When the tool ecosystem is large and tool documentation is verbose, naive approaches that preload all tool definitions into the model context (MCP-style) become infeasible: they blow the token budget, increase latency and cost, and reduce robustness.
We study selection and loading strategies that minimize token/compute cost and latency while preserving (or improving) correctness and reliability of tool usage.

### Motivation

Real-world deployments (enterprise skill stores, platform plugins, multi-tenant APIs) contain many overlapping implementations (multiple mail providers, multiple browsing adapters, etc.).
Efficient and accurate tool selection reduces operational cost, improves uptime and user experience, and is fundamental for scaling agent systems in production.

### Methodologies Explored
We conduct a series of experiments to evaluate different tool selection strategies under varying conditions of tool count and documentation verbosity.
The study includes the following configurations:
- **Naive Context Loading**: All tool definitions are loaded into the model context, serving as a baseline for comparison.
- **Clustering-Based Two-Step Selection**: Tools are clustered, and a two-step selection process is employed, first selecting a cluster and then a specific tool within that cluster.
- **Hybrid RAG-Based Category Selection**: A hybrid approach using Retrieval-Augmented Generation (RAG) for category selection to inject into context.
- **RAG-Based Tool Selection**: Directly selecting top-k tools using RAG to minimize context load.
- **Adaptive RAG-Based Selection**: An adaptive variant of RAG-based selection that dynamically adjusts the number of tools selected based on the query.

### Experiments & Results

* Assumption: tools do not need to work but need to be correctly selected (allowed us to simply define the tools as simple yamls files and not implement them).
* Tools definition:
    * Different tools divided into categories.
    * Each tool has a name, description with different levels of verbosity, parameters, prompts (clear and concise) that should invoke this tool.
    * Similar tools to introduce ambiguity (e.g. manage_connection_pool, manage_database_connection_pool).
* Datasets:
    * Created synthetic dataset of tools as defined above.
    * Secondary public dataset to validate synthetic dataset results on more accurate real-world data.
* Different LLMs used:
    * LLama 3 70B on cloud
        * context of 64k tokens
        * free APIs, therefore limited amount of tools tested per test (10 samples x 3 times with different seeds for each configuration).
    * Llama 2 3B locally with Ollama:
        * context of 4k tokens
        * allowed us for more precise testing by testing all tools loaded from configuration files.
* Metrics:
    * Tool Selection Accuracy: Percentage of correctly selected tools.
    * Latency: Time taken to select and invoke tools.
    * Context Size: Number of tokens used in the model context.
    * Category Selection Accuracy (for clustering and hybrid rag-based approaches).

[EDIT]: add more ?



## Methodologies [EDIT]: Or change to Implementation?

In this section, we detail the various tool selection strategies implemented and evaluated in our experiments.
For each methodology, we describe the approach, its implementation, and the reasoning behind its design.
After each introduction, we test the methodology through a series of experiments to assess its performance in terms of tool selection accuracy, latency, and context size, aiming to find its limitations.

In this section, the results shown are based on experiments conducted with Llama 3 70B model on cloud. 

### Naive Context Loading

The naive context loading approach involves preloading all tool definitions into the model's context.
This is the simplest method but is also the style commonly used by the MCP protocol.
This method serves as a baseline for comparison against more sophisticated selection strategies.

[EDIT]: add more details about the imp

Different experiments are conducted with an increasing number of tools:
* 10 tools
* 25 tools
* 50 tools
* 100 tools
* 200 tools
* 300 tools
* 350 tools

It was impossible to test with 400 tools due to context window limitations of the LLM used. In fact, with 400 tools the context size exceeded the model's maximum token limit and all API requests failed.

#### Results
[IMAGE PLACEHOLDER: MCP accuracy]
[IMAGE PLACEHOLDER: MCP token usage]

These images show how the accuracy of MCP is high and stable until the context window limit is approached.
However, as the number of tools increases beyond approximately 300, accuracy begins to degrade significantly with 350 tools.

The token usage graph shows a linear increase in context size as the number of tools increases, leading to practical limitations in scalability.
The last test configuration with 350 tools resulted in a context size of approximately 60k tokens, which is close to the maximum limit of the LLM used (64k tokens). No further tests could be conducted beyond this point.

### Clustering-Based Two-Step Selection

In order to address the scalability issues observed with naive context loading, we implemented a clustering-based two-step selection strategy.
This approach involves first clustering tools into categories based on their functionality, and then performing a two-step selection process:
1. **Category Selection**: The model first selects the most relevant category of tools based on the user query.
2. **Tool Selection**: Within the selected category, the model then selects the specific tool to invoke.

This method aims to reduce the context size by only loading tools from the selected category into the model's context.
However, this approach introduces complexity in accurately selecting the correct category, which can impact overall tool selection accuracy.
In addition, compared to naive context loading, this method requires two separate calls to the LLM: one for category selection and another for tool selection within that category, potentially increasing latency.

For this methodology, tests included tools ranging from 10 to 200 tools, similarly to the naive context loading, and then two additional layers of tests with respectively 500 and 985 tools.

> Note: Altough this methodology involves clustering tools, in this project we did not implement an automatic clustering algorithm but we manually defined the categories and assigned tools to them based on their functionality. This is a simplification to focus on evaluating the two-step selection process itself rather than the clustering quality.

#### Results
[IMAGE PLACEHOLDER: Clustering accuracy]
[IMAGE PLACEHOLDER: Clustering token usage]

The graphs indicate that the clustering-based approach:
1. drastically reduces the context size used compared to naive context loading, as only tools from the selected category are loaded.
2. however, it also significantly reduces tool selection accuracy. The two-step process introduces errors, particularly in the category selection phase, which results in a much lower tool selection accuracy compared to the naive context loading.

The accuracy degradation is primarily due to the model's difficulty in correctly identifying the relevant category from the user query.
The accuracy remains relatively stable and low about 40-50% across different tool counts due to how the tests have been designed. Even when creating a test with only 10 tools, they are chosen from different categories, therefore the model still needs to correctly identify the category first, which is challenging.

[IMAGE PLACEHOLDER: Clustering Category Selection Confusion Matrix]

The confusion matrix above illustrates the model's performance in selecting the correct category.
It shows that certain categories are frequently confused with others, leading to incorrect tool selections downstream.

> Note: only the most used 15 categories are shown in the confusion matrix for better clarity.

#### Additional design choices

The category selection prompt defines the list of categories available for selection as a list of tools that the model can call.
For example, if we have 3 categories: "AI ML Operations", "Analytics Operations" and "Database Operations", the prompt will define these categories as 3 tools with their respective descriptions:
* select_category_ai_ml_operations: "AI and machine learning operations including text generation, classification, embeddings, model inference, and natural language processing"
* select_category_analytics_operations: "Data analytics, statistics, reporting, dashboards, and business intelligence operations"
* select_category_database_operations: "Database queries, CRUD operations, schema management, migrations, and data retrieval from SQL/NoSQL databases"

To further improve the category selection accuracy, a `backtrack` mechanism has been implemented. To the list of categories defined as tools in the prompt, an additional tool called `__backtrack__` is added that enables the model to return back to the previous step if it realizes that the selected category was incorrect. A maximum number of 10 max steps (5 backtracks) is allowed to avoid infinite loops. The list of previously selected categories is also provided in the prompt to avoid repeating the same mistakes.

### Hybrid RAG-Based Category Selection

After seeing the limitations of the clustering-based two-step selection, particularly in terms of category selection accuracy, we explored a hybrid approach that leverages Retrieval-Augmented Generation (RAG) for category selection.

In this hybrid RAG-based category selection method, the model uses RAG to select the most relevant categories based on the user query.
The selected categories are then injected into the model context as tools, similar to the clustering-based approach.
This allows the model to focus on a smaller subset of tools while benefiting from the retrieval capabilities of RAG.

> Note: Hybrid between full clustering-based and full RAG-based tool selection, as this methodology uses RAG to select the category, and then uses the selected category to load tools into context (like clustering-based two-step selection).

This methodology introduces an additional layer of complexity, as it relies on the effectiveness of RAG in accurately retrieving relevant categories.
The RAG component for category selection uses a vector store built from the tool categories' descriptions.
When a user query is received, the RAG system retrieves the top-k most relevant categories based on similarity to the query.

For this methodology, tests included tools ranging from 10 to 985 tools, similarly to the clustering-based two-step selection.


##### Results
[IMAGE PLACEHOLDER: Hybrid RAG accuracy]
[IMAGE PLACEHOLDER: Hybrid RAG token usage]

The results indicate that the hybrid RAG-based category selection approach greatly improves tool selection accuracy compared to the pure clustering-based method, coming closer to the performance of naive context loading.
However, the context size remains significantly lower than that of naive context loading, demonstrating the effectiveness of this approach in balancing accuracy and context efficiency.

[IMAGE PLACEHOLDER: Hybrid RAG Category Selection Confusion Matrix]
The confusion matrix for category selection in the hybrid RAG-based approach shows improved performance over the pure clustering method.
While some categories are still confused, the overall accuracy is higher, leading to better downstream tool selection accuracy.

A further analysis of this methodology involved the impact of changing the number of categories retrieved by RAG (k) on overall tool selection accuracy.
[IMAGE PLACEHOLDER: Hybrid RAG k impact on accuracy]
The graph illustrates that increasing the number of categories retrieved (k) generally leads to improved tool selection accuracy, as the model has access to a broader set of relevant tools. While generally all tests indicated that higher k leads to better accuracy, the major accuracy improvements are observed when increasing k from 3 to 5 for the 985 tools configuration.
This suggests that there is a trade-off between context size and accuracy, as retrieving too many categories may lead to increased context load without significant accuracy gains. Having categories generally have from 30 to 40 tools each, therefore retrieving 5 categories means loading approximately 150-200 tools into context, which is the naive context load managed with high accuracy. The optimal k value may vary depending on the specific toolset, tool categorization and model used, but this analysis is out of the scope of this project and may be explored in future work.


#### Additional design choices

[EDIT: add some more details about RAG implementation and how it works? Maybe talk about that greater category accuracy is also due to rag being able to see the tools in the category descriptions when creating the vector store? Also, talk about single embedding model used and other testing could be future work?]

### RAG-Based Tool Selection

Building upon the insights gained from the hybrid RAG-based category selection, we implemented a full RAG-based tool selection strategy.
With this approach, we aim to skip the category selection step and directly select the most relevant tools using RAG, thereby minimizing context load while maximizing tool selection accuracy. Context load is minimized by only injecting the top-k tools retrieved by RAG into the model context, instead of loading all tools for a subset of categories. Additionally, this methodology aims to retrieve tools that may belong to different categories but are all relevant to the user query.

For this methodology, tests included tools ranging from 10 to 985 tools, similarly to the previous Clustering and Hybrid methodologies.
The RAG component for tool selection uses a vector store built from the tool definitions, including their names and descriptions.
When a user query is received, the RAG system retrieves a fixed number of the most relevant tools based on similarity to the query. The number of tools retrieved (k) is depended on the specific test configuration and is chosen to balance context size and accuracy.

> Note: The RAG system uses the same embedding model and parameters as the hybrid RAG-based category selection for consistency across experiments.

#### Results

[IMAGE PLACEHOLDER: RAG accuracy]
[IMAGE PLACEHOLDER: RAG token usage]

The results indicate that the RAG-based tool selection approach achieves an higher tool selection accuracy compared to both the clustering-based and hybrid RAG-based methods, outperforming even the naive context loading in lower-tools configurations.
The context size used is significantly lower than that of naive context loading, and is also lower than both the clustering-based and hybrid RAG-based approaches, demonstrating the effectiveness of this method in balancing accuracy and context efficiency.

The accuracy remains relatively high across different tool counts, although there is a gradual decrease as the number of tools increases.
The context size remains stable and low, even lower than the clustering approach, as only the top-k relevant tools are loaded into context and not all tools from selected categories.

Two additional analysis were conducted in this methodology:
1. Impact of changing the number of tools retrieved by RAG (k) on overall tool selection accuracy.
2. Impact of changing the verbosity level of tool descriptions on overall tool selection accuracy.

[IMAGE PLACEHOLDER: RAG k impact on accuracy]
[IMAGE PLACEHOLDER: RAG token usage k impact]
The first graph illustrates that increasing the number of tools retrieved (k), in these test configurations, did not lead to significant improvements in tool selection accuracy. Particularly, beyond k=15, accuracy was observed to be lower, probably due to the model being confused by the larger number of similar tools provided in context. In these tests, k values of 10 and 15 seem to be optimal, but this may vary depending on the specific toolset used.

The second graph shows that increasing k leads to a fixed increase in context size, as more tools are loaded into the model context. However, the context size occupied remains fixed also when increasing the total number of tools available, as only the top-k tools are loaded regardless of the total tool count. Being approximately 5500 tokens for k=30 and around 2700 tokens for k=15, it is manageable even for models with smaller context windows.

[IMAGE PLACEHOLDER: RAG verbosity impact on accuracy]
[IMAGE PLACEHOLDER: RAG token usage verbosity impact]

These graphs instead illustrates that increasing the verbosity level of tool descriptions has a positive impact on tool selection accuracy, while maintaining a relatively low context size.
Higher verbosity provides the model with more detailed information about each tool and its parameters examples, enabling better understanding and selection.


### Adaptive RAG-Based Selection

Building upon the RAG-based tool selection, we implemented an adaptive variant that dynamically adjusts the number of tools selected based on the user query.
The adaptive RAG-based selection method uses a heuristic to determine the optimal number of tools (k) to retrieve for each query, aiming to balance context size and tool selection accuracy more effectively.
The heuristic considers factors such as query complexity and ambiguity to decide how many tools to retrieve.

For this methodology, tests included tools ranging from 10 to 985 tools, similarly to the previous RAG-based tool selection.

[EDIT]: add more details about the heuristic used to adaptively select k? And check previous paragraph for correctness.

> Note: This approach is more of a variant of the RAG-based tool selection rather than a completely new methodology, as it builds upon the same RAG framework but introduces adaptivity in the selection process. It was introduced to explore whether dynamically adjusting k could yield better performance compared to a fixed k value.

#### Results

[EDIT]: talk about small performance improvements and similar context size (a bit on the lower end) compared to fixed k RAG-based selection. show average input tokens chart not linear with chainging num tools. Talk about test done clear vs concise prompts -> big improvement on higher tools configs (but not tested enough to be conclusive). Chart of distribution of adaptive k values?
        

## Experiments & Analysis

[EDIT]: Summarize the experiments conducted for each methodology, the metrics used for evaluation, and the key findings from the analysis. Show general 


Considerations:
* more difficult to handle the clustering prompt as the llm a lot of times fails to understand the categories selection process. Probably it could be improved, but it is out of the scope of this project.
* Overall charts shown for each methodology are all created based on aggregated data from a set of 130+ experiments. While not assuring all methodologies were tested with exactly the same number of tools/configurations, the trends shown are representative of the general performance observed for each approach. For example, verbosity level tests are shown only in RAG-based methodology, but they are included in all methodology tests, as the results are all aggregated together.
* Majority of tests conducted with prompt clarity set to concise. Some tests with clear prompt were also conducted (mainly in adaptive rag-based methodology) and they showed improved accuracy when the prompt was more clear. A more extensive analysis of prompt clarity on the other methodologies and its impact on accuracy could be a future work.




* Note for me: check also about notool/similar tools, and other info to be added here. An example of a tool? Also add in another section a sort of list of the configs? Also section about problems with cloud/local LLMs? -> Probably in experimentation section?


