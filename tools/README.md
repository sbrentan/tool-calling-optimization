# Tool Definitions

This directory contains YAML files defining tools for testing LLM tool-calling accuracy.

## Structure

Each YAML file represents a **category** of related tools:
- `file_operations.yaml` - File system operations
- `data_operations.yaml` - Database and data manipulation
- `math_operations.yaml` - Calculations and conversions
- `web_operations.yaml` - HTTP, email, web search
- `text_operations.yaml` - Text processing and NLP
- `system_operations.yaml` - System commands and monitoring

## Schema

```yaml
category: category_name
tools:
  - name: tool_name              # Unique identifier (snake_case)
    
    # Descriptions at different verbosity levels
    descriptions:
      minimal: "Brief description"
      short: "One sentence description"
      medium: "Detailed description with basic usage"
      long: "Full description with examples"
      verbose: "Comprehensive docs with examples, notes, warnings"
    
    # Parameters following JSON Schema
    parameters:
      - name: param_name
        type: string|integer|number|boolean|array
        description: "Parameter description"
        required: true|false
        enum: ["option1", "option2"]  # Optional
        default: "default_value"       # Optional
    
    # Tags for grouping and similarity
    tags: ["tag1", "tag2"]
    
    # Test prompts
    test_prompts:
      # Single tool invocation
      single:
        - "Natural language prompt that should trigger this tool"
        - "Alternative prompt for the same tool"
      
      # Multi-tool scenarios (for future use)
      multi:
        - prompt: "Prompt requiring multiple tools"
          required_tools: ["tool1", "tool2"]
          order_matters: false
    
    # For generating similar/confusing tools
    similar_variants:
      - suffix: "_v2"
        description_modifier: "Alternative version that"
      - suffix: "_async"
        description_modifier: "Asynchronous version that"
```

## Adding New Tools

1. Find the appropriate category file (or create a new one)
2. Add your tool following the schema above
3. Include meaningful test prompts that clearly require this tool
4. Add tags for future similarity grouping

## Best Practices

- **Descriptions**: Write realistic descriptions at each level
- **Test prompts**: Make them specific enough that only this tool fits
- **Parameters**: Include realistic parameters with good descriptions
- **Tags**: Use consistent tags across tools for similarity grouping
