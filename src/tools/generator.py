"""
Tool generator for creating synthetic tools with configurable parameters.

This module generates tools with varying:
- Number of tools
- Documentation length (minimal to verbose)
- Parameter complexity
- Semantic similarity between tools
"""
import random
from typing import Optional
from .base import Tool, ToolParameter, TestCase


# Tool templates organized by category
TOOL_TEMPLATES = {
    "file_operations": [
        {
            "name": "read_file",
            "base_desc": "Read contents of a file",
            "params": [
                {"name": "file_path", "type": "string", "desc": "Path to the file"},
                {"name": "encoding", "type": "string", "desc": "File encoding", "required": False}
            ],
            "tags": ["file", "read", "io"],
            "test_prompt": "Read the contents of config.json"
        },
        {
            "name": "write_file",
            "base_desc": "Write content to a file",
            "params": [
                {"name": "file_path", "type": "string", "desc": "Path to the file"},
                {"name": "content", "type": "string", "desc": "Content to write"},
                {"name": "append", "type": "boolean", "desc": "Append instead of overwrite", "required": False}
            ],
            "tags": ["file", "write", "io"],
            "test_prompt": "Write 'Hello World' to output.txt"
        },
        {
            "name": "delete_file",
            "base_desc": "Delete a file from the filesystem",
            "params": [
                {"name": "file_path", "type": "string", "desc": "Path to the file to delete"},
                {"name": "force", "type": "boolean", "desc": "Force deletion without confirmation", "required": False}
            ],
            "tags": ["file", "delete", "io"],
            "test_prompt": "Delete the temporary file temp.txt"
        },
        {
            "name": "list_directory",
            "base_desc": "List files in a directory",
            "params": [
                {"name": "directory_path", "type": "string", "desc": "Path to the directory"},
                {"name": "recursive", "type": "boolean", "desc": "List recursively", "required": False},
                {"name": "pattern", "type": "string", "desc": "Filter pattern", "required": False}
            ],
            "tags": ["file", "directory", "list", "io"],
            "test_prompt": "Show me all files in the src folder"
        },
        {
            "name": "copy_file",
            "base_desc": "Copy a file to a new location",
            "params": [
                {"name": "source_path", "type": "string", "desc": "Source file path"},
                {"name": "destination_path", "type": "string", "desc": "Destination file path"},
                {"name": "overwrite", "type": "boolean", "desc": "Overwrite if exists", "required": False}
            ],
            "tags": ["file", "copy", "io"],
            "test_prompt": "Copy config.json to config.backup.json"
        },
    ],
    "data_operations": [
        {
            "name": "query_database",
            "base_desc": "Execute a database query",
            "params": [
                {"name": "query", "type": "string", "desc": "SQL query to execute"},
                {"name": "database", "type": "string", "desc": "Database name"},
                {"name": "limit", "type": "integer", "desc": "Maximum rows to return", "required": False}
            ],
            "tags": ["database", "query", "sql"],
            "test_prompt": "Get all users from the users table"
        },
        {
            "name": "insert_record",
            "base_desc": "Insert a record into the database",
            "params": [
                {"name": "table", "type": "string", "desc": "Table name"},
                {"name": "data", "type": "string", "desc": "JSON data to insert"},
                {"name": "database", "type": "string", "desc": "Database name", "required": False}
            ],
            "tags": ["database", "insert", "sql"],
            "test_prompt": "Add a new user named John to the users table"
        },
        {
            "name": "update_record",
            "base_desc": "Update records in the database",
            "params": [
                {"name": "table", "type": "string", "desc": "Table name"},
                {"name": "data", "type": "string", "desc": "JSON data with updates"},
                {"name": "condition", "type": "string", "desc": "WHERE condition"},
                {"name": "database", "type": "string", "desc": "Database name", "required": False}
            ],
            "tags": ["database", "update", "sql"],
            "test_prompt": "Update the email of user with id 5"
        },
        {
            "name": "delete_record",
            "base_desc": "Delete records from the database",
            "params": [
                {"name": "table", "type": "string", "desc": "Table name"},
                {"name": "condition", "type": "string", "desc": "WHERE condition"},
                {"name": "database", "type": "string", "desc": "Database name", "required": False}
            ],
            "tags": ["database", "delete", "sql"],
            "test_prompt": "Remove all inactive users from the database"
        },
    ],
    "math_operations": [
        {
            "name": "calculate",
            "base_desc": "Perform a mathematical calculation",
            "params": [
                {"name": "expression", "type": "string", "desc": "Math expression to evaluate"},
                {"name": "precision", "type": "integer", "desc": "Decimal precision", "required": False}
            ],
            "tags": ["math", "calculate", "arithmetic"],
            "test_prompt": "Calculate 15 * 23 + 47"
        },
        {
            "name": "convert_units",
            "base_desc": "Convert between units of measurement",
            "params": [
                {"name": "value", "type": "number", "desc": "Value to convert"},
                {"name": "from_unit", "type": "string", "desc": "Source unit"},
                {"name": "to_unit", "type": "string", "desc": "Target unit"}
            ],
            "tags": ["math", "convert", "units"],
            "test_prompt": "Convert 100 kilometers to miles"
        },
        {
            "name": "statistics",
            "base_desc": "Calculate statistical measures",
            "params": [
                {"name": "numbers", "type": "string", "desc": "Comma-separated list of numbers"},
                {"name": "operation", "type": "string", "desc": "Statistical operation", 
                 "enum": ["mean", "median", "mode", "std", "variance"]}
            ],
            "tags": ["math", "statistics", "analysis"],
            "test_prompt": "Find the average of 10, 20, 30, 40, 50"
        },
    ],
    "web_operations": [
        {
            "name": "fetch_url",
            "base_desc": "Fetch content from a URL",
            "params": [
                {"name": "url", "type": "string", "desc": "URL to fetch"},
                {"name": "method", "type": "string", "desc": "HTTP method", "enum": ["GET", "POST", "PUT", "DELETE"], "required": False},
                {"name": "headers", "type": "string", "desc": "JSON headers", "required": False}
            ],
            "tags": ["web", "http", "fetch"],
            "test_prompt": "Get the content from https://api.example.com/data"
        },
        {
            "name": "send_email",
            "base_desc": "Send an email message",
            "params": [
                {"name": "to", "type": "string", "desc": "Recipient email address"},
                {"name": "subject", "type": "string", "desc": "Email subject"},
                {"name": "body", "type": "string", "desc": "Email body content"},
                {"name": "cc", "type": "string", "desc": "CC recipients", "required": False}
            ],
            "tags": ["email", "send", "communication"],
            "test_prompt": "Send an email to john@example.com about the meeting"
        },
        {
            "name": "search_web",
            "base_desc": "Search the web for information",
            "params": [
                {"name": "query", "type": "string", "desc": "Search query"},
                {"name": "num_results", "type": "integer", "desc": "Number of results", "required": False}
            ],
            "tags": ["web", "search", "information"],
            "test_prompt": "Search for Python tutorial resources"
        },
    ],
    "text_operations": [
        {
            "name": "translate_text",
            "base_desc": "Translate text between languages",
            "params": [
                {"name": "text", "type": "string", "desc": "Text to translate"},
                {"name": "source_language", "type": "string", "desc": "Source language code"},
                {"name": "target_language", "type": "string", "desc": "Target language code"}
            ],
            "tags": ["text", "translate", "language"],
            "test_prompt": "Translate 'Hello world' to Spanish"
        },
        {
            "name": "summarize_text",
            "base_desc": "Generate a summary of text",
            "params": [
                {"name": "text", "type": "string", "desc": "Text to summarize"},
                {"name": "max_length", "type": "integer", "desc": "Maximum summary length", "required": False}
            ],
            "tags": ["text", "summarize", "nlp"],
            "test_prompt": "Summarize this long article into key points"
        },
        {
            "name": "extract_keywords",
            "base_desc": "Extract keywords from text",
            "params": [
                {"name": "text", "type": "string", "desc": "Text to analyze"},
                {"name": "num_keywords", "type": "integer", "desc": "Number of keywords to extract", "required": False}
            ],
            "tags": ["text", "keywords", "nlp"],
            "test_prompt": "Find the main keywords in this document"
        },
    ],
    "system_operations": [
        {
            "name": "get_system_info",
            "base_desc": "Get system information",
            "params": [
                {"name": "info_type", "type": "string", "desc": "Type of info", 
                 "enum": ["cpu", "memory", "disk", "network", "all"]}
            ],
            "tags": ["system", "info", "monitoring"],
            "test_prompt": "Show me the current memory usage"
        },
        {
            "name": "run_command",
            "base_desc": "Execute a system command",
            "params": [
                {"name": "command", "type": "string", "desc": "Command to execute"},
                {"name": "timeout", "type": "integer", "desc": "Timeout in seconds", "required": False}
            ],
            "tags": ["system", "command", "shell"],
            "test_prompt": "Run the ls command to list files"
        },
        {
            "name": "schedule_task",
            "base_desc": "Schedule a task to run later",
            "params": [
                {"name": "task_name", "type": "string", "desc": "Name of the task"},
                {"name": "command", "type": "string", "desc": "Command to run"},
                {"name": "schedule", "type": "string", "desc": "Cron expression or datetime"}
            ],
            "tags": ["system", "schedule", "automation"],
            "test_prompt": "Schedule a backup to run every night at midnight"
        },
    ],
}


# Description length templates
DESCRIPTION_TEMPLATES = {
    "minimal": "{base}",
    "short": "{base}. {usage}",
    "medium": "{base}. {usage} {params_desc}",
    "long": "{base}. {usage} {params_desc} {examples}",
    "verbose": "{base}. {usage} {params_desc} {examples} {notes} {warnings}"
}

USAGE_TEMPLATES = [
    "Use this tool when you need to {action}.",
    "This tool allows you to {action}.",
    "Call this function to {action}.",
]

EXAMPLE_TEMPLATES = [
    "Example: {example}.",
    "For instance, you can {example}.",
]

NOTE_TEMPLATES = [
    "Note: This operation may take some time depending on the input size.",
    "Note: Ensure you have the necessary permissions before using this tool.",
    "Note: Results are cached for 5 minutes.",
]

WARNING_TEMPLATES = [
    "Warning: This action cannot be undone.",
    "Warning: Use with caution in production environments.",
    "Warning: Large inputs may cause performance issues.",
]


class ToolGenerator:
    """
    Generator for creating synthetic tools with configurable parameters.
    
    Supports varying:
    - Number of tools
    - Documentation length
    - Parameter complexity
    - Semantic similarity
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator with an optional seed for reproducibility."""
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def generate_tools(
        self,
        num_tools: int,
        doc_length: str = "medium",
        include_similar: int = 0,
        categories: Optional[list[str]] = None
    ) -> list[Tool]:
        """
        Generate a list of tools with specified parameters.
        
        Args:
            num_tools: Number of tools to generate
            doc_length: Description length (minimal, short, medium, long, verbose)
            include_similar: Number of similar/distractor tools to include
            categories: Categories to include (None = all)
            
        Returns:
            List of generated Tool objects
        """
        if categories is None:
            categories = list(TOOL_TEMPLATES.keys())
        
        # Collect all available tool templates
        all_templates = []
        for category in categories:
            if category in TOOL_TEMPLATES:
                for template in TOOL_TEMPLATES[category]:
                    all_templates.append((category, template))
        
        # Select tools
        if num_tools > len(all_templates):
            # Need to generate additional synthetic tools
            selected = all_templates.copy()
            for i in range(num_tools - len(all_templates)):
                selected.append(self._generate_synthetic_template(i, categories))
        else:
            selected = random.sample(all_templates, num_tools)
        
        # Generate Tool objects
        tools = []
        for category, template in selected:
            tool = self._create_tool_from_template(category, template, doc_length)
            tools.append(tool)
        
        # Add similar tools if requested
        if include_similar > 0:
            similar_tools = self._generate_similar_tools(tools[:include_similar], doc_length)
            tools.extend(similar_tools)
        
        return tools
    
    def _create_tool_from_template(
        self,
        category: str,
        template: dict,
        doc_length: str
    ) -> Tool:
        """Create a Tool object from a template with specified doc length."""
        # Build description based on length
        description = self._build_description(template, doc_length)
        
        # Create parameters
        parameters = []
        for param in template.get("params", []):
            parameters.append(ToolParameter(
                name=param["name"],
                type=param.get("type", "string"),
                description=param.get("desc", ""),
                required=param.get("required", True),
                enum=param.get("enum"),
                default=param.get("default")
            ))
        
        return Tool(
            name=template["name"],
            description=description,
            category=category,
            parameters=parameters,
            tags=template.get("tags", []),
            complexity=self._calculate_complexity(parameters)
        )
    
    def _build_description(self, template: dict, doc_length: str) -> str:
        """Build a description string based on the requested length."""
        base = template["base_desc"]
        action = base.lower().replace(".", "")
        
        if doc_length == "minimal":
            return base
        
        usage = random.choice(USAGE_TEMPLATES).format(action=action)
        
        if doc_length == "short":
            return f"{base}. {usage}"
        
        # Build parameter descriptions
        params_desc = "Parameters: " + ", ".join(
            f"{p['name']} ({p.get('type', 'string')})"
            for p in template.get("params", [])
        )
        
        if doc_length == "medium":
            return f"{base}. {usage} {params_desc}"
        
        example = template.get("test_prompt", f"use {template['name']}")
        examples = random.choice(EXAMPLE_TEMPLATES).format(example=example)
        
        if doc_length == "long":
            return f"{base}. {usage} {params_desc} {examples}"
        
        # verbose
        notes = random.choice(NOTE_TEMPLATES)
        warnings = random.choice(WARNING_TEMPLATES)
        return f"{base}. {usage} {params_desc} {examples} {notes} {warnings}"
    
    def _calculate_complexity(self, parameters: list[ToolParameter]) -> str:
        """Calculate tool complexity based on parameters."""
        num_params = len(parameters)
        has_enum = any(p.enum for p in parameters)
        has_optional = any(not p.required for p in parameters)
        
        if num_params <= 2 and not has_enum:
            return "simple"
        elif num_params <= 4 or (num_params <= 3 and has_enum):
            return "medium"
        else:
            return "complex"
    
    def _generate_synthetic_template(self, index: int, categories: list[str]) -> tuple:
        """Generate a synthetic tool template when we need more tools than templates."""
        category = random.choice(categories)
        
        # Generate synthetic tool
        actions = ["process", "analyze", "transform", "validate", "generate", "extract", "filter", "merge"]
        objects = ["data", "records", "items", "entries", "content", "values", "results", "output"]
        
        action = random.choice(actions)
        obj = random.choice(objects)
        
        template = {
            "name": f"{action}_{obj}_{index}",
            "base_desc": f"{action.capitalize()} {obj} based on specified criteria",
            "params": [
                {"name": "input", "type": "string", "desc": f"Input {obj} to {action}"},
                {"name": "options", "type": "string", "desc": "Processing options", "required": False}
            ],
            "tags": [action, obj, "synthetic"],
            "test_prompt": f"{action.capitalize()} the given {obj}"
        }
        
        return (category, template)
    
    def _generate_similar_tools(self, base_tools: list[Tool], doc_length: str) -> list[Tool]:
        """Generate similar/distractor tools based on existing tools."""
        similar_tools = []
        
        for tool in base_tools:
            # Create a similar tool with slight name variation
            similar_name = f"{tool.name}_v2"
            similar_desc = tool.description.replace(
                tool.name.replace("_", " "),
                f"alternative {tool.name.replace('_', ' ')}"
            )
            
            similar_tool = Tool(
                name=similar_name,
                description=f"Alternative version: {similar_desc}",
                category=tool.category,
                parameters=tool.parameters.copy(),
                tags=tool.tags + ["similar", "alternative"],
                complexity=tool.complexity
            )
            similar_tools.append(similar_tool)
        
        return similar_tools
    
    def generate_test_cases(self, tools: list[Tool]) -> list[TestCase]:
        """Generate test cases for the given tools."""
        test_cases = []
        
        for tool in tools:
            # Find the original template to get test prompt
            test_prompt = None
            for category, templates in TOOL_TEMPLATES.items():
                for template in templates:
                    if template["name"] == tool.name:
                        test_prompt = template.get("test_prompt")
                        break
            
            if test_prompt is None:
                # Generate a generic test prompt
                test_prompt = f"Use {tool.name.replace('_', ' ')} to perform the operation"
            
            test_cases.append(TestCase(
                prompt=test_prompt,
                expected_tool=tool.name,
                category=tool.category,
                difficulty="easy" if tool.complexity == "simple" else "medium",
                description=f"Test that {tool.name} is called for: {test_prompt}"
            ))
        
        return test_cases
