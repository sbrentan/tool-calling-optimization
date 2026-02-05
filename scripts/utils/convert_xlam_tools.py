#!/usr/bin/env python3
"""
Convert xLAM Function Calling dataset tools to project YAML format.

This script reads the xlam_function_calling_60k.json file and converts tools
to the project's YAML-based tool definition format.
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any

import yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert xLAM Function Calling dataset to project YAML format"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="xlam_function_calling_60k.json",
        help="Path to the xLAM dataset JSON file (default: xlam_function_calling_60k.json)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Output directory for generated YAML files",
    )
    parser.add_argument(
        "--num-tools",
        "-n",
        type=int,
        default=985,
        help="Number of unique tools to extract (default: 985)",
    )
    parser.add_argument(
        "--max-answer-tools",
        "-m",
        type=int,
        default=1,
        help="Maximum number of tools called in answers to include (default: 1, max: 3)",
    )
    parser.add_argument(
        "--tools-per-file",
        type=int,
        default=25,
        help="Maximum number of tools per YAML file (default: 25)",
    )
    return parser.parse_args()


def load_xlam_dataset(input_path: str) -> list[dict]:
    """Load the xLAM dataset from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_entry(entry: dict) -> dict:
    """Parse a single xLAM entry, deserializing JSON strings."""
    return {
        "id": entry.get("id"),
        "query": entry.get("query", ""),
        "tools": json.loads(entry.get("tools", "[]")),
        "answers": json.loads(entry.get("answers", "[]")),
    }


def filter_entries_by_answer_count(
    entries: list[dict], max_answer_tools: int
) -> list[dict]:
    """Filter entries based on number of tools called in answers."""
    filtered = []
    for entry in entries:
        parsed = parse_entry(entry)
        num_answers = len(parsed["answers"])
        if 1 <= num_answers <= max_answer_tools:
            filtered.append(parsed)
    return filtered


def normalize_type(type_str: str) -> tuple[str, bool]:
    """
    Normalize xLAM type strings to project types.
    
    Returns (normalized_type, is_optional).
    """
    type_str_lower = type_str.lower().strip()
    
    # Check if optional
    is_optional = "optional" in type_str_lower
    
    # Remove 'optional' suffix
    type_str_clean = re.sub(r",?\s*optional$", "", type_str_lower)
    
    # Map common types
    type_mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "integer": "integer",
        "string": "string",
        "number": "number",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
    }
    
    return type_mapping.get(type_str_clean, "string"), is_optional


def generate_descriptions(description: str) -> dict[str, str]:
    """Generate multi-verbosity descriptions from a single description."""
    description = description.strip()
    
    # Minimal: First 50 chars or first sentence fragment
    if len(description) <= 50:
        minimal = description
    else:
        # Try to cut at a word boundary
        minimal = description[:50].rsplit(" ", 1)[0]
        if len(minimal) < 20:
            minimal = description[:50]
        minimal = minimal.rstrip(".,;:") + "..."
    
    # Short: First sentence or 100 chars
    first_sentence_match = re.match(r"^([^.!?]+[.!?])", description)
    if first_sentence_match:
        short = first_sentence_match.group(1)
    elif len(description) <= 100:
        short = description
    else:
        short = description[:100].rsplit(" ", 1)[0] + "..."
    
    # Medium: Full description as-is
    medium = description
    
    # Long: Description + generic enhancement
    long = description
    if not long.endswith("."):
        long += "."
    
    # Verbose: Same as long for simplicity (since we don't have more context)
    verbose = long
    
    return {
        "minimal": minimal,
        "short": short,
        "medium": medium,
        "long": long,
        "verbose": verbose,
    }


def convert_parameters(xlam_params: dict) -> list[dict]:
    """Convert xLAM parameter format to project format."""
    parameters = []
    
    for param_name, param_info in xlam_params.items():
        param_type_str = param_info.get("type", "string")
        normalized_type, is_optional = normalize_type(param_type_str)
        
        # Determine required status: explicit 'required' field takes precedence,
        # otherwise infer from 'optional' in type string
        if "required" in param_info:
            is_required = param_info["required"]
        else:
            is_required = not is_optional
        
        param = {
            "name": param_name,
            "type": normalized_type,
            "description": param_info.get("description", ""),
            "required": is_required,
        }
        
        # Add default if present
        if "default" in param_info and param_info["default"] != "":
            param["default"] = param_info["default"]
        
        # Add enum if present
        if "enum" in param_info:
            param["enum"] = param_info["enum"]
        
        parameters.append(param)
    
    return parameters


def extract_tools_with_prompts(
    filtered_entries: list[dict], num_tools: int
) -> dict[str, dict]:
    """
    Extract unique tools with their associated prompts.
    
    Returns a dict mapping tool_name -> {
        "definition": tool_definition,
        "prompts": [list of query prompts],
        "answers": [list of expected answers]
    }
    """
    tools_data = defaultdict(
        lambda: {"definition": None, "prompts": [], "answers": []}
    )
    
    for entry in filtered_entries:
        query = entry["query"]
        answers = entry["answers"]
        tools = entry["tools"]
        
        # Get the tool(s) that are called in the answer
        answered_tool_names = {ans["name"] for ans in answers}
        
        # Find the tool definitions for answered tools
        for tool in tools:
            tool_name = tool["name"]
            if tool_name in answered_tool_names:
                # Store or update tool definition
                if tools_data[tool_name]["definition"] is None:
                    tools_data[tool_name]["definition"] = tool
                
                # Add prompt and answer
                tools_data[tool_name]["prompts"].append(query)
                
                # Find the specific answer for this tool
                for ans in answers:
                    if ans["name"] == tool_name:
                        tools_data[tool_name]["answers"].append(ans)
                        break
        
        # Stop if we have enough tools
        if len(tools_data) >= num_tools:
            break
    
    # Limit to requested number of tools
    limited_tools = dict(list(tools_data.items())[:num_tools])
    return limited_tools


def convert_tool_to_yaml_format(
    tool_name: str, tool_data: dict
) -> dict:
    """Convert a tool with its prompts to project YAML format."""
    definition = tool_data["definition"]
    prompts = tool_data["prompts"]
    
    # Generate descriptions
    descriptions = generate_descriptions(definition.get("description", tool_name))
    
    # Convert parameters
    parameters = convert_parameters(definition.get("parameters", {}))
    
    # Generate tags from tool name
    tags = tool_name.replace("_", " ").split()[:3]
    
    # Build the tool entry
    tool_entry = {
        "name": tool_name,
        "descriptions": descriptions,
        "parameters": parameters,
        "tags": tags,
        "test_prompts": {
            "single": prompts.copy(),
            "single_clear": prompts.copy(),  # Same as single per user request
        },
    }
    
    return tool_entry


def group_tools_into_categories(
    tools: dict[str, dict], tools_per_file: int
) -> list[tuple[str, list[dict]]]:
    """Group tools into categories with specified max tools per file."""
    categories = []
    tool_list = list(tools.items())
    
    for i in range(0, len(tool_list), tools_per_file):
        category_num = (i // tools_per_file) + 1
        category_name = f"category_{category_num}"
        
        category_tools = []
        for tool_name, tool_data in tool_list[i : i + tools_per_file]:
            tool_entry = convert_tool_to_yaml_format(tool_name, tool_data)
            category_tools.append(tool_entry)
        
        categories.append((category_name, category_tools))
    
    return categories


def write_yaml_files(
    categories: list[tuple[str, list[dict]]], output_dir: Path
):
    """Write category YAML files to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for category_name, tools in categories:
        yaml_content = {
            "category": category_name,
            "tools": tools,
        }
        
        output_file = output_dir / f"{category_name}.yaml"
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(
                yaml_content,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )
        
        print(f"Written: {output_file} ({len(tools)} tools)")


def write_prompts_mapping(
    tools: dict[str, dict], output_dir: Path
):
    """Write a JSON file mapping tools to their prompts and expected answers."""
    mapping = {}
    
    for tool_name, tool_data in tools.items():
        mapping[tool_name] = {
            "prompts": tool_data["prompts"],
            "expected_answers": tool_data["answers"],
        }
    
    output_file = output_dir / "prompts_mapping.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"Written prompts mapping: {output_file}")


def main():
    args = parse_args()
    
    # Validate max_answer_tools
    if args.max_answer_tools < 1 or args.max_answer_tools > 3:
        print("Error: --max-answer-tools must be between 1 and 3")
        return 1
    
    # Resolve paths
    input_path = Path(args.input)
    if not input_path.is_absolute():
        # Try relative to script location, then current directory
        script_dir = Path(__file__).parent.parent
        if (script_dir / args.input).exists():
            input_path = script_dir / args.input
    
    output_dir = Path(args.output_dir)
    
    print(f"Loading xLAM dataset from: {input_path}")
    entries = load_xlam_dataset(input_path)
    print(f"Loaded {len(entries)} entries")
    
    print(f"Filtering entries with 1-{args.max_answer_tools} tool(s) in answers...")
    filtered_entries = filter_entries_by_answer_count(entries, args.max_answer_tools)
    print(f"Found {len(filtered_entries)} matching entries")
    
    print(f"Extracting up to {args.num_tools} unique tools...")
    tools = extract_tools_with_prompts(filtered_entries, args.num_tools)
    print(f"Extracted {len(tools)} unique tools")
    
    # Count total prompts
    total_prompts = sum(len(t["prompts"]) for t in tools.values())
    print(f"Total prompts collected: {total_prompts}")
    
    print(f"Grouping into categories ({args.tools_per_file} tools per file)...")
    categories = group_tools_into_categories(tools, args.tools_per_file)
    print(f"Created {len(categories)} category files")
    
    print(f"Writing YAML files to: {output_dir}")
    write_yaml_files(categories, output_dir)
    
    print("Writing prompts mapping file...")
    write_prompts_mapping(tools, output_dir)
    
    print("\nConversion complete!")
    print(f"  - Tools: {len(tools)}")
    print(f"  - Categories: {len(categories)}")
    print(f"  - Total prompts: {total_prompts}")
    
    return 0


if __name__ == "__main__":
    exit(main())
