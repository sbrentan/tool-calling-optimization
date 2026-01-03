"""
Tool generator for creating tools from YAML definitions with configurable parameters.

This module loads tool definitions from YAML files and generates Tool objects with:
- Configurable documentation length (minimal to verbose)
- Multi-tool test scenarios
- Similar tool variants for testing
- No-tool test scenarios (where no tool should be called)
"""
import os
import random
from pathlib import Path
from typing import Optional, Union
import yaml
from loguru import logger
from .base import Tool, ToolParameter, TestCase, MultiToolTestCase

# Type alias for any test case
AnyTestCase = Union[TestCase, MultiToolTestCase]


class ToolGenerator:
    """
    Generator for creating tools from YAML definitions with configurable parameters.
    
    Supports:
    - Loading tools from YAML files
    - Variable documentation verbosity (5 levels)
    - Multi-tool test scenarios
    - Similar tool generation for distractor testing
    """
    
    # Mapping from verbosity names to YAML description keys
    VERBOSITY_LEVELS = ["minimal", "short", "medium", "long", "verbose"]
    
    def __init__(self, tools_dir: Optional[str] = None, seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Args:
            tools_dir: Path to directory containing YAML tool definitions.
                      Defaults to 'tools/' in project root.
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        # Find tools directory
        if tools_dir is None:
            # Default: look for 'tools/' relative to project root
            project_root = Path(__file__).parent.parent.parent
            tools_dir = project_root / "tools"
        else:
            tools_dir = Path(tools_dir)
        
        self.tools_dir = tools_dir
        self._tools_cache: dict[str, list[dict]] = {}  # category -> tools
        self._load_all_tools()
    
    def _load_all_tools(self) -> None:
        """Load all tool definitions from YAML files."""
        if not self.tools_dir.exists():
            logger.warning(f"Tools directory not found: {self.tools_dir}")
            return
        
        yaml_files = list(self.tools_dir.glob("*.yaml")) + list(self.tools_dir.glob("*.yml"))
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if data and "tools" in data:
                    category = data.get("category", yaml_file.stem)
                    self._tools_cache[category] = data["tools"]
                    logger.info(f"Loaded {len(data['tools'])} tools from {yaml_file.name} (category: {category})")
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")
        
        total_tools = sum(len(tools) for tools in self._tools_cache.values())
        logger.info(f"Total tools loaded: {total_tools} across {len(self._tools_cache)} categories")
    
    def get_categories(self) -> list[str]:
        """Get list of available tool categories."""
        return list(self._tools_cache.keys())
    
    def get_tool_count(self, categories: Optional[list[str]] = None) -> int:
        """Get total number of tools available, optionally filtered by categories."""
        if categories is None:
            return sum(len(tools) for tools in self._tools_cache.values())
        return sum(len(self._tools_cache.get(cat, [])) for cat in categories)
    
    def generate_tools(
        self,
        num_tools: Optional[int] = None,
        doc_length: str = "medium",
        include_similar: int = 0,
        categories: Optional[list[str]] = None
    ) -> list[Tool]:
        """
        Generate a list of tools with specified parameters.
        
        Args:
            num_tools: Number of tools to generate. None = all available tools.
            doc_length: Description verbosity (minimal, short, medium, long, verbose)
            include_similar: Number of similar/distractor tools to include
            categories: Categories to include (None = all)
            
        Returns:
            List of generated Tool objects
        """
        if doc_length not in self.VERBOSITY_LEVELS:
            logger.warning(f"Unknown doc_length '{doc_length}', defaulting to 'medium'")
            doc_length = "medium"
        
        if categories is None:
            categories = list(self._tools_cache.keys())
        
        # Collect all available tool definitions
        all_tools: list[tuple[str, dict]] = []
        for category in categories:
            if category in self._tools_cache:
                for tool_def in self._tools_cache[category]:
                    all_tools.append((category, tool_def))
        
        if not all_tools:
            logger.warning("No tools available! Check your tools directory.")
            return []
        
        # Select tools
        if num_tools is None:
            selected = all_tools
        elif num_tools > len(all_tools):
            logger.warning(
                f"Requested {num_tools} tools but only {len(all_tools)} available. "
                f"Using all available tools."
            )
            selected = all_tools
        else:
            selected = random.sample(all_tools, num_tools)
        
        # Generate Tool objects
        tools = []
        for category, tool_def in selected:
            tool = self._create_tool_from_definition(category, tool_def, doc_length)
            tools.append(tool)
        
        # Add similar tools if requested
        if include_similar > 0 and tools:
            base_tools = tools[:min(include_similar, len(tools))]
            similar_tools = self._generate_similar_tools(base_tools, doc_length)
            tools.extend(similar_tools)
        
        return tools
    
    def _create_tool_from_definition(
        self,
        category: str,
        tool_def: dict,
        doc_length: str
    ) -> Tool:
        """Create a Tool object from a YAML definition with specified doc length."""
        # Get description for the specified verbosity level
        descriptions = tool_def.get("descriptions", {})
        
        # Try to get the exact verbosity level, fall back to closest available
        description = descriptions.get(doc_length)
        if description is None:
            # Try to find a fallback
            for level in self.VERBOSITY_LEVELS:
                if level in descriptions:
                    description = descriptions[level]
                    if level == doc_length or self.VERBOSITY_LEVELS.index(level) >= self.VERBOSITY_LEVELS.index(doc_length):
                        break
        
        if description is None:
            description = tool_def.get("name", "Unknown tool")
            logger.warning(f"No description found for tool '{tool_def.get('name')}', using name")
        
        # Create parameters
        parameters = []
        for param_def in tool_def.get("parameters", []):
            parameters.append(ToolParameter(
                name=param_def["name"],
                type=param_def.get("type", "string"),
                description=param_def.get("description", ""),
                required=param_def.get("required", True),
                enum=param_def.get("enum"),
                default=param_def.get("default")
            ))
        
        return Tool(
            name=tool_def["name"],
            description=description,
            category=category,
            parameters=parameters,
            tags=tool_def.get("tags", []),
            complexity=self._calculate_complexity(parameters)
        )
    
    def _calculate_complexity(self, parameters: list[ToolParameter]) -> str:
        """Calculate tool complexity based on parameters."""
        num_params = len(parameters)
        num_required = sum(1 for p in parameters if p.required)
        has_enum = any(p.enum for p in parameters)
        
        if num_required <= 1 and num_params <= 2:
            return "simple"
        elif num_required <= 2 and num_params <= 4:
            return "medium"
        else:
            return "complex"
    
    def _generate_similar_tools(self, base_tools: list[Tool], doc_length: str) -> list[Tool]:
        """Generate similar/distractor tools based on existing tools."""
        similar_tools = []
        
        # Synonyms for generating similar tool names
        synonyms = {
            "read": ["load", "fetch", "get", "retrieve"],
            "write": ["save", "store", "put", "export"],
            "delete": ["remove", "erase", "clear", "purge"],
            "create": ["make", "new", "add", "generate"],
            "update": ["modify", "change", "edit", "alter"],
            "list": ["enumerate", "show", "display", "get_all"],
            "search": ["find", "query", "lookup", "locate"],
            "copy": ["duplicate", "clone", "replicate"],
            "move": ["transfer", "relocate", "migrate"],
            "get": ["fetch", "retrieve", "obtain", "acquire"],
            "set": ["configure", "assign", "define"],
            "check": ["verify", "validate", "test", "confirm"],
            "calculate": ["compute", "evaluate", "determine"],
            "convert": ["transform", "change", "translate"],
            "parse": ["analyze", "extract", "process"],
            "format": ["style", "arrange", "structure"],
            "encode": ["encrypt", "cipher"],
            "decode": ["decrypt", "decipher"],
            "compress": ["zip", "pack", "shrink"],
            "decompress": ["unzip", "unpack", "expand"],
            "http": ["request", "call", "invoke"],
            "query": ["search", "find", "select"],
            "insert": ["add", "create", "put"],
            "count": ["tally", "sum", "total"],
            "round": ["truncate", "floor", "ceil"],
            "hash": ["digest", "checksum"],
            "trim": ["strip", "clean"],
            "split": ["separate", "divide", "tokenize"],
            "join": ["combine", "merge", "concatenate"],
            "replace": ["substitute", "swap"],
            "extract": ["pull", "get", "parse"],
            "validate": ["verify", "check", "confirm"],
            "translate": ["convert", "transform"],
            "summarize": ["condense", "brief", "shorten"],
            "detect": ["identify", "recognize", "find"],
            "download": ["fetch", "get", "retrieve"],
            "upload": ["send", "push", "transfer"],
            "scrape": ["extract", "crawl", "harvest"],
        }
        
        for tool in base_tools:
            # Find a synonym for the first word of the tool name
            name_parts = tool.name.split("_")
            first_word = name_parts[0]
            
            if first_word in synonyms:
                new_first = random.choice(synonyms[first_word])
                new_name = "_".join([new_first] + name_parts[1:])
            else:
                new_name = f"{tool.name}_alt"
            
            # Modify description slightly
            similar_desc = tool.description
            for original, alts in synonyms.items():
                if original in similar_desc.lower():
                    replacement = random.choice(alts)
                    similar_desc = similar_desc.replace(original, replacement)
                    similar_desc = similar_desc.replace(original.capitalize(), replacement.capitalize())
                    break
            
            similar_tool = Tool(
                name=new_name,
                description=similar_desc,
                category=tool.category,
                parameters=tool.parameters.copy(),
                tags=tool.tags + ["similar", "alternative"],
                complexity=tool.complexity
            )
            similar_tools.append(similar_tool)
        
        return similar_tools
    
    def generate_test_cases(
        self,
        tools: list[Tool],
        include_multi_tool: bool = False,
        include_no_tool: bool = False,
        prompt_type: str = "concise"
    ) -> list[AnyTestCase]:
        """
        Generate test cases for the given tools.
        
        Args:
            tools: List of tools to generate tests for
            include_multi_tool: Whether to include multi-tool test scenarios
            include_no_tool: Whether to include no-tool test scenarios
            prompt_type: Type of prompts to use ('concise' or 'clear')
            
        Returns:
            List of TestCase and MultiToolTestCase objects
        """
        test_cases: list[AnyTestCase] = []
        tool_name_to_category = {t.name: t.category for t in tools}
        
        for tool in tools:
            # Find the original definition to get test prompts
            test_prompts = self._get_test_prompts(tool.name, tool.category)
            
            if test_prompts:
                # Select prompts based on prompt_type
                if prompt_type == "clear":
                    single_prompts = test_prompts.get("single_clear", test_prompts.get("single", []))
                else:
                    single_prompts = test_prompts.get("single", [])
                
                if single_prompts:
                    prompt = random.choice(single_prompts) if isinstance(single_prompts, list) else single_prompts
                else:
                    prompt = f"Use {tool.name.replace('_', ' ')} to perform the operation"
            else:
                prompt = f"Use {tool.name.replace('_', ' ')} to perform the operation"
            
            test_cases.append(TestCase(
                prompt=prompt,
                expected_tool=tool.name,
                category=tool.category,
                difficulty=self._get_difficulty(tool.complexity, len(tools)),
                description=f"Test: {prompt}",
                prompt_type=prompt_type
            ))
        
        # Add no-tool test cases if requested
        if include_no_tool:
            no_tool_cases = self._generate_no_tool_cases(tools, prompt_type)
            test_cases.extend(no_tool_cases)
        
        # Add multi-tool test cases if requested
        if include_multi_tool:
            multi_tool_cases = self._generate_multi_tool_cases(tools, prompt_type)
            test_cases.extend(multi_tool_cases)
        
        return test_cases
    
    def _get_test_prompts(self, tool_name: str, category: str) -> Optional[dict]:
        """Get test prompts for a tool from its YAML definition."""
        if category not in self._tools_cache:
            return None
        
        for tool_def in self._tools_cache[category]:
            if tool_def.get("name") == tool_name:
                return tool_def.get("test_prompts")
        
        return None
    
    def _get_difficulty(self, complexity: str, total_tools: int) -> str:
        """Determine test difficulty based on complexity and tool count."""
        if total_tools <= 5:
            if complexity == "simple":
                return "easy"
            return "medium"
        elif total_tools <= 15:
            if complexity == "simple":
                return "medium"
            return "hard"
        else:
            if complexity == "simple":
                return "medium"
            elif complexity == "medium":
                return "hard"
            return "expert"
    
    def _generate_no_tool_cases(self, tools: list[Tool], prompt_type: str = "concise") -> list[TestCase]:
        """
        Generate no-tool test cases where no tool should be called.
        
        Categories of no-tool tests:
        1. Informational: Asking about tool capabilities
        2. Clarification: Vague requests needing more info
        3. Out-of-scope: Requests for tools not available
        4. Conversational: General conversation, greetings, thanks
        
        Args:
            tools: List of available tools (used for context-aware generation)
            prompt_type: Type of prompts to use ('concise' or 'clear')
            
        Returns:
            List of TestCase objects with expected_tool=None
        """
        no_tool_cases = []
        
        # 1. Informational prompts - asking about tools
        informational_prompts = [
            ("What does the {tool_name} function do?", "informational", "Asking about tool functionality"),
            ("Can you explain how {tool_name} works?", "informational", "Requesting tool explanation"),
            ("What parameters does {tool_name} accept?", "informational", "Asking about tool parameters"),
            ("What are all the available tools I can use?", "informational", "Asking about available tools"),
            ("List the tools you have access to", "informational", "Listing available tools"),
            ("What can you help me with?", "informational", "General capability inquiry"),
        ]
        
        # 2. Clarification-needed prompts - too vague
        clarification_prompts = [
            ("I want to do something with files", "clarification", "Vague file request"),
            ("Help me with data", "clarification", "Vague data request"),
            ("Can you process this?", "clarification", "Unclear processing request"),
            ("I need to work with some information", "clarification", "Vague information request"),
            ("Do something useful", "clarification", "Completely vague request"),
            ("Handle my request", "clarification", "No specific action mentioned"),
        ]
        
        # 3. Out-of-scope prompts - no matching tool
        out_of_scope_prompts = [
            ("What's the weather in Paris?", "out_of_scope", "Weather request - no weather tool"),
            ("Translate this text to French: Hello world", "out_of_scope", "Translation request - no translate tool"),
            ("Book a flight to New York", "out_of_scope", "Booking request - no booking tool"),
            ("Order a pizza for me", "out_of_scope", "Food order - no ordering tool"),
            ("Play some music", "out_of_scope", "Music request - no music tool"),
            ("Set an alarm for 7am", "out_of_scope", "Alarm request - no alarm tool"),
        ]
        
        # 4. Conversational prompts - no action needed
        conversational_prompts = [
            ("Hello!", "conversational", "Greeting"),
            ("Thanks for your help!", "conversational", "Thank you message"),
            ("That's great, goodbye!", "conversational", "Farewell message"),
            ("You're doing a great job", "conversational", "Compliment"),
            ("How are you today?", "conversational", "Small talk"),
            ("I appreciate your assistance", "conversational", "Appreciation"),
        ]
        
        # Generate informational cases using actual tool names
        if tools:
            sample_tools = random.sample(tools, min(3, len(tools)))
            for tool in sample_tools:
                prompt_template, category, reason = random.choice(informational_prompts[:3])
                no_tool_cases.append(TestCase(
                    prompt=prompt_template.format(tool_name=tool.name.replace("_", " ")),
                    expected_tool=None,
                    no_tool_reason=reason,
                    category="no_tool",
                    difficulty="medium",
                    description=f"No-tool test ({category}): {reason}",
                    prompt_type=prompt_type
                ))
        
        # Add general informational cases
        for prompt, category, reason in informational_prompts[3:]:
            no_tool_cases.append(TestCase(
                prompt=prompt,
                expected_tool=None,
                no_tool_reason=reason,
                category="no_tool",
                difficulty="easy",
                description=f"No-tool test ({category}): {reason}",
                prompt_type=prompt_type
            ))
        
        # Add clarification-needed cases
        for prompt, category, reason in clarification_prompts:
            no_tool_cases.append(TestCase(
                prompt=prompt,
                expected_tool=None,
                no_tool_reason=reason,
                category="no_tool",
                difficulty="hard",
                description=f"No-tool test ({category}): {reason}",
                prompt_type=prompt_type
            ))
        
        # Add out-of-scope cases
        for prompt, category, reason in out_of_scope_prompts:
            no_tool_cases.append(TestCase(
                prompt=prompt,
                expected_tool=None,
                no_tool_reason=reason,
                category="no_tool",
                difficulty="medium",
                description=f"No-tool test ({category}): {reason}",
                prompt_type=prompt_type
            ))
        
        # Add conversational cases
        for prompt, category, reason in conversational_prompts:
            no_tool_cases.append(TestCase(
                prompt=prompt,
                expected_tool=None,
                no_tool_reason=reason,
                category="no_tool",
                difficulty="easy",
                description=f"No-tool test ({category}): {reason}",
                prompt_type=prompt_type
            ))
        
        return no_tool_cases
    
    def _generate_multi_tool_cases(self, tools: list[Tool], prompt_type: str = "concise") -> list[MultiToolTestCase]:
        """Generate multi-tool test cases from YAML definitions."""
        multi_cases: list[MultiToolTestCase] = []
        tool_names = {t.name for t in tools}
        
        for category, tool_defs in self._tools_cache.items():
            for tool_def in tool_defs:
                test_prompts = tool_def.get("test_prompts", {})
                
                # Select multi prompts based on prompt_type
                if prompt_type == "clear":
                    multi_prompts = test_prompts.get("multi_clear", test_prompts.get("multi", []))
                else:
                    multi_prompts = test_prompts.get("multi", [])
                
                for multi in multi_prompts:
                    if isinstance(multi, dict):
                        prompt = multi.get("prompt", "")
                        required_tools = multi.get("required_tools", [])
                        require_sequence = multi.get("require_sequence", False)
                        
                        # Only include if all required tools are in our set
                        if all(rt in tool_names for rt in required_tools) and len(required_tools) >= 2:
                            multi_cases.append(MultiToolTestCase(
                                prompt=prompt,
                                expected_tools=required_tools,
                                require_sequence=require_sequence,
                                category=category,
                                difficulty="hard",
                                description=f"Multi-tool test requiring: {', '.join(required_tools)}",
                                prompt_type=prompt_type
                            ))
        
        return multi_cases
    
    def list_all_tools(self) -> list[dict]:
        """List all available tools with their metadata."""
        all_tools = []
        for category, tools in self._tools_cache.items():
            for tool in tools:
                all_tools.append({
                    "name": tool["name"],
                    "category": category,
                    "tags": tool.get("tags", []),
                    "param_count": len(tool.get("parameters", [])),
                    "has_multi_tool_tests": bool(tool.get("test_prompts", {}).get("multi"))
                })
        return all_tools


# Keep TOOL_TEMPLATES for backward compatibility (deprecated)
TOOL_TEMPLATES = {
}
