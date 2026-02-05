"""
Tool generator for creating tools from YAML definitions with configurable parameters.

This module loads tool definitions from YAML files and generates Tool objects with:
- Configurable documentation length (minimal to verbose)
- Multi-tool test scenarios
- Similar tool variants for testing
- No-tool test scenarios (where no tool should be called)
"""
import json
import os
import random
from pathlib import Path
from typing import Any, Optional, Union
import yaml
from loguru import logger
from .base import Tool, ToolParameter, TestCase, MultiToolTestCase, AmbiguousTestCase

# Type alias for any test case
AnyTestCase = Union[TestCase, MultiToolTestCase, AmbiguousTestCase]


TOOLS_DIR = Path(__file__).parent.parent.parent / "tools"


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
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        # Find tools directory
        if tools_dir is None:
            # Default: look for 'tools/' relative to project root
            tools_dir = TOOLS_DIR
        else:
            tools_dir = Path(tools_dir)
        
        self.tools_dir = tools_dir
        self._tools_cache: dict[str, list[dict]] = {}  # category -> tools
        self._prompts_mapping: dict[str, dict[str, Any]] = {}  # tool_name -> {prompts, expected_answers}
        self._load_all_tools()
        self._load_prompts_mapping()
    
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
    
    def _load_prompts_mapping(self) -> None:
        """
        Load prompts_mapping.json if it exists in the tools directory.
        
        This file contains expected parameters for each prompt, enabling
        parameter validation during evaluation.
        """
        mapping_file = self.tools_dir / "prompts_mapping.json"
        if not mapping_file.exists():
            logger.debug(f"No prompts_mapping.json found in {self.tools_dir}")
            return
        
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self._prompts_mapping = json.load(f)
            logger.info(f"Loaded prompts mapping for {len(self._prompts_mapping)} tools")
        except Exception as e:
            logger.error(f"Failed to load prompts_mapping.json: {e}")
            self._prompts_mapping = {}
    
    def has_prompts_mapping(self) -> bool:
        """Check if prompts mapping is available for parameter validation."""
        return len(self._prompts_mapping) > 0
    
    def get_expected_params_for_prompt(
        self, tool_name: str, prompt: str
    ) -> Optional[dict[str, Any]]:
        """
        Get expected parameters for a specific prompt.
        
        Args:
            tool_name: Name of the tool
            prompt: The test prompt
            
        Returns:
            Expected parameters dict or None if not found
        """
        if tool_name not in self._prompts_mapping:
            return None
        
        tool_data = self._prompts_mapping[tool_name]
        prompts = tool_data.get("prompts", [])
        answers = tool_data.get("expected_answers", [])
        
        # Find the prompt index
        try:
            idx = prompts.index(prompt)
            if idx < len(answers):
                answer = answers[idx]
                return answer.get("arguments", {})
        except ValueError:
            pass
        
        return None
    
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
        include_ambiguous: bool = False,
        prompt_type: str = "concise",
        validate_params: bool = False
    ) -> list[AnyTestCase]:
        """
        Generate test cases for the given tools.
        
        Args:
            tools: List of tools to generate tests for
            include_multi_tool: Whether to include multi-tool test scenarios
            include_no_tool: Whether to include no-tool test scenarios
            include_ambiguous: Whether to include ambiguous test scenarios
            prompt_type: Type of prompts to use ('concise' or 'clear')
            validate_params: Whether to include expected parameters from prompts_mapping.json
            
        Returns:
            List of TestCase, MultiToolTestCase, and AmbiguousTestCase objects
        """
        test_cases: list[AnyTestCase] = []
        tool_name_to_category = {t.name: t.category for t in tools}
        
        # Warn if validate_params requested but no mapping available
        if validate_params and not self.has_prompts_mapping():
            logger.warning(
                "validate_params=True but no prompts_mapping.json found in tools directory. "
                "Parameter validation will be skipped."
            )
        
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
            
            # Get expected parameters if validation is enabled
            expected_params = None
            if validate_params and self.has_prompts_mapping():
                expected_params = self.get_expected_params_for_prompt(tool.name, prompt)
            
            test_cases.append(TestCase(
                prompt=prompt,
                expected_tool=tool.name,
                expected_params=expected_params,
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
        
        # Add ambiguous test cases if requested
        if include_ambiguous:
            ambiguous_cases = self._generate_ambiguous_cases(tools, prompt_type)
            test_cases.extend(ambiguous_cases)
        
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
    
    def _generate_ambiguous_cases(self, tools: list[Tool], prompt_type: str = "concise") -> list[AmbiguousTestCase]:
        """Generate ambiguous test cases where multiple similar tools could apply.
        
        These cases test the clarification capability - when a request is vague
        and could be fulfilled by multiple tools, the system should ask for clarification.
        """
        ambiguous_cases: list[AmbiguousTestCase] = []
        tool_names = {t.name for t in tools}
        
        # Group tools by their tags to find similar tools
        tag_to_tools: dict[str, list[Tool]] = {}
        for tool in tools:
            for tag in tool.tags:
                if tag not in tag_to_tools:
                    tag_to_tools[tag] = []
                tag_to_tools[tag].append(tool)
        
        # Also group by category/prefix (e.g., all "file_*" tools)
        prefix_to_tools: dict[str, list[Tool]] = {}
        for tool in tools:
            if "_" in tool.name:
                prefix = tool.name.split("_")[0]
                if prefix not in prefix_to_tools:
                    prefix_to_tools[prefix] = []
                prefix_to_tools[prefix].append(tool)
        
        # First, look for ambiguous prompts defined in YAML
        for category, tool_defs in self._tools_cache.items():
            for tool_def in tool_defs:
                test_prompts = tool_def.get("test_prompts", {})
                
                # Check for ambiguous prompt definitions
                ambiguous_prompts = test_prompts.get("ambiguous", [])
                
                for ambig in ambiguous_prompts:
                    if isinstance(ambig, dict):
                        prompt = ambig.get("prompt", "")
                        candidate_tools = ambig.get("candidate_tools", [])
                        correct_tool = ambig.get("correct_tool", tool_def["name"])
                        ambiguity_type = ambig.get("ambiguity_type", "similar_function")
                        
                        # Only include if all candidate tools exist and correct tool is valid
                        if (all(ct in tool_names for ct in candidate_tools) and 
                            correct_tool in tool_names and 
                            len(candidate_tools) >= 2):
                            ambiguous_cases.append(AmbiguousTestCase(
                                prompt=prompt,
                                expected_candidate_tools=candidate_tools,
                                correct_tool=correct_tool,
                                ambiguity_type=ambiguity_type,
                                category=category,
                                difficulty="hard",
                                description=f"Ambiguous test ({ambiguity_type}): could be {', '.join(candidate_tools)}",
                                prompt_type=prompt_type
                            ))
        
        # Generate synthetic ambiguous cases from similar tools
        # Find groups of similar tools (sharing tags or prefix)
        for tag, similar_tools in tag_to_tools.items():
            if len(similar_tools) >= 2 and len(similar_tools) <= 5:
                # Create an ambiguous prompt that could apply to any of these tools
                tool_names_list = [t.name for t in similar_tools[:4]]  # Limit to 4
                
                # Create vague prompts based on the tag
                vague_prompts = self._create_vague_prompts_for_tag(tag, similar_tools[:4])
                
                for prompt, ambiguity_type in vague_prompts:
                    ambiguous_cases.append(AmbiguousTestCase(
                        prompt=prompt,
                        expected_candidate_tools=tool_names_list,
                        correct_tool=tool_names_list[0],  # First tool as default correct
                        ambiguity_type=ambiguity_type,
                        category=similar_tools[0].category or "general",
                        difficulty="hard",
                        description=f"Synthetic ambiguous ({tag}): {', '.join(tool_names_list)}",
                        prompt_type=prompt_type
                    ))
        
        return ambiguous_cases
    
    def _create_vague_prompts_for_tag(self, tag: str, tools: list[Tool]) -> list[tuple[str, str]]:
        """Create vague prompts that could apply to multiple tools with the same tag."""
        prompts = []
        
        # Map common tags to vague request patterns
        tag_patterns = {
            "file": [
                ("I need to do something with a file", "vague_action"),
                ("Can you help me with file operations?", "underspecified"),
                ("Handle this file for me", "vague_action"),
            ],
            "data": [
                ("I need to work with some data", "vague_action"),
                ("Process this data somehow", "underspecified"),
                ("Do something with the data", "vague_action"),
            ],
            "search": [
                ("Find something for me", "underspecified"),
                ("I need to search", "vague_action"),
                ("Look something up", "underspecified"),
            ],
            "create": [
                ("Create something new", "underspecified"),
                ("Make a new one", "vague_action"),
                ("I need to create", "underspecified"),
            ],
            "delete": [
                ("Remove this", "underspecified"),
                ("Delete something", "vague_action"),
                ("Get rid of it", "underspecified"),
            ],
            "update": [
                ("Update this", "underspecified"),
                ("Make some changes", "vague_action"),
                ("Modify it somehow", "underspecified"),
            ],
            "send": [
                ("Send something", "underspecified"),
                ("I need to send a message", "underspecified"),
                ("Dispatch this", "vague_action"),
            ],
            "get": [
                ("Get me that thing", "underspecified"),
                ("Retrieve something", "vague_action"),
                ("Fetch the data", "underspecified"),
            ],
            "list": [
                ("Show me what's there", "underspecified"),
                ("List everything", "vague_action"),
                ("What do I have?", "underspecified"),
            ],
            "convert": [
                ("Convert this", "underspecified"),
                ("Change the format", "underspecified"),
                ("Transform it", "vague_action"),
            ],
            "async": [
                ("Do this in the background", "underspecified"),
                ("Run it asynchronously", "vague_action"),
            ],
            "batch": [
                ("Process all of these", "underspecified"),
                ("Do this for multiple items", "vague_action"),
            ],
        }
        
        # Check if the tag matches any pattern
        for pattern_tag, patterns in tag_patterns.items():
            if pattern_tag in tag.lower():
                prompts.extend(patterns[:2])  # Add up to 2 patterns per tag
                break
        
        # If no specific pattern, create generic ones based on tool descriptions
        if not prompts and tools:
            # Use first tool's description as a base for vague prompt
            first_desc = tools[0].description[:50] if tools[0].description else ""
            prompts.append((f"Help me with {tag} operations", "vague_action"))
            prompts.append((f"I need {tag} functionality", "underspecified"))
        
        return prompts
    
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
