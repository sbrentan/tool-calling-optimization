"""Base tool classes and schemas following MCP protocol."""
from typing import Any, Optional
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """A parameter for a tool, following JSON Schema format."""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    enum: Optional[list[str]] = None
    default: Optional[Any] = None


class Tool(BaseModel):
    """
    A tool definition following MCP (Model Context Protocol) format.
    
    MCP tools use JSON-RPC 2.0 with JSON Schema for parameters.
    This class represents a single callable tool with its metadata.
    """
    name: str = Field(..., description="Unique identifier for the tool")
    description: str = Field(..., description="Human-readable description of what the tool does")
    category: str = Field(default="general", description="Tool category for grouping")
    parameters: list[ToolParameter] = Field(default_factory=list)
    
    # Metadata for testing
    tags: list[str] = Field(default_factory=list, description="Tags for similarity grouping")
    complexity: str = Field(default="simple", description="Complexity level: simple, medium, complex")
    
    def to_mcp_schema(self) -> dict[str, Any]:
        """Convert to MCP tool schema format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
                
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def to_gemini_function(self) -> dict[str, Any]:
        """Convert to Gemini function declaration format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            # Map types to Gemini-compatible types
            gemini_type = param.type
            if gemini_type == "integer":
                gemini_type = "integer"
            elif gemini_type == "number":
                gemini_type = "number"
            elif gemini_type == "boolean":
                gemini_type = "boolean"
            elif gemini_type == "array":
                gemini_type = "array"
            else:
                gemini_type = "string"
            
            prop = {
                "type": gemini_type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
                
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
            }
        }
        
        if required:
            schema["parameters"]["required"] = required
            
        return schema


class TestCase(BaseModel):
    """A test case for tool calling evaluation."""
    prompt: str = Field(..., description="User prompt to send to the model")
    expected_tool: Optional[str] = Field(default=None, description="Name of the expected tool to be called. None means no tool should be called.")
    expected_params: Optional[dict[str, Any]] = Field(default=None, description="Expected parameters")
    category: str = Field(default="general", description="Test category")
    difficulty: str = Field(default="easy", description="Difficulty: easy, medium, hard")
    description: str = Field(default="", description="Description of what this test verifies")
    prompt_type: str = Field(default="concise", description="Prompt type: concise or clear")
    
    # No-tool test case fields
    no_tool_reason: Optional[str] = Field(default=None, description="Why no tool should be called (for no-tool test cases)")
    
    @property
    def expects_tool_call(self) -> bool:
        """Whether this test case expects a tool to be called."""
        return self.expected_tool is not None


class MultiToolTestCase(BaseModel):
    """
    A test case that expects multiple tools to be called.
    
    Used for testing scenarios where a user request requires
    calling multiple tools in sequence or parallel.
    """
    prompt: str = Field(..., description="User prompt to send to the model")
    expected_tools: list[str] = Field(..., description="Ordered list of expected tools to be called")
    expected_params: Optional[list[dict[str, Any]]] = Field(default=None, description="Expected parameters for each tool (same order as expected_tools)")
    require_sequence: bool = Field(default=False, description="If True, tools must be called in exact order. If False, treated as a set.")
    category: str = Field(default="multi_tool", description="Test category")
    difficulty: str = Field(default="hard", description="Difficulty: easy, medium, hard")
    description: str = Field(default="", description="Description of what this test verifies")
    prompt_type: str = Field(default="concise", description="Prompt type: concise or clear")
    
    @property
    def expects_tool_call(self) -> bool:
        """Multi-tool test cases always expect at least one tool call."""
        return len(self.expected_tools) > 0
    
    @property
    def primary_tool(self) -> Optional[str]:
        """Return the first expected tool (for compatibility with single-tool metrics)."""
        return self.expected_tools[0] if self.expected_tools else None
