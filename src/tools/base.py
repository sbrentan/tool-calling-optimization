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
    expected_tool: str = Field(..., description="Name of the expected tool to be called")
    expected_params: Optional[dict[str, Any]] = Field(default=None, description="Expected parameters")
    category: str = Field(default="general", description="Test category")
    difficulty: str = Field(default="easy", description="Difficulty: easy, medium, hard")
    description: str = Field(default="", description="Description of what this test verifies")
    prompt_type: str = Field(default="concise", description="Prompt type: concise or clear")
