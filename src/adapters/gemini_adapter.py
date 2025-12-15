"""
Adapter for converting MCP tool definitions to Gemini function declarations.

Gemini uses OpenAPI-compatible schemas for function definitions.
This adapter handles the conversion from our internal Tool format.
"""
from typing import Any
from google.genai import types

from src.tools.base import Tool


class GeminiAdapter:
    """
    Converts Tool objects to Gemini-compatible function declarations.
    
    Gemini's function calling uses a subset of OpenAPI schema format.
    """
    
    @staticmethod
    def tool_to_function_declaration(tool: Tool) -> dict[str, Any]:
        """
        Convert a Tool to a Gemini function declaration dict.
        
        Args:
            tool: Tool object to convert
            
        Returns:
            Dictionary compatible with Gemini's function declaration format
        """
        return tool.to_gemini_function()
    
    @staticmethod
    def tools_to_gemini_tools(tools: list[Tool]) -> types.Tool:
        """
        Convert a list of Tools to a Gemini Tool object.
        
        Args:
            tools: List of Tool objects
            
        Returns:
            google.genai.types.Tool object with all function declarations
        """
        function_declarations = [
            GeminiAdapter.tool_to_function_declaration(tool)
            for tool in tools
        ]
        
        return types.Tool(function_declarations=function_declarations)
    
    @staticmethod
    def extract_function_call(response) -> dict[str, Any] | None:
        """
        Extract function call information from a Gemini response.
        
        Args:
            response: Gemini API response object
            
        Returns:
            Dictionary with 'name' and 'args' if a function was called, None otherwise
        """
        try:
            if not response.candidates:
                return None
            
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                return None
            
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    return {
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {}
                    }
            
            return None
        except Exception:
            return None
    
    @staticmethod
    def extract_all_function_calls(response) -> list[dict[str, Any]]:
        """
        Extract all function calls from a Gemini response (for parallel calling).
        
        Args:
            response: Gemini API response object
            
        Returns:
            List of dictionaries with 'name' and 'args' for each function call
        """
        calls = []
        try:
            if not response.candidates:
                return calls
            
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                return calls
            
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    calls.append({
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {}
                    })
            
            return calls
        except Exception:
            return calls
