"""
MCP (Model Context Protocol) methodology.

The standard approach where all tools are passed to the LLM at once
and it selects which tool(s) to call in a single step.
"""
from typing import Optional
from loguru import logger

from src.tools.base import Tool
from src.clients.base import BaseLLMClient
from .base import BaseMethodology, MethodologyResult, StepInfo, StepType, StepBasedMethodology


class MCPMethodology(BaseMethodology):
    """
    Model Context Protocol methodology.
    
    All tools are passed to the LLM in the context at once.
    The LLM selects which tool to call in a single API call.
    
    This is the baseline/standard approach for tool calling.
    """
    
    NAME: str = "mcp"
    DECLINE_TOOL: str = StepBasedMethodology.DECLINE_TOOL
    
    def __init__(self, allow_no_tool_call: bool = False):
        """
        Initialize MCP methodology.
        
        Args:
            allow_no_tool_call: If True, add a decline option for cases
                               where no tool call is needed.
        """
        self.allow_no_tool_call = allow_no_tool_call
        logger.debug(f"[MCP] Initialized with allow_no_tool_call={allow_no_tool_call}")
    
    def run_single(
        self,
        prompt: str,
        tools: list[Tool],
        client: BaseLLMClient,
        system_instruction: Optional[str] = None,
    ) -> MethodologyResult:
        """
        Run MCP methodology for a single prompt.
        
        Passes all tools to the LLM and gets the tool selection.
        
        Args:
            prompt: User prompt to process
            tools: All available tools
            client: LLM client to use
            system_instruction: Optional system instruction
            
        Returns:
            MethodologyResult with tool selection
        """
        logger.debug(f"[MCP] Running with prompt: {prompt[:100]}...")
        logger.debug(f"[MCP] Number of tools available: {len(tools)}")
        logger.debug(f"[MCP] Tool names: {[t.name for t in tools]}")
        
        # Optionally add decline pseudo-tool
        tools_to_use = list(tools)
        if self.allow_no_tool_call:
            decline_tool = Tool(
                name=self.DECLINE_TOOL,
                description="Indicate that no tool call is needed for this request.",
                category="system",
                parameters=[],
                tags=["system"],
                complexity="simple",
            )
            tools_to_use.append(decline_tool)
            logger.debug(f"[MCP] Added decline tool, total tools: {len(tools_to_use)}")
        
        if system_instruction:
            logger.debug(f"[MCP] System instruction: {system_instruction[:200]}...")
        
        # Make the API call
        logger.debug(f"[MCP] Making API call to {client.PROVIDER_NAME}...")
        call_result = client.call_with_tools(
            prompt=prompt,
            tools=tools_to_use,
            system_instruction=system_instruction,
        )
        
        logger.debug(f"[MCP] API call completed in {call_result.latency_ms:.1f}ms")
        logger.debug(f"[MCP] Success: {call_result.success}")
        logger.debug(f"[MCP] Called tool: {call_result.called_tool}")
        logger.debug(f"[MCP] Called args: {call_result.called_args}")
        if call_result.error:
            logger.debug(f"[MCP] Error: {call_result.error}")
        if call_result.all_calls:
            logger.debug(f"[MCP] All calls: {call_result.all_calls}")
        
        # Check for decline
        declined = False
        called_tool = call_result.called_tool
        if called_tool == self.DECLINE_TOOL:
            declined = True
            called_tool = None
            logger.debug(f"[MCP] LLM declined to call any tool")
        
        # Create step info
        step = StepInfo(
            step_number=1,
            step_type=StepType.DECLINE if declined else StepType.SELECT_TOOL,
            selection=call_result.called_tool,
            latency_ms=call_result.latency_ms,
            raw_response=call_result.raw_response,
            error=call_result.error,
        )
        
        result = MethodologyResult(
            success=call_result.success,
            called_tool=called_tool,
            called_args=call_result.called_args,
            all_calls=call_result.all_calls,
            latency_ms=call_result.latency_ms,
            error=call_result.error,
            raw_response=call_result.raw_response,
            model=call_result.model,
            provider=call_result.provider,
            methodology=self.NAME,
            steps=[step],
            categories_selected=[],
            backtrack_count=0,
            declined_tool_call=declined,
            final_category=None,
        )
        
        logger.debug(f"[MCP] Final result - tool: {result.called_tool}, success: {result.success}")
        return result
