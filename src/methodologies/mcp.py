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
    CLARIFICATION_TOOL: str = StepBasedMethodology.CLARIFICATION_TOOL
    
    def __init__(
        self,
        allow_no_tool_call: bool = False,
        allow_clarification: bool = False,
    ):
        """
        Initialize MCP methodology.
        
        Args:
            allow_no_tool_call: If True, add a decline option for cases
                               where no tool call is needed.
            allow_clarification: If True, add a clarification option for
                                ambiguous requests.
        """
        self.allow_no_tool_call = allow_no_tool_call
        self.allow_clarification = allow_clarification
        logger.debug(f"[MCP] Initialized with allow_no_tool_call={allow_no_tool_call}, allow_clarification={allow_clarification}")
    
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
        
        # Build tools list with optional pseudo-tools
        tools_to_use = list(tools)
        
        # Optionally add decline pseudo-tool
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
        
        # Optionally add clarification pseudo-tool
        if self.allow_clarification:
            from src.tools.base import ToolParameter
            clarification_tool = Tool(
                name=self.CLARIFICATION_TOOL,
                description="Request clarification when the user's request is ambiguous "
                           "and could match multiple tools. Use this when you cannot "
                           "determine which tool the user wants with high confidence.",
                category="system",
                parameters=[
                    ToolParameter(
                        name="question",
                        type="string",
                        description="The clarifying question to ask the user",
                        required=True,
                    ),
                    ToolParameter(
                        name="candidate_tools",
                        type="array",
                        description="List of tool names that could potentially match the request",
                        required=True,
                    ),
                ],
                tags=["system"],
                complexity="simple",
            )
            tools_to_use.append(clarification_tool)
            logger.debug(f"[MCP] Added clarification tool, total tools: {len(tools_to_use)}")
        
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
        
        # Check for decline or clarification
        declined = False
        clarification_requested = False
        clarification_question = None
        candidate_tools = []
        called_tool = call_result.called_tool
        step_type = StepType.SELECT_TOOL
        
        if called_tool == self.DECLINE_TOOL:
            declined = True
            called_tool = None
            step_type = StepType.DECLINE
            logger.debug(f"[MCP] LLM declined to call any tool")
        elif called_tool == self.CLARIFICATION_TOOL:
            clarification_requested = True
            called_tool = None
            step_type = StepType.CLARIFICATION
            # Extract clarification details from args
            args = call_result.called_args or {}
            clarification_question = args.get("question", "")
            candidate_tools = args.get("candidate_tools", [])
            # Ensure candidate_tools is a list
            if isinstance(candidate_tools, str):
                candidate_tools = [candidate_tools]
            logger.debug(f"[MCP] LLM requested clarification: {clarification_question}")
            logger.debug(f"[MCP] Candidate tools: {candidate_tools}")
        
        # Create step info
        step = StepInfo(
            step_number=1,
            step_type=step_type,
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
            clarification_requested=clarification_requested,
            clarification_question=clarification_question,
            candidate_tools=candidate_tools,
            # Token usage from the API call
            tokens_input=call_result.tokens_input,
            tokens_output=call_result.tokens_output,
            tokens_total=call_result.tokens_total,
        )
        
        logger.debug(f"[MCP] Final result - tool: {result.called_tool}, success: {result.success}, clarification: {clarification_requested}")
        return result
