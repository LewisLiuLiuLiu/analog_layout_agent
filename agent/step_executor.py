"""
Step Executor for Layout Agent Loop

Executes individual steps from the workflow plan.
"""

import logging
from typing import TYPE_CHECKING, Optional, Any
from dataclasses import dataclass

from ..state.models import StepDefinition, StepResult, VerificationResult

if TYPE_CHECKING:
    from ..mcp_server.server import LayoutMCPServer

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context for step execution"""
    mcp_server: "LayoutMCPServer"
    design_dir: Any  # Path
    constitution: str = ""
    temp_modified_params: Optional[dict] = None


class StepExecutor:
    """
    Step executor - executes individual workflow steps.
    
    Responsibilities:
    1. Check step dependencies
    2. Execute tool calls
    3. Verify execution results
    """
    
    def __init__(self, context: ExecutionContext):
        """
        Initialize step executor.
        
        Args:
            context: Execution context with MCP server reference
        """
        self.context = context
        self.mcp_server = context.mcp_server
    
    async def execute_step(
        self,
        step: StepDefinition,
        completed: list[bool],
        override_params: Optional[dict] = None
    ) -> StepResult:
        """
        Execute a single step.
        
        Process:
        1. Check if dependencies are satisfied
        2. Execute the tool call
        3. Verify execution result
        4. Return result
        
        Args:
            step: Step definition to execute
            completed: Current completion status array
            override_params: Optional parameter override (from failure recovery)
            
        Returns:
            StepResult: Execution result
        """
        logger.info(f"Executing step {step.step_id}: {step.description}")
        
        # 1. Check dependencies
        if not self._check_dependencies(step, completed):
            return StepResult(
                success=False,
                error={
                    "type": "dependency_not_met",
                    "message": f"Dependencies not satisfied: {step.depends_on}"
                }
            )
        
        # 2. Prepare parameters
        params = override_params if override_params else step.parameters.copy()
        
        # 3. Execute tool
        try:
            tool_result = await self._call_tool(step.tool, params)
            logger.debug(f"Tool {step.tool} returned: {tool_result}")
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return StepResult(
                success=False,
                error={
                    "type": "execution_error",
                    "message": str(e)
                }
            )
        
        # 4. Check if tool call succeeded
        if not tool_result.get("success", False):
            return StepResult(
                success=False,
                data=tool_result,
                error={
                    "type": "tool_error",
                    "message": tool_result.get("error", "Tool returned failure")
                }
            )
        
        # 5. Verify result
        verification = await self._verify_step(step, tool_result)
        
        if verification.passed:
            return StepResult(
                success=True,
                data=tool_result,
                message=f"Step {step.step_id} completed and verified"
            )
        else:
            return StepResult(
                success=False,
                data=tool_result,
                error={
                    "type": "verification_failed",
                    "message": verification.message
                }
            )
    
    def _check_dependencies(self, step: StepDefinition, completed: list[bool]) -> bool:
        """Check if all dependencies are satisfied"""
        for dep_id in step.depends_on:
            # Convert to 0-indexed
            dep_index = dep_id - 1
            if dep_index < 0 or dep_index >= len(completed):
                logger.error(f"Invalid dependency: {dep_id}")
                return False
            if not completed[dep_index]:
                logger.warning(f"Dependency {dep_id} not completed")
                return False
        return True
    
    async def _call_tool(self, tool_name: str, params: dict) -> dict:
        """
        Call a tool through the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            params: Tool parameters
            
        Returns:
            dict: Tool execution result
        """
        logger.debug(f"Calling tool: {tool_name}")
        logger.debug(f"Parameters: {params}")
        
        # Get the tool handler from MCP server
        # The MCP server has tools registered - we need to call them
        result = self.mcp_server.call_tool(tool_name, params)
        
        return result
    
    async def _verify_step(
        self,
        step: StepDefinition,
        execution_result: dict
    ) -> VerificationResult:
        """
        Verify step execution result.
        
        Args:
            step: Step definition
            execution_result: Result from tool execution
            
        Returns:
            VerificationResult: Verification result
        """
        verification_type = step.verification.type
        
        if verification_type == "component_exists":
            return await self._verify_component_exists(step, execution_result)
        elif verification_type == "placement_check":
            return await self._verify_placement(step, execution_result)
        elif verification_type == "routing_check":
            return await self._verify_routing(step, execution_result)
        elif verification_type == "drc_clean":
            return await self._verify_drc(step, execution_result)
        elif verification_type == "file_exists":
            return await self._verify_file_exists(step, execution_result)
        else:
            # Default: check if tool returned success
            passed = execution_result.get("success", False)
            return VerificationResult(
                passed=passed,
                message=execution_result.get("message", "")
            )
    
    async def _verify_component_exists(
        self,
        step: StepDefinition,
        result: dict
    ) -> VerificationResult:
        """Verify component was created"""
        expected_name = step.expected_output.get("component_name")
        if not expected_name:
            return VerificationResult(passed=True, message="No component name to verify")
        
        # Check in result
        created_name = result.get("component_name") or result.get("name")
        if created_name == expected_name:
            return VerificationResult(
                passed=True,
                message=f"Component {expected_name} created successfully"
            )
        
        # Check in component registry
        try:
            layout_ctx = self.mcp_server.state_handler.get_context()
            if layout_ctx:
                components = layout_ctx.component_registry.list_components()
                if expected_name in [c["name"] for c in components]:
                    return VerificationResult(
                        passed=True,
                        message=f"Component {expected_name} found in registry"
                    )
        except Exception as e:
            logger.warning(f"Could not check component registry: {e}")
        
        return VerificationResult(
            passed=False,
            message=f"Component {expected_name} not found"
        )
    
    async def _verify_placement(
        self,
        step: StepDefinition,
        result: dict
    ) -> VerificationResult:
        """Verify component placement"""
        # Check if placement succeeded
        if result.get("placed") or result.get("success"):
            return VerificationResult(
                passed=True,
                message="Placement completed"
            )
        return VerificationResult(
            passed=False,
            message="Placement verification failed"
        )
    
    async def _verify_routing(
        self,
        step: StepDefinition,
        result: dict
    ) -> VerificationResult:
        """Verify routing connection"""
        if result.get("routed") or result.get("success"):
            return VerificationResult(
                passed=True,
                message="Routing completed"
            )
        return VerificationResult(
            passed=False,
            message="Routing verification failed"
        )
    
    async def _verify_drc(
        self,
        step: StepDefinition,
        result: dict
    ) -> VerificationResult:
        """Verify DRC is clean"""
        drc_clean = result.get("drc_clean") or result.get("clean", False)
        error_count = result.get("error_count", -1)
        
        if drc_clean or error_count == 0:
            return VerificationResult(
                passed=True,
                message="DRC clean"
            )
        return VerificationResult(
            passed=False,
            message=f"DRC errors found: {error_count}"
        )
    
    async def _verify_file_exists(
        self,
        step: StepDefinition,
        result: dict
    ) -> VerificationResult:
        """Verify file was created"""
        from pathlib import Path
        
        output_path = step.parameters.get("output_path")
        if output_path:
            path = self.context.design_dir / output_path
            if path.exists():
                return VerificationResult(
                    passed=True,
                    message=f"File created: {output_path}"
                )
        
        # Check result
        if result.get("exported") or result.get("success"):
            return VerificationResult(
                passed=True,
                message="Export completed"
            )
        
        return VerificationResult(
            passed=False,
            message=f"File not found: {output_path}"
        )
    
    def get_full_context(self) -> dict:
        """
        Get full layout context for failure analysis.
        
        Returns:
            dict: Complete context state
        """
        layout_ctx = self.mcp_server.state_handler.get_context()
        
        if not layout_ctx:
            return {
                "pdk": "unknown",
                "components": [],
                "design_dir": str(self.context.design_dir) if self.context.design_dir else None
            }
        
        try:
            components = layout_ctx.component_registry.list_components()
        except Exception:
            components = []
        
        return {
            "pdk": layout_ctx.pdk_name if hasattr(layout_ctx, 'pdk_name') else "unknown",
            "components": components,
            "design_dir": str(self.context.design_dir) if self.context.design_dir else None
        }
