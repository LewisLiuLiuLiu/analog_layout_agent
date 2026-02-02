"""
Step Executor for Layout Agent Loop
布局代理循环的步骤执行器

Executes individual steps from the workflow plan.
执行工作流计划中的单个步骤。
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
    """
    Context for step execution
    步骤执行的上下文
    """
    mcp_server: "LayoutMCPServer"
    design_dir: Any  # Path / 路径
    constitution: str = ""
    temp_modified_params: Optional[dict] = None
    completed_step_results: list = None  # 新增：已完成步骤的结果
    
    def __post_init__(self):
        if self.completed_step_results is None:
            self.completed_step_results = []


class StepExecutor:
    """
    Step executor - executes individual workflow steps.
    步骤执行器 - 执行单个工作流步骤。
    
    支持两种执行模式：
    1. 目标导向模式（新）：根据 objective 调用 Agent 自主选择工具
    2. 直接调用模式（旧）：直接使用 step.tool 调用
    """
    
    def __init__(self, context: ExecutionContext):
        """
        Initialize step executor.
        初始化步骤执行器。
        
        Args:
            context: Execution context with MCP server reference / 包含 MCP 服务器引用的执行上下文
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
        执行单个步骤。
        
        自动判断执行模式：
        - 如果 step 有 objective 且无 tool → 目标导向模式
        - 如果 step 有 tool → 直接调用模式
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
        
        # 2. 判断执行模式
        use_agent_mode = self._should_use_agent_mode(step)
        
        if use_agent_mode:
            # 目标导向模式：调用 Agent 自主选择工具
            return await self._execute_with_agent(step)
        else:
            # 直接调用模式：使用 step.tool
            return await self._execute_direct(step, override_params)
    
    def _should_use_agent_mode(self, step: StepDefinition) -> bool:
        """判断是否使用 Agent 模式执行"""
        # 有 objective 且无 tool → 使用 Agent 模式
        has_objective = bool(step.objective) if hasattr(step, 'objective') else False
        has_tool = bool(step.tool)
        
        # 新架构优先
        if has_objective and not has_tool:
            return True
        
        # 兼容：如果同时有 objective 和 tool，优先使用 objective
        if has_objective:
            logger.info("Step has both objective and tool, using Agent mode")
            return True
        
        return False
    
    async def _execute_with_agent(self, step: StepDefinition) -> StepResult:
        """使用 Agent 执行步骤（目标导向模式）"""
        from .pydantic_agent import execute_step_with_agent
        
        logger.info(f"Using Agent mode for step {step.step_id}")
        objective = step.objective if hasattr(step, 'objective') else ''
        logger.debug(f"Objective: {objective[:100] if objective else 'N/A'}...")
        
        try:
            # 调用 pydantic_agent 的执行函数
            result = await execute_step_with_agent(
                step=step.model_dump(),
                mcp_server=self.mcp_server,
                completed_results=self.context.completed_step_results
            )
            
            if result.get("success"):
                # 记录成功结果
                self.context.completed_step_results.append(result)
                
                # 验证结果
                verification = await self._verify_step(step, result)
                if verification.passed:
                    return StepResult(
                        success=True,
                        data=result,
                        message=f"Step {step.step_id} completed via Agent"
                    )
                else:
                    return StepResult(
                        success=False,
                        data=result,
                        error={"type": "verification_failed", "message": verification.message}
                    )
            else:
                return StepResult(
                    success=False,
                    data=result,
                    error=result.get("error", {"message": "Agent execution failed"})
                )
                
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return StepResult(
                success=False,
                error={"type": "agent_error", "message": str(e)}
            )
    
    async def _execute_direct(
        self,
        step: StepDefinition,
        override_params: Optional[dict] = None
    ) -> StepResult:
        """直接调用工具执行（兼容旧模式）"""
        logger.info(f"Using direct mode for step {step.step_id}")
        
        # Prepare parameters
        params = override_params if override_params else step.parameters.copy()
        
        # 自动注入 expected_output.component_name 为 name 参数
        # 修复旧格式 workflow_state.json 中缺少 name 参数的问题
        # 当 device-creation 步骤定义了 expected_output.component_name 但 parameters 中没有 name 时，
        # 自动将 component_name 注入为 name 参数，确保组件使用预期的名称创建
        if step.category == "device-creation":
            expected_name = step.expected_output.get("component_name")
            
            # 处理 override_params 可能包含 component_name 而非 name 的情况
            # (Reasoning Agent 的 failure analysis 可能返回 component_name 作为 key)
            if "component_name" in params and "name" not in params:
                params["name"] = params.pop("component_name")
                logger.info(f"Converted component_name to name: {params['name']}")
            
            if expected_name and "name" not in params:
                params["name"] = expected_name
                logger.info(f"Injected component name from expected_output: {expected_name}")
        
        # Execute tool
        try:
            tool_result = await self._call_tool(step.tool, params)
            logger.debug(f"Tool {step.tool} returned: {tool_result}")
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return StepResult(
                success=False,
                error={"type": "execution_error", "message": str(e)}
            )
        
        # Check if tool call succeeded
        if not tool_result.get("success", False):
            return StepResult(
                success=False,
                data=tool_result,
                error={"type": "tool_error", "message": tool_result.get("error", "Tool returned failure")}
            )
        
        # 记录成功结果
        self.context.completed_step_results.append(tool_result)
        
        # Verify result
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
                error={"type": "verification_failed", "message": verification.message}
            )
    
    def _check_dependencies(self, step: StepDefinition, completed: list[bool]) -> bool:
        """
        Check if all dependencies are satisfied
        检查所有依赖是否满足
        """
        for dep_id in step.depends_on:
            # Convert to 0-indexed / 转换为 0 索引
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
        通过 MCP 服务器调用工具。
        
        Args:
            tool_name: Name of the tool to call / 要调用的工具名称
            params: Tool parameters / 工具参数
            
        Returns:
            dict: Tool execution result / 工具执行结果
        """
        logger.debug(f"Calling tool: {tool_name}")
        logger.debug(f"Parameters: {params}")
        
        # Get the tool handler from MCP server
        # 从 MCP 服务器获取工具处理器
        # The MCP server has tools registered - we need to call them
        # MCP 服务器已注册工具 - 我们需要调用它们
        result = self.mcp_server.call_tool(tool_name, params)
        
        return result
    
    async def _verify_step(
        self,
        step: StepDefinition,
        execution_result: dict
    ) -> VerificationResult:
        """
        Verify step execution result.
        验证步骤执行结果。
        
        Args:
            step: Step definition / 步骤定义
            execution_result: Result from tool execution / 工具执行的结果
            
        Returns:
            VerificationResult: Verification result / 验证结果
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
            # 默认: 检查工具是否返回成功
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
        """
        Verify component was created
        验证组件是否已创建
        """
        expected_name = step.expected_output.get("component_name")
        if not expected_name:
            return VerificationResult(passed=True, message="No component name to verify")
        
        # Check in result / 在结果中检查
        # MCP Server 返回的结构是 {"success": True, "data": {...}}
        # 器件创建结果在 data 字段中，需要同时检查顶层和嵌套结构
        data = result.get("data", {})
        created_name = (
            result.get("component_name") or 
            result.get("name") or 
            data.get("component_name") or 
            data.get("name")
        )
        if created_name == expected_name:
            return VerificationResult(
                passed=True,
                message=f"Component {expected_name} created successfully"
            )
        
        # Check in component registry / 在组件注册表中检查
        try:
            layout_ctx = self.mcp_server.state_handler.get_context()
            if layout_ctx:
                # registry.list_all() 返回组件名称字符串列表
                components = layout_ctx.registry.list_all()
                if expected_name in components:
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
        """
        Verify component placement
        验证组件布局
        """
        # MCP Server 返回的结构是 {"success": True, "data": {...}}
        # 需要同时检查顶层和嵌套结构
        data = result.get("data", {})
        
        # Check if placement succeeded / 检查布局是否成功
        placed = (
            result.get("placed") or 
            result.get("success") or
            data.get("placed") or
            data.get("success")
        )
        
        if placed:
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
        """
        Verify routing connection
        验证路由连接
        """
        # MCP Server 返回的结构是 {"success": True, "data": {...}}
        # 需要同时检查顶层和嵌套结构
        data = result.get("data", {})
        
        routed = (
            result.get("routed") or 
            result.get("success") or
            data.get("routed") or
            data.get("success")
        )
        
        if routed:
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
        """
        Verify DRC is clean
        验证 DRC 是否通过
        """
        # MCP Server 返回的结构是 {"success": True, "data": {...}}
        # 需要同时检查顶层和嵌套结构
        data = result.get("data", {})
        
        drc_clean = (
            result.get("drc_clean") or 
            result.get("clean", False) or
            data.get("drc_clean") or
            data.get("clean", False)
        )
        
        error_count = result.get("error_count", data.get("error_count", -1))
        
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
        """
        Verify file was created
        验证文件是否已创建
        """
        from pathlib import Path
        
        output_path = step.parameters.get("output_path")
        if output_path:
            path = self.context.design_dir / output_path
            if path.exists():
                return VerificationResult(
                    passed=True,
                    message=f"File created: {output_path}"
                )
        
        # MCP Server 返回的结构是 {"success": True, "data": {...}}
        # 需要同时检查顶层和嵌套结构
        data = result.get("data", {})
        
        # Check result / 检查结果
        exported = (
            result.get("exported") or 
            result.get("success") or
            data.get("exported") or
            data.get("success") or
            data.get("path")  # 导出工具返回 path 字段表示成功
        )
        
        if exported:
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
        获取用于失败分析的完整布局上下文。
        
        Returns:
            dict: Complete context state / 完整的上下文状态
        """
        layout_ctx = self.mcp_server.state_handler.get_context()
        
        if not layout_ctx:
            return {
                "pdk": "unknown",
                "components": [],
                "design_dir": str(self.context.design_dir) if self.context.design_dir else None
            }
        
        try:
            components = layout_ctx.registry.list_all()
        except Exception:
            components = []
        
        return {
            "pdk": layout_ctx.pdk_name if hasattr(layout_ctx, 'pdk_name') else "unknown",
            "components": components,
            "design_dir": str(self.context.design_dir) if self.context.design_dir else None
        }
