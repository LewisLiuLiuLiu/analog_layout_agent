"""
Workflow Graph for Layout Agent Loop
工作流图 - 用于布局代理循环

Uses pydantic-graph to manage workflow state machine.
使用 pydantic-graph 管理工作流状态机。
"""

import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Any, Dict, List

# Try to import pydantic-graph
# 尝试导入 pydantic-graph 库
try:
    from pydantic_graph import Graph, BaseNode, End, GraphRunContext
    PYDANTIC_GRAPH_AVAILABLE = True
except ImportError:
    PYDANTIC_GRAPH_AVAILABLE = False
    # Provide stub classes for type hints
    # 提供用于类型提示的占位类
    class Graph:
        pass
    class BaseNode:
        pass
    class End:
        pass
    class GraphRunContext:
        pass

from ..state.models import LayoutWorkflowState, StepResult

if TYPE_CHECKING:
    from .step_executor import StepExecutor
    from .reasoning_agent import ReasoningAgent

# Import load_constitution for automatic loading
# 导入 load_constitution 用于自动加载宪法配置
from .reasoning_agent import load_constitution

logger = logging.getLogger(__name__)


# ============================================================================
# Dependencies / 依赖项
# ============================================================================

@dataclass
class LayoutAgentDeps:
    """Dependencies for workflow graph nodes
    工作流图节点的依赖项
    
    Constitution is automatically loaded via __post_init__ if not provided.
    如果未提供，宪法配置将通过 __post_init__ 自动加载。
    """
    design_dir: Path
    step_executor: "StepExecutor"
    reasoning_agent: Optional["ReasoningAgent"] = None
    constitution: str = ""
    temp_modified_params: Optional[dict] = None
    init_status: Dict[str, Any] = field(default_factory=dict)
    completed_step_results: List[Dict] = field(default_factory=list)  # 新增：已完成步骤的结果
    
    def __post_init__(self):
        """Auto-load constitution if not provided
        如果未提供则自动加载宪法配置"""
        if not self.constitution:
            self.constitution = load_constitution()
            if self.constitution:
                logger.info(f"Constitution auto-loaded: {len(self.constitution)} chars")
            else:
                logger.warning("Constitution not loaded - workflow may not follow all rules")
    
    def record_init_executed(self, success: bool, output: str = ""):
        """Record init.sh execution status (for LLM awareness)
        记录 init.sh 执行状态（供 LLM 感知）"""
        self.init_status["init_sh_executed"] = True
        self.init_status["init_sh_success"] = success
        self.init_status["init_sh_output"] = output[:200] if output else ""
    
    def record_progress_read(self, last_lines: List[str] = None):
        """Record progress.md read status (for LLM awareness)
        记录 progress.md 读取状态（供 LLM 感知）"""
        self.init_status["progress_read"] = True
        if last_lines is not None:
            self.init_status["progress_last_lines"] = last_lines
    
    async def call_tool(self, tool_name: str, params: dict) -> dict:
        """
        Call a tool through step executor
        通过步骤执行器调用工具
        """
        from ..state.models import StepDefinition, VerificationConfig
        
        # Create a minimal step definition for the executor
        # 为执行器创建最小化的步骤定义
        step = StepDefinition(
            step_id=0,
            category="direct-call",
            description=f"Direct call to {tool_name}",
            tool=tool_name,
            parameters=params,
            verification=VerificationConfig(type="default", conditions=[])
        )
        
        result = await self.step_executor.execute_step(step, [True] * 100)  # Bypass dependency check / 绕过依赖检查
        return result.to_dict() if hasattr(result, 'to_dict') else {"success": result.success}
    
    async def verify_step(self, step: dict, tool_result: dict) -> bool:
        """
        Verify a step execution
        验证步骤执行结果
        """
        # Simple verification - check if tool succeeded
        # 简单验证 - 检查工具是否执行成功
        return tool_result.get("success", False)
    
    async def save_workflow_state(self, state: LayoutWorkflowState) -> None:
        """
        Save workflow state to file
        保存工作流状态到文件
        """
        from ..state.workflow_manager import save_workflow_state
        
        workflow = state.to_workflow_state()
        save_workflow_state(workflow, self.design_dir / "workflow_state.json")
    
    async def append_progress(self, step: dict, result: dict) -> None:
        """
        Append progress record
        追加进度记录
        """
        from ..state.progress_writer import append_session_record
        from ..state.models import StepDefinition, StepResult, VerificationConfig
        
        # Convert dict to StepDefinition
        # 将字典转换为 StepDefinition 对象
        if isinstance(step.get('verification'), dict):
            step['verification'] = VerificationConfig(**step['verification'])
        step_def = StepDefinition(**step)
        step_result = StepResult(
            success=result.get("success", False),
            data=result.get("data"),
            message=result.get("message", "")
        )
        
        progress_path = self.design_dir / "progress.md"
        if progress_path.exists():
            append_session_record(progress_path, step_def, step_result)
    
    def get_full_context(self) -> dict:
        """
        Get full layout context
        获取完整的布局上下文
        """
        return self.step_executor.get_full_context()


# ============================================================================
# Graph Nodes (only defined if pydantic-graph is available)
# 图节点      （仅当 pydantic-graph 可用时定义）
# ============================================================================

if PYDANTIC_GRAPH_AVAILABLE:
    
    @dataclass
    class InitNode(BaseNode[LayoutWorkflowState, LayoutAgentDeps, str]):
        """
        Initialization node - runs init.sh, reads progress, determines next step.
        初始化节点 - 运行 init.sh，读取进度，确定下一步骤。
        
        宪法第一条强制执行点 + 状态记录（供后续 LLM session 感知）
        Constitution Article 1 enforcement point + status recording (for subsequent LLM session awareness)
        """
        
        async def run(
            self, 
            ctx: GraphRunContext[LayoutWorkflowState, LayoutAgentDeps]
        ) -> "ExecuteStepNode | End[str]":
            design_dir = ctx.deps.design_dir
            
            # 1. Execute init.sh (Constitution 1.1)
            # 1. 执行 init.sh (宪法 1.1)
            logger.info("[Constitution 1.1] Executing init.sh...")
            init_script = design_dir / "init.sh"
            
            if init_script.exists():
                result = subprocess.run(
                    ["bash", str(init_script)],
                    cwd=design_dir,
                    capture_output=True,
                    text=True
                )
                success = (result.returncode == 0)
                
                # Record status for subsequent LLM session awareness
                # 记录状态供后续 LLM session 感知
                ctx.deps.record_init_executed(
                    success=success,
                    output=result.stdout or result.stderr
                )
                
                if not success:
                    logger.error(f"init.sh failed: {result.stderr}")
                    return End(f"Initialization failed: {result.stderr}")
                logger.info(f"init.sh completed (exit code: {result.returncode})")
            else:
                logger.warning(f"init.sh not found at {init_script}")
            
            # 2. Read progress.md (Constitution 1.2)
            # 2. 读取 progress.md (宪法 1.2)
            progress_file = design_dir / "progress.md"
            if progress_file.exists() and progress_file.stat().st_size > 0:
                logger.debug("Reading progress.md last 50 lines...")
                lines = progress_file.read_text().splitlines()
                last_50 = lines[-50:] if len(lines) > 50 else lines
                
                # Record status for subsequent LLM session awareness
                # 记录状态供后续 LLM session 感知
                ctx.deps.record_progress_read(last_50)
                
                logger.debug(f"Progress summary: {len(last_50)} lines")
            else:
                logger.info("First run, skipping progress.md read")
            
            # 3. Constitution status logging
            # 3. 宪法状态日志
            if ctx.deps.constitution:
                logger.debug(f"Constitution available: {len(ctx.deps.constitution)} chars")
            else:
                logger.warning("Constitution NOT loaded - agent rules may not be enforced")
            
            # 4. Find the first False step (Constitution 1.3)
            # 4. 找第一个 False 步骤 (宪法 1.3)
            first_false = next(
                (i for i, done in enumerate(ctx.state.completed) if not done),
                None
            )
            
            if first_false is None:
                logger.info("All steps completed!")
                return End("All steps completed")
            
            ctx.state.current_step_index = first_false
            ctx.state.reset_retry_count()
            step = ctx.state.steps[first_false]
            logger.info(f"Next step: Step {step['step_id']} - {step['description']}")
            
            return ExecuteStepNode()
    
    
    @dataclass
    class ExecuteStepNode(BaseNode[LayoutWorkflowState, LayoutAgentDeps, str]):
        """
        Execute the current step using StepExecutor
        使用 StepExecutor 执行当前步骤
        
        支持两种执行模式：
        1. 目标导向模式：step 有 objective，通过 StepExecutor 调用 Agent 自主选择工具
        2. 直接调用模式：step 有 tool，直接调用工具
        
        执行模式由 StepExecutor 自动判断。
        """
        
        async def run(
            self,
            ctx: GraphRunContext[LayoutWorkflowState, LayoutAgentDeps]
        ) -> "VerifyStepNode | FailureAnalysisNode":
            step = ctx.state.steps[ctx.state.current_step_index]
            
            step_id = step.get('step_id', ctx.state.current_step_index + 1)
            description = step.get('description', 'Unknown step')
            objective = step.get('objective', '')
            
            logger.info(f"Executing step {step_id}: {description}")
            if objective:
                logger.debug(f"Objective: {objective[:100]}...")
            
            # 将已完成步骤结果传递给执行器
            ctx.deps.step_executor.context.completed_step_results = ctx.deps.completed_step_results
            
            # 使用 modified params 如果有的话（来自失败恢复）
            override_params = ctx.deps.temp_modified_params
            ctx.deps.temp_modified_params = None
            
            try:
                # 调用 StepExecutor（自动判断执行模式）
                from ..state.models import StepDefinition, VerificationConfig
                
                # 转换 step dict 为 StepDefinition
                step_copy = step.copy()
                if isinstance(step_copy.get('verification'), dict):
                    step_copy['verification'] = VerificationConfig(**step_copy['verification'])
                step_def = StepDefinition(**step_copy)
                
                result = await ctx.deps.step_executor.execute_step(
                    step_def,
                    ctx.state.completed,
                    override_params
                )
                
                # 转换结果格式
                result_dict = result.to_dict() if hasattr(result, 'to_dict') else {
                    'success': result.success,
                    'data': result.data,
                    'message': result.message,
                    'error': result.error
                }
                
                if result.success:
                    # 记录成功结果
                    ctx.deps.completed_step_results.append(result_dict)
                    return VerifyStepNode(tool_result=result_dict)
                else:
                    error_msg = result.error
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get('message', 'Unknown error')
                    return FailureAnalysisNode(error=str(error_msg))
                    
            except Exception as e:
                logger.error(f"Step execution failed: {e}")
                return FailureAnalysisNode(error=str(e))
    
    
    @dataclass
    class VerifyStepNode(BaseNode[LayoutWorkflowState, LayoutAgentDeps, str]):
        """Verify step execution and update state
        验证步骤执行并更新状态"""
        tool_result: dict = field(default_factory=dict)
        
        async def run(
            self,
            ctx: GraphRunContext[LayoutWorkflowState, LayoutAgentDeps]
        ) -> "InitNode | End[str]":
            step = ctx.state.steps[ctx.state.current_step_index]
            
            logger.info(f"Verifying step: {step['verification']['type']}")
            
            # Execute verification
            # 执行验证
            verification_passed = await ctx.deps.verify_step(step, self.tool_result)
            
            if verification_passed:
                logger.info("Verification passed")
                
                # Update state: False -> True (only allowed modification)
                # 更新状态: False -> True (唯一允许的修改)
                ctx.state.completed[ctx.state.current_step_index] = True
                logger.info(f"Updated state: Step {step['step_id']} -> true")
                
                # Save state to file
                # 保存状态到文件
                await ctx.deps.save_workflow_state(ctx.state)
                
                # Update progress.md
                # 更新 progress.md
                await ctx.deps.append_progress(step, self.tool_result)
                
                # Check if all completed
                # 检查是否全部完成
                if all(ctx.state.completed):
                    logger.info("=== All steps completed ===")
                    return End("Design completed")
                
                # Continue to next step
                # 继续下一步骤
                return InitNode()
            else:
                logger.warning("Verification failed")
                return FailureAnalysisNode(error="Verification failed")
    
    
    @dataclass
    class FailureAnalysisNode(BaseNode[LayoutWorkflowState, LayoutAgentDeps, str]):
        """Analyze failure and attempt recovery
        分析失败原因并尝试恢复"""
        error: str = ""
        
        async def run(
            self,
            ctx: GraphRunContext[LayoutWorkflowState, LayoutAgentDeps]
        ) -> "ExecuteStepNode | End[str]":
            step = ctx.state.steps[ctx.state.current_step_index]
            
            ctx.state.retry_count += 1
            logger.info(f"Failure analysis: retry {ctx.state.retry_count}/{ctx.state.max_retries}")
            
            if ctx.state.retry_count >= ctx.state.max_retries:
                logger.error("Max retries reached, task failed")
                return End(f"Step {step['step_id']} failed: {self.error}")
            
            # Call Reasoning Agent for analysis
            # 调用推理代理进行分析
            if ctx.deps.reasoning_agent:
                logger.info("Calling Reasoning Agent for failure analysis...")
                try:
                    analysis = await ctx.deps.reasoning_agent.analyze_failure(
                        failed_step=step,
                        error_info={"error": self.error},
                        layout_context=ctx.deps.get_full_context()
                    )
                    
                    if analysis.recoverable:
                        logger.info(f"Analysis: recoverable - {analysis.analysis[:100]}...")
                        if analysis.modified_step:
                            params = analysis.modified_step.get('parameters')
                            if params is None and 'component_name' in analysis.modified_step:
                                params = analysis.modified_step
                                logger.warning("Modified step missing 'parameters' field, using directly")
                            if params:
                                logger.info(f"Using modified parameters: {params}")
                                ctx.deps.temp_modified_params = params
                        return ExecuteStepNode()
                    else:
                        logger.error(f"Analysis: NOT recoverable - {analysis.recommendation}")
                        return End(f"Unrecoverable error: {analysis.recommendation}")
                        
                except Exception as e:
                    logger.error(f"Reasoning Agent failed: {e}")
            
            # Retry without modification
            # 不做修改直接重试
            return ExecuteStepNode()
    
    
    # Build workflow graph
    # 构建工作流图
    layout_workflow_graph = Graph(
        nodes=[InitNode, ExecuteStepNode, VerifyStepNode, FailureAnalysisNode]
    )

else:
    # Stubs when pydantic-graph is not available
    # 当 pydantic-graph 不可用时的占位符
    layout_workflow_graph = None
    InitNode = None
    ExecuteStepNode = None
    VerifyStepNode = None
    FailureAnalysisNode = None


# ============================================================================
# Workflow Runner / 工作流运行器
# ============================================================================

async def run_workflow(
    design_dir: Path,
    deps: LayoutAgentDeps,
    max_iterations: int = 10
) -> str:
    """
    Run the workflow graph.
    运行工作流图。
    
    Args:
        design_dir: Design directory path / 设计目录路径
        deps: Workflow dependencies / 工作流依赖项
        max_iterations: Maximum iterations to prevent infinite loops / 最大迭代次数，防止无限循环
        
    Returns:
        str: Final result message / 最终结果消息
    """
    if not PYDANTIC_GRAPH_AVAILABLE:
        raise ImportError(
            "pydantic-graph is not installed. "
            "Install with: pip install pydantic-graph"
        )
    
    from ..state.workflow_manager import load_workflow_state
    
    # Load initial state
    # 加载初始状态
    workflow_state = load_workflow_state(design_dir / "workflow_state.json")
    graph_state = LayoutWorkflowState.from_workflow_state(workflow_state)
    
    logger.info(f"Starting workflow: {graph_state.design_name}")
    logger.info(f"Progress: {sum(graph_state.completed)}/{len(graph_state.completed)} steps")
    
    # Run graph
    # 运行图
    try:
        result = await layout_workflow_graph.run(
            InitNode(),
            state=graph_state,
            deps=deps
        )
        return result.output
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise
