"""
Workflow Graph for Layout Agent Loop

Uses pydantic-graph to manage workflow state machine.
"""

import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Any

# Try to import pydantic-graph
try:
    from pydantic_graph import Graph, BaseNode, End, GraphRunContext
    PYDANTIC_GRAPH_AVAILABLE = True
except ImportError:
    PYDANTIC_GRAPH_AVAILABLE = False
    # Provide stub classes for type hints
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

logger = logging.getLogger(__name__)


# ============================================================================
# Dependencies
# ============================================================================

@dataclass
class LayoutAgentDeps:
    """Dependencies for workflow graph nodes"""
    design_dir: Path
    step_executor: "StepExecutor"
    reasoning_agent: Optional["ReasoningAgent"] = None
    constitution: str = ""
    temp_modified_params: Optional[dict] = None
    
    async def call_tool(self, tool_name: str, params: dict) -> dict:
        """Call a tool through step executor"""
        from ..state.models import StepDefinition, VerificationConfig
        
        # Create a minimal step definition for the executor
        step = StepDefinition(
            step_id=0,
            category="direct-call",
            description=f"Direct call to {tool_name}",
            tool=tool_name,
            parameters=params,
            verification=VerificationConfig(type="default", conditions=[])
        )
        
        result = await self.step_executor.execute_step(step, [True] * 100)  # Bypass dependency check
        return result.to_dict() if hasattr(result, 'to_dict') else {"success": result.success}
    
    async def verify_step(self, step: dict, tool_result: dict) -> bool:
        """Verify a step execution"""
        # Simple verification - check if tool succeeded
        return tool_result.get("success", False)
    
    async def save_workflow_state(self, state: LayoutWorkflowState) -> None:
        """Save workflow state to file"""
        from ..state.workflow_manager import save_workflow_state
        
        workflow = state.to_workflow_state()
        save_workflow_state(workflow, self.design_dir / "workflow_state.json")
    
    async def append_progress(self, step: dict, result: dict) -> None:
        """Append progress record"""
        from ..state.progress_writer import append_session_record
        from ..state.models import StepDefinition, StepResult, VerificationConfig
        
        # Convert dict to StepDefinition
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
        """Get full layout context"""
        return self.step_executor.get_full_context()


# ============================================================================
# Graph Nodes (only defined if pydantic-graph is available)
# ============================================================================

if PYDANTIC_GRAPH_AVAILABLE:
    
    @dataclass
    class InitNode(BaseNode[LayoutWorkflowState, LayoutAgentDeps, str]):
        """
        Initialization node - runs init.sh, reads progress, determines next step.
        
        This is the entry point for each workflow iteration.
        """
        
        async def run(
            self, 
            ctx: GraphRunContext[LayoutWorkflowState, LayoutAgentDeps]
        ) -> "ExecuteStepNode | End[str]":
            design_dir = ctx.deps.design_dir
            
            # 1. Execute init.sh (mandatory first step)
            logger.info("Executing init.sh...")
            init_script = design_dir / "init.sh"
            
            if init_script.exists():
                result = subprocess.run(
                    ["bash", str(init_script)],
                    cwd=design_dir,
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    logger.error(f"init.sh failed: {result.stderr}")
                    return End(f"Initialization failed: {result.stderr}")
                logger.info(f"init.sh completed (exit code: {result.returncode})")
            else:
                logger.warning(f"init.sh not found at {init_script}")
            
            # 2. Read progress.md last 50 lines (skip on first run)
            progress_file = design_dir / "progress.md"
            if progress_file.exists() and progress_file.stat().st_size > 0:
                logger.debug("Reading progress.md last 50 lines...")
                lines = progress_file.read_text().splitlines()
                last_50 = lines[-50:] if len(lines) > 50 else lines
                logger.debug(f"Progress summary: {len(last_50)} lines")
            else:
                logger.info("First run, skipping progress.md read")
            
            # 3. Load AGENT_CONSTITUTION.md
            constitution_path = Path(__file__).parent / "AGENT_CONSTITUTION.md"
            if constitution_path.exists():
                ctx.deps.constitution = constitution_path.read_text()
            
            # 4. Find first False step (mandatory step)
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
        """Execute the current step"""
        
        async def run(
            self,
            ctx: GraphRunContext[LayoutWorkflowState, LayoutAgentDeps]
        ) -> "VerifyStepNode | FailureAnalysisNode":
            step = ctx.state.steps[ctx.state.current_step_index]
            
            logger.debug(f"Calling tool: {step['tool']}")
            logger.debug(f"  Parameters: {step['parameters']}")
            
            # Use modified params if available (from failure recovery)
            params = ctx.deps.temp_modified_params or step['parameters']
            ctx.deps.temp_modified_params = None  # Clear after use
            
            try:
                result = await ctx.deps.call_tool(step['tool'], params)
                logger.debug(f"  Result: {result}")
                
                if result.get('success', False):
                    return VerifyStepNode(tool_result=result)
                else:
                    error_msg = result.get('error', {})
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get('message', 'Unknown error')
                    return FailureAnalysisNode(error=str(error_msg))
                    
            except Exception as e:
                logger.error(f"Tool call failed: {e}")
                return FailureAnalysisNode(error=str(e))
    
    
    @dataclass
    class VerifyStepNode(BaseNode[LayoutWorkflowState, LayoutAgentDeps, str]):
        """Verify step execution and update state"""
        tool_result: dict = field(default_factory=dict)
        
        async def run(
            self,
            ctx: GraphRunContext[LayoutWorkflowState, LayoutAgentDeps]
        ) -> "InitNode | End[str]":
            step = ctx.state.steps[ctx.state.current_step_index]
            
            logger.info(f"Verifying step: {step['verification']['type']}")
            
            # Execute verification
            verification_passed = await ctx.deps.verify_step(step, self.tool_result)
            
            if verification_passed:
                logger.info("Verification passed")
                
                # Update state: False -> True (only allowed modification)
                ctx.state.completed[ctx.state.current_step_index] = True
                logger.info(f"Updated state: Step {step['step_id']} -> true")
                
                # Save state to file
                await ctx.deps.save_workflow_state(ctx.state)
                
                # Update progress.md
                await ctx.deps.append_progress(step, self.tool_result)
                
                # Check if all completed
                if all(ctx.state.completed):
                    logger.info("=== All steps completed ===")
                    return End("Design completed")
                
                # Continue to next step
                return InitNode()
            else:
                logger.warning("Verification failed")
                return FailureAnalysisNode(error="Verification failed")
    
    
    @dataclass
    class FailureAnalysisNode(BaseNode[LayoutWorkflowState, LayoutAgentDeps, str]):
        """Analyze failure and attempt recovery"""
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
                            ctx.deps.temp_modified_params = analysis.modified_step.get('parameters')
                        return ExecuteStepNode()
                    else:
                        logger.error(f"Analysis: NOT recoverable - {analysis.recommendation}")
                        return End(f"Unrecoverable error: {analysis.recommendation}")
                        
                except Exception as e:
                    logger.error(f"Reasoning Agent failed: {e}")
            
            # Retry without modification
            return ExecuteStepNode()
    
    
    # Build workflow graph
    layout_workflow_graph = Graph(
        nodes=[InitNode, ExecuteStepNode, VerifyStepNode, FailureAnalysisNode]
    )

else:
    # Stubs when pydantic-graph is not available
    layout_workflow_graph = None
    InitNode = None
    ExecuteStepNode = None
    VerifyStepNode = None
    FailureAnalysisNode = None


# ============================================================================
# Workflow Runner
# ============================================================================

async def run_workflow(
    design_dir: Path,
    deps: LayoutAgentDeps,
    max_iterations: int = 10
) -> str:
    """
    Run the workflow graph.
    
    Args:
        design_dir: Design directory path
        deps: Workflow dependencies
        max_iterations: Maximum iterations to prevent infinite loops
        
    Returns:
        str: Final result message
    """
    if not PYDANTIC_GRAPH_AVAILABLE:
        raise ImportError(
            "pydantic-graph is not installed. "
            "Install with: pip install pydantic-graph"
        )
    
    from ..state.workflow_manager import load_workflow_state
    
    # Load initial state
    workflow_state = load_workflow_state(design_dir / "workflow_state.json")
    graph_state = LayoutWorkflowState.from_workflow_state(workflow_state)
    
    logger.info(f"Starting workflow: {graph_state.design_name}")
    logger.info(f"Progress: {sum(graph_state.completed)}/{len(graph_state.completed)} steps")
    
    # Run graph
    try:
        result = await layout_workflow_graph.run(
            InitNode(),
            state=graph_state,
            deps=deps
        )
        return result.data
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise
