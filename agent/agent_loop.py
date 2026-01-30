"""
Agent Loop Controller for Layout Agent

Main entry point for running the Reasoning-Execution loop.
"""

import logging
import asyncio
from pathlib import Path
from typing import Optional
from enum import Enum

from .logging_config import setup_agent_logging, get_logger
from .reasoning_agent import ReasoningAgent, load_available_skills
from .step_executor import StepExecutor, ExecutionContext
from .workflow_graph import LayoutAgentDeps, run_workflow, PYDANTIC_GRAPH_AVAILABLE

from ..state.models import WorkflowState
from ..state.workflow_manager import (
    load_workflow_state,
    save_workflow_state,
    create_initial_workflow_state,
    validate_workflow_state,
    get_workflow_summary
)
from ..state.progress_writer import create_progress_file, update_progress_header

logger = get_logger()


class DRCStrategy(Enum):
    """DRC verification strategy"""
    CRITICAL_STEPS = "critical_steps"  # Only after routing/placement
    EVERY_STEP = "every_step"          # After every step


class AgentLoop:
    """
    Agent Loop main controller.
    
    Orchestrates the Reasoning-Execution loop:
    1. First run: Reasoning Agent plans workflow
    2. Subsequent runs: Execute steps one by one
    3. On failure: Call Reasoning Agent for analysis
    """
    
    def __init__(
        self,
        mcp_server,
        designs_dir: Path,
        reasoning_api_key: Optional[str] = None,
        reasoning_base_url: str = "https://api.deepseek.com",
        drc_strategy: DRCStrategy = DRCStrategy.CRITICAL_STEPS,
        max_iterations: int = 10
    ):
        """
        Initialize Agent Loop.
        
        Args:
            mcp_server: MCP server instance with tools
            designs_dir: Base directory for designs
            reasoning_api_key: API key for Reasoning Agent
            reasoning_base_url: API base URL
            drc_strategy: DRC verification strategy
            max_iterations: Maximum iterations to prevent loops
        """
        self.mcp_server = mcp_server
        self.designs_dir = Path(designs_dir)
        self.drc_strategy = drc_strategy
        self.max_iterations = max_iterations
        
        # Initialize Reasoning Agent
        try:
            self.reasoning_agent = ReasoningAgent(
                api_key=reasoning_api_key,
                base_url=reasoning_base_url
            )
            logger.info("Reasoning Agent initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Reasoning Agent: {e}")
            self.reasoning_agent = None
        
        # Load available skills
        skills_dir = Path(__file__).parent.parent / "skills"
        self.available_skills = load_available_skills(skills_dir)
    
    async def run(
        self,
        instruction: str,
        pdk: str = "sky130",
        design_name: Optional[str] = None
    ) -> dict:
        """
        Run the Agent Loop.
        
        Process:
        1. Check if workflow exists
        2. If not, call Reasoning Agent to plan
        3. Run execution loop
        
        Args:
            instruction: User instruction for the design
            pdk: PDK name
            design_name: Design name (generated if not provided)
            
        Returns:
            dict: Execution result
        """
        # Generate design name if not provided
        if not design_name:
            design_name = self._generate_design_name(instruction)
        
        design_dir = self.designs_dir / design_name
        workflow_path = design_dir / "workflow_state.json"
        progress_path = design_dir / "progress.md"
        
        # Setup logging for this design
        log_file = design_dir / "agent.log"
        setup_agent_logging(
            project_name=f"layout-agent-{design_name}",
            log_file=log_file
        )
        
        logger.info(f"=== Agent Loop Started ===")
        logger.info(f"Design: {design_name}")
        logger.info(f"PDK: {pdk}")
        logger.info(f"Instruction: {instruction[:100]}...")
        
        # Check if workflow exists
        if not workflow_path.exists():
            logger.info("No existing workflow found, planning new workflow...")
            
            if not self.reasoning_agent:
                return {
                    "success": False,
                    "error": "Reasoning Agent not available for planning"
                }
            
            # Plan workflow
            try:
                plan = await self.reasoning_agent.plan_workflow(
                    instruction=instruction,
                    pdk=pdk,
                    available_skills=self.available_skills
                )
                
                # Save workflow state
                design_dir.mkdir(parents=True, exist_ok=True)
                workflow = create_initial_workflow_state(
                    design_name=plan.design_name,
                    pdk=plan.pdk,
                    steps=[s.model_dump() for s in plan.steps],
                    output_path=workflow_path
                )
                
                # Create progress file
                create_progress_file(workflow, progress_path)
                
                # Create init.sh
                self._create_init_script(design_dir)
                
                logger.info(f"Workflow planned with {len(plan.steps)} steps")
                
            except Exception as e:
                logger.error(f"Failed to plan workflow: {e}")
                return {
                    "success": False,
                    "error": f"Planning failed: {e}"
                }
        
        # Validate workflow
        is_valid, error = validate_workflow_state(workflow_path)
        if not is_valid:
            return {
                "success": False,
                "error": f"Invalid workflow state: {error}"
            }
        
        # Load workflow
        workflow = load_workflow_state(workflow_path)
        
        # Check if already completed
        if workflow.is_all_completed():
            logger.info("Workflow already completed!")
            return {
                "success": True,
                "message": "Design already completed",
                "summary": get_workflow_summary(workflow)
            }
        
        # Run execution loop
        if PYDANTIC_GRAPH_AVAILABLE:
            result = await self._run_with_graph(design_dir, workflow)
        else:
            result = await self._run_simple_loop(design_dir, workflow)
        
        # Update progress header
        workflow = load_workflow_state(workflow_path)
        update_progress_header(workflow, progress_path)
        
        return result
    
    async def _run_with_graph(self, design_dir: Path, workflow: WorkflowState) -> dict:
        """Run using pydantic-graph workflow"""
        logger.info("Running with pydantic-graph workflow")
        
        # Create execution context
        context = ExecutionContext(
            mcp_server=self.mcp_server,
            design_dir=design_dir
        )
        
        # Create step executor
        executor = StepExecutor(context)
        
        # Create dependencies
        deps = LayoutAgentDeps(
            design_dir=design_dir,
            step_executor=executor,
            reasoning_agent=self.reasoning_agent
        )
        
        try:
            result_msg = await run_workflow(design_dir, deps, self.max_iterations)
            return {
                "success": "completed" in result_msg.lower() or "all steps" in result_msg.lower(),
                "message": result_msg,
                "summary": get_workflow_summary(load_workflow_state(design_dir / "workflow_state.json"))
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_simple_loop(self, design_dir: Path, workflow: WorkflowState) -> dict:
        """Run simple execution loop without pydantic-graph"""
        logger.info("Running simple execution loop (pydantic-graph not available)")
        
        workflow_path = design_dir / "workflow_state.json"
        
        # Create execution context
        context = ExecutionContext(
            mcp_server=self.mcp_server,
            design_dir=design_dir
        )
        executor = StepExecutor(context)
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            
            # Reload workflow state
            workflow = load_workflow_state(workflow_path)
            
            # Find next step
            step_idx = workflow.get_first_incomplete_step()
            if step_idx is None:
                logger.info("All steps completed!")
                return {
                    "success": True,
                    "message": "Design completed",
                    "summary": get_workflow_summary(workflow)
                }
            
            step = workflow.steps[step_idx]
            logger.info(f"Executing step {step.step_id}: {step.description}")
            
            # Execute step
            result = await executor.execute_step(step, workflow.completed)
            
            if result.success:
                # Mark as completed
                workflow.mark_step_complete(step_idx)
                save_workflow_state(workflow, workflow_path)
                logger.info(f"Step {step.step_id} completed")
            else:
                logger.error(f"Step {step.step_id} failed: {result.error}")
                
                # Try failure analysis
                if self.reasoning_agent:
                    try:
                        analysis = await self.reasoning_agent.analyze_failure(
                            failed_step=step.model_dump(),
                            error_info=result.error or {},
                            layout_context=executor.get_full_context()
                        )
                        
                        if not analysis.recoverable:
                            return {
                                "success": False,
                                "error": f"Step {step.step_id} failed: {analysis.recommendation}"
                            }
                        # Continue with retry
                        logger.info("Retrying with modified parameters...")
                    except Exception as e:
                        logger.error(f"Failure analysis failed: {e}")
                else:
                    return {
                        "success": False,
                        "error": f"Step {step.step_id} failed: {result.error}"
                    }
        
        return {
            "success": False,
            "error": f"Max iterations ({self.max_iterations}) reached"
        }
    
    def _generate_design_name(self, instruction: str) -> str:
        """Generate design name from instruction"""
        import re
        from datetime import datetime
        
        # Extract key words
        words = re.findall(r'\b\w+\b', instruction.lower())
        key_words = [w for w in words if w not in ['a', 'an', 'the', 'create', 'make', 'design', 'with']]
        
        if key_words:
            name = '_'.join(key_words[:3])
        else:
            name = "design"
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{name}_{timestamp}"
    
    def _create_init_script(self, design_dir: Path) -> None:
        """Create init.sh script for design directory"""
        project_root = Path(__file__).parent.parent.parent
        
        init_content = f'''#!/bin/bash
# Layout Agent initialization script
# Auto-generated for design directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="{project_root}"

echo "=== Layout Agent Initialization ==="
echo "Design directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"

# 1. Activate virtual environment
echo "[1/5] Activating virtual environment..."
source "$PROJECT_ROOT/venv311/bin/activate"

if [[ "$VIRTUAL_ENV" != *"venv311"* ]]; then
    echo "Error: Virtual environment activation failed"
    exit 1
fi
echo "      Python: $(which python3)"

# 2. Check dependencies
echo "[2/5] Checking dependencies..."
GDSFACTORY_VERSION=$(pip show gdsfactory 2>/dev/null | grep "^Version:" | awk '{{print $2}}')
echo "      gdsfactory: $GDSFACTORY_VERSION"

# 3. Check state files
echo "[3/5] Checking state files..."
if [[ ! -f "$SCRIPT_DIR/workflow_state.json" ]]; then
    echo "Error: workflow_state.json not found"
    exit 1
fi
echo "      workflow_state.json: exists"

if [[ ! -f "$SCRIPT_DIR/progress.md" ]]; then
    touch "$SCRIPT_DIR/progress.md"
fi
echo "      progress.md: exists"

# 4. Validate JSON
echo "[4/5] Validating workflow_state.json..."
python3 -c "
import json
import sys

with open('$SCRIPT_DIR/workflow_state.json', 'r') as f:
    data = json.load(f)

required = ['design_name', 'pdk', 'steps', 'completed']
for field in required:
    if field not in data:
        print(f'Error: missing field {{field}}')
        sys.exit(1)

if len(data['completed']) != len(data['steps']):
    print(f'Error: completed/steps length mismatch')
    sys.exit(1)

for i, val in enumerate(data['completed']):
    if not isinstance(val, bool):
        print(f'Error: completed[{{i}}] is not boolean')
        sys.exit(1)

print(f'      Design: {{data[\"design_name\"]}}')
print(f'      Steps: {{len(data[\"steps\"])}}')
print(f'      Progress: {{sum(data[\"completed\"])}}/{{len(data[\"completed\"])}}')
"

# 5. Create output directory
echo "[5/5] Checking output directory..."
mkdir -p "$SCRIPT_DIR/output"
echo "      output/: ready"

echo ""
echo "=== Initialization Complete ==="
echo ""
'''
        
        init_path = design_dir / "init.sh"
        init_path.write_text(init_content)
        init_path.chmod(0o755)
        logger.info(f"Created init.sh at {init_path}")


# ============================================================================
# Public API
# ============================================================================

async def run_layout_agent_loop(
    instruction: str,
    pdk: str = "sky130",
    design_name: Optional[str] = None,
    mcp_server = None,
    designs_dir: Optional[Path] = None
) -> dict:
    """
    Run the Layout Agent Loop.
    
    This is the main entry point for running the agent loop.
    
    Args:
        instruction: User instruction for the design
        pdk: PDK name (default: sky130)
        design_name: Optional design name
        mcp_server: MCP server instance (created if not provided)
        designs_dir: Base directory for designs
        
    Returns:
        dict: Execution result with success status and summary
    """
    # Set default designs directory
    if designs_dir is None:
        designs_dir = Path(__file__).parent.parent.parent / "designs"
    
    # Create MCP server if not provided
    if mcp_server is None:
        from ..mcp_server.server import create_mcp_server
        mcp_server = create_mcp_server(pdk=pdk)
    
    # Create and run agent loop
    loop = AgentLoop(
        mcp_server=mcp_server,
        designs_dir=designs_dir
    )
    
    return await loop.run(
        instruction=instruction,
        pdk=pdk,
        design_name=design_name
    )


def run_layout_agent_loop_sync(
    instruction: str,
    pdk: str = "sky130",
    design_name: Optional[str] = None,
    mcp_server = None,
    designs_dir: Optional[Path] = None
) -> dict:
    """
    Synchronous wrapper for run_layout_agent_loop.
    """
    return asyncio.run(run_layout_agent_loop(
        instruction=instruction,
        pdk=pdk,
        design_name=design_name,
        mcp_server=mcp_server,
        designs_dir=designs_dir
    ))
