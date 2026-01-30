"""
Reasoning Agent for Layout Agent Loop

Uses deepseek-reasoner model to:
1. Plan workflow from user instructions
2. Analyze failures and suggest fixes

All outputs use PydanticAI structured output for type safety.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

logger = logging.getLogger(__name__)


# ============================================================================
# Output Schemas (Pydantic models for structured output)
# ============================================================================

class StepDefinitionOutput(BaseModel):
    """Single step output format from Reasoning Agent"""
    step_id: int
    category: str  # device-creation, placement-layout, routing-connection, verification-drc, export-query
    description: str
    skill: str = ""
    tool: str
    parameters: dict = Field(default_factory=dict)
    expected_output: dict = Field(default_factory=dict)
    verification: dict = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    routing_justification: Optional[str] = None
    max_retries: int = 3


class WorkflowPlanOutput(BaseModel):
    """Complete workflow plan output from Reasoning Agent"""
    design_name: str
    pdk: str
    steps: list[StepDefinitionOutput]
    
    def to_workflow_state_dict(self) -> dict:
        """Convert to workflow_state.json format"""
        return {
            "design_name": self.design_name,
            "pdk": self.pdk,
            "steps": [s.model_dump() for s in self.steps],
            "completed": [False] * len(self.steps)
        }


class FailureAnalysisOutput(BaseModel):
    """Failure analysis output from Reasoning Agent"""
    recoverable: bool
    analysis: str
    modified_step: Optional[dict] = None
    recommendation: Optional[str] = None


# ============================================================================
# Prompts
# ============================================================================

PLANNING_PROMPT = """# Analog Layout Reasoning Agent - Workflow Planner

You are an expert analog circuit layout planner. Your task is to analyze user requirements
and decompose them into a sequence of minimal executable steps.

## Output Format Requirements

You must output a strict JSON format with the following structure:
- design_name: Name of the design
- pdk: PDK name (sky130)
- steps: Array of step definitions

Each step must contain:
- step_id: Sequential number starting from 1
- category: Step type (device-creation, placement-layout, routing-connection, verification-drc, export-query)
- description: Human-readable step description
- tool: Tool name to use
- parameters: Tool parameters as object
- expected_output: Expected output for verification
- verification: Verification configuration with type and conditions
- depends_on: List of step_ids this step depends on
- routing_justification: (Only for routing steps) Explain metal layer choice

## Available Tools by Category

### device-creation
- create_nmos: Create NMOS transistor
- create_pmos: Create PMOS transistor
- create_mimcap: Create MIM capacitor
- create_resistor: Create resistor
- create_via_stack: Create via stack

### placement-layout
- place_component: Place component relative to another
- align_to_port: Align component to a port
- move_component: Move component to position
- interdigitize: Interdigitate multiple components

### routing-connection
- smart_route: Smart routing between ports
- c_route: C-shaped route
- l_route: L-shaped route
- straight_route: Straight route

### verification-drc
- run_drc: Run DRC check

### export-query
- export_gds: Export to GDS file
- list_components: List all components
- get_component_info: Get component details

## Planning Principles (MANDATORY)

1. **Minimal Action**: Each step does ONE thing only
2. **Clear Dependencies**: Explicitly mark step dependencies in depends_on
3. **Verifiable**: Each step must have clear verification conditions
4. **Correct Order**: Device creation -> Placement -> Routing -> DRC -> Export
5. **DRC Checkpoints**: Add DRC verification after placement and routing steps
6. **Metal Layer Planning**: For routing steps, MUST specify layer parameter to prevent shorts

## Metal Layer Rules (CRITICAL - Prevent Short Circuits)

1. Horizontal signals: met1
2. Vertical signals: met2
3. Crossing signals: MUST use different layers
4. Power/Ground: met3 or higher
5. EVERY routing step MUST have a 'layer' parameter
"""

FAILURE_ANALYSIS_PROMPT = """# Analog Layout Reasoning Agent - Failure Analyzer

A step execution has failed. Analyze the root cause and provide a fix.

## Your Task

1. Analyze the root cause of the failure
2. Determine if it can be fixed by modifying parameters
3. If recoverable: Output modified step definition
4. If not recoverable: Explain why and recommend manual intervention

## Output Format

For recoverable errors:
- recoverable: true
- analysis: Explanation of failure cause
- modified_step: Modified step definition with fixed parameters

For non-recoverable errors:
- recoverable: false
- analysis: Explanation of failure cause
- recommendation: Suggested manual action
"""


# ============================================================================
# Reasoning Agent Class
# ============================================================================

class ReasoningAgent:
    """
    Reasoning Agent using deepseek-reasoner model with structured output.
    
    Responsibilities:
    1. Plan workflow from user instructions
    2. Analyze failures and suggest fixes
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model_name: str = "deepseek-reasoner"
    ):
        """
        Initialize Reasoning Agent.
        
        Args:
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            base_url: API base URL
            model_name: Model name to use
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")
        
        self.base_url = base_url
        self.model_name = model_name
        
        # Configure model
        self.model = OpenAIModel(
            model_name,
            base_url=base_url,
            api_key=self.api_key
        )
        
        # Create planning agent with structured output
        self.planning_agent = Agent(
            self.model,
            output_type=WorkflowPlanOutput,
            system_prompt=PLANNING_PROMPT
        )
        
        # Create failure analysis agent with structured output
        self.analysis_agent = Agent(
            self.model,
            output_type=FailureAnalysisOutput,
            system_prompt=FAILURE_ANALYSIS_PROMPT
        )
        
        logger.info(f"ReasoningAgent initialized with model: {model_name}")
    
    async def plan_workflow(
        self,
        instruction: str,
        pdk: str,
        available_skills: list[dict]
    ) -> WorkflowPlanOutput:
        """
        Plan workflow from user instruction.
        
        Uses structured output to ensure valid JSON format.
        
        Args:
            instruction: User instruction for the design
            pdk: PDK name (e.g., 'sky130')
            available_skills: List of available skills and tools
            
        Returns:
            WorkflowPlanOutput: Type-safe workflow plan
        """
        logger.info(f"Planning workflow for: {instruction[:50]}...")
        
        # Build prompt with context
        skills_str = self._format_skills(available_skills)
        user_prompt = f"""PDK: {pdk}

User Instruction: {instruction}

Available Skills and Tools:
{skills_str}

Please create a detailed workflow plan following all the rules in the system prompt.
Remember: Every routing step MUST have a 'layer' parameter specified.
"""
        
        # Run planning agent
        result = await self.planning_agent.run(user_prompt)
        plan = result.data
        
        # Validate the plan
        self._validate_plan(plan)
        
        logger.info(f"Workflow planned: {plan.design_name} with {len(plan.steps)} steps")
        return plan
    
    async def analyze_failure(
        self,
        failed_step: dict,
        error_info: dict,
        layout_context: dict
    ) -> FailureAnalysisOutput:
        """
        Analyze a failed step and suggest fix.
        
        Args:
            failed_step: The step definition that failed
            error_info: Error information from execution
            layout_context: Current layout context state
            
        Returns:
            FailureAnalysisOutput: Analysis result with fix suggestion
        """
        logger.info(f"Analyzing failure for step {failed_step.get('step_id')}")
        
        import json
        user_prompt = f"""## Failed Step Information
```json
{json.dumps(failed_step, indent=2, ensure_ascii=False)}
```

## Error Information
```json
{json.dumps(error_info, indent=2, ensure_ascii=False)}
```

## Current Layout Context
```json
{json.dumps(layout_context, indent=2, ensure_ascii=False, default=str)}
```

Please analyze the failure and provide a fix if possible.
"""
        
        result = await self.analysis_agent.run(user_prompt)
        analysis = result.data
        
        if analysis.recoverable:
            logger.info(f"Failure is recoverable: {analysis.analysis[:100]}...")
        else:
            logger.warning(f"Failure is NOT recoverable: {analysis.recommendation}")
        
        return analysis
    
    def _format_skills(self, skills: list[dict]) -> str:
        """Format skills list for prompt"""
        if not skills:
            return "No specific skills provided. Use standard tools."
        
        lines = []
        for skill in skills:
            name = skill.get('name', 'unknown')
            tools = skill.get('tools', [])
            lines.append(f"- {name}: {', '.join(tools)}")
        return '\n'.join(lines)
    
    def _validate_plan(self, plan: WorkflowPlanOutput) -> None:
        """
        Validate workflow plan after Reasoning Agent output.
        
        Checks:
        1. At least one step
        2. step_id is sequential from 1
        3. depends_on references are valid
        4. routing steps have layer parameter
        """
        if len(plan.steps) == 0:
            raise ValueError("Workflow must have at least one step")
        
        # Check step_id sequence
        step_ids = [s.step_id for s in plan.steps]
        expected = list(range(1, len(plan.steps) + 1))
        if step_ids != expected:
            raise ValueError(f"step_id must be sequential from 1: got {step_ids}")
        
        # Check dependencies
        for step in plan.steps:
            for dep in step.depends_on:
                if dep >= step.step_id:
                    raise ValueError(
                        f"Step {step.step_id} cannot depend on step {dep} (forward reference)"
                    )
                if dep < 1:
                    raise ValueError(
                        f"Step {step.step_id} has invalid dependency: {dep}"
                    )
        
        # Check routing steps have layer
        for step in plan.steps:
            if step.category == "routing-connection":
                if "layer" not in step.parameters:
                    raise ValueError(
                        f"Routing step {step.step_id} must specify 'layer' parameter"
                    )
                if not step.routing_justification:
                    logger.warning(
                        f"Routing step {step.step_id} missing routing_justification"
                    )


def load_available_skills(skills_dir: Path) -> list[dict]:
    """
    Load available skills from skills directory.
    
    Args:
        skills_dir: Path to skills directory
        
    Returns:
        list[dict]: List of skill definitions
    """
    skills = []
    
    if not skills_dir.exists():
        logger.warning(f"Skills directory not found: {skills_dir}")
        return skills
    
    for skill_dir in skills_dir.iterdir():
        if not skill_dir.is_dir():
            continue
        
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        
        # Parse skill name and tools
        skill_name = skill_dir.name
        scripts_dir = skill_dir / "scripts"
        tools = []
        
        if scripts_dir.exists():
            for script in scripts_dir.glob("*.py"):
                if script.name != "__init__.py":
                    tools.append(script.stem)
        
        skills.append({
            "name": skill_name,
            "tools": tools,
            "path": str(skill_dir)
        })
    
    logger.info(f"Loaded {len(skills)} skills from {skills_dir}")
    return skills
