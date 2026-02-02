"""Reasoning Agent for Layout Agent Loop
布局代理循环的推理代理

Uses deepseek-reasoner model to:
使用 deepseek-reasoner 模型来:
1. Plan workflow from user instructions / 根据用户指令规划工作流
2. Analyze failures and suggest fixes / 分析失败并建议修复方案

All outputs use PydanticAI structured output for type safety.
所有输出使用 PydanticAI 结构化输出以确保类型安全。

Agent Constitution Integration:
代理宪法集成:
- AGENT_CONSTITUTION.md is automatically loaded and injected into system prompts
- AGENT_CONSTITUTION.md 自动加载并注入系统提示词
- This ensures all planning decisions comply with constitutional rules
- 这确保所有规划决策都符合宪法规则
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

logger = logging.getLogger(__name__)

# ============================================================================
# Constitution Loading / 宪法加载
# ============================================================================

# Path to AGENT_CONSTITUTION.md / AGENT_CONSTITUTION.md 的路径
CONSTITUTION_PATH = Path(__file__).parent / "AGENT_CONSTITUTION.md"

# Cache for constitution content / 宪法内容缓存
_constitution_cache: Optional[str] = None


def load_constitution() -> str:
    """
    Load AGENT_CONSTITUTION.md content.
    加载 AGENT_CONSTITUTION.md 内容。
    
    Uses module-level cache to avoid repeated file reads.
    使用模块级缓存以避免重复读取文件。
    
    Returns:
        Constitution content as string, empty string if file not found
        宪法内容字符串，如果文件未找到则返回空字符串
    """
    global _constitution_cache
    
    if _constitution_cache is not None:
        return _constitution_cache
    
    if CONSTITUTION_PATH.exists():
        _constitution_cache = CONSTITUTION_PATH.read_text(encoding="utf-8")
        logger.info(f"Constitution loaded: {len(_constitution_cache)} chars")
    else:
        logger.warning(f"Constitution file not found: {CONSTITUTION_PATH}")
        _constitution_cache = ""
    
    return _constitution_cache


# ============================================================================
# Output Schemas (Pydantic models for structured output)
# 输出模式（用于结构化输出的 Pydantic 模型）
# ============================================================================

class StepDefinitionOutput(BaseModel):
    """Single step output format from Reasoning Agent
    推理代理的单步输出格式
    
    重构说明：
    - 移除 tool 和 parameters 字段，改用目标导向描述
    - Reasoning Agent 只描述"做什么"，不描述"用什么工具"
    - Act Agent（pydantic_agent）根据 objective 自主选择工具
    """
    step_id: int
    category: str  # device-creation, placement-layout, routing-connection, verification-drc, export-query
    description: str  # 人类可读的步骤描述
    
    # 目标导向字段（新架构）
    objective: str = ""  # 具体任务目标，描述需要完成什么（必填）
    expected_behavior: dict = Field(default_factory=dict)  # 期望的执行结果
    context_hints: dict = Field(default_factory=dict)  # 上下文提示，帮助 act agent 理解
    
    # 验证和依赖
    verification: dict = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    routing_justification: Optional[str] = None  # 仅 routing 步骤需要
    max_retries: int = 3
    
    # 兼容性字段（过渡期保留，新生成的计划不再填充）
    skill: str = ""
    tool: str = ""  # 已废弃
    parameters: dict = Field(default_factory=dict)  # 已废弃
    expected_output: dict = Field(default_factory=dict)  # 映射到 expected_behavior


class WorkflowPlanOutput(BaseModel):
    """Complete workflow plan output from Reasoning Agent
    推理代理的完整工作流计划输出"""
    design_name: str
    pdk: str
    steps: list[StepDefinitionOutput]
    
    def to_workflow_state_dict(self) -> dict:
        """Convert to workflow_state.json format
        转换为 workflow_state.json 格式"""
        return {
            "design_name": self.design_name,
            "pdk": self.pdk,
            "steps": [s.model_dump() for s in self.steps],
            "completed": [False] * len(self.steps)
        }


class FailureAnalysisOutput(BaseModel):
    """Failure analysis output from Reasoning Agent
    推理代理的失败分析输出"""
    recoverable: bool
    analysis: str
    modified_step: Optional[dict] = None
    recommendation: Optional[str] = None


# ============================================================================
# Prompts / 提示词
# ============================================================================

PLANNING_PROMPT = """# Analog Layout Reasoning Agent - Workflow Planner

You are an expert analog circuit layout planner. Your task is to analyze user requirements
and decompose them into a sequence of minimal executable steps.

## 目标导向输出（CRITICAL）

你的输出**不再指定具体的工具名称和参数**。相反，你需要：
1. 描述每个步骤的**任务目标 (objective)**
2. 说明**期望的执行结果 (expected_behavior)**
3. 提供**上下文提示 (context_hints)** 帮助执行代理理解任务

执行代理(Act Agent)会根据你的描述**自主选择**合适的工具。

## Output Format Requirements

You must output a strict JSON format with the following structure:
- design_name: Name of the design
- pdk: PDK name (sky130)
- steps: Array of step definitions

Each step must contain:
- step_id: Sequential number starting from 1
- category: Step type (device-creation, placement-layout, routing-connection, verification-drc, export-query)
- description: Human-readable step description (简短)
- objective: **Clear task objective describing WHAT to achieve** (详细，必填)
- expected_behavior: Expected outcome as object
- context_hints: Additional context to help execution agent
- verification: Verification config with "type" and "conditions"
- depends_on: List of step_ids this step depends on
- routing_justification: (Only for routing steps) Explain metal layer choice

## JSON Format Example

```json
{
  "design_name": "simple_ota",
  "pdk": "sky130",
  "steps": [
    {
      "step_id": 1,
      "category": "device-creation",
      "description": "创建 PMOS 电流镜参考管",
      "objective": "创建一个 PMOS 晶体管作为电流镜的参考管，沟道宽度 1.0um，沟道长度 0.15um，使用 2 个 fingers 以提高匹配性，添加 dummy 结构保护边缘",
      "expected_behavior": {
        "result_type": "pmos_transistor",
        "component_created": true,
        "has_standard_ports": true,
        "ports_include": ["gate", "drain", "source", "bulk"]
      },
      "context_hints": {
        "device_type": "pmos",
        "width_um": 1.0,
        "length_um": 0.15,
        "fingers": 2,
        "purpose": "current_mirror_reference",
        "matching_requirement": "high"
      },
      "verification": {"type": "component_exists", "conditions": []},
      "depends_on": [],
      "max_retries": 3
    },
    {
      "step_id": 2,
      "category": "placement-layout",
      "description": "将 PMOS 放置在原点",
      "objective": "将步骤1创建的 PMOS 晶体管放置到版图的原点位置 (0, 0)，作为电流镜布局的起始参考点",
      "expected_behavior": {
        "placement_completed": true,
        "position": {"x": 0, "y": 0}
      },
      "context_hints": {
        "target_component": "step_1_output",
        "position_x": 0,
        "position_y": 0,
        "rotation": 0
      },
      "verification": {"type": "placement_check", "conditions": []},
      "depends_on": [1],
      "max_retries": 3
    },
    {
      "step_id": 3,
      "category": "routing-connection",
      "description": "连接电流镜栅极",
      "objective": "将两个 PMOS 晶体管的栅极端口连接在一起，形成电流镜的栅极共接结构，使用 met2 层进行水平方向的布线",
      "expected_behavior": {
        "route_completed": true,
        "connection_type": "gate_tie",
        "layer_used": "met2"
      },
      "context_hints": {
        "source_component": "pmos_1",
        "source_port": "gate_W",
        "dest_component": "pmos_2",
        "dest_port": "gate_W",
        "preferred_layer": "met2",
        "routing_direction": "horizontal"
      },
      "verification": {"type": "routing_check", "conditions": []},
      "depends_on": [1, 2],
      "routing_justification": "使用 met2 层进行水平布线，避免与 met1 层的器件内部连线冲突",
      "max_retries": 3
    }
  ]
}
```

## 如何编写高质量的 objective

### 器件创建 (device-creation)
好的 objective 示例：
"创建一个 NMOS 晶体管用于差分对输入级，沟道宽度 2.0um，沟道长度 0.5um（长沟道以改善匹配性），使用 4 个 fingers，添加 dummy 结构和衬底连接"

### 布局放置 (placement-layout)
好的 objective 示例：
"将差分对的两个 NMOS 晶体管进行互指式放置（interdigitated layout），采用 ABBA 对称结构，4 列排列，以实现最佳匹配性"

### 路由连接 (routing-connection)
好的 objective 示例：
"连接 NMOS 输入管的漏极到 PMOS 负载管的漏极，形成输出节点。使用 met2 层进行垂直方向布线，避免与 met1 层的栅极连线交叉"

### 验证 (verification-drc)
好的 objective 示例：
"执行设计规则检查(DRC)，验证当前版图是否存在间距违规、最小宽度违规或包围违规"

### 导出 (export-query)
好的 objective 示例：
"将完成的版图导出为 GDS 格式文件，文件名使用设计名称"

## Planning Principles (MANDATORY)

1. **Objective-Oriented**: 描述"做什么"，不描述"用什么工具"
2. **Context Rich**: context_hints 提供足够的数值和上下文信息
3. **Clear Intent**: objective 必须明确无歧义
4. **Minimal Action**: 每个步骤只做一件事
5. **Clear Dependencies**: 在 depends_on 中明确标注依赖
6. **Verifiable**: 每个步骤必须有验证条件
7. **Correct Order**: 器件创建 → 放置 → 路由 → DRC → 导出

## context_hints 规范

### 器件创建
```json
{
  "device_type": "nmos|pmos",
  "width_um": 1.0,
  "length_um": 0.15,
  "fingers": 2,
  "multiplier": 1,
  "with_dummy": true,
  "with_tie": true,
  "purpose": "current_mirror|diff_pair|load|bias"
}
```

### 布局放置
```json
{
  "target_component": "component_name_or_step_reference",
  "position_x": 0.0,
  "position_y": 0.0,
  "rotation": 0,
  "align_to": "port_reference",
  "layout_style": "interdigitated|common_centroid"
}
```

### 路由连接
```json
{
  "source_component": "comp1",
  "source_port": "drain_E",
  "dest_component": "comp2",
  "dest_port": "drain_E",
  "preferred_layer": "met1|met2|met3",
  "routing_direction": "horizontal|vertical|auto"
}
```

## Metal Layer Rules

在 routing 步骤的 context_hints 中指定 preferred_layer：
1. 水平信号: met1
2. 垂直信号: met2
3. 交叉信号: 必须使用不同层
4. 电源/地: met3 或更高层
"""

FAILURE_ANALYSIS_PROMPT = """# Analog Layout Reasoning Agent - Failure Analyzer

A step execution has failed. Analyze the root cause and provide a fix.

## Tool Parameter Reference

### move_component
CORRECT: {"component_name": "xxx", "dx": 5.0, "dy": 0.0}
WRONG: {"component_name": "xxx", "x": 5.0, "y": 0.0}  <- x/y are NOT valid
WRONG: {"component_name": "xxx", "position": [5, 0]} <- position is NOT valid

### place_component
CORRECT: {"component_name": "xxx", "x": 0.0, "y": 10.0}

## Your Task

1. Analyze the root cause of the failure
2. Determine if it can be fixed by modifying parameters
3. If recoverable: Output modified step definition
4. If not recoverable: Explain why and recommend manual intervention

## Output Format

Output MUST be valid JSON with this EXACT structure:

For recoverable errors:
``json
{
  "recoverable": true,
  "analysis": "Explanation of failure cause",
  "modified_step": {
    "parameters": {
      "component_name": "pmos_1",
      "dx": 0,
      "dy": 10
    }
  }
}

```

For non-recoverable errors:
```
{
  "recoverable": false,
  "analysis": "Explanation of failure cause",
  "recommendation": "Suggested manual action"
}


```
"""


# ============================================================================
# Reasoning Agent Class / 推理代理类
# ============================================================================

class ReasoningAgent:
    """
    Reasoning Agent using deepseek-reasoner model with structured output.
    使用 deepseek-reasoner 模型的推理代理，支持结构化输出。
    
    Responsibilities / 职责:
    1. Plan workflow from user instructions / 根据用户指令规划工作流
    2. Analyze failures and suggest fixes
    
    宪法注入: 完整宪法在 __init__ 时注入到 system_prompt
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
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY or OPENAI_API_KEY not set")
        
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        self.model_name = model_name
        
        # Configure model using OpenAIProvider (compatible with DeepSeek API)
        provider = OpenAIProvider(base_url=self.base_url, api_key=self.api_key)
        self.model = OpenAIChatModel(model_name, provider=provider)
        
        # Check if model supports structured output (tool_choice)
        # deepseek-reasoner does NOT support tool_choice, need to parse JSON manually
        self.use_structured_output = "reasoner" not in model_name.lower()
        
        # ============== 完整宪法注入到规划 prompt ==============
        constitution = load_constitution()
        if constitution:
            # 在 PLANNING_PROMPT 后追加完整宪法
            planning_prompt_with_constitution = f"""{PLANNING_PROMPT}

═══════════════════════════════════════════════════════════════
              AGENT CONSTITUTION (MANDATORY)
═══════════════════════════════════════════════════════════════

{constitution}

## 规划时必须遵守的宪法规则

1. **步骤顺序**: device-creation → placement → routing → drc → export
2. **Routing 规则**: 每个 routing 步骤必须指定 layer 参数
3. **依赖关系**: depends_on 只能引用之前的步骤
4. **验证配置**: 每个步骤必须有 verification 配置
"""
            logger.info(f"Constitution integrated: {len(constitution)} chars")
        else:
            planning_prompt_with_constitution = PLANNING_PROMPT
            logger.warning("Constitution not loaded for ReasoningAgent")
        
        if self.use_structured_output:
            self.planning_agent = Agent(
                self.model,
                output_type=WorkflowPlanOutput,
                system_prompt=planning_prompt_with_constitution
            )
            
            self.analysis_agent = Agent(
                self.model,
                output_type=FailureAnalysisOutput,
                system_prompt=FAILURE_ANALYSIS_PROMPT
            )
        else:
            self.planning_agent = Agent(
                self.model,
                output_type=str,
                system_prompt=planning_prompt_with_constitution + "\n\nIMPORTANT: Output ONLY valid JSON, no markdown code blocks, no extra text."
            )
            
            self.analysis_agent = Agent(
                self.model,
                output_type=str,
                system_prompt=FAILURE_ANALYSIS_PROMPT + "\n\nIMPORTANT: Output ONLY valid JSON, no markdown code blocks, no extra text."
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
        
        if self.use_structured_output:
            plan = result.output
        else:
            # Parse JSON from text output (for reasoner models)
            plan = self._parse_workflow_json(result.output)
        
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
        
        if self.use_structured_output:
            analysis = result.output
        else:
            # Parse JSON from text output (for reasoner models)
            analysis = self._parse_analysis_json(result.output)
        
        if analysis.recoverable:
            logger.info(f"Failure is recoverable: {analysis.analysis[:100]}...")
        else:
            logger.warning(f"Failure is NOT recoverable: {analysis.recommendation}")
        
        return analysis
    
    def _parse_workflow_json(self, text: str) -> WorkflowPlanOutput:
        """
        Parse workflow JSON from plain text output (for reasoner models).
        
        Handles various formats:
        - Pure JSON
        - JSON wrapped in markdown code blocks
        - JSON with extra text before/after
        """
        # Try to extract JSON from markdown code blocks first
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError(f"Could not find JSON in response: {text[:200]}...")
        
        try:
            data = json.loads(json_str)
            return WorkflowPlanOutput(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}\nText: {json_str[:500]}...")
        except Exception as e:
            raise ValueError(f"Failed to parse workflow plan: {e}")
    
    def _parse_analysis_json(self, text: str) -> FailureAnalysisOutput:
        """
        Parse failure analysis JSON from plain text output (for reasoner models).
        """
        # Try to extract JSON from markdown code blocks first
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError(f"Could not find JSON in response: {text[:200]}...")
        
        try:
            data = json.loads(json_str)
            return FailureAnalysisOutput(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}\nText: {json_str[:500]}...")
        except Exception as e:
            raise ValueError(f"Failed to parse failure analysis: {e}")
    
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
        4. Each step has objective (required for new architecture)
        5. Routing steps have layer info in context_hints
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
        
        # Check each step has valid task definition
        for step in plan.steps:
            has_objective = bool(step.objective) if hasattr(step, 'objective') else False
            has_tool = bool(step.tool)
            
            # 新架构要求必须有 objective
            if not has_objective and not has_tool:
                raise ValueError(
                    f"Step {step.step_id} must have 'objective' defined (new architecture) "
                    f"or 'tool' for backward compatibility"
                )
            
            # 如果有 tool 但没有 objective，发出警告（旧格式）
            if has_tool and not has_objective:
                logger.warning(
                    f"Step {step.step_id} uses deprecated 'tool' field. "
                    f"Please migrate to 'objective' based format."
                )
        
        # Check routing steps have layer info
        for step in plan.steps:
            if step.category == "routing-connection":
                context_hints = getattr(step, 'context_hints', {}) or {}
                has_layer_hint = 'preferred_layer' in context_hints or 'layer' in context_hints
                has_layer_param = "layer" in step.parameters if step.parameters else False
                
                if not has_layer_hint and not has_layer_param:
                    logger.warning(
                        f"Routing step {step.step_id} should specify 'preferred_layer' in context_hints"
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
