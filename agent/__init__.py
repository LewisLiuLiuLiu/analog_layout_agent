"""
Agent module - Agent orchestration layer

Contains Agent classes, task planners, prompt templates.
Supports PydanticAI framework (primary) and OpenAI Agent SDK (deprecated).

New: Agent Loop for Reasoning-Execution workflow
"""

# Original module imports (optional, if layout_agent.py exists)
try:
    from .layout_agent import AnalogLayoutAgent, TaskPlanner, ToolExecutor, Task, TaskStatus
    _LAYOUT_AGENT_AVAILABLE = True
except ImportError:
    _LAYOUT_AGENT_AVAILABLE = False
    AnalogLayoutAgent = None
    TaskPlanner = None
    ToolExecutor = None
    Task = None
    TaskStatus = None

from .prompt_templates import (
    get_system_prompt,
    get_task_decomposition_prompt,
    get_error_recovery_prompt,
    get_design_review_prompt,
    get_tool_example,
    format_context_for_llm,
    format_result_for_llm,
)

# PydanticAI integration (recommended)
from .pydantic_agent import (
    create_layout_agent,
    run_layout_agent,
    run_layout_agent_sync,
    run_layout_agent_stream,
    LayoutAgentDeps,
)

# Agent Loop (new - Reasoning-Execution workflow)
from .agent_loop import (
    AgentLoop,
    DRCStrategy,
    run_layout_agent_loop,
    run_layout_agent_loop_sync,
)

from .reasoning_agent import (
    ReasoningAgent,
    WorkflowPlanOutput,
    FailureAnalysisOutput,
    load_available_skills,
)

from .step_executor import (
    StepExecutor,
    ExecutionContext,
)

from .logging_config import (
    setup_agent_logging,
    get_logger,
    log_step_execution,
    log_workflow_event,
    WorkflowLogContext,
)

# OpenAI Agent SDK integration (deprecated, kept for backward compatibility)
try:
    from .openai_agent import (
        create_layout_agent as create_layout_agent_openai,
        run_layout_agent as run_layout_agent_openai,
        LayoutAgentContext,
    )
    _OPENAI_AGENT_AVAILABLE = True
except ImportError:
    _OPENAI_AGENT_AVAILABLE = False
    LayoutAgentContext = None
    create_layout_agent_openai = None
    run_layout_agent_openai = None

__all__ = [
    # Original exports
    "AnalogLayoutAgent",
    "TaskPlanner",
    "ToolExecutor",
    "Task",
    "TaskStatus",
    "get_system_prompt",
    "get_task_decomposition_prompt",
    "get_error_recovery_prompt",
    "get_design_review_prompt",
    "get_tool_example",
    "format_context_for_llm",
    "format_result_for_llm",
    # PydanticAI integration (recommended)
    "create_layout_agent",
    "run_layout_agent",
    "run_layout_agent_sync",
    "run_layout_agent_stream",
    "LayoutAgentDeps",
    # Agent Loop (new)
    "AgentLoop",
    "DRCStrategy",
    "run_layout_agent_loop",
    "run_layout_agent_loop_sync",
    "ReasoningAgent",
    "WorkflowPlanOutput",
    "FailureAnalysisOutput",
    "load_available_skills",
    "StepExecutor",
    "ExecutionContext",
    "setup_agent_logging",
    "get_logger",
    "log_step_execution",
    "log_workflow_event",
    "WorkflowLogContext",
    # OpenAI Agent SDK integration (deprecated)
    "LayoutAgentContext",
    "create_layout_agent_openai",
    "run_layout_agent_openai",
]
