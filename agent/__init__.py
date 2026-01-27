"""
Agent module - Agent编排层

包含Agent主类、任务规划器、Prompt模板等
支持 PydanticAI 框架（主要）和 OpenAI Agent SDK（已废弃）
"""

# 原有模块导入（可选，如果 layout_agent.py 存在）
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

# PydanticAI 集成（推荐使用）
from .pydantic_agent import (
    create_layout_agent,
    run_layout_agent,
    run_layout_agent_sync,
    run_layout_agent_stream,
    LayoutAgentDeps,
)

# OpenAI Agent SDK 集成（已废弃，保留向后兼容）
# 注意：推荐使用 pydantic_agent 中的实现
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
    # 原有导出
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
    # PydanticAI 集成（推荐）
    "create_layout_agent",
    "run_layout_agent",
    "run_layout_agent_sync",
    "run_layout_agent_stream",
    "LayoutAgentDeps",
    # OpenAI Agent SDK 集成（已废弃）
    "LayoutAgentContext",
    "create_layout_agent_openai",
    "run_layout_agent_openai",
]
