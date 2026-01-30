"""
State management module for Layout Agent Loop
布局代理循环的状态管理模块

This module provides:
本模块提供:
- WorkflowState: Workflow state data model / 工作流状态数据模型
- StepDefinition: Step definition data model / 步骤定义数据模型
- LayoutWorkflowState: State class for pydantic-graph / 用于 pydantic-graph 的状态类
"""

from .models import (
    VerificationConfig,
    StepDefinition,
    WorkflowState,
    LayoutWorkflowState,
    StepResult,
    VerificationResult,
)

from .workflow_manager import (
    load_workflow_state,
    save_workflow_state,
    validate_workflow_state,
    backup_workflow_state,
    create_initial_workflow_state,
)

from .progress_writer import (
    create_progress_file,
    append_session_record,
    get_last_n_lines,
    update_progress_header,
)

__all__ = [
    # Models
    "VerificationConfig",
    "StepDefinition",
    "WorkflowState",
    "LayoutWorkflowState",
    "StepResult",
    "VerificationResult",
    # Workflow manager
    "load_workflow_state",
    "save_workflow_state",
    "validate_workflow_state",
    "backup_workflow_state",
    "create_initial_workflow_state",
    # Progress writer
    "create_progress_file",
    "append_session_record",
    "get_last_n_lines",
    "update_progress_header",
]
