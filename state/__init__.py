"""
State management module for Layout Agent Loop

This module provides:
- WorkflowState: Workflow state data model
- StepDefinition: Step definition data model
- LayoutWorkflowState: State class for pydantic-graph
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
