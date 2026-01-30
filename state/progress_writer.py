"""
Progress file writer for Layout Agent Loop

Manages progress.md file that tracks execution history and current state.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

from .models import WorkflowState, StepDefinition, StepResult

logger = logging.getLogger(__name__)


def _format_dict(d: dict, indent: int = 2) -> str:
    """Format dictionary as indented JSON-like string"""
    return json.dumps(d, indent=indent, ensure_ascii=False, default=str)


def create_progress_file(workflow: WorkflowState, path: Path) -> None:
    """
    Create initial progress.md file from workflow state.
    
    Args:
        workflow: WorkflowState to create progress file for
        path: Path to progress.md file
    """
    content = f"""# Layout Design Progress: {workflow.design_name}

## Design Information
- **PDK**: {workflow.pdk}
- **Design Name**: {workflow.design_name}
- **Created**: {workflow.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- **Last Updated**: {workflow.updated_at.strftime('%Y-%m-%d %H:%M:%S')}

## Current Status
**Progress**: 0/{len(workflow.steps)} steps completed (0%)

## Steps Overview

"""
    
    # Add steps overview
    for i, step in enumerate(workflow.steps):
        status = "[x]" if workflow.completed[i] else "[ ]"
        content += f"{status} Step {step.step_id}: {step.description}\n"
    
    content += "\n---\n\n## Session Records\n\n"
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write file
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Created progress file: {path}")


def update_progress_header(workflow: WorkflowState, path: Path) -> None:
    """
    Update the progress header section with current status.
    
    Args:
        workflow: Current WorkflowState
        path: Path to progress.md file
    """
    if not path.exists():
        create_progress_file(workflow, path)
        return
    
    content = path.read_text(encoding='utf-8')
    
    # Update progress line
    completed_count = sum(workflow.completed)
    total_count = len(workflow.completed)
    progress_percent = workflow.get_progress_percentage()
    
    new_progress = f"**Progress**: {completed_count}/{total_count} steps completed ({progress_percent:.0f}%)"
    
    # Find and replace progress line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('**Progress**:'):
            lines[i] = new_progress
            break
        if line.startswith('- **Last Updated**:'):
            lines[i] = f"- **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Update steps overview checkboxes
    in_steps_section = False
    for i, line in enumerate(lines):
        if line.startswith('## Steps Overview'):
            in_steps_section = True
            continue
        if in_steps_section and line.startswith('---'):
            break
        if in_steps_section and line.strip():
            # Match step lines like "[ ] Step 1:" or "[x] Step 1:"
            for j, step in enumerate(workflow.steps):
                if f"Step {step.step_id}:" in line:
                    status = "[x]" if workflow.completed[j] else "[ ]"
                    lines[i] = f"{status} Step {step.step_id}: {step.description}"
                    break
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def append_session_record(
    path: Path,
    step: StepDefinition,
    result: StepResult,
    session_number: Optional[int] = None
) -> None:
    """
    Append a session record to progress.md.
    
    Args:
        path: Path to progress.md file
        step: Step definition that was executed
        result: Result of step execution
        session_number: Optional session number (auto-calculated if None)
    """
    if not path.exists():
        raise FileNotFoundError(f"Progress file not found: {path}")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Calculate session number if not provided
    if session_number is None:
        content = path.read_text(encoding='utf-8')
        session_number = content.count('### Session ') + 1
    
    # Build session record
    status_icon = "[x]" if result.success else "[ ]"
    status_text = "Success" if result.success else "Failed"
    
    params_str = _format_dict(step.parameters)
    
    record = f"""
### Session {session_number} - {timestamp}

**Target**: Step {step.step_id} - {step.description}

**Tool**: `{step.tool}`
**Parameters**:
```json
{params_str}
```

**Execution Result**:
- {status_icon} Step {step.step_id}: {step.description} ({status_text})
"""
    
    if result.data:
        data_str = _format_dict(result.data)
        record += f"""
**Output Data**:
```json
{data_str}
```
"""
    
    if result.success:
        msg = result.message or 'Step completed successfully'
        record += f"""
**Verification**: PASS
- {msg}
"""
    else:
        error_msg = 'Unknown error'
        error_type = 'unknown'
        if result.error:
            error_msg = result.error.get('message', 'Unknown error')
            error_type = result.error.get('type', 'unknown')
        record += f"""
**Verification**: FAIL
- Error: {error_msg}
- Type: {error_type}
"""
    
    record += "\n---\n"
    
    # Append to file
    with open(path, 'a', encoding='utf-8') as f:
        f.write(record)
    
    logger.debug(f"Appended session {session_number} record to {path}")


def get_last_n_lines(path: Path, n: int = 50) -> list[str]:
    """
    Get the last N lines from progress.md file.
    
    Args:
        path: Path to progress.md file
        n: Number of lines to return (default: 50)
        
    Returns:
        list[str]: Last N lines from the file
    """
    if not path.exists():
        return []
    
    content = path.read_text(encoding='utf-8')
    lines = content.splitlines()
    
    if len(lines) <= n:
        return lines
    
    return lines[-n:]


def get_progress_summary(path: Path) -> dict:
    """
    Parse progress.md and return summary information.
    
    Args:
        path: Path to progress.md file
        
    Returns:
        dict: Summary including completed steps, failed steps, etc.
    """
    if not path.exists():
        return {
            "exists": False,
            "total_sessions": 0,
            "successful_sessions": 0,
            "failed_sessions": 0
        }
    
    content = path.read_text(encoding='utf-8')
    
    # Count sessions
    total_sessions = content.count('### Session ')
    successful = content.count('**Verification**: PASS')
    failed = content.count('**Verification**: FAIL')
    
    return {
        "exists": True,
        "total_sessions": total_sessions,
        "successful_sessions": successful,
        "failed_sessions": failed,
        "last_lines": get_last_n_lines(path, 20)
    }
"""
Progress file writer for Layout Agent Loop

Manages progress.md file that tracks execution history and current state.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

from .models import WorkflowState, StepDefinition, StepResult

logger = logging.getLogger(__name__)


def create_progress_file(workflow: WorkflowState, path: Path) -> None:
    """
    Create initial progress.md file from workflow state.
    
    Args:
        workflow: WorkflowState to create progress file for
        path: Path to progress.md file
    """
    content = f"""# Layout Design Progress: {workflow.design_name}

## Design Information
- **PDK**: {workflow.pdk}
- **Design Name**: {workflow.design_name}
- **Created**: {workflow.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- **Last Updated**: {workflow.updated_at.strftime('%Y-%m-%d %H:%M:%S')}

## Current Status
**Progress**: 0/{len(workflow.steps)} steps completed (0%)

## Steps Overview

"""
    
    # Add steps overview
    for i, step in enumerate(workflow.steps):
        status = "[x]" if workflow.completed[i] else "[ ]"
        content += f"{status} Step {step.step_id}: {step.description}\n"
    
    content += "\n---\n\n## Session Records\n\n"
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write file
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Created progress file: {path}")


def update_progress_header(workflow: WorkflowState, path: Path) -> None:
    """
    Update the progress header section with current status.
    
    Args:
        workflow: Current WorkflowState
        path: Path to progress.md file
    """
    if not path.exists():
        create_progress_file(workflow, path)
        return
    
    content = path.read_text(encoding='utf-8')
    
    # Update progress line
    completed_count = sum(workflow.completed)
    total_count = len(workflow.completed)
    progress_percent = workflow.get_progress_percentage()
    
    new_progress = f"**Progress**: {completed_count}/{total_count} steps completed ({progress_percent:.0f}%)"
    
    # Find and replace progress line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('**Progress**:'):
            lines[i] = new_progress
            break
        if line.startswith('- **Last Updated**:'):
            lines[i] = f"- **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Update steps overview checkboxes
    in_steps_section = False
    for i, line in enumerate(lines):
        if line.startswith('## Steps Overview'):
            in_steps_section = True
            continue
        if in_steps_section and line.startswith('---'):
            break
        if in_steps_section and line.strip():
            # Match step lines like "[ ] Step 1:" or "[x] Step 1:"
            for j, step in enumerate(workflow.steps):
                if f"Step {step.step_id}:" in line:
                    status = "[x]" if workflow.completed[j] else "[ ]"
                    lines[i] = f"{status} Step {step.step_id}: {step.description}"
                    break
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def append_session_record(
    path: Path,
    step: StepDefinition,
    result: StepResult,
    session_number: Optional[int] = None
) -> None:
    """
    Append a session record to progress.md.
    
    Args:
        path: Path to progress.md file
        step: Step definition that was executed
        result: Result of step execution
        session_number: Optional session number (auto-calculated if None)
    """
    if not path.exists():
        raise FileNotFoundError(f"Progress file not found: {path}")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Calculate session number if not provided
    if session_number is None:
        content = path.read_text(encoding='utf-8')
        session_number = content.count('### Session ') + 1
    
    # Build session record
    status_icon = "[x]" if result.success else "[ ]"
    status_text = "Success" if result.success else "Failed"
    
    record = f"""
### Session {session_number} - {timestamp}

**Target**: Step {step.step_id} - {step.description}

**Tool**: `{step.tool}`
**Parameters**:
```json
{_format_dict(step.parameters)}
```

**Execution Result**:
- {status_icon} Step {step.step_id}: {step.description} ({status_text})
"""
    
    if result.data:
        record += f"""
**Output Data**:
```json
{_format_dict(result.data)}
```
"""
    
    if result.success:
        record += f"""
**Verification**: PASS
- {result.message or 'Step completed successfully'}
"""
    else:
        record += f"""
**Verification**: FAIL
- Error: {result.error.get('message', 'Unknown error') if result.error else 'Unknown error'}
- Type: {result.error.get('type', 'unknown') if result.error else 'unknown'}
"""
    
    record += "\n---\n"
    
    # Append to file
    with open(path, 'a', encoding='utf-8') as f:
        f.write(record)
    
    logger.debug(f"Appended session {session_number} record to {path}")


def get_last_n_lines(path: Path, n: int = 50) -> list[str]:
    """
    Get the last N lines from progress.md file.
    
    Args:
        path: Path to progress.md file
        n: Number of lines to return (default: 50)
        
    Returns:
        list[str]: Last N lines from the file
    """
    if not path.exists():
        return []
    
    content = path.read_text(encoding='utf-8')
    lines = content.splitlines()
    
    if len(lines) <= n:
        return lines
    
    return lines[-n:]


def get_progress_summary(path: Path) -> dict:
    """
    Parse progress.md and return summary information.
    
    Args:
        path: Path to progress.md file
        
    Returns:
        dict: Summary including completed steps, failed steps, etc.
    """
    if not path.exists():
        return {
            "exists": False,
            "total_sessions": 0,
            "successful_sessions": 0,
            "failed_sessions": 0
        }
    
    content = path.read_text(encoding='utf-8')
    
    # Count sessions
    total_sessions = content.count('### Session ')
    successful = content.count('**Verification**: PASS')
    failed = content.count('**Verification**: FAIL')
    
    return {
        "exists": True,
        "total_sessions": total_sessions,
        "successful_sessions": successful,
        "failed_sessions": failed,
        "last_lines": get_last_n_lines(path, 20)
    }


def _format_dict(d: dict, indent: int = 2) -> str:
    """Format dictionary as indented JSON-like string"""
    import json
    return json.dumps(d, indent=indent, ensure_ascii=False, default=str)
