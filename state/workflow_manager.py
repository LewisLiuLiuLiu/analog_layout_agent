"""
Workflow state manager for Layout Agent Loop

Handles reading, writing, validation, and backup of workflow_state.json files.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

from .models import WorkflowState, StepDefinition, VerificationConfig

logger = logging.getLogger(__name__)


def load_workflow_state(path: Path) -> WorkflowState:
    """
    Load workflow state from JSON file.
    
    Args:
        path: Path to workflow_state.json
        
    Returns:
        WorkflowState: Parsed and validated workflow state
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or doesn't match schema
    """
    if not path.exists():
        raise FileNotFoundError(f"Workflow state file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Parse datetime strings
    if 'created_at' in data and isinstance(data['created_at'], str):
        data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
    if 'updated_at' in data and isinstance(data['updated_at'], str):
        data['updated_at'] = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))
    
    # Parse steps
    if 'steps' in data:
        parsed_steps = []
        for step_data in data['steps']:
            # Parse verification config
            if 'verification' in step_data and isinstance(step_data['verification'], dict):
                step_data['verification'] = VerificationConfig(**step_data['verification'])
            parsed_steps.append(StepDefinition(**step_data))
        data['steps'] = parsed_steps
    
    return WorkflowState(**data)


def save_workflow_state(workflow: WorkflowState, path: Path, backup: bool = True) -> None:
    """
    Save workflow state to JSON file.
    
    Only saves changes to 'completed' array and 'updated_at' timestamp.
    Creates a backup before writing if backup=True.
    
    Args:
        workflow: WorkflowState to save
        path: Path to workflow_state.json
        backup: Whether to create backup before saving (default: True)
    """
    # Update timestamp
    workflow.updated_at = datetime.now()
    
    # Backup existing file
    if backup and path.exists():
        backup_workflow_state(path)
    
    # Serialize to JSON
    data = {
        "design_name": workflow.design_name,
        "pdk": workflow.pdk,
        "created_at": workflow.created_at.isoformat(),
        "updated_at": workflow.updated_at.isoformat(),
        "steps": [step.model_dump() for step in workflow.steps],
        "completed": workflow.completed
    }
    
    # Convert nested objects
    for step in data['steps']:
        if isinstance(step.get('verification'), VerificationConfig):
            step['verification'] = step['verification'].model_dump()
    
    # Write to file
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved workflow state to {path}")


def validate_workflow_state(path: Path) -> tuple[bool, Optional[str]]:
    """
    Validate workflow state JSON file format.
    
    Args:
        path: Path to workflow_state.json
        
    Returns:
        tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        if not path.exists():
            return False, f"File not found: {path}"
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = ['design_name', 'pdk', 'steps', 'completed']
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Check steps is a list
        if not isinstance(data['steps'], list):
            return False, "steps must be a list"
        
        # Check completed is a list
        if not isinstance(data['completed'], list):
            return False, "completed must be a list"
        
        # Check array lengths match
        if len(data['completed']) != len(data['steps']):
            return False, f"completed length ({len(data['completed'])}) != steps length ({len(data['steps'])})"
        
        # Check completed values are all boolean
        for i, val in enumerate(data['completed']):
            if not isinstance(val, bool):
                return False, f"completed[{i}] = {val} (type: {type(val).__name__}), must be boolean"
        
        # Check each step has required fields
        step_required = ['step_id', 'category', 'description', 'tool', 'parameters', 'verification']
        for i, step in enumerate(data['steps']):
            for field in step_required:
                if field not in step:
                    return False, f"Step {i} missing required field: {field}"
        
        # Validate step_id sequence
        step_ids = [s['step_id'] for s in data['steps']]
        expected_ids = list(range(1, len(data['steps']) + 1))
        if step_ids != expected_ids:
            return False, f"step_id must be sequential starting from 1: got {step_ids}"
        
        # Validate depends_on references
        for step in data['steps']:
            for dep_id in step.get('depends_on', []):
                if dep_id >= step['step_id']:
                    return False, f"Step {step['step_id']} cannot depend on step {dep_id} (forward reference)"
                if dep_id < 1 or dep_id > len(data['steps']):
                    return False, f"Step {step['step_id']} has invalid dependency: {dep_id}"
        
        return True, None
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def backup_workflow_state(path: Path) -> Path:
    """
    Create a timestamped backup of workflow state file.
    
    Args:
        path: Path to workflow_state.json
        
    Returns:
        Path: Path to backup file
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot backup non-existent file: {path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"workflow_state.backup_{timestamp}.json"
    backup_path = path.parent / backup_name
    
    shutil.copy2(path, backup_path)
    logger.debug(f"Created backup: {backup_path}")
    
    return backup_path


def create_initial_workflow_state(
    design_name: str,
    pdk: str,
    steps: list[dict],
    output_path: Path
) -> WorkflowState:
    """
    Create initial workflow state from Reasoning Agent output.
    
    Args:
        design_name: Name of the design
        pdk: PDK name (e.g., 'sky130')
        steps: List of step definitions from Reasoning Agent
        output_path: Path to save workflow_state.json
        
    Returns:
        WorkflowState: Created workflow state
    """
    # Parse steps
    parsed_steps = []
    for step_data in steps:
        # Ensure verification is a VerificationConfig
        if 'verification' in step_data and isinstance(step_data['verification'], dict):
            step_data['verification'] = VerificationConfig(**step_data['verification'])
        parsed_steps.append(StepDefinition(**step_data))
    
    # Create workflow state with all steps initially incomplete
    workflow = WorkflowState(
        design_name=design_name,
        pdk=pdk,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        steps=parsed_steps,
        completed=[False] * len(parsed_steps)
    )
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    save_workflow_state(workflow, output_path, backup=False)
    
    logger.info(f"Created initial workflow state: {output_path}")
    logger.info(f"  Design: {design_name}, PDK: {pdk}, Steps: {len(parsed_steps)}")
    
    return workflow


def get_workflow_summary(workflow: WorkflowState) -> dict:
    """
    Get a summary of workflow state for logging/display.
    
    Args:
        workflow: WorkflowState to summarize
        
    Returns:
        dict: Summary information
    """
    completed_count = sum(workflow.completed)
    total_count = len(workflow.completed)
    next_step = workflow.get_first_incomplete_step_definition()
    
    return {
        "design_name": workflow.design_name,
        "pdk": workflow.pdk,
        "total_steps": total_count,
        "completed_steps": completed_count,
        "progress_percent": workflow.get_progress_percentage(),
        "is_complete": workflow.is_all_completed(),
        "next_step": {
            "step_id": next_step.step_id if next_step else None,
            "description": next_step.description if next_step else None,
            "tool": next_step.tool if next_step else None
        } if next_step else None,
        "created_at": workflow.created_at.isoformat(),
        "updated_at": workflow.updated_at.isoformat()
    }
