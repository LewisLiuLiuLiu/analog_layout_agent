"""
Logging configuration for Layout Agent Loop

Supports both Pydantic Logfire (for observability) and standard Python logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Try to import logfire (optional dependency)
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None


def setup_agent_logging(
    project_name: str = "layout-agent",
    log_file: Optional[Path] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    use_logfire: bool = True
) -> logging.Logger:
    """
    Configure logging for Layout Agent Loop.
    
    Sets up:
    - Console handler (INFO level by default)
    - File handler (DEBUG level by default)
    - Logfire integration (if available and enabled)
    
    Args:
        project_name: Name for Logfire project
        log_file: Path to log file (optional)
        console_level: Log level for console output
        file_level: Log level for file output
        use_logfire: Whether to enable Logfire integration
        
    Returns:
        logging.Logger: Configured logger for layout_agent
    """
    # Create logger
    logger = logging.getLogger("layout_agent")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if path provided)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Logfire integration
    if use_logfire and LOGFIRE_AVAILABLE:
        try:
            logfire.configure(
                project_name=project_name,
                send_to_logfire='if-token-present'
            )
            # Instrument PydanticAI if available
            logfire.instrument_pydantic_ai()
            logger.info(f"Logfire configured for project: {project_name}")
        except Exception as e:
            logger.warning(f"Failed to configure Logfire: {e}")
    
    return logger


def get_logger(name: str = "layout_agent") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (default: layout_agent)
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


def log_step_execution(
    logger: logging.Logger,
    step_id: int,
    tool: str,
    parameters: dict,
    success: bool,
    result: Optional[dict] = None,
    error: Optional[str] = None
) -> None:
    """
    Log step execution with structured data.
    
    Args:
        logger: Logger instance
        step_id: Step ID
        tool: Tool name
        parameters: Tool parameters
        success: Whether execution succeeded
        result: Result data (if success)
        error: Error message (if failed)
    """
    if success:
        logger.info(
            f"Step {step_id} completed: {tool}",
            extra={
                "step_id": step_id,
                "tool": tool,
                "parameters": parameters,
                "success": True,
                "result": result
            }
        )
    else:
        logger.error(
            f"Step {step_id} failed: {tool} - {error}",
            extra={
                "step_id": step_id,
                "tool": tool,
                "parameters": parameters,
                "success": False,
                "error": error
            }
        )


def log_workflow_event(
    logger: logging.Logger,
    event: str,
    design_name: str,
    **kwargs
) -> None:
    """
    Log workflow lifecycle events.
    
    Args:
        logger: Logger instance
        event: Event type (start, complete, error, etc.)
        design_name: Design name
        **kwargs: Additional event data
    """
    logger.info(
        f"Workflow {event}: {design_name}",
        extra={
            "event": event,
            "design_name": design_name,
            **kwargs
        }
    )


class WorkflowLogContext:
    """Context manager for workflow logging with automatic timing."""
    
    def __init__(self, logger: logging.Logger, workflow_name: str, step_id: Optional[int] = None):
        self.logger = logger
        self.workflow_name = workflow_name
        self.step_id = step_id
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        if self.step_id:
            self.logger.info(f"Starting step {self.step_id} in {self.workflow_name}")
        else:
            self.logger.info(f"Starting workflow: {self.workflow_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type:
            self.logger.error(
                f"Workflow {self.workflow_name} failed after {duration:.2f}s: {exc_val}"
            )
        else:
            if self.step_id:
                self.logger.info(f"Step {self.step_id} completed in {duration:.2f}s")
            else:
                self.logger.info(f"Workflow {self.workflow_name} completed in {duration:.2f}s")
        return False  # Don't suppress exceptions
