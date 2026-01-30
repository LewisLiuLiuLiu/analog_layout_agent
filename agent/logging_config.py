"""
Logging configuration for Layout Agent Loop
布局代理循环的日志配置

Supports both Pydantic Logfire (for observability) and standard Python logging.
支持 Pydantic Logfire（用于可观测性）和标准 Python 日志。
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Try to import logfire (optional dependency)
# 尝试导入 logfire（可选依赖）
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
    为布局代理循环配置日志。
    
    Sets up:
    设置内容:
    - Console handler (INFO level by default) / 控制台处理器（默认 INFO 级别）
    - File handler (DEBUG level by default) / 文件处理器（默认 DEBUG 级别）
    - Logfire integration (if available and enabled) / Logfire 集成（如果可用且已启用）
    
    Args:
        project_name: Name for Logfire project / Logfire 项目名称
        log_file: Path to log file (optional) / 日志文件路径（可选）
        console_level: Log level for console output / 控制台输出的日志级别
        file_level: Log level for file output / 文件输出的日志级别
        use_logfire: Whether to enable Logfire integration / 是否启用 Logfire 集成
        
    Returns:
        logging.Logger: Configured logger for layout_agent / 已配置的 layout_agent 日志记录器
    """
    # Create logger / 创建日志记录器
    logger = logging.getLogger("layout_agent")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers / 移除现有处理器
    logger.handlers.clear()
    
    # Create formatters / 创建格式化器
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler / 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if path provided) / 文件处理器（如果提供了路径）
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Logfire integration / Logfire 集成
    if use_logfire and LOGFIRE_AVAILABLE:
        try:
            logfire.configure(
                project_name=project_name,
                send_to_logfire='if-token-present'
            )
            # Instrument PydanticAI if available
            # 如果 PydanticAI 可用则进行插桩
            logfire.instrument_pydantic_ai()
            logger.info(f"Logfire configured for project: {project_name}")
        except Exception as e:
            logger.warning(f"Failed to configure Logfire: {e}")
    
    return logger


def get_logger(name: str = "layout_agent") -> logging.Logger:
    """
    Get a logger instance.
    获取日志记录器实例。
    
    Args:
        name: Logger name (default: layout_agent) / 日志记录器名称（默认: layout_agent）
        
    Returns:
        logging.Logger: Logger instance / 日志记录器实例
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
    使用结构化数据记录步骤执行。
    
    Args:
        logger: Logger instance / 日志记录器实例
        step_id: Step ID / 步骤 ID
        tool: Tool name / 工具名称
        parameters: Tool parameters / 工具参数
        success: Whether execution succeeded / 执行是否成功
        result: Result data (if success) / 结果数据（如果成功）
        error: Error message (if failed) / 错误消息（如果失败）
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
    记录工作流生命周期事件。
    
    Args:
        logger: Logger instance / 日志记录器实例
        event: Event type (start, complete, error, etc.) / 事件类型（start, complete, error 等）
        design_name: Design name / 设计名称
        **kwargs: Additional event data / 额外的事件数据
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
    """Context manager for workflow logging with automatic timing.
    工作流日志的上下文管理器，带自动计时功能。"""
    
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
        return False  # Don't suppress exceptions / 不抑制异常
