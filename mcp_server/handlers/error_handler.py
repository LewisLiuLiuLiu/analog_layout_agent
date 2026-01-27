"""
Error Handler - 错误处理器

统一的错误处理和恢复机制
"""

import traceback
import logging
from typing import Optional, Dict, Any, List, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps


# 配置日志
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度"""
    INFO = "info"           # 信息级别，不影响执行
    WARNING = "warning"     # 警告级别，可能影响结果
    ERROR = "error"         # 错误级别，操作失败
    CRITICAL = "critical"   # 严重错误，需要停止


class ErrorCategory(Enum):
    """错误类别"""
    VALIDATION = "validation"      # 参数验证错误
    PDK = "pdk"                    # PDK相关错误
    DEVICE = "device"              # 器件创建错误
    ROUTING = "routing"            # 路由错误
    PLACEMENT = "placement"        # 放置错误
    DRC = "drc"                    # DRC违规
    LVS = "lvs"                    # LVS不匹配
    EXPORT = "export"              # 导出错误
    SYSTEM = "system"              # 系统错误
    UNKNOWN = "unknown"            # 未知错误


@dataclass
class ErrorRecord:
    """错误记录"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    traceback: Optional[str] = None
    recovered: bool = False
    recovery_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback,
            "recovered": self.recovered,
            "recovery_action": self.recovery_action
        }


class LayoutError(Exception):
    """布局错误基类"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}


class ValidationError(LayoutError):
    """参数验证错误"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            details=details
        )


class PDKError(LayoutError):
    """PDK相关错误"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            category=ErrorCategory.PDK,
            severity=ErrorSeverity.ERROR,
            details=details
        )


class DeviceError(LayoutError):
    """器件创建错误"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            category=ErrorCategory.DEVICE,
            severity=ErrorSeverity.ERROR,
            details=details
        )


class RoutingError(LayoutError):
    """路由错误"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            category=ErrorCategory.ROUTING,
            severity=ErrorSeverity.ERROR,
            details=details
        )


class DRCError(LayoutError):
    """DRC违规错误"""
    
    def __init__(
        self,
        message: str,
        violations: Optional[List[Dict[str, Any]]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["violations"] = violations or []
        super().__init__(
            message,
            category=ErrorCategory.DRC,
            severity=ErrorSeverity.WARNING,
            details=details
        )


class LVSError(LayoutError):
    """LVS不匹配错误"""
    
    def __init__(
        self,
        message: str,
        mismatches: Optional[List[Dict[str, Any]]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        details["mismatches"] = mismatches or []
        super().__init__(
            message,
            category=ErrorCategory.LVS,
            severity=ErrorSeverity.ERROR,
            details=details
        )


class ErrorHandler:
    """错误处理器
    
    提供统一的错误处理和恢复机制：
    - 错误捕获和记录
    - 错误分类和严重程度评估
    - 恢复策略执行
    - 错误历史查询
    
    Usage:
        >>> handler = ErrorHandler()
        >>> @handler.catch_errors
        ... def my_function():
        ...     pass
        >>> handler.handle_error(ValueError("test"))
    """
    
    _instance: Optional['ErrorHandler'] = None
    
    def __new__(cls) -> 'ErrorHandler':
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化错误处理器"""
        if self._initialized:
            return
        
        self._error_history: List[ErrorRecord] = []
        self._max_history: int = 100
        self._error_counter: int = 0
        self._recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self._initialized = True
        
        # 注册默认恢复策略
        self._register_default_strategies()
    
    def _register_default_strategies(self) -> None:
        """注册默认恢复策略"""
        self._recovery_strategies[ErrorCategory.VALIDATION] = self._recover_validation
        self._recovery_strategies[ErrorCategory.DRC] = self._recover_drc
        self._recovery_strategies[ErrorCategory.ROUTING] = self._recover_routing
    
    def handle_error(
        self,
        error: Exception,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorRecord:
        """处理错误
        
        Args:
            error: 异常对象
            category: 错误类别（如果是LayoutError会自动获取）
            severity: 严重程度（如果是LayoutError会自动获取）
            context: 额外上下文信息
            
        Returns:
            ErrorRecord对象
        """
        # 生成错误ID
        self._error_counter += 1
        error_id = f"ERR_{self._error_counter:04d}"
        
        # 确定错误类别和严重程度
        if isinstance(error, LayoutError):
            category = category or error.category
            severity = severity or error.severity
            details = error.details.copy()
            message = error.message
        else:
            category = category or self._infer_category(error)
            severity = severity or ErrorSeverity.ERROR
            details = {}
            message = str(error)
        
        # 添加上下文信息
        if context:
            details.update(context)
        
        # 获取堆栈跟踪
        tb = traceback.format_exc()
        
        # 创建错误记录
        record = ErrorRecord(
            error_id=error_id,
            category=category,
            severity=severity,
            message=message,
            details=details,
            traceback=tb
        )
        
        # 记录错误
        self._record_error(record)
        
        # 尝试恢复
        if category in self._recovery_strategies:
            try:
                recovery_action = self._recovery_strategies[category](error, details)
                if recovery_action:
                    record.recovered = True
                    record.recovery_action = recovery_action
            except Exception as e:
                logger.warning(f"恢复策略执行失败: {e}")
        
        # 日志记录
        self._log_error(record)
        
        return record
    
    def _infer_category(self, error: Exception) -> ErrorCategory:
        """推断错误类别
        
        Args:
            error: 异常对象
            
        Returns:
            错误类别
        """
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # 基于错误类型推断
        type_mapping = {
            "ValueError": ErrorCategory.VALIDATION,
            "TypeError": ErrorCategory.VALIDATION,
            "KeyError": ErrorCategory.VALIDATION,
            "ImportError": ErrorCategory.SYSTEM,
            "RuntimeError": ErrorCategory.SYSTEM,
            "FileNotFoundError": ErrorCategory.EXPORT,
        }
        
        if error_type in type_mapping:
            return type_mapping[error_type]
        
        # 基于错误消息推断
        keyword_mapping = {
            "pdk": ErrorCategory.PDK,
            "layer": ErrorCategory.PDK,
            "width": ErrorCategory.VALIDATION,
            "length": ErrorCategory.VALIDATION,
            "route": ErrorCategory.ROUTING,
            "connect": ErrorCategory.ROUTING,
            "place": ErrorCategory.PLACEMENT,
            "drc": ErrorCategory.DRC,
            "lvs": ErrorCategory.LVS,
            "export": ErrorCategory.EXPORT,
            "gds": ErrorCategory.EXPORT,
        }
        
        for keyword, category in keyword_mapping.items():
            if keyword in error_msg:
                return category
        
        return ErrorCategory.UNKNOWN
    
    def _record_error(self, record: ErrorRecord) -> None:
        """记录错误到历史
        
        Args:
            record: 错误记录
        """
        self._error_history.append(record)
        
        # 限制历史长度
        if len(self._error_history) > self._max_history:
            self._error_history = self._error_history[-self._max_history:]
    
    def _log_error(self, record: ErrorRecord) -> None:
        """记录错误到日志
        
        Args:
            record: 错误记录
        """
        log_msg = (
            f"[{record.error_id}] {record.category.value.upper()}: "
            f"{record.message}"
        )
        
        if record.severity == ErrorSeverity.INFO:
            logger.info(log_msg)
        elif record.severity == ErrorSeverity.WARNING:
            logger.warning(log_msg)
        elif record.severity == ErrorSeverity.ERROR:
            logger.error(log_msg)
        elif record.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_msg)
    
    # 恢复策略
    def _recover_validation(
        self,
        error: Exception,
        details: Dict[str, Any]
    ) -> Optional[str]:
        """验证错误恢复策略
        
        Args:
            error: 异常对象
            details: 错误详情
            
        Returns:
            恢复操作描述
        """
        # 尝试提供修正建议
        if "width" in str(error).lower():
            return "建议: 检查宽度参数是否满足PDK最小尺寸要求"
        if "length" in str(error).lower():
            return "建议: 检查长度参数是否满足PDK最小尺寸要求"
        return None
    
    def _recover_drc(
        self,
        error: Exception,
        details: Dict[str, Any]
    ) -> Optional[str]:
        """DRC错误恢复策略
        
        Args:
            error: 异常对象
            details: 错误详情
            
        Returns:
            恢复操作描述
        """
        violations = details.get("violations", [])
        if not violations:
            return None
        
        # 分析违规类型
        actions = []
        for v in violations[:3]:  # 只处理前3个
            rule = v.get("rule", "")
            if "spacing" in rule.lower():
                actions.append(f"增大{rule}相关的间距")
            elif "width" in rule.lower():
                actions.append(f"增大{rule}相关的宽度")
            elif "enclosure" in rule.lower():
                actions.append(f"增大{rule}相关的包络")
        
        if actions:
            return "建议: " + "; ".join(actions)
        return None
    
    def _recover_routing(
        self,
        error: Exception,
        details: Dict[str, Any]
    ) -> Optional[str]:
        """路由错误恢复策略
        
        Args:
            error: 异常对象
            details: 错误详情
            
        Returns:
            恢复操作描述
        """
        error_msg = str(error).lower()
        
        if "parallel" in error_msg or "inline" in error_msg:
            return "建议: 尝试使用smart_route自动选择路由策略"
        if "layer" in error_msg:
            return "建议: 检查路由层是否与端口层匹配，可能需要via连接"
        
        return None
    
    def register_recovery_strategy(
        self,
        category: ErrorCategory,
        strategy: Callable[[Exception, Dict[str, Any]], Optional[str]]
    ) -> None:
        """注册恢复策略
        
        Args:
            category: 错误类别
            strategy: 恢复策略函数
        """
        self._recovery_strategies[category] = strategy
    
    def catch_errors(
        self,
        category: Optional[ErrorCategory] = None,
        reraise: bool = True
    ) -> Callable:
        """错误捕获装饰器
        
        Args:
            category: 默认错误类别
            reraise: 是否重新抛出异常
            
        Returns:
            装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.handle_error(e, category=category)
                    if reraise:
                        raise
                    return None
            return wrapper
        return decorator
    
    def get_error_history(
        self,
        limit: Optional[int] = None,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None
    ) -> List[ErrorRecord]:
        """获取错误历史
        
        Args:
            limit: 返回的最大记录数
            category: 过滤类别
            severity: 过滤严重程度
            
        Returns:
            错误记录列表
        """
        records = self._error_history.copy()
        
        if category:
            records = [r for r in records if r.category == category]
        
        if severity:
            records = [r for r in records if r.severity == severity]
        
        if limit:
            records = records[-limit:]
        
        return records
    
    def get_last_error(self) -> Optional[ErrorRecord]:
        """获取最后一个错误
        
        Returns:
            最后的错误记录
        """
        return self._error_history[-1] if self._error_history else None
    
    def clear_history(self) -> None:
        """清空错误历史"""
        self._error_history.clear()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要
        
        Returns:
            错误摘要字典
        """
        summary = {
            "total_errors": len(self._error_history),
            "by_category": {},
            "by_severity": {},
            "recovered_count": 0
        }
        
        for record in self._error_history:
            # 按类别统计
            cat = record.category.value
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
            
            # 按严重程度统计
            sev = record.severity.value
            summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1
            
            # 恢复统计
            if record.recovered:
                summary["recovered_count"] += 1
        
        return summary
    
    def format_error_for_llm(self, record: ErrorRecord) -> str:
        """格式化错误信息供LLM理解
        
        Args:
            record: 错误记录
            
        Returns:
            LLM友好的错误描述
        """
        lines = [
            f"错误 [{record.error_id}]:",
            f"  类型: {record.category.value}",
            f"  严重程度: {record.severity.value}",
            f"  描述: {record.message}"
        ]
        
        if record.details:
            lines.append("  详情:")
            for key, value in record.details.items():
                if key != "violations" and key != "mismatches":
                    lines.append(f"    - {key}: {value}")
        
        if record.recovery_action:
            lines.append(f"  恢复建议: {record.recovery_action}")
        
        return "\n".join(lines)
