"""
State Handler - 状态管理器

管理MCP Server的全局状态，包括会话、上下文、缓存等
"""

import uuid
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import sys
from pathlib import Path
_CORE_PATH = Path(__file__).parent.parent.parent / "core"
if str(_CORE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(_CORE_PATH.parent))

from core.layout_context import LayoutContext
from core.pdk_manager import PDKManager


class SessionState(Enum):
    """会话状态"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class Session:
    """会话信息"""
    session_id: str
    created_at: datetime
    context: LayoutContext
    state: SessionState = SessionState.ACTIVE
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self) -> None:
        """更新最后活动时间"""
        self.last_activity = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "state": self.state.value,
            "last_activity": self.last_activity.isoformat(),
            "context_summary": self.context.summary(),
            "metadata": self.metadata
        }


class StateHandler:
    """状态管理器
    
    管理MCP Server的全局状态，提供：
    - 会话管理（创建、获取、销毁）
    - 全局配置管理
    - 缓存管理
    - 线程安全的状态访问
    
    Usage:
        >>> handler = StateHandler()
        >>> session = handler.create_session(pdk_name="sky130")
        >>> context = handler.get_context(session.session_id)
    """
    
    _instance: Optional['StateHandler'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'StateHandler':
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化状态管理器"""
        if self._initialized:
            return
        
        self._sessions: Dict[str, Session] = {}
        self._active_session_id: Optional[str] = None
        self._config: Dict[str, Any] = {
            "default_pdk": "sky130",
            "max_sessions": 10,
            "session_timeout_minutes": 60,
            "auto_cleanup": True
        }
        self._cache: Dict[str, Any] = {}
        self._session_lock = threading.Lock()
        self._initialized = True
    
    def create_session(
        self,
        pdk_name: Optional[str] = None,
        design_name: str = "top_level",
        metadata: Optional[Dict[str, Any]] = None,
        make_active: bool = True
    ) -> Session:
        """创建新会话
        
        Args:
            pdk_name: PDK名称，默认使用配置中的default_pdk
            design_name: 设计名称
            metadata: 额外元数据
            make_active: 是否设为活动会话
            
        Returns:
            新创建的Session对象
        """
        with self._session_lock:
            # 检查会话数量限制
            if len(self._sessions) >= self._config["max_sessions"]:
                self._cleanup_old_sessions()
            
            # 生成会话ID
            session_id = str(uuid.uuid4())[:8]
            
            # 创建布局上下文
            pdk_name = pdk_name or self._config["default_pdk"]
            try:
                context = LayoutContext(
                    pdk_name=pdk_name,
                    design_name=design_name
                )
            except ImportError:
                # 如果无法加载PDK（依赖未安装），创建无PDK的上下文
                context = LayoutContext(
                    pdk=None,
                    pdk_name=None,
                    design_name=design_name
                )
            
            # 创建会话
            session = Session(
                session_id=session_id,
                created_at=datetime.now(),
                context=context,
                metadata=metadata or {}
            )
            
            # 注册会话
            self._sessions[session_id] = session
            
            if make_active:
                self._active_session_id = session_id
            
            return session
    
    def get_session(self, session_id: Optional[str] = None) -> Optional[Session]:
        """获取会话
        
        Args:
            session_id: 会话ID，None表示获取活动会话
            
        Returns:
            Session对象，如果不存在则返回None
        """
        session_id = session_id or self._active_session_id
        if session_id is None:
            return None
        
        session = self._sessions.get(session_id)
        if session:
            session.touch()
        
        return session
    
    def get_context(self, session_id: Optional[str] = None) -> Optional[LayoutContext]:
        """获取会话的布局上下文
        
        Args:
            session_id: 会话ID，None表示获取活动会话的上下文
            
        Returns:
            LayoutContext对象
        """
        session = self.get_session(session_id)
        return session.context if session else None
    
    def get_active_session(self) -> Optional[Session]:
        """获取活动会话
        
        Returns:
            活动Session对象
        """
        return self.get_session(self._active_session_id)
    
    def set_active_session(self, session_id: str) -> bool:
        """设置活动会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功设置
        """
        if session_id in self._sessions:
            self._active_session_id = session_id
            return True
        return False
    
    def close_session(self, session_id: Optional[str] = None) -> bool:
        """关闭会话
        
        Args:
            session_id: 会话ID，None表示关闭活动会话
            
        Returns:
            是否成功关闭
        """
        with self._session_lock:
            session_id = session_id or self._active_session_id
            if session_id is None or session_id not in self._sessions:
                return False
            
            session = self._sessions[session_id]
            session.state = SessionState.COMPLETED
            
            # 清理资源
            session.context.clear()
            
            del self._sessions[session_id]
            
            if self._active_session_id == session_id:
                self._active_session_id = None
            
            return True
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话
        
        Returns:
            会话信息列表
        """
        return [
            session.to_dict() 
            for session in self._sessions.values()
        ]
    
    def _cleanup_old_sessions(self) -> None:
        """清理过期会话"""
        if not self._config["auto_cleanup"]:
            return
        
        timeout_minutes = self._config["session_timeout_minutes"]
        now = datetime.now()
        
        to_remove = []
        for session_id, session in self._sessions.items():
            age_minutes = (now - session.last_activity).total_seconds() / 60
            if age_minutes > timeout_minutes and session.state != SessionState.ACTIVE:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            self.close_session(session_id)
    
    # 配置管理
    def get_config(self, key: Optional[str] = None) -> Any:
        """获取配置
        
        Args:
            key: 配置键，None表示获取全部配置
            
        Returns:
            配置值
        """
        if key is None:
            return self._config.copy()
        return self._config.get(key)
    
    def set_config(self, key: str, value: Any) -> None:
        """设置配置
        
        Args:
            key: 配置键
            value: 配置值
        """
        self._config[key] = value
    
    # 缓存管理
    def set_cache(self, key: str, value: Any) -> None:
        """设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        self._cache[key] = value
    
    def get_cache(self, key: str, default: Any = None) -> Any:
        """获取缓存
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            缓存值
        """
        return self._cache.get(key, default)
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态管理器状态
        
        Returns:
            状态信息字典
        """
        return {
            "session_count": len(self._sessions),
            "active_session_id": self._active_session_id,
            "config": self._config,
            "cache_size": len(self._cache)
        }
    
    def reset(self) -> None:
        """重置状态管理器"""
        with self._session_lock:
            for session in self._sessions.values():
                session.context.clear()
            
            self._sessions.clear()
            self._active_session_id = None
            self._cache.clear()
