"""
Layout Context - 布局上下文

管理整个布局过程的状态，包括PDK、组件、连接等信息
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# 添加gLayout路径
_GLAYOUT_PATH = Path(__file__).parent.parent.parent / "gLayout" / "src"
if str(_GLAYOUT_PATH) not in sys.path:
    sys.path.insert(0, str(_GLAYOUT_PATH))

# 设置 PDK_ROOT 环境变量（glayout 初始化时需要）
_PDK_ROOT = Path(__file__).parent.parent.parent / "skywater-pdk"
if _PDK_ROOT.exists() and not os.environ.get('PDK_ROOT'):
    os.environ['PDK_ROOT'] = str(_PDK_ROOT)

from .pdk_manager import PDKManager
from .component_registry import ComponentRegistry, ComponentInfo


class ContextState(Enum):
    """上下文状态枚举"""
    INITIALIZED = "initialized"
    DESIGNING = "designing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ComponentPlacement:
    """组件放置信息"""
    component_name: str
    x: float = 0.0
    y: float = 0.0
    rotation: int = 0  # 旋转角度 (0, 90, 180, 270)
    mirror: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_name": self.component_name,
            "x": self.x,
            "y": self.y,
            "rotation": self.rotation,
            "mirror": self.mirror
        }


@dataclass
class Connection:
    """连接信息"""
    source: str           # 源端口 (格式: component_name.port_name)
    target: str           # 目标端口
    layer: str            # 路由层
    route_type: str       # 路由类型 (smart/c/l/straight)
    route_component: Optional[str] = None  # 路由组件名称
    width: Optional[float] = None  # 路由宽度
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "source": self.source,
            "target": self.target,
            "layer": self.layer,
            "route_type": self.route_type,
            "route_component": self.route_component,
            "width": self.width,
            "created_at": self.created_at.isoformat()
        }


class LayoutContext:
    """布局上下文
    
    管理整个布局过程的状态，包括：
    - PDK和设计规则
    - 组件注册和管理
    - 连接关系跟踪
    - 顶层组件组装
    - 历史记录和撤销
    
    Usage:
        >>> context = LayoutContext(pdk_name="sky130")
        >>> name = context.register_component(comp, "nmos", params)
        >>> context.add_connection("nmos_1.drain", "pmos_1.drain", "met2")
        >>> top = context.build_top_level()
    """
    
    def __init__(
        self,
        pdk: Optional[Any] = None,
        pdk_name: Optional[str] = None,
        design_name: str = "top_level"
    ):
        """初始化布局上下文
        
        Args:
            pdk: MappedPDK实例（可选）
            pdk_name: PDK名称，如果没有提供pdk则使用此名称加载
            design_name: 设计名称
        """
        # PDK设置
        if pdk is not None:
            self._pdk = pdk
            self._pdk_name = pdk.name if hasattr(pdk, 'name') else 'unknown'
        elif pdk_name is not None:
            self._pdk = PDKManager.load_pdk(pdk_name)
            self._pdk_name = pdk_name
        else:
            self._pdk = None
            self._pdk_name = None
        
        # 基本属性
        self.design_name = design_name
        self.state = ContextState.INITIALIZED
        self.created_at = datetime.now()
        
        # 组件管理
        self.registry = ComponentRegistry()
        
        # 组件放置信息管理
        self._placements: Dict[str, ComponentPlacement] = {}  # component_name -> ComponentPlacement
        
        # 连接管理
        self._connections: List[Connection] = []
        
        # 顶层组件（懒加载）
        self._top_level: Optional[Any] = None
        self._top_level_dirty: bool = True  # 标记是否需要重建
        
        # 历史记录（用于撤销）
        self._history: List[Dict[str, Any]] = []
        self._max_history: int = 50
        
        # 验证结果缓存
        self._drc_result: Optional[Dict[str, Any]] = None
        self._lvs_result: Optional[Dict[str, Any]] = None
        
        # 输出目录
        self._output_dir: Optional[Path] = None
    
    @property
    def pdk(self) -> Any:
        """获取PDK实例"""
        if self._pdk is None:
            raise RuntimeError("PDK未初始化。请先设置PDK。")
        return self._pdk
    
    @property
    def pdk_name(self) -> Optional[str]:
        """获取PDK名称"""
        return self._pdk_name
    
    def set_pdk(self, pdk_name: str) -> None:
        """设置/切换PDK
        
        Args:
            pdk_name: PDK名称
        """
        self._pdk = PDKManager.load_pdk(pdk_name)
        self._pdk_name = pdk_name
        self._mark_dirty()
    
    def register_component(
        self,
        component: Any,
        device_type: str,
        params: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        parent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """注册组件到上下文
        
        Args:
            component: gdsfactory Component实例
            device_type: 器件类型
            params: 创建参数
            name: 指定名称
            prefix: 名称前缀
            parent: 父组件名称
            metadata: 额外元数据
            
        Returns:
            注册的组件名称
        """
        # 记录历史
        self._record_history("register_component", {
            "device_type": device_type,
            "params": params,
            "name": name
        })
        
        # 注册组件
        registered_name = self.registry.register(
            component=component,
            device_type=device_type,
            params=params,
            name=name,
            prefix=prefix,
            parent=parent,
            metadata=metadata
        )
        
        # 初始化放置信息（默认在原点）
        if registered_name not in self._placements:
            self._placements[registered_name] = ComponentPlacement(
                component_name=registered_name,
                x=0.0,
                y=0.0,
                rotation=0
            )
        
        # 标记需要重建顶层组件
        self._mark_dirty()
        
        # 更新状态
        if self.state == ContextState.INITIALIZED:
            self.state = ContextState.DESIGNING
        
        return registered_name
    
    def get_component(self, name: str) -> Optional[Any]:
        """获取组件实例
        
        Args:
            name: 组件名称
            
        Returns:
            Component实例
        """
        return self.registry.get_component(name)
    
    def get_component_info(self, name: str) -> Optional[ComponentInfo]:
        """获取组件信息
        
        Args:
            name: 组件名称
            
        Returns:
            ComponentInfo实例
        """
        return self.registry.get(name)
    
    def list_components(self, device_type: Optional[str] = None) -> List[str]:
        """列出组件
        
        Args:
            device_type: 过滤器件类型，None表示全部
            
        Returns:
            组件名称列表
        """
        if device_type:
            return self.registry.list_by_type(device_type)
        return self.registry.list_all()
    
    def get_port(self, port_ref: str) -> Optional[Any]:
        """获取端口对象
        
        Args:
            port_ref: 端口引用，格式 "component_name.port_name"
            
        Returns:
            端口对象
        """
        return self.registry.get_port(port_ref)
    
    def add_connection(
        self,
        source: str,
        target: str,
        layer: str = "met2",
        route_type: str = "smart",
        route_component: Optional[str] = None,
        width: Optional[float] = None
    ) -> int:
        """添加连接
        
        Args:
            source: 源端口引用
            target: 目标端口引用
            layer: 路由层
            route_type: 路由类型
            route_component: 路由组件名称
            width: 路由宽度
            
        Returns:
            连接索引
        """
        # 验证端口存在
        if '.' in source:
            src_port = self.get_port(source)
            if src_port is None:
                raise ValueError(f"源端口不存在: {source}")
        
        if '.' in target:
            tgt_port = self.get_port(target)
            if tgt_port is None:
                raise ValueError(f"目标端口不存在: {target}")
        
        # 创建连接
        conn = Connection(
            source=source,
            target=target,
            layer=layer,
            route_type=route_type,
            route_component=route_component,
            width=width
        )
        
        # 记录历史
        self._record_history("add_connection", conn.to_dict())
        
        # 添加连接
        self._connections.append(conn)
        self._mark_dirty()
        
        return len(self._connections) - 1
    
    def set_placement(
        self,
        component_name: str,
        x: float = 0.0,
        y: float = 0.0,
        rotation: int = 0,
        mirror: bool = False
    ) -> bool:
        """设置组件放置位置
        
        Args:
            component_name: 组件名称
            x: X坐标
            y: Y坐标
            rotation: 旋转角度 (0, 90, 180, 270)
            mirror: 是否镜像
            
        Returns:
            是否成功设置
        """
        if component_name not in self.registry:
            return False
        
        self._placements[component_name] = ComponentPlacement(
            component_name=component_name,
            x=x,
            y=y,
            rotation=rotation,
            mirror=mirror
        )
        
        self._mark_dirty()
        return True
    
    def move_component(self, component_name: str, dx: float = 0.0, dy: float = 0.0) -> bool:
        """移动组件（相对位移）
        
        Args:
            component_name: 组件名称
            dx: X方向移动距离
            dy: Y方向移动距离
            
        Returns:
            是否成功移动
        """
        if component_name not in self._placements:
            return False
        
        placement = self._placements[component_name]
        placement.x += dx
        placement.y += dy
        
        self._mark_dirty()
        return True
    
    def get_placement(self, component_name: str) -> Optional[ComponentPlacement]:
        """获取组件放置信息
        
        Args:
            component_name: 组件名称
            
        Returns:
            ComponentPlacement 或 None
        """
        return self._placements.get(component_name)
    
    def get_connections(self, component_name: Optional[str] = None) -> List[Connection]:
        """获取连接列表
        
        Args:
            component_name: 过滤特定组件的连接
            
        Returns:
            连接列表
        """
        if component_name is None:
            return self._connections.copy()
        
        return [
            conn for conn in self._connections
            if (conn.source.startswith(f"{component_name}.") or
                conn.target.startswith(f"{component_name}."))
        ]
    
    def remove_connection(self, index: int) -> bool:
        """移除连接
        
        Args:
            index: 连接索引
            
        Returns:
            是否成功移除
        """
        if 0 <= index < len(self._connections):
            self._connections.pop(index)
            self._mark_dirty()
            return True
        return False
    
    def build_top_level(self, force: bool = False) -> Any:
        """构建顶层组件
        
        Args:
            force: 强制重建
            
        Returns:
            顶层Component
        """
        if not force and not self._top_level_dirty and self._top_level is not None:
            return self._top_level
        
        try:
            from gdsfactory import Component
        except ImportError:
            raise ImportError("需要安装gdsfactory来构建顶层组件")
        
        # 创建顶层组件
        top = Component(self.design_name)
        
        # 添加所有已注册的组件，并应用放置信息
        for name, info in self.registry:
            if info.component is not None:
                ref = top.add_ref(info.component, alias=name)
                
                # 应用放置信息
                placement = self._placements.get(name)
                if placement:
                    # 设置位置
                    ref.move((placement.x, placement.y))
                    
                    # 应用旋转
                    if placement.rotation != 0:
                        ref.rotate(placement.rotation)
                    
                    # 应用镜像
                    if placement.mirror:
                        ref.mirror()
        
        # 应用连接（添加路由组件）
        added_routes = set()  # 避免重复添加
        for conn in self._connections:
            if conn.route_component and conn.route_component not in added_routes:
                route_info = self.registry.get_info(conn.route_component)
                if route_info and route_info.component is not None:
                    route_ref = top.add_ref(route_info.component, alias=conn.route_component)
                    
                    # 应用路由组件的放置信息
                    route_placement = self._placements.get(conn.route_component)
                    if route_placement:
                        route_ref.move((route_placement.x, route_placement.y))
                        if route_placement.rotation != 0:
                            route_ref.rotate(route_placement.rotation)
                    
                    added_routes.add(conn.route_component)
                else:
                    # 路由组件不存在，记录警告
                    import logging
                    logging.warning(
                        f"路由组件 '{conn.route_component}' 未找到，"
                        f"连接 {conn.source} -> {conn.target} 未应用路由"
                    )
        
        self._top_level = top
        self._top_level_dirty = False
        
        return top
    
    def _mark_dirty(self) -> None:
        """标记顶层组件需要重建"""
        self._top_level_dirty = True
        self._drc_result = None  # 清除验证缓存
        self._lvs_result = None
    
    def _record_history(self, action: str, data: Dict[str, Any]) -> None:
        """记录历史
        
        Args:
            action: 操作类型
            data: 操作数据
        """
        record = {
            "action": action,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        self._history.append(record)
        
        # 限制历史记录长度
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取历史记录
        
        Args:
            limit: 返回的最大记录数
            
        Returns:
            历史记录列表
        """
        if limit:
            return self._history[-limit:]
        return self._history.copy()
    
    def set_output_dir(self, path: Union[str, Path]) -> None:
        """设置输出目录
        
        Args:
            path: 输出目录路径
        """
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_output_dir(self) -> Path:
        """获取输出目录
        
        Returns:
            输出目录路径
        """
        if self._output_dir is None:
            self._output_dir = Path(tempfile.mkdtemp(prefix="layout_"))
        return self._output_dir
    
    def export_gds(self, filename: Optional[str] = None) -> Path:
        """导出GDS文件
        
        Args:
            filename: 文件名，默认使用设计名称
            
        Returns:
            导出的文件路径
        """
        top = self.build_top_level()
        
        if filename:
            filepath = Path(filename)
            # 如果 filepath 是相对路径，但包含 ./ 或 ../，则相对于当前工作目录
            if not filepath.is_absolute():
                if str(filename).startswith('./') or str(filename).startswith('../'):
                    # 相对于当前工作目录
                    filepath = Path.cwd() / filepath
                else:
                    # 放在默认输出目录中
                    output_dir = self.get_output_dir()
                    filepath = output_dir / filepath
            # 确保父目录存在
            filepath.parent.mkdir(parents=True, exist_ok=True)
        else :
            output_dir = self.get_output_dir()
            filepath = output_dir / f"{self.design_name}.gds"
        
        top.write_gds(str(filepath))
        
        return filepath
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取布局统计信息
        
        Returns:
            统计信息字典
        """
        total_area = 0.0
        device_counts = {}
        
        for name, info in self.registry:
            total_area += info.area
            device_type = info.device_type
            device_counts[device_type] = device_counts.get(device_type, 0) + 1
        
        return {
            "design_name": self.design_name,
            "pdk_name": self._pdk_name,
            "state": self.state.value,
            "component_count": self.registry.count(),
            "connection_count": len(self._connections),
            "device_counts": device_counts,
            "total_area": total_area,
            "created_at": self.created_at.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典格式
        
        Returns:
            上下文状态字典
        """
        return {
            "design_name": self.design_name,
            "pdk_name": self._pdk_name,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "components": self.registry.to_dict(),
            "connections": [conn.to_dict() for conn in self._connections],
            "statistics": self.get_statistics()
        }
    
    def to_natural_language(self) -> str:
        """转换为自然语言描述
        
        Returns:
            自然语言描述字符串
        """
        lines = [
            f"当前布局状态 ({self.design_name}):",
            f"  PDK: {self._pdk_name}",
            f"  状态: {self.state.value}",
            f""
        ]
        
        # 组件信息
        comp_count = self.registry.count()
        lines.append(f"组件 ({comp_count}个):")
        
        for name, info in self.registry:
            size_str = f"{info.size[0]:.2f}x{info.size[1]:.2f}um"
            ports_str = ", ".join(info.ports[:5])
            if len(info.ports) > 5:
                ports_str += f"... (+{len(info.ports)-5}个)"
            lines.append(f"  - {name}: {info.device_type}, 尺寸={size_str}")
            lines.append(f"    端口: {ports_str}")
        
        # 连接信息
        lines.append(f"\n连接 ({len(self._connections)}条):")
        for conn in self._connections:
            lines.append(f"  - {conn.source} -> {conn.target} (层:{conn.layer}, 类型:{conn.route_type})")
        
        return "\n".join(lines)
    
    def summary(self) -> str:
        """生成简短摘要
        
        Returns:
            摘要字符串
        """
        stats = self.get_statistics()
        return (
            f"设计: {self.design_name} | "
            f"PDK: {self._pdk_name} | "
            f"组件: {stats['component_count']}个 | "
            f"连接: {stats['connection_count']}条 | "
            f"状态: {self.state.value}"
        )
    
    def clear(self) -> None:
        """清空上下文"""
        self.registry.clear()
        self._placements.clear()
        self._connections.clear()
        self._top_level = None
        self._top_level_dirty = True
        self._history.clear()
        self._drc_result = None
        self._lvs_result = None
        self.state = ContextState.INITIALIZED
    
    def __repr__(self) -> str:
        return f"<LayoutContext design='{self.design_name}' pdk='{self._pdk_name}' components={self.registry.count()}>"
