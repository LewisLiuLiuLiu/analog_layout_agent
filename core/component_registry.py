"""
Component Registry - 组件注册表

管理布局过程中创建的所有组件，提供命名、查找、跟踪功能
"""

import re
from typing import Optional, Dict, Any, List, Tuple, Iterator
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ComponentInfo:
    """组件信息数据类"""
    
    name: str                          # 组件名称
    component: Any                     # gdsfactory Component实例
    device_type: str                   # 器件类型 (nmos, pmos, current_mirror等)
    params: Dict[str, Any]             # 创建参数
    created_at: datetime = field(default_factory=datetime.now)  # 创建时间
    ports: List[str] = field(default_factory=list)  # 端口列表
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # 边界框 (xmin, ymin, xmax, ymax)
    parent: Optional[str] = None       # 父组件名称（用于层次化设计）
    children: List[str] = field(default_factory=list)  # 子组件列表
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    @property
    def size(self) -> Tuple[float, float]:
        """获取组件尺寸 (width, height)"""
        return (self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])
    
    @property
    def area(self) -> float:
        """获取组件面积"""
        w, h = self.size
        return w * h
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "device_type": self.device_type,
            "params": self.params,
            "created_at": self.created_at.isoformat(),
            "ports": self.ports,
            "bbox": self.bbox,
            "size": self.size,
            "area": self.area,
            "parent": self.parent,
            "children": self.children,
            "metadata": self.metadata
        }


class ComponentRegistry:
    """组件注册表
    
    负责管理布局过程中创建的所有组件，提供：
    - 自动命名和重名处理
    - 组件查找和遍历
    - 层次结构管理
    - 组件信息跟踪
    
    Usage:
        >>> registry = ComponentRegistry()
        >>> name = registry.register(component, "nmos", params)
        >>> comp_info = registry.get(name)
        >>> registry.list_by_type("nmos")
    """
    
    def __init__(self):
        """初始化组件注册表"""
        self._components: Dict[str, ComponentInfo] = {}
        self._name_counters: Dict[str, int] = {}  # 用于自动编号
        self._type_index: Dict[str, List[str]] = {}  # 类型索引
        
    def register(
        self,
        component: Any,
        device_type: str,
        params: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        parent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """注册组件
        
        Args:
            component: gdsfactory Component实例
            device_type: 器件类型 (如 "nmos", "pmos", "current_mirror")
            params: 创建参数字典
            name: 指定名称，如果为None则自动生成
            prefix: 名称前缀，用于自动生成名称
            parent: 父组件名称（用于层次化设计）
            metadata: 额外元数据
            
        Returns:
            注册的组件名称
            
        Raises:
            ValueError: 如果指定的名称已存在
        """
        # 生成或验证名称
        if name is None:
            name = self._generate_name(device_type, prefix)
        elif name in self._components:
            raise ValueError(f"组件名称已存在: {name}")
        
        # 提取组件信息
        ports = []
        bbox = (0, 0, 0, 0)
        
        if component is not None:
            # 提取端口
            if hasattr(component, 'ports'):
                ports = list(component.ports.keys())
            
            # 提取边界框
            if hasattr(component, 'xmin'):
                bbox = (
                    float(component.xmin),
                    float(component.ymin),
                    float(component.xmax),
                    float(component.ymax)
                )
        
        # 创建组件信息
        comp_info = ComponentInfo(
            name=name,
            component=component,
            device_type=device_type,
            params=params or {},
            ports=ports,
            bbox=bbox,
            parent=parent,
            metadata=metadata or {}
        )
        
        # 注册组件
        self._components[name] = comp_info
        
        # 更新类型索引
        if device_type not in self._type_index:
            self._type_index[device_type] = []
        self._type_index[device_type].append(name)
        
        # 更新父组件的children列表
        if parent and parent in self._components:
            self._components[parent].children.append(name)
        
        return name
    
    def _generate_name(self, device_type: str, prefix: Optional[str] = None) -> str:
        """生成唯一的组件名称
        
        Args:
            device_type: 器件类型
            prefix: 可选的名称前缀
            
        Returns:
            生成的唯一名称
        """
        base = prefix or device_type
        
        if base not in self._name_counters:
            self._name_counters[base] = 0
        
        # 递增计数器直到找到未使用的名称
        while True:
            self._name_counters[base] += 1
            name = f"{base}_{self._name_counters[base]}"
            if name not in self._components:
                return name
    
    def get(self, name: str) -> Optional[ComponentInfo]:
        """获取组件信息
        
        Args:
            name: 组件名称
            
        Returns:
            ComponentInfo实例，如果不存在则返回None
        """
        return self._components.get(name)
    
    def get_component(self, name: str) -> Optional[Any]:
        """获取组件实例
        
        Args:
            name: 组件名称
            
        Returns:
            Component实例，如果不存在则返回None
        """
        info = self._components.get(name)
        return info.component if info else None
    
    def exists(self, name: str) -> bool:
        """检查组件是否存在
        
        Args:
            name: 组件名称
            
        Returns:
            是否存在
        """
        return name in self._components
    
    def remove(self, name: str) -> bool:
        """移除组件
        
        Args:
            name: 组件名称
            
        Returns:
            是否成功移除
        """
        if name not in self._components:
            return False
        
        comp_info = self._components[name]
        
        # 从类型索引中移除
        if comp_info.device_type in self._type_index:
            self._type_index[comp_info.device_type].remove(name)
        
        # 从父组件的children列表中移除
        if comp_info.parent and comp_info.parent in self._components:
            parent_info = self._components[comp_info.parent]
            if name in parent_info.children:
                parent_info.children.remove(name)
        
        # 处理子组件（设为无父组件）
        for child_name in comp_info.children:
            if child_name in self._components:
                self._components[child_name].parent = None
        
        # 删除组件
        del self._components[name]
        return True
    
    def rename(self, old_name: str, new_name: str) -> bool:
        """重命名组件
        
        Args:
            old_name: 原名称
            new_name: 新名称
            
        Returns:
            是否成功重命名
        """
        if old_name not in self._components:
            return False
        if new_name in self._components:
            raise ValueError(f"新名称已存在: {new_name}")
        
        # 获取组件信息并更新名称
        comp_info = self._components[old_name]
        comp_info.name = new_name
        
        # 更新注册表
        del self._components[old_name]
        self._components[new_name] = comp_info
        
        # 更新类型索引
        if comp_info.device_type in self._type_index:
            idx = self._type_index[comp_info.device_type].index(old_name)
            self._type_index[comp_info.device_type][idx] = new_name
        
        # 更新父组件的children列表
        if comp_info.parent and comp_info.parent in self._components:
            parent_info = self._components[comp_info.parent]
            if old_name in parent_info.children:
                idx = parent_info.children.index(old_name)
                parent_info.children[idx] = new_name
        
        # 更新子组件的parent引用
        for child_name in comp_info.children:
            if child_name in self._components:
                self._components[child_name].parent = new_name
        
        return True
    
    def list_all(self) -> List[str]:
        """列出所有组件名称
        
        Returns:
            组件名称列表
        """
        return list(self._components.keys())
    
    def list_by_type(self, device_type: str) -> List[str]:
        """按类型列出组件
        
        Args:
            device_type: 器件类型
            
        Returns:
            该类型的组件名称列表
        """
        return self._type_index.get(device_type, []).copy()
    
    def list_types(self) -> List[str]:
        """列出所有器件类型
        
        Returns:
            器件类型列表
        """
        return list(self._type_index.keys())
    
    def count(self) -> int:
        """获取组件总数
        
        Returns:
            组件数量
        """
        return len(self._components)
    
    def count_by_type(self, device_type: str) -> int:
        """获取特定类型的组件数量
        
        Args:
            device_type: 器件类型
            
        Returns:
            该类型的组件数量
        """
        return len(self._type_index.get(device_type, []))
    
    def get_children(self, name: str) -> List[str]:
        """获取组件的子组件列表
        
        Args:
            name: 组件名称
            
        Returns:
            子组件名称列表
        """
        info = self._components.get(name)
        return info.children.copy() if info else []
    
    def get_parent(self, name: str) -> Optional[str]:
        """获取组件的父组件
        
        Args:
            name: 组件名称
            
        Returns:
            父组件名称，如果没有则返回None
        """
        info = self._components.get(name)
        return info.parent if info else None
    
    def find_by_port(self, port_pattern: str) -> List[Tuple[str, str]]:
        """查找包含指定端口的组件
        
        Args:
            port_pattern: 端口名称模式（支持正则表达式）
            
        Returns:
            (组件名称, 端口名称) 元组列表
        """
        results = []
        pattern = re.compile(port_pattern)
        
        for name, info in self._components.items():
            for port in info.ports:
                if pattern.match(port):
                    results.append((name, port))
        
        return results
    
    def search(self, query: str) -> List[str]:
        """搜索组件（按名称或类型）
        
        Args:
            query: 搜索关键词
            
        Returns:
            匹配的组件名称列表
        """
        results = []
        query_lower = query.lower()
        
        for name, info in self._components.items():
            if query_lower in name.lower() or query_lower in info.device_type.lower():
                results.append(name)
        
        return results
    
    def get_port(self, port_ref: str) -> Optional[Any]:
        """通过端口引用获取端口对象
        
        Args:
            port_ref: 端口引用，格式为 "component_name.port_name"
            
        Returns:
            端口对象，如果不存在则返回None
        """
        if '.' not in port_ref:
            return None
        
        comp_name, port_name = port_ref.rsplit('.', 1)
        info = self._components.get(comp_name)
        
        if info is None or info.component is None:
            return None
        
        if hasattr(info.component, 'ports') and port_name in info.component.ports:
            return info.component.ports[port_name]
        
        return None
    
    def update_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """更新组件元数据
        
        Args:
            name: 组件名称
            metadata: 要更新的元数据
            
        Returns:
            是否成功更新
        """
        if name not in self._components:
            return False
        
        self._components[name].metadata.update(metadata)
        return True
    
    def clear(self) -> None:
        """清空注册表"""
        self._components.clear()
        self._name_counters.clear()
        self._type_index.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典格式
        
        Returns:
            包含所有组件信息的字典
        """
        return {
            "component_count": len(self._components),
            "types": list(self._type_index.keys()),
            "components": {
                name: info.to_dict() 
                for name, info in self._components.items()
            }
        }
    
    def summary(self) -> str:
        """生成注册表摘要
        
        Returns:
            摘要字符串
        """
        lines = [
            f"组件注册表摘要:",
            f"  总组件数: {self.count()}"
        ]
        
        for device_type in sorted(self._type_index.keys()):
            count = self.count_by_type(device_type)
            lines.append(f"  - {device_type}: {count}个")
        
        return "\n".join(lines)
    
    def __iter__(self) -> Iterator[Tuple[str, ComponentInfo]]:
        """迭代所有组件"""
        return iter(self._components.items())
    
    def __len__(self) -> int:
        """获取组件数量"""
        return len(self._components)
    
    def __contains__(self, name: str) -> bool:
        """检查组件是否存在"""
        return name in self._components
    
    def __getitem__(self, name: str) -> ComponentInfo:
        """通过名称获取组件信息"""
        if name not in self._components:
            raise KeyError(f"组件不存在: {name}")
        return self._components[name]
