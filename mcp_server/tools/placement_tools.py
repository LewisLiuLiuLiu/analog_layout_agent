"""
Placement Tools - 放置工具

封装gLayout的放置功能，提供MCP工具接口
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# 添加路径
_BASE_PATH = Path(__file__).parent.parent.parent
if str(_BASE_PATH) not in sys.path:
    sys.path.insert(0, str(_BASE_PATH))

_GLAYOUT_PATH = _BASE_PATH.parent / "gLayout" / "src"
if str(_GLAYOUT_PATH) not in sys.path:
    sys.path.insert(0, str(_GLAYOUT_PATH))

# 设置 PDK_ROOT 环境变量（glayout 初始化时需要）
_PDK_ROOT = _BASE_PATH.parent / "skywater-pdk"
if _PDK_ROOT.exists() and not os.environ.get('PDK_ROOT'):
    os.environ['PDK_ROOT'] = str(_PDK_ROOT)

from core.layout_context import LayoutContext
from mcp_server.schemas.common_schemas import (
    PLACE_COMPONENT_SCHEMA, ALIGN_TO_PORT_SCHEMA,
    MOVE_COMPONENT_SCHEMA, INTERDIGITIZE_SCHEMA
)
from mcp_server.handlers.error_handler import (
    ValidationError, LayoutError, ErrorHandler
)


class PlacementError(LayoutError):
    """放置错误"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        from mcp_server.handlers.error_handler import ErrorCategory, ErrorSeverity
        super().__init__(
            message,
            category=ErrorCategory.PLACEMENT,
            severity=ErrorSeverity.ERROR,
            details=details
        )


class PlacementToolExecutor:
    """放置工具执行器
    
    封装gLayout的放置功能，提供统一的接口。
    支持绝对放置、相对放置、对齐、互指式布局等操作。
    
    Usage:
        >>> executor = PlacementToolExecutor(context)
        >>> result = executor.place_component("nmos_1", x=10, y=20)
        >>> result = executor.align_to_port("pmos_1", "nmos_1.gate_W")
    """
    
    def __init__(self, context: LayoutContext):
        """初始化执行器
        
        Args:
            context: 布局上下文
        """
        self.context = context
        self.error_handler = ErrorHandler()
        self._glayout_available = self._check_glayout()
    
    def _check_glayout(self) -> bool:
        """检查gLayout是否可用"""
        try:
            from glayout import move, movex, movey
            return True
        except ImportError:
            return False
    
    def _get_component(self, name: str):
        """获取组件
        
        Args:
            name: 组件名称
            
        Returns:
            组件对象
            
        Raises:
            ValidationError: 如果组件不存在
        """
        comp = self.context.get_component(name)
        if comp is None:
            raise ValidationError(
                f"组件不存在: {name}",
                {"available_components": self.context.list_components()[:10]}
            )
        return comp
    
    def place_component(
        self,
        component_name: str,
        x: float = 0,
        y: float = 0,
        rotation: int = 0,
        mirror: bool = False
    ) -> Dict[str, Any]:
        """放置组件到指定位置
        
        Args:
            component_name: 组件名称
            x: X坐标(um)
            y: Y坐标(um)
            rotation: 旋转角度(0/90/180/270)
            mirror: 是否镜像
            
        Returns:
            放置结果
        """
        if rotation not in [0, 90, 180, 270]:
            raise ValidationError(
                f"无效的旋转角度: {rotation}",
                {"valid_angles": [0, 90, 180, 270]}
            )
        
        # 获取组件信息
        info = self.context.get_component_info(component_name)
        if info is None:
            raise ValidationError(
                f"组件不存在: {component_name}",
                {"available_components": self.context.list_components()[:10]}
            )
        
        comp = info.component
        
        if comp is None or not self._glayout_available:
            # 模拟模式：使用新的放置API
            self.context.set_placement(
                component_name,
                x=x,
                y=y,
                rotation=rotation,
                mirror=mirror
            )
            
            return {
                "success": True,
                "component_name": component_name,
                "position": {"x": x, "y": y},
                "rotation": rotation,
                "mirror": mirror,
                "_mock": True
            }
        
        try:
            from glayout import move
            
            # 移动组件
            move(comp, (x, y))
            
            # 旋转（如果需要）
            if rotation != 0:
                comp.rotate(rotation)
            
            # 镜像（如果需要）
            if mirror:
                comp.mirror()
            
            # 更新元数据
            self.context.registry.update_metadata(component_name, {
                "position": (x, y),
                "rotation": rotation,
                "mirror": mirror
            })
            
            return {
                "success": True,
                "component_name": component_name,
                "position": {"x": x, "y": y},
                "rotation": rotation,
                "mirror": mirror,
                "new_bbox": {
                    "xmin": float(comp.xmin),
                    "ymin": float(comp.ymin),
                    "xmax": float(comp.xmax),
                    "ymax": float(comp.ymax)
                }
            }
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise PlacementError(
                f"放置组件失败: {e}",
                {"component": component_name, "position": (x, y)}
            )
    
    def align_to_port(
        self,
        component_name: str,
        target_port: str,
        alignment: str = "center",
        offset_x: float = 0,
        offset_y: float = 0
    ) -> Dict[str, Any]:
        """将组件对齐到目标端口
        
        Args:
            component_name: 要对齐的组件名称
            target_port: 目标端口(格式: component_name.port_name)
            alignment: 对齐方式(center/left/right/top/bottom)
            offset_x: X方向偏移
            offset_y: Y方向偏移
            
        Returns:
            对齐结果
        """
        if alignment not in ["center", "left", "right", "top", "bottom"]:
            raise ValidationError(
                f"无效的对齐方式: {alignment}",
                {"valid_alignments": ["center", "left", "right", "top", "bottom"]}
            )
        
        # 获取组件信息
        info = self.context.get_component_info(component_name)
        if info is None:
            raise ValidationError(
                f"组件不存在: {component_name}",
                {"available_components": self.context.list_components()[:10]}
            )
        
        # 解析目标端口
        if '.' not in target_port:
            raise ValidationError(
                f"无效的端口引用格式: {target_port}",
                {"expected_format": "component_name.port_name"}
            )
        
        comp = info.component
        port = self.context.get_port(target_port)
        
        if comp is None or port is None or not self._glayout_available:
            # 模拟模式
            self.context.registry.update_metadata(component_name, {
                "aligned_to": target_port,
                "alignment": alignment,
                "offset": (offset_x, offset_y)
            })
            
            return {
                "success": True,
                "component_name": component_name,
                "aligned_to": target_port,
                "alignment": alignment,
                "offset": {"x": offset_x, "y": offset_y},
                "_mock": True
            }
        
        try:
            from glayout import align_comp_to_port
            
            # 执行对齐
            align_comp_to_port(comp, port, alignment=alignment)
            
            # 应用偏移
            if offset_x != 0 or offset_y != 0:
                from glayout import move
                current_pos = (comp.xmin, comp.ymin)
                move(comp, (current_pos[0] + offset_x, current_pos[1] + offset_y))
            
            # 更新元数据
            self.context.registry.update_metadata(component_name, {
                "aligned_to": target_port,
                "alignment": alignment,
                "offset": (offset_x, offset_y),
                "position": (float(comp.xmin), float(comp.ymin))
            })
            
            return {
                "success": True,
                "component_name": component_name,
                "aligned_to": target_port,
                "alignment": alignment,
                "offset": {"x": offset_x, "y": offset_y},
                "new_position": {
                    "x": float(comp.xmin),
                    "y": float(comp.ymin)
                }
            }
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise PlacementError(
                f"对齐组件失败: {e}",
                {"component": component_name, "target": target_port}
            )
    
    def move_component(
        self,
        component_name: str,
        dx: float = 0,
        dy: float = 0
    ) -> Dict[str, Any]:
        """移动组件（相对位移）
        
        Args:
            component_name: 组件名称
            dx: X方向移动距离
            dy: Y方向移动距离
            
        Returns:
            移动结果
        """
        info = self.context.get_component_info(component_name)
        if info is None:
            raise ValidationError(
                f"组件不存在: {component_name}",
                {"available_components": self.context.list_components()[:10]}
            )
        
        comp = info.component
        
        if comp is None or not self._glayout_available:
            # 模拟模式：使用新的移动API
            self.context.move_component(component_name, dx=dx, dy=dy)
            
            # 获取新的放置信息
            placement = self.context.get_placement(component_name)
            if placement:
                new_pos = (placement.x, placement.y)
                old_pos = (placement.x - dx, placement.y - dy)
            else:
                new_pos = (dx, dy)
                old_pos = (0, 0)
            
            return {
                "success": True,
                "component_name": component_name,
                "moved_by": {"dx": dx, "dy": dy},
                "old_position": {"x": old_pos[0], "y": old_pos[1]},
                "new_position": {"x": new_pos[0], "y": new_pos[1]},
                "_mock": True
            }
        
        try:
            from glayout import movex, movey
            
            old_pos = (float(comp.xmin), float(comp.ymin))
            
            if dx != 0:
                movex(comp, dx)
            if dy != 0:
                movey(comp, dy)
            
            new_pos = (float(comp.xmin), float(comp.ymin))
            
            self.context.registry.update_metadata(component_name, {
                "position": new_pos
            })
            
            return {
                "success": True,
                "component_name": component_name,
                "moved_by": {"dx": dx, "dy": dy},
                "old_position": {"x": old_pos[0], "y": old_pos[1]},
                "new_position": {"x": new_pos[0], "y": new_pos[1]}
            }
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise PlacementError(
                f"移动组件失败: {e}",
                {"component": component_name, "delta": (dx, dy)}
            )
    
    def interdigitize(
        self,
        comp_a: str,
        comp_b: str,
        num_cols: int = 4,
        layout_style: str = "ABAB"
    ) -> Dict[str, Any]:
        """互指式放置两个晶体管
        
        用于改善匹配性的放置技术。
        
        Args:
            comp_a: 组件A名称
            comp_b: 组件B名称
            num_cols: 互指列数
            layout_style: 布局风格(ABAB/ABBA/common_centroid)
            
        Returns:
            放置结果
        """
        if layout_style not in ["ABAB", "ABBA", "common_centroid"]:
            raise ValidationError(
                f"无效的布局风格: {layout_style}",
                {"valid_styles": ["ABAB", "ABBA", "common_centroid"]}
            )
        
        if num_cols < 2:
            raise ValidationError(
                f"列数必须至少为2: {num_cols}",
                {"min_cols": 2}
            )
        
        # 获取组件
        info_a = self.context.get_component_info(comp_a)
        info_b = self.context.get_component_info(comp_b)
        
        if info_a is None:
            raise ValidationError(f"组件A不存在: {comp_a}")
        if info_b is None:
            raise ValidationError(f"组件B不存在: {comp_b}")
        
        if not self._glayout_available:
            # 模拟模式
            self.context.registry.update_metadata(comp_a, {
                "interdigitized_with": comp_b,
                "layout_style": layout_style,
                "num_cols": num_cols
            })
            self.context.registry.update_metadata(comp_b, {
                "interdigitized_with": comp_a,
                "layout_style": layout_style,
                "num_cols": num_cols
            })
            
            return {
                "success": True,
                "comp_a": comp_a,
                "comp_b": comp_b,
                "layout_style": layout_style,
                "num_cols": num_cols,
                "_mock": True
            }
        
        try:
            from glayout import two_nfet_interdigitized, two_pfet_interdigitized
            from glayout.placement.common_centroid_ab_ba import common_centroid_ab_ba
            
            pdk = self.context.pdk
            
            # 确定器件类型
            device_type_a = info_a.device_type
            device_type_b = info_b.device_type
            
            if device_type_a != device_type_b:
                raise PlacementError(
                    f"互指式放置需要相同类型的器件: {device_type_a} != {device_type_b}"
                )
            
            # 选择合适的放置函数
            if layout_style == "common_centroid":
                # TODO: 实现common_centroid
                raise NotImplementedError("common_centroid布局尚未实现")
            else:
                if device_type_a == "nmos":
                    place_func = two_nfet_interdigitized
                elif device_type_a == "pmos":
                    place_func = two_pfet_interdigitized
                else:
                    raise PlacementError(
                        f"不支持的器件类型: {device_type_a}"
                    )
            
            # 执行互指式放置
            # 注意：实际实现需要获取器件参数
            params_a = info_a.params
            
            result_comp = place_func(
                pdk=pdk,
                width=params_a.get("width", 1.0),
                length=params_a.get("length"),
                numcols=num_cols
            )
            
            # 更新元数据
            self.context.registry.update_metadata(comp_a, {
                "interdigitized_with": comp_b,
                "layout_style": layout_style,
                "num_cols": num_cols
            })
            self.context.registry.update_metadata(comp_b, {
                "interdigitized_with": comp_a,
                "layout_style": layout_style,
                "num_cols": num_cols
            })
            
            return {
                "success": True,
                "comp_a": comp_a,
                "comp_b": comp_b,
                "layout_style": layout_style,
                "num_cols": num_cols,
                "result_bbox": {
                    "width": float(result_comp.xsize),
                    "height": float(result_comp.ysize)
                }
            }
            
        except (ValidationError, PlacementError, NotImplementedError):
            raise
        except Exception as e:
            self.error_handler.handle_error(e)
            raise PlacementError(
                f"互指式放置失败: {e}",
                {"comp_a": comp_a, "comp_b": comp_b}
            )


# ============== MCP工具定义 ==============

def get_placement_tools() -> List[Dict[str, Any]]:
    """获取放置工具定义列表
    
    Returns:
        MCP工具定义列表
    """
    return [
        {
            "name": "place_component",
            "description": "放置组件到指定位置，可以设置旋转和镜像",
            "inputSchema": PLACE_COMPONENT_SCHEMA,
            "category": "placement"
        },
        {
            "name": "align_to_port",
            "description": "将组件对齐到指定端口，支持多种对齐方式和偏移",
            "inputSchema": ALIGN_TO_PORT_SCHEMA,
            "category": "placement"
        },
        {
            "name": "move_component",
            "description": "相对移动组件，指定X和Y方向的位移",
            "inputSchema": MOVE_COMPONENT_SCHEMA,
            "category": "placement"
        },
        {
            "name": "interdigitize",
            "description": "互指式放置两个晶体管，用于改善匹配性（如差分对、电流镜）",
            "inputSchema": INTERDIGITIZE_SCHEMA,
            "category": "placement"
        }
    ]
