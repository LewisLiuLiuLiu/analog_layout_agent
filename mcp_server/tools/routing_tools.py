"""
Routing Tools - 路由工具

封装gLayout的路由功能，提供MCP工具接口
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
from mcp_server.schemas.routing_schemas import (
    SMART_ROUTE_SCHEMA, C_ROUTE_SCHEMA, L_ROUTE_SCHEMA, STRAIGHT_ROUTE_SCHEMA
)
from mcp_server.handlers.error_handler import (
    ValidationError, RoutingError, ErrorHandler
)


class RoutingToolExecutor:
    """路由工具执行器
    
    封装gLayout的路由功能，提供统一的接口。
    支持smart_route、c_route、l_route、straight_route等路由策略。
    
    Usage:
        >>> executor = RoutingToolExecutor(context)
        >>> result = executor.smart_route("nmos_1.drain_E", "pmos_1.drain_E", layer="met2")
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
            from glayout import smart_route, c_route
            return True
        except ImportError:
            return False
    
    def _get_pdk(self):
        """获取PDK实例"""
        try:
            return self.context.pdk
        except RuntimeError:
            raise RoutingError(
                "PDK未初始化，请先设置PDK",
                {"suggestion": "调用 set_pdk('sky130') 设置PDK"}
            )
    
    def _resolve_port(self, port_ref: str):
        """解析端口引用
        
        Args:
            port_ref: 端口引用，格式 "component_name.port_name"
            
        Returns:
            端口对象
            
        Raises:
            ValidationError: 如果端口不存在
        """
        if '.' not in port_ref:
            raise ValidationError(
                f"无效的端口引用格式: {port_ref}",
                {"expected_format": "component_name.port_name"}
            )
        
        port = self.context.get_port(port_ref)
        if port is None:
            comp_name, port_name = port_ref.rsplit('.', 1)
            
            # 检查组件是否存在
            info = self.context.get_component_info(comp_name)
            if info is None:
                raise ValidationError(
                    f"组件不存在: {comp_name}",
                    {"available_components": self.context.list_components()[:10]}
                )
            
            # 检查端口是否存在
            raise ValidationError(
                f"端口不存在: {port_name}",
                {"component": comp_name, "available_ports": info.ports[:10]}
            )
        
        return port
    
    def smart_route(
        self,
        source_port: str,
        dest_port: str,
        layer: str = "met2",
        width: Optional[float] = None,
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """智能路由
        
        自动选择最优的路由策略（straight/c/l）。
        
        Args:
            source_port: 源端口(格式: component_name.port_name)
            dest_port: 目标端口
            layer: 布线层
            width: 布线宽度，默认使用端口宽度
            name: 路由组件名称
            
        Returns:
            路由结果
        """
        if not self._glayout_available:
            return self._mock_route_result(
                "smart_route", source_port, dest_port, layer, name
            )
        
        try:
            from glayout import smart_route
            
            pdk = self._get_pdk()
            
            # 解析端口
            port1 = self._resolve_port(source_port)
            port2 = self._resolve_port(dest_port)
            
            # 执行路由
            route_kwargs = {
                "pdk": pdk,
                "edge1": port1,
                "edge2": port2,
                "glayer": layer
            }
            
            if width:
                route_kwargs["width"] = width
            
            comp = smart_route(**route_kwargs, **kwargs)
            
            # 注册路由组件
            registered_name = self.context.register_component(
                component=comp,
                device_type="route",
                params={
                    "source_port": source_port,
                    "dest_port": dest_port,
                    "layer": layer,
                    "route_type": "smart"
                },
                name=name,
                prefix="route"
            )
            
            # 记录连接
            self.context.add_connection(
                source=source_port,
                target=dest_port,
                layer=layer,
                route_type="smart",
                route_component=registered_name,
                width=width
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "route_type": "smart",
                "source": source_port,
                "target": dest_port,
                "layer": layer,
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                }
            }
            
        except (ValidationError, RoutingError):
            raise
        except Exception as e:
            self.error_handler.handle_error(e)
            raise RoutingError(
                f"智能路由失败: {e}",
                {"source": source_port, "dest": dest_port, "layer": layer}
            )
    
    def c_route(
        self,
        source_port: str,
        dest_port: str,
        extension: Optional[float] = None,
        cglayer: str = "met2",
        cwidth: Optional[float] = None,
        width1: Optional[float] = None,
        width2: Optional[float] = None,
        e1glayer: Optional[str] = None,
        e2glayer: Optional[str] = None,
        viaoffset: Optional[bool] = True,
        fullbottom: Optional[bool] = False,
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """C型路由
        
        适用于同向平行端口的连接。
        
        Args:
            source_port: 源端口
            dest_port: 目标端口
            extension: 延伸长度，默认自动计算
            cglayer: 连接层金属层
            cwidth: 连接线宽度
            width1: 源端宽度
            width2: 目标端宽度
            e1glayer: 源端金属层
            e2glayer: 目标端金属层
            viaoffset: 过孔偏移
            fullbottom: 过孔底部填满
            name: 路由组件名称
            
        Returns:
            路由结果
        """
        if not self._glayout_available:
            return self._mock_route_result(
                "c_route", source_port, dest_port, cglayer, name
            )
        
        try:
            from glayout import c_route
            
            pdk = self._get_pdk()
            
            port1 = self._resolve_port(source_port)
            port2 = self._resolve_port(dest_port)
            
            # 计算默认延伸长度
            if extension is None:
                extension = 3 * pdk.util_max_metal_seperation()
            
            # 直接传递到底层函数的参数
            route_kwargs = {
                "pdk": pdk,
                "edge1": port1,
                "edge2": port2,
                "extension": extension,
                "cglayer": cglayer,
                "viaoffset": viaoffset,
                "fullbottom": fullbottom
            }
            
            # 可选参数
            if cwidth is not None:
                route_kwargs["cwidth"] = cwidth
            if width1 is not None:
                route_kwargs["width1"] = width1
            if width2 is not None:
                route_kwargs["width2"] = width2
            if e1glayer is not None:
                route_kwargs["e1glayer"] = e1glayer
            if e2glayer is not None:
                route_kwargs["e2glayer"] = e2glayer
            
            comp = c_route(**route_kwargs)
            
            registered_name = self.context.register_component(
                component=comp,
                device_type="route",
                params={
                    "source_port": source_port,
                    "dest_port": dest_port,
                    "layer": cglayer,
                    "route_type": "c",
                    "extension": extension
                },
                name=name,
                prefix="route"
            )
            
            self.context.add_connection(
                source=source_port,
                target=dest_port,
                layer=cglayer,
                route_type="c",
                route_component=registered_name,
                width=cwidth
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "route_type": "c",
                "source": source_port,
                "target": dest_port,
                "layer": cglayer,
                "extension": extension,
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                }
            }
            
        except (ValidationError, RoutingError):
            raise
        except Exception as e:
            self.error_handler.handle_error(e)
            raise RoutingError(
                f"C型路由失败: {e}",
                {"source": source_port, "dest": dest_port}
            )
    
    def l_route(
        self,
        source_port: str,
        dest_port: str,
        hglayer: str = "met2",
        vglayer: str = "met2",
        hwidth: Optional[float] = None,
        vwidth: Optional[float] = None,
        viaoffset: Optional[bool] = True,
        fullbottom: Optional[bool] = True,
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """L型路由
        
        适用于垂直端口的连接。
        
        Args:
            source_port: 源端口
            dest_port: 目标端口
            hglayer: 水平连接金属层
            vglayer: 垂直连接金属层
            hwidth: 水平线宽度
            vwidth: 垂直线宽度
            viaoffset: 过孔偏移
            fullbottom: 过孔底部填满
            name: 路由组件名称
            
        Returns:
            路由结果
        """
        if not self._glayout_available:
            return self._mock_route_result(
                "l_route", source_port, dest_port, hglayer, name
            )
        
        try:
            from glayout import L_route
            
            pdk = self._get_pdk()
            
            port1 = self._resolve_port(source_port)
            port2 = self._resolve_port(dest_port)
            
            # 直接传递到底层函数的参数
            route_kwargs = {
                "pdk": pdk,
                "edge1": port1,
                "edge2": port2,
                "hglayer": hglayer,
                "vglayer": vglayer,
                "viaoffset": viaoffset,
                "fullbottom": fullbottom
            }
            
            # 可选参数
            if hwidth is not None:
                route_kwargs["hwidth"] = hwidth
            if vwidth is not None:
                route_kwargs["vwidth"] = vwidth
            
            comp = L_route(**route_kwargs)
            
            # 使用 hglayer 作为主要 layer 记录
            layer = hglayer
            
            registered_name = self.context.register_component(
                component=comp,
                device_type="route",
                params={
                    "source_port": source_port,
                    "dest_port": dest_port,
                    "layer": layer,
                    "route_type": "l"
                },
                name=name,
                prefix="route"
            )
            
            self.context.add_connection(
                source=source_port,
                target=dest_port,
                layer=layer,
                route_type="l",
                route_component=registered_name,
                width=hwidth
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "route_type": "l",
                "source": source_port,
                "target": dest_port,
                "layer": layer,
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                }
            }
            
        except (ValidationError, RoutingError):
            raise
        except Exception as e:
            self.error_handler.handle_error(e)
            raise RoutingError(
                f"L型路由失败: {e}",
                {"source": source_port, "dest": dest_port}
            )
    
    def straight_route(
        self,
        source_port: str,
        dest_port: str,
        glayer1: Optional[str] = None,
        glayer2: Optional[str] = None,
        width: Optional[float] = None,
        fullbottom: Optional[bool] = False,
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """直线路由
        
        适用于共线端口的连接。
        
        Args:
            source_port: 源端口
            dest_port: 目标端口
            glayer1: 起始层金属层，默认使用端口层
            glayer2: 结束层金属层，默认使用端口层
            width: 布线宽度
            fullbottom: 过孔底部填满
            name: 路由组件名称
            
        Returns:
            路由结果
        """
        # 用于记录的 layer，默认 met2
        layer = glayer1 if glayer1 else "met2"
        
        if not self._glayout_available:
            return self._mock_route_result(
                "straight_route", source_port, dest_port, layer, name
            )
        
        try:
            from glayout import straight_route
            
            pdk = self._get_pdk()
            
            port1 = self._resolve_port(source_port)
            port2 = self._resolve_port(dest_port)
            
            # 直接传递到底层函数的参数
            route_kwargs = {
                "pdk": pdk,
                "edge1": port1,
                "edge2": port2,
                "fullbottom": fullbottom
            }
            
            # 可选参数
            if width is not None:
                route_kwargs["width"] = width
            if glayer1 is not None:
                route_kwargs["glayer1"] = glayer1
            if glayer2 is not None:
                route_kwargs["glayer2"] = glayer2
            
            comp = straight_route(**route_kwargs)
            
            registered_name = self.context.register_component(
                component=comp,
                device_type="route",
                params={
                    "source_port": source_port,
                    "dest_port": dest_port,
                    "layer": layer,
                    "route_type": "straight"
                },
                name=name,
                prefix="route"
            )
            
            self.context.add_connection(
                source=source_port,
                target=dest_port,
                layer=layer,
                route_type="straight",
                route_component=registered_name,
                width=width
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "route_type": "straight",
                "source": source_port,
                "target": dest_port,
                "layer": layer,
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                }
            }
            
        except (ValidationError, RoutingError):
            raise
        except Exception as e:
            self.error_handler.handle_error(e)
            raise RoutingError(
                f"直线路由失败: {e}",
                {"source": source_port, "dest": dest_port}
            )
    
    # ============== 模拟模式结果 ==============
    
    def _mock_route_result(
        self,
        route_type: str,
        source_port: str,
        dest_port: str,
        layer: str,
        name: Optional[str]
    ) -> Dict[str, Any]:
        """生成模拟路由结果"""
        registered_name = self.context.register_component(
            component=None,
            device_type="route",
            params={
                "source_port": source_port,
                "dest_port": dest_port,
                "layer": layer,
                "route_type": route_type.replace("_route", "")
            },
            name=name,
            prefix="route"
        )
        
        # 记录连接
        # 在mock模式下，跳过端口验证
        try:
            self.context.add_connection(
                source=source_port,
                target=dest_port,
                layer=layer,
                route_type=route_type.replace("_route", ""),
                route_component=registered_name
            )
        except ValueError:
            pass  # 模拟模式下忽略端口不存在错误
        
        return {
            "success": True,
            "component_name": registered_name,
            "route_type": route_type.replace("_route", ""),
            "source": source_port,
            "target": dest_port,
            "layer": layer,
            "bbox": {"width": 1.0, "height": 1.0},
            "_mock": True
        }


# ============== MCP工具定义 ==============

def get_routing_tools() -> List[Dict[str, Any]]:
    """获取路由工具定义列表
    
    Returns:
        MCP工具定义列表
    """
    return [
        {
            "name": "smart_route",
            "description": "智能路由，自动选择最优路由策略（straight/c/l）连接两个端口",
            "inputSchema": SMART_ROUTE_SCHEMA,
            "category": "routing"
        },
        {
            "name": "c_route",
            "description": "C型路由，适用于同向平行端口的连接（如两个朝右的端口）",
            "inputSchema": C_ROUTE_SCHEMA,
            "category": "routing"
        },
        {
            "name": "l_route",
            "description": "L型路由，适用于垂直端口的连接（如一个朝上一个朝右）",
            "inputSchema": L_ROUTE_SCHEMA,
            "category": "routing"
        },
        {
            "name": "straight_route",
            "description": "直线路由，适用于共线端口的直接连接",
            "inputSchema": STRAIGHT_ROUTE_SCHEMA,
            "category": "routing"
        }
    ]
