"""
Device Tools - 器件创建工具

封装gLayout的器件创建API，提供MCP工具接口
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
from core.pdk_manager import PDKManager
from mcp_server.schemas.device_schemas import (
    NMOS_SCHEMA, PMOS_SCHEMA, VIA_STACK_SCHEMA, VIA_ARRAY_SCHEMA,
    MIMCAP_SCHEMA, RESISTOR_SCHEMA, TAPRING_SCHEMA,
    CURRENT_MIRROR_SCHEMA, DIFF_PAIR_SCHEMA
)
from mcp_server.handlers.error_handler import (
    ValidationError, DeviceError, ErrorHandler
)


class DeviceToolExecutor:
    """器件工具执行器
    
    封装gLayout的器件创建功能，提供统一的接口。
    
    Usage:
        >>> executor = DeviceToolExecutor(context)
        >>> result = executor.create_nmos(width=1.0, fingers=2)
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
            from glayout import nmos, pmos
            return True
        except ImportError:
            return False
    
    def _get_pdk(self):
        """获取PDK实例"""
        try:
            return self.context.pdk
        except RuntimeError:
            raise DeviceError(
                "PDK未初始化，请先设置PDK",
                {"suggestion": "调用 set_pdk('sky130') 设置PDK"}
            )
    
    def _validate_mosfet_params(
        self,
        width: float,
        length: Optional[float],
        device_type: str = "nfet"
    ) -> Dict[str, Any]:
        """验证MOSFET参数
        
        Args:
            width: 沟道宽度
            length: 沟道长度
            device_type: 器件类型
            
        Returns:
            验证结果
        """
        result = PDKManager.validate_device_params(
            device_type=device_type,
            width=width,
            length=length,
            pdk_name=self.context.pdk_name
        )
        
        if not result["valid"]:
            raise ValidationError(
                f"参数验证失败: {'; '.join(result['errors'])}",
                {"validation_result": result}
            )
        
        return result
    
    def create_nmos(
        self,
        width: float,
        length: Optional[float] = None,
        fingers: int = 1,
        multiplier: int = 1,
        with_dummy: bool = True,
        with_tie: bool = True,
        sd_route_topmet: str = "met2",
        gate_route_topmet: str = "met2",
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """创建NMOS晶体管
        
        Args:
            width: 沟道宽度(um)
            length: 沟道长度(um)，默认使用PDK最小长度
            fingers: 指数
            multiplier: 并联倍数
            with_dummy: 是否添加dummy
            with_tie: 是否添加衬底连接
            sd_route_topmet: 源漏路由的顶层金属
            gate_route_topmet: 栅极路由的顶层金属
            name: 组件名称
            
        Returns:
            创建结果字典
        """
        # 验证参数
        validation = self._validate_mosfet_params(width, length, "nfet")
        
        # 使用验证后的参数
        width = validation["adjusted_params"]["width"]
        length = validation["adjusted_params"]["length"]
        
        if not self._glayout_available:
            # 模拟模式：返回模拟结果
            return self._mock_mosfet_result("nmos", width, length, fingers, multiplier, name)
        
        try:
            from glayout import nmos
            
            pdk = self._get_pdk()
            
            # 创建NMOS
            comp = nmos(
                pdk=pdk,
                width=width,
                length=length,
                fingers=fingers,
                multipliers=multiplier,
                with_dummy=with_dummy,
                with_tie=with_tie,
                sd_route_topmet=sd_route_topmet,
                gate_route_topmet=gate_route_topmet,
                **kwargs
            )
            
            # 注册组件
            registered_name = self.context.register_component(
                component=comp,
                device_type="nmos",
                params={
                    "width": width,
                    "length": length,
                    "fingers": fingers,
                    "multiplier": multiplier,
                    "with_dummy": with_dummy,
                    "with_tie": with_tie
                },
                name=name
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "device_type": "nmos",
                "params": {
                    "width": width,
                    "length": length,
                    "fingers": fingers,
                    "multiplier": multiplier
                },
                "ports": list(comp.ports.keys()),
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                }
            }
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise DeviceError(f"创建NMOS失败: {e}", {"params": {"width": width, "length": length}})
    
    def create_pmos(
        self,
        width: float,
        length: Optional[float] = None,
        fingers: int = 1,
        multiplier: int = 1,
        with_dummy: bool = True,
        with_tie: bool = True,
        sd_route_topmet: str = "met2",
        gate_route_topmet: str = "met2",
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """创建PMOS晶体管
        
        参数同create_nmos
        """
        # 验证参数
        validation = self._validate_mosfet_params(width, length, "pfet")
        width = validation["adjusted_params"]["width"]
        length = validation["adjusted_params"]["length"]
        
        if not self._glayout_available:
            return self._mock_mosfet_result("pmos", width, length, fingers, multiplier, name)
        
        try:
            from glayout import pmos
            
            pdk = self._get_pdk()
            
            comp = pmos(
                pdk=pdk,
                width=width,
                length=length,
                fingers=fingers,
                multipliers=multiplier,
                with_dummy=with_dummy,
                with_tie=with_tie,
                sd_route_topmet=sd_route_topmet,
                gate_route_topmet=gate_route_topmet,
                **kwargs
            )
            
            registered_name = self.context.register_component(
                component=comp,
                device_type="pmos",
                params={
                    "width": width,
                    "length": length,
                    "fingers": fingers,
                    "multiplier": multiplier,
                    "with_dummy": with_dummy,
                    "with_tie": with_tie
                },
                name=name
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "device_type": "pmos",
                "params": {
                    "width": width,
                    "length": length,
                    "fingers": fingers,
                    "multiplier": multiplier
                },
                "ports": list(comp.ports.keys()),
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                }
            }
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise DeviceError(f"创建PMOS失败: {e}")
    
    def create_via_stack(
        self,
        from_layer: str,
        to_layer: str,
        size: Optional[List[float]] = None,
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """创建层间Via堆叠
        
        Args:
            from_layer: 起始层
            to_layer: 目标层
            size: Via尺寸[宽,高]
            name: 组件名称
            
        Returns:
            创建结果
        """
        if not self._glayout_available:
            return self._mock_via_result("via_stack", from_layer, to_layer, name)
        
        try:
            from glayout import via_stack
            
            pdk = self._get_pdk()
            
            comp_kwargs = {
                "pdk": pdk,
                "glayer1": from_layer,
                "glayer2": to_layer
            }
            
            if size:
                comp_kwargs["size"] = tuple(size)
            
            comp = via_stack(**comp_kwargs, **kwargs)
            
            registered_name = self.context.register_component(
                component=comp,
                device_type="via_stack",
                params={"from_layer": from_layer, "to_layer": to_layer, "size": size},
                name=name
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "device_type": "via_stack",
                "params": {
                    "from_layer": from_layer,
                    "to_layer": to_layer,
                    "size": size
                },
                "ports": list(comp.ports.keys()),
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                }
            }
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise DeviceError(f"创建Via Stack失败: {e}")
    
    def create_via_array(
        self,
        from_layer: str,
        to_layer: str,
        num_vias: Optional[List[int]] = None,
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """创建Via阵列
        
        Args:
            from_layer: 起始层
            to_layer: 目标层
            num_vias: Via数量[行,列]
            name: 组件名称
            
        Returns:
            创建结果
        """
        if not self._glayout_available:
            return self._mock_via_result("via_array", from_layer, to_layer, name)
        
        try:
            from glayout import via_array
            
            pdk = self._get_pdk()
            
            comp_kwargs = {
                "pdk": pdk,
                "glayer1": from_layer,
                "glayer2": to_layer
            }
            
            if num_vias:
                comp_kwargs["num_vias"] = tuple(num_vias)
            
            comp = via_array(**comp_kwargs, **kwargs)
            
            registered_name = self.context.register_component(
                component=comp,
                device_type="via_array",
                params={"from_layer": from_layer, "to_layer": to_layer, "num_vias": num_vias},
                name=name
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "device_type": "via_array",
                "params": {
                    "from_layer": from_layer,
                    "to_layer": to_layer,
                    "num_vias": num_vias
                },
                "ports": list(comp.ports.keys()),
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                }
            }
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise DeviceError(f"创建Via Array失败: {e}")
    
    def create_mimcap(
        self,
        width: float,
        length: float,
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """创建MIM电容
        
        Args:
            width: 电容宽度(um)
            length: 电容长度(um)
            name: 组件名称
            
        Returns:
            创建结果
        """
        if width < 0.5 or length < 0.5:
            raise ValidationError(
                f"MIM电容尺寸过小: width={width}, length={length}",
                {"min_width": 0.5, "min_length": 0.5}
            )
        
        if not self._glayout_available:
            return self._mock_passive_result("mimcap", width, length, name)
        
        try:
            from glayout import mimcap
            
            pdk = self._get_pdk()
            
            comp = mimcap(
                pdk=pdk,
                width=width,
                length=length,
                **kwargs
            )
            
            registered_name = self.context.register_component(
                component=comp,
                device_type="mimcap",
                params={"width": width, "length": length},
                name=name
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "device_type": "mimcap",
                "params": {"width": width, "length": length},
                "ports": list(comp.ports.keys()),
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                }
            }
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise DeviceError(f"创建MIM电容失败: {e}")
    
    def create_resistor(
        self,
        width: float,
        length: float,
        num_series: int = 1,
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """创建多晶硅电阻
        
        Args:
            width: 电阻宽度(um)
            length: 电阻长度(um)
            num_series: 串联段数
            name: 组件名称
            
        Returns:
            创建结果
        """
        if width < 0.3 or length < 0.5:
            raise ValidationError(
                f"电阻尺寸过小: width={width}, length={length}",
                {"min_width": 0.3, "min_length": 0.5}
            )
        
        if not self._glayout_available:
            return self._mock_passive_result("resistor", width, length, name)
        
        try:
            from glayout import resistor
            
            pdk = self._get_pdk()
            
            comp = resistor(
                pdk=pdk,
                width=width,
                length=length,
                **kwargs
            )
            
            registered_name = self.context.register_component(
                component=comp,
                device_type="resistor",
                params={"width": width, "length": length, "num_series": num_series},
                name=name
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "device_type": "resistor",
                "params": {"width": width, "length": length, "num_series": num_series},
                "ports": list(comp.ports.keys()),
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                }
            }
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise DeviceError(f"创建电阻失败: {e}")
    
    def create_tapring(
        self,
        enclosed_rectangle: List[float],
        sdlayer: str = "p+s/d",
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """创建Tap环
        
        Args:
            enclosed_rectangle: 包围区域尺寸[宽,高]
            sdlayer: source/drain层类型
            name: 组件名称
            
        Returns:
            创建结果
        """
        if not self._glayout_available:
            return self._mock_tapring_result(enclosed_rectangle, sdlayer, name)
        
        try:
            from glayout import tapring
            
            pdk = self._get_pdk()
            
            comp = tapring(
                pdk=pdk,
                enclosed_rectangle=tuple(enclosed_rectangle),
                sdlayer=sdlayer,
                **kwargs
            )
            
            registered_name = self.context.register_component(
                component=comp,
                device_type="tapring",
                params={"enclosed_rectangle": enclosed_rectangle, "sdlayer": sdlayer},
                name=name
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "device_type": "tapring",
                "params": {"enclosed_rectangle": enclosed_rectangle, "sdlayer": sdlayer},
                "ports": list(comp.ports.keys()),
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                }
            }
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise DeviceError(f"创建Tap环失败: {e}")
    
    # ============== 模拟模式结果（用于测试）==============
    
    def _mock_mosfet_result(
        self,
        device_type: str,
        width: float,
        length: float,
        fingers: int,
        multiplier: int,
        name: Optional[str]
    ) -> Dict[str, Any]:
        """生成模拟MOSFET结果"""
        registered_name = self.context.register_component(
            component=None,
            device_type=device_type,
            params={
                "width": width,
                "length": length,
                "fingers": fingers,
                "multiplier": multiplier
            },
            name=name
        )
        
        # 估算尺寸
        est_width = width * fingers * multiplier + 2.0  # 包括dummy和tie
        est_height = length + 1.0
        
        return {
            "success": True,
            "component_name": registered_name,
            "device_type": device_type,
            "params": {
                "width": width,
                "length": length,
                "fingers": fingers,
                "multiplier": multiplier
            },
            "ports": ["drain_E", "gate_W", "source_E", "well_W"],
            "bbox": {
                "width": est_width,
                "height": est_height
            },
            "_mock": True
        }
    
    def _mock_via_result(
        self,
        device_type: str,
        from_layer: str,
        to_layer: str,
        name: Optional[str]
    ) -> Dict[str, Any]:
        """生成模拟Via结果"""
        registered_name = self.context.register_component(
            component=None,
            device_type=device_type,
            params={"from_layer": from_layer, "to_layer": to_layer},
            name=name
        )
        
        return {
            "success": True,
            "component_name": registered_name,
            "device_type": device_type,
            "params": {"from_layer": from_layer, "to_layer": to_layer},
            "ports": [f"{from_layer}_port", f"{to_layer}_port"],
            "bbox": {"width": 0.3, "height": 0.3},
            "_mock": True
        }
    
    def _mock_passive_result(
        self,
        device_type: str,
        width: float,
        length: float,
        name: Optional[str]
    ) -> Dict[str, Any]:
        """生成模拟无源器件结果"""
        registered_name = self.context.register_component(
            component=None,
            device_type=device_type,
            params={"width": width, "length": length},
            name=name
        )
        
        return {
            "success": True,
            "component_name": registered_name,
            "device_type": device_type,
            "params": {"width": width, "length": length},
            "ports": ["plus", "minus"],
            "bbox": {"width": width + 0.5, "height": length + 0.5},
            "_mock": True
        }
    
    def _mock_tapring_result(
        self,
        enclosed_rectangle: List[float],
        sdlayer: str,
        name: Optional[str]
    ) -> Dict[str, Any]:
        """生成模拟Tapring结果"""
        registered_name = self.context.register_component(
            component=None,
            device_type="tapring",
            params={"enclosed_rectangle": enclosed_rectangle, "sdlayer": sdlayer},
            name=name
        )
        
        return {
            "success": True,
            "component_name": registered_name,
            "device_type": "tapring",
            "params": {"enclosed_rectangle": enclosed_rectangle, "sdlayer": sdlayer},
            "ports": ["tap_N", "tap_S", "tap_E", "tap_W"],
            "bbox": {
                "width": enclosed_rectangle[0] + 1.0,
                "height": enclosed_rectangle[1] + 1.0
            },
            "_mock": True
        }


# ============== MCP工具定义 ==============

def get_device_tools() -> List[Dict[str, Any]]:
    """获取器件工具定义列表
    
    Returns:
        MCP工具定义列表
    """
    return [
        {
            "name": "create_nmos",
            "description": "创建NMOS晶体管，支持多指(fingers)和多倍数(multiplier)结构",
            "inputSchema": NMOS_SCHEMA,
            "category": "device"
        },
        {
            "name": "create_pmos",
            "description": "创建PMOS晶体管，支持多指(fingers)和多倍数(multiplier)结构",
            "inputSchema": PMOS_SCHEMA,
            "category": "device"
        },
        {
            "name": "create_via_stack",
            "description": "创建层间Via堆叠，用于连接不同金属层",
            "inputSchema": VIA_STACK_SCHEMA,
            "category": "device"
        },
        {
            "name": "create_via_array",
            "description": "创建Via阵列，用于大电流连接",
            "inputSchema": VIA_ARRAY_SCHEMA,
            "category": "device"
        },
        {
            "name": "create_mimcap",
            "description": "创建MIM(Metal-Insulator-Metal)电容",
            "inputSchema": MIMCAP_SCHEMA,
            "category": "device"
        },
        {
            "name": "create_resistor",
            "description": "创建多晶硅电阻",
            "inputSchema": RESISTOR_SCHEMA,
            "category": "device"
        },
        {
            "name": "create_tapring",
            "description": "创建Tap环(保护环)，用于衬底连接和隔离",
            "inputSchema": TAPRING_SCHEMA,
            "category": "device"
        }
    ]
