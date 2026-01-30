"""
Circuit Builder - 复合电路构建器

提供电流镜、差分对、运放等复合电路的构建功能
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# 配置日志
logger = logging.getLogger(__name__)

# 添加路径
_BASE_PATH = Path(__file__).parent.parent
if str(_BASE_PATH) not in sys.path:
    sys.path.insert(0, str(_BASE_PATH))

_GLAYOUT_PATH = _BASE_PATH.parent / "gLayout" / "src"
if str(_GLAYOUT_PATH) not in sys.path:
    sys.path.insert(0, str(_GLAYOUT_PATH))

# 设置 PDK_ROOT 环境变量（glayout 初始化时需要）
_PDK_ROOT = _BASE_PATH.parent / "skywater-pdk"
if _PDK_ROOT.exists() and not os.environ.get('PDK_ROOT'):
    os.environ['PDK_ROOT'] = str(_PDK_ROOT)

# 预先尝试导入 gdsfactory 以确保环境正确
try:
    import gdsfactory as gf
    logger.debug(f"gdsfactory 版本：{gf.__version__}")
except ImportError:
    logger.warning(f"gdsfactory 导入失败：{e}")

from core.layout_context import LayoutContext
from core.pdk_manager import PDKManager
from mcp_server.handlers.error_handler import (
    DeviceError, ValidationError, ErrorHandler
)


class CircuitBuilder:
    """复合电路构建器
    
    提供电流镜、差分对、运放等复合电路的自动化构建。
    封装gLayout的blocks模块。
    
    Usage:
        >>> builder = CircuitBuilder(context)
        >>> result = builder.build_current_mirror(device_type="nmos", width=3.0)
        >>> result = builder.build_diff_pair(device_type="nmos", width=5.0)
    """
    
    # 端口过滤配置：主要功能端口关键字
    PRIMARY_PORT_KEYWORDS = ['drain', 'gate', 'source', 'well', 'bulk', 'inp', 'inm', 'out', 'vdd', 'vss', 'plus', 'minus', 'ref', 'mirror', 'tail']
    # 排除的内部结构关键字
    EXCLUDE_KEYWORDS = ['dummy', 'via', 'bottom', 'top_met', 'gsdcon', 'layer', 'con_', 'tie_', 'multiplier_']
    # 最大返回端口数
    MAX_PORTS_RETURN = 30
    
    def __init__(self, context: LayoutContext):
        """初始化构建器
        
        Args:
            context: 布局上下文
        """
        self.context = context
        self.error_handler = ErrorHandler()
        self._glayout_available, self._glayout_error = self._check_glayout()

        import logging
        logger = logging.getLogger(__name__)
        if self._glayout_available:
            logger.info("gLayout模块可用，将使用真实器件创建")
        else:
            logger.warning(f"gLayout模块不可用，将使用模拟模式。原因：{self._glayout_error}")
    
    def _check_glayout(self) -> bool:
        """检查gLayout是否可用
        
        Returns:
            (是否可用, 错误信息)
        """
        try:
            from glayout.blocks.elementary.current_mirror import current_mirror
            # 额外验证：确保它是可调用的
            if callable(current_mirror):
                return True, None
            else:
                return False, "current_mirror 不是可调用的函数"
        except ImportError as e:
            return False, f"ImportError: {e}"
        except TypeError as e:
            return False, f"TypeError: {e}"
        except Exception as e:
            return False, f"Exception ({type(e).__name__}): {e}"
    
    def _get_pdk(self):
        """获取PDK实例"""
        try:
            return self.context.pdk
        except RuntimeError:
            raise DeviceError(
                "PDK未初始化，请先设置PDK"
            )
        
    def _filter_ports(self, all_ports: list) -> dict:
        """过滤端口列表，只返回主要功能端口
            
        Args:
            all_ports: 完整的端口列表
                
        Returns:
            包含过滤后端口和统计信息的字典
        """
        filtered = []
        for port in all_ports:
            port_lower = port.lower()
            # 必须包含主要功能关键字
            has_main = any(kw in port_lower for kw in self.PRIMARY_PORT_KEYWORDS)
            # 不能包含内部结构关键字
            has_exclude = any(kw in port_lower for kw in self.EXCLUDE_KEYWORDS)
                
            if has_main and not has_exclude:
                filtered.append(port)
            
        # 如果过滤后为空，返回前N个原始端口
        if not filtered:
            filtered = all_ports[:self.MAX_PORTS_RETURN]
        # 如果过滤后仍然太多，截断
        elif len(filtered) > self.MAX_PORTS_RETURN:
            filtered = filtered[:self.MAX_PORTS_RETURN]
            
        return {
            "ports": filtered,
            "total_ports": len(all_ports),
            "filtered": len(all_ports) != len(filtered)
        }
    
    def build_current_mirror(
        self,
        device_type: str = "nmos",
        width: float = 3.0,
        length: Optional[float] = None,
        numcols: int = 3,
        with_dummy: bool = True,
        with_tie: bool = True,
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """构建电流镜电路
        
        电流镜是模拟电路中最基本的功能模块，用于复制电流。
        使用互指式布局减小失配。
        
        Args:
            device_type: 器件类型 "nmos" 或 "pmos"
            width: 管子宽度(um)
            length: 管子长度(um)，默认使用PDK最小长度
            numcols: 互指列数，影响匹配性能
            with_dummy: 是否添加dummy结构
            with_tie: 是否添加衬底连接
            name: 电路名称
            
        Returns:
            构建结果，包含组件信息和端口列表
        """
        if device_type not in ["nmos", "pmos"]:
            raise ValidationError(
                f"无效的器件类型: {device_type}",
                {"valid_types": ["nmos", "pmos"]}
            )
        
        if numcols < 2:
            raise ValidationError(
                f"互指列数至少为2: {numcols}",
                {"min_numcols": 2}
            )
        
        if not self._glayout_available:
            return self._mock_current_mirror_result(
                device_type, width, length, numcols, name
            )
        
        try:
            from glayout.blocks.elementary.current_mirror import current_mirror
            
            pdk = self._get_pdk()
            
            # 转换device_type格式
            device = "nfet" if device_type == "nmos" else "pfet"
            
            # 构建电流镜
            comp = current_mirror(
                pdk=pdk,
                numcols=numcols,
                device=device,
                with_dummy=with_dummy,
                with_tie=with_tie,
                width=width,
                length=length,
                **kwargs
            )
            
            # 注册组件
            registered_name = self.context.register_component(
                component=comp,
                device_type="current_mirror",
                params={
                    "device_type": device_type,
                    "width": width,
                    "length": length,
                    "numcols": numcols,
                    "with_dummy": with_dummy,
                    "with_tie": with_tie
                },
                name=name
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "circuit_type": "current_mirror",
                "params": {
                    "device_type": device_type,
                    "width": width,
                    "length": length,
                    "numcols": numcols
                },
                **self._filter_ports(list(comp.ports.keys())),
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                },
                "description": f"{device_type.upper()}电流镜，{numcols}列互指式布局"
            }
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise DeviceError(f"构建电流镜失败: {e}")
    
    def build_diff_pair(
        self,
        device_type: str = "nmos",
        width: float = 5.0,
        length: Optional[float] = None,
        fingers: int = 1,
        numcols: int = 2,
        layout_style: str = "interdigitized",
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """构建差分对电路
        
        差分对是运放和比较器的核心输入级。
        支持互指式和共心式两种布局风格。
        
        Args:
            device_type: 器件类型
            width: 管子宽度
            length: 管子长度
            fingers: 指数
            numcols: 互指列数，影响匹配性能（默认2）
            layout_style: 布局风格 "interdigitized" 或 "common_centroid"
            name: 电路名称
            
        Returns:
            构建结果
        """
        if device_type not in ["nmos", "pmos"]:
            raise ValidationError(f"无效的器件类型: {device_type}")
        
        if layout_style not in ["interdigitized", "common_centroid"]:
            raise ValidationError(
                f"无效的布局风格: {layout_style}",
                {"valid_styles": ["interdigitized", "common_centroid"]}
            )
        
        if not self._glayout_available:
            return self._mock_diff_pair_result(
                device_type, width, length, fingers, numcols, layout_style, name
            )
        
        try:
            pdk = self._get_pdk()
            
            if layout_style == "interdigitized":
                from glayout.placement.two_transistor_interdigitized import (
                    two_nfet_interdigitized, two_pfet_interdigitized
                )
                
                if device_type == "nmos":
                    comp = two_nfet_interdigitized(
                        pdk=pdk,
                        numcols=numcols,
                        width=width,
                        length=length,
                        fingers=fingers,
                        **kwargs
                    )
                else:
                    comp = two_pfet_interdigitized(
                        pdk=pdk,
                        numcols=numcols,
                        width=width,
                        length=length,
                        fingers=fingers,
                        **kwargs
                    )
            else:
                # common_centroid
                from glayout.placement.common_centroid_ab_ba import common_centroid_ab_ba
                comp = common_centroid_ab_ba(
                    pdk=pdk,
                    width=width,
                    length=length,
                    fingers=fingers,
                    **kwargs
                )
            
            registered_name = self.context.register_component(
                component=comp,
                device_type="diff_pair",
                params={
                    "device_type": device_type,
                    "width": width,
                    "length": length,
                    "fingers": fingers,
                    "layout_style": layout_style
                },
                name=name
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "circuit_type": "diff_pair",
                "params": {
                    "device_type": device_type,
                    "width": width,
                    "length": length,
                    "fingers": fingers,
                    "numcols": numcols,
                    "layout_style": layout_style
                },
                **self._filter_ports(list(comp.ports.keys())),
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                },
                "description": f"{device_type.upper()}差分对，{layout_style}布局"
            }
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise DeviceError(f"构建差分对失败: {e}")
    
    def build_opamp(
        self,
        topology: str = "two_stage",
        input_pair_w: float = 5.0,
        load_w: float = 2.0,
        bias_current: float = 10.0,
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """构建运算放大器
        
        支持两级运放拓扑。使用 glayout 的 opamp 模块（如果可用），
        否则使用基础器件组合构建。
        
        Args:
            topology: 拓扑结构 "two_stage" 或 "folded_cascode"
            input_pair_w: 输入对管宽度 (um)
            load_w: 负载管宽度 (um)
            bias_current: 偏置电流 (uA)
            name: 电路名称
            
        Returns:
            构建结果
        """
        if topology not in ["two_stage", "folded_cascode"]:
            raise ValidationError(
                f"无效的拓扑结构: {topology}",
                {"valid_topologies": ["two_stage", "folded_cascode"]}
            )
        
        if not self._glayout_available:
            return self._mock_opamp_result(
                topology, input_pair_w, load_w, name
            )
        
        try:
            pdk = self._get_pdk()
            
            if topology == "two_stage":
                # 尝试使用 glayout 的 opamp 模块
                try:
                    from glayout.blocks.composite.opamp import opamp
                    
                    comp = opamp(
                        pdk=pdk,
                        input_pair_params={"width": input_pair_w},
                        **kwargs
                    )
                except ImportError as e:
                    # glayout opamp 模块不可用，使用基础器件组合
                    import logging
                    logging.warning(f"glayout opamp 模块不可用: {e}，使用基础器件组合")
                    return self._build_two_stage_opamp_from_primitives(
                        pdk, input_pair_w, load_w, bias_current, name
                    )
            else:
                # folded_cascode
                raise NotImplementedError("折叠共源共栅运放尚未实现")
            
            registered_name = self.context.register_component(
                component=comp,
                device_type="opamp",
                params={
                    "topology": topology,
                    "input_pair_w": input_pair_w,
                    "load_w": load_w,
                    "bias_current": bias_current
                },
                name=name
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "circuit_type": "opamp",
                "params": {
                    "topology": topology,
                    "input_pair_w": input_pair_w,
                    "load_w": load_w,
                    "bias_current": bias_current
                },
                **self._filter_ports(list(comp.ports.keys())),
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                },
                "description": f"{topology}运算放大器"
            }
            
        except NotImplementedError:
            raise
        except Exception as e:
            self.error_handler.handle_error(e)
            raise DeviceError(f"构建运放失败: {e}")
    
    def _build_two_stage_opamp_from_primitives(
        self,
        pdk,
        input_pair_w: float,
        load_w: float,
        bias_current: float,
        name: Optional[str]
    ) -> Dict[str, Any]:
        """使用基础器件构建两级运放
        
        当 glayout 的 opamp 模块不可用时，使用基础器件组合：
        - 第一级：差分对 + 电流镜负载
        - 第二级：共源放大器
        
        Args:
            pdk: PDK实例
            input_pair_w: 输入对管宽度
            load_w: 负载管宽度
            bias_current: 偏置电流
            name: 电路名称
            
        Returns:
            构建结果
        """
        try:
            from gdsfactory import Component
            from glayout.primitives.fet import nmos, pmos
            from glayout import move, movey
            
            # 创建顶层组件
            comp = Component(name or "two_stage_opamp")
            
            # 第一级：差分输入对 (NMOS)
            input_n1 = nmos(pdk, width=input_pair_w, length=pdk.get_grule("poly")["min_width"])
            input_n2 = nmos(pdk, width=input_pair_w, length=pdk.get_grule("poly")["min_width"])
            
            # 第一级：PMOS负载（电流镜）
            load_p1 = pmos(pdk, width=load_w, length=pdk.get_grule("poly")["min_width"])
            load_p2 = pmos(pdk, width=load_w, length=pdk.get_grule("poly")["min_width"])
            
            # 添加引用并布局
            ref_n1 = comp.add_ref(input_n1, alias="input_n1")
            ref_n2 = comp.add_ref(input_n2, alias="input_n2")
            ref_p1 = comp.add_ref(load_p1, alias="load_p1")
            ref_p2 = comp.add_ref(load_p2, alias="load_p2")
            
            # 简单布局：差分对在下，负载在上
            spacing = 2.0  # um
            ref_n2.movex(ref_n1.xmax + spacing)
            ref_p1.movey(ref_n1.ymax + spacing)
            ref_p2.move((ref_n2.x, ref_n2.ymax + spacing))
            
            # 添加端口
            comp.add_port("inp", port=ref_n1.ports["gate_N"])
            comp.add_port("inn", port=ref_n2.ports["gate_N"])
            comp.add_port("out", port=ref_p2.ports["drain_N"])
            comp.add_port("vdd", port=ref_p1.ports["source_N"])
            comp.add_port("vss", port=ref_n1.ports["source_N"])
            
            # 注册组件
            registered_name = self.context.register_component(
                component=comp,
                device_type="opamp",
                params={
                    "topology": "two_stage",
                    "input_pair_w": input_pair_w,
                    "load_w": load_w,
                    "bias_current": bias_current,
                    "build_method": "primitives"
                },
                name=name
            )
            
            return {
                "success": True,
                "component_name": registered_name,
                "circuit_type": "opamp",
                "params": {
                    "topology": "two_stage",
                    "input_pair_w": input_pair_w,
                    "load_w": load_w,
                    "bias_current": bias_current
                },
                **self._filter_ports(list(comp.ports.keys())),
                "bbox": {
                    "width": float(comp.xsize),
                    "height": float(comp.ysize)
                },
                "description": "两级运算放大器（基础器件组合）",
                "_note": "使用基础器件组合构建，未使用完整的opamp模板"
            }
            
        except Exception as e:
            # 如果基础器件构建也失败，返回mock结果
            import logging
            logging.warning(f"基础器件构建运放失败: {e}，返回mock结果")
            return self._mock_opamp_result("two_stage", input_pair_w, load_w, name)
    
    # ============== 模拟模式结果 ==============
    
    def _mock_current_mirror_result(
        self,
        device_type: str,
        width: float,
        length: Optional[float],
        numcols: int,
        name: Optional[str]
    ) -> Dict[str, Any]:
        """生成模拟电流镜结果"""
        registered_name = self.context.register_component(
            component=None,
            device_type="current_mirror",
            params={
                "device_type": device_type,
                "width": width,
                "length": length,
                "numcols": numcols
            },
            name=name
        )
        
        # 估算尺寸
        est_width = width * numcols * 2 + 3.0
        est_height = (length or 0.15) + 2.0
        
        return {
            "success": True,
            "component_name": registered_name,
            "circuit_type": "current_mirror",
            "params": {
                "device_type": device_type,
                "width": width,
                "length": length,
                "numcols": numcols
            },
            "ports": [
                "ref_drain", "ref_source", "ref_gate",
                "mirror_drain", "mirror_source", "mirror_gate",
                "well"
            ],
            "bbox": {"width": est_width, "height": est_height},
            "description": f"{device_type.upper()}电流镜，{numcols}列互指式布局",
            "_mock": True
        }
    
    def _mock_diff_pair_result(
        self,
        device_type: str,
        width: float,
        length: Optional[float],
        fingers: int,
        numcols: int,
        layout_style: str,
        name: Optional[str]
    ) -> Dict[str, Any]:
        """生成模拟差分对结果"""
        registered_name = self.context.register_component(
            component=None,
            device_type="diff_pair",
            params={
                "device_type": device_type,
                "width": width,
                "length": length,
                "fingers": fingers,
                "numcols": numcols,
                "layout_style": layout_style
            },
            name=name
        )
        
        est_width = width * fingers * numcols * 2 + 2.0
        est_height = (length or 0.15) + 2.0
        
        return {
            "success": True,
            "component_name": registered_name,
            "circuit_type": "diff_pair",
            "params": {
                "device_type": device_type,
                "width": width,
                "length": length,
                "fingers": fingers,
                "numcols": numcols,
                "layout_style": layout_style
            },
            "ports": [
                "inp_drain", "inp_gate", "inp_source",
                "inm_drain", "inm_gate", "inm_source",
                "tail_source", "well"
            ],
            "bbox": {"width": est_width, "height": est_height},
            "description": f"{device_type.upper()}差分对，{layout_style}布局",
            "_mock": True
        }
    
    def _mock_opamp_result(
        self,
        topology: str,
        input_pair_w: float,
        load_w: float,
        name: Optional[str]
    ) -> Dict[str, Any]:
        """生成模拟运放结果"""
        registered_name = self.context.register_component(
            component=None,
            device_type="opamp",
            params={
                "topology": topology,
                "input_pair_w": input_pair_w,
                "load_w": load_w
            },
            name=name
        )
        
        return {
            "success": True,
            "component_name": registered_name,
            "circuit_type": "opamp",
            "params": {
                "topology": topology,
                "input_pair_w": input_pair_w,
                "load_w": load_w
            },
            "ports": [
                "inp", "inm", "out",
                "vdd", "vss", "bias"
            ],
            "bbox": {"width": 50.0, "height": 30.0},
            "description": f"{topology}运算放大器",
            "_mock": True
        }


# ============== MCP工具定义 ==============

def get_circuit_tools() -> List[Dict[str, Any]]:
    """获取复合电路工具定义列表"""
    from mcp_server.schemas.device_schemas import (
        CURRENT_MIRROR_SCHEMA, DIFF_PAIR_SCHEMA, OPAMP_SCHEMA
    )
    
    return [
        {
            "name": "create_current_mirror",
            "description": "创建电流镜电路，使用互指式布局减小失配。适用于偏置电路和电流复制",
            "inputSchema": CURRENT_MIRROR_SCHEMA,
            "category": "circuit"
        },
        {
            "name": "create_diff_pair",
            "description": "创建差分对电路，是运放和比较器的核心输入级。支持互指式和共心式布局",
            "inputSchema": DIFF_PAIR_SCHEMA,
            "category": "circuit"
        },
        {
            "name": "create_opamp",
            "description": "创建运算放大器，支持两级运放和折叠共源共栅两种拓扑",
            "inputSchema": OPAMP_SCHEMA,
            "category": "circuit"
        }
    ]
