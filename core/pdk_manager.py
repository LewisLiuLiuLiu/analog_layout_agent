"""
PDK Manager - 统一PDK管理器

支持sky130、gf180、ihp130三种开源PDK的统一管理
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Optional, Dict, Any, List, ClassVar

# 添加gLayout路径到sys.path
_GLAYOUT_PATH = Path(__file__).parent.parent.parent / "gLayout" / "src"
if str(_GLAYOUT_PATH) not in sys.path:
    sys.path.insert(0, str(_GLAYOUT_PATH))

# 自动设置 PDK_ROOT 环境变量（glayout 初始化时需要）
_PDK_ROOT = Path(__file__).parent.parent.parent / "skywater-pdk"
if _PDK_ROOT.exists() and not os.environ.get('PDK_ROOT'):
    os.environ['PDK_ROOT'] = str(_PDK_ROOT)


class PDKManager:
    """统一PDK管理器
    
    负责加载、切换和管理不同的PDK（Process Design Kit）。
    支持sky130、gf180、ihp130三种开源工艺。
    
    Usage:
        >>> pdk = PDKManager.load_pdk("sky130")
        >>> rules = PDKManager.get_design_rules()
        >>> layers = PDKManager.get_layer_mapping()
    """
    
    # PDK配置信息
    PDK_CONFIG: ClassVar[Dict[str, Dict[str, Any]]] = {
        "sky130": {
            "module": "glayout.pdk.sky130_mapped.sky130_mapped",
            "pdk_var": "sky130_mapped_pdk",
            "pdk_root_env": "PDK_ROOT",
            "tech_node": "130nm",
            "metal_layers": 5,
            "min_dimensions": {
                "nfet_w": 0.15,
                "nfet_l": 0.15,
                "pfet_w": 0.15,
                "pfet_l": 0.15,
            },
            "drc_tool": "klayout",
            "lvs_tool": "netgen",
            "models": {
                "nfet": "sky130_fd_pr__nfet_01v8",
                "pfet": "sky130_fd_pr__pfet_01v8",
            },
            "description": "Skywater 130nm开源CMOS工艺"
        },
        "gf180": {
            "module": "glayout.pdk.gf180_mapped.gf180_mapped",
            "pdk_var": "gf180_mapped_pdk",
            "pdk_root_env": "PDK_ROOT",
            "tech_node": "180nm",
            "metal_layers": 5,
            "min_dimensions": {
                "nfet_w": 0.22,
                "nfet_l": 0.18,
                "pfet_w": 0.22,
                "pfet_l": 0.18,
            },
            "drc_tool": "klayout",
            "lvs_tool": "magic",
            "models": {
                "nfet": "nfet_03v3",
                "pfet": "pfet_03v3",
            },
            "description": "GlobalFoundries 180nm开源工艺"
        },
        "ihp130": {
            "module": "glayout.pdk.ihp130_mapped.ihp130_mapped",
            "pdk_var": "ihp130_mapped_pdk",
            "pdk_root_env": "PDK_ROOT",
            "tech_node": "130nm",
            "metal_layers": 5,
            "min_dimensions": {
                "nfet_w": 0.13,
                "nfet_l": 0.13,
                "pfet_w": 0.13,
                "pfet_l": 0.13,
            },
            "drc_tool": "klayout",
            "lvs_tool": None,  # 暂不支持LVS
            "models": {
                "nfet": "sg13_lv_nmos",
                "pfet": "sg13_lv_pmos",
            },
            "description": "IHP 130nm SiGe BiCMOS开源工艺"
        }
    }
    
    # 类变量：当前激活的PDK
    _active_pdk_name: ClassVar[Optional[str]] = None
    _pdk_instance: ClassVar[Optional[Any]] = None
    
    @classmethod
    def load_pdk(cls, pdk_name: str) -> Any:
        """加载并激活指定的PDK
        
        Args:
            pdk_name: PDK名称，支持 "sky130", "gf180", "ihp130"
            
        Returns:
            MappedPDK实例
            
        Raises:
            ValueError: 如果PDK名称不支持
            ImportError: 如果PDK模块无法导入
        """
        pdk_name = pdk_name.lower()
        
        if pdk_name not in cls.PDK_CONFIG:
            available = list(cls.PDK_CONFIG.keys())
            raise ValueError(
                f"不支持的PDK: {pdk_name}。可用选项: {available}"
            )
        
        config = cls.PDK_CONFIG[pdk_name]
        
        try:
            # 动态导入PDK模块
            module = importlib.import_module(config["module"])
            pdk = getattr(module, config["pdk_var"])
            
            # 激活PDK
            pdk.activate()
            
            # 更新类变量
            cls._active_pdk_name = pdk_name
            cls._pdk_instance = pdk
            
            return pdk
            
        except ImportError as e:
            raise ImportError(
                f"无法导入PDK模块 {config['module']}: {e}。"
                f"请确保gLayout已正确安装。"
            )
        except AttributeError as e:
            raise ImportError(
                f"PDK模块中找不到 {config['pdk_var']}: {e}"
            )
    
    @classmethod
    def get_active_pdk(cls) -> Any:
        """获取当前激活的PDK实例
        
        Returns:
            当前激活的MappedPDK实例
            
        Raises:
            RuntimeError: 如果没有PDK被激活
        """
        if cls._pdk_instance is None:
            raise RuntimeError(
                "没有PDK被激活。请先调用 PDKManager.load_pdk(pdk_name)"
            )
        return cls._pdk_instance
    
    @classmethod
    def get_active_pdk_name(cls) -> Optional[str]:
        """获取当前激活的PDK名称
        
        Returns:
            当前PDK名称，如果没有激活则返回None
        """
        return cls._active_pdk_name
    
    @classmethod
    def get_pdk_config(cls, pdk_name: Optional[str] = None) -> Dict[str, Any]:
        """获取PDK配置信息
        
        Args:
            pdk_name: PDK名称，默认使用当前激活的PDK
            
        Returns:
            PDK配置字典
        """
        pdk_name = pdk_name or cls._active_pdk_name
        if pdk_name is None:
            raise RuntimeError("没有指定PDK名称且没有激活的PDK")
        
        if pdk_name not in cls.PDK_CONFIG:
            raise ValueError(f"不支持的PDK: {pdk_name}")
        
        return cls.PDK_CONFIG[pdk_name].copy()
    
    @classmethod
    def get_design_rules(cls, pdk_name: Optional[str] = None) -> Dict[str, Any]:
        """获取PDK设计规则
        
        Args:
            pdk_name: PDK名称，默认使用当前激活的PDK
            
        Returns:
            设计规则字典，包含技术节点、金属层数、最小尺寸等
        """
        pdk_name = pdk_name or cls._active_pdk_name
        config = cls.get_pdk_config(pdk_name)
        pdk = cls.get_active_pdk() if pdk_name == cls._active_pdk_name else cls.load_pdk(pdk_name)
        
        # 构建设计规则字典
        rules = {
            "tech_node": config["tech_node"],
            "metal_layers": config["metal_layers"],
            "min_dimensions": config["min_dimensions"],
            "grules": {}
        }
        
        # 尝试获取关键层的设计规则
        key_layers = ["poly", "met1", "met2", "active_diff"]
        for layer in key_layers:
            try:
                rules["grules"][layer] = pdk.get_grule(layer)
            except (NotImplementedError, ValueError):
                # 某些层可能在某些PDK中不存在
                pass
        
        return rules
    
    @classmethod
    def get_layer_mapping(cls, pdk_name: Optional[str] = None) -> Dict[str, Any]:
        """获取层映射信息
        
        Args:
            pdk_name: PDK名称，默认使用当前激活的PDK
            
        Returns:
            glayer到实际层的映射字典
        """
        pdk = cls.get_active_pdk()
        
        mapping = {}
        if hasattr(pdk, 'glayers'):
            for glayer in pdk.glayers:
                try:
                    mapping[glayer] = pdk.get_glayer(glayer)
                except (KeyError, ValueError):
                    pass
        
        return mapping
    
    @classmethod
    def get_min_dimension(cls, dim_type: str, pdk_name: Optional[str] = None) -> float:
        """获取特定类型的最小尺寸
        
        Args:
            dim_type: 尺寸类型，如 "nfet_w", "nfet_l", "pfet_w", "pfet_l"
            pdk_name: PDK名称，默认使用当前激活的PDK
            
        Returns:
            最小尺寸值（单位：um）
        """
        config = cls.get_pdk_config(pdk_name)
        min_dims = config.get("min_dimensions", {})
        
        if dim_type not in min_dims:
            raise ValueError(f"未知的尺寸类型: {dim_type}")
        
        return min_dims[dim_type]
    
    @classmethod
    def get_model_name(cls, device_type: str, pdk_name: Optional[str] = None) -> str:
        """获取器件的SPICE模型名称
        
        Args:
            device_type: 器件类型，如 "nfet", "pfet", "mimcap"
            pdk_name: PDK名称，默认使用当前激活的PDK
            
        Returns:
            SPICE模型名称
        """
        config = cls.get_pdk_config(pdk_name)
        models = config.get("models", {})
        
        if device_type not in models:
            raise ValueError(
                f"未知的器件类型: {device_type}。"
                f"支持的类型: {list(models.keys())}"
            )
        
        return models[device_type]
    
    @classmethod
    def list_available_pdks(cls) -> List[Dict[str, str]]:
        """列出所有可用的PDK
        
        Returns:
            PDK信息列表
        """
        return [
            {
                "name": name,
                "tech_node": config["tech_node"],
                "description": config.get("description", ""),
                "drc_tool": config["drc_tool"],
                "lvs_tool": config["lvs_tool"] or "不支持"
            }
            for name, config in cls.PDK_CONFIG.items()
        ]
    
    @classmethod
    def snap_to_grid(cls, value: float) -> float:
        """将值对齐到制造网格
        
        Args:
            value: 需要对齐的值
            
        Returns:
            对齐后的值
        """
        pdk = cls.get_active_pdk()
        return pdk.snap_to_2xgrid(value)
    
    @classmethod
    def get_max_metal_separation(cls, metal_levels: Optional[List[int]] = None) -> float:
        """获取金属层的最大间距要求
        
        Args:
            metal_levels: 金属层级别列表，默认为1-5层
            
        Returns:
            最大间距值
        """
        pdk = cls.get_active_pdk()
        if metal_levels is None:
            metal_levels = list(range(1, 6))
        return pdk.util_max_metal_seperation(metal_levels)
    
    @classmethod
    def validate_device_params(
        cls, 
        device_type: str,
        width: float,
        length: Optional[float] = None,
        pdk_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """验证器件参数是否符合PDK规则
        
        Args:
            device_type: 器件类型 "nfet" 或 "pfet"
            width: 沟道宽度
            length: 沟道长度，如果为None则使用最小长度
            pdk_name: PDK名称
            
        Returns:
            验证结果字典，包含是否有效、错误信息、调整后的参数等
        """
        config = cls.get_pdk_config(pdk_name)
        min_dims = config["min_dimensions"]
        
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "adjusted_params": {}
        }
        
        # 验证宽度
        min_w = min_dims.get(f"{device_type}_w", 0.15)
        if width < min_w:
            result["valid"] = False
            result["errors"].append(
                f"宽度 {width}um 小于最小值 {min_w}um"
            )
            result["adjusted_params"]["width"] = min_w
        else:
            result["adjusted_params"]["width"] = width
        
        # 验证长度
        min_l = min_dims.get(f"{device_type}_l", 0.15)
        if length is None:
            result["adjusted_params"]["length"] = min_l
            result["warnings"].append(
                f"使用默认最小长度 {min_l}um"
            )
        elif length < min_l:
            result["valid"] = False
            result["errors"].append(
                f"长度 {length}um 小于最小值 {min_l}um"
            )
            result["adjusted_params"]["length"] = min_l
        else:
            result["adjusted_params"]["length"] = length
        
        return result
    
    @classmethod
    def reset(cls) -> None:
        """重置PDK管理器状态"""
        cls._active_pdk_name = None
        cls._pdk_instance = None
