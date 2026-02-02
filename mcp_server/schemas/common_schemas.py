"""
Common Schemas - 通用Schema定义

定义放置、验证、导出等工具的参数格式
"""

from typing import Dict, Any, List


# ============== 放置Schema ==============

PLACE_COMPONENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["component_name"],
    "properties": {
        "component_name": {
            "type": "string",
            "description": "要放置的组件名称"
        },
        "x": {
            "type": "number",
            "description": "X坐标(um)",
            "default": 0
        },
        "y": {
            "type": "number",
            "description": "Y坐标(um)",
            "default": 0
        },
        "rotation": {
            "type": "number",
            "description": "旋转角度",
            "enum": [0, 90, 180, 270],
            "default": 0
        },
        "mirror": {
            "type": "boolean",
            "description": "是否镜像",
            "default": False
        }
    }
}


ALIGN_TO_PORT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["component_name", "target_port"],
    "properties": {
        "component_name": {
            "type": "string",
            "description": "要对齐的组件名称"
        },
        "target_port": {
            "type": "string",
            "description": "目标端口(格式: component_name.port_name)"
        },
        "alignment": {
            "type": "string",
            "description": "对齐方式",
            "enum": ["center", "left", "right", "top", "bottom"],
            "default": "center"
        },
        "offset_x": {
            "type": "number",
            "description": "X方向偏移(um)",
            "default": 0
        },
        "offset_y": {
            "type": "number",
            "description": "Y方向偏移(um)",
            "default": 0
        }
    }
}


MOVE_COMPONENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["component_name"],
    "properties": {
        "component_name": {
            "type": "string",
            "description": "要移动的组件名称"
        },
        "dx": {
            "type": "number",
            "description": "X方向移动距离(um)",
            "default": 0
        },
        "dy": {
            "type": "number",
            "description": "Y方向移动距离(um)",
            "default": 0
        }
    }
}


INTERDIGITIZE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["comp_a", "comp_b"],
    "properties": {
        "comp_a": {
            "type": "string",
            "description": "组件A名称"
        },
        "comp_b": {
            "type": "string",
            "description": "组件B名称"
        },
        "num_cols": {
            "type": "integer",
            "description": "互指列数",
            "default": 4,
            "minimum": 2
        },
        "layout_style": {
            "type": "string",
            "description": "布局风格",
            "enum": ["ABAB", "ABBA", "common_centroid"],
            "default": "ABAB"
        }
    }
}


# ============== 验证Schema ==============

RUN_DRC_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "component_name": {
            "type": "string",
            "description": "组件名称，默认检查顶层"
        },
        "output_format": {
            "type": "string",
            "description": "输出格式",
            "enum": ["summary", "detailed"],
            "default": "summary"
        },
        "output_dir": {
            "type": "string",
            "description": "输出目录"
        },
        "include_suggestions": {
            "type": "boolean",
            "description": "是否生成修复建议",
            "default": True
        }
    }
}


RUN_LVS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["schematic_netlist"],
    "properties": {
        "component_name": {
            "type": "string",
            "description": "组件名称，默认使用顶层"
        },
        "schematic_netlist": {
            "type": "string",
            "description": "参考原理图网表路径或内容"
        },
        "output_dir": {
            "type": "string",
            "description": "输出目录"
        }
    }
}


EXTRACT_NETLIST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "component_name": {
            "type": "string",
            "description": "组件名称，默认使用顶层"
        },
        "format": {
            "type": "string",
            "description": "输出格式",
            "enum": ["spice", "spectre"],
            "default": "spice"
        },
        "output_file": {
            "type": "string",
            "description": "输出文件路径"
        }
    }
}


# ============== 导出Schema ==============

EXPORT_GDS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "filename": {
            "type": "string",
            "description": "输出文件名"
        },
        "component_name": {
            "type": "string",
            "description": "组件名称，默认导出顶层"
        },
        "with_ports": {
            "type": "boolean",
            "description": "是否包含端口标记",
            "default": True
        }
    }
}


EXPORT_IMAGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "filename": {
            "type": "string",
            "description": "输出文件名"
        },
        "component_name": {
            "type": "string",
            "description": "组件名称，默认导出顶层"
        },
        "show_ports": {
            "type": "boolean",
            "description": "是否显示端口",
            "default": True
        },
        "show_subports": {
            "type": "boolean",
            "description": "是否显示子端口",
            "default": False
        }
    }
}


# ============== Schema注册表 ==============

PLACEMENT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "place_component": PLACE_COMPONENT_SCHEMA,
    "align_to_port": ALIGN_TO_PORT_SCHEMA,
    "move_component": MOVE_COMPONENT_SCHEMA,
    "interdigitize": INTERDIGITIZE_SCHEMA,
}


VERIFICATION_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "run_drc": RUN_DRC_SCHEMA,
    "run_lvs": RUN_LVS_SCHEMA,
    "extract_netlist": EXTRACT_NETLIST_SCHEMA,
}


EXPORT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "export_gds": EXPORT_GDS_SCHEMA,
    "export_image": EXPORT_IMAGE_SCHEMA,
}


def get_placement_schema(name: str) -> Dict[str, Any]:
    """获取放置Schema"""
    if name in PLACEMENT_SCHEMAS:
        return PLACEMENT_SCHEMAS[name]
    raise KeyError(f"未知的放置Schema: {name}")


def get_verification_schema(name: str) -> Dict[str, Any]:
    """获取验证Schema"""
    if name in VERIFICATION_SCHEMAS:
        return VERIFICATION_SCHEMAS[name]
    raise KeyError(f"未知的验证Schema: {name}")


def get_export_schema(name: str) -> Dict[str, Any]:
    """获取导出Schema"""
    if name in EXPORT_SCHEMAS:
        return EXPORT_SCHEMAS[name]
    raise KeyError(f"未知的导出Schema: {name}")


def list_all_schemas() -> Dict[str, List[str]]:
    """列出所有Schema
    
    Returns:
        按类别组织的Schema名称列表
    """
    from .device_schemas import DEVICE_SCHEMAS, CIRCUIT_SCHEMAS
    from .routing_schemas import ROUTING_SCHEMAS
    
    return {
        "devices": list(DEVICE_SCHEMAS.keys()),
        "circuits": list(CIRCUIT_SCHEMAS.keys()),
        "routing": list(ROUTING_SCHEMAS.keys()),
        "placement": list(PLACEMENT_SCHEMAS.keys()),
        "verification": list(VERIFICATION_SCHEMAS.keys()),
        "export": list(EXPORT_SCHEMAS.keys())
    }
