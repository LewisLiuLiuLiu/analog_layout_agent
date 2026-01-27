"""
Device Schemas - 器件参数Schema定义

定义器件创建工具的输入参数格式
"""

from typing import Dict, Any, List


# ============== 基础器件Schema ==============

NMOS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["width"],
    "properties": {
        "width": {
            "type": "number",
            "description": "沟道宽度(um)",
            "minimum": 0.15
        },
        "length": {
            "type": "number",
            "description": "沟道长度(um)，默认使用PDK最小长度",
            "minimum": 0.15
        },
        "fingers": {
            "type": "integer",
            "description": "指数(每个MOS的栅极数量)",
            "default": 1,
            "minimum": 1
        },
        "multiplier": {
            "type": "integer",
            "description": "并联倍数(重复的MOS数量)",
            "default": 1,
            "minimum": 1
        },
        "with_dummy": {
            "type": "boolean",
            "description": "是否添加dummy结构（改善匹配性）",
            "default": True
        },
        "with_tie": {
            "type": "boolean",
            "description": "是否添加衬底连接(substrate tap)",
            "default": True
        },
        "sd_route_topmet": {
            "type": "string",
            "description": "源漏路由的顶层金属",
            "enum": ["met1", "met2", "met3", "met4", "met5"],
            "default": "met2"
        },
        "gate_route_topmet": {
            "type": "string",
            "description": "栅极路由的顶层金属",
            "enum": ["met1", "met2", "met3", "met4", "met5"],
            "default": "met2"
        },
        "name": {
            "type": "string",
            "description": "组件名称，不指定则自动生成"
        }
    }
}


PMOS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["width"],
    "properties": {
        "width": {
            "type": "number",
            "description": "沟道宽度(um)",
            "minimum": 0.15
        },
        "length": {
            "type": "number",
            "description": "沟道长度(um)，默认使用PDK最小长度",
            "minimum": 0.15
        },
        "fingers": {
            "type": "integer",
            "description": "指数",
            "default": 1,
            "minimum": 1
        },
        "multiplier": {
            "type": "integer",
            "description": "并联倍数",
            "default": 1,
            "minimum": 1
        },
        "with_dummy": {
            "type": "boolean",
            "description": "是否添加dummy结构",
            "default": True
        },
        "with_tie": {
            "type": "boolean",
            "description": "是否添加衬底连接",
            "default": True
        },
        "sd_route_topmet": {
            "type": "string",
            "description": "源漏路由的顶层金属",
            "enum": ["met1", "met2", "met3", "met4", "met5"],
            "default": "met2"
        },
        "gate_route_topmet": {
            "type": "string",
            "description": "栅极路由的顶层金属",
            "enum": ["met1", "met2", "met3", "met4", "met5"],
            "default": "met2"
        },
        "name": {
            "type": "string",
            "description": "组件名称，不指定则自动生成"
        }
    }
}


VIA_STACK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["from_layer", "to_layer"],
    "properties": {
        "from_layer": {
            "type": "string",
            "description": "起始层",
            "enum": ["met1", "met2", "met3", "met4", "met5", "poly", "active_diff"]
        },
        "to_layer": {
            "type": "string",
            "description": "目标层",
            "enum": ["met1", "met2", "met3", "met4", "met5"]
        },
        "size": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Via尺寸[宽,高](um)",
            "minItems": 2,
            "maxItems": 2
        },
        "name": {
            "type": "string",
            "description": "组件名称"
        }
    }
}


VIA_ARRAY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["from_layer", "to_layer"],
    "properties": {
        "from_layer": {
            "type": "string",
            "description": "起始层",
            "enum": ["met1", "met2", "met3", "met4", "met5"]
        },
        "to_layer": {
            "type": "string",
            "description": "目标层",
            "enum": ["met1", "met2", "met3", "met4", "met5"]
        },
        "num_vias": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Via数量[行,列]",
            "default": [1, 1]
        },
        "name": {
            "type": "string",
            "description": "组件名称"
        }
    }
}


MIMCAP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["width", "length"],
    "properties": {
        "width": {
            "type": "number",
            "description": "电容宽度(um)",
            "minimum": 0.5
        },
        "length": {
            "type": "number",
            "description": "电容长度(um)",
            "minimum": 0.5
        },
        "name": {
            "type": "string",
            "description": "组件名称"
        }
    }
}


RESISTOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["width", "length"],
    "properties": {
        "width": {
            "type": "number",
            "description": "电阻宽度(um)",
            "minimum": 0.3
        },
        "length": {
            "type": "number",
            "description": "电阻长度(um)",
            "minimum": 0.5
        },
        "num_series": {
            "type": "integer",
            "description": "串联段数",
            "default": 1,
            "minimum": 1
        },
        "name": {
            "type": "string",
            "description": "组件名称"
        }
    }
}


TAPRING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["enclosed_rectangle"],
    "properties": {
        "enclosed_rectangle": {
            "type": "array",
            "items": {"type": "number"},
            "description": "包围区域尺寸[宽,高](um)",
            "minItems": 2,
            "maxItems": 2
        },
        "sdlayer": {
            "type": "string",
            "description": "source/drain层类型",
            "enum": ["n+s/d", "p+s/d"],
            "default": "p+s/d"
        },
        "name": {
            "type": "string",
            "description": "组件名称"
        }
    }
}


# ============== 复合电路Schema ==============

CURRENT_MIRROR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "device_type": {
            "type": "string",
            "description": "器件类型",
            "enum": ["nmos", "pmos"],
            "default": "nmos"
        },
        "width": {
            "type": "number",
            "description": "管子宽度(um)",
            "default": 3.0,
            "minimum": 0.15
        },
        "length": {
            "type": "number",
            "description": "管子长度(um)",
            "minimum": 0.15
        },
        "numcols": {
            "type": "integer",
            "description": "互指列数",
            "default": 3,
            "minimum": 2
        },
        "with_dummy": {
            "type": "boolean",
            "description": "是否添加dummy",
            "default": True
        },
        "with_tie": {
            "type": "boolean",
            "description": "是否添加衬底连接",
            "default": True
        },
        "name": {
            "type": "string",
            "description": "组件名称"
        }
    }
}


DIFF_PAIR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["width"],
    "properties": {
        "device_type": {
            "type": "string",
            "description": "器件类型",
            "enum": ["nmos", "pmos"],
            "default": "nmos"
        },
        "width": {
            "type": "number",
            "description": "管子宽度(um)",
            "minimum": 0.15
        },
        "length": {
            "type": "number",
            "description": "管子长度(um)",
            "minimum": 0.15
        },
        "fingers": {
            "type": "integer",
            "description": "指数",
            "default": 1,
            "minimum": 1
        },
        "numcols": {
            "type": "integer",
            "description": "互指列数，影响匹配性能",
            "default": 2,
            "minimum": 2
        },
        "layout_style": {
            "type": "string",
            "description": "布局风格",
            "enum": ["interdigitized", "common_centroid"],
            "default": "interdigitized"
        },
        "name": {
            "type": "string",
            "description": "组件名称"
        }
    }
}


OPAMP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "topology": {
            "type": "string",
            "description": "运放拓扑",
            "enum": ["two_stage", "folded_cascode"],
            "default": "two_stage"
        },
        "input_pair_w": {
            "type": "number",
            "description": "输入对管宽度(um)",
            "default": 5.0
        },
        "load_w": {
            "type": "number",
            "description": "负载管宽度(um)",
            "default": 2.0
        },
        "bias_current": {
            "type": "number",
            "description": "偏置电流(uA)",
            "default": 10.0
        },
        "name": {
            "type": "string",
            "description": "组件名称"
        }
    }
}


# ============== Schema注册表 ==============

DEVICE_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "nmos": NMOS_SCHEMA,
    "pmos": PMOS_SCHEMA,
    "via_stack": VIA_STACK_SCHEMA,
    "via_array": VIA_ARRAY_SCHEMA,
    "mimcap": MIMCAP_SCHEMA,
    "resistor": RESISTOR_SCHEMA,
    "tapring": TAPRING_SCHEMA,
}


CIRCUIT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "current_mirror": CURRENT_MIRROR_SCHEMA,
    "diff_pair": DIFF_PAIR_SCHEMA,
    "opamp": OPAMP_SCHEMA,
}


def get_schema(name: str) -> Dict[str, Any]:
    """获取指定器件/电路的Schema
    
    Args:
        name: 器件或电路名称
        
    Returns:
        Schema字典
        
    Raises:
        KeyError: 如果名称不存在
    """
    if name in DEVICE_SCHEMAS:
        return DEVICE_SCHEMAS[name]
    if name in CIRCUIT_SCHEMAS:
        return CIRCUIT_SCHEMAS[name]
    raise KeyError(f"未知的Schema: {name}")


def list_schemas() -> Dict[str, List[str]]:
    """列出所有可用的Schema
    
    Returns:
        按类别组织的Schema名称列表
    """
    return {
        "devices": list(DEVICE_SCHEMAS.keys()),
        "circuits": list(CIRCUIT_SCHEMAS.keys())
    }
