"""
Routing Schemas - 路由参数Schema定义

定义路由工具的输入参数格式
"""

from typing import Dict, Any


# ============== 路由Schema ==============

SMART_ROUTE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["source_port", "dest_port"],
    "properties": {
        "source_port": {
            "type": "string",
            "description": "源端口(格式: component_name.port_name)"
        },
        "dest_port": {
            "type": "string",
            "description": "目标端口(格式: component_name.port_name)"
        },
        "layer": {
            "type": "string",
            "description": "布线层",
            "enum": ["met1", "met2", "met3", "met4", "met5"],
            "default": "met2"
        },
        "width": {
            "type": "number",
            "description": "布线宽度(um)，默认使用端口宽度"
        }
    }
}


C_ROUTE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["source_port", "dest_port"],
    "properties": {
        "source_port": {
            "type": "string",
            "description": "源端口(格式: component_name.port_name)"
        },
        "dest_port": {
            "type": "string",
            "description": "目标端口(格式: component_name.port_name)"
        },
        "extension": {
            "type": "number",
            "description": "延伸长度(um)，默认自动计算"
        },
        "cglayer": {
            "type": "string",
            "description": "连接层金属层",
            "enum": ["met1", "met2", "met3", "met4", "met5"],
            "default": "met2"
        },
        "cwidth": {
            "type": "number",
            "description": "连接线宽度(um)"
        },
        "width1": {
            "type": "number",
            "description": "源端宽度(um)，默认使用端口宽度"
        },
        "width2": {
            "type": "number",
            "description": "目标端宽度(um)，默认使用端口宽度"
        },
        "e1glayer": {
            "type": "string",
            "description": "源端金属层，默认使用端口层",
            "enum": ["met1", "met2", "met3", "met4", "met5"]
        },
        "e2glayer": {
            "type": "string",
            "description": "目标端金属层，默认使用端口层",
            "enum": ["met1", "met2", "met3", "met4", "met5"]
        },
        "viaoffset": {
            "type": "boolean",
            "description": "过孔偏移，默认true",
            "default": True
        },
        "fullbottom": {
            "type": "boolean",
            "description": "过孔底部填满",
            "default": False
        }
    }
}


L_ROUTE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["source_port", "dest_port"],
    "properties": {
        "source_port": {
            "type": "string",
            "description": "源端口(格式: component_name.port_name)"
        },
        "dest_port": {
            "type": "string",
            "description": "目标端口(格式: component_name.port_name)"
        },
        "hglayer": {
            "type": "string",
            "description": "水平连接金属层",
            "enum": ["met1", "met2", "met3", "met4", "met5"],
            "default": "met2"
        },
        "vglayer": {
            "type": "string",
            "description": "垂直连接金属层",
            "enum": ["met1", "met2", "met3", "met4", "met5"],
            "default": "met2"
        },
        "hwidth": {
            "type": "number",
            "description": "水平线宽度(um)，默认使用端口宽度"
        },
        "vwidth": {
            "type": "number",
            "description": "垂直线宽度(um)，默认使用端口宽度"
        },
        "viaoffset": {
            "type": "boolean",
            "description": "过孔偏移，默认true",
            "default": True
        },
        "fullbottom": {
            "type": "boolean",
            "description": "过孔底部填满",
            "default": True
        }
    }
}


STRAIGHT_ROUTE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["source_port", "dest_port"],
    "properties": {
        "source_port": {
            "type": "string",
            "description": "源端口(格式: component_name.port_name)"
        },
        "dest_port": {
            "type": "string",
            "description": "目标端口(格式: component_name.port_name)"
        },
        "glayer1": {
            "type": "string",
            "description": "起始层金属层，默认使用端口层",
            "enum": ["met1", "met2", "met3", "met4", "met5"]
        },
        "glayer2": {
            "type": "string",
            "description": "结束层金属层，默认使用端口层",
            "enum": ["met1", "met2", "met3", "met4", "met5"]
        },
        "width": {
            "type": "number",
            "description": "布线宽度(um)，默认使用端口宽度"
        },
        "fullbottom": {
            "type": "boolean",
            "description": "过孔底部填满",
            "default": False
        }
    }
}


# ============== 路由Schema注册表 ==============

ROUTING_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "smart_route": SMART_ROUTE_SCHEMA,
    "c_route": C_ROUTE_SCHEMA,
    "l_route": L_ROUTE_SCHEMA,
    "straight_route": STRAIGHT_ROUTE_SCHEMA,
}


def get_routing_schema(name: str) -> Dict[str, Any]:
    """获取指定路由类型的Schema
    
    Args:
        name: 路由类型名称
        
    Returns:
        Schema字典
        
    Raises:
        KeyError: 如果名称不存在
    """
    if name in ROUTING_SCHEMAS:
        return ROUTING_SCHEMAS[name]
    raise KeyError(f"未知的路由Schema: {name}")


def list_routing_schemas() -> list:
    """列出所有可用的路由Schema
    
    Returns:
        路由Schema名称列表
    """
    return list(ROUTING_SCHEMAS.keys())
