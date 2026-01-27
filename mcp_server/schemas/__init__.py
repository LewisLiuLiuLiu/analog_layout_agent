"""
MCP Schemas - 参数Schema定义

定义各工具的输入输出参数格式
"""

from .device_schemas import (
    DEVICE_SCHEMAS,
    CIRCUIT_SCHEMAS,
    NMOS_SCHEMA,
    PMOS_SCHEMA,
    VIA_STACK_SCHEMA,
    VIA_ARRAY_SCHEMA,
    MIMCAP_SCHEMA,
    RESISTOR_SCHEMA,
    TAPRING_SCHEMA,
    CURRENT_MIRROR_SCHEMA,
    DIFF_PAIR_SCHEMA,
    OPAMP_SCHEMA,
    get_schema,
    list_schemas,
)

from .routing_schemas import (
    ROUTING_SCHEMAS,
    SMART_ROUTE_SCHEMA,
    C_ROUTE_SCHEMA,
    L_ROUTE_SCHEMA,
    STRAIGHT_ROUTE_SCHEMA,
    get_routing_schema,
    list_routing_schemas,
)

from .common_schemas import (
    PLACEMENT_SCHEMAS,
    VERIFICATION_SCHEMAS,
    EXPORT_SCHEMAS,
    PLACE_COMPONENT_SCHEMA,
    ALIGN_TO_PORT_SCHEMA,
    MOVE_COMPONENT_SCHEMA,
    INTERDIGITIZE_SCHEMA,
    RUN_DRC_SCHEMA,
    RUN_LVS_SCHEMA,
    EXTRACT_NETLIST_SCHEMA,
    EXPORT_GDS_SCHEMA,
    EXPORT_IMAGE_SCHEMA,
    get_placement_schema,
    get_verification_schema,
    get_export_schema,
    list_all_schemas,
)

__all__ = [
    # Device schemas
    "DEVICE_SCHEMAS",
    "CIRCUIT_SCHEMAS",
    "NMOS_SCHEMA",
    "PMOS_SCHEMA",
    "VIA_STACK_SCHEMA",
    "VIA_ARRAY_SCHEMA",
    "MIMCAP_SCHEMA",
    "RESISTOR_SCHEMA",
    "TAPRING_SCHEMA",
    "CURRENT_MIRROR_SCHEMA",
    "DIFF_PAIR_SCHEMA",
    "OPAMP_SCHEMA",
    "get_schema",
    "list_schemas",
    # Routing schemas
    "ROUTING_SCHEMAS",
    "SMART_ROUTE_SCHEMA",
    "C_ROUTE_SCHEMA",
    "L_ROUTE_SCHEMA",
    "STRAIGHT_ROUTE_SCHEMA",
    "get_routing_schema",
    "list_routing_schemas",
    # Common schemas
    "PLACEMENT_SCHEMAS",
    "VERIFICATION_SCHEMAS",
    "EXPORT_SCHEMAS",
    "PLACE_COMPONENT_SCHEMA",
    "ALIGN_TO_PORT_SCHEMA",
    "MOVE_COMPONENT_SCHEMA",
    "INTERDIGITIZE_SCHEMA",
    "RUN_DRC_SCHEMA",
    "RUN_LVS_SCHEMA",
    "EXTRACT_NETLIST_SCHEMA",
    "EXPORT_GDS_SCHEMA",
    "EXPORT_IMAGE_SCHEMA",
    "get_placement_schema",
    "get_verification_schema",
    "get_export_schema",
    "list_all_schemas",
]
