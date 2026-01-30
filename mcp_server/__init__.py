"""
MCP Server module - Model Context Protocol 服务端
MCP Server module - Model Context Protocol server-side

提供LLM友好的工具抽象层，支持 PydanticAI 框架集成
Provides LLM-friendly tool abstraction layer, supports PydanticAI framework integration
"""

from .server import MCPServer, get_server
from .pydantic_toolset import (
    MCPLayoutToolset,
    MCPLayoutToolsetAsync,
    create_layout_toolset,
    DEVICE_TOOLS,
    ROUTING_TOOLS,
    PLACEMENT_TOOLS,
    QUERY_TOOLS,
    EXPORT_TOOLS,
    ALL_LAYOUT_TOOLS,
)

__all__ = [
    # MCP Server
    "MCPServer",
    "get_server",
    # PydanticAI Toolset
    "MCPLayoutToolset",
    "MCPLayoutToolsetAsync",
    "create_layout_toolset",
    # 工具类别常量 / Tool category constants
    "DEVICE_TOOLS",
    "ROUTING_TOOLS",
    "PLACEMENT_TOOLS",
    "QUERY_TOOLS",
    "EXPORT_TOOLS",
    "ALL_LAYOUT_TOOLS",
]
