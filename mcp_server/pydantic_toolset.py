"""
PydanticAI MCP Toolset 集成模块

将现有的 MCP Server 工具包装为 PydanticAI 的 Toolset，
实现与 PydanticAI Agent 的无缝集成。

支持两种使用模式：
1. 内部工具模式：通过 LayoutAgentDeps.call_tool() 调用
2. Toolset 模式：作为 PydanticAI 的 toolsets 参数传入

Usage:
    # 模式1: 内部工具模式（推荐，当前默认）
    from pydantic_agent import create_layout_agent, LayoutAgentDeps
    agent = create_layout_agent()
    deps = LayoutAgentDeps(mcp_server=mcp_server, ...)
    result = await agent.run(instruction, deps=deps)
    
    # 模式2: Toolset 模式
    from mcp_server.pydantic_toolset import MCPLayoutToolset
    toolset = MCPLayoutToolset(mcp_server)
    agent = Agent(model, toolsets=[toolset])
"""

import json
import logging
from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import Tool

logger = logging.getLogger(__name__)


@dataclass
class MCPToolDefinition:
    """MCP 工具定义"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    category: str = "general"


class MCPLayoutToolset:
    """MCP Server 的 PydanticAI Toolset 包装器
    
    将 MCP Server 的工具转换为 PydanticAI 兼容的工具集。
    
    特点：
    - 自动从 MCP Server 发现和注册工具
    - 支持工具分类过滤
    - 提供统一的工具调用接口
    
    Usage:
        >>> from mcp_server.server import MCPServer
        >>> mcp_server = MCPServer()
        >>> mcp_server.initialize(pdk_name="sky130")
        >>> 
        >>> toolset = MCPLayoutToolset(mcp_server)
        >>> tools = toolset.get_tools()  # 获取所有工具
    """
    
    def __init__(
        self,
        mcp_server: "MCPServer",  # type: ignore
        categories: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None
    ):
        """初始化 Toolset
        
        Args:
            mcp_server: MCP Server 实例
            categories: 要包含的工具类别，None 表示全部
            exclude_tools: 要排除的工具名称列表
        """
        self.mcp_server = mcp_server
        self.categories = categories
        self.exclude_tools = exclude_tools or []
        self._tools_cache: Optional[List[Tool]] = None
    
    def get_tools(self) -> List[Tool]:
        """获取所有工具
        
        将 MCP Server 的工具转换为 PydanticAI Tool 对象。
        
        Returns:
            PydanticAI Tool 对象列表
        """
        if self._tools_cache is not None:
            return self._tools_cache
        
        tools = []
        mcp_tools = self.mcp_server.list_tools(category=None)
        
        for tool_def in mcp_tools:
            tool_name = tool_def["name"]
            
            # 跳过排除的工具
            if tool_name in self.exclude_tools:
                continue
            
            # 过滤类别
            if self.categories and tool_def.get("category") not in self.categories:
                continue
            
            # 创建工具函数
            tool_func = self._create_tool_function(tool_name)
            
            # 创建 PydanticAI Tool
            tool = Tool(
                function=tool_func,
                takes_ctx=False,  # 不需要上下文，直接调用 MCP
                name=tool_name,
                description=tool_def.get("description", ""),
            )
            tools.append(tool)
        
        self._tools_cache = tools
        return tools
    
    def _create_tool_function(self, tool_name: str) -> Callable[..., str]:
        """为指定工具创建包装函数
        
        Args:
            tool_name: 工具名称
            
        Returns:
            工具包装函数
        """
        mcp_server = self.mcp_server  # 闭包捕获
        
        def tool_wrapper(**kwargs) -> str:
            """调用 MCP Server 工具"""
            result = mcp_server.call_tool(tool_name, kwargs)
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        # 设置函数名和文档
        tool_wrapper.__name__ = tool_name
        tool_def = self.mcp_server.get_tool(tool_name)
        if tool_def:
            tool_wrapper.__doc__ = tool_def.description
        
        return tool_wrapper
    
    def call_tool(self, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """直接调用 MCP 工具
        
        Args:
            tool_name: 工具名称
            params: 工具参数
            
        Returns:
            工具执行结果
        """
        return self.mcp_server.call_tool(tool_name, params or {})


class MCPLayoutToolsetAsync:
    """异步版本的 MCP Toolset
    
    支持异步工具调用，适用于需要非阻塞操作的场景。
    """
    
    def __init__(
        self,
        mcp_server: "MCPServer",  # type: ignore
        categories: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None
    ):
        self.mcp_server = mcp_server
        self.categories = categories
        self.exclude_tools = exclude_tools or []
        self._tools_cache: Optional[List[Tool]] = None
    
    def get_tools(self) -> List[Tool]:
        """获取异步工具列表"""
        if self._tools_cache is not None:
            return self._tools_cache
        
        tools = []
        mcp_tools = self.mcp_server.list_tools(category=None)
        
        for tool_def in mcp_tools:
            tool_name = tool_def["name"]
            
            if tool_name in self.exclude_tools:
                continue
            
            if self.categories and tool_def.get("category") not in self.categories:
                continue
            
            tool_func = self._create_async_tool_function(tool_name)
            
            tool = Tool(
                function=tool_func,
                takes_ctx=False,
                name=tool_name,
                description=tool_def.get("description", ""),
            )
            tools.append(tool)
        
        self._tools_cache = tools
        return tools
    
    def _create_async_tool_function(self, tool_name: str) -> Callable[..., Awaitable[str]]:
        """创建异步工具包装函数"""
        mcp_server = self.mcp_server
        
        async def async_tool_wrapper(**kwargs) -> str:
            """异步调用 MCP Server 工具"""
            # 实际的 MCP 调用是同步的，这里包装为异步
            result = mcp_server.call_tool(tool_name, kwargs)
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        async_tool_wrapper.__name__ = tool_name
        tool_def = self.mcp_server.get_tool(tool_name)
        if tool_def:
            async_tool_wrapper.__doc__ = tool_def.description
        
        return async_tool_wrapper


def create_layout_toolset(
    pdk_name: str = "sky130",
    design_name: str = "top_level",
    categories: Optional[List[str]] = None,
    exclude_tools: Optional[List[str]] = None,
    async_mode: bool = False
) -> MCPLayoutToolset | MCPLayoutToolsetAsync:
    """创建布局工具集的便捷函数
    
    Args:
        pdk_name: PDK 名称
        design_name: 设计名称
        categories: 要包含的工具类别
        exclude_tools: 要排除的工具
        async_mode: 是否使用异步模式
        
    Returns:
        MCPLayoutToolset 或 MCPLayoutToolsetAsync 实例
    """
    from mcp_server.server import MCPServer
    
    mcp_server = MCPServer()
    init_result = mcp_server.initialize(pdk_name=pdk_name, design_name=design_name)
    
    if not init_result.get("success"):
        raise RuntimeError(f"MCP Server 初始化失败: {init_result.get('error')}")
    
    if async_mode:
        return MCPLayoutToolsetAsync(
            mcp_server=mcp_server,
            categories=categories,
            exclude_tools=exclude_tools
        )
    else:
        return MCPLayoutToolset(
            mcp_server=mcp_server,
            categories=categories,
            exclude_tools=exclude_tools
        )


# ============== 工具类别常量 ==============

DEVICE_TOOLS = ["create_nmos", "create_pmos", "create_via_stack", "create_mimcap", "create_resistor"]
ROUTING_TOOLS = ["smart_route", "c_route", "l_route", "straight_route"]
PLACEMENT_TOOLS = ["place_component", "align_to_port", "move_component", "interdigitize"]
QUERY_TOOLS = ["list_components", "get_component_info", "get_context_status"]
EXPORT_TOOLS = ["export_gds"]
PDK_TOOLS = ["list_pdks", "get_pdk_info", "switch_pdk"]
SESSION_TOOLS = ["new_session", "clear_layout"]

ALL_LAYOUT_TOOLS = (
    DEVICE_TOOLS + ROUTING_TOOLS + PLACEMENT_TOOLS + 
    QUERY_TOOLS + EXPORT_TOOLS + PDK_TOOLS + SESSION_TOOLS
)
