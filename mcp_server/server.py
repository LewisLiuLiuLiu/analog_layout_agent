"""
MCP Server - Model Context Protocol Server主入口
MCP Server - Model Context Protocol Server main entry

提供LLM友好的工具调用接口，遵循MCP协议规范
Provides LLM-friendly tool invocation interface, following MCP protocol specification
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

# 添加路径 / Add paths
_BASE_PATH = Path(__file__).parent.parent
if str(_BASE_PATH) not in sys.path:
    sys.path.insert(0, str(_BASE_PATH))

from core.pdk_manager import PDKManager
from core.layout_context import LayoutContext
from mcp_server.handlers.state_handler import StateHandler
from mcp_server.handlers.error_handler import ErrorHandler, LayoutError
from mcp_server.tools.device_tools import DeviceToolExecutor
from mcp_server.tools.routing_tools import RoutingToolExecutor
from mcp_server.tools.placement_tools import PlacementToolExecutor
from mcp_server.schemas.device_schemas import (
    NMOS_SCHEMA, PMOS_SCHEMA, VIA_STACK_SCHEMA, MIMCAP_SCHEMA, RESISTOR_SCHEMA
)
from mcp_server.schemas.routing_schemas import (
    SMART_ROUTE_SCHEMA, C_ROUTE_SCHEMA, L_ROUTE_SCHEMA, STRAIGHT_ROUTE_SCHEMA
)
from mcp_server.schemas.common_schemas import (
    PLACE_COMPONENT_SCHEMA, ALIGN_TO_PORT_SCHEMA, MOVE_COMPONENT_SCHEMA,
    INTERDIGITIZE_SCHEMA
)


# 配置日志 / Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """工具定义 / Tool definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable
    category: str = "general"


@dataclass
class ToolResult:
    """工具执行结果 / Tool execution result"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 / Convert to dictionary"""
        result = {"success": self.success}
        if self.data:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        return result


class MCPServer:
    """MCP Server - Model Context Protocol服务端
    MCP Server - Model Context Protocol server
    
    提供LLM工具调用的标准化接口：
    Provides standardized interface for LLM tool invocation:
    - 工具注册和发现 / Tool registration and discovery
    - 工具执行和结果返回 / Tool execution and result return
    - 状态管理 / State management
    - 错误处理 / Error handling
    
    Usage:
        >>> server = MCPServer()
        >>> server.initialize(pdk_name="sky130")
        >>> tools = server.list_tools()
        >>> result = server.call_tool("create_nmos", {"width": 1.0})
    """
    
    def __init__(self):
        """初始化MCP Server"""
        self.state_handler = StateHandler()
        self.error_handler = ErrorHandler()
        
        # 工具注册表
        self._tools: Dict[str, ToolDefinition] = {}
        
        # 工具执行器（延迟初始化，需要context）
        self._device_executor: Optional[DeviceToolExecutor] = None
        self._routing_executor: Optional[RoutingToolExecutor] = None
        self._placement_executor: Optional[PlacementToolExecutor] = None
        
        # 初始化状态
        self._initialized = False
        
        # 注册内置工具
        self._register_builtin_tools()
    
    def initialize(
        self,
        pdk_name: str = "sky130",
        design_name: str = "top_level"
    ) -> Dict[str, Any]:
        """初始化服务器
        
        Args:
            pdk_name: PDK名称
            design_name: 设计名称
            
        Returns:
            初始化结果
        """
        try:
            # 创建会话
            session = self.state_handler.create_session(
                pdk_name=pdk_name,
                design_name=design_name
            )
            
            # 初始化工具执行器
            context = session.context
            self._device_executor = DeviceToolExecutor(context)
            self._routing_executor = RoutingToolExecutor(context)
            self._placement_executor = PlacementToolExecutor(context)
            
            self._initialized = True
            
            # 检查PDK是否成功加载
            actual_pdk = session.context.pdk_name or "none (gdsfactory not installed)"
            
            return {
                "success": True,
                "session_id": session.session_id,
                "pdk": actual_pdk,
                "design_name": design_name,
                "available_tools": len(self._tools)
            }
            
        except Exception as e:
            error_record = self.error_handler.handle_error(e)
            return {
                "success": False,
                "error": str(e),
                "error_id": error_record.error_id
            }
    
    def _register_builtin_tools(self) -> None:
        """注册内置工具"""
        # PDK管理工具
        self.register_tool(ToolDefinition(
            name="list_pdks",
            description="列出所有可用的PDK（工艺设计套件）",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            handler=self._tool_list_pdks,
            category="pdk"
        ))
        
        self.register_tool(ToolDefinition(
            name="get_pdk_info",
            description="获取当前PDK的详细信息，包括技术节点、设计规则等",
            input_schema={
                "type": "object",
                "properties": {
                    "pdk_name": {
                        "type": "string",
                        "description": "PDK名称，默认使用当前激活的PDK",
                        "enum": ["sky130", "gf180", "ihp130"]
                    }
                },
                "required": []
            },
            handler=self._tool_get_pdk_info,
            category="pdk"
        ))
        
        self.register_tool(ToolDefinition(
            name="switch_pdk",
            description="切换到另一个PDK",
            input_schema={
                "type": "object",
                "properties": {
                    "pdk_name": {
                        "type": "string",
                        "description": "目标PDK名称",
                        "enum": ["sky130", "gf180", "ihp130"]
                    }
                },
                "required": ["pdk_name"]
            },
            handler=self._tool_switch_pdk,
            category="pdk"
        ))
        
        # 上下文管理工具
        self.register_tool(ToolDefinition(
            name="get_context_status",
            description="获取当前布局上下文的状态，包括已创建的组件和连接",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            handler=self._tool_get_context_status,
            category="context"
        ))
        
        self.register_tool(ToolDefinition(
            name="list_components",
            description="列出当前布局中的所有组件",
            input_schema={
                "type": "object",
                "properties": {
                    "device_type": {
                        "type": "string",
                        "description": "过滤特定器件类型"
                    }
                },
                "required": []
            },
            handler=self._tool_list_components,
            category="context"
        ))
        
        self.register_tool(ToolDefinition(
            name="get_component_info",
            description="获取指定组件的详细信息",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "组件名称"
                    }
                },
                "required": ["name"]
            },
            handler=self._tool_get_component_info,
            category="context"
        ))
        
        # 导出工具
        self.register_tool(ToolDefinition(
            name="export_gds",
            description="导出当前布局为GDS文件",
            input_schema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "输出文件名"
                    }
                },
                "required": []
            },
            handler=self._tool_export_gds,
            category="export"
        ))
        
        # 会话管理工具
        self.register_tool(ToolDefinition(
            name="new_session",
            description="创建新的设计会话",
            input_schema={
                "type": "object",
                "properties": {
                    "pdk_name": {
                        "type": "string",
                        "description": "PDK名称",
                        "enum": ["sky130", "gf180", "ihp130"]
                    },
                    "design_name": {
                        "type": "string",
                        "description": "设计名称"
                    }
                },
                "required": []
            },
            handler=self._tool_new_session,
            category="session"
        ))
        
        self.register_tool(ToolDefinition(
            name="clear_layout",
            description="清空当前布局，移除所有组件和连接",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            handler=self._tool_clear_layout,
            category="context"
        ))
        
        # ========== 器件创建工具 ==========
        self.register_tool(ToolDefinition(
            name="create_nmos",
            description="创建NMOS晶体管",
            input_schema=NMOS_SCHEMA,
            handler=self._tool_create_nmos,
            category="device"
        ))
        
        self.register_tool(ToolDefinition(
            name="create_pmos",
            description="创建PMOS晶体管",
            input_schema=PMOS_SCHEMA,
            handler=self._tool_create_pmos,
            category="device"
        ))
        
        self.register_tool(ToolDefinition(
            name="create_via_stack",
            description="创建过孔堆栈，用于连接不同金属层",
            input_schema=VIA_STACK_SCHEMA,
            handler=self._tool_create_via_stack,
            category="device"
        ))
        
        self.register_tool(ToolDefinition(
            name="create_mimcap",
            description="创建MIM电容",
            input_schema=MIMCAP_SCHEMA,
            handler=self._tool_create_mimcap,
            category="device"
        ))
        
        self.register_tool(ToolDefinition(
            name="create_resistor",
            description="创建电阻",
            input_schema=RESISTOR_SCHEMA,
            handler=self._tool_create_resistor,
            category="device"
        ))
        
        # ========== 路由工具 ==========
        self.register_tool(ToolDefinition(
            name="smart_route",
            description="智能路由，自动选择最佳路径连接两个端口",
            input_schema=SMART_ROUTE_SCHEMA,
            handler=self._tool_smart_route,
            category="routing"
        ))
        
        self.register_tool(ToolDefinition(
            name="c_route",
            description="C形路由，用于垂直方向的端口连接",
            input_schema=C_ROUTE_SCHEMA,
            handler=self._tool_c_route,
            category="routing"
        ))
        
        self.register_tool(ToolDefinition(
            name="l_route",
            description="L形路由，用于直角转弯连接",
            input_schema=L_ROUTE_SCHEMA,
            handler=self._tool_l_route,
            category="routing"
        ))
        
        self.register_tool(ToolDefinition(
            name="straight_route",
            description="直线路由，用于同方向端口的简单连接",
            input_schema=STRAIGHT_ROUTE_SCHEMA,
            handler=self._tool_straight_route,
            category="routing"
        ))
        
        # ========== 放置工具 ==========
        self.register_tool(ToolDefinition(
            name="place_component",
            description="将组件放置到指定位置",
            input_schema=PLACE_COMPONENT_SCHEMA,
            handler=self._tool_place_component,
            category="placement"
        ))
        
        self.register_tool(ToolDefinition(
            name="align_to_port",
            description="将组件对齐到另一个组件的端口",
            input_schema=ALIGN_TO_PORT_SCHEMA,
            handler=self._tool_align_to_port,
            category="placement"
        ))
        
        self.register_tool(ToolDefinition(
            name="move_component",
            description="移动组件相对位置",
            input_schema=MOVE_COMPONENT_SCHEMA,
            handler=self._tool_move_component,
            category="placement"
        ))
        
        self.register_tool(ToolDefinition(
            name="interdigitize",
            description="互指式放置两个晶体管，用于改善匹配性（如差分对、电流镜）",
            input_schema=INTERDIGITIZE_SCHEMA,
            handler=self._tool_interdigitize,
            category="placement"
        ))
    
    # 工具处理函数
    def _tool_list_pdks(self, params: Dict[str, Any]) -> ToolResult:
        """列出可用PDK"""
        pdks = PDKManager.list_available_pdks()
        return ToolResult(
            success=True,
            data={"pdks": pdks}
        )
    
    def _tool_get_pdk_info(self, params: Dict[str, Any]) -> ToolResult:
        """获取PDK信息"""
        pdk_name = params.get("pdk_name")
        
        try:
            config = PDKManager.get_pdk_config(pdk_name)
            
            # 如果PDK已激活，获取更多信息
            if pdk_name == PDKManager.get_active_pdk_name():
                try:
                    rules = PDKManager.get_design_rules(pdk_name)
                    config["design_rules"] = rules
                except:
                    pass
            
            return ToolResult(success=True, data=config)
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _tool_switch_pdk(self, params: Dict[str, Any]) -> ToolResult:
        """切换PDK"""
        pdk_name = params.get("pdk_name")
        
        try:
            context = self.state_handler.get_context()
            if context:
                context.set_pdk(pdk_name)
            
            return ToolResult(
                success=True,
                data={"message": f"已切换到 {pdk_name} PDK"}
            )
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _tool_get_context_status(self, params: Dict[str, Any]) -> ToolResult:
        """获取上下文状态"""
        context = self.state_handler.get_context()
        
        if context is None:
            return ToolResult(
                success=False,
                error="没有活动的布局上下文"
            )
        
        return ToolResult(
            success=True,
            data=context.to_dict()
        )
    
    def _tool_list_components(self, params: Dict[str, Any]) -> ToolResult:
        """列出组件"""
        context = self.state_handler.get_context()
        
        if context is None:
            return ToolResult(
                success=False,
                error="没有活动的布局上下文"
            )
        
        device_type = params.get("device_type")
        components = context.list_components(device_type)
        
        # 获取每个组件的简要信息
        component_info = []
        for name in components:
            info = context.get_component_info(name)
            if info:
                component_info.append({
                    "name": name,
                    "device_type": info.device_type,
                    "size": info.size,
                    "ports": info.ports[:5],  # 只显示前5个端口
                    "port_count": len(info.ports)
                })
        
        return ToolResult(
            success=True,
            data={
                "count": len(components),
                "components": component_info
            }
        )
    
    def _tool_get_component_info(self, params: Dict[str, Any]) -> ToolResult:
        """获取组件信息"""
        context = self.state_handler.get_context()
        
        if context is None:
            return ToolResult(
                success=False,
                error="没有活动的布局上下文"
            )
        
        name = params.get("name")
        info = context.get_component_info(name)
        
        if info is None:
            return ToolResult(
                success=False,
                error=f"组件不存在: {name}"
            )
        
        return ToolResult(
            success=True,
            data=info.to_dict()
        )
    
    def _tool_export_gds(self, params: Dict[str, Any]) -> ToolResult:
        """导出GDS"""
        context = self.state_handler.get_context()
        
        if context is None:
            return ToolResult(
                success=False,
                error="没有活动的布局上下文"
            )
        
        try:
            filename = params.get("filename")
            filepath = context.export_gds(filename)
            
            return ToolResult(
                success=True,
                data={
                    "message": "GDS文件导出成功",
                    "path": str(filepath)
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _tool_new_session(self, params: Dict[str, Any]) -> ToolResult:
        """创建新会话"""
        try:
            session = self.state_handler.create_session(
                pdk_name=params.get("pdk_name", "sky130"),
                design_name=params.get("design_name", "top_level")
            )
            
            return ToolResult(
                success=True,
                data={
                    "session_id": session.session_id,
                    "pdk": session.context.pdk_name,
                    "design_name": session.context.design_name
                }
            )
            
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _tool_clear_layout(self, params: Dict[str, Any]) -> ToolResult:
        """清空布局"""
        context = self.state_handler.get_context()
        
        if context is None:
            return ToolResult(
                success=False,
                error="没有活动的布局上下文"
            )
        
        context.clear()
        
        return ToolResult(
            success=True,
            data={"message": "布局已清空"}
        )
    
    # ========== 器件工具处理函数 ==========
    
    def _tool_create_nmos(self, params: Dict[str, Any]) -> ToolResult:
        """创建NMOS"""
        if not self._device_executor:
            return ToolResult(success=False, error="服务器未初始化，请先调用initialize")
        
        try:
            result = self._device_executor.create_nmos(**params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _tool_create_pmos(self, params: Dict[str, Any]) -> ToolResult:
        """创建PMOS"""
        if not self._device_executor:
            return ToolResult(success=False, error="服务器未初始化，请先调用initialize")
        
        try:
            result = self._device_executor.create_pmos(**params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _tool_create_via_stack(self, params: Dict[str, Any]) -> ToolResult:
        """创建过孔堆栈"""
        if not self._device_executor:
            return ToolResult(success=False, error="服务器未初始化，请先调用initialize")
        
        try:
            result = self._device_executor.create_via_stack(**params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _tool_create_mimcap(self, params: Dict[str, Any]) -> ToolResult:
        """创建MIM电容"""
        if not self._device_executor:
            return ToolResult(success=False, error="服务器未初始化，请先调用initialize")
        
        try:
            result = self._device_executor.create_mimcap(**params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _tool_create_resistor(self, params: Dict[str, Any]) -> ToolResult:
        """创建电阻"""
        if not self._device_executor:
            return ToolResult(success=False, error="服务器未初始化，请先调用initialize")
        
        try:
            result = self._device_executor.create_resistor(**params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    # ========== 路由工具处理函数 ==========
    
    def _tool_smart_route(self, params: Dict[str, Any]) -> ToolResult:
        """智能路由"""
        if not self._routing_executor:
            return ToolResult(success=False, error="服务器未初始化，请先调用initialize")
        
        try:
            result = self._routing_executor.smart_route(**params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _tool_c_route(self, params: Dict[str, Any]) -> ToolResult:
        """C形路由"""
        if not self._routing_executor:
            return ToolResult(success=False, error="服务器未初始化，请先调用initialize")
        
        try:
            result = self._routing_executor.c_route(**params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _tool_l_route(self, params: Dict[str, Any]) -> ToolResult:
        """L形路由"""
        if not self._routing_executor:
            return ToolResult(success=False, error="服务器未初始化，请先调用initialize")
        
        try:
            result = self._routing_executor.l_route(**params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _tool_straight_route(self, params: Dict[str, Any]) -> ToolResult:
        """直线路由"""
        if not self._routing_executor:
            return ToolResult(success=False, error="服务器未初始化，请先调用initialize")
        
        try:
            result = self._routing_executor.straight_route(**params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    # ========== 放置工具处理函数 ==========
    
    def _tool_place_component(self, params: Dict[str, Any]) -> ToolResult:
        """放置组件"""
        if not self._placement_executor:
            return ToolResult(success=False, error="服务器未初始化，请先调用initialize")
        
        try:
            result = self._placement_executor.place_component(**params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _tool_align_to_port(self, params: Dict[str, Any]) -> ToolResult:
        """对齐到端口"""
        if not self._placement_executor:
            return ToolResult(success=False, error="服务器未初始化，请先调用initialize")
        
        try:
            result = self._placement_executor.align_to_port(**params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _tool_move_component(self, params: Dict[str, Any]) -> ToolResult:
        """移动组件"""
        if not self._placement_executor:
            return ToolResult(success=False, error="服务器未初始化，请先调用initialize")
        
        try:
            result = self._placement_executor.move_component(**params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _tool_interdigitize(self, params: Dict[str, Any]) -> ToolResult:
        """互指式放置"""
        if not self._placement_executor:
            return ToolResult(success=False, error="服务器未初始化，请先调用initialize")
        
        try:
            result = self._placement_executor.interdigitize(**params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    # 工具管理方法
    def register_tool(self, tool: ToolDefinition) -> None:
        """注册工具
        
        Args:
            tool: 工具定义
        """
        self._tools[tool.name] = tool
        logger.debug(f"注册工具: {tool.name}")
    
    def unregister_tool(self, name: str) -> bool:
        """取消注册工具
        
        Args:
            name: 工具名称
            
        Returns:
            是否成功
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有工具
        
        Args:
            category: 过滤类别
            
        Returns:
            工具列表
        """
        tools = []
        for tool in self._tools.values():
            if category and tool.category != category:
                continue
            
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "inputSchema": tool.input_schema
            })
        
        return tools
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义
        
        Args:
            name: 工具名称
            
        Returns:
            工具定义
        """
        return self._tools.get(name)
    
    def call_tool(self, name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """调用工具
        
        Args:
            name: 工具名称
            params: 工具参数
            
        Returns:
            执行结果
        """
        params = params or {}
        
        # 检查工具是否存在
        tool = self._tools.get(name)
        if tool is None:
            return {
                "success": False,
                "error": f"未知工具: {name}",
                "available_tools": list(self._tools.keys())
            }
        
        try:
            # 执行工具
            result = tool.handler(params)
            return result.to_dict()
            
        except LayoutError as e:
            error_record = self.error_handler.handle_error(e)
            return {
                "success": False,
                "error": str(e),
                "error_id": error_record.error_id,
                "category": e.category.value,
                "recovery_suggestion": error_record.recovery_action
            }
            
        except Exception as e:
            error_record = self.error_handler.handle_error(e)
            return {
                "success": False,
                "error": str(e),
                "error_id": error_record.error_id
            }
    
    # MCP协议方法
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理MCP请求
        
        Args:
            request: MCP请求对象
            
        Returns:
            MCP响应对象
        """
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        response = {
            "jsonrpc": "2.0",
            "id": request_id
        }
        
        try:
            if method == "initialize":
                result = self.initialize(**params)
                response["result"] = result
                
            elif method == "tools/list":
                result = self.list_tools(params.get("category"))
                response["result"] = {"tools": result}
                
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_params = params.get("arguments", {})
                result = self.call_tool(tool_name, tool_params)
                response["result"] = result
                
            else:
                response["error"] = {
                    "code": -32601,
                    "message": f"未知方法: {method}"
                }
                
        except Exception as e:
            response["error"] = {
                "code": -32000,
                "message": str(e)
            }
        
        return response
    
    def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息
        
        Returns:
            服务器信息
        """
        return {
            "name": "Analog Layout Agent MCP Server",
            "version": "0.1.0",
            "protocol_version": "2024-11-05",
            "capabilities": {
                "tools": True,
                "prompts": False,
                "resources": False
            },
            "tool_count": len(self._tools),
            "tool_categories": list(set(t.category for t in self._tools.values()))
        }


# 创建全局实例
_server_instance: Optional[MCPServer] = None


def get_server() -> MCPServer:
    """获取MCP Server单例
    
    Returns:
        MCPServer实例
    """
    global _server_instance
    if _server_instance is None:
        _server_instance = MCPServer()
    return _server_instance


def main():
    """主入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analog Layout Agent MCP Server")
    parser.add_argument("--pdk", default="sky130", help="默认PDK")
    parser.add_argument("--design", default="top_level", help="默认设计名称")
    parser.add_argument("--stdio", action="store_true", help="使用STDIO模式")
    
    args = parser.parse_args()
    
    server = get_server()
    
    if args.stdio:
        # STDIO模式：从标准输入读取请求，输出到标准输出
        print(json.dumps(server.get_server_info()), file=sys.stderr)
        
        # 初始化
        init_result = server.initialize(pdk_name=args.pdk, design_name=args.design)
        print(json.dumps(init_result), file=sys.stderr)
        
        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                response = server.handle_request(request)
                print(json.dumps(response))
                sys.stdout.flush()
            except json.JSONDecodeError:
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": "解析错误"}
                }))
                sys.stdout.flush()
    else:
        # 交互模式
        print("Analog Layout Agent MCP Server")
        print(f"PDK: {args.pdk}")
        print(f"设计: {args.design}")
        print("-" * 40)
        
        init_result = server.initialize(pdk_name=args.pdk, design_name=args.design)
        print(f"初始化结果: {init_result}")
        
        print("\n可用工具:")
        for tool in server.list_tools():
            print(f"  - {tool['name']}: {tool['description']}")


if __name__ == "__main__":
    main()
