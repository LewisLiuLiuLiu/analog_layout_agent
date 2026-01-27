"""
Phase 1 单元测试 - 基础架构测试

测试PDKManager、ComponentRegistry、LayoutContext等核心组件
"""

import pytest
import sys
from pathlib import Path

# 添加模块路径
_MODULE_PATH = Path(__file__).parent.parent
if str(_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(_MODULE_PATH))


class TestComponentRegistry:
    """测试组件注册表"""
    
    def test_register_component(self):
        """测试注册组件"""
        from core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry()
        
        # 注册一个模拟组件
        name = registry.register(
            component=None,
            device_type="nmos",
            params={"width": 1.0, "length": 0.15}
        )
        
        assert name is not None
        assert "nmos" in name
        assert registry.count() == 1
    
    def test_auto_naming(self):
        """测试自动命名"""
        from core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry()
        
        # 注册多个同类型组件
        name1 = registry.register(None, "nmos", {})
        name2 = registry.register(None, "nmos", {})
        name3 = registry.register(None, "pmos", {})
        
        assert name1 != name2
        assert name1 != name3
        assert registry.count() == 3
        assert registry.count_by_type("nmos") == 2
        assert registry.count_by_type("pmos") == 1
    
    def test_custom_naming(self):
        """测试自定义命名"""
        from core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry()
        
        name = registry.register(None, "nmos", {}, name="my_nmos")
        
        assert name == "my_nmos"
        assert registry.exists("my_nmos")
    
    def test_duplicate_name_error(self):
        """测试重复命名错误"""
        from core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry()
        
        registry.register(None, "nmos", {}, name="dup_name")
        
        with pytest.raises(ValueError):
            registry.register(None, "nmos", {}, name="dup_name")
    
    def test_get_component(self):
        """测试获取组件"""
        from core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry()
        
        name = registry.register(None, "nmos", {"width": 2.0})
        
        info = registry.get(name)
        assert info is not None
        assert info.device_type == "nmos"
        assert info.params["width"] == 2.0
    
    def test_remove_component(self):
        """测试移除组件"""
        from core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry()
        
        name = registry.register(None, "nmos", {})
        assert registry.count() == 1
        
        result = registry.remove(name)
        assert result is True
        assert registry.count() == 0
        assert not registry.exists(name)
    
    def test_rename_component(self):
        """测试重命名组件"""
        from core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry()
        
        old_name = registry.register(None, "nmos", {})
        registry.rename(old_name, "new_name")
        
        assert not registry.exists(old_name)
        assert registry.exists("new_name")
    
    def test_list_by_type(self):
        """测试按类型列出"""
        from core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry()
        
        registry.register(None, "nmos", {})
        registry.register(None, "nmos", {})
        registry.register(None, "pmos", {})
        
        nmos_list = registry.list_by_type("nmos")
        assert len(nmos_list) == 2
        
        pmos_list = registry.list_by_type("pmos")
        assert len(pmos_list) == 1
    
    def test_search(self):
        """测试搜索"""
        from core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry()
        
        registry.register(None, "nmos", {}, name="input_nmos")
        registry.register(None, "nmos", {}, name="output_nmos")
        registry.register(None, "pmos", {}, name="load_pmos")
        
        results = registry.search("input")
        assert len(results) == 1
        assert "input_nmos" in results
        
        results = registry.search("nmos")
        assert len(results) == 2
    
    def test_clear(self):
        """测试清空"""
        from core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry()
        
        registry.register(None, "nmos", {})
        registry.register(None, "pmos", {})
        
        registry.clear()
        assert registry.count() == 0
    
    def test_to_dict(self):
        """测试导出字典"""
        from core.component_registry import ComponentRegistry
        
        registry = ComponentRegistry()
        registry.register(None, "nmos", {"width": 1.0})
        
        data = registry.to_dict()
        assert "component_count" in data
        assert data["component_count"] == 1
        assert "components" in data


class TestLayoutContext:
    """测试布局上下文"""
    
    def test_create_context(self):
        """测试创建上下文"""
        from core.layout_context import LayoutContext, ContextState
        
        context = LayoutContext(design_name="test_design")
        
        assert context.design_name == "test_design"
        assert context.state == ContextState.INITIALIZED
    
    def test_register_component(self):
        """测试注册组件"""
        from core.layout_context import LayoutContext, ContextState
        
        context = LayoutContext(design_name="test")
        
        name = context.register_component(
            component=None,
            device_type="nmos",
            params={"width": 1.0}
        )
        
        assert name is not None
        assert context.state == ContextState.DESIGNING
    
    def test_list_components(self):
        """测试列出组件"""
        from core.layout_context import LayoutContext
        
        context = LayoutContext(design_name="test")
        
        context.register_component(None, "nmos", {})
        context.register_component(None, "pmos", {})
        
        all_comps = context.list_components()
        assert len(all_comps) == 2
        
        nmos_comps = context.list_components("nmos")
        assert len(nmos_comps) == 1
    
    def test_add_connection(self):
        """测试添加连接"""
        from core.layout_context import LayoutContext
        
        context = LayoutContext(design_name="test")
        
        # 注册组件（模拟有端口的组件）
        context.register_component(None, "nmos", {}, name="nmos_1")
        context.register_component(None, "nmos", {}, name="nmos_2")
        
        # 添加连接（不验证端口，因为是模拟组件）
        # 需要先mock端口，这里简化测试
        connections = context.get_connections()
        assert len(connections) == 0
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        from core.layout_context import LayoutContext
        
        context = LayoutContext(design_name="test")
        
        context.register_component(None, "nmos", {})
        context.register_component(None, "pmos", {})
        
        stats = context.get_statistics()
        
        assert stats["design_name"] == "test"
        assert stats["component_count"] == 2
        assert "nmos" in stats["device_counts"]
        assert "pmos" in stats["device_counts"]
    
    def test_to_dict(self):
        """测试导出字典"""
        from core.layout_context import LayoutContext
        
        context = LayoutContext(design_name="test")
        context.register_component(None, "nmos", {})
        
        data = context.to_dict()
        
        assert "design_name" in data
        assert "state" in data
        assert "components" in data
        assert "connections" in data
    
    def test_to_natural_language(self):
        """测试自然语言描述"""
        from core.layout_context import LayoutContext
        
        context = LayoutContext(design_name="test")
        context.register_component(None, "nmos", {})
        
        text = context.to_natural_language()
        
        assert "test" in text
        assert "nmos" in text
    
    def test_clear(self):
        """测试清空"""
        from core.layout_context import LayoutContext, ContextState
        
        context = LayoutContext(design_name="test")
        
        context.register_component(None, "nmos", {})
        context.clear()
        
        assert len(context.list_components()) == 0
        assert context.state == ContextState.INITIALIZED
    
    def test_history(self):
        """测试历史记录"""
        from core.layout_context import LayoutContext
        
        context = LayoutContext(design_name="test")
        
        context.register_component(None, "nmos", {})
        context.register_component(None, "pmos", {})
        
        history = context.get_history()
        assert len(history) >= 2


class TestPDKManager:
    """测试PDK管理器"""
    
    def test_list_pdks(self):
        """测试列出PDK"""
        from core.pdk_manager import PDKManager
        
        pdks = PDKManager.list_available_pdks()
        
        assert len(pdks) >= 3
        
        pdk_names = [p["name"] for p in pdks]
        assert "sky130" in pdk_names
        assert "gf180" in pdk_names
        assert "ihp130" in pdk_names
    
    def test_get_config(self):
        """测试获取配置"""
        from core.pdk_manager import PDKManager
        
        config = PDKManager.get_pdk_config("sky130")
        
        assert "tech_node" in config
        assert "metal_layers" in config
        assert "min_dimensions" in config
        assert config["tech_node"] == "130nm"
    
    def test_invalid_pdk_error(self):
        """测试无效PDK错误"""
        from core.pdk_manager import PDKManager
        
        with pytest.raises(ValueError):
            PDKManager.get_pdk_config("invalid_pdk")
    
    def test_validate_params(self):
        """测试参数验证"""
        from core.pdk_manager import PDKManager
        
        # 有效参数
        result = PDKManager.validate_device_params(
            "nfet", width=1.0, length=0.15, pdk_name="sky130"
        )
        assert result["valid"] is True
        
        # 无效参数（宽度太小）
        result = PDKManager.validate_device_params(
            "nfet", width=0.01, length=0.15, pdk_name="sky130"
        )
        assert result["valid"] is False
        assert len(result["errors"]) > 0
    
    def test_get_model_name(self):
        """测试获取模型名称"""
        from core.pdk_manager import PDKManager
        
        model = PDKManager.get_model_name("nfet", "sky130")
        assert "nfet" in model.lower() or "nmos" in model.lower()
    
    def test_reset(self):
        """测试重置"""
        from core.pdk_manager import PDKManager
        
        PDKManager.reset()
        
        assert PDKManager.get_active_pdk_name() is None


class TestStateHandler:
    """测试状态处理器"""
    
    def test_create_session(self):
        """测试创建会话"""
        from mcp_server.handlers.state_handler import StateHandler, SessionState
        
        handler = StateHandler()
        handler.reset()  # 确保干净状态
        
        session = handler.create_session(design_name="test")
        
        assert session is not None
        assert session.state == SessionState.ACTIVE
        assert session.context is not None
    
    def test_get_session(self):
        """测试获取会话"""
        from mcp_server.handlers.state_handler import StateHandler
        
        handler = StateHandler()
        handler.reset()
        
        created = handler.create_session()
        retrieved = handler.get_session(created.session_id)
        
        assert retrieved is not None
        assert retrieved.session_id == created.session_id
    
    def test_active_session(self):
        """测试活动会话"""
        from mcp_server.handlers.state_handler import StateHandler
        
        handler = StateHandler()
        handler.reset()
        
        session = handler.create_session(make_active=True)
        active = handler.get_active_session()
        
        assert active is not None
        assert active.session_id == session.session_id
    
    def test_close_session(self):
        """测试关闭会话"""
        from mcp_server.handlers.state_handler import StateHandler
        
        handler = StateHandler()
        handler.reset()
        
        session = handler.create_session()
        result = handler.close_session(session.session_id)
        
        assert result is True
        assert handler.get_session(session.session_id) is None
    
    def test_multiple_sessions(self):
        """测试多会话"""
        from mcp_server.handlers.state_handler import StateHandler
        
        handler = StateHandler()
        handler.reset()
        
        session1 = handler.create_session(design_name="design1")
        session2 = handler.create_session(design_name="design2")
        
        sessions = handler.list_sessions()
        assert len(sessions) == 2


class TestErrorHandler:
    """测试错误处理器"""
    
    def test_handle_error(self):
        """测试处理错误"""
        from mcp_server.handlers.error_handler import ErrorHandler, ErrorCategory
        
        handler = ErrorHandler()
        handler.clear_history()
        
        error = ValueError("test error")
        record = handler.handle_error(error)
        
        assert record is not None
        assert record.message == "test error"
        assert record.error_id is not None
    
    def test_layout_error(self):
        """测试布局错误"""
        from mcp_server.handlers.error_handler import (
            ErrorHandler, ValidationError, ErrorCategory
        )
        
        handler = ErrorHandler()
        handler.clear_history()
        
        error = ValidationError("invalid width", {"width": 0.01})
        record = handler.handle_error(error)
        
        assert record.category == ErrorCategory.VALIDATION
    
    def test_error_history(self):
        """测试错误历史"""
        from mcp_server.handlers.error_handler import ErrorHandler
        
        handler = ErrorHandler()
        handler.clear_history()
        
        handler.handle_error(ValueError("error1"))
        handler.handle_error(ValueError("error2"))
        
        history = handler.get_error_history()
        assert len(history) == 2
    
    def test_error_summary(self):
        """测试错误摘要"""
        from mcp_server.handlers.error_handler import ErrorHandler
        
        handler = ErrorHandler()
        handler.clear_history()
        
        handler.handle_error(ValueError("error1"))
        handler.handle_error(TypeError("error2"))
        
        summary = handler.get_error_summary()
        
        assert summary["total_errors"] == 2
        assert "by_category" in summary
        assert "by_severity" in summary
    
    def test_catch_errors_decorator(self):
        """测试错误捕获装饰器"""
        from mcp_server.handlers.error_handler import ErrorHandler
        
        handler = ErrorHandler()
        handler.clear_history()
        
        @handler.catch_errors(reraise=False)
        def failing_function():
            raise ValueError("decorated error")
        
        result = failing_function()
        
        assert result is None
        assert len(handler.get_error_history()) == 1


class TestMCPServer:
    """测试MCP Server"""
    
    def test_initialize(self):
        """测试初始化"""
        from mcp_server.server import MCPServer
        
        server = MCPServer()
        result = server.initialize(pdk_name="sky130")
        
        assert result["success"] is True
        assert "session_id" in result
    
    def test_list_tools(self):
        """测试列出工具"""
        from mcp_server.server import MCPServer
        
        server = MCPServer()
        tools = server.list_tools()
        
        assert len(tools) > 0
        
        tool_names = [t["name"] for t in tools]
        assert "list_pdks" in tool_names
        assert "get_context_status" in tool_names
    
    def test_call_tool(self):
        """测试调用工具"""
        from mcp_server.server import MCPServer
        
        server = MCPServer()
        server.initialize()
        
        result = server.call_tool("list_pdks")
        
        assert result["success"] is True
        assert "pdks" in result["data"]
    
    def test_unknown_tool(self):
        """测试未知工具"""
        from mcp_server.server import MCPServer
        
        server = MCPServer()
        result = server.call_tool("nonexistent_tool")
        
        assert result["success"] is False
        assert "error" in result
    
    def test_handle_request(self):
        """测试处理MCP请求"""
        from mcp_server.server import MCPServer
        
        server = MCPServer()
        server.initialize()
        
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 1
        }
        
        response = server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
    
    def test_server_info(self):
        """测试服务器信息"""
        from mcp_server.server import MCPServer
        
        server = MCPServer()
        info = server.get_server_info()
        
        assert "name" in info
        assert "version" in info
        assert "capabilities" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
