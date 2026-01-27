"""
集成测试 - 端到端流程测试

测试 MCP Server、OpenAI Agent、DRC Advisor 的完整集成
"""

import pytest
import sys
from pathlib import Path

# 添加模块路径
_MODULE_PATH = Path(__file__).parent.parent
if str(_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(_MODULE_PATH))


class TestMCPServerIntegration:
    """MCP Server 集成测试"""
    
    def test_server_initialization(self):
        """测试服务器初始化"""
        from mcp_server.server import MCPServer
        
        server = MCPServer()
        result = server.initialize(pdk_name="sky130", design_name="test_init")
        
        assert result["success"] is True
        assert "session_id" in result
        assert result["available_tools"] > 0
    
    def test_tool_listing(self):
        """测试工具列表"""
        from mcp_server.server import MCPServer
        
        server = MCPServer()
        server.initialize(pdk_name="sky130")
        
        tools = server.list_tools()
        tool_names = [t["name"] for t in tools]
        
        # 检查核心工具存在
        assert "create_nmos" in tool_names
        assert "create_pmos" in tool_names
        assert "smart_route" in tool_names
        assert "place_component" in tool_names
        assert "interdigitize" in tool_names
    
    def test_device_creation_via_mcp(self):
        """测试通过MCP创建器件"""
        from mcp_server.server import MCPServer
        
        server = MCPServer()
        server.initialize(pdk_name="sky130", design_name="test_device")
        
        # 创建 NMOS
        result = server.call_tool("create_nmos", {
            "width": 1.0,
            "fingers": 2
        })
        
        assert result["success"] is True
        assert "data" in result
    
    def test_placement_via_mcp(self):
        """测试通过MCP放置组件"""
        from mcp_server.server import MCPServer
        
        server = MCPServer()
        server.initialize(pdk_name="sky130", design_name="test_place")
        
        # 先创建器件
        server.call_tool("create_nmos", {"width": 1.0, "name": "m1"})
        
        # 放置组件
        result = server.call_tool("place_component", {
            "component_name": "m1",
            "x": 10.0,
            "y": 20.0
        })
        
        assert result["success"] is True
    
    def test_mcp_request_handling(self):
        """测试MCP协议请求处理"""
        from mcp_server.server import MCPServer
        
        server = MCPServer()
        server.initialize(pdk_name="sky130")
        
        # 模拟 MCP 请求
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        
        response = server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "tools" in response["result"]


class TestOpenAIAgentIntegration:
    """OpenAI Agent 集成测试"""
    
    def test_agent_creation(self):
        """测试Agent创建"""
        from agent.openai_agent import create_layout_agent
        
        agent, ctx = create_layout_agent(pdk="sky130", design_name="test_agent")
        
        assert agent is not None
        assert agent.name == "AnalogLayoutAgent"
        assert len(agent.tools) >= 20
        assert ctx.mcp_server._initialized is True
    
    def test_agent_tool_call(self):
        """测试Agent工具调用"""
        from agent.openai_agent import create_layout_agent
        
        agent, ctx = create_layout_agent(pdk="sky130", design_name="test_tool")
        
        # 通过 context 调用工具
        result = ctx.call_tool("create_nmos", {"width": 2.0, "fingers": 1})
        
        assert result["success"] is True
    
    def test_agent_has_drc_tools(self):
        """测试Agent包含DRC工具"""
        from agent.openai_agent import create_layout_agent
        
        agent, ctx = create_layout_agent(pdk="sky130")
        
        tool_names = [t.name for t in agent.tools]
        
        assert "run_drc" in tool_names
        assert "get_drc_fix_suggestions" in tool_names


class TestDRCAdvisorIntegration:
    """DRC Advisor 集成测试"""
    
    def test_advisor_initialization(self):
        """测试Advisor初始化"""
        from core.drc_advisor import DRCAdvisor
        
        advisor = DRCAdvisor(pdk_name="sky130")
        
        assert advisor.pdk_name == "sky130"
        assert "met1_spacing" in advisor.rules
    
    def test_advisor_spacing_suggestion(self):
        """测试间距违规建议"""
        from core.drc_advisor import DRCAdvisor, analyze_drc_result
        from core.verification import DRCResult, DRCViolation
        
        mock_result = DRCResult(
            passed=False,
            violations=[
                DRCViolation(
                    rule="met1.spacing",
                    category="spacing",
                    location=(1.0, 2.0),
                    description="Minimum spacing violation",
                    severity="error"
                )
            ],
            report_path="/tmp/test.rpt"
        )
        
        analysis = analyze_drc_result(mock_result, "sky130")
        
        assert analysis["violation_count"] == 1
        assert analysis["suggestion_count"] == 1
        assert analysis["suggestions"][0]["action"] == "increase_spacing"
    
    def test_advisor_enclosure_suggestion(self):
        """测试包络违规建议"""
        from core.drc_advisor import DRCAdvisor, analyze_drc_result
        from core.verification import DRCResult, DRCViolation
        
        mock_result = DRCResult(
            passed=False,
            violations=[
                DRCViolation(
                    rule="via.enclosure",
                    category="enclosure",
                    location=(3.0, 4.0),
                    description="Via enclosure too small",
                    severity="warning"
                )
            ],
            report_path="/tmp/test.rpt"
        )
        
        analysis = analyze_drc_result(mock_result, "sky130")
        
        assert analysis["suggestions"][0]["action"] == "increase_enclosure"
    
    def test_advisor_summary_generation(self):
        """测试摘要生成"""
        from core.drc_advisor import DRCAdvisor
        from core.verification import DRCResult, DRCViolation
        
        advisor = DRCAdvisor(pdk_name="sky130")
        
        mock_result = DRCResult(
            passed=False,
            violations=[
                DRCViolation("met1.spacing", "spacing", (0, 0), "test", "error"),
                DRCViolation("met2.spacing", "spacing", (1, 1), "test", "error"),
                DRCViolation("via.enc", "enclosure", (2, 2), "test", "warning"),
            ],
            report_path="/tmp/test.rpt"
        )
        
        suggestions = advisor.analyze(mock_result)
        summary = advisor.get_summary(suggestions)
        
        assert "增加间距" in summary
        assert "增加包络" in summary


class TestCircuitBuilderIntegration:
    """电路构建器集成测试"""
    
    def test_current_mirror_creation(self):
        """测试电流镜创建"""
        from core.layout_context import LayoutContext
        from core.circuit_builder import CircuitBuilder
        
        ctx = LayoutContext(pdk_name="sky130", design_name="test_mirror")
        builder = CircuitBuilder(ctx)
        
        result = builder.build_current_mirror(
            device_type="nmos",
            width=3.0,
            numcols=3
        )
        
        assert result["success"] is True
        assert "component_name" in result
    
    def test_diff_pair_creation(self):
        """测试差分对创建"""
        from core.layout_context import LayoutContext
        from core.circuit_builder import CircuitBuilder
        
        ctx = LayoutContext(pdk_name="sky130", design_name="test_diff")
        builder = CircuitBuilder(ctx)
        
        result = builder.build_diff_pair(
            device_type="nmos",
            width=5.0,
            numcols=3  # glayout requires numcols parameter
        )
        
        assert result["success"] is True


class TestEndToEndWorkflow:
    """端到端工作流测试"""
    
    def test_simple_inverter_workflow(self):
        """测试简单反相器设计流程"""
        from mcp_server.server import MCPServer
        
        server = MCPServer()
        server.initialize(pdk_name="sky130", design_name="inverter")
        
        # 1. 创建 NMOS
        result = server.call_tool("create_nmos", {
            "width": 1.0,
            "fingers": 1,
            "name": "mn"
        })
        assert result["success"] is True
        
        # 2. 创建 PMOS
        result = server.call_tool("create_pmos", {
            "width": 2.0,
            "fingers": 1,
            "name": "mp"
        })
        assert result["success"] is True
        
        # 3. 放置 PMOS
        result = server.call_tool("place_component", {
            "component_name": "mp",
            "x": 0,
            "y": 10
        })
        assert result["success"] is True
        
        # 4. 列出组件
        result = server.call_tool("list_components", {})
        assert result["success"] is True
        assert result["data"]["count"] == 2
    
    def test_mcp_to_agent_consistency(self):
        """测试MCP和Agent工具一致性"""
        from mcp_server.server import MCPServer
        from agent.openai_agent import create_layout_agent
        
        # MCP Server 工具
        server = MCPServer()
        server.initialize(pdk_name="sky130")
        mcp_tools = set(t["name"] for t in server.list_tools())
        
        # Agent 工具（通过 MCP 调用的那些）
        agent, ctx = create_layout_agent(pdk="sky130")
        
        # Agent 应该能调用的核心工具
        core_tools = [
            "create_nmos", "create_pmos", "create_mimcap", "create_resistor",
            "smart_route", "c_route", "l_route", "straight_route",
            "place_component", "move_component", "align_to_port",
            "list_components", "get_component_info", "export_gds"
        ]
        
        for tool in core_tools:
            assert tool in mcp_tools, f"Tool {tool} missing from MCP Server"


class TestMultiPDKSupport:
    """多PDK支持测试"""
    
    @pytest.mark.parametrize("pdk_name", ["sky130", "gf180", "ihp130"])
    def test_pdk_initialization(self, pdk_name):
        """测试各PDK初始化"""
        from mcp_server.server import MCPServer
        
        server = MCPServer()
        result = server.initialize(pdk_name=pdk_name, design_name=f"test_{pdk_name}")
        
        # 即使PDK不完全支持，初始化应该不会崩溃
        assert "success" in result
    
    def test_drc_advisor_pdk_rules(self):
        """测试DRC Advisor PDK规则"""
        from core.drc_advisor import DRCAdvisor
        
        for pdk in ["sky130", "gf180", "ihp130"]:
            advisor = DRCAdvisor(pdk_name=pdk)
            assert advisor.rules is not None
            assert len(advisor.rules) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
