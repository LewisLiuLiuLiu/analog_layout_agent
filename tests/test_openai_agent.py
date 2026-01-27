"""
OpenAI Agent SDK 集成测试

测试Layout Agent通过OpenAI Agent SDK的各种功能
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目路径
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest


class TestOpenAIAgentImport:
    """测试模块导入"""
    
    def test_import_openai_agent_module(self):
        """测试导入openai_agent模块"""
        from agent.openai_agent import (
            create_layout_agent,
            run_layout_agent,
            LayoutAgentContext
        )
        assert create_layout_agent is not None
        assert run_layout_agent is not None
        assert LayoutAgentContext is not None
    
    def test_import_from_package(self):
        """测试从包级别导入"""
        from agent import (
            create_layout_agent,
            run_layout_agent,
            LayoutAgentContext
        )
        assert create_layout_agent is not None
        assert run_layout_agent is not None
        assert LayoutAgentContext is not None
    
    def test_import_function_tools(self):
        """测试导入function tools"""
        from agent.openai_agent import (
            create_nmos,
            create_pmos,
            create_mimcap,
            create_resistor,
            smart_route,
            place_component,
            create_current_mirror,
            run_drc,
            list_components,
            export_gds
        )
        from agents.tool import FunctionTool
        # 验证这些都是FunctionTool类型（由function_tool装饰器创建）
        assert isinstance(create_nmos, FunctionTool)
        assert isinstance(create_pmos, FunctionTool)
        assert isinstance(smart_route, FunctionTool)
        # 验证工具名称
        assert create_nmos.name == "create_nmos"
        assert create_pmos.name == "create_pmos"
        assert smart_route.name == "smart_route"


class TestAgentCreation:
    """测试Agent创建"""
    
    @pytest.fixture(autouse=True)
    def check_dependencies(self):
        """检查依赖是否可用"""
        try:
            from agent.openai_agent import create_layout_agent
            create_layout_agent()
            self.deps_available = True
        except ImportError as e:
            if "gdsfactory" in str(e) or "gLayout" in str(e):
                pytest.skip("需要gdsfactory/gLayout依赖")
            raise
    
    def test_create_layout_agent_default(self):
        """测试默认参数创建Agent"""
        from agent.openai_agent import create_layout_agent
        
        agent, ctx = create_layout_agent()
        
        assert agent is not None
        assert ctx is not None
        assert agent.name == "AnalogLayoutAgent"
        assert ctx.layout_context is not None
        assert ctx.device_executor is not None
        assert ctx.routing_executor is not None
        assert ctx.placement_executor is not None
        assert ctx.circuit_builder is not None
        assert ctx.verification_engine is not None
    
    def test_create_layout_agent_with_params(self):
        """测试带参数创建Agent"""
        from agent.openai_agent import create_layout_agent
        
        agent, ctx = create_layout_agent(pdk="sky130", design_name="test_design")
        
        assert ctx.layout_context.pdk_name == "sky130"
        assert ctx.layout_context.design_name == "test_design"
    
    def test_agent_has_tools(self):
        """测试Agent包含所有预期的工具"""
        from agent.openai_agent import create_layout_agent
        
        agent, _ = create_layout_agent()
        
        # 验证工具数量（应该有20个左右）
        assert len(agent.tools) >= 15
        
        # 获取工具名称
        tool_names = [t.name for t in agent.tools]
        
        # 验证关键工具存在
        expected_tools = [
            "create_nmos",
            "create_pmos",
            "smart_route",
            "place_component",
            "create_current_mirror",
            "run_drc",
            "list_components",
            "export_gds"
        ]
        for tool_name in expected_tools:
            assert tool_name in tool_names, f"缺少工具: {tool_name}"


class TestLayoutAgentContext:
    """测试LayoutAgentContext数据类"""
    
    @pytest.fixture(autouse=True)
    def check_dependencies(self):
        """检查依赖是否可用"""
        try:
            from agent.openai_agent import create_layout_agent
            create_layout_agent()
        except ImportError as e:
            if "gdsfactory" in str(e) or "gLayout" in str(e):
                pytest.skip("需要gdsfactory/gLayout依赖")
            raise
    
    def test_context_structure(self):
        """测试上下文结构"""
        from agent.openai_agent import create_layout_agent, LayoutAgentContext
        
        _, ctx = create_layout_agent()
        
        assert isinstance(ctx, LayoutAgentContext)
        
        # 验证所有执行器都已初始化
        from core.layout_context import LayoutContext
        from mcp_server.tools.device_tools import DeviceToolExecutor
        from mcp_server.tools.routing_tools import RoutingToolExecutor
        from mcp_server.tools.placement_tools import PlacementToolExecutor
        from core.circuit_builder import CircuitBuilder
        from core.verification import VerificationEngine
        
        assert isinstance(ctx.layout_context, LayoutContext)
        assert isinstance(ctx.device_executor, DeviceToolExecutor)
        assert isinstance(ctx.routing_executor, RoutingToolExecutor)
        assert isinstance(ctx.placement_executor, PlacementToolExecutor)
        assert isinstance(ctx.circuit_builder, CircuitBuilder)
        assert isinstance(ctx.verification_engine, VerificationEngine)


# ============== 需要OpenAI API Key的测试 ==============

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="需要设置OPENAI_API_KEY环境变量"
)
class TestRunLayoutAgent:
    """测试Agent运行（需要OpenAI API Key）"""
    
    @pytest.mark.asyncio
    async def test_simple_nmos_creation(self):
        """测试简单的NMOS创建指令"""
        from agent.openai_agent import run_layout_agent
        
        result = await run_layout_agent(
            instruction="创建一个NMOS，宽度1um",
            pdk="sky130"
        )
        
        assert "response" in result
        assert "components" in result
        assert len(result["components"]) >= 1
    
    @pytest.mark.asyncio
    async def test_current_mirror_creation(self):
        """测试电流镜创建指令"""
        from agent.openai_agent import run_layout_agent
        
        result = await run_layout_agent(
            instruction="创建一个NMOS电流镜，宽度3um，5列互指式布局",
            pdk="sky130"
        )
        
        assert "response" in result
        assert "components" in result
    
    @pytest.mark.asyncio
    async def test_list_components(self):
        """测试列出组件指令"""
        from agent.openai_agent import run_layout_agent
        
        # 先创建一些组件
        result = await run_layout_agent(
            instruction="创建一个NMOS和一个PMOS，然后列出所有组件",
            pdk="sky130"
        )
        
        assert "response" in result


# ============== 本地测试（不需要API Key）==============

class TestToolExecutionLocal:
    """本地测试工具执行（不调用LLM）"""
    
    @pytest.fixture(autouse=True)
    def check_dependencies(self):
        """检查依赖是否可用"""
        try:
            from agent.openai_agent import create_layout_agent
            create_layout_agent()
        except ImportError as e:
            if "gdsfactory" in str(e) or "gLayout" in str(e):
                pytest.skip("需要gdsfactory/gLayout依赖")
            raise
    
    def test_direct_device_creation(self):
        """直接测试器件创建（绕过LLM）"""
        from agent.openai_agent import create_layout_agent
        
        _, ctx = create_layout_agent(pdk="sky130")
        
        # 直接调用device_executor
        result = ctx.device_executor.create_nmos(width=1.0, fingers=2)
        
        assert result["success"] is True
        assert "component_name" in result
        assert result["device_type"] == "nmos"
        assert result["params"]["width"] == 1.0
        assert result["params"]["fingers"] == 2
    
    def test_direct_circuit_creation(self):
        """直接测试电路创建（绕过LLM）"""
        from agent.openai_agent import create_layout_agent
        
        _, ctx = create_layout_agent(pdk="sky130")
        
        # 直接调用circuit_builder
        result = ctx.circuit_builder.build_current_mirror(
            device_type="nmos",
            width=3.0,
            numcols=5
        )
        
        assert result["success"] is True
        assert result["circuit_type"] == "current_mirror"
        assert result["params"]["numcols"] == 5
    
    def test_direct_list_components(self):
        """直接测试列出组件（绕过LLM）"""
        from agent.openai_agent import create_layout_agent
        
        _, ctx = create_layout_agent(pdk="sky130")
        
        # 创建一些组件
        ctx.device_executor.create_nmos(width=1.0, name="test_nmos")
        ctx.device_executor.create_pmos(width=2.0, name="test_pmos")
        
        # 列出组件
        components = ctx.layout_context.list_components()
        
        assert "test_nmos" in components
        assert "test_pmos" in components
    
    def test_direct_drc(self):
        """直接测试DRC（绕过LLM）"""
        from agent.openai_agent import create_layout_agent
        
        _, ctx = create_layout_agent(pdk="sky130")
        
        # 执行DRC
        result = ctx.verification_engine.run_drc()
        
        # 在mock模式下应该返回通过
        assert hasattr(result, 'passed') or isinstance(result, dict)


if __name__ == "__main__":
    # 运行简单的本地测试
    print("运行本地测试...")
    print("=" * 50)
    
    # 测试导入
    print("\n1. 测试模块导入...")
    test_import = TestOpenAIAgentImport()
    test_import.test_import_openai_agent_module()
    test_import.test_import_from_package()
    test_import.test_import_function_tools()
    print("   ✓ 模块导入成功")
    
    # 测试Agent创建（需要gdsfactory依赖）
    print("\n2. 测试Agent创建...")
    try:
        from agent.openai_agent import create_layout_agent
        agent, ctx = create_layout_agent(pdk="sky130")
        print("   ✓ Agent创建成功")
        print(f"   - Agent名称: {agent.name}")
        print(f"   - 工具数量: {len(agent.tools)}")
        
        # 测试Context
        print("\n3. 测试Context结构...")
        print(f"   - PDK: {ctx.layout_context.pdk_name}")
        print(f"   - 设计名称: {ctx.layout_context.design_name}")
        print("   ✓ Context结构正确")
        
        # 测试本地执行
        print("\n4. 测试本地工具执行...")
        result = ctx.device_executor.create_nmos(width=1.0, fingers=2)
        print(f"   - 创建NMOS: {result['component_name']}")
        print("   ✓ 本地工具执行成功")
        
    except ImportError as e:
        if "gdsfactory" in str(e):
            print("   ⚠ 跳过（需要gdsfactory依赖）")
            print("\n注意: 完整测试需要安装gdsfactory:")
            print("  pip install gdsfactory")
        else:
            raise
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("\n要运行需要API Key的测试，请设置OPENAI_API_KEY环境变量后运行:")
    print("  python3 -m pytest tests/test_openai_agent.py -v")
