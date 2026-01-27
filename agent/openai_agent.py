"""
OpenAI Agent SDK 集成模块

使用OpenAI Agent SDK构建Layout Agent，通过LLM实现智能指令解析和工具调用。
统一通过 MCP Server 的 call_tool() 作为单一工具调用入口（Single Source of Truth）。
"""

import sys
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from agents import Agent, Runner, function_tool
from agents.run_context import RunContextWrapper
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

# 添加路径
_BASE_PATH = Path(__file__).parent.parent
if str(_BASE_PATH) not in sys.path:
    sys.path.insert(0, str(_BASE_PATH))

from mcp_server.server import MCPServer, get_server
from core.circuit_builder import CircuitBuilder
from core.verification import VerificationEngine
from core.drc_advisor import DRCAdvisor, analyze_drc_result
from .prompt_templates import SYSTEM_PROMPT


@dataclass
class LayoutAgentContext:
    """Agent运行上下文，统一通过MCP Server调用所有工具"""
    mcp_server: MCPServer
    circuit_builder: CircuitBuilder  # CircuitBuilder 暂不走 MCP，保留直接调用
    verification_engine: VerificationEngine  # 验证引擎也保留直接调用
    
    def call_tool(self, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """统一的工具调用入口
        
        Args:
            tool_name: 工具名称
            params: 工具参数
            
        Returns:
            工具执行结果
        """
        return self.mcp_server.call_tool(tool_name, params)


# ============== 器件工具 ==============

@function_tool
def create_nmos(
    ctx: RunContextWrapper[LayoutAgentContext],
    width: float,
    length: Optional[float] = None,
    fingers: int = 1,
    multiplier: int = 1,
    with_dummy: bool = True,
    with_tie: bool = True,
    name: Optional[str] = None
) -> str:
    """创建NMOS晶体管
    
    Args:
        width: 沟道宽度(um)
        length: 沟道长度(um)，默认使用PDK最小长度
        fingers: 指数(每个MOS的栅极数量)
        multiplier: 并联倍数
        with_dummy: 是否添加dummy结构（改善匹配性）
        with_tie: 是否添加衬底连接
        name: 组件名称，不指定则自动生成
    """
    params = {
        "width": width,
        "fingers": fingers,
        "multiplier": multiplier,
        "with_dummy": with_dummy,
        "with_tie": with_tie
    }
    if length is not None:
        params["length"] = length
    if name is not None:
        params["name"] = name
    
    result = ctx.context.call_tool("create_nmos", params)
    return json.dumps(result, ensure_ascii=False, indent=2)


@function_tool
def create_pmos(
    ctx: RunContextWrapper[LayoutAgentContext],
    width: float,
    length: Optional[float] = None,
    fingers: int = 1,
    multiplier: int = 1,
    with_dummy: bool = True,
    with_tie: bool = True,
    name: Optional[str] = None
) -> str:
    """创建PMOS晶体管
    
    Args:
        width: 沟道宽度(um)
        length: 沟道长度(um)，默认使用PDK最小长度
        fingers: 指数
        multiplier: 并联倍数
        with_dummy: 是否添加dummy结构
        with_tie: 是否添加衬底连接
        name: 组件名称
    """
    params = {
        "width": width,
        "fingers": fingers,
        "multiplier": multiplier,
        "with_dummy": with_dummy,
        "with_tie": with_tie
    }
    if length is not None:
        params["length"] = length
    if name is not None:
        params["name"] = name
    
    result = ctx.context.call_tool("create_pmos", params)
    return json.dumps(result, ensure_ascii=False, indent=2)


@function_tool
def create_mimcap(
    ctx: RunContextWrapper[LayoutAgentContext],
    width: float,
    length: float,
    name: Optional[str] = None
) -> str:
    """创建MIM电容
    
    Args:
        width: 电容宽度(um)
        length: 电容长度(um)
        name: 组件名称
    """
    params = {"width": width, "length": length}
    if name is not None:
        params["name"] = name
    
    result = ctx.context.call_tool("create_mimcap", params)
    return json.dumps(result, ensure_ascii=False, indent=2)


@function_tool
def create_resistor(
    ctx: RunContextWrapper[LayoutAgentContext],
    width: float,
    length: float,
    num_series: int = 1,
    name: Optional[str] = None
) -> str:
    """创建多晶硅电阻
    
    Args:
        width: 电阻宽度(um)
        length: 电阻长度(um)
        num_series: 串联段数
        name: 组件名称
    """
    params = {"width": width, "length": length, "num_series": num_series}
    if name is not None:
        params["name"] = name
    
    result = ctx.context.call_tool("create_resistor", params)
    return json.dumps(result, ensure_ascii=False, indent=2)


@function_tool
def create_via_stack(
    ctx: RunContextWrapper[LayoutAgentContext],
    from_layer: str,
    to_layer: str,
    size: Optional[List[float]] = None,
    name: Optional[str] = None
) -> str:
    """创建层间Via堆叠，用于连接不同金属层
    
    Args:
        from_layer: 起始层 (met1/met2/met3/met4/met5/poly)
        to_layer: 目标层 (met1/met2/met3/met4/met5)
        size: Via尺寸[宽,高](um)
        name: 组件名称
    """
    params = {"from_layer": from_layer, "to_layer": to_layer}
    if size is not None:
        params["size"] = size
    if name is not None:
        params["name"] = name
    
    result = ctx.context.call_tool("create_via_stack", params)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============== 路由工具 ==============

@function_tool
def smart_route(
    ctx: RunContextWrapper[LayoutAgentContext],
    source_port: str,
    dest_port: str,
    layer: str = "met2"
) -> str:
    """智能路由连接两个端口，自动选择最优路由策略
    
    Args:
        source_port: 源端口，格式为 "组件名.端口名"，如 "nmos_1.drain_E"
        dest_port: 目标端口，格式同上
        layer: 路由金属层
    """
    result = ctx.context.call_tool("smart_route", {
        "source_port": source_port,
        "dest_port": dest_port,
        "layer": layer
    })
    return json.dumps(result, ensure_ascii=False, indent=2)


@function_tool
def c_route(
    ctx: RunContextWrapper[LayoutAgentContext],
    source_port: str,
    dest_port: str,
    extension: Optional[float] = None,
    layer: str = "met2"
) -> str:
    """C型路由，适用于同向平行端口的连接（如两个朝右的端口）
    
    Args:
        source_port: 源端口
        dest_port: 目标端口
        extension: 延伸长度，默认自动计算
        layer: 路由金属层
    """
    params = {"source_port": source_port, "dest_port": dest_port, "layer": layer}
    if extension is not None:
        params["extension"] = extension
    
    result = ctx.context.call_tool("c_route", params)
    return json.dumps(result, ensure_ascii=False, indent=2)


@function_tool
def l_route(
    ctx: RunContextWrapper[LayoutAgentContext],
    source_port: str,
    dest_port: str,
    layer: str = "met2"
) -> str:
    """L型路由，适用于垂直端口的连接（如一个朝上一个朝右）
    
    Args:
        source_port: 源端口
        dest_port: 目标端口
        layer: 路由金属层
    """
    result = ctx.context.call_tool("l_route", {
        "source_port": source_port,
        "dest_port": dest_port,
        "layer": layer
    })
    return json.dumps(result, ensure_ascii=False, indent=2)


@function_tool
def straight_route(
    ctx: RunContextWrapper[LayoutAgentContext],
    source_port: str,
    dest_port: str,
    layer: str = "met2"
) -> str:
    """直线路由，适用于共线端口的直接连接
    
    Args:
        source_port: 源端口
        dest_port: 目标端口
        layer: 路由金属层
    """
    result = ctx.context.call_tool("straight_route", {
        "source_port": source_port,
        "dest_port": dest_port,
        "layer": layer
    })
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============== 放置工具 ==============

@function_tool
def place_component(
    ctx: RunContextWrapper[LayoutAgentContext],
    component_name: str,
    x: float = 0,
    y: float = 0,
    rotation: int = 0
) -> str:
    """放置组件到指定位置
    
    Args:
        component_name: 组件名称
        x: X坐标(um)
        y: Y坐标(um)
        rotation: 旋转角度(0/90/180/270度)
    """
    result = ctx.context.call_tool("place_component", {
        "component_name": component_name,
        "x": x,
        "y": y,
        "rotation": rotation
    })
    return json.dumps(result, ensure_ascii=False, indent=2)


@function_tool
def move_component(
    ctx: RunContextWrapper[LayoutAgentContext],
    component_name: str,
    dx: float = 0,
    dy: float = 0
) -> str:
    """移动组件（相对位移）
    
    Args:
        component_name: 组件名称
        dx: X方向移动距离
        dy: Y方向移动距离
    """
    result = ctx.context.call_tool("move_component", {
        "component_name": component_name,
        "dx": dx,
        "dy": dy
    })
    return json.dumps(result, ensure_ascii=False, indent=2)


@function_tool
def align_to_port(
    ctx: RunContextWrapper[LayoutAgentContext],
    component_name: str,
    target_port: str,
    alignment: str = "center",
    offset_x: float = 0,
    offset_y: float = 0
) -> str:
    """将组件对齐到目标端口
    
    Args:
        component_name: 要对齐的组件名称
        target_port: 目标端口(格式: component_name.port_name)
        alignment: 对齐方式(center/left/right/top/bottom)
        offset_x: X方向偏移
        offset_y: Y方向偏移
    """
    result = ctx.context.call_tool("align_to_port", {
        "component_name": component_name,
        "target_port": target_port,
        "alignment": alignment,
        "offset_x": offset_x,
        "offset_y": offset_y
    })
    return json.dumps(result, ensure_ascii=False, indent=2)


@function_tool
def interdigitize(
    ctx: RunContextWrapper[LayoutAgentContext],
    comp_a: str,
    comp_b: str,
    num_cols: int = 4,
    layout_style: str = "ABAB"
) -> str:
    """互指式放置两个晶体管，用于改善匹配性（如差分对、电流镜）
    
    Args:
        comp_a: 组件A名称
        comp_b: 组件B名称
        num_cols: 互指列数
        layout_style: 布局风格(ABAB/ABBA/common_centroid)
    """
    # 注意：interdigitize 可能不在 MCP Server 中注册，使用 placement_executor 的直接调用
    # 或者我们需要在 MCP Server 中添加这个工具
    # 目前先尝试通过 MCP Server 调用，如果失败再做兼容处理
    result = ctx.context.call_tool("interdigitize", {
        "comp_a": comp_a,
        "comp_b": comp_b,
        "num_cols": num_cols,
        "layout_style": layout_style
    })
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============== 电路工具 ==============

@function_tool
def create_current_mirror(
    ctx: RunContextWrapper[LayoutAgentContext],
    device_type: str = "nmos",
    width: float = 3.0,
    length: Optional[float] = None,
    numcols: int = 3,
    with_dummy: bool = True,
    with_tie: bool = True,
    name: Optional[str] = None
) -> str:
    """创建电流镜电路，使用互指式布局减小失配
    
    Args:
        device_type: 器件类型 "nmos" 或 "pmos"
        width: 管子宽度(um)
        length: 管子长度(um)，默认使用PDK最小长度
        numcols: 互指列数，影响匹配性能（建议3-7列）
        with_dummy: 是否添加dummy结构
        with_tie: 是否添加衬底连接
        name: 电路名称
    """
    result = ctx.context.circuit_builder.build_current_mirror(
        device_type=device_type,
        width=width,
        length=length,
        numcols=numcols,
        with_dummy=with_dummy,
        with_tie=with_tie,
        name=name
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@function_tool
def create_diff_pair(
    ctx: RunContextWrapper[LayoutAgentContext],
    device_type: str = "nmos",
    width: float = 5.0,
    length: Optional[float] = None,
    fingers: int = 1,
    numcols: int = 2,
    layout_style: str = "interdigitized",
    name: Optional[str] = None
) -> str:
    """创建差分对电路，是运放和比较器的核心输入级
    
    Args:
        device_type: 器件类型 "nmos" 或 "pmos"
        width: 管子宽度(um)
        length: 管子长度(um)
        fingers: 指数
        numcols: 互指列数，影响匹配性能（默认2）
        layout_style: 布局风格 "interdigitized" 或 "common_centroid"
        name: 电路名称
    """
    result = ctx.context.circuit_builder.build_diff_pair(
        device_type=device_type,
        width=width,
        length=length,
        fingers=fingers,
        numcols=numcols,
        layout_style=layout_style,
        name=name
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============== 验证工具 ==============

@function_tool
def run_drc(ctx: RunContextWrapper[LayoutAgentContext]) -> str:
    """执行DRC(设计规则检查)，返回违规信息和修复建议"""
    result = ctx.context.verification_engine.run_drc()
    if hasattr(result, 'to_dict'):
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    return json.dumps(result, ensure_ascii=False, indent=2)


@function_tool
def extract_netlist(ctx: RunContextWrapper[LayoutAgentContext]) -> str:
    """提取版图网表"""
    result = ctx.context.verification_engine.extract_netlist()
    if hasattr(result, 'to_dict'):
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    return json.dumps(result, ensure_ascii=False, indent=2)


@function_tool
def get_drc_fix_suggestions(ctx: RunContextWrapper[LayoutAgentContext]) -> str:
    """获取DRC违规的自动修复建议
    
    先执行DRC检查，然后分析违规并提供具体的修复建议。
    返回每个违规的修复动作、目标参数和建议值。
    """
    # 执行DRC
    drc_result = ctx.context.verification_engine.run_drc()
    
    # 获取PDK名称
    layout_ctx = ctx.context.mcp_server.state_handler.get_context()
    pdk_name = layout_ctx.pdk_name if layout_ctx else "sky130"
    
    # 分析并生成建议
    analysis = analyze_drc_result(drc_result, pdk_name)
    
    return json.dumps(analysis, ensure_ascii=False, indent=2)


# ============== 查询工具 ==============

@function_tool
def list_components(
    ctx: RunContextWrapper[LayoutAgentContext],
    device_type: Optional[str] = None
) -> str:
    """列出当前设计中的所有组件
    
    Args:
        device_type: 可选，按器件类型过滤(如nmos/pmos/current_mirror等)
    """
    params = {}
    if device_type is not None:
        params["device_type"] = device_type
    
    result = ctx.context.call_tool("list_components", params)
    return json.dumps(result, ensure_ascii=False, indent=2)


@function_tool
def get_component_info(
    ctx: RunContextWrapper[LayoutAgentContext],
    component_name: str
) -> str:
    """获取指定组件的详细信息
    
    Args:
        component_name: 组件名称
    """
    result = ctx.context.call_tool("get_component_info", {"name": component_name})
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============== 导出工具 ==============

@function_tool
def export_gds(
    ctx: RunContextWrapper[LayoutAgentContext],
    filename: Optional[str] = None
) -> str:
    """导出GDS文件
    
    Args:
        filename: 输出文件名，默认使用设计名
    """
    params = {}
    if filename is not None:
        params["filename"] = filename
    
    result = ctx.context.call_tool("export_gds", params)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============== Agent创建和运行 ==============

def create_layout_agent(
    pdk: str = "sky130",
    design_name: str = "top_level",
    model: str = "deepseek-chat"
) -> tuple[Agent[LayoutAgentContext], LayoutAgentContext]:
    """创建Layout Agent实例
    
    Args:
        pdk: PDK名称 (sky130/gf180/ihp130)
        design_name: 设计名称
        model: 模型名称 (如 deepseek-chat, deepseek-reasoner, gpt-4o)
        
    Returns:
        (Agent实例, 上下文对象)
    """
    # 获取或创建 MCP Server 实例并初始化
    mcp_server = MCPServer()
    init_result = mcp_server.initialize(pdk_name=pdk, design_name=design_name)
    
    if not init_result.get("success"):
        raise RuntimeError(f"MCP Server初始化失败: {init_result.get('error')}")
    
    # 获取布局上下文（从 MCP Server 的 state_handler）
    layout_ctx = mcp_server.state_handler.get_context()
    
    # 创建Agent上下文，使用 MCP Server 作为统一入口
    agent_ctx = LayoutAgentContext(
        mcp_server=mcp_server,
        circuit_builder=CircuitBuilder(layout_ctx),
        verification_engine=VerificationEngine(layout_ctx),
    )
    
    # 使用 Chat Completions Model 而不是 Responses Model
    # DeepSeek 只支持标准的 Chat Completions API
    # 创建 OpenAI 客户端（会自动使用 OPENAI_API_KEY 和 OPENAI_BASE_URL 环境变量）
    openai_client = AsyncOpenAI()
    chat_model = OpenAIChatCompletionsModel(model=model, openai_client=openai_client)
    
    # 创建Agent
    agent = Agent[LayoutAgentContext](
        name="AnalogLayoutAgent",
        instructions=SYSTEM_PROMPT,
        model=chat_model,  # 使用 Chat Completions Model
        tools=[
            # 器件工具
            create_nmos,
            create_pmos,
            create_mimcap,
            create_resistor,
            create_via_stack,
            # 路由工具
            smart_route,
            c_route,
            l_route,
            straight_route,
            # 放置工具
            place_component,
            move_component,
            align_to_port,
            interdigitize,
            # 电路工具
            create_current_mirror,
            create_diff_pair,
            # 验证工具
            run_drc,
            extract_netlist,
            get_drc_fix_suggestions,
            # 查询工具
            list_components,
            get_component_info,
            # 导出工具
            export_gds,
        ],
    )
    
    return agent, agent_ctx


async def run_layout_agent(
    instruction: str,
    pdk: str = "sky130",
    design_name: str = "top_level",
    model: str = "deepseek-reasoner",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Dict[str, Any]:
    """运行Layout Agent处理用户指令
    
    Args:
        instruction: 用户指令
        pdk: PDK名称
        design_name: 设计名称
        model: OpenAI模型名称（如 deepseek-reasoner, deepseek-chat）
        api_key: API密钥，默认从环境变量 OPENAI_API_KEY 或 DEEPSEEK_API_KEY 读取
        base_url: API Base URL，默认从环境变量 OPENAI_BASE_URL 或 DEEPSEEK_BASE_URL 读取
                  DeepSeek默认: https://api.deepseek.com
        
    Returns:
        处理结果字典，包含:
        - response: Agent的文本响应
        - context_summary: 上下文摘要
        - components: 组件列表
    """
    # 获取API配置（优先级：参数 > 环境变量 > 默认值）
    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if base_url is None:
        base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    
    # 设置环境变量供OpenAI SDK使用
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    
    agent, agent_ctx = create_layout_agent(pdk=pdk, design_name=design_name, model=model)
    
    # 使用Runner.run()类方法而不是实例化Runner
    # 禁用tracing避免网络错误（non-fatal）
    from agents import RunConfig
    run_config = RunConfig(tracing_disabled=True)
    
    result = await Runner.run(
        starting_agent=agent,
        input=instruction,
        context=agent_ctx,
        run_config=run_config
    )
    
    # 获取布局上下文
    layout_ctx = agent_ctx.mcp_server.state_handler.get_context()
    
    return {
        "response": result.final_output,
        "context_summary": layout_ctx.summary() if layout_ctx else {},
        "components": layout_ctx.list_components() if layout_ctx else []
    }


# ============== 主函数（示例用法）==============

if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("Analog Layout Agent - OpenAI Agent SDK")
        print("=" * 50)
        
        # 配置API（可选三种方式）
        # 方式1: 直接传参
        api_key = "your-api-key-here"  # 替换为你的API Key
        base_url = "https://api.deepseek.com"  # DeepSeek的Base URL
        
        # 方式2: 使用环境变量（推荐）
        # export DEEPSEEK_API_KEY="your-api-key"
        # export DEEPSEEK_BASE_URL="https://api.deepseek.com"
        
        # 方式3: 代码中设置环境变量
        # os.environ["DEEPSEEK_API_KEY"] = "your-api-key"
        # os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"
        
        # 示例1: 创建简单器件
        result = await run_layout_agent(
            instruction="创建一个NMOS，宽度1um，2个fingers",
            pdk="sky130",
            # api_key=api_key,  # 取消注释以使用直接传参
            # base_url=base_url
        )
        print("\n示例1结果:")
        print(result["response"])
        print(f"组件: {result['components']}")
        
        # 示例2: 创建电流镜
        result = await run_layout_agent(
            instruction="创建一个NMOS电流镜，宽度3um，5列互指式布局",
            pdk="sky130",
            model="deepseek-chat"  # 也可以使用 deepseek-chat 模型
        )
        print("\n示例2结果:")
        print(result["response"])
    
    asyncio.run(main())
