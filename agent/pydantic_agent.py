"""
PydanticAI Agent 集成模块

使用 PydanticAI 框架构建 Layout Agent，通过 LLM 实现智能指令解析和工具调用。
统一通过 MCP Server 的 call_tool() 作为单一工具调用入口（Single Source of Truth）。

支持两种工具调用模式：
1. Skills 模式（推荐）：使用 PydanticAI Skills 实现渐进式披露，按需加载技能
2. 传统模式：直接注册所有工具到 Agent

迁移自 OpenAI Agent SDK 实现，保持相同的功能和接口。
"""

import sys
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# 添加路径
_BASE_PATH = Path(__file__).parent.parent
if str(_BASE_PATH) not in sys.path:
    sys.path.insert(0, str(_BASE_PATH))

from mcp_server.server import MCPServer
from core.circuit_builder import CircuitBuilder
from core.verification import VerificationEngine
from core.drc_advisor import analyze_drc_result
from .prompt_templates import SYSTEM_PROMPT
from .reasoning_agent import load_constitution


# ============== 依赖类型定义 ==============

@dataclass
class LayoutAgentDeps:
    """Agent 运行时依赖项
    
    包含所有运行时需要的服务和状态。
    在 PydanticAI 中，依赖项通过 deps_type 定义，运行时通过 deps 参数传入。
    
    Attributes:
        mcp_server: MCP Server 实例，提供统一的工具调用入口
        circuit_builder: 电路构建器，用于创建复合电路
        verification_engine: 验证引擎，用于 DRC/LVS 验证
        constitution: Agent 宪法内容（强制遵循规则）
        session_id: 当前 session 标识（用于追踪）
        init_status: 初始化状态信息（供 LLM 感知 Flow 执行结果）
    """
    mcp_server: MCPServer
    circuit_builder: CircuitBuilder
    verification_engine: VerificationEngine
    constitution: str = ""
    session_id: str = ""
    init_status: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化时自动加载宪法并生成 session_id"""
        if not self.constitution:
            self.constitution = load_constitution()
        if not self.session_id:
            import uuid
            self.session_id = str(uuid.uuid4())[:8]
    
    def call_tool(self, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """统一的工具调用入口"""
        return self.mcp_server.call_tool(tool_name, params or {})


# ============== Agent 工厂函数 ==============

def create_layout_agent(
    model_name: str = "deepseek-reasoner",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    use_skills: bool = False
) -> Tuple[Agent[LayoutAgentDeps, str], Optional[Any]]:
    """创建 Layout Agent 实例"""
    
    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if base_url is None:
        base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    model = OpenAIChatModel(model_name, provider=provider)
    
    skills_toolset = None
    
    if use_skills:
        try:
            from ..skills import create_layout_skills_toolset
            
            skills_toolset = create_layout_skills_toolset()
            
            layout_agent = Agent(
                model,
                deps_type=LayoutAgentDeps,
                output_type=str,
                system_prompt=SYSTEM_PROMPT,
                retries=2,
                toolsets=[skills_toolset]
            )
            
            @layout_agent.instructions
            async def add_skills_instructions(ctx: RunContext[LayoutAgentDeps]) -> str | None:
                """动态添加技能列表到系统提示"""
                return await skills_toolset.get_instructions(ctx)
            
            # Skills 模式也需要注入宪法
            @layout_agent.instructions
            async def inject_constitution_skills(ctx: RunContext[LayoutAgentDeps]) -> str:
                """Skills 模式下的宪法注入"""
                return _build_constitution_injection(ctx.deps)
            
            return layout_agent, skills_toolset
            
        except ImportError as e:
            import warnings
            warnings.warn(f"Skills 模块导入失败: {e}，回退到传统模式")
            use_skills = False
    
    # 传统模式
    layout_agent = Agent(
        model,
        deps_type=LayoutAgentDeps,
        output_type=str,
        system_prompt=SYSTEM_PROMPT,
        retries=2,
    )
    
    # ============== Session 级宪法完整注入 ==============
    @layout_agent.instructions
    async def inject_constitution(ctx: RunContext[LayoutAgentDeps]) -> str:
        """
        Session 级宪法完整注入
        
        此函数在每次 agent.run() 调用时执行，确保：
        1. 每个新 session 都注入完整宪法（AGENT_CONSTITUTION.md）
        2. 宪法内容作为 LLM 收到的第一部分指令
        3. 包含 session 标识和初始化状态
        
        PydanticAI 机制：
        - @agent.instructions 装饰的函数返回值追加到 system_prompt 之后
        - 在 LLM 收到用户指令之前执行
        - 每次 agent.run() 都会触发（session 级别）
        """
        return _build_constitution_injection(ctx.deps)
    
    # 注册所有工具
    _register_device_tools(layout_agent)
    _register_routing_tools(layout_agent)
    _register_placement_tools(layout_agent)
    _register_circuit_tools(layout_agent)
    _register_verification_tools(layout_agent)
    _register_query_tools(layout_agent)
    _register_export_tools(layout_agent)
    
    return layout_agent, None


def _build_constitution_injection(deps: LayoutAgentDeps) -> str:
    """
    构建宪法注入内容
    
    Args:
        deps: Agent 依赖项，包含宪法内容和 session 信息
        
    Returns:
        格式化的宪法注入字符串
    """
    parts = []
    
    # 1. Session 标识
    parts.append(f"""
═══════════════════════════════════════════════════════════════
                    SESSION INITIALIZED
                    ID: {deps.session_id}
═══════════════════════════════════════════════════════════════
""")
    
    # 2. 完整宪法内容（强制）
    constitution = deps.constitution or load_constitution()
    if constitution:
        parts.append("""
## 🚨 AGENT CONSTITUTION (最高优先级 - 必须遵守)

以下是 Agent 宪法的完整内容。任何违反都将导致任务失败。
在处理任何请求之前，请确保理解并遵守所有规则。

""")
        parts.append(constitution)
    else:
        parts.append("\n⚠️ 警告: 宪法文件未加载，请检查 AGENT_CONSTITUTION.md\n")
    
    # 3. 初始化状态感知（如果有）
    if deps.init_status:
        parts.append("\n\n## 当前初始化状态\n")
        if deps.init_status.get("init_sh_executed"):
            status = "✓ 成功" if deps.init_status.get("init_sh_success") else "✗ 失败"
            parts.append(f"- [宪法1.1] init.sh: {status}\n")
        if deps.init_status.get("progress_read"):
            parts.append(f"- [宪法1.2] progress.md: 已读取\n")
    
    # 4. 合规确认提示
    parts.append("""

## 执行前确认

在执行任何操作前，我已确认：
- ✓ 已阅读并理解上述宪法全部内容
- ✓ 将按照宪法规定的顺序执行步骤
- ✓ routing 操作将指定 layer 参数
- ✓ 只有验证通过才会修改 completed 状态
""")
    
    return "".join(parts)


# ============== 器件工具 ==============

def _register_device_tools(agent: Agent[LayoutAgentDeps, str]) -> None:
    """注册器件创建工具"""
    
    @agent.tool
    async def create_nmos(
        ctx: RunContext[LayoutAgentDeps],
        width: float,
        length: float | None = None,
        fingers: int = 1,
        multiplier: int = 1,
        with_dummy: bool = True,
        with_tie: bool = True,
        name: str | None = None
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
        params: Dict[str, Any] = {
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
        
        result = ctx.deps.call_tool("create_nmos", params)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @agent.tool
    async def create_pmos(
        ctx: RunContext[LayoutAgentDeps],
        width: float,
        length: float | None = None,
        fingers: int = 1,
        multiplier: int = 1,
        with_dummy: bool = True,
        with_tie: bool = True,
        name: str | None = None
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
        params: Dict[str, Any] = {
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
        
        result = ctx.deps.call_tool("create_pmos", params)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @agent.tool
    async def create_mimcap(
        ctx: RunContext[LayoutAgentDeps],
        width: float,
        length: float,
        name: str | None = None
    ) -> str:
        """创建MIM电容
        
        Args:
            width: 电容宽度(um)
            length: 电容长度(um)
            name: 组件名称
        """
        params: Dict[str, Any] = {"width": width, "length": length}
        if name is not None:
            params["name"] = name
        
        result = ctx.deps.call_tool("create_mimcap", params)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @agent.tool
    async def create_resistor(
        ctx: RunContext[LayoutAgentDeps],
        width: float,
        length: float,
        num_series: int = 1,
        name: str | None = None
    ) -> str:
        """创建多晶硅电阻
        
        Args:
            width: 电阻宽度(um)
            length: 电阻长度(um)
            num_series: 串联段数
            name: 组件名称
        """
        params: Dict[str, Any] = {"width": width, "length": length, "num_series": num_series}
        if name is not None:
            params["name"] = name
        
        result = ctx.deps.call_tool("create_resistor", params)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @agent.tool
    async def create_via_stack(
        ctx: RunContext[LayoutAgentDeps],
        from_layer: str,
        to_layer: str,
        size: List[float] | None = None,
        name: str | None = None
    ) -> str:
        """创建层间Via堆叠，用于连接不同金属层
        
        Args:
            from_layer: 起始层 (met1/met2/met3/met4/met5/poly)
            to_layer: 目标层 (met1/met2/met3/met4/met5)
            size: Via尺寸[宽,高](um)
            name: 组件名称
        """
        params: Dict[str, Any] = {"from_layer": from_layer, "to_layer": to_layer}
        if size is not None:
            params["size"] = size
        if name is not None:
            params["name"] = name
        
        result = ctx.deps.call_tool("create_via_stack", params)
        return json.dumps(result, ensure_ascii=False, indent=2)


# ============== 路由工具 ==============

def _register_routing_tools(agent: Agent[LayoutAgentDeps, str]) -> None:
    """注册路由工具"""
    
    @agent.tool
    async def smart_route(
        ctx: RunContext[LayoutAgentDeps],
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
        result = ctx.deps.call_tool("smart_route", {
            "source_port": source_port,
            "dest_port": dest_port,
            "layer": layer
        })
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @agent.tool
    async def c_route(
        ctx: RunContext[LayoutAgentDeps],
        source_port: str,
        dest_port: str,
        extension: float | None = None,
        layer: str = "met2"
    ) -> str:
        """C型路由，适用于同向平行端口的连接（如两个朝右的端口）
        
        Args:
            source_port: 源端口
            dest_port: 目标端口
            extension: 延伸长度，默认自动计算
            layer: 路由金属层
        """
        params: Dict[str, Any] = {"source_port": source_port, "dest_port": dest_port, "layer": layer}
        if extension is not None:
            params["extension"] = extension
        
        result = ctx.deps.call_tool("c_route", params)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @agent.tool
    async def l_route(
        ctx: RunContext[LayoutAgentDeps],
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
        result = ctx.deps.call_tool("l_route", {
            "source_port": source_port,
            "dest_port": dest_port,
            "layer": layer
        })
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @agent.tool
    async def straight_route(
        ctx: RunContext[LayoutAgentDeps],
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
        result = ctx.deps.call_tool("straight_route", {
            "source_port": source_port,
            "dest_port": dest_port,
            "layer": layer
        })
        return json.dumps(result, ensure_ascii=False, indent=2)


# ============== 放置工具 ==============

def _register_placement_tools(agent: Agent[LayoutAgentDeps, str]) -> None:
    """注册放置工具"""
    
    @agent.tool
    async def place_component(
        ctx: RunContext[LayoutAgentDeps],
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
        result = ctx.deps.call_tool("place_component", {
            "component_name": component_name,
            "x": x,
            "y": y,
            "rotation": rotation
        })
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @agent.tool
    async def move_component(
        ctx: RunContext[LayoutAgentDeps],
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
        result = ctx.deps.call_tool("move_component", {
            "component_name": component_name,
            "dx": dx,
            "dy": dy
        })
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @agent.tool
    async def align_to_port(
        ctx: RunContext[LayoutAgentDeps],
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
        result = ctx.deps.call_tool("align_to_port", {
            "component_name": component_name,
            "target_port": target_port,
            "alignment": alignment,
            "offset_x": offset_x,
            "offset_y": offset_y
        })
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @agent.tool
    async def interdigitize(
        ctx: RunContext[LayoutAgentDeps],
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
        result = ctx.deps.call_tool("interdigitize", {
            "comp_a": comp_a,
            "comp_b": comp_b,
            "num_cols": num_cols,
            "layout_style": layout_style
        })
        return json.dumps(result, ensure_ascii=False, indent=2)


# ============== 电路工具 ==============

def _register_circuit_tools(agent: Agent[LayoutAgentDeps, str]) -> None:
    """注册电路工具"""
    
    @agent.tool
    async def create_current_mirror(
        ctx: RunContext[LayoutAgentDeps],
        device_type: str = "nmos",
        width: float = 3.0,
        length: float | None = None,
        numcols: int = 3,
        with_dummy: bool = True,
        with_tie: bool = True,
        name: str | None = None
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
        # 直接调用 CircuitBuilder（保留原有设计）
        result = ctx.deps.circuit_builder.build_current_mirror(
            device_type=device_type,
            width=width,
            length=length,
            numcols=numcols,
            with_dummy=with_dummy,
            with_tie=with_tie,
            name=name
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @agent.tool
    async def create_diff_pair(
        ctx: RunContext[LayoutAgentDeps],
        device_type: str = "nmos",
        width: float = 5.0,
        length: float | None = None,
        fingers: int = 1,
        numcols: int = 2,
        layout_style: str = "interdigitized",
        name: str | None = None
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
        result = ctx.deps.circuit_builder.build_diff_pair(
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

def _register_verification_tools(agent: Agent[LayoutAgentDeps, str]) -> None:
    """注册验证工具"""
    
    @agent.tool
    async def run_drc(ctx: RunContext[LayoutAgentDeps]) -> str:
        """执行DRC(设计规则检查)，返回违规信息和修复建议"""
        result = ctx.deps.verification_engine.run_drc()
        if hasattr(result, 'to_dict'):
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @agent.tool
    async def extract_netlist(ctx: RunContext[LayoutAgentDeps]) -> str:
        """提取版图网表"""
        result = ctx.deps.verification_engine.extract_netlist()
        if hasattr(result, 'to_dict'):
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @agent.tool
    async def get_drc_fix_suggestions(ctx: RunContext[LayoutAgentDeps]) -> str:
        """获取DRC违规的自动修复建议
        
        先执行DRC检查，然后分析违规并提供具体的修复建议。
        返回每个违规的修复动作、目标参数和建议值。
        """
        # 执行DRC
        drc_result = ctx.deps.verification_engine.run_drc()
        
        # 获取PDK名称
        layout_ctx = ctx.deps.mcp_server.state_handler.get_context()
        pdk_name = layout_ctx.pdk_name if layout_ctx else "sky130"
        
        # 分析并生成建议
        analysis = analyze_drc_result(drc_result, pdk_name)
        
        return json.dumps(analysis, ensure_ascii=False, indent=2)


# ============== 查询工具 ==============

def _register_query_tools(agent: Agent[LayoutAgentDeps, str]) -> None:
    """注册查询工具"""
    
    @agent.tool
    async def list_components(
        ctx: RunContext[LayoutAgentDeps],
        device_type: str | None = None
    ) -> str:
        """列出当前设计中的所有组件
        
        Args:
            device_type: 可选，按器件类型过滤(如nmos/pmos/current_mirror等)
        """
        params: Dict[str, Any] = {}
        if device_type is not None:
            params["device_type"] = device_type
        
        result = ctx.deps.call_tool("list_components", params)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @agent.tool
    async def get_component_info(
        ctx: RunContext[LayoutAgentDeps],
        component_name: str
    ) -> str:
        """获取指定组件的详细信息
        
        Args:
            component_name: 组件名称
        """
        result = ctx.deps.call_tool("get_component_info", {"name": component_name})
        return json.dumps(result, ensure_ascii=False, indent=2)


# ============== 导出工具 ==============

def _register_export_tools(agent: Agent[LayoutAgentDeps, str]) -> None:
    """注册导出工具"""
    
    @agent.tool
    async def export_gds(
        ctx: RunContext[LayoutAgentDeps],
        filename: str | None = None
    ) -> str:
        """导出GDS文件
        
        Args:
            filename: 输出文件名，默认使用设计名
        """
        params: Dict[str, Any] = {}
        if filename is not None:
            params["filename"] = filename
        
        result = ctx.deps.call_tool("export_gds", params)
        return json.dumps(result, ensure_ascii=False, indent=2)


# ============== 运行入口 ==============

async def run_layout_agent(
    instruction: str,
    pdk: str = "sky130",
    design_name: str = "top_level",
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    use_skills: bool = False
) -> Dict[str, Any]:
    """运行 Layout Agent 处理用户指令
    
    Args:
        instruction: 用户指令
        pdk: PDK名称
        design_name: 设计名称
        model: 模型名称（如 deepseek-chat, deepseek-reasoner）
        api_key: API密钥，默认从环境变量读取
        base_url: API Base URL，默认从环境变量读取
        use_skills: 是否使用 Skills 模式（推荐 True，实现渐进式披露减少Token）
        
    Returns:
        处理结果字典，包含:
        - response: Agent的文本响应
        - context_summary: 上下文摘要
        - components: 组件列表
        - usage: Token 使用信息
        - mode: 使用的模式 ("skills" 或 "traditional")
    """
    # 创建 Agent
    agent, skills_toolset = create_layout_agent(
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        use_skills=use_skills
    )
    
    # 初始化 MCP Server 和依赖
    mcp_server = MCPServer()
    init_result = mcp_server.initialize(pdk_name=pdk, design_name=design_name)
    
    if not init_result.get("success"):
        raise RuntimeError(f"MCP Server初始化失败: {init_result.get('error')}")
    
    # 获取布局上下文
    layout_ctx = mcp_server.state_handler.get_context()
    
    # 创建依赖对象
    deps = LayoutAgentDeps(
        mcp_server=mcp_server,
        circuit_builder=CircuitBuilder(layout_ctx),
        verification_engine=VerificationEngine(layout_ctx),
    )
    
    # 运行 Agent（异步方式）
    result = await agent.run(instruction, deps=deps)
    
    # 构建返回结果
    usage_info = {}
    if result.usage():
        usage_info = {
            "total_tokens": result.usage().total_tokens,
            "request_tokens": result.usage().request_tokens,
            "response_tokens": result.usage().response_tokens,
        }
    
    return {
        "response": result.output,
        "context_summary": layout_ctx.summary() if layout_ctx else {},
        "components": layout_ctx.list_components() if layout_ctx else [],
        "usage": usage_info,
        "mode": "skills" if skills_toolset is not None else "traditional"
    }


def run_layout_agent_sync(
    instruction: str,
    pdk: str = "sky130",
    design_name: str = "top_level",
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    use_skills: bool = True
) -> Dict[str, Any]:
    """同步运行 Layout Agent（便捷方法）
    
    内部使用 asyncio.run() 调用异步版本。
    """
    import asyncio
    return asyncio.run(run_layout_agent(
        instruction=instruction,
        pdk=pdk,
        design_name=design_name,
        model=model,
        api_key=api_key,
        base_url=base_url,
        use_skills=use_skills
    ))


async def run_layout_agent_stream(
    instruction: str,
    pdk: str = "sky130",
    design_name: str = "top_level",
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    on_text: Optional[callable] = None,
    use_skills: bool = True
) -> Dict[str, Any]:
    """流式运行 Layout Agent
    
    Args:
        instruction: 用户指令
        pdk: PDK名称
        design_name: 设计名称
        model: 模型名称
        api_key: API密钥
        base_url: API Base URL
        on_text: 文本回调函数，每次收到新文本时调用
        use_skills: 是否使用 Skills 模式
        
    Returns:
        完整的处理结果
    """
    # 创建 Agent
    agent, skills_toolset = create_layout_agent(
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        use_skills=use_skills
    )
    
    # 初始化 MCP Server 和依赖
    mcp_server = MCPServer()
    init_result = mcp_server.initialize(pdk_name=pdk, design_name=design_name)
    
    if not init_result.get("success"):
        raise RuntimeError(f"MCP Server初始化失败: {init_result.get('error')}")
    
    layout_ctx = mcp_server.state_handler.get_context()
    
    deps = LayoutAgentDeps(
        mcp_server=mcp_server,
        circuit_builder=CircuitBuilder(layout_ctx),
        verification_engine=VerificationEngine(layout_ctx),
    )
    
    # 流式运行
    full_response = ""
    async with agent.run_stream(instruction, deps=deps) as response:
        async for text in response.stream_text():
            full_response += text
            if on_text:
                on_text(text)
    
    return {
        "response": full_response,
        "context_summary": layout_ctx.summary() if layout_ctx else {},
        "components": layout_ctx.list_components() if layout_ctx else [],
        "mode": "skills" if skills_toolset is not None else "traditional"
    }


# ============== 主函数（示例用法）==============

if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("Analog Layout Agent - PydanticAI")
        print("=" * 50)
        
        # 示例1: 创建简单器件
        try:
            result = await run_layout_agent(
                instruction="创建一个NMOS，宽度1um，2个fingers",
                pdk="sky130",
            )
            print("\n示例1结果:")
            print(result["response"])
            print(f"组件: {result['components']}")
            print(f"Token使用: {result.get('usage', {})}")
        except Exception as e:
            print(f"示例1出错: {e}")
        
        # 示例2: 创建电流镜
        try:
            result = await run_layout_agent(
                instruction="创建一个NMOS电流镜，宽度3um，5列互指式布局",
                pdk="sky130",
            )
            print("\n示例2结果:")
            print(result["response"])
        except Exception as e:
            print(f"示例2出错: {e}")
    
    asyncio.run(main())
