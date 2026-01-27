"""
Prompt Templates - Prompt模板

为LLM提供结构化的提示模板，用于任务分解、错误恢复等
"""

from typing import Dict, Any, List, Optional


# ============== 系统Prompt ==============

SYSTEM_PROMPT = """你是一个专业的模拟集成电路版图设计助手(Analog Layout Agent)。

## 能力范围
你可以帮助用户:
1. 创建模拟器件: NMOS, PMOS, BJT, 电阻, MIM电容, Via
2. 设计基础电路: 电流镜, 差分对, 传输门
3. 设计复合电路: 运算放大器, 级联电流镜
4. 执行布局布线: 智能路由, 器件放置
5. 执行验证: DRC检查, LVS验证, 网表提取
6. 导出文件: GDS, SPICE网表

## 支持的PDK
- sky130: Skywater 130nm开源CMOS工艺
- gf180: GlobalFoundries 180nm开源工艺
- ihp130: IHP 130nm SiGe BiCMOS开源工艺

## 工作流程
1. 确认设计需求和PDK选择
2. 创建所需器件，使用有意义的命名
3. 进行器件放置（考虑匹配和对称性）
4. 执行布线连接
5. 运行DRC检查，修复违规
6. 生成网表并可选进行LVS
7. 导出GDS文件

## DRC错误恢复流程
当DRC检查发现违规时，按以下步骤处理:
1. 调用 get_drc_fix_suggestions 获取自动修复建议
2. 根据建议的action类型采取对应措施:
   - increase_spacing: 使用 move_component 增加组件间距
   - increase_width: 重新创建更宽的器件或路由
   - increase_enclosure: 调整via尺寸
   - adjust_position: 使用 move_component 调整位置
   - add_fill: 在稀疏区域添加dummy结构
   - manual_review: 报告给用户需要手动检查
3. 修复后重新运行DRC验证
4. 重复直到DRC通过或达到最大尝试次数

## 设计规则提示
- 所有尺寸单位为微米(um)
- sky130最小尺寸: W>=0.15um, L>=0.15um
- gf180最小尺寸: W>=0.22um, L>=0.18um
- 默认使用met2层布线
- 电流镜/差分对建议使用互指式布局减小失配

## 响应格式
每次操作后，我会提供:
- 操作结果状态
- 创建的组件信息（名称、端口列表、尺寸）
- DRC状态（如适用）
- 下一步建议
"""


# ============== 任务分解Prompt ==============

TASK_DECOMPOSITION_PROMPT = """
请将用户的设计需求分解为具体的操作步骤。

用户需求: {user_instruction}
当前PDK: {current_pdk}
已有组件: {existing_components}

请输出JSON格式的任务列表，每个任务包含:
- tool: 工具名称
- params: 参数字典
- description: 操作描述

示例输出:
[
    {{"tool": "create_nmos", "params": {{"width": 1.0, "length": 0.15}}, "description": "创建输入NMOS"}},
    {{"tool": "create_pmos", "params": {{"width": 2.0}}, "description": "创建负载PMOS"}},
    {{"tool": "smart_route", "params": {{"source_port": "nmos_1.drain_E", "dest_port": "pmos_1.drain_E"}}, "description": "连接漏极"}}
]

可用的工具:
- 器件: create_nmos, create_pmos, create_mimcap, create_resistor, create_via_stack, create_tapring
- 电路: create_current_mirror, create_diff_pair, create_opamp
- 路由: smart_route, c_route, l_route, straight_route
- 放置: place_component, align_to_port, move_component, interdigitize
- 验证: run_drc, run_lvs, extract_netlist
- 导出: export_gds
"""


# ============== 错误恢复Prompt ==============

ERROR_RECOVERY_PROMPT = """
执行过程中遇到错误，请分析原因并提供修复建议。

任务信息:
- 工具: {tool_name}
- 参数: {tool_params}

错误信息:
- 类型: {error_type}
- 描述: {error_message}
- 详情: {error_details}

当前上下文:
- PDK: {current_pdk}
- 已有组件: {existing_components}

请提供:
1. 可能的错误原因分析
2. 建议的修复方案
3. 修改后的参数（如适用）

响应格式:
{{
    "analysis": "错误原因分析",
    "suggestions": ["建议1", "建议2"],
    "fixed_params": {{"param1": "new_value"}},
    "retry": true/false
}}
"""


# ============== 设计验证Prompt ==============

DESIGN_REVIEW_PROMPT = """
请审查当前的版图设计，检查是否存在潜在问题。

设计概览:
- 设计名称: {design_name}
- PDK: {pdk_name}
- 组件数量: {component_count}

组件列表:
{component_list}

连接列表:
{connection_list}

请检查:
1. 器件尺寸是否合理（是否过小/过大）
2. 匹配性要求是否满足（差分对、电流镜是否使用匹配布局）
3. 布线是否可能导致DRC违规
4. 是否缺少必要的连接

输出格式:
{{
    "issues": [
        {{"severity": "warning/error", "category": "matching/drc/connectivity", "description": "问题描述", "suggestion": "修复建议"}}
    ],
    "score": 0-100,
    "summary": "总体评估"
}}
"""


# ============== 电路特定Prompt ==============

CURRENT_MIRROR_PROMPT = """
设计电流镜电路需要考虑以下因素:

## 设计参数
- 输入电流: {input_current}uA
- 镜像比例: {mirror_ratio}
- 器件类型: {device_type}

## 设计约束
- 最小长度: {min_length}um（通常>最小长度的2-4倍以提高匹配性）
- 输出电阻要求: {output_resistance}
- 工作电压范围: {voltage_range}V

## 建议
1. 使用互指式布局减小工艺偏差
2. 列数建议: 3-7列（列数越多匹配性越好但面积增大）
3. 添加dummy结构保护边缘管子
4. 使用共源共栅结构提高输出电阻

请生成电流镜的创建任务。
"""


DIFF_PAIR_PROMPT = """
设计差分对电路需要考虑以下因素:

## 设计参数
- 输入偏置电流: {bias_current}uA
- 跨导要求: {transconductance}uS
- 输入共模范围: {input_cm_range}V

## 设计约束
- 偏移电压要求: <{offset_voltage}mV
- 噪声性能: {noise_spec}

## 建议
1. 使用互指式或共心式布局
2. W/L比例影响跨导和噪声
3. 长沟道管子（L>2*Lmin）改善匹配性
4. 差分对两管尽量对称放置

请生成差分对的创建任务。
"""


# ============== 工具使用示例 ==============

TOOL_EXAMPLES = {
    "create_nmos": """
创建NMOS示例:
- 基本创建: {{"width": 1.0}}
- 带长度: {{"width": 2.0, "length": 0.5}}
- 多指结构: {{"width": 1.0, "fingers": 4}}
- 完整参数: {{"width": 1.0, "length": 0.15, "fingers": 2, "multiplier": 1, "with_dummy": true, "with_tie": true}}
""",
    
    "create_current_mirror": """
创建电流镜示例:
- 基本NMOS电流镜: {{"device_type": "nmos", "width": 3.0, "numcols": 3}}
- PMOS电流镜: {{"device_type": "pmos", "width": 5.0, "numcols": 5}}
- 高匹配性: {{"device_type": "nmos", "width": 3.0, "length": 0.5, "numcols": 7, "with_dummy": true}}
""",
    
    "smart_route": """
智能路由示例:
- 基本连接: {{"source_port": "nmos_1.drain_E", "dest_port": "pmos_1.drain_E"}}
- 指定层: {{"source_port": "m1.gate_W", "dest_port": "m2.gate_W", "layer": "met2"}}
""",
    
    "place_component": """
放置组件示例:
- 绝对位置: {{"component_name": "nmos_1", "x": 10.0, "y": 20.0}}
- 带旋转: {{"component_name": "pmos_1", "x": 0, "y": 0, "rotation": 180}}
""",
    
    "run_drc": """
DRC检查示例:
- 检查顶层: {{}}
- 检查特定组件: {{"component_name": "nmos_1"}}
- 详细报告: {{"output_format": "detailed"}}
"""
}


def get_system_prompt() -> str:
    """获取系统Prompt"""
    return SYSTEM_PROMPT


def get_task_decomposition_prompt(
    user_instruction: str,
    current_pdk: str,
    existing_components: List[str]
) -> str:
    """获取任务分解Prompt
    
    Args:
        user_instruction: 用户指令
        current_pdk: 当前PDK
        existing_components: 已有组件列表
        
    Returns:
        格式化后的Prompt
    """
    return TASK_DECOMPOSITION_PROMPT.format(
        user_instruction=user_instruction,
        current_pdk=current_pdk,
        existing_components=", ".join(existing_components) if existing_components else "无"
    )


def get_error_recovery_prompt(
    tool_name: str,
    tool_params: Dict[str, Any],
    error_type: str,
    error_message: str,
    error_details: Dict[str, Any],
    current_pdk: str,
    existing_components: List[str]
) -> str:
    """获取错误恢复Prompt
    
    Args:
        tool_name: 工具名称
        tool_params: 工具参数
        error_type: 错误类型
        error_message: 错误消息
        error_details: 错误详情
        current_pdk: 当前PDK
        existing_components: 已有组件
        
    Returns:
        格式化后的Prompt
    """
    return ERROR_RECOVERY_PROMPT.format(
        tool_name=tool_name,
        tool_params=str(tool_params),
        error_type=error_type,
        error_message=error_message,
        error_details=str(error_details),
        current_pdk=current_pdk,
        existing_components=", ".join(existing_components) if existing_components else "无"
    )


def get_design_review_prompt(
    design_name: str,
    pdk_name: str,
    component_count: int,
    component_list: str,
    connection_list: str
) -> str:
    """获取设计审查Prompt
    
    Args:
        design_name: 设计名称
        pdk_name: PDK名称
        component_count: 组件数量
        component_list: 组件列表描述
        connection_list: 连接列表描述
        
    Returns:
        格式化后的Prompt
    """
    return DESIGN_REVIEW_PROMPT.format(
        design_name=design_name,
        pdk_name=pdk_name,
        component_count=component_count,
        component_list=component_list,
        connection_list=connection_list
    )


def get_tool_example(tool_name: str) -> Optional[str]:
    """获取工具使用示例
    
    Args:
        tool_name: 工具名称
        
    Returns:
        示例字符串，如果工具不存在返回None
    """
    return TOOL_EXAMPLES.get(tool_name)


def get_all_tool_examples() -> Dict[str, str]:
    """获取所有工具示例"""
    return TOOL_EXAMPLES.copy()


# ============== 中间表示格式化 ==============

def format_context_for_llm(context_dict: Dict[str, Any]) -> str:
    """将上下文格式化为LLM友好的描述
    
    Args:
        context_dict: 上下文字典
        
    Returns:
        格式化的描述字符串
    """
    lines = [
        f"当前设计: {context_dict.get('design_name', 'unknown')}",
        f"PDK: {context_dict.get('pdk_name', 'unknown')}",
        f"状态: {context_dict.get('state', 'unknown')}",
        ""
    ]
    
    # 组件信息
    components = context_dict.get('components', {}).get('components', {})
    if components:
        lines.append(f"组件 ({len(components)}个):")
        for name, info in list(components.items())[:10]:  # 最多显示10个
            device_type = info.get('device_type', 'unknown')
            size = info.get('size', (0, 0))
            ports = info.get('ports', [])[:5]
            lines.append(f"  - {name} ({device_type})")
            lines.append(f"    尺寸: {size[0]:.2f}x{size[1]:.2f}um")
            lines.append(f"    端口: {', '.join(ports)}")
    else:
        lines.append("组件: 无")
    
    # 连接信息
    connections = context_dict.get('connections', [])
    if connections:
        lines.append(f"\n连接 ({len(connections)}条):")
        for conn in connections[:10]:
            lines.append(f"  - {conn.get('source')} -> {conn.get('target')}")
            lines.append(f"    层: {conn.get('layer')}, 类型: {conn.get('route_type')}")
    else:
        lines.append("\n连接: 无")
    
    return "\n".join(lines)


def format_result_for_llm(result: Dict[str, Any]) -> str:
    """将执行结果格式化为LLM友好的描述
    
    Args:
        result: 执行结果字典
        
    Returns:
        格式化的描述字符串
    """
    lines = []
    
    if result.get("success"):
        lines.append("✓ 操作成功")
        
        if "component_name" in result:
            lines.append(f"创建组件: {result['component_name']}")
        
        if "device_type" in result:
            lines.append(f"类型: {result['device_type']}")
        
        if "params" in result:
            params_str = ", ".join(f"{k}={v}" for k, v in result['params'].items())
            lines.append(f"参数: {params_str}")
        
        if "ports" in result:
            ports = result['ports'][:5]
            lines.append(f"端口: {', '.join(ports)}")
            if len(result['ports']) > 5:
                lines.append(f"  ... 还有{len(result['ports'])-5}个端口")
        
        if "bbox" in result:
            bbox = result['bbox']
            lines.append(f"尺寸: {bbox.get('width', 0):.2f}x{bbox.get('height', 0):.2f}um")
    
    else:
        lines.append("✗ 操作失败")
        if "error" in result:
            lines.append(f"错误: {result['error']}")
    
    return "\n".join(lines)
