"""
器件创建技能模块 (Device Creation Skill)

使用 PydanticAI Skills 封装器件创建相关工具，实现渐进式披露。
包含 NMOS、PMOS、MIM电容、多晶硅电阻、Via堆叠等基础器件的创建功能。

该技能通过 MCP Server 的统一工具调用入口与底层实现交互，
保持与现有 LayoutAgentDeps 依赖类型的完全兼容。
"""

from __future__ import annotations

import json
from typing import Optional, Dict, Any, List

from pydantic_ai import RunContext
from pydantic_ai_skills import Skill, SkillResource


# ============== 技能指令文档 ==============

DEVICE_SKILL_INSTRUCTIONS = """
# 器件创建技能 (Device Creation Skill)

## 何时使用此技能

当你需要创建以下基础模拟电路器件时使用此技能：
- **NMOS/PMOS 晶体管**: 模拟电路的核心器件
- **MIM 电容**: 金属-绝缘体-金属电容，用于滤波、补偿等
- **多晶硅电阻**: 精密电阻，用于偏置、反馈网络
- **Via 过孔堆叠**: 连接不同金属层的垂直互连

## 使用流程

1. 根据电路需求确定器件类型和参数
2. 调用相应的创建脚本（如 `create_nmos`、`create_pmos`）
3. 记录返回的器件名称，用于后续布线和放置操作
4. 如需参考参数范围，使用 `read_skill_resource` 读取 `device-reference`

## 关键参数说明

### 晶体管参数
| 参数 | 说明 | 典型值 |
|------|------|--------|
| `width` | 沟道宽度(μm)，影响电流驱动能力 | 1-10μm |
| `length` | 沟道长度(μm)，影响输出阻抗 | 0.15-1μm |
| `fingers` | 指数，将宽MOS分成多个并联单元 | 1-8 |
| `multiplier` | 并联倍数，用于电流镜匹配 | 1-4 |
| `with_dummy` | 添加dummy结构改善匹配性 | True |
| `with_tie` | 添加衬底连接 | True |

### 电容参数
| 参数 | 说明 | 典型值 |
|------|------|--------|
| `width` | 电容宽度(μm) | 5-20μm |
| `length` | 电容长度(μm) | 5-20μm |

### 电阻参数
| 参数 | 说明 | 典型值 |
|------|------|--------|
| `width` | 电阻宽度(μm) | 0.5-2μm |
| `length` | 电阻长度(μm) | 5-50μm |
| `num_series` | 串联段数 | 1-10 |

## 脚本列表

- `create_nmos`: 创建NMOS晶体管
- `create_pmos`: 创建PMOS晶体管  
- `create_mimcap`: 创建MIM电容
- `create_resistor`: 创建多晶硅电阻
- `create_via_stack`: 创建层间Via堆叠

## 使用示例

创建一个3μm宽、4指的NMOS：
```python
run_skill_script(
    skill_name="device-creation",
    script_name="create_nmos",
    args={"width": 3.0, "fingers": 4, "with_dummy": True}
)
```

## 注意事项

- 宽度过小可能导致 DRC 违规
- 建议对匹配敏感电路（如差分对、电流镜）使用 `with_dummy=True`
- 器件名称不指定时会自动生成唯一 ID
- 创建后的器件默认位于原点，需使用 placement-layout 技能进行放置
"""


DEVICE_REFERENCE = """
# 器件参数参考文档

## NMOS/PMOS 典型参数范围 (Sky130 PDK)

| 参数 | 最小值 | 典型值 | 最大值 | 单位 | 说明 |
|------|--------|--------|--------|------|------|
| width | 0.42 | 1-10 | 100 | μm | 沟道宽度 |
| length | 0.15 | 0.15-1 | 10 | μm | 沟道长度 |
| fingers | 1 | 1-8 | 64 | - | 栅极指数 |
| multiplier | 1 | 1-4 | 32 | - | 并联倍数 |

## 设计规则要点

### 最小间距要求
- poly 最小间距: 0.21μm
- metal1 最小间距: 0.14μm
- metal2 最小间距: 0.14μm

### 电流镜设计建议
- 使用相同的 width/length 确保匹配
- multiplier 建议为 1:N 整数比
- 启用 with_dummy 改善边缘效应
- 使用 interdigitize 放置方式

### Via 堆叠层序
支持的层: poly, met1, met2, met3, met4, met5
Via类型:
- poly → met1: contact
- met1 → met2: via1
- met2 → met3: via2
- met3 → met4: via3
- met4 → met5: via4

## MIM电容计算

电容值估算公式（Sky130）:
```
C = 2.0 fF/μm² × width × length
```

例如：10μm × 10μm = 200 fF

## 多晶硅电阻计算

电阻值估算公式（Sky130 高阻poly）:
```
R = 1000 Ω/□ × length / width × num_series
```

例如：width=1μm, length=10μm, num_series=1 → R ≈ 10kΩ
"""


# ============== 技能工厂函数 ==============

def create_device_skill() -> Skill:
    """创建器件创建技能
    
    封装所有器件创建相关的脚本和资源，提供渐进式披露机制。
    
    Returns:
        Skill: 配置好的器件创建技能实例
    """
    
    device_skill = Skill(
        name='device-creation',
        description='创建基础模拟器件：NMOS、PMOS、MIM电容、多晶硅电阻、Via过孔堆叠。支持指定尺寸、指数、dummy结构等参数。',
        content=DEVICE_SKILL_INSTRUCTIONS,
        resources=[
            SkillResource(
                name='device-reference',
                content=DEVICE_REFERENCE
            )
        ]
    )
    
    # ========== 注册器件创建脚本 ==========
    
    @device_skill.script
    async def create_nmos(
        ctx: RunContext[Any],
        width: float,
        length: Optional[float] = None,
        fingers: int = 1,
        multiplier: int = 1,
        with_dummy: bool = True,
        with_tie: bool = True,
        name: Optional[str] = None
    ) -> str:
        """创建 NMOS 晶体管
        
        创建一个N沟道金属氧化物半导体场效应晶体管，是模拟电路的核心器件。
        
        Args:
            width: 沟道宽度(μm)，影响电流驱动能力，典型值1-10μm
            length: 沟道长度(μm)，默认使用PDK最小长度(0.15μm)
            fingers: 指数(每个MOS的栅极数量)，用于分割大宽度MOS
            multiplier: 并联倍数，用于电流镜等需要精确比例的电路
            with_dummy: 是否添加dummy结构（改善匹配性），匹配敏感电路建议True
            with_tie: 是否添加衬底连接（body tie）
            name: 组件名称，不指定则自动生成唯一ID
            
        Returns:
            JSON格式的创建结果，包含器件名称、端口列表、尺寸等信息
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
    
    @device_skill.script
    async def create_pmos(
        ctx: RunContext[Any],
        width: float,
        length: Optional[float] = None,
        fingers: int = 1,
        multiplier: int = 1,
        with_dummy: bool = True,
        with_tie: bool = True,
        name: Optional[str] = None
    ) -> str:
        """创建 PMOS 晶体管
        
        创建一个P沟道金属氧化物半导体场效应晶体管，常用于电流源、负载管等。
        
        Args:
            width: 沟道宽度(μm)
            length: 沟道长度(μm)，默认使用PDK最小长度
            fingers: 指数
            multiplier: 并联倍数
            with_dummy: 是否添加dummy结构
            with_tie: 是否添加衬底连接(N阱接VDD)
            name: 组件名称
            
        Returns:
            JSON格式的创建结果
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
    
    @device_skill.script
    async def create_mimcap(
        ctx: RunContext[Any],
        width: float,
        length: float,
        name: Optional[str] = None
    ) -> str:
        """创建 MIM 电容
        
        创建金属-绝缘体-金属电容，用于滤波、补偿、采样保持等应用。
        电容值约为 2.0 fF/μm² × width × length。
        
        Args:
            width: 电容宽度(μm)，典型值5-20μm
            length: 电容长度(μm)，典型值5-20μm
            name: 组件名称
            
        Returns:
            JSON格式的创建结果，包含估算电容值
        """
        params: Dict[str, Any] = {"width": width, "length": length}
        if name is not None:
            params["name"] = name
        
        result = ctx.deps.call_tool("create_mimcap", params)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @device_skill.script
    async def create_resistor(
        ctx: RunContext[Any],
        width: float,
        length: float,
        num_series: int = 1,
        name: Optional[str] = None
    ) -> str:
        """创建多晶硅电阻
        
        创建高阻多晶硅电阻，用于偏置网络、反馈电路等。
        电阻值约为 1000Ω/□ × length/width × num_series。
        
        Args:
            width: 电阻宽度(μm)，典型值0.5-2μm
            length: 电阻长度(μm)，典型值5-50μm
            num_series: 串联段数，用于实现大电阻值
            name: 组件名称
            
        Returns:
            JSON格式的创建结果，包含估算电阻值
        """
        params: Dict[str, Any] = {
            "width": width, 
            "length": length, 
            "num_series": num_series
        }
        if name is not None:
            params["name"] = name
        
        result = ctx.deps.call_tool("create_resistor", params)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @device_skill.script
    async def create_via_stack(
        ctx: RunContext[Any],
        from_layer: str,
        to_layer: str,
        size: Optional[List[float]] = None,
        name: Optional[str] = None
    ) -> str:
        """创建层间 Via 堆叠
        
        创建连接不同金属层的垂直互连结构。
        支持的层: poly, met1, met2, met3, met4, met5
        
        Args:
            from_layer: 起始层 (poly/met1/met2/met3/met4/met5)
            to_layer: 目标层 (met1/met2/met3/met4/met5)
            size: Via尺寸[宽,高](μm)，默认使用最小Via尺寸
            name: 组件名称
            
        Returns:
            JSON格式的创建结果
        """
        params: Dict[str, Any] = {
            "from_layer": from_layer, 
            "to_layer": to_layer
        }
        if size is not None:
            params["size"] = size
        if name is not None:
            params["name"] = name
        
        result = ctx.deps.call_tool("create_via_stack", params)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    # ========== 动态资源 ==========
    
    @device_skill.resource
    async def get_current_pdk_info(ctx: RunContext[Any]) -> str:
        """获取当前PDK信息
        
        返回当前激活的PDK名称和基本设计规则信息。
        """
        try:
            layout_ctx = ctx.deps.mcp_server.state_handler.get_context()
            if layout_ctx:
                pdk_name = layout_ctx.pdk_name or "unknown"
                return f"""## 当前PDK信息

- **PDK名称**: {pdk_name}
- **设计名称**: {layout_ctx.design_name}

如需详细参数范围，请参考 `device-reference` 资源。
"""
            return "PDK未初始化，请先调用初始化。"
        except Exception as e:
            return f"获取PDK信息失败: {str(e)}"
    
    return device_skill
