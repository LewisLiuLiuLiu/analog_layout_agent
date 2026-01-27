"""
电路构建技能模块 (Circuit Building Skill)

使用 PydanticAI Skills 封装复合电路构建工具，实现渐进式披露。
包含电流镜、差分对等常用模拟电路模块的自动化构建。

该技能封装了 CircuitBuilder 的功能，提供高层次的电路构建接口，
自动处理器件创建、互指式布局、内部连接等复杂操作。
"""

from __future__ import annotations

import json
from typing import Optional, Dict, Any

from pydantic_ai import RunContext
from pydantic_ai_skills import Skill, SkillResource


# ============== 技能指令文档 ==============

CIRCUIT_SKILL_INSTRUCTIONS = """
# 电路构建技能 (Circuit Building Skill)

## 何时使用此技能

当你需要创建完整的模拟电路模块时使用此技能：
- **电流镜**: 用于偏置电流复制和镜像
- **差分对**: 运放和比较器的核心输入级

这些是高层次的电路构建命令，会自动：
1. 创建所需的晶体管
2. 应用互指式布局
3. 完成内部连接
4. 导出统一的端口

## 可用电路类型

### 1. 电流镜 (Current Mirror)

用于产生与参考电流成比例的输出电流。

**特点**:
- 自动互指式布局
- 支持NMOS/PMOS
- 可配置镜像比例

**典型应用**:
- 偏置电流源
- 有源负载
- 电流DAC

### 2. 差分对 (Differential Pair)

用于放大两个输入之间的差异。

**特点**:
- 自动互指式布局
- 高共模抑制
- 支持多种布局风格

**典型应用**:
- 运放输入级
- 比较器输入级
- 混频器

## 脚本列表

- `create_current_mirror`: 创建电流镜电路
- `create_diff_pair`: 创建差分对电路

## 使用示例

### 创建NMOS电流镜
```python
run_skill_script(
    skill_name="circuit-building",
    script_name="create_current_mirror",
    args={
        "device_type": "nmos",
        "width": 3.0,
        "numcols": 4,
        "with_dummy": True
    }
)
```

### 创建差分对
```python
run_skill_script(
    skill_name="circuit-building",
    script_name="create_diff_pair",
    args={
        "device_type": "nmos",
        "width": 5.0,
        "fingers": 2,
        "layout_style": "interdigitized"
    }
)
```

## 输出端口说明

### 电流镜端口
| 端口名 | 说明 |
|--------|------|
| `input` | 参考电流输入（diode连接端） |
| `output` | 镜像电流输出 |
| `vdd/vss` | 电源/地连接 |

### 差分对端口
| 端口名 | 说明 |
|--------|------|
| `inp` | 正输入 |
| `inn` | 负输入 |
| `outp` | 正输出 |
| `outn` | 负输出 |
| `tail` | 尾电流连接点 |

## 注意事项

- 电路构建后会返回组合后的组件，内部器件不再单独可见
- 建议先创建电路模块，再进行顶层放置和连接
- 互指列数影响匹配性能，建议3-7列
"""


CIRCUIT_REFERENCE = """
# 电路构建参考文档

## 电流镜设计指南

### 基本原理

电流镜通过将一个晶体管接成diode配置（栅漏短接），
使另一个晶体管复制其电流。

```
        VDD
         │
    ┌────┴────┐
    │   M1    │ ← 源管 (diode连接)
    │ G─┬─D  │
    │   │    │
    └───┼────┘
        │
        │ (gate共接)
        │
    ┌───┼────┐
    │   │    │
    │ G─┘ D  │ ← 镜像管
    │   M2   │
    └────┬───┘
         │
        Iout
```

### 镜像比关系

```
Iout/Iin = (W2/L2) / (W1/L1) × (multiplier2/multiplier1)
```

对于1:1镜像，两管参数应完全相同。

### 互指列数选择

| 精度要求 | 建议列数 | 预期失配 |
|---------|---------|---------|
| 一般 | 2-3 | ~1% |
| 较高 | 4-5 | ~0.5% |
| 高精度 | 6-8 | ~0.2% |

### 设计建议

1. **长度选择**: 使用较长的沟道长度可减少沟道长度调制效应
2. **宽度选择**: 较大宽度可减少失配，但增加面积和寄生电容
3. **dummy结构**: 始终启用with_dummy以改善边缘效应
4. **衬底连接**: 始终启用with_tie确保衬底电位稳定

## 差分对设计指南

### 基本原理

差分对放大两个输入之间的差值，抑制共模信号。

```
              VDD
               │
    ┌──────────┴──────────┐
    │                     │
┌───┴───┐             ┌───┴───┐
│ Load1 │             │ Load2 │
└───┬───┘             └───┬───┘
    │ Voutn           Voutp │
    │                     │
    └──────┬───────┬──────┘
           │       │
       ┌───┴───┬───┴───┐
       │  M1   │  M2   │ ← 差分对
       │ Vinp  │ Vinn  │
       └───┬───┴───┬───┘
           │       │
           └───┬───┘
               │
           ┌───┴───┐
           │ Mtail │ ← 尾电流
           └───┬───┘
               │
              GND
```

### 关键性能指标

| 指标 | 影响因素 |
|------|---------|
| 增益 | gm × Rload |
| 带宽 | 1/(Rload × Cload) |
| 输入失调 | 晶体管匹配 |
| 共模抑制 | 尾电流源阻抗 |

### 布局风格对比

| 风格 | 匹配性 | 复杂度 | 适用场景 |
|------|--------|--------|---------|
| interdigitized | 良好 | 中等 | 一般应用 |
| common_centroid | 最佳 | 较高 | 高精度应用 |

### 设计建议

1. **尺寸选择**: 差分管宽度取决于gm需求，典型5-20μm
2. **匹配要求**: 使用互指式或共质心布局
3. **尾电流**: 尾电流源应有足够高的输出阻抗
4. **负载**: 负载管也应采用互指式布局

## 复合电路示例

### 简单两级运放

```
Stage 1: 差分输入级 (create_diff_pair)
  ↓
Stage 2: 共源放大级 (单管NMOS/PMOS)
  ↓
Output: 输出缓冲 (可选)
```

### 折叠共源共栅

```
Input: PMOS差分对
  ↓
Cascode: NMOS共栅管
  ↓
Load: PMOS电流镜负载
```
"""


# ============== 技能工厂函数 ==============

def create_circuit_skill() -> Skill:
    """创建电路构建技能
    
    封装复合电路构建功能，提供高层次的电路模块创建接口。
    
    Returns:
        Skill: 配置好的电路构建技能实例
    """
    
    circuit_skill = Skill(
        name='circuit-building',
        description='构建复合模拟电路模块：电流镜、差分对等。自动处理器件创建、互指式布局和内部连接，返回统一的电路模块。',
        content=CIRCUIT_SKILL_INSTRUCTIONS,
        resources=[
            SkillResource(
                name='circuit-reference',
                content=CIRCUIT_REFERENCE
            )
        ]
    )
    
    # ========== 注册电路构建脚本 ==========
    
    @circuit_skill.script
    async def create_current_mirror(
        ctx: RunContext[Any],
        device_type: str = "nmos",
        width: float = 3.0,
        length: Optional[float] = None,
        numcols: int = 3,
        with_dummy: bool = True,
        with_tie: bool = True,
        name: Optional[str] = None
    ) -> str:
        """创建电流镜电路
        
        使用互指式布局创建电流镜，自动处理器件创建和内部连接。
        电流镜包含源管（diode连接）和镜像管，输出与输入电流成比例。
        
        Args:
            device_type: 器件类型，"nmos"或"pmos"
            width: 管子宽度(μm)，影响电流能力
            length: 管子长度(μm)，默认使用PDK最小长度
            numcols: 互指列数，影响匹配性能（建议3-7列）
            with_dummy: 是否添加dummy结构（强烈建议True）
            with_tie: 是否添加衬底连接
            name: 电路名称，不指定则自动生成
            
        Returns:
            JSON格式的创建结果，包含电路端口和尺寸信息
        """
        # 使用 CircuitBuilder 创建电流镜
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
    
    @circuit_skill.script
    async def create_diff_pair(
        ctx: RunContext[Any],
        device_type: str = "nmos",
        width: float = 5.0,
        length: Optional[float] = None,
        fingers: int = 1,
        numcols: int = 2,
        layout_style: str = "interdigitized",
        name: Optional[str] = None
    ) -> str:
        """创建差分对电路
        
        创建用于运放和比较器输入级的差分对，是模拟电路的核心模块。
        自动应用互指式或共质心布局以确保匹配。
        
        Args:
            device_type: 器件类型，"nmos"或"pmos"
            width: 管子宽度(μm)，影响gm和电流
            length: 管子长度(μm)，默认使用PDK最小长度
            fingers: 指数，用于分割大宽度
            numcols: 互指列数（默认2）
            layout_style: 布局风格
                - "interdigitized": 互指式（推荐）
                - "common_centroid": 共质心（最佳匹配）
            name: 电路名称
            
        Returns:
            JSON格式的创建结果，包含inp/inn/outp/outn/tail等端口
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
    
    return circuit_skill
