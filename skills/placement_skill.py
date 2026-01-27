"""
放置布局技能模块 (Placement Layout Skill)

使用 PydanticAI Skills 封装放置相关工具，实现渐进式披露。
包含组件放置、移动、对齐、互指式布局等功能。

该技能用于控制器件在版图中的物理位置，
是实现高性能模拟电路布局的关键环节。
"""

from __future__ import annotations

import json
from typing import Optional, Dict, Any

from pydantic_ai import RunContext
from pydantic_ai_skills import Skill, SkillResource


# ============== 技能指令文档 ==============

PLACEMENT_SKILL_INSTRUCTIONS = """
# 放置布局技能 (Placement Layout Skill)

## 何时使用此技能

当你需要控制器件在版图中的物理位置时使用此技能：
- 将器件放置到指定坐标
- 移动已放置的器件
- 将器件对齐到其他器件的端口
- 使用互指式布局改善匹配性

## 可用放置操作

| 操作 | 用途 | 说明 |
|------|------|------|
| `place_component` | 绝对放置 | 将器件放置到指定(x,y)坐标 |
| `move_component` | 相对移动 | 按相对位移移动器件 |
| `align_to_port` | 端口对齐 | 将器件对齐到另一个器件的端口 |
| `interdigitize` | 互指式放置 | 两个晶体管交叉放置，改善匹配 |

## 坐标系统

- 原点(0, 0)位于设计区域左下角
- X轴正向为右，Y轴正向为上
- 单位为微米(μm)
- 旋转角度为逆时针方向

## 放置策略建议

### 模拟电路布局原则

1. **对称性**: 差分电路应关于中心轴对称
2. **紧凑性**: 相关器件紧密放置减少寄生
3. **匹配性**: 使用互指式布局改善晶体管匹配
4. **热对称**: 避免热敏感器件靠近发热源

### 典型布局流程

```
1. 放置核心器件（差分对/电流镜）
2. 放置负载器件
3. 放置偏置器件
4. 添加去耦电容
5. 调整对齐和间距
```

## 脚本列表

- `place_component`: 放置组件到指定位置
- `move_component`: 移动组件（相对位移）
- `align_to_port`: 将组件对齐到目标端口
- `interdigitize`: 互指式放置两个晶体管

## 使用示例

### 放置器件到指定位置
```python
run_skill_script(
    skill_name="placement-layout",
    script_name="place_component",
    args={
        "component_name": "nmos_1",
        "x": 10.0,
        "y": 5.0,
        "rotation": 0
    }
)
```

### 对齐到另一个器件的端口
```python
run_skill_script(
    skill_name="placement-layout",
    script_name="align_to_port",
    args={
        "component_name": "pmos_1",
        "target_port": "nmos_1.drain_E",
        "alignment": "center",
        "offset_y": 5.0
    }
)
```

### 互指式放置（电流镜/差分对）
```python
run_skill_script(
    skill_name="placement-layout",
    script_name="interdigitize",
    args={
        "comp_a": "nmos_m1",
        "comp_b": "nmos_m2",
        "num_cols": 4,
        "layout_style": "ABAB"
    }
)
```

## 注意事项

- 放置前请确保器件已创建
- 旋转只支持0/90/180/270度
- 互指式布局要求两个器件参数相同
- 放置后可能需要重新路由
"""


PLACEMENT_REFERENCE = """
# 放置布局参考文档

## 互指式布局详解

### 什么是互指式布局？

互指式布局(Interdigitated Layout)将两个需要匹配的晶体管交错放置，
而不是简单地并排放置。这种布局可以平均化工艺梯度的影响，显著改善匹配性。

### 布局风格对比

**ABAB 风格（推荐）**:
```
┌───┬───┬───┬───┐
│ A │ B │ A │ B │
└───┴───┴───┴───┘
```
- 优点: 简单、布线方便
- 适用: 一般匹配要求

**ABBA 风格**:
```
┌───┬───┬───┬───┐
│ A │ B │ B │ A │
└───┴───┴───┴───┘
```
- 优点: 对线性梯度更好的免疫
- 适用: 高精度匹配

**Common Centroid 风格**:
```
┌───┬───┬───┬───┐
│ A │ B │ B │ A │
├───┼───┼───┼───┤
│ B │ A │ A │ B │
└───┴───┴───┴───┘
```
- 优点: 最佳匹配性能
- 适用: 最高精度要求

### 互指列数建议

| 应用 | 建议列数 | 说明 |
|------|---------|------|
| 一般电流镜 | 2-4 | 基本匹配 |
| 精密电流镜 | 4-8 | 较好匹配 |
| 差分对 | 2-4 | 关键匹配 |
| 带隙基准 | 8+ | 最高精度 |

## 对齐方式详解

### alignment 参数选项

| 值 | 对齐方式 | 说明 |
|----|---------|------|
| `center` | 中心对齐 | 组件中心对齐到目标点 |
| `left` | 左对齐 | 组件左边界对齐到目标点 |
| `right` | 右对齐 | 组件右边界对齐到目标点 |
| `top` | 顶部对齐 | 组件顶边界对齐到目标点 |
| `bottom` | 底部对齐 | 组件底边界对齐到目标点 |

### 偏移量使用

`offset_x` 和 `offset_y` 用于在对齐基础上添加额外偏移：
- 正值：向右/向上偏移
- 负值：向左/向下偏移
- 单位：微米(μm)

## 典型电路布局模板

### 简单电流镜
```
        VDD
         │
    ┌────┴────┐
    │ PMOS_M1 │ ← 源管
    └────┬────┘
         │
    ┌────┴────┐
    │ NMOS_M1 │ ← diode连接
    └────┬────┘
         │
        GND
```

### 差分放大器
```
           VDD
            │
    ┌───────┴───────┐
    │               │
┌───┴───┐       ┌───┴───┐
│ PMOS1 │       │ PMOS2 │  ← 负载
└───┬───┘       └───┬───┘
    │   OUT-   OUT+  │
    │       ×       │
    │  ┌────┴────┐  │
    └──┤ NMOS1,2 ├──┘  ← 差分对(互指式)
       └────┬────┘
            │
       ┌────┴────┐
       │ NMOS_T  │  ← 尾电流
       └────┬────┘
            │
           GND
```

## 间距建议

| 器件类型 | 最小间距 | 推荐间距 | 说明 |
|---------|---------|---------|------|
| NMOS-NMOS | 0.5μm | 1-2μm | 同类型 |
| PMOS-PMOS | 0.5μm | 1-2μm | 同类型 |
| NMOS-PMOS | 2μm | 3-5μm | 需要隔离 |
| MOS-电容 | 1μm | 2-3μm | 防止噪声耦合 |
"""


# ============== 技能工厂函数 ==============

def create_placement_skill() -> Skill:
    """创建放置布局技能
    
    封装所有放置相关的脚本，支持多种布局方式。
    
    Returns:
        Skill: 配置好的放置布局技能实例
    """
    
    placement_skill = Skill(
        name='placement-layout',
        description='组件放置与布局：绝对放置、相对移动、端口对齐、互指式布局。用于控制器件在版图中的物理位置，实现匹配和对称。',
        content=PLACEMENT_SKILL_INSTRUCTIONS,
        resources=[
            SkillResource(
                name='placement-reference',
                content=PLACEMENT_REFERENCE
            )
        ]
    )
    
    # ========== 注册放置脚本 ==========
    
    @placement_skill.script
    async def place_component(
        ctx: RunContext[Any],
        component_name: str,
        x: float = 0,
        y: float = 0,
        rotation: int = 0
    ) -> str:
        """放置组件到指定位置
        
        将组件放置到版图中的指定坐标位置，支持旋转。
        
        Args:
            component_name: 组件名称
            x: X坐标(μm)，默认0
            y: Y坐标(μm)，默认0
            rotation: 旋转角度，仅支持0/90/180/270度
            
        Returns:
            JSON格式的放置结果，包含新位置信息
        """
        result = ctx.deps.call_tool("place_component", {
            "component_name": component_name,
            "x": x,
            "y": y,
            "rotation": rotation
        })
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @placement_skill.script
    async def move_component(
        ctx: RunContext[Any],
        component_name: str,
        dx: float = 0,
        dy: float = 0
    ) -> str:
        """移动组件（相对位移）
        
        按相对位移移动已放置的组件。
        
        Args:
            component_name: 组件名称
            dx: X方向移动距离(μm)，正值向右
            dy: Y方向移动距离(μm)，正值向上
            
        Returns:
            JSON格式的移动结果，包含新位置信息
        """
        result = ctx.deps.call_tool("move_component", {
            "component_name": component_name,
            "dx": dx,
            "dy": dy
        })
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @placement_skill.script
    async def align_to_port(
        ctx: RunContext[Any],
        component_name: str,
        target_port: str,
        alignment: str = "center",
        offset_x: float = 0,
        offset_y: float = 0
    ) -> str:
        """将组件对齐到目标端口
        
        将组件对齐到另一个组件的端口位置，支持多种对齐方式和偏移。
        常用于建立器件之间的相对位置关系。
        
        Args:
            component_name: 要对齐的组件名称
            target_port: 目标端口，格式"组件名.端口名"
            alignment: 对齐方式(center/left/right/top/bottom)
            offset_x: X方向偏移(μm)
            offset_y: Y方向偏移(μm)
            
        Returns:
            JSON格式的对齐结果
        """
        result = ctx.deps.call_tool("align_to_port", {
            "component_name": component_name,
            "target_port": target_port,
            "alignment": alignment,
            "offset_x": offset_x,
            "offset_y": offset_y
        })
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @placement_skill.script
    async def interdigitize(
        ctx: RunContext[Any],
        comp_a: str,
        comp_b: str,
        num_cols: int = 4,
        layout_style: str = "ABAB"
    ) -> str:
        """互指式放置两个晶体管
        
        将两个晶体管交错放置，用于改善匹配性。
        适用于电流镜、差分对等需要精确匹配的电路。
        
        Args:
            comp_a: 组件A名称（如电流镜的源管）
            comp_b: 组件B名称（如电流镜的镜像管）
            num_cols: 互指列数，影响匹配精度，建议2-8
            layout_style: 布局风格
                - "ABAB": A-B-A-B排列（推荐）
                - "ABBA": A-B-B-A排列（更好的梯度免疫）
                - "common_centroid": 共质心布局（最佳匹配）
            
        Returns:
            JSON格式的布局结果，包含组合后的尺寸和端口
        """
        result = ctx.deps.call_tool("interdigitize", {
            "comp_a": comp_a,
            "comp_b": comp_b,
            "num_cols": num_cols,
            "layout_style": layout_style
        })
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    return placement_skill
