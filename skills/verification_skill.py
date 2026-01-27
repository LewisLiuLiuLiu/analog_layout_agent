"""
验证技能模块 (Verification Skill)

使用 PydanticAI Skills 封装设计验证工具，实现渐进式披露。
包含DRC检查、网表提取、LVS验证等功能。

该技能封装了 VerificationEngine 的功能，提供自动化的设计规则检查
和错误修复建议。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai_skills import Skill, SkillResource


# ============== 技能指令文档 ==============

VERIFICATION_SKILL_INSTRUCTIONS = """
# 验证技能 (Verification Skill)

## 何时使用此技能

当你需要验证版图设计是否符合工艺要求时使用此技能：
- **DRC检查**: 检测设计规则违规（间距、宽度、重叠等）
- **网表提取**: 从版图提取电路网表
- **修复建议**: 获取DRC违规的自动修复建议

## 验证流程

### 典型验证工作流

```
1. 完成版图设计
   ↓
2. 执行 run_drc 检查设计规则
   ↓
3. 如有违规 → get_drc_fix_suggestions 获取修复建议
   ↓
4. 根据建议修改设计
   ↓
5. 重复步骤2-4直到DRC通过
   ↓
6. extract_netlist 提取网表进行功能验证
```

## 脚本列表

- `run_drc`: 执行设计规则检查
- `extract_netlist`: 提取版图网表
- `get_drc_fix_suggestions`: 获取DRC违规的修复建议

## 使用示例

### 执行DRC检查
```python
run_skill_script(
    skill_name="verification-drc",
    script_name="run_drc"
)
```

### 获取修复建议
```python
run_skill_script(
    skill_name="verification-drc",
    script_name="get_drc_fix_suggestions"
)
```

### 提取网表
```python
run_skill_script(
    skill_name="verification-drc",
    script_name="extract_netlist"
)
```

## DRC违规类型

### 常见违规类别

| 类别 | 描述 | 典型原因 |
|------|------|---------|
| `spacing` | 间距违规 | 器件/走线过近 |
| `width` | 宽度违规 | 走线/器件过窄 |
| `enclosure` | 包围违规 | Via未被金属充分覆盖 |
| `overlap` | 重叠违规 | 不同层意外重叠 |
| `density` | 密度违规 | 金属密度过高/过低 |

### 违规严重程度

| 级别 | 说明 | 处理方式 |
|------|------|---------|
| ERROR | 必须修复 | 流片前必须解决 |
| WARNING | 建议修复 | 可能影响成品率 |
| INFO | 信息提示 | 仅供参考 |

## 修复建议解读

修复建议包含以下信息：
- `violation_type`: 违规类型
- `location`: 违规位置坐标
- `affected_layers`: 涉及的层
- `suggested_action`: 建议的修复动作
- `parameters`: 建议的参数调整

## 注意事项

- DRC检查可能需要几秒钟，取决于设计复杂度
- 某些违规可能有多种修复方式
- 修复一个违规可能引入新的违规，需迭代检查
- 网表提取前应确保DRC干净
"""


VERIFICATION_REFERENCE = """
# 验证参考文档

## Sky130 PDK 关键设计规则

### 金属层规则

| 层 | 最小宽度 | 最小间距 | 说明 |
|----|---------|---------|------|
| met1 | 0.14μm | 0.14μm | 最底层金属 |
| met2 | 0.14μm | 0.14μm | 常用布线层 |
| met3 | 0.30μm | 0.30μm | 较宽 |
| met4 | 0.30μm | 0.30μm | 电源分配 |
| met5 | 1.60μm | 1.60μm | 顶层厚金属 |

### Via规则

| Via类型 | 最小尺寸 | 金属包围 |
|---------|---------|---------|
| via1 | 0.15×0.15μm | 0.055μm |
| via2 | 0.20×0.20μm | 0.065μm |
| via3 | 0.20×0.20μm | 0.065μm |
| via4 | 0.80×0.80μm | 0.19μm |

### 晶体管规则

| 规则 | NMOS | PMOS | 说明 |
|------|------|------|------|
| 最小栅长 | 0.15μm | 0.15μm | 技术节点限制 |
| 最小栅宽 | 0.42μm | 0.55μm | 最小有效宽度 |
| 栅到接触 | 0.15μm | 0.15μm | poly到contact |

## 常见DRC问题及解决方案

### 1. 金属间距违规

**问题**: 两条金属走线过近
```
[ERROR] M2.S1: Metal2 spacing < 0.14um at (10.5, 20.3)
```

**解决方案**:
- 增大走线间距
- 使用不同金属层
- 调整器件位置

### 2. Via包围违规

**问题**: Via周围金属不足
```
[ERROR] V1.EN1: Via1 metal1 enclosure < 0.055um
```

**解决方案**:
- 增大Via周围的金属面积
- 使用自动Via生成确保包围

### 3. 金属宽度违规

**问题**: 金属走线过窄
```
[ERROR] M1.W1: Metal1 width < 0.14um at (5.2, 10.1)
```

**解决方案**:
- 增大金属宽度参数
- 检查路由设置

### 4. 密度违规

**问题**: 金属密度不符合要求
```
[WARNING] M1.DN1: Metal1 density < 20% in region
```

**解决方案**:
- 添加填充金属(metal fill)
- 调整布局分布

## 网表格式

### SPICE网表示例

```spice
* Netlist extracted from layout
* Design: current_mirror

.SUBCKT current_mirror input output vdd vss

M1 input input vss vss sky130_fd_pr__nfet_01v8 W=3u L=0.15u
M2 output input vss vss sky130_fd_pr__nfet_01v8 W=3u L=0.15u

.ENDS current_mirror
```

### 网表元素说明

| 元素 | 说明 |
|------|------|
| 子电路定义 | .SUBCKT ... .ENDS |
| MOS管 | M<name> D G S B <model> W=... L=... |
| 电阻 | R<name> N1 N2 <value> |
| 电容 | C<name> N1 N2 <value> |

## LVS检查要点

### 什么是LVS？

LVS (Layout vs Schematic) 比较版图提取的网表与原理图网表是否一致。

### 常见LVS错误

| 错误类型 | 描述 | 常见原因 |
|---------|------|---------|
| NET mismatch | 网络不匹配 | 连接错误或遗漏 |
| DEVICE mismatch | 器件不匹配 | 参数错误 |
| PORT mismatch | 端口不匹配 | 端口名称或数量错误 |
| SHORT | 短路 | 不应连接的网络连在一起 |
| OPEN | 开路 | 应连接的网络未连接 |
"""


# ============== 技能工厂函数 ==============

def create_verification_skill() -> Skill:
    """创建验证技能
    
    封装设计验证功能，提供DRC检查和修复建议。
    
    Returns:
        Skill: 配置好的验证技能实例
    """
    
    verification_skill = Skill(
        name='verification-drc',
        description='设计规则验证：DRC检查、网表提取、修复建议。检测间距/宽度/包围等违规，提供自动修复方案。',
        content=VERIFICATION_SKILL_INSTRUCTIONS,
        resources=[
            SkillResource(
                name='verification-reference',
                content=VERIFICATION_REFERENCE
            )
        ]
    )
    
    # ========== 注册验证脚本 ==========
    
    @verification_skill.script
    async def run_drc(ctx: RunContext[Any]) -> str:
        """执行设计规则检查（DRC）
        
        检查当前版图是否符合PDK的设计规则要求，
        包括金属间距、宽度、Via包围、密度等检查。
        
        Returns:
            JSON格式的DRC结果，包含：
            - total_violations: 违规总数
            - errors: 错误级别违规列表
            - warnings: 警告级别违规列表
            - passed: 是否通过（无ERROR）
        """
        result = ctx.deps.verification_engine.run_drc()
        if hasattr(result, 'to_dict'):
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @verification_skill.script
    async def extract_netlist(ctx: RunContext[Any]) -> str:
        """提取版图网表
        
        从当前版图提取SPICE格式的电路网表，
        用于仿真验证或LVS比对。
        
        Returns:
            JSON格式的网表信息，包含：
            - devices: 器件列表及参数
            - nets: 网络连接关系
            - ports: 外部端口
            - spice_netlist: SPICE格式网表字符串
        """
        result = ctx.deps.verification_engine.extract_netlist()
        if hasattr(result, 'to_dict'):
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @verification_skill.script
    async def get_drc_fix_suggestions(ctx: RunContext[Any]) -> str:
        """获取DRC违规的修复建议
        
        先执行DRC检查，然后分析违规并提供具体的修复建议。
        每个建议包含修复动作、目标参数和建议值。
        
        Returns:
            JSON格式的分析结果，包含：
            - drc_summary: DRC检查摘要
            - violations: 违规列表
            - suggestions: 修复建议列表，每个包含：
                - violation_type: 违规类型
                - location: 位置
                - suggested_action: 建议动作
                - parameters: 建议参数
                - priority: 修复优先级
        """
        # 导入DRC分析器
        from ..core.drc_advisor import analyze_drc_result
        
        # 执行DRC
        drc_result = ctx.deps.verification_engine.run_drc()
        
        # 获取PDK名称
        layout_ctx = ctx.deps.mcp_server.state_handler.get_context()
        pdk_name = layout_ctx.pdk_name if layout_ctx else "sky130"
        
        # 分析并生成建议
        analysis = analyze_drc_result(drc_result, pdk_name)
        
        return json.dumps(analysis, ensure_ascii=False, indent=2)
    
    return verification_skill
