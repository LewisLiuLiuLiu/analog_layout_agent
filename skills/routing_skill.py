"""
路由连接技能模块 (Routing Connection Skill)

使用 PydanticAI Skills 封装路由相关工具，实现渐进式披露。
包含智能路由、C型路由、L型路由、直线路由等多种布线方式。

该技能通过 MCP Server 的统一工具调用入口与底层实现交互，
自动分析端口方向选择最优路由策略。
"""

from __future__ import annotations

import json
from typing import Optional, Dict, Any

from pydantic_ai import RunContext
from pydantic_ai_skills import Skill, SkillResource


# ============== 技能指令文档 ==============

ROUTING_SKILL_INSTRUCTIONS = """
# 路由连接技能 (Routing Connection Skill)

## 何时使用此技能

当你需要连接两个器件的端口时使用此技能，适用于：
- 连接晶体管的源/漏/栅极端口
- 连接电容、电阻的端口
- 创建电源和地线连接
- 构建信号通路

## 可用路由类型

| 路由类型 | 适用场景 | 描述 |
|---------|---------|------|
| `smart_route` | **推荐首选** | 自动分析端口方向，选择最优路由 |
| `straight_route` | 两端口共线 | 直接连接，无转弯 |
| `l_route` | 垂直端口 | L型连接（一个朝上一个朝右） |
| `c_route` | 同向平行端口 | C型连接（两个都朝右） |

## 端口命名格式

端口使用 `组件名.端口名` 格式，例如：
- `nmos_1.drain_E` - NMOS的漏极东向端口
- `pmos_2.gate_N` - PMOS的栅极北向端口
- `current_mirror_1.output` - 电流镜的输出端口
- `cap_1.top` - 电容的顶板端口

### 常用端口名称

**晶体管端口**:
- `drain_E`, `drain_W` - 漏极（东/西向）
- `source_E`, `source_W` - 源极（东/西向）
- `gate_N`, `gate_S` - 栅极（北/南向）

**电容端口**:
- `top`, `bottom` - 顶板/底板

**电阻端口**:
- `port_a`, `port_b` - 两端端口

## 金属层选择

| 层名 | 用途 | 说明 |
|------|------|------|
| `met1` | 本地互连 | 最底层金属，常用于器件内部连接 |
| `met2` | 水平走线 | **默认层**，用于大部分水平连接 |
| `met3` | 垂直走线 | 用于跨越met2走线 |
| `met4` | 电源分配 | 较宽，用于电源轨 |
| `met5` | 顶层电源 | 最宽，用于主电源/地 |

## 使用示例

### 使用智能路由（推荐）
```python
run_skill_script(
    skill_name="routing-connection",
    script_name="smart_route",
    args={
        "source_port": "nmos_1.drain_E",
        "dest_port": "pmos_1.drain_E",
        "layer": "met2"
    }
)
```

### 使用L型路由
```python
run_skill_script(
    skill_name="routing-connection", 
    script_name="l_route",
    args={
        "source_port": "nmos_1.gate_N",
        "dest_port": "input_pad.signal",
        "layer": "met2"
    }
)
```

## 使用建议

1. **优先使用 `smart_route`**：它会自动分析端口方向选择最佳路由
2. **指定正确的金属层**：默认met2，复杂电路可能需要met3避免拥塞
3. **检查端口方向**：确保端口存在且方向正确
4. **复杂路由分段**：非常复杂的路由可能需要多段连接

## 常见问题

- **路由失败**: 检查端口名称是否正确，组件是否已创建
- **DRC违规**: 可能需要调整金属层或添加Via
- **连接不上**: 尝试使用不同的路由类型
"""


ROUTING_REFERENCE = """
# 路由参考文档

## 路由类型详解

### 1. smart_route（智能路由）

**工作原理**:
1. 分析源端口和目标端口的方向
2. 计算最短路径
3. 自动选择合适的路由类型（直线/L型/C型）
4. 自动添加必要的Via

**适用场景**: 
- 不确定用哪种路由时
- 快速原型设计
- 大部分常规连接

### 2. straight_route（直线路由）

**工作原理**:
直接在两个共线端口之间画一条直线金属

**适用场景**:
- 两个端口在同一条线上
- 端口方向相对（一个朝左，一个朝右）

**示意图**:
```
    [Port A] ═══════════════ [Port B]
```

### 3. l_route（L型路由）

**工作原理**:
在两个垂直方向的端口之间画L型连接

**适用场景**:
- 一个端口朝上/下，另一个朝左/右
- 需要转一个弯

**示意图**:
```
    [Port A]
        ║
        ║
        ╚═════════ [Port B]
```

### 4. c_route（C型路由）

**工作原理**:
在两个同向平行端口之间画C型连接

**适用场景**:
- 两个端口都朝同一方向
- 需要绕过障碍物

**示意图**:
```
    [Port A] ═══╗
                ║
                ║
    [Port B] ═══╝
```

## 端口方向参考

| 方向后缀 | 含义 | 角度 |
|---------|------|------|
| `_E` | 东（右） | 0° |
| `_N` | 北（上） | 90° |
| `_W` | 西（左） | 180° |
| `_S` | 南（下） | 270° |

## 常见连接模式

### 电流镜连接
```
源管漏极 → 镜像管漏极（共接）
源管栅极 → 镜像管栅极（共接）
源管栅极 → 源管漏极（diode连接）
```

### 差分对连接
```
M1源极 → M2源极（尾电流节点）
M1漏极 → 负载1
M2漏极 → 负载2
```
"""


# ============== 技能工厂函数 ==============

def create_routing_skill() -> Skill:
    """创建路由连接技能
    
    封装所有路由相关的脚本，支持多种布线方式。
    
    Returns:
        Skill: 配置好的路由连接技能实例
    """
    
    routing_skill = Skill(
        name='routing-connection',
        description='智能路由连接器件端口：支持自动路由、直线路由、L型路由、C型路由。自动分析端口方向选择最优连接方式。',
        content=ROUTING_SKILL_INSTRUCTIONS,
        resources=[
            SkillResource(
                name='routing-reference',
                content=ROUTING_REFERENCE
            )
        ]
    )
    
    # ========== 注册路由脚本 ==========
    
    @routing_skill.script
    async def smart_route(
        ctx: RunContext[Any],
        source_port: str,
        dest_port: str,
        layer: str = "met2"
    ) -> str:
        """智能路由连接两个端口
        
        自动分析端口方向并选择最优路由策略（直线/L型/C型）。
        这是推荐的默认路由方式。
        
        Args:
            source_port: 源端口，格式为 "组件名.端口名"，如 "nmos_1.drain_E"
            dest_port: 目标端口，格式同上
            layer: 路由金属层，默认met2
            
        Returns:
            JSON格式的路由结果，包含路由类型和路径信息
        """
        result = ctx.deps.call_tool("smart_route", {
            "source_port": source_port,
            "dest_port": dest_port,
            "layer": layer
        })
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @routing_skill.script
    async def c_route(
        ctx: RunContext[Any],
        source_port: str,
        dest_port: str,
        extension: Optional[float] = None,
        layer: str = "met2"
    ) -> str:
        """C型路由连接
        
        适用于两个同向平行端口的连接（如两个都朝右的端口）。
        路由形成C形，绕过中间区域。
        
        Args:
            source_port: 源端口
            dest_port: 目标端口
            extension: 延伸长度(μm)，默认自动计算
            layer: 路由金属层
            
        Returns:
            JSON格式的路由结果
        """
        params: Dict[str, Any] = {
            "source_port": source_port, 
            "dest_port": dest_port, 
            "layer": layer
        }
        if extension is not None:
            params["extension"] = extension
        
        result = ctx.deps.call_tool("c_route", params)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @routing_skill.script
    async def l_route(
        ctx: RunContext[Any],
        source_port: str,
        dest_port: str,
        layer: str = "met2"
    ) -> str:
        """L型路由连接
        
        适用于垂直方向端口的连接（如一个朝上一个朝右）。
        路由形成L形，在拐点处转弯。
        
        Args:
            source_port: 源端口
            dest_port: 目标端口
            layer: 路由金属层
            
        Returns:
            JSON格式的路由结果
        """
        result = ctx.deps.call_tool("l_route", {
            "source_port": source_port,
            "dest_port": dest_port,
            "layer": layer
        })
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @routing_skill.script
    async def straight_route(
        ctx: RunContext[Any],
        source_port: str,
        dest_port: str,
        layer: str = "met2"
    ) -> str:
        """直线路由连接
        
        适用于共线端口的直接连接，两端口需在同一直线上。
        这是最简单的路由方式，无转弯。
        
        Args:
            source_port: 源端口
            dest_port: 目标端口
            layer: 路由金属层
            
        Returns:
            JSON格式的路由结果
        """
        result = ctx.deps.call_tool("straight_route", {
            "source_port": source_port,
            "dest_port": dest_port,
            "layer": layer
        })
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    return routing_skill
