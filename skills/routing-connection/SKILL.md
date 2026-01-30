---
name: routing-connection
description: 智能布线连接：支持smart_route自动寻路、c_route/l_route手动布线、straight_route直连等多种布线方式，用于连接器件端口。
---

# 布线连接技能 (Routing Connection Skill)

## 何时使用此技能

当你需要连接器件端口时使用此技能：

- **smart_route**: 自动寻路布线，智能避障和层切换
- **c_route**: C形布线，适合需要绕行的连接
- **l_route**: L形布线，适合直角转弯连接
- **straight_route**: 直连布线，适合简单直线连接

## 使用流程

1. 确定需要连接的源端口和目标端口
2. 选择合适的布线方式（推荐先尝试 smart_route）
3. 指定布线层（默认 met1）
4. 如布线失败，尝试其他方式或调整端口位置

## 可用脚本

| 脚本路径 | 说明 | 必需参数 | 可选参数 |
|--------|------|----------|----------|
| `scripts/smart_route.py` | 智能自动布线 | `--source`, `--target` | `--layer`, `--width`, `--via-spacing` |
| `scripts/c_route.py` | C形布线 | `--source`, `--target` | `--layer`, `--width`, `--extension` |
| `scripts/l_route.py` | L形布线 | `--source`, `--target` | `--layer`, `--width` |
| `scripts/straight_route.py` | 直连布线 | `--source`, `--target` | `--layer`, `--width` |

## 端口命名规则

端口格式为 `组件名.端口名`，例如：
- `M1.drain_N` - 组件M1的N侧drain端口
- `M2.gate_W` - 组件M2的W侧gate端口
- `current_mirror_1.output` - 电流镜的输出端口

### 常见端口名称

| 器件类型 | 可用端口 |
|----------|----------|
| NMOS/PMOS | `drain_N`, `drain_S`, `gate_E`, `gate_W`, `source_N`, `source_S`, `bulk` |
| 电流镜 | `input`, `output`, `vss/vdd` |
| 差分对 | `inp`, `inn`, `outp`, `outn`, `tail` |

## 布线层选择

| 布线层 | 推荐用途 | 最小宽度 |
|--------|----------|----------|
| met1 | 短距离局部连接 | 0.14μm |
| met2 | 中距离水平连接 | 0.14μm |
| met3 | 中距离垂直连接 | 0.30μm |
| met4 | 长距离连接、电源线 | 0.30μm |
| met5 | 顶层电源/地线 | 1.60μm |

## 使用示例

### 智能布线（推荐）

自动连接两个NMOS的drain：

```
run_skill_script(
    skill_name="routing-connection",
    script_name="scripts/smart_route.py",
    args=["--source", "M1.drain_N", "--target", "M2.drain_S"]
)
```

### 指定布线层

使用met2层布线：

```
run_skill_script(
    skill_name="routing-connection",
    script_name="scripts/smart_route.py",
    args=["--source", "M1.gate_E", "--target", "M2.gate_W", "--layer", "met2"]
)
```

### L形布线

当smart_route失败时尝试L形布线：

```
run_skill_script(
    skill_name="routing-connection",
    script_name="scripts/l_route.py",
    args=["--source", "M1.drain_N", "--target", "R1.port_a"]
)
```

### C形布线（绕行）

需要绕过障碍物时：

```
run_skill_script(
    skill_name="routing-connection",
    script_name="scripts/c_route.py",
    args=["--source", "M1.source_S", "--target", "M3.source_S", "--extension", "5.0"]
)
```

## 布线策略建议

1. **优先使用 smart_route**
   - 自动处理障碍物避让
   - 自动选择最优路径
   - 自动添加必要的via

2. **布线顺序**
   - 先布关键信号线（如差分对）
   - 再布普通信号线
   - 最后布电源地线

3. **层分配原则**
   - 信号线用低层金属（met1/met2）
   - 电源线用高层金属（met4/met5）
   - 避免信号线跨越多层

## 注意事项

- 布线前确保器件已正确放置（使用 placement-layout 技能）
- 端口名称区分大小写
- 布线宽度过小可能导致 DRC 违规
- smart_route 可能因空间限制失败，此时尝试手动布线方式

## 相关资源

- `references/routing_rules.md` - 布线设计规则详解
- `references/layer_stack.md` - 金属层堆叠结构
