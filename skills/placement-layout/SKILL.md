---
name: placement-layout
description: 组件放置与布局：支持绝对/相对放置、端口对齐、移动组件、互指式排列等，用于优化版图布局和改善匹配性。
---

# 放置布局技能 (Placement Layout Skill)

## 何时使用此技能

当你需要放置和排列器件时使用此技能：

- **place_component**: 将组件放置到指定位置
- **move_component**: 移动已放置的组件
- **align_to_port**: 将组件对齐到另一个组件的端口
- **interdigitize**: 互指式放置，用于改善匹配性

## 使用流程

1. 创建器件（使用 device-creation 技能）
2. 使用 place_component 或 align_to_port 放置器件
3. 如需改善匹配，使用 interdigitize 进行互指式排列
4. 微调位置（使用 move_component）
5. 完成后使用 routing-connection 技能连接端口

## 可用脚本

| 脚本路径 | 说明 | 必需参数 | 可选参数 |
|--------|------|----------|----------|
| `scripts/place_component.py` | 放置组件到指定位置 | `--name`, `--x`, `--y` | `--rotation` |
| `scripts/move_component.py` | 移动组件 | `--name`, `--dx`, `--dy` | - |
| `scripts/align_to_port.py` | 对齐到端口 | `--name`, `--target-port` | `--alignment`, `--offset-x`, `--offset-y` |
| `scripts/interdigitize.py` | 互指式放置 | `--comp-a`, `--comp-b` | `--num-cols`, `--layout-style` |

## 放置策略

### 基本原则

1. **信号流向**: 从左到右或从下到上
2. **匹配敏感器件**: 相邻放置，使用 interdigitize
3. **电源线**: 靠近边缘，便于引出
4. **热敏感器件**: 远离发热源

### 推荐布局顺序

1. 放置核心器件（差分对、电流镜）
2. 放置偏置电路
3. 放置负载/输出级
4. 放置补偿电容/电阻

## 使用示例

### 绝对位置放置

将NMOS放置到坐标(10, 20)：

```
run_skill_script(
    skill_name="placement-layout",
    script_name="scripts/place_component.py",
    args=["--name", "M1", "--x", "10.0", "--y", "20.0"]
)
```

### 对齐到端口

将M2的gate对齐到M1的gate：

```
run_skill_script(
    skill_name="placement-layout",
    script_name="scripts/align_to_port.py",
    args=["--name", "M2", "--target-port", "M1.gate_E", "--alignment", "center"]
)
```

### 互指式放置（电流镜）

将电流镜的两个管子互指放置：

```
run_skill_script(
    skill_name="placement-layout",
    script_name="scripts/interdigitize.py",
    args=["--comp-a", "M_ref", "--comp-b", "M_out", "--num-cols", "4", "--layout-style", "ABAB"]
)
```

### 移动组件微调

将M3向右移动5μm：

```
run_skill_script(
    skill_name="placement-layout",
    script_name="scripts/move_component.py",
    args=["--name", "M3", "--dx", "5.0", "--dy", "0.0"]
)
```

## 互指式布局详解

### 布局风格

| 风格 | 排列方式 | 适用场景 |
|------|----------|----------|
| ABAB | A-B-A-B-... | 简单电流镜 |
| ABBA | A-B-B-A | 差分对，更好的匹配 |
| common_centroid | 中心对称 | 高精度匹配要求 |

### 参数说明

- `num_cols`: 列数，影响总宽度和匹配精度
- 更多列数 = 更好的梯度补偿
- 典型值: 4-8 列

## 对齐方式

| alignment | 说明 |
|-----------|------|
| center | 中心对齐（默认） |
| left | 左对齐 |
| right | 右对齐 |
| top | 顶部对齐 |
| bottom | 底部对齐 |

## 注意事项

- 放置前确保器件已创建
- 坐标单位为 μm
- rotation 支持 0, 90, 180, 270 度
- interdigitize 会重新排列两个组件的所有单元
- 放置后建议运行 DRC 检查（使用 verification-drc 技能）

## 相关资源

- `references/layout_guidelines.md` - 版图布局最佳实践
- `references/matching_techniques.md` - 匹配技术详解
