---
name: circuit-building
description: 复合电路构建：一键创建电流镜、差分对等常用模拟电路模块，自动处理器件创建、放置和内部连接。
---

# 电路构建技能 (Circuit Building Skill)

## 何时使用此技能

当你需要创建完整的模拟电路模块时使用此技能：

- **create_current_mirror**: 创建电流镜电路
- **create_diff_pair**: 创建差分对电路

这些是高级构建器，会自动完成：
1. 创建所需的晶体管
2. 使用互指式布局优化匹配
3. 连接内部节点
4. 生成标准化端口

## 使用流程

1. 确定电路需求（电流比、宽度等）
2. 调用相应的构建脚本
3. 使用返回的端口进行外部连接
4. 如需自定义，可使用 device-creation 和 placement-layout 技能手动构建

## 可用脚本

| 脚本路径 | 说明 | 必需参数 | 可选参数 |
|--------|------|----------|----------|
| `scripts/create_current_mirror.py` | 创建电流镜 | `--width`, `--ratio` | `--length`, `--fingers`, `--type`, `--name` |
| `scripts/create_diff_pair.py` | 创建差分对 | `--width` | `--length`, `--fingers`, `--tail-width`, `--name` |

## 电流镜参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `width` | 单管宽度(μm) | 2-10μm |
| `ratio` | 电流比(输出:输入) | 1:1, 2:1, 4:1 |
| `length` | 沟道长度(μm) | 0.5-2μm |
| `fingers` | 指数 | 2-4 |
| `type` | NMOS或PMOS | nmos(默认), pmos |

### 电流镜端口

| 端口名 | 说明 |
|--------|------|
| `input` | 输入端（二极管连接） |
| `output` | 输出端（镜像电流） |
| `vss` | 地端（NMOS镜）|
| `vdd` | 电源端（PMOS镜）|

## 差分对参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `width` | 输入管宽度(μm) | 5-20μm |
| `length` | 沟道长度(μm) | 0.5-2μm |
| `fingers` | 指数 | 4-8 |
| `tail_width` | 尾电流管宽度(μm) | 2×width |

### 差分对端口

| 端口名 | 说明 |
|--------|------|
| `inp` | 正输入端 |
| `inn` | 负输入端 |
| `outp` | 正输出端 |
| `outn` | 负输出端 |
| `tail` | 尾电流端 |
| `vss` | 地端 |

## 使用示例

### 创建1:4电流镜

```
run_skill_script(
    skill_name="circuit-building",
    script_name="scripts/create_current_mirror.py",
    args=["--width", "2.0", "--ratio", "4", "--fingers", "2", "--name", "CM1"]
)
```

### 创建PMOS电流镜

```
run_skill_script(
    skill_name="circuit-building",
    script_name="scripts/create_current_mirror.py",
    args=["--width", "4.0", "--ratio", "2", "--type", "pmos", "--name", "CM_load"]
)
```

### 创建差分输入对

```
run_skill_script(
    skill_name="circuit-building",
    script_name="scripts/create_diff_pair.py",
    args=["--width", "10.0", "--fingers", "4", "--tail-width", "20.0", "--name", "DP1"]
)
```

## 电路拓扑说明

### 电流镜内部结构

```
        VDD/VSS
           |
    +------+------+
    |             |
  [M_ref]      [M_out]
    |             |
  input        output
```

- M_ref: 参考管，二极管连接
- M_out: 输出管，multiplier = ratio

### 差分对内部结构

```
      outp    outn
        |      |
     [M_inp] [M_inn]
        |      |
        +--+---+
           |
        [M_tail]
           |
          vss
```

- M_inp/M_inn: 差分输入管，互指式放置
- M_tail: 尾电流管

## 设计建议

1. **电流镜**
   - ratio 建议使用整数比
   - 高精度应用增加 fingers
   - PMOS镜用于有源负载

2. **差分对**
   - 大 width 提高 gm
   - 适当的 tail_width 保证线性区工作
   - 互指式布局自动应用

## 注意事项

- 复合电路创建后位于原点，可用 placement-layout 移动
- 内部连线已完成，只需连接外部端口
- 如需更复杂的电路，建议分步使用基础技能构建

## 相关资源

- `references/current_mirror_design.md` - 电流镜设计指南
- `references/diff_pair_design.md` - 差分对设计指南
