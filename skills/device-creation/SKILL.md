---
name: device-creation
description: 创建基础模拟器件：NMOS、PMOS、MIM电容、多晶硅电阻、Via过孔堆叠。支持指定尺寸、指数、dummy结构等参数。
---

# 器件创建技能 (Device Creation Skill)

## 何时使用此技能

当你需要创建以下基础模拟电路器件时使用此技能：

- **NMOS/PMOS 晶体管**: 模拟电路的核心器件，用于放大、开关、电流源等
- **MIM 电容**: 金属-绝缘体-金属电容，用于滤波、补偿、采样保持等
- **多晶硅电阻**: 精密电阻，用于偏置网络、反馈电路
- **Via 过孔堆叠**: 连接不同金属层的垂直互连结构

## 使用流程

1. 根据电路需求确定器件类型和参数
2. 调用相应的创建脚本（如 `create_nmos`、`create_pmos`）
3. 记录返回的器件名称，用于后续布线和放置操作
4. 如需参考参数范围，读取 `references/sky130_parameters.md`

## 可用脚本

| 脚本名 | 说明 | 必需参数 | 可选参数 |
|--------|------|----------|----------|
| `create_nmos` | 创建NMOS晶体管 | `--width` | `--length`, `--fingers`, `--multiplier`, `--with-dummy`, `--with-tie`, `--name` |
| `create_pmos` | 创建PMOS晶体管 | `--width` | `--length`, `--fingers`, `--multiplier`, `--with-dummy`, `--with-tie`, `--name` |
| `create_mimcap` | 创建MIM电容 | `--width`, `--length` | `--name` |
| `create_resistor` | 创建多晶硅电阻 | `--width`, `--length` | `--num-series`, `--name` |
| `create_via_stack` | 创建层间Via堆叠 | `--from-layer`, `--to-layer` | `--size`, `--name` |

## 关键参数说明

### 晶体管参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `width` | 沟道宽度(μm)，影响电流驱动能力 | 1-10μm |
| `length` | 沟道长度(μm)，影响输出阻抗，默认使用PDK最小长度 | 0.15-1μm |
| `fingers` | 指数，将宽MOS分成多个并联单元 | 1-8 |
| `multiplier` | 并联倍数，用于电流镜匹配 | 1-4 |
| `with_dummy` | 添加dummy结构改善匹配性 | True (推荐) |
| `with_tie` | 添加衬底连接 | True |

### 电容参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `width` | 电容宽度(μm) | 5-20μm |
| `length` | 电容长度(μm) | 5-20μm |

电容值估算（Sky130）: `C = 2.0 fF/μm² × width × length`

### 电阻参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `width` | 电阻宽度(μm) | 0.5-2μm |
| `length` | 电阻长度(μm) | 5-50μm |
| `num_series` | 串联段数 | 1-10 |

电阻值估算（Sky130高阻poly）: `R = 1000 Ω/□ × length/width × num_series`

### Via堆叠参数

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `from_layer` | 起始层 | poly, met1, met2, met3, met4, met5 |
| `to_layer` | 目标层 | met1, met2, met3, met4, met5 |

## 使用示例

### 创建NMOS晶体管

创建一个3μm宽、4指、带dummy结构的NMOS：

```
run_skill_script(
    skill_name="device-creation",
    script_name="create_nmos",
    args=["--width", "3.0", "--fingers", "4", "--with-dummy"]
)
```

### 创建电流镜用NMOS对

创建两个匹配的NMOS用于电流镜：

```
# 参考管（1倍）
run_skill_script(
    skill_name="device-creation",
    script_name="create_nmos",
    args=["--width", "2.0", "--multiplier", "1", "--with-dummy", "--name", "M_ref"]
)

# 输出管（4倍）
run_skill_script(
    skill_name="device-creation",
    script_name="create_nmos",
    args=["--width", "2.0", "--multiplier", "4", "--with-dummy", "--name", "M_out"]
)
```

### 创建MIM电容

创建约200fF的补偿电容：

```
run_skill_script(
    skill_name="device-creation",
    script_name="create_mimcap",
    args=["--width", "10.0", "--length", "10.0", "--name", "Cc"]
)
```

## 注意事项

- 宽度过小可能导致 DRC 违规，请参考 `references/sky130_parameters.md` 中的最小尺寸要求
- 建议对匹配敏感电路（如差分对、电流镜）使用 `--with-dummy` 选项
- 器件名称不指定时会自动生成唯一 ID
- 创建后的器件默认位于原点，需使用 `placement-layout` 技能进行放置
- Via堆叠会自动创建中间层的所有via

## 相关资源

- `references/sky130_parameters.md` - Sky130 PDK 器件参数范围和设计规则
- `references/matching_guidelines.md` - 匹配敏感电路的设计建议
