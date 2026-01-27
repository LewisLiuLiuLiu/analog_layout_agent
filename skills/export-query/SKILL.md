---
name: export-query
description: 导出与查询：导出GDS文件、查询组件列表、获取组件详细信息等，用于版图输出和状态查询。
---

# 导出查询技能 (Export Query Skill)

## 何时使用此技能

当你需要导出版图或查询设计状态时使用此技能：

- **export_gds**: 导出GDSII格式文件
- **list_components**: 列出所有组件
- **get_component_info**: 获取组件详细信息
- **get_context_status**: 获取当前设计状态

## 使用流程

1. 设计完成并通过DRC检查后
2. 使用 list_components 确认组件完整性
3. 使用 export_gds 导出版图文件
4. 可随时使用查询功能了解设计状态

## 可用脚本

| 脚本名 | 说明 | 必需参数 | 可选参数 |
|--------|------|----------|----------|
| `export_gds` | 导出GDS文件 | `--output` | `--top-cell` |
| `list_components` | 列出组件 | 无 | `--type` |
| `get_component_info` | 获取组件信息 | `--name` | - |
| `get_context_status` | 获取设计状态 | 无 | - |

## GDS导出

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `output` | 输出文件路径 | 必需 |
| `top_cell` | 顶层单元名 | 设计名称 |

### 导出示例

```
run_skill_script(
    skill_name="export-query",
    script_name="export_gds",
    args=["--output", "./output/my_design.gds"]
)
```

返回：
```json
{
  "success": true,
  "file_path": "./output/my_design.gds",
  "file_size": "125.4 KB",
  "top_cell": "top_level",
  "layers_used": ["poly", "met1", "met2", "via1"]
}
```

## 组件查询

### 列出所有组件

```
run_skill_script(
    skill_name="export-query",
    script_name="list_components",
    args=[]
)
```

返回：
```json
{
  "total": 5,
  "components": [
    {"name": "M1", "type": "nmos", "position": [0, 0]},
    {"name": "M2", "type": "nmos", "position": [10, 0]},
    {"name": "CM1", "type": "current_mirror", "position": [0, 20]},
    {"name": "R1", "type": "resistor", "position": [20, 0]},
    {"name": "C1", "type": "mimcap", "position": [30, 0]}
  ]
}
```

### 按类型过滤

```
run_skill_script(
    skill_name="export-query",
    script_name="list_components",
    args=["--type", "nmos"]
)
```

### 获取组件详情

```
run_skill_script(
    skill_name="export-query",
    script_name="get_component_info",
    args=["--name", "M1"]
)
```

返回：
```json
{
  "name": "M1",
  "type": "nmos",
  "parameters": {
    "width": 3.0,
    "length": 0.15,
    "fingers": 4,
    "multiplier": 1
  },
  "position": [0, 0],
  "rotation": 0,
  "ports": [
    {"name": "drain_N", "position": [1.5, 2.0], "layer": "met1"},
    {"name": "gate_E", "position": [3.0, 1.0], "layer": "poly"},
    {"name": "source_S", "position": [1.5, 0], "layer": "met1"}
  ],
  "bounding_box": [[0, 0], [3.0, 2.0]]
}
```

## 设计状态查询

### 获取上下文状态

```
run_skill_script(
    skill_name="export-query",
    script_name="get_context_status",
    args=[]
)
```

返回：
```json
{
  "design_name": "opamp_v1",
  "pdk": "sky130",
  "components_count": 15,
  "routes_count": 23,
  "last_modified": "2024-01-15T10:30:00",
  "drc_status": "passed",
  "bounding_box": [[0, 0], [100, 80]]
}
```

## 组件类型列表

| 类型 | 说明 |
|------|------|
| `nmos` | NMOS晶体管 |
| `pmos` | PMOS晶体管 |
| `current_mirror` | 电流镜 |
| `diff_pair` | 差分对 |
| `resistor` | 电阻 |
| `mimcap` | MIM电容 |
| `via_stack` | Via堆叠 |

## 注意事项

- 导出前建议先运行DRC检查
- GDS文件路径需要有写入权限
- 组件名称区分大小写
- 查询操作不会修改设计状态

## 相关资源

- `references/gds_format.md` - GDSII格式说明
- `references/layer_mapping.md` - 层映射表
