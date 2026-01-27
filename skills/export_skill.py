"""
导出查询技能模块 (Export Query Skill)

使用 PydanticAI Skills 封装导出和查询工具，实现渐进式披露。
包含GDS导出、组件列表查询、组件详情查询等功能。

该技能用于将完成的版图导出为工业标准格式，
以及查询当前设计中的组件信息。
"""

from __future__ import annotations

import json
from typing import Optional, Dict, Any

from pydantic_ai import RunContext
from pydantic_ai_skills import Skill, SkillResource


# ============== 技能指令文档 ==============

EXPORT_SKILL_INSTRUCTIONS = """
# 导出查询技能 (Export Query Skill)

## 何时使用此技能

当你需要以下操作时使用此技能：
- **导出GDS文件**: 将版图导出为GDSII格式
- **查询组件列表**: 查看当前设计中的所有组件
- **获取组件详情**: 查看特定组件的参数和端口

## 功能说明

### GDS导出

GDSII是集成电路版图的工业标准格式，用于：
- 流片生产
- 版图查看器打开
- 与其他EDA工具交互
- 版图存档

### 组件查询

查询功能帮助你了解当前设计状态：
- 已创建哪些器件
- 每个器件的参数
- 可用的端口名称

## 脚本列表

- `export_gds`: 导出GDS文件
- `list_components`: 列出所有组件
- `get_component_info`: 获取组件详细信息

## 使用示例

### 导出GDS文件
```python
run_skill_script(
    skill_name="export-query",
    script_name="export_gds",
    args={"filename": "my_design.gds"}
)
```

### 列出所有组件
```python
run_skill_script(
    skill_name="export-query",
    script_name="list_components"
)
```

### 按类型过滤组件
```python
run_skill_script(
    skill_name="export-query",
    script_name="list_components",
    args={"device_type": "nmos"}
)
```

### 获取组件详情
```python
run_skill_script(
    skill_name="export-query",
    script_name="get_component_info",
    args={"component_name": "nmos_1"}
)
```

## 输出信息

### list_components 返回

```json
{
  "count": 3,
  "components": [
    {
      "name": "nmos_1",
      "device_type": "nmos",
      "size": [5.2, 3.1],
      "ports": ["drain_E", "gate_N", "source_W"],
      "port_count": 6
    },
    ...
  ]
}
```

### get_component_info 返回

```json
{
  "name": "nmos_1",
  "device_type": "nmos",
  "parameters": {
    "width": 3.0,
    "length": 0.15,
    "fingers": 2,
    "multiplier": 1
  },
  "position": [10.0, 5.0],
  "rotation": 0,
  "size": [5.2, 3.1],
  "ports": [
    {"name": "drain_E", "position": [15.2, 6.5], "direction": 0},
    {"name": "gate_N", "position": [12.6, 8.1], "direction": 90},
    ...
  ]
}
```

## 注意事项

- GDS导出前建议先完成DRC检查
- 文件名不指定时使用设计名称
- 组件列表可按device_type过滤
- 端口位置是绝对坐标
"""


EXPORT_REFERENCE = """
# 导出查询参考文档

## GDSII格式说明

### 什么是GDSII？

GDSII (Graphic Data System II) 是集成电路版图的标准二进制文件格式，
由Calma公司于1978年开发，至今仍是IC设计的事实标准。

### GDSII文件结构

```
GDSII文件
├── HEADER (版本信息)
├── LIBRARY (库定义)
│   ├── Library name
│   ├── Units
│   └── STRUCTURE (单元/cell)
│       ├── Cell name
│       ├── BOUNDARY (多边形)
│       ├── PATH (路径)
│       ├── SREF (单元引用)
│       ├── AREF (阵列引用)
│       └── TEXT (文本标签)
└── ENDLIB
```

### 层定义

GDSII使用层号(layer)和数据类型(datatype)标识不同的版图层：

| Sky130层 | Layer | Datatype | 用途 |
|---------|-------|----------|------|
| nwell | 64 | 20 | N阱 |
| diff | 65 | 20 | 有源区 |
| poly | 66 | 20 | 多晶硅 |
| li1 | 67 | 20 | 局部互连 |
| met1 | 68 | 20 | Metal1 |
| met2 | 69 | 20 | Metal2 |
| met3 | 70 | 20 | Metal3 |
| met4 | 71 | 20 | Metal4 |
| met5 | 72 | 20 | Metal5 |

## 组件类型说明

### 基础器件类型

| device_type | 说明 | 典型端口 |
|-------------|------|---------|
| `nmos` | NMOS晶体管 | drain, gate, source, body |
| `pmos` | PMOS晶体管 | drain, gate, source, body |
| `mimcap` | MIM电容 | top, bottom |
| `resistor` | 多晶硅电阻 | port_a, port_b |
| `via_stack` | Via堆叠 | top, bottom |

### 复合电路类型

| device_type | 说明 | 典型端口 |
|-------------|------|---------|
| `current_mirror` | 电流镜 | input, output, vdd/vss |
| `diff_pair` | 差分对 | inp, inn, outp, outn, tail |

## 端口方向定义

| 方向值 | 角度 | 含义 |
|--------|------|------|
| 0 | 0° | 东（右） |
| 90 | 90° | 北（上） |
| 180 | 180° | 西（左） |
| 270 | 270° | 南（下） |

## 导出检查清单

在导出GDS前，建议检查：

### 必须检查
- [ ] DRC检查通过
- [ ] 所有必要的连接已完成
- [ ] 端口标签正确

### 建议检查
- [ ] 设计名称有意义
- [ ] 金属密度在范围内
- [ ] 无悬空网络

### 可选检查
- [ ] 添加设计标题
- [ ] 版本号标注
- [ ] 设计者信息

## 使用GDS文件

### 查看GDS

推荐工具：
- KLayout (免费、跨平台)
- Klive (gLayout集成)
- Magic (开源)

### KLayout快捷键

| 快捷键 | 功能 |
|--------|------|
| F2 | 适应视图 |
| + / - | 放大/缩小 |
| Ctrl+F | 搜索 |
| L | 图层管理 |

## 常见问题

### Q: 为什么导出的GDS很大？

A: 可能原因：
1. 使用了大量Via（正常）
2. 金属填充过多
3. 重复的几何图形

### Q: 为什么看不到某些层？

A: 可能原因：
1. 层在查看器中被隐藏
2. 层定义与查看器不匹配
3. 该层确实没有图形

### Q: 如何合并多个GDS？

A: 使用KLayout或脚本工具，将多个单元导入同一库。
"""


# ============== 技能工厂函数 ==============

def create_export_skill() -> Skill:
    """创建导出查询技能
    
    封装GDS导出和组件查询功能。
    
    Returns:
        Skill: 配置好的导出查询技能实例
    """
    
    export_skill = Skill(
        name='export-query',
        description='版图导出与查询：GDS文件导出、组件列表、组件详情。用于保存设计和查看当前状态。',
        content=EXPORT_SKILL_INSTRUCTIONS,
        resources=[
            SkillResource(
                name='export-reference',
                content=EXPORT_REFERENCE
            )
        ]
    )
    
    # ========== 注册导出查询脚本 ==========
    
    @export_skill.script
    async def export_gds(
        ctx: RunContext[Any],
        filename: Optional[str] = None
    ) -> str:
        """导出GDS文件
        
        将当前版图导出为GDSII格式文件，用于流片或版图查看。
        
        Args:
            filename: 输出文件名（不含路径），默认使用设计名称。
                      如不含.gds后缀会自动添加。
            
        Returns:
            JSON格式的导出结果，包含：
            - success: 是否成功
            - path: 导出文件的完整路径
            - file_size: 文件大小(bytes)
        """
        params: Dict[str, Any] = {}
        if filename is not None:
            params["filename"] = filename
        
        result = ctx.deps.call_tool("export_gds", params)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @export_skill.script
    async def list_components(
        ctx: RunContext[Any],
        device_type: Optional[str] = None
    ) -> str:
        """列出当前设计中的所有组件
        
        查看当前版图中已创建的所有组件，可按类型过滤。
        
        Args:
            device_type: 可选，按器件类型过滤。
                有效值: nmos, pmos, mimcap, resistor, via_stack,
                       current_mirror, diff_pair 等
            
        Returns:
            JSON格式的组件列表，包含：
            - count: 组件数量
            - components: 组件数组，每个包含name, device_type, size, ports
        """
        params: Dict[str, Any] = {}
        if device_type is not None:
            params["device_type"] = device_type
        
        result = ctx.deps.call_tool("list_components", params)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @export_skill.script
    async def get_component_info(
        ctx: RunContext[Any],
        component_name: str
    ) -> str:
        """获取指定组件的详细信息
        
        查看特定组件的完整参数、位置和端口信息。
        
        Args:
            component_name: 组件名称（如 "nmos_1", "current_mirror_1"）
            
        Returns:
            JSON格式的组件详情，包含：
            - name: 组件名称
            - device_type: 器件类型
            - parameters: 创建参数（width, length等）
            - position: 当前位置 [x, y]
            - rotation: 旋转角度
            - size: 尺寸 [width, height]
            - ports: 端口列表，每个包含name, position, direction
        """
        result = ctx.deps.call_tool("get_component_info", {"name": component_name})
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    return export_skill
