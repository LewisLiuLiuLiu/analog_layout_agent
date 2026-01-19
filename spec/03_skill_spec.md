# 第3章 技能模块实现

---

## 3.1 技能模块总体设计

### 3.1.1 Skill 在系统中的角色

技能模块（Skill）是 Agent 的"执行单元"，负责完成确定性的版图操作。其核心特征：

1. **确定性**：给定相同输入，始终产生相同输出
2. **原子性**：每个 Skill 完成一个独立、完整的操作
3. **可组合**：多个 Skill 可以按顺序组合完成复杂任务
4. **工具化**：通过 MCP 协议暴露给 Agent 调用

### 3.1.2 Skill 与其他模块的关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Agent Orchestrator                             │
│                                                                         │
│  ┌───────────────┐                          ┌───────────────────────┐  │
│  │ Planning      │──── Plan 步骤 ──────────►│ Action Module         │  │
│  │ Module        │                          │                       │  │
│  └───────────────┘                          │  ┌─────────────────┐  │  │
│                                             │  │ Skill Registry  │  │  │
│                                             │  │  ┌───────────┐  │  │  │
│                                             │  │  │ Skill A   │  │  │  │
│                                             │  │  ├───────────┤  │  │  │
│                                             │  │  │ Skill B   │  │  │  │
│                                             │  │  ├───────────┤  │  │  │
│                                             │  │  │ Skill C   │  │  │  │
│                                             │  │  └───────────┘  │  │  │
│                                             │  └────────┬────────┘  │  │
│                                             └───────────┼───────────┘  │
└─────────────────────────────────────────────────────────┼──────────────┘
                                                          │
                                                          ▼
                                              ┌───────────────────────┐
                                              │   KLayout MCP Server  │
                                              │   (版图操作执行)       │
                                              └───────────────────────┘
```

### 3.1.3 命名规范

技能模块采用分层命名，格式为 `{category}.{action}_{target}`：

| 类别前缀 | 含义 | 示例 |
|----------|------|------|
| `netlist.*` | 网表处理相关 | `netlist.parse` |
| `layout.*` | 版图生成相关 | `layout.create_common_centroid_pair` |
| `drc.*` | DRC 检查相关 | `drc.run_check` |
| `eval.*` | 评估相关 | `eval.compute_metrics` |
| `export.*` | 导出相关 | `export.gds` |

---

## 3.2 通用约定

### 3.2.1 通用输入/输出包装结构

所有 Skill 的返回值采用统一的包装结构：

```python
@dataclass
class SkillResult:
    """Skill 执行结果"""
    ok: bool                          # 是否成功
    error: Optional[SkillError]       # 错误信息（失败时）
    data: Optional[Dict[str, Any]]    # 返回数据（成功时）
    duration_ms: int = 0              # 执行耗时
    
@dataclass
class SkillError:
    """Skill 错误"""
    code: str                         # 错误码
    message: str                      # 错误消息
    details: Optional[Dict] = None    # 详细信息
```

**JSON 格式示例**：

```json
// 成功响应
{
    "ok": true,
    "error": null,
    "data": {
        "cell_name": "diff_pair_cc",
        "device_instances": ["M1_A", "M2_B", "M1_B", "M2_A"],
        "bounding_box": {"x": 0, "y": 0, "width": 10.5, "height": 8.2}
    },
    "duration_ms": 234
}

// 失败响应
{
    "ok": false,
    "error": {
        "code": "INVALID_PARAM",
        "message": "Device 'M3' not found in circuit",
        "details": {
            "param_name": "device_a",
            "provided_value": "M3",
            "available_devices": ["M1", "M2"]
        }
    },
    "data": null,
    "duration_ms": 12
}
```

### 3.2.2 错误处理约定

#### 错误码定义

| 错误码 | 含义 | 典型场景 |
|--------|------|----------|
| `INVALID_PARAM` | 参数非法 | 参数类型错误、必填参数缺失、值超出范围 |
| `DEVICE_NOT_FOUND` | 器件未找到 | 指定的器件 ID 在电路中不存在 |
| `DRC_VIOLATION` | DRC 违规 | 生成的版图存在 DRC 错误 |
| `LAYOUT_CONFLICT` | 布局冲突 | 器件放置位置重叠 |
| `KLAYOUT_ERROR` | KLayout 错误 | KLayout API 调用失败 |
| `SERVICE_UNAVAILABLE` | 服务不可用 | KLayout MCP 服务未启动 |
| `TIMEOUT` | 超时 | 操作超过时间限制 |
| `INTERNAL_ERROR` | 内部错误 | 未预期的程序异常 |
| `RULE_NOT_FOUND` | 规则未找到 | 指定的 DRC 规则不存在 |
| `UNSUPPORTED_OPERATION` | 不支持的操作 | 当前版本不支持的功能 |

#### 错误处理代码模式

```python
class SkillBase:
    """Skill 基类"""
    
    def execute(self, params: Dict[str, Any]) -> SkillResult:
        """执行 Skill"""
        start_time = time.time()
        
        try:
            # 1. 参数校验
            validated_params = self._validate_params(params)
            
            # 2. 执行核心逻辑
            result_data = self._execute_impl(validated_params)
            
            # 3. 返回成功结果
            return SkillResult(
                ok=True,
                error=None,
                data=result_data,
                duration_ms=self._elapsed_ms(start_time)
            )
            
        except ParamValidationError as e:
            return SkillResult(
                ok=False,
                error=SkillError(
                    code="INVALID_PARAM",
                    message=str(e),
                    details=e.details
                ),
                data=None,
                duration_ms=self._elapsed_ms(start_time)
            )
            
        except KLayoutError as e:
            return SkillResult(
                ok=False,
                error=SkillError(
                    code="KLAYOUT_ERROR",
                    message=str(e)
                ),
                data=None,
                duration_ms=self._elapsed_ms(start_time)
            )
            
        except Exception as e:
            self.logger.exception("Unexpected error in skill execution")
            return SkillResult(
                ok=False,
                error=SkillError(
                    code="INTERNAL_ERROR",
                    message=f"Unexpected error: {str(e)}"
                ),
                data=None,
                duration_ms=self._elapsed_ms(start_time)
            )
```

---

## 3.3 单个 Skill 的详细规格

### 3.3.1 `netlist.parse` 技能规格

#### 功能描述

解析 JSON 格式的网表文件，构建内部电路表示（`Circuit` 对象）。该 Skill 负责：
- 读取并验证网表 JSON 结构
- 提取器件信息（类型、参数、端口）
- 构建连接图（Net 拓扑）
- 识别常见电路模块（差分对、电流镜等）

#### 调用方式

- **MCP 工具名**: `netlist.parse`
- **Python 函数签名**:

```python
def parse_netlist(
    netlist_path: str,
    *,
    identify_modules: bool = True
) -> SkillResult:
    """
    解析网表 JSON
    
    参数:
        netlist_path: 网表 JSON 文件路径
        identify_modules: 是否自动识别电路模块
        
    返回:
        SkillResult: 包含 Circuit 对象的结果
    """
```

#### 输入参数定义

| 字段名 | 类型 | 单位 | 必填 | 含义 | 约束 |
|--------|------|------|------|------|------|
| `netlist_path` | string | - | 是 | 网表 JSON 文件路径 | 有效文件路径 |
| `identify_modules` | boolean | - | 否 | 是否自动识别模块 | 默认 true |

#### 输出数据结构

```json
{
    "circuit": {
        "name": "opamp_diff_stage",
        "devices": [
            {
                "id": "M1",
                "type": "nmos",
                "model": "nmos_1v8",
                "params": {"w": 2e-6, "l": 0.18e-6, "m": 2},
                "terminals": {"g": "vinp", "d": "outp", "s": "tail", "b": "vss"}
            }
        ],
        "nets": [
            {"name": "vinp", "connections": [{"device": "M1", "terminal": "g"}]}
        ],
        "modules": [
            {
                "id": "diff_pair_1",
                "type": "differential_pair",
                "devices": ["M1", "M2"],
                "properties": {"matched": true}
            }
        ]
    },
    "parse_info": {
        "device_count": 6,
        "net_count": 12,
        "module_count": 2
    }
}
```

#### 内部实现逻辑概述

1. **读取 JSON 文件**：使用标准 JSON 解析器读取文件
2. **Schema 校验**：验证 JSON 结构符合预定义 Schema
3. **器件解析**：遍历 `devices` 数组，创建 `Device` 对象
4. **网络构建**：根据 `terminals` 信息构建 `Net` 连接图
5. **模块识别**（可选）：
   - 识别差分对：查找共源极且栅极为输入的 NMOS/PMOS 对
   - 识别电流镜：查找共栅共源的 MOS 管组合
6. **返回结果**：封装为 `Circuit` 对象

#### 错误场景与错误码

| 场景 | 错误码 | 错误信息模板 |
|------|--------|--------------|
| 文件不存在 | `INVALID_PARAM` | `Netlist file not found: {path}` |
| JSON 解析失败 | `INVALID_PARAM` | `Invalid JSON format: {parse_error}` |
| Schema 校验失败 | `INVALID_PARAM` | `Schema validation failed: {field} - {reason}` |
| 器件类型未知 | `INVALID_PARAM` | `Unknown device type: {type}` |

#### 测试建议

| 测试用例 | 输入 | 预期结果 |
|----------|------|----------|
| 正常解析 | 标准差分运放网表 | 成功，器件数=6，识别出差分对 |
| 空网表 | `{"cells": [], "top": ""}` | 成功，器件数=0 |
| 无效路径 | 不存在的文件路径 | 失败，INVALID_PARAM |
| 格式错误 | 非 JSON 文件 | 失败，INVALID_PARAM |

---

### 3.3.2 `layout.create_nmos_pcell` / `layout.create_pmos_pcell` 技能规格

#### 功能描述

基于工艺参数和器件尺寸，生成单个 MOS 晶体管的参数化单元（PCell）。该 Skill 负责：
- 根据 W/L/M 参数计算几何尺寸
- 生成符合 DRC 的晶体管版图
- 创建端口标记和连接点

#### 调用方式

- **MCP 工具名**: `layout.create_nmos_pcell` / `layout.create_pmos_pcell`
- **Python 函数签名**:

```python
def create_mos_pcell(
    device_type: Literal["nmos", "pmos"],
    device_id: str,
    w: float,
    l: float,
    m: int = 1,
    nf: int = 1,
    *,
    layer_map: Optional[Dict[str, int]] = None,
    position: Tuple[float, float] = (0, 0),
    orientation: str = "R0"
) -> SkillResult:
    """
    生成 MOS 晶体管 PCell
    
    参数:
        device_type: 器件类型 (nmos/pmos)
        device_id: 器件实例名
        w: 沟道宽度 (米)
        l: 沟道长度 (米)
        m: 倍数
        nf: 指数（finger 数量）
        layer_map: 层映射表（可选，使用默认）
        position: 放置位置 (微米)
        orientation: 方向 (R0, R90, R180, R270, MX, MY)
    """
```

#### 输入参数定义

| 字段名 | 类型 | 单位 | 必填 | 含义 | 约束 |
|--------|------|------|------|------|------|
| `device_type` | string | - | 是 | 器件类型 | "nmos" 或 "pmos" |
| `device_id` | string | - | 是 | 器件实例名 | 非空字符串 |
| `w` | float | 米 | 是 | 沟道宽度 | > 0, 通常 0.1um~100um |
| `l` | float | 米 | 是 | 沟道长度 | > 0, 通常 >= min_L |
| `m` | int | - | 否 | 倍数 | >= 1, 默认 1 |
| `nf` | int | - | 否 | 指数 | >= 1, 默认 1 |
| `layer_map` | object | - | 否 | 层映射 | 见下方说明 |
| `position` | array | 微米 | 否 | 位置 [x, y] | 默认 [0, 0] |
| `orientation` | string | - | 否 | 方向 | 默认 "R0" |

**layer_map 结构**:
```json
{
    "active": 1,
    "poly": 2,
    "contact": 3,
    "metal1": 4,
    "nwell": 5,
    "pwell": 6
}
```

#### 输出数据结构

```json
{
    "cell_name": "nmos_M1",
    "device_id": "M1",
    "bounding_box": {
        "x": 0,
        "y": 0,
        "width": 2.5,
        "height": 1.8
    },
    "pins": {
        "G": {"layer": 2, "center": [1.25, 0.9]},
        "D": {"layer": 4, "center": [2.0, 0.9]},
        "S": {"layer": 4, "center": [0.5, 0.9]},
        "B": {"layer": 4, "center": [1.25, 0.2]}
    },
    "geometry_stats": {
        "poly_count": 1,
        "contact_count": 4,
        "metal1_count": 3
    }
}
```

#### 内部实现逻辑概述

1. **参数验证**：检查 W/L 是否在工艺允许范围内
2. **计算几何尺寸**：
   - 有源区宽度 = W * nf + (nf-1) * poly_spacing
   - 有源区长度 = L + 2 * gate_extension
3. **绘制有源区**：创建矩形图形
4. **绘制栅极**：根据 nf 绘制 poly 条
5. **添加 Contact**：在 S/D 区域添加接触孔
6. **绘制 Metal1**：连接 S/D/G/B 端口
7. **创建 Pin 标记**：标注各端口位置
8. **基本 DRC 检查**：验证最小间距和宽度

#### 错误场景与错误码

| 场景 | 错误码 | 错误信息模板 |
|------|--------|--------------|
| W/L 小于工艺最小值 | `INVALID_PARAM` | `Width {w} is below minimum {min_w}` |
| nf 非正整数 | `INVALID_PARAM` | `Number of fingers must be positive integer` |
| 无效方向 | `INVALID_PARAM` | `Invalid orientation: {orientation}` |
| 层映射缺失 | `INVALID_PARAM` | `Missing layer mapping for: {layer_name}` |
| KLayout 创建失败 | `KLAYOUT_ERROR` | `Failed to create cell: {error}` |

#### 测试建议

| 测试用例 | 输入 | 预期结果 |
|----------|------|----------|
| 标准 NMOS | w=2um, l=0.18um, nf=2 | 成功，生成 2 指 NMOS |
| 最小尺寸 | w=min_w, l=min_l | 成功，生成最小尺寸器件 |
| 多倍数 | m=4, nf=1 | 成功，生成 4 个并联器件 |
| 旋转放置 | orientation="R90" | 成功，器件旋转 90 度 |
| 非法参数 | w=-1 | 失败，INVALID_PARAM |

---

### 3.3.3 `layout.create_common_centroid_pair` 技能规格

#### 功能描述

为差分对或需要匹配的两个 MOS 器件生成共质心/交叉耦合布局。该 Skill 负责：
- 根据指定的排列方式（ABBA、ABAB 等）生成器件实例
- 计算并放置器件以实现对称性
- 生成对称轴和约束锚点信息

#### 调用方式

- **MCP 工具名**: `layout.create_common_centroid_pair`
- **Python 函数签名**:

```python
def create_common_centroid_pair(
    device_a: DeviceParams,
    device_b: DeviceParams,
    arrangement: str = "ABBA",
    *,
    symmetry_axis: str = "vertical",
    spacing: float = 0.5,
    interdigitate: bool = True
) -> SkillResult:
    """
    生成共质心配对布局
    
    参数:
        device_a: 器件 A 的参数
        device_b: 器件 B 的参数
        arrangement: 排列方式 (ABBA, ABAB, AABB)
        symmetry_axis: 对称轴方向 (vertical, horizontal)
        spacing: 器件间距 (微米)
        interdigitate: 是否交叉耦合
    """
```

#### 输入参数定义

| 字段名 | 类型 | 单位 | 必填 | 含义 | 约束 |
|--------|------|------|------|------|------|
| `device_a` | object | - | 是 | 器件 A 参数 | 见 DeviceParams |
| `device_b` | object | - | 是 | 器件 B 参数 | 见 DeviceParams |
| `arrangement` | string | - | 否 | 排列方式 | "ABBA"/"ABAB"/"AABB", 默认 "ABBA" |
| `symmetry_axis` | string | - | 否 | 对称轴 | "vertical"/"horizontal", 默认 "vertical" |
| `spacing` | float | 微米 | 否 | 间距 | > 0, 默认 0.5 |
| `interdigitate` | boolean | - | 否 | 交叉耦合 | 默认 true |

**DeviceParams 结构**:
```json
{
    "device_id": "M1",
    "type": "nmos",
    "w": 2e-6,
    "l": 0.18e-6,
    "m": 2,
    "nf": 2
}
```

#### 输出数据结构

```json
{
    "cell_name": "diff_pair_cc",
    "arrangement": "ABBA",
    "device_instances": [
        {"id": "M1_A1", "parent": "M1", "position": [0, 0], "orientation": "R0"},
        {"id": "M2_B1", "parent": "M2", "position": [3.0, 0], "orientation": "R0"},
        {"id": "M2_B2", "parent": "M2", "position": [6.0, 0], "orientation": "MX"},
        {"id": "M1_A2", "parent": "M1", "position": [9.0, 0], "orientation": "MX"}
    ],
    "symmetry_info": {
        "axis": "vertical",
        "center": [4.5, 0.9],
        "matched_pairs": [["M1_A1", "M1_A2"], ["M2_B1", "M2_B2"]]
    },
    "anchors": {
        "center": [4.5, 0.9],
        "left": [0, 0.9],
        "right": [9.0, 0.9]
    },
    "bounding_box": {
        "x": 0,
        "y": 0,
        "width": 12.0,
        "height": 1.8
    }
}
```

#### 内部实现逻辑概述

1. **参数验证**：
   - 检查两个器件是否兼容（相同类型、相似尺寸）
   - 验证排列方式有效性
2. **计算排列**：
   - 解析排列字符串（如 "ABBA"）
   - 计算每个器件实例的位置
3. **生成器件实例**：
   - 调用 `create_mos_pcell` 生成基础器件
   - 按计算位置放置并应用镜像变换
4. **对称性处理**：
   - 对于 ABBA 模式，内侧器件使用镜像方向
   - 计算对称轴位置
5. **生成锚点**：
   - 中心锚点用于整体定位
   - 边界锚点用于布线连接
6. **基本 DRC 检查**：
   - 验证器件间距满足 DRC

#### 错误场景与错误码

| 场景 | 错误码 | 错误信息模板 |
|------|--------|--------------|
| 器件类型不匹配 | `INVALID_PARAM` | `Devices must be same type for matching` |
| 无效排列方式 | `INVALID_PARAM` | `Invalid arrangement: {arrangement}` |
| 间距过小 | `DRC_VIOLATION` | `Spacing {spacing} violates min spacing {min}` |
| 器件参数缺失 | `INVALID_PARAM` | `Missing required param: {param}` |

#### 测试建议

| 测试用例 | 输入 | 预期结果 |
|----------|------|----------|
| 标准 ABBA | 两个相同 NMOS, arrangement="ABBA" | 成功，4 个实例 |
| ABAB 模式 | arrangement="ABAB" | 成功，交替排列 |
| 水平对称 | symmetry_axis="horizontal" | 成功，上下对称 |
| 不同尺寸 | device_a.w != device_b.w | 成功（或警告） |
| 类型不匹配 | NMOS + PMOS | 失败，INVALID_PARAM |

---

### 3.3.4 `layout.create_current_mirror` 技能规格

#### 功能描述

识别电流镜结构并生成匹配布局。支持简单电流镜和共源共栅电流镜。

#### 调用方式

- **MCP 工具名**: `layout.create_current_mirror`
- **Python 函数签名**:

```python
def create_current_mirror(
    devices: List[DeviceParams],
    mirror_type: str = "simple",
    *,
    reference_device: Optional[str] = None,
    matching_ratio: Optional[Dict[str, float]] = None,
    arrangement: str = "interdigitated"
) -> SkillResult:
    """
    生成电流镜匹配布局
    
    参数:
        devices: 参与电流镜的器件列表
        mirror_type: 电流镜类型 (simple, cascode)
        reference_device: 参考器件 ID
        matching_ratio: 匹配比例 {device_id: ratio}
        arrangement: 排列方式 (interdigitated, stacked)
    """
```

#### 输入参数定义

| 字段名 | 类型 | 单位 | 必填 | 含义 | 约束 |
|--------|------|------|------|------|------|
| `devices` | array | - | 是 | 器件列表 | 至少 2 个器件 |
| `mirror_type` | string | - | 否 | 类型 | "simple"/"cascode", 默认 "simple" |
| `reference_device` | string | - | 否 | 参考器件 | 默认第一个器件 |
| `matching_ratio` | object | - | 否 | 匹配比例 | 如 {"M1": 1, "M2": 2} |
| `arrangement` | string | - | 否 | 排列方式 | 默认 "interdigitated" |

#### 输出数据结构

```json
{
    "cell_name": "current_mirror",
    "mirror_type": "simple",
    "device_instances": [
        {"id": "M1_ref_1", "parent": "M1", "position": [0, 0]},
        {"id": "M2_out_1", "parent": "M2", "position": [3.0, 0]},
        {"id": "M2_out_2", "parent": "M2", "position": [6.0, 0]}
    ],
    "matching_info": {
        "reference": "M1",
        "mirrors": {"M2": 2.0},
        "unit_width": 1e-6
    },
    "common_nodes": {
        "gate": "bias",
        "source": "vss"
    }
}
```

#### 内部实现逻辑概述

1. **分析电流镜结构**：识别共栅共源连接
2. **计算单元分割**：根据匹配比例计算器件分割
3. **交替排列**：将不同器件的单元交替放置
4. **生成共用连接**：创建共栅和共源的布线锚点

#### 测试建议

| 测试用例 | 输入 | 预期结果 |
|----------|------|----------|
| 1:1 镜像 | 2 个相同 NMOS | 成功，交替排列 |
| 1:2 镜像 | M2.m = 2 * M1.m | 成功，M2 分 2 个单元 |
| Cascode | 4 个器件 | 成功，上下两层 |

---

### 3.3.5 `layout.route_signal_nets_basic` 技能规格

#### 功能描述

在局部区域内为少量网络进行规则约束下的基础布线。使用简单的曼哈顿布线算法。

#### 调用方式

- **MCP 工具名**: `layout.route_signal_nets_basic`
- **Python 函数签名**:

```python
def route_signal_nets_basic(
    nets: List[NetRouteSpec],
    drc_rules: DRCRuleSet,
    *,
    routing_layers: List[str] = ["metal1", "metal2"],
    via_layer: str = "via1",
    routing_area: Optional[BoundingBox] = None
) -> SkillResult:
    """
    基础信号线布线
    
    参数:
        nets: 需要布线的网络列表
        drc_rules: DRC 规则集
        routing_layers: 可用的布线层
        via_layer: 过孔层
        routing_area: 布线区域限制
    """
```

#### 输入参数定义

| 字段名 | 类型 | 单位 | 必填 | 含义 | 约束 |
|--------|------|------|------|------|------|
| `nets` | array | - | 是 | 网络列表 | 见 NetRouteSpec |
| `drc_rules` | object | - | 是 | DRC 规则 | DRCRuleSet 格式 |
| `routing_layers` | array | - | 否 | 布线层 | 默认 ["metal1", "metal2"] |
| `via_layer` | string | - | 否 | 过孔层 | 默认 "via1" |
| `routing_area` | object | - | 否 | 区域限制 | BoundingBox 格式 |

**NetRouteSpec 结构**:
```json
{
    "net_name": "outp",
    "endpoints": [
        {"x": 10.5, "y": 5.0, "layer": "metal1"},
        {"x": 25.0, "y": 5.0, "layer": "metal1"}
    ],
    "width": 0.2,
    "priority": 1
}
```

#### 输出数据结构

```json
{
    "routed_nets": [
        {
            "net_name": "outp",
            "success": true,
            "path": [
                {"layer": "metal1", "points": [[10.5, 5.0], [15.0, 5.0], [15.0, 8.0]]},
                {"layer": "via1", "position": [15.0, 8.0]},
                {"layer": "metal2", "points": [[15.0, 8.0], [25.0, 8.0], [25.0, 5.0]]}
            ],
            "wire_length": 25.5,
            "via_count": 1
        }
    ],
    "routing_stats": {
        "total_nets": 5,
        "success_count": 5,
        "failed_count": 0,
        "total_wire_length": 85.2
    }
}
```

#### 内部实现逻辑概述

1. **网络排序**：按优先级和长度排序
2. **路径规划**：使用曼哈顿距离 + 简单避障
3. **DRC 检查**：每步检查间距规则
4. **过孔插入**：层间切换时自动插入过孔
5. **冲突处理**：简单重试或绕行

#### 算法选择

- **推荐算法**：迷宫布线（Lee's Algorithm）简化版
- **备选算法**：A* 搜索（考虑线长优化）

---

### 3.3.6 `drc.run_check` 技能规格

#### 功能描述

调用 KLayout DRC 引擎对当前版图进行规则检查，返回违规列表和统计信息。

#### 调用方式

- **MCP 工具名**: `drc.run_check`
- **Python 函数签名**:

```python
def run_drc_check(
    layout_path: Optional[str] = None,
    cell_name: Optional[str] = None,
    drc_rules_path: str = None,
    *,
    rule_ids: Optional[List[str]] = None,
    max_violations: int = 100
) -> SkillResult:
    """
    执行 DRC 检查
    
    参数:
        layout_path: 版图文件路径（可选，使用当前版图）
        cell_name: 检查的 cell 名称
        drc_rules_path: DRC 规则文件路径
        rule_ids: 只检查指定规则（可选）
        max_violations: 最大违规数量限制
    """
```

#### 输入参数定义

| 字段名 | 类型 | 单位 | 必填 | 含义 | 约束 |
|--------|------|------|------|------|------|
| `layout_path` | string | - | 否 | 版图路径 | 有效 GDS/OAS 路径 |
| `cell_name` | string | - | 否 | Cell 名称 | 默认顶层 cell |
| `drc_rules_path` | string | - | 是 | 规则路径 | 有效 JSON/DRC 文件 |
| `rule_ids` | array | - | 否 | 规则 ID 列表 | 过滤特定规则 |
| `max_violations` | int | - | 否 | 最大违规数 | 默认 100 |

#### 输出数据结构

```json
{
    "summary": {
        "total_violations": 3,
        "rules_checked": 15,
        "rules_violated": 2,
        "check_time_ms": 1234
    },
    "violations": [
        {
            "rule_id": "M1_MIN_WIDTH",
            "rule_description": "Metal1 minimum width",
            "severity": "error",
            "count": 2,
            "locations": [
                {"x": 10.5, "y": 5.2, "layer": "metal1"},
                {"x": 15.0, "y": 8.0, "layer": "metal1"}
            ]
        }
    ],
    "passed": false
}
```

#### 内部实现逻辑概述

1. **加载版图**：打开 GDS/OAS 文件或使用当前会话
2. **解析规则**：将 JSON 规则转换为 KLayout DRC 脚本
3. **执行检查**：调用 KLayout DRC 引擎
4. **收集结果**：解析 DRC 报告，提取违规信息
5. **格式化输出**：转换为标准化 JSON 结构

---

### 3.3.7 `export.gds` 技能规格

#### 功能描述

将当前设计导出为 GDSII 文件，供后续仿真或流片使用。

#### 调用方式

- **MCP 工具名**: `export.gds`
- **Python 函数签名**:

```python
def export_gds(
    output_path: str,
    cell_name: Optional[str] = None,
    *,
    flatten: bool = False,
    layer_filter: Optional[List[int]] = None,
    unit: float = 1e-6,
    precision: float = 1e-9
) -> SkillResult:
    """
    导出 GDSII 文件
    
    参数:
        output_path: 输出文件路径
        cell_name: 导出的顶层 cell
        flatten: 是否展平层次
        layer_filter: 只导出指定层
        unit: 用户单位 (默认微米)
        precision: 数据库精度
    """
```

#### 输入参数定义

| 字段名 | 类型 | 单位 | 必填 | 含义 | 约束 |
|--------|------|------|------|------|------|
| `output_path` | string | - | 是 | 输出路径 | 有效写入路径 |
| `cell_name` | string | - | 否 | Cell 名称 | 默认顶层 |
| `flatten` | boolean | - | 否 | 展平 | 默认 false |
| `layer_filter` | array | - | 否 | 层过滤 | 层号列表 |
| `unit` | float | 米 | 否 | 单位 | 默认 1e-6 |
| `precision` | float | 米 | 否 | 精度 | 默认 1e-9 |

#### 输出数据结构

```json
{
    "output_path": "/path/to/output.gds",
    "cell_name": "opamp_top",
    "file_size_bytes": 125678,
    "export_stats": {
        "cell_count": 5,
        "layer_count": 8,
        "polygon_count": 1234,
        "path_count": 56
    }
}
```

---

### 3.3.8 `eval.compute_metrics` 技能规格

#### 功能描述

基于几何信息计算版图面积、基本匹配度和约束指标。

#### 调用方式

- **MCP 工具名**: `eval.compute_metrics`
- **Python 函数签名**:

```python
def compute_metrics(
    cell_name: Optional[str] = None,
    *,
    matched_pairs: Optional[List[Tuple[str, str]]] = None,
    area_targets: Optional[Dict[str, float]] = None
) -> SkillResult:
    """
    计算版图评估指标
    
    参数:
        cell_name: 评估的 cell 名称
        matched_pairs: 需要评估匹配度的器件对
        area_targets: 面积目标值
    """
```

#### 输出数据结构

```json
{
    "metrics": {
        "total_area": 125.5,
        "active_area": 45.2,
        "utilization": 0.36,
        "aspect_ratio": 1.25
    },
    "matching_metrics": [
        {
            "pair": ["M1", "M2"],
            "centroid_distance": 0.05,
            "area_ratio": 1.02,
            "matching_score": 0.95
        }
    ],
    "area_comparison": {
        "target": 100.0,
        "actual": 125.5,
        "ratio": 1.255
    }
}
```

---

## 3.4 Skill Registry 与版本管理

### 3.4.1 Skill 注册机制

```python
@dataclass
class SkillDefinition:
    """Skill 定义"""
    name: str                         # 工具名 (如 layout.create_common_centroid_pair)
    version: str                      # 版本号 (如 1.0.0)
    handler: Callable                 # 处理函数
    input_schema: Dict                # 输入参数 Schema
    output_schema: Dict               # 输出数据 Schema
    description: str                  # 功能描述
    tags: List[str]                   # 标签 (用于分类)

class SkillRegistry:
    """Skill 注册表"""
    
    def __init__(self):
        self._skills: Dict[str, Dict[str, SkillDefinition]] = {}
        # 结构: {skill_name: {version: SkillDefinition}}
        
    def register(self, skill: SkillDefinition) -> None:
        """
        注册 Skill
        
        如果同名 Skill 已存在，作为新版本添加
        """
        if skill.name not in self._skills:
            self._skills[skill.name] = {}
        self._skills[skill.name][skill.version] = skill
        
    def get_skill(
        self,
        name: str,
        version: Optional[str] = None
    ) -> SkillDefinition:
        """
        获取 Skill
        
        参数:
            name: Skill 名称
            version: 版本号（可选，默认最新版本）
        """
        if name not in self._skills:
            raise SkillNotFoundError(f"Skill not found: {name}")
            
        versions = self._skills[name]
        if version:
            if version not in versions:
                raise SkillVersionNotFoundError(
                    f"Version {version} not found for skill {name}"
                )
            return versions[version]
        
        # 返回最新版本
        latest = max(versions.keys(), key=self._parse_version)
        return versions[latest]
        
    def list_skills(self) -> List[SkillInfo]:
        """列出所有可用 Skill"""
        result = []
        for name, versions in self._skills.items():
            latest_version = max(versions.keys(), key=self._parse_version)
            skill = versions[latest_version]
            result.append(SkillInfo(
                name=name,
                version=latest_version,
                description=skill.description,
                tags=skill.tags
            ))
        return result
```

### 3.4.2 版本管理与向后兼容

```python
class SkillVersionManager:
    """Skill 版本管理"""
    
    # 版本兼容策略
    COMPATIBILITY_POLICY = {
        "major": "breaking",      # 主版本变更：不兼容
        "minor": "additive",      # 次版本变更：新增功能，向后兼容
        "patch": "compatible"     # 补丁版本：Bug 修复，完全兼容
    }
    
    def is_compatible(
        self,
        requested: str,
        available: str
    ) -> bool:
        """
        检查版本兼容性
        
        规则:
        - 相同主版本号：兼容
        - 不同主版本号：不兼容
        """
        req_parts = self._parse_version(requested)
        avail_parts = self._parse_version(available)
        
        return req_parts[0] == avail_parts[0]
        
    def migrate_params(
        self,
        params: Dict,
        from_version: str,
        to_version: str,
        skill_name: str
    ) -> Dict:
        """
        参数迁移
        
        处理不同版本间的参数格式变化
        """
        migrations = self._get_migrations(skill_name)
        current_params = params.copy()
        
        for migration in migrations:
            if self._should_apply(migration, from_version, to_version):
                current_params = migration.apply(current_params)
                
        return current_params
```

### 3.4.3 Skill 装饰器注册

```python
def skill(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    tags: List[str] = None
):
    """
    Skill 装饰器
    
    使用示例:
    
    @skill(
        name="layout.create_common_centroid_pair",
        version="1.0.0",
        description="生成共质心配对布局",
        tags=["layout", "matching"]
    )
    def create_common_centroid_pair(
        device_a: DeviceParams,
        device_b: DeviceParams,
        **kwargs
    ) -> SkillResult:
        ...
    """
    def decorator(func):
        # 提取输入 Schema（从类型注解）
        input_schema = _extract_input_schema(func)
        
        # 创建 Skill 定义
        skill_def = SkillDefinition(
            name=name,
            version=version,
            handler=func,
            input_schema=input_schema,
            output_schema={},  # 可从返回类型提取
            description=description or func.__doc__,
            tags=tags or []
        )
        
        # 注册到全局注册表
        global_registry.register(skill_def)
        
        return func
    return decorator
```

---

## 附录 D：Skill 实现假设

1. **实现语言**：Python 3.10+，使用 KLayout pya API
2. **坐标单位**：内部使用 KLayout DBU（数据库单位），接口使用微米
3. **层映射**：通过配置文件定义，支持不同 PDK 切换
4. **错误处理**：所有异常捕获并转换为 SkillResult
5. **日志记录**：每个 Skill 调用记录输入参数和执行结果
