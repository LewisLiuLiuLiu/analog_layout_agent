# 第5章 数据模型与接口

---

## 5.1 数据模型总览

### 5.1.1 核心数据结构

系统中的核心数据结构分为三类：

1. **输入数据结构**：由感知模块解析器生成
2. **中间数据结构**：在规划和执行过程中使用
3. **输出数据结构**：Skill 执行结果和最终报告

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         数据结构关系图                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐     ┌─────────────────────┐                   │
│  │   输入 JSON 文件     │     │    内部数据结构      │                   │
│  │  ─────────────────  │     │  ─────────────────  │                   │
│  │  • netlist.json     │────►│  • Circuit          │                   │
│  │  • drc_rules.json   │────►│  • DRCRuleSet       │                   │
│  │  • objectives.json  │────►│  • DesignObjectives │                   │
│  └─────────────────────┘     └──────────┬──────────┘                   │
│                                         │                               │
│                                         ▼                               │
│                              ┌─────────────────────┐                   │
│                              │    TaskContext      │                   │
│                              └──────────┬──────────┘                   │
│                                         │                               │
│                    ┌────────────────────┼────────────────────┐         │
│                    ▼                    ▼                    ▼         │
│         ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│         │      Plan       │  │ToolCallRequest │  │ ToolCallResult  │ │
│         │   PlanStep[]    │  │                 │  │   SkillResult   │ │
│         └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.1.2 数据结构与模块对应关系

| 数据结构 | 生成模块 | 消费模块 |
|----------|----------|----------|
| `Circuit` | NetlistParser | Planning, Action |
| `DRCRuleSet` | DRCParser | Action (DRC Skill) |
| `DesignObjectives` | ObjectiveParser | Planning, Evaluation |
| `TaskContext` | Perception | Planning, Memory |
| `Plan` | Planning | Action |
| `ToolCallRequest` | Action | MCP Client |
| `ToolCallResult` | MCP Client | Action, Memory |
| `EvaluationReport` | Evaluation | Memory, Orchestrator |

---

## 5.2 网表 JSON 解析器规格

### 5.2.1 网表 JSON Schema

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Netlist",
    "description": "电路网表定义",
    "type": "object",
    "required": ["cells", "top"],
    "properties": {
        "schema_version": {
            "type": "string",
            "description": "Schema 版本号",
            "default": "1.0.0"
        },
        "cells": {
            "type": "array",
            "description": "Cell 定义列表",
            "items": {"$ref": "#/definitions/Cell"}
        },
        "top": {
            "type": "string",
            "description": "顶层 Cell 名称"
        }
    },
    "definitions": {
        "Cell": {
            "type": "object",
            "required": ["name", "devices"],
            "properties": {
                "name": {"type": "string"},
                "devices": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/Device"}
                },
                "nets": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "subcells": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/SubcellInstance"}
                }
            }
        },
        "Device": {
            "type": "object",
            "required": ["id", "type", "terminals"],
            "properties": {
                "id": {
                    "type": "string",
                    "description": "器件实例名"
                },
                "type": {
                    "type": "string",
                    "enum": ["nmos", "pmos", "resistor", "capacitor"],
                    "description": "器件类型"
                },
                "model": {
                    "type": "string",
                    "description": "器件模型名"
                },
                "w": {
                    "type": "number",
                    "description": "宽度 (米)",
                    "minimum": 0
                },
                "l": {
                    "type": "number",
                    "description": "长度 (米)",
                    "minimum": 0
                },
                "m": {
                    "type": "integer",
                    "description": "倍数",
                    "minimum": 1,
                    "default": 1
                },
                "nf": {
                    "type": "integer",
                    "description": "指数",
                    "minimum": 1,
                    "default": 1
                },
                "terminals": {
                    "type": "object",
                    "description": "端口连接",
                    "additionalProperties": {"type": "string"}
                }
            }
        },
        "SubcellInstance": {
            "type": "object",
            "required": ["name", "cell"],
            "properties": {
                "name": {"type": "string"},
                "cell": {"type": "string"},
                "connections": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
            }
        }
    }
}
```

### 5.2.2 字段详细说明

#### Device 字段

| 字段 | 类型 | 必填 | 单位 | 约束 | 说明 |
|------|------|------|------|------|------|
| `id` | string | 是 | - | 非空，在 Cell 内唯一 | 器件实例名称 |
| `type` | string | 是 | - | 枚举值 | 器件类型 |
| `model` | string | 否 | - | - | 工艺模型名称 |
| `w` | number | 是* | 米 | > 0 | 沟道宽度（MOS 管必填） |
| `l` | number | 是* | 米 | > 0 | 沟道长度（MOS 管必填） |
| `m` | integer | 否 | - | >= 1 | 并联倍数，默认 1 |
| `nf` | integer | 否 | - | >= 1 | 指数，默认 1 |
| `terminals` | object | 是 | - | 见下表 | 端口-网络映射 |

#### MOS 管 terminals 规范

| 端口名 | 含义 | 必填 |
|--------|------|------|
| `g` | 栅极 Gate | 是 |
| `d` | 漏极 Drain | 是 |
| `s` | 源极 Source | 是 |
| `b` | 衬底 Bulk | 是 |

### 5.2.3 内部数据结构定义

```python
@dataclass
class Device:
    """器件"""
    id: str                           # 实例名
    type: DeviceType                  # 类型枚举
    model: Optional[str]              # 模型名
    params: DeviceParams              # 参数
    terminals: Dict[str, str]         # 端口 -> 网络名

@dataclass
class DeviceParams:
    """器件参数"""
    w: Optional[float] = None         # 宽度 (米)
    l: Optional[float] = None         # 长度 (米)
    m: int = 1                        # 倍数
    nf: int = 1                       # 指数
    # 扩展参数
    extra: Dict[str, Any] = field(default_factory=dict)

class DeviceType(Enum):
    """器件类型"""
    NMOS = "nmos"
    PMOS = "pmos"
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"

@dataclass
class Net:
    """网络"""
    name: str                         # 网络名
    connections: List[TerminalRef]    # 连接的端口列表
    net_type: NetType = NetType.SIGNAL  # 网络类型

@dataclass
class TerminalRef:
    """端口引用"""
    device_id: str                    # 器件 ID
    terminal_name: str                # 端口名

class NetType(Enum):
    """网络类型"""
    SIGNAL = "signal"
    POWER = "power"
    GROUND = "ground"

@dataclass
class CircuitModule:
    """电路模块"""
    id: str                           # 模块 ID
    type: ModuleType                  # 模块类型
    devices: List[str]                # 包含的器件 ID
    properties: Dict[str, Any]        # 模块属性

class ModuleType(Enum):
    """模块类型"""
    DIFFERENTIAL_PAIR = "differential_pair"
    CURRENT_MIRROR = "current_mirror"
    CASCODE = "cascode"
    TAIL_CURRENT_SOURCE = "tail_current_source"

@dataclass
class Circuit:
    """电路"""
    name: str                         # 电路名
    devices: List[Device]             # 器件列表
    nets: List[Net]                   # 网络列表
    modules: List[CircuitModule]      # 识别的模块
    
    def get_device(self, device_id: str) -> Optional[Device]:
        """根据 ID 获取器件"""
        for d in self.devices:
            if d.id == device_id:
                return d
        return None
        
    def get_net(self, net_name: str) -> Optional[Net]:
        """根据名称获取网络"""
        for n in self.nets:
            if n.name == net_name:
                return n
        return None
        
    def get_module_types(self) -> List[str]:
        """获取所有模块类型"""
        return [m.type.value for m in self.modules]
        
    def summarize(self) -> str:
        """生成电路摘要（用于 Prompt）"""
        lines = [
            f"电路名: {self.name}",
            f"器件数: {len(self.devices)}",
            f"网络数: {len(self.nets)}",
            "器件列表:"
        ]
        for d in self.devices:
            lines.append(f"  - {d.id}: {d.type.value}, W={d.params.w}, L={d.params.l}")
        if self.modules:
            lines.append("识别的模块:")
            for m in self.modules:
                lines.append(f"  - {m.id}: {m.type.value}")
        return "\n".join(lines)
```

### 5.2.4 解析流程

```python
class NetlistParser:
    """网表解析器"""
    
    def __init__(self, schema_path: Optional[str] = None):
        self.schema = self._load_schema(schema_path)
        self.module_recognizer = ModuleRecognizer()
        
    def parse(
        self,
        netlist_path: str,
        *,
        identify_modules: bool = True
    ) -> Circuit:
        """
        解析网表 JSON
        
        流程:
        1. 读取 JSON 文件
        2. Schema 校验
        3. 构建 Device 列表
        4. 构建 Net 连接图
        5. (可选) 识别电路模块
        """
        # 1. 读取 JSON
        with open(netlist_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 2. Schema 校验
        self._validate_schema(data)
        
        # 3. 获取顶层 Cell
        top_cell_name = data['top']
        top_cell = self._find_cell(data['cells'], top_cell_name)
        
        # 4. 构建 Device 列表
        devices = []
        for device_data in top_cell.get('devices', []):
            device = self._parse_device(device_data)
            devices.append(device)
            
        # 5. 构建 Net 连接图
        nets = self._build_nets(devices, top_cell.get('nets', []))
        
        # 6. 识别电路模块
        modules = []
        if identify_modules:
            modules = self.module_recognizer.recognize(devices, nets)
            
        return Circuit(
            name=top_cell_name,
            devices=devices,
            nets=nets,
            modules=modules
        )
        
    def _validate_schema(self, data: Dict) -> None:
        """Schema 校验"""
        try:
            jsonschema.validate(data, self.schema)
        except jsonschema.ValidationError as e:
            raise NetlistParseError(
                f"Schema validation failed: {e.path} - {e.message}"
            )
            
    def _parse_device(self, data: Dict) -> Device:
        """解析单个器件"""
        device_type = DeviceType(data['type'])
        
        params = DeviceParams(
            w=data.get('w'),
            l=data.get('l'),
            m=data.get('m', 1),
            nf=data.get('nf', 1)
        )
        
        return Device(
            id=data['id'],
            type=device_type,
            model=data.get('model'),
            params=params,
            terminals=data['terminals']
        )
        
    def _build_nets(
        self,
        devices: List[Device],
        net_names: List[str]
    ) -> List[Net]:
        """构建网络连接图"""
        net_map: Dict[str, List[TerminalRef]] = {}
        
        for device in devices:
            for terminal, net_name in device.terminals.items():
                if net_name not in net_map:
                    net_map[net_name] = []
                net_map[net_name].append(TerminalRef(
                    device_id=device.id,
                    terminal_name=terminal
                ))
                
        nets = []
        for net_name, connections in net_map.items():
            net_type = self._infer_net_type(net_name)
            nets.append(Net(
                name=net_name,
                connections=connections,
                net_type=net_type
            ))
            
        return nets
        
    def _infer_net_type(self, net_name: str) -> NetType:
        """推断网络类型"""
        name_lower = net_name.lower()
        if name_lower in ('vdd', 'vcc', 'avdd', 'dvdd'):
            return NetType.POWER
        elif name_lower in ('vss', 'gnd', 'avss', 'dvss'):
            return NetType.GROUND
        return NetType.SIGNAL
```

### 5.2.5 错误处理

| 错误场景 | 处理方式 | 错误消息 |
|----------|----------|----------|
| 文件不存在 | 抛出 `FileNotFoundError` | `Netlist file not found: {path}` |
| JSON 解析失败 | 抛出 `NetlistParseError` | `Invalid JSON: {parse_error}` |
| Schema 校验失败 | 抛出 `NetlistParseError` | `Schema validation failed: {field} - {reason}` |
| 器件类型未知 | 抛出 `NetlistParseError` | `Unknown device type: {type}` |
| 顶层 Cell 未找到 | 抛出 `NetlistParseError` | `Top cell not found: {name}` |

---

## 5.3 DRC 规则 JSON 解析器规格

### 5.3.1 DRC JSON Schema

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "DRCRules",
    "description": "设计规则检查规则集",
    "type": "object",
    "required": ["tech", "layers", "rules"],
    "properties": {
        "schema_version": {
            "type": "string",
            "default": "1.0.0"
        },
        "tech": {
            "type": "string",
            "description": "工艺名称"
        },
        "layers": {
            "type": "object",
            "description": "层名到层号的映射",
            "additionalProperties": {"type": "integer"}
        },
        "rules": {
            "type": "array",
            "items": {"$ref": "#/definitions/Rule"}
        }
    },
    "definitions": {
        "Rule": {
            "type": "object",
            "required": ["id", "type", "layer"],
            "properties": {
                "id": {
                    "type": "string",
                    "description": "规则 ID"
                },
                "type": {
                    "type": "string",
                    "enum": ["width", "spacing", "enclosure", "extension", "overlap", "density"],
                    "description": "规则类型"
                },
                "layer": {
                    "type": "string",
                    "description": "主层"
                },
                "layer2": {
                    "type": "string",
                    "description": "第二层（用于 spacing/enclosure 等）"
                },
                "min": {
                    "type": "number",
                    "description": "最小值 (米)"
                },
                "max": {
                    "type": "number",
                    "description": "最大值 (米)"
                },
                "description": {
                    "type": "string",
                    "description": "规则描述"
                }
            }
        }
    }
}
```

### 5.3.2 字段详细说明

| 字段 | 类型 | 必填 | 单位 | 说明 |
|------|------|------|------|------|
| `id` | string | 是 | - | 规则唯一标识 |
| `type` | string | 是 | - | 规则类型（见下表） |
| `layer` | string | 是 | - | 主层名称 |
| `layer2` | string | 否 | - | 第二层（双层规则时使用） |
| `min` | number | 否 | 米 | 最小值约束 |
| `max` | number | 否 | 米 | 最大值约束 |
| `description` | string | 否 | - | 规则描述 |

#### 规则类型说明

| 类型 | 含义 | 典型用法 |
|------|------|----------|
| `width` | 宽度规则 | 最小线宽 |
| `spacing` | 间距规则 | 同层最小间距 |
| `enclosure` | 包围规则 | Via 必须被 Metal 包围 |
| `extension` | 延伸规则 | Poly 必须延伸出有源区 |
| `overlap` | 重叠规则 | 两层必须重叠 |
| `density` | 密度规则 | 金属覆盖率 |

### 5.3.3 内部数据结构定义

```python
@dataclass
class DRCRule:
    """单条 DRC 规则"""
    id: str                           # 规则 ID
    type: RuleType                    # 规则类型
    layer: str                        # 主层
    layer2: Optional[str] = None      # 第二层
    min_value: Optional[float] = None # 最小值 (米)
    max_value: Optional[float] = None # 最大值 (米)
    description: Optional[str] = None # 描述
    
    def to_klayout_check(self) -> str:
        """转换为 KLayout DRC 检查语句"""
        if self.type == RuleType.WIDTH:
            return f"{self.layer}.width({self.min_value * 1e6}).output(\"{self.id}\")"
        elif self.type == RuleType.SPACING:
            return f"{self.layer}.space({self.min_value * 1e6}).output(\"{self.id}\")"
        # ... 其他类型
        raise NotImplementedError(f"Rule type {self.type} not implemented")

class RuleType(Enum):
    """规则类型"""
    WIDTH = "width"
    SPACING = "spacing"
    ENCLOSURE = "enclosure"
    EXTENSION = "extension"
    OVERLAP = "overlap"
    DENSITY = "density"

@dataclass
class DRCRuleSet:
    """DRC 规则集"""
    tech: str                         # 工艺名称
    layers: Dict[str, int]            # 层名 -> 层号映射
    rules: List[DRCRule]              # 规则列表
    schema_version: str = "1.0.0"     # Schema 版本
    
    def get_rule(self, rule_id: str) -> Optional[DRCRule]:
        """根据 ID 获取规则"""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None
        
    def get_rules_for_layer(self, layer: str) -> List[DRCRule]:
        """获取指定层的所有规则"""
        return [r for r in self.rules if r.layer == layer or r.layer2 == layer]
        
    def get_layer_number(self, layer_name: str) -> Optional[int]:
        """获取层号"""
        return self.layers.get(layer_name)
        
    def summarize(self) -> str:
        """生成规则摘要（用于 Prompt）"""
        lines = [
            f"工艺: {self.tech}",
            f"规则数: {len(self.rules)}",
            "层定义:"
        ]
        for name, num in self.layers.items():
            lines.append(f"  - {name}: {num}")
        lines.append("关键规则:")
        for rule in self.rules[:5]:  # 只显示前 5 条
            lines.append(f"  - {rule.id}: {rule.type.value}, min={rule.min_value}")
        return "\n".join(lines)
```

### 5.3.4 解析流程与错误处理

```python
class DRCParser:
    """DRC 规则解析器"""
    
    def parse(self, drc_path: str) -> DRCRuleSet:
        """
        解析 DRC 规则 JSON
        
        流程:
        1. 读取 JSON 文件
        2. Schema 校验
        3. 解析层映射
        4. 解析规则列表
        5. 验证规则引用的层存在
        """
        with open(drc_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self._validate_schema(data)
        
        tech = data['tech']
        layers = data['layers']
        
        rules = []
        for rule_data in data['rules']:
            rule = self._parse_rule(rule_data)
            
            # 验证层存在
            if rule.layer not in layers:
                raise DRCParseError(f"Unknown layer in rule {rule.id}: {rule.layer}")
            if rule.layer2 and rule.layer2 not in layers:
                raise DRCParseError(f"Unknown layer2 in rule {rule.id}: {rule.layer2}")
                
            rules.append(rule)
            
        return DRCRuleSet(
            tech=tech,
            layers=layers,
            rules=rules,
            schema_version=data.get('schema_version', '1.0.0')
        )
        
    def _parse_rule(self, data: Dict) -> DRCRule:
        """解析单条规则"""
        try:
            rule_type = RuleType(data['type'])
        except ValueError:
            raise DRCParseError(f"Unknown rule type: {data['type']}")
            
        return DRCRule(
            id=data['id'],
            type=rule_type,
            layer=data['layer'],
            layer2=data.get('layer2'),
            min_value=data.get('min'),
            max_value=data.get('max'),
            description=data.get('description')
        )
```

---

## 5.4 设计目标 JSON 解析器规格

### 5.4.1 设计目标 JSON Schema

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "DesignObjectives",
    "description": "设计目标和约束",
    "type": "object",
    "required": ["objectives"],
    "properties": {
        "schema_version": {
            "type": "string",
            "default": "1.0.0"
        },
        "objectives": {
            "type": "array",
            "items": {"$ref": "#/definitions/Objective"}
        },
        "constraints": {
            "$ref": "#/definitions/Constraints"
        }
    },
    "definitions": {
        "Objective": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string",
                    "enum": ["area_min", "matching_max", "performance_max", "cost_min"],
                    "description": "目标名称"
                },
                "weight": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 1.0,
                    "description": "权重"
                },
                "target": {
                    "type": "number",
                    "description": "目标值（可选）"
                }
            }
        },
        "Constraints": {
            "type": "object",
            "properties": {
                "max_iterations": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 5
                },
                "max_area": {
                    "type": "number",
                    "description": "最大面积 (平方微米)"
                },
                "min_matching_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "最小匹配度分数"
                }
            }
        }
    }
}
```

### 5.4.2 内部数据结构定义

```python
@dataclass
class Objective:
    """设计目标"""
    name: ObjectiveName               # 目标名称
    weight: float = 1.0               # 权重
    target: Optional[float] = None    # 目标值
    
    def get_evaluation_function(self) -> Callable:
        """获取评估函数"""
        return OBJECTIVE_EVALUATORS.get(self.name)

class ObjectiveName(Enum):
    """目标名称"""
    AREA_MIN = "area_min"
    MATCHING_MAX = "matching_max"
    PERFORMANCE_MAX = "performance_max"
    COST_MIN = "cost_min"

@dataclass
class DesignConstraints:
    """设计约束"""
    max_iterations: int = 5           # 最大迭代次数
    max_area: Optional[float] = None  # 最大面积 (平方微米)
    min_matching_score: Optional[float] = None  # 最小匹配度

@dataclass
class DesignObjectives:
    """设计目标集"""
    objectives: List[Objective]       # 目标列表
    constraints: DesignConstraints    # 约束条件
    
    def get_primary_objective(self) -> Objective:
        """获取主要目标（权重最高）"""
        return max(self.objectives, key=lambda o: o.weight)
        
    def summarize(self) -> str:
        """生成目标摘要（用于 Prompt）"""
        lines = ["设计目标:"]
        for obj in self.objectives:
            lines.append(f"  - {obj.name.value}: 权重={obj.weight}")
        lines.append("约束条件:")
        lines.append(f"  - 最大迭代: {self.constraints.max_iterations}")
        if self.constraints.max_area:
            lines.append(f"  - 最大面积: {self.constraints.max_area} um^2")
        return "\n".join(lines)

# 目标评估函数映射
OBJECTIVE_EVALUATORS: Dict[ObjectiveName, Callable] = {
    ObjectiveName.AREA_MIN: lambda metrics: 1.0 / (1.0 + metrics.total_area),
    ObjectiveName.MATCHING_MAX: lambda metrics: metrics.matching_score,
    # ...
}
```

### 5.4.3 目标到评估函数的映射

| 目标名称 | 评估函数 | 输入 | 输出范围 |
|----------|----------|------|----------|
| `area_min` | `1 / (1 + area)` | 版图总面积 | (0, 1] |
| `matching_max` | `matching_score` | 匹配度分数 | [0, 1] |
| `performance_max` | 自定义 | 性能指标 | [0, 1] |
| `cost_min` | `1 / (1 + cost)` | 成本估算 | (0, 1] |

---

## 5.5 MCP 工具调用输入/输出格式

### 5.5.1 工具调用请求结构

```python
@dataclass
class ToolCallRequest:
    """MCP 工具调用请求"""
    name: str                         # 工具名（如 "layout.create_common_centroid_pair"）
    input: Dict[str, Any]             # 输入参数
    trace_id: str                     # 追踪 ID（用于日志关联）
    timeout_ms: int = 30000           # 超时时间 (毫秒)
    
    def to_mcp_message(self) -> Dict:
        """转换为 MCP JSON-RPC 消息"""
        return {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": self.name,
                "arguments": self.input
            },
            "id": self.trace_id
        }
```

**JSON 示例**:

```json
{
    "name": "layout.create_common_centroid_pair",
    "input": {
        "device_a": {
            "device_id": "M1",
            "type": "nmos",
            "w": 2e-6,
            "l": 0.18e-6,
            "m": 2
        },
        "device_b": {
            "device_id": "M2",
            "type": "nmos",
            "w": 2e-6,
            "l": 0.18e-6,
            "m": 2
        },
        "arrangement": "ABBA"
    },
    "trace_id": "trace_abc123"
}
```

### 5.5.2 工具调用响应结构

```python
@dataclass
class ToolCallResult:
    """MCP 工具调用结果"""
    ok: bool                          # 是否成功
    error: Optional[ToolError]        # 错误信息
    data: Optional[Dict[str, Any]]    # 返回数据
    duration_ms: int = 0              # 执行耗时

@dataclass
class ToolError:
    """工具错误"""
    code: str                         # 错误码
    message: str                      # 错误消息
    details: Optional[Dict] = None    # 详细信息
    
    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details
        }
```

**JSON 示例**:

```json
// 成功
{
    "ok": true,
    "error": null,
    "data": {
        "cell_name": "diff_pair_cc",
        "device_instances": ["M1_A1", "M2_B1", "M2_B2", "M1_A2"]
    },
    "duration_ms": 234
}

// 失败
{
    "ok": false,
    "error": {
        "code": "INVALID_PARAM",
        "message": "Devices must be same type for matching",
        "details": {
            "device_a_type": "nmos",
            "device_b_type": "pmos"
        }
    },
    "data": null,
    "duration_ms": 12
}
```

### 5.5.3 错误码设计

| 错误码 | HTTP 等价 | 含义 | 示例场景 |
|--------|-----------|------|----------|
| `INVALID_PARAM` | 400 | 参数非法 | 缺少必填参数、类型错误 |
| `NOT_FOUND` | 404 | 资源不存在 | 器件/规则未找到 |
| `CONFLICT` | 409 | 冲突 | 器件位置重叠 |
| `DRC_VIOLATION` | 422 | DRC 违规 | 生成的版图不合规 |
| `TIMEOUT` | 408 | 超时 | 操作超时 |
| `SERVICE_UNAVAILABLE` | 503 | 服务不可用 | KLayout 未启动 |
| `INTERNAL_ERROR` | 500 | 内部错误 | 未预期的异常 |

### 5.5.4 Orchestrator 中的封装与解析

```python
class MCPClient:
    """MCP 客户端"""
    
    def call_tool(self, request: ToolCallRequest) -> ToolCallResult:
        """
        调用 MCP 工具
        
        流程:
        1. 构造 JSON-RPC 消息
        2. 发送到 MCP Server
        3. 等待响应
        4. 解析响应为 ToolCallResult
        """
        start_time = time.time()
        
        try:
            # 发送请求
            mcp_message = request.to_mcp_message()
            response = self._send_and_receive(
                mcp_message,
                timeout_ms=request.timeout_ms
            )
            
            # 解析响应
            return self._parse_response(response, start_time)
            
        except TimeoutError:
            return ToolCallResult(
                ok=False,
                error=ToolError(code="TIMEOUT", message="Tool call timed out"),
                data=None,
                duration_ms=int((time.time() - start_time) * 1000)
            )
            
    def _parse_response(
        self,
        response: Dict,
        start_time: float
    ) -> ToolCallResult:
        """解析 MCP 响应"""
        duration_ms = int((time.time() - start_time) * 1000)
        
        if "error" in response:
            return ToolCallResult(
                ok=False,
                error=ToolError(
                    code=response["error"].get("code", "UNKNOWN"),
                    message=response["error"].get("message", "Unknown error"),
                    details=response["error"].get("data")
                ),
                data=None,
                duration_ms=duration_ms
            )
            
        result = response.get("result", {})
        
        # 检查 result 内部的 ok 字段（Skill 层面的成功/失败）
        if isinstance(result, dict) and "ok" in result:
            return ToolCallResult(
                ok=result["ok"],
                error=ToolError(**result["error"]) if result.get("error") else None,
                data=result.get("data"),
                duration_ms=duration_ms
            )
            
        # 直接返回 result 作为 data
        return ToolCallResult(
            ok=True,
            error=None,
            data=result,
            duration_ms=duration_ms
        )
```

---

## 5.6 版本管理与兼容性

### 5.6.1 Schema 版本字段

所有关键 JSON 格式都包含 `schema_version` 字段：

```json
{
    "schema_version": "1.0.0",
    ...
}
```

### 5.6.2 版本号规范

采用语义化版本号 (SemVer)：`MAJOR.MINOR.PATCH`

| 变更类型 | 版本号变化 | 兼容性 |
|----------|------------|--------|
| 新增可选字段 | MINOR + 1 | 向后兼容 |
| 修改字段类型 | MAJOR + 1 | 不兼容 |
| 删除字段 | MAJOR + 1 | 不兼容 |
| 修复文档/描述 | PATCH + 1 | 完全兼容 |

### 5.6.3 兼容性处理策略

```python
class SchemaVersionManager:
    """Schema 版本管理"""
    
    SUPPORTED_VERSIONS = {
        "netlist": ["1.0.0", "1.1.0"],
        "drc_rules": ["1.0.0"],
        "objectives": ["1.0.0"]
    }
    
    def check_compatibility(
        self,
        schema_type: str,
        version: str
    ) -> CompatibilityResult:
        """检查版本兼容性"""
        supported = self.SUPPORTED_VERSIONS.get(schema_type, [])
        
        if version in supported:
            return CompatibilityResult(
                compatible=True,
                migration_needed=False
            )
            
        # 检查主版本号是否相同
        major_version = version.split('.')[0]
        compatible_major = [v for v in supported if v.startswith(major_version)]
        
        if compatible_major:
            return CompatibilityResult(
                compatible=True,
                migration_needed=True,
                target_version=max(compatible_major)
            )
            
        return CompatibilityResult(
            compatible=False,
            error=f"Unsupported schema version: {version}"
        )
        
    def migrate(
        self,
        data: Dict,
        schema_type: str,
        from_version: str,
        to_version: str
    ) -> Dict:
        """迁移数据到目标版本"""
        migrations = self._get_migrations(schema_type)
        
        current = data.copy()
        for migration in migrations:
            if self._should_apply(migration, from_version, to_version):
                current = migration.apply(current)
                
        current["schema_version"] = to_version
        return current
```

---

## 附录 E：完整数据结构定义

```python
# 完整的 Python 类型定义文件 (types.py)

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime

# ============ 枚举定义 ============

class DeviceType(Enum):
    NMOS = "nmos"
    PMOS = "pmos"
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"

class NetType(Enum):
    SIGNAL = "signal"
    POWER = "power"
    GROUND = "ground"

class ModuleType(Enum):
    DIFFERENTIAL_PAIR = "differential_pair"
    CURRENT_MIRROR = "current_mirror"
    CASCODE = "cascode"
    TAIL_CURRENT_SOURCE = "tail_current_source"

class RuleType(Enum):
    WIDTH = "width"
    SPACING = "spacing"
    ENCLOSURE = "enclosure"
    EXTENSION = "extension"
    OVERLAP = "overlap"
    DENSITY = "density"

class ObjectiveName(Enum):
    AREA_MIN = "area_min"
    MATCHING_MAX = "matching_max"
    PERFORMANCE_MAX = "performance_max"
    COST_MIN = "cost_min"

class AgentState(Enum):
    INIT = "init"
    PARSING = "parsing"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    ITERATING = "iterating"
    COMPLETED = "completed"
    FAILED = "failed"
    DONE = "done"

# ============ 电路相关 ============

@dataclass
class DeviceParams:
    w: Optional[float] = None
    l: Optional[float] = None
    m: int = 1
    nf: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Device:
    id: str
    type: DeviceType
    model: Optional[str]
    params: DeviceParams
    terminals: Dict[str, str]

@dataclass
class TerminalRef:
    device_id: str
    terminal_name: str

@dataclass
class Net:
    name: str
    connections: List[TerminalRef]
    net_type: NetType = NetType.SIGNAL

@dataclass
class CircuitModule:
    id: str
    type: ModuleType
    devices: List[str]
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Circuit:
    name: str
    devices: List[Device]
    nets: List[Net]
    modules: List[CircuitModule] = field(default_factory=list)

# ============ DRC 相关 ============

@dataclass
class DRCRule:
    id: str
    type: RuleType
    layer: str
    layer2: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: Optional[str] = None

@dataclass
class DRCRuleSet:
    tech: str
    layers: Dict[str, int]
    rules: List[DRCRule]
    schema_version: str = "1.0.0"

# ============ 目标相关 ============

@dataclass
class Objective:
    name: ObjectiveName
    weight: float = 1.0
    target: Optional[float] = None

@dataclass
class DesignConstraints:
    max_iterations: int = 5
    max_area: Optional[float] = None
    min_matching_score: Optional[float] = None

@dataclass
class DesignObjectives:
    objectives: List[Objective]
    constraints: DesignConstraints = field(default_factory=DesignConstraints)

# ============ 任务上下文 ============

@dataclass
class TaskContext:
    circuit: Circuit
    drc_rules: DRCRuleSet
    objectives: DesignObjectives
    session_id: str = ""

# ============ 工具调用 ============

@dataclass
class ToolError:
    code: str
    message: str
    details: Optional[Dict] = None

@dataclass
class ToolCallRequest:
    name: str
    input: Dict[str, Any]
    trace_id: str
    timeout_ms: int = 30000

@dataclass
class ToolCallResult:
    ok: bool
    error: Optional[ToolError]
    data: Optional[Dict[str, Any]]
    duration_ms: int = 0

# ============ 规划相关 ============

@dataclass
class PlanStep:
    id: str
    name: str
    skill_name: str
    params: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"

@dataclass
class Plan:
    steps: List[PlanStep]
    summary: str = ""
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)

# ============ 评估相关 ============

@dataclass
class LayoutMetrics:
    drc_error_count: int = 0
    total_area: float = 0.0
    matching_score: float = 0.0
    utilization: float = 0.0

@dataclass
class EvaluationReport:
    metrics: LayoutMetrics
    qualitative_feedback: str
    score: float
    passed: bool
    suggestions: List[str] = field(default_factory=list)
```
