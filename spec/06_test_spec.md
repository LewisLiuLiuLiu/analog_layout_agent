# 第6章 测试与评估框架

---

## 6.1 测试目标与范围

### 6.1.1 测试目标

本测试框架的核心目标是确保 Agent 系统在各个层面的正确性、稳定性和可回归性：

| 目标 | 描述 | 对应测试类型 |
|------|------|--------------|
| **功能正确性** | Skill 模块输出符合预期 | 单元测试 |
| **端到端可用性** | 完整流程能够生成合规版图 | 集成测试 |
| **性能稳定性** | 关键指标不退化 | 回归测试 |
| **可靠性** | 异常情况下优雅降级 | 异常测试 |

### 6.1.2 关注指标

| 指标 | 定义 | 重要性 |
|------|------|--------|
| **DRC 错误数** | 版图违反设计规则的数量 | 核心指标，MVP 目标为 0 |
| **版图面积** | 生成版图的总面积 (平方微米) | 优化指标 |
| **匹配度分数** | 差分对/电流镜的匹配程度 | 优化指标 |
| **执行成功率** | 任务完成的成功率 | 可用性指标 |
| **执行耗时** | 单次任务的总耗时 | 性能指标 |

### 6.1.3 测试范围

**当前版本测试范围**：

| 范围 | 包含 | 不包含 |
|------|------|--------|
| 电路类型 | 单级差分运放 | 多级运放、复杂模拟系统 |
| 工艺 | Toy PDK / 开源 PDK 子集 | 完整商业 PDK |
| DRC 规则 | 基本宽度/间距/覆盖规则 | 复杂 DFM、天线规则 |
| Skill 模块 | P0/P1 优先级模块 | P2 及后续模块 |

---

## 6.2 单元测试 (Skill 级)

### 6.2.1 测试覆盖范围

需要覆盖的 Skill 模块：

| Skill 名称 | 优先级 | 测试重点 |
|------------|--------|----------|
| `netlist.parse` | P0 | 解析正确性、错误处理 |
| `layout.create_nmos_pcell` | P0 | 几何正确性、参数范围 |
| `layout.create_pmos_pcell` | P0 | 几何正确性、参数范围 |
| `layout.create_common_centroid_pair` | P0 | 对称性、排列正确性 |
| `layout.create_current_mirror` | P1 | 匹配布局正确性 |
| `layout.route_signal_nets_basic` | P1 | 布线连通性、DRC |
| `drc.run_check` | P0 | 检查准确性、报告格式 |
| `export.gds` | P0 | 导出完整性、格式正确 |
| `eval.compute_metrics` | P1 | 指标计算准确性 |

### 6.2.2 测试用例设计原则

每个 Skill 的测试用例应覆盖：

1. **正常输入**：标准参数组合
2. **边界条件**：最小值、最大值、临界值
3. **异常参数**：非法值、缺失参数、类型错误
4. **特殊场景**：空输入、极端尺寸

### 6.2.3 `netlist.parse` 测试用例

```python
# tests/unit/test_netlist_parser.py

import pytest
from opamp_agent.perception.netlist_parser import NetlistParser
from opamp_agent.exceptions import NetlistParseError

class TestNetlistParser:
    """网表解析器测试"""
    
    @pytest.fixture
    def parser(self):
        return NetlistParser()
    
    # ============ 正常输入测试 ============
    
    def test_parse_simple_diff_amp(self, parser, tmp_path):
        """测试解析简单差分运放网表"""
        netlist_content = '''
        {
            "schema_version": "1.0.0",
            "cells": [{
                "name": "diff_amp",
                "devices": [
                    {"id": "M1", "type": "nmos", "w": 2e-6, "l": 0.18e-6,
                     "terminals": {"g": "vinp", "d": "outp", "s": "tail", "b": "vss"}},
                    {"id": "M2", "type": "nmos", "w": 2e-6, "l": 0.18e-6,
                     "terminals": {"g": "vinn", "d": "outn", "s": "tail", "b": "vss"}}
                ],
                "nets": ["vinp", "vinn", "outp", "outn", "tail", "vss"]
            }],
            "top": "diff_amp"
        }
        '''
        netlist_file = tmp_path / "netlist.json"
        netlist_file.write_text(netlist_content)
        
        circuit = parser.parse(str(netlist_file))
        
        assert circuit.name == "diff_amp"
        assert len(circuit.devices) == 2
        assert len(circuit.nets) == 6
        assert circuit.get_device("M1") is not None
        assert circuit.get_device("M1").params.w == 2e-6
        
    def test_parse_with_module_identification(self, parser, tmp_path):
        """测试自动模块识别"""
        # ... 差分对应被识别为 DIFFERENTIAL_PAIR
        circuit = parser.parse(str(netlist_file), identify_modules=True)
        
        assert len(circuit.modules) >= 1
        assert any(m.type.value == "differential_pair" for m in circuit.modules)
    
    # ============ 边界条件测试 ============
    
    def test_parse_empty_devices(self, parser, tmp_path):
        """测试空器件列表"""
        netlist_content = '''
        {
            "cells": [{"name": "empty", "devices": [], "nets": []}],
            "top": "empty"
        }
        '''
        netlist_file = tmp_path / "empty.json"
        netlist_file.write_text(netlist_content)
        
        circuit = parser.parse(str(netlist_file))
        
        assert len(circuit.devices) == 0
        
    def test_parse_minimum_device(self, parser, tmp_path):
        """测试最小尺寸器件"""
        # w = l = 工艺最小值
        pass
    
    # ============ 异常参数测试 ============
    
    def test_parse_file_not_found(self, parser):
        """测试文件不存在"""
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path.json")
            
    def test_parse_invalid_json(self, parser, tmp_path):
        """测试无效 JSON"""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not a json")
        
        with pytest.raises(NetlistParseError) as exc_info:
            parser.parse(str(invalid_file))
        assert "Invalid JSON" in str(exc_info.value)
        
    def test_parse_missing_required_field(self, parser, tmp_path):
        """测试缺少必填字段"""
        netlist_content = '{"cells": []}'  # 缺少 "top"
        netlist_file = tmp_path / "missing.json"
        netlist_file.write_text(netlist_content)
        
        with pytest.raises(NetlistParseError) as exc_info:
            parser.parse(str(netlist_file))
        assert "required" in str(exc_info.value).lower()
        
    def test_parse_unknown_device_type(self, parser, tmp_path):
        """测试未知器件类型"""
        netlist_content = '''
        {
            "cells": [{
                "name": "test",
                "devices": [{"id": "X1", "type": "unknown_type", "terminals": {}}]
            }],
            "top": "test"
        }
        '''
        netlist_file = tmp_path / "unknown.json"
        netlist_file.write_text(netlist_content)
        
        with pytest.raises(NetlistParseError) as exc_info:
            parser.parse(str(netlist_file))
        assert "Unknown device type" in str(exc_info.value)
```

### 6.2.4 `layout.create_common_centroid_pair` 测试用例

```python
# tests/unit/test_skill_common_centroid.py

import pytest
from opamp_agent.skills.layout import create_common_centroid_pair
from opamp_agent.types import DeviceParams, SkillResult

class TestCreateCommonCentroidPair:
    """共质心配对布局技能测试"""
    
    @pytest.fixture
    def device_a(self):
        return DeviceParams(
            device_id="M1",
            type="nmos",
            w=2e-6,
            l=0.18e-6,
            m=2,
            nf=2
        )
        
    @pytest.fixture
    def device_b(self):
        return DeviceParams(
            device_id="M2",
            type="nmos",
            w=2e-6,
            l=0.18e-6,
            m=2,
            nf=2
        )
    
    # ============ 正常输入测试 ============
    
    def test_abba_arrangement(self, device_a, device_b):
        """测试 ABBA 排列"""
        result = create_common_centroid_pair(
            device_a=device_a,
            device_b=device_b,
            arrangement="ABBA"
        )
        
        assert result.ok is True
        assert result.data["arrangement"] == "ABBA"
        assert len(result.data["device_instances"]) == 4
        
        # 验证实例顺序
        instances = result.data["device_instances"]
        assert instances[0]["parent"] == "M1"  # A
        assert instances[1]["parent"] == "M2"  # B
        assert instances[2]["parent"] == "M2"  # B
        assert instances[3]["parent"] == "M1"  # A
        
    def test_abab_arrangement(self, device_a, device_b):
        """测试 ABAB 排列"""
        result = create_common_centroid_pair(
            device_a=device_a,
            device_b=device_b,
            arrangement="ABAB"
        )
        
        assert result.ok is True
        instances = result.data["device_instances"]
        assert instances[0]["parent"] == "M1"
        assert instances[1]["parent"] == "M2"
        assert instances[2]["parent"] == "M1"
        assert instances[3]["parent"] == "M2"
        
    def test_symmetry_info(self, device_a, device_b):
        """测试对称性信息"""
        result = create_common_centroid_pair(
            device_a=device_a,
            device_b=device_b,
            arrangement="ABBA",
            symmetry_axis="vertical"
        )
        
        assert result.ok is True
        symmetry = result.data["symmetry_info"]
        assert symmetry["axis"] == "vertical"
        assert "center" in symmetry
        assert len(symmetry["matched_pairs"]) == 2
        
    def test_bounding_box_calculation(self, device_a, device_b):
        """测试边界框计算"""
        result = create_common_centroid_pair(
            device_a=device_a,
            device_b=device_b,
            arrangement="ABBA"
        )
        
        bbox = result.data["bounding_box"]
        assert bbox["width"] > 0
        assert bbox["height"] > 0
    
    # ============ 边界条件测试 ============
    
    def test_minimum_spacing(self, device_a, device_b):
        """测试最小间距"""
        result = create_common_centroid_pair(
            device_a=device_a,
            device_b=device_b,
            spacing=0.1  # 最小间距
        )
        
        assert result.ok is True
        
    def test_single_finger(self):
        """测试单指器件"""
        device_a = DeviceParams(device_id="M1", type="nmos", w=1e-6, l=0.18e-6, nf=1)
        device_b = DeviceParams(device_id="M2", type="nmos", w=1e-6, l=0.18e-6, nf=1)
        
        result = create_common_centroid_pair(device_a, device_b)
        
        assert result.ok is True
    
    # ============ 异常参数测试 ============
    
    def test_different_device_types(self):
        """测试不同器件类型（应失败）"""
        device_a = DeviceParams(device_id="M1", type="nmos", w=2e-6, l=0.18e-6)
        device_b = DeviceParams(device_id="M2", type="pmos", w=2e-6, l=0.18e-6)
        
        result = create_common_centroid_pair(device_a, device_b)
        
        assert result.ok is False
        assert result.error.code == "INVALID_PARAM"
        assert "same type" in result.error.message.lower()
        
    def test_invalid_arrangement(self, device_a, device_b):
        """测试无效排列方式"""
        result = create_common_centroid_pair(
            device_a=device_a,
            device_b=device_b,
            arrangement="INVALID"
        )
        
        assert result.ok is False
        assert result.error.code == "INVALID_PARAM"
        
    def test_negative_spacing(self, device_a, device_b):
        """测试负间距"""
        result = create_common_centroid_pair(
            device_a=device_a,
            device_b=device_b,
            spacing=-1.0
        )
        
        assert result.ok is False
        assert result.error.code == "INVALID_PARAM"

    # ============ 几何验证 ============
    
    def test_symmetry_axis_position(self, device_a, device_b):
        """验证对称轴位置正确"""
        result = create_common_centroid_pair(
            device_a=device_a,
            device_b=device_b,
            arrangement="ABBA"
        )
        
        instances = result.data["device_instances"]
        center = result.data["symmetry_info"]["center"]
        
        # 验证左右实例关于对称轴对称
        left_center = (instances[0]["position"][0] + instances[1]["position"][0]) / 2
        right_center = (instances[2]["position"][0] + instances[3]["position"][0]) / 2
        
        assert abs(center[0] - (left_center + right_center) / 2) < 1e-6
```

### 6.2.5 `drc.run_check` 测试用例

```python
# tests/unit/test_skill_drc.py

import pytest
from opamp_agent.skills.drc import run_drc_check
from opamp_agent.types import SkillResult

class TestDRCRunCheck:
    """DRC 检查技能测试"""
    
    @pytest.fixture
    def valid_layout_path(self, tmp_path):
        """创建有效测试版图"""
        # 实际实现需要创建测试 GDS 文件
        return str(tmp_path / "valid_layout.gds")
        
    @pytest.fixture
    def drc_rules_path(self, tmp_path):
        """创建测试 DRC 规则"""
        rules_content = '''
        {
            "tech": "test_180nm",
            "layers": {"metal1": 1, "metal2": 2, "via1": 3},
            "rules": [
                {"id": "M1_MIN_WIDTH", "type": "width", "layer": "metal1", "min": 0.18e-6}
            ]
        }
        '''
        rules_file = tmp_path / "drc_rules.json"
        rules_file.write_text(rules_content)
        return str(rules_file)
    
    def test_drc_pass(self, valid_layout_path, drc_rules_path):
        """测试 DRC 通过"""
        result = run_drc_check(
            layout_path=valid_layout_path,
            drc_rules_path=drc_rules_path
        )
        
        assert result.ok is True
        assert result.data["passed"] is True
        assert result.data["summary"]["total_violations"] == 0
        
    def test_drc_report_format(self, valid_layout_path, drc_rules_path):
        """测试 DRC 报告格式"""
        result = run_drc_check(
            layout_path=valid_layout_path,
            drc_rules_path=drc_rules_path
        )
        
        assert "summary" in result.data
        assert "violations" in result.data
        assert "rules_checked" in result.data["summary"]
        
    def test_filter_specific_rules(self, valid_layout_path, drc_rules_path):
        """测试过滤特定规则"""
        result = run_drc_check(
            layout_path=valid_layout_path,
            drc_rules_path=drc_rules_path,
            rule_ids=["M1_MIN_WIDTH"]
        )
        
        assert result.ok is True
        # 仅检查指定规则
```

### 6.2.6 测试框架与工具

| 工具 | 用途 |
|------|------|
| **pytest** | 测试框架 |
| **pytest-cov** | 测试覆盖率 |
| **pytest-mock** | Mock 支持 |
| **pytest-asyncio** | 异步测试支持 |

**pytest 配置**:
```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
```

---

## 6.3 集成测试 (MVP 场景端到端)

### 6.3.1 标准 MVP 输入集合

#### 网表定义

```json
// tests/fixtures/mvp_diff_amp.json
{
    "schema_version": "1.0.0",
    "cells": [{
        "name": "mvp_diff_amp",
        "devices": [
            {
                "id": "M1",
                "type": "nmos",
                "model": "nmos_1v8",
                "w": 2e-6,
                "l": 0.18e-6,
                "m": 2,
                "nf": 2,
                "terminals": {"g": "vinp", "d": "outp", "s": "tail", "b": "vss"}
            },
            {
                "id": "M2",
                "type": "nmos",
                "model": "nmos_1v8",
                "w": 2e-6,
                "l": 0.18e-6,
                "m": 2,
                "nf": 2,
                "terminals": {"g": "vinn", "d": "outn", "s": "tail", "b": "vss"}
            },
            {
                "id": "M3",
                "type": "nmos",
                "model": "nmos_1v8",
                "w": 4e-6,
                "l": 0.5e-6,
                "m": 1,
                "terminals": {"g": "vbias", "d": "tail", "s": "vss", "b": "vss"}
            },
            {
                "id": "M4",
                "type": "pmos",
                "model": "pmos_1v8",
                "w": 4e-6,
                "l": 0.18e-6,
                "m": 2,
                "terminals": {"g": "outp", "d": "outp", "s": "vdd", "b": "vdd"}
            },
            {
                "id": "M5",
                "type": "pmos",
                "model": "pmos_1v8",
                "w": 4e-6,
                "l": 0.18e-6,
                "m": 2,
                "terminals": {"g": "outp", "d": "outn", "s": "vdd", "b": "vdd"}
            }
        ],
        "nets": ["vinp", "vinn", "outp", "outn", "tail", "vbias", "vdd", "vss"]
    }],
    "top": "mvp_diff_amp"
}
```

#### DRC 规则定义

```json
// tests/fixtures/mvp_drc_rules.json
{
    "schema_version": "1.0.0",
    "tech": "toy_180nm",
    "layers": {
        "active": 1,
        "poly": 2,
        "contact": 3,
        "metal1": 4,
        "via1": 5,
        "metal2": 6,
        "nwell": 7,
        "pwell": 8
    },
    "rules": [
        {"id": "POLY_MIN_WIDTH", "type": "width", "layer": "poly", "min": 0.18e-6},
        {"id": "POLY_MIN_SPACE", "type": "spacing", "layer": "poly", "min": 0.24e-6},
        {"id": "M1_MIN_WIDTH", "type": "width", "layer": "metal1", "min": 0.22e-6},
        {"id": "M1_MIN_SPACE", "type": "spacing", "layer": "metal1", "min": 0.22e-6},
        {"id": "M2_MIN_WIDTH", "type": "width", "layer": "metal2", "min": 0.28e-6},
        {"id": "M2_MIN_SPACE", "type": "spacing", "layer": "metal2", "min": 0.28e-6},
        {"id": "CONT_MIN_SIZE", "type": "width", "layer": "contact", "min": 0.22e-6},
        {"id": "CONT_MIN_SPACE", "type": "spacing", "layer": "contact", "min": 0.25e-6},
        {"id": "VIA1_MIN_SIZE", "type": "width", "layer": "via1", "min": 0.26e-6},
        {"id": "M1_CONT_ENCLOSURE", "type": "enclosure", "layer": "metal1", "layer2": "contact", "min": 0.06e-6}
    ]
}
```

#### 设计目标定义

```json
// tests/fixtures/mvp_objectives.json
{
    "schema_version": "1.0.0",
    "objectives": [
        {"name": "area_min", "weight": 0.5},
        {"name": "matching_max", "weight": 0.5}
    ],
    "constraints": {
        "max_iterations": 3,
        "max_area": 500.0
    }
}
```

### 6.3.2 集成测试流程

```python
# tests/integration/test_mvp_e2e.py

import pytest
from pathlib import Path
from opamp_agent.orchestrator import AgentOrchestrator
from opamp_agent.config import Config

class TestMVPEndToEnd:
    """MVP 场景端到端集成测试"""
    
    @pytest.fixture
    def config(self):
        """加载测试配置"""
        return Config.load("config/config.test.yaml")
        
    @pytest.fixture
    def orchestrator(self, config):
        """创建 Orchestrator"""
        return AgentOrchestrator(config)
        
    @pytest.fixture
    def mvp_inputs(self):
        """加载 MVP 输入"""
        fixtures = Path("tests/fixtures")
        return {
            "netlist_path": str(fixtures / "mvp_diff_amp.json"),
            "drc_rules_path": str(fixtures / "mvp_drc_rules.json"),
            "objectives_path": str(fixtures / "mvp_objectives.json")
        }
    
    def test_mvp_full_pipeline(self, orchestrator, mvp_inputs, tmp_path):
        """
        测试完整 MVP 流程
        
        通过标准:
        1. 任务成功完成
        2. DRC 错误数 = 0
        3. 生成有效 GDS 文件
        """
        output_path = tmp_path / "output.gds"
        
        result = orchestrator.run(
            netlist_path=mvp_inputs["netlist_path"],
            drc_rules_path=mvp_inputs["drc_rules_path"],
            objectives_path=mvp_inputs["objectives_path"],
            output_path=str(output_path)
        )
        
        # 验证任务成功
        assert result.success is True, f"Task failed: {result.error}"
        
        # 验证 DRC
        assert result.evaluation.metrics.drc_error_count == 0, \
            f"DRC errors: {result.evaluation.metrics.drc_error_count}"
        
        # 验证输出文件
        assert output_path.exists(), "GDS file not created"
        assert output_path.stat().st_size > 0, "GDS file is empty"
        
    def test_mvp_drc_zero(self, orchestrator, mvp_inputs, tmp_path):
        """测试 DRC 错误数为 0"""
        result = orchestrator.run(**mvp_inputs, output_path=str(tmp_path / "out.gds"))
        
        assert result.evaluation.metrics.drc_error_count == 0
        
    def test_mvp_area_within_limit(self, orchestrator, mvp_inputs, tmp_path):
        """测试面积在限制范围内"""
        result = orchestrator.run(**mvp_inputs, output_path=str(tmp_path / "out.gds"))
        
        max_area = 500.0  # 从 objectives 读取
        assert result.evaluation.metrics.total_area <= max_area, \
            f"Area {result.evaluation.metrics.total_area} exceeds limit {max_area}"
            
    def test_mvp_matching_score(self, orchestrator, mvp_inputs, tmp_path):
        """测试匹配度分数"""
        result = orchestrator.run(**mvp_inputs, output_path=str(tmp_path / "out.gds"))
        
        min_matching = 0.8
        assert result.evaluation.metrics.matching_score >= min_matching, \
            f"Matching score {result.evaluation.metrics.matching_score} below {min_matching}"
            
    def test_mvp_execution_time(self, orchestrator, mvp_inputs, tmp_path):
        """测试执行时间"""
        import time
        
        start = time.time()
        result = orchestrator.run(**mvp_inputs, output_path=str(tmp_path / "out.gds"))
        duration = time.time() - start
        
        max_duration = 300  # 5 分钟
        assert duration < max_duration, \
            f"Execution time {duration}s exceeds limit {max_duration}s"
```

### 6.3.3 通过标准定义

| 指标 | 通过标准 | 失败处理 |
|------|----------|----------|
| DRC 错误数 | = 0 | 测试失败 |
| 执行成功 | success = True | 测试失败 |
| 面积 | <= max_area (如配置) | 警告 |
| 匹配度 | >= 0.8 | 警告 |
| 执行时间 | < 300s | 警告 |

---

## 6.4 回归测试与基准用例集

### 6.4.1 基准用例集构成

| 用例 ID | 电路描述 | 拓扑复杂度 | 参数规格 |
|---------|----------|------------|----------|
| `mvp_001` | 基础差分运放 | 简单 | M1/M2: W=2u, L=0.18u |
| `mvp_002` | 宽器件差分运放 | 简单 | M1/M2: W=10u, L=0.18u |
| `mvp_003` | 长沟道差分运放 | 简单 | M1/M2: W=2u, L=0.5u |
| `mvp_004` | 高倍数差分运放 | 中等 | M1/M2: m=4 |
| `mvp_005` | 多指差分运放 | 中等 | M1/M2: nf=4 |

### 6.4.2 回归测试触发时机

| 触发条件 | 运行范围 | 目的 |
|----------|----------|------|
| Pull Request | 快速子集 (2-3 用例) | 快速验证 |
| 合并到 main | 完整基准集 | 确保无退化 |
| 每日构建 | 完整基准集 + 扩展集 | 持续监控 |
| 发布前 | 全量 + 压力测试 | 发布验证 |

### 6.4.3 回归测试脚本

```python
# tests/regression/run_regression.py

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class RegressionResult:
    """单个用例的回归结果"""
    case_id: str
    success: bool
    drc_errors: int
    area: float
    matching_score: float
    duration_seconds: float
    error_message: Optional[str] = None

@dataclass
class RegressionReport:
    """回归测试报告"""
    version: str
    timestamp: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    results: List[RegressionResult]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

class RegressionRunner:
    """回归测试运行器"""
    
    def __init__(self, config_path: str, baseline_path: str):
        self.config = Config.load(config_path)
        self.baseline = self._load_baseline(baseline_path)
        self.orchestrator = AgentOrchestrator(self.config)
        
    def run_all(self, cases_dir: str) -> RegressionReport:
        """运行所有回归用例"""
        cases = list(Path(cases_dir).glob("mvp_*.json"))
        results = []
        
        for case_path in cases:
            result = self._run_single_case(case_path)
            results.append(result)
            
        passed = sum(1 for r in results if r.success and r.drc_errors == 0)
        
        return RegressionReport(
            version=self._get_version(),
            timestamp=datetime.now().isoformat(),
            total_cases=len(results),
            passed_cases=passed,
            failed_cases=len(results) - passed,
            results=results
        )
        
    def _run_single_case(self, case_path: Path) -> RegressionResult:
        """运行单个用例"""
        case_id = case_path.stem
        
        try:
            import time
            start = time.time()
            
            result = self.orchestrator.run(
                netlist_path=str(case_path),
                drc_rules_path=self.config.drc_rules_path,
                objectives_path=self.config.objectives_path
            )
            
            duration = time.time() - start
            
            return RegressionResult(
                case_id=case_id,
                success=result.success,
                drc_errors=result.evaluation.metrics.drc_error_count,
                area=result.evaluation.metrics.total_area,
                matching_score=result.evaluation.metrics.matching_score,
                duration_seconds=duration
            )
            
        except Exception as e:
            return RegressionResult(
                case_id=case_id,
                success=False,
                drc_errors=-1,
                area=0,
                matching_score=0,
                duration_seconds=0,
                error_message=str(e)
            )
            
    def compare_with_baseline(self, report: RegressionReport) -> dict:
        """与基线比较"""
        comparison = {
            "regressions": [],
            "improvements": [],
            "unchanged": []
        }
        
        for result in report.results:
            baseline_result = self.baseline.get(result.case_id)
            if baseline_result is None:
                continue
                
            # DRC 退化
            if result.drc_errors > baseline_result["drc_errors"]:
                comparison["regressions"].append({
                    "case_id": result.case_id,
                    "metric": "drc_errors",
                    "baseline": baseline_result["drc_errors"],
                    "current": result.drc_errors
                })
            # 面积退化 (>10%)
            elif result.area > baseline_result["area"] * 1.1:
                comparison["regressions"].append({
                    "case_id": result.case_id,
                    "metric": "area",
                    "baseline": baseline_result["area"],
                    "current": result.area
                })
            # 改进
            elif result.area < baseline_result["area"] * 0.95:
                comparison["improvements"].append({
                    "case_id": result.case_id,
                    "metric": "area",
                    "baseline": baseline_result["area"],
                    "current": result.area
                })
            else:
                comparison["unchanged"].append(result.case_id)
                
        return comparison
```

### 6.4.4 回归报告格式

```json
{
    "version": "1.0.0",
    "timestamp": "2024-01-15T10:30:00Z",
    "total_cases": 5,
    "passed_cases": 5,
    "failed_cases": 0,
    "results": [
        {
            "case_id": "mvp_001",
            "success": true,
            "drc_errors": 0,
            "area": 125.5,
            "matching_score": 0.95,
            "duration_seconds": 45.2
        }
    ],
    "comparison": {
        "regressions": [],
        "improvements": [
            {
                "case_id": "mvp_001",
                "metric": "area",
                "baseline": 140.0,
                "current": 125.5
            }
        ],
        "unchanged": ["mvp_002", "mvp_003", "mvp_004", "mvp_005"]
    }
}
```

---

## 6.5 自动化与 CI 集成

### 6.5.1 CI 流水线设计

```yaml
# .github/workflows/ci.yml

name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # 快速检查 - 每次提交
  lint-and-typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Lint
        run: ruff check .
      - name: Type check
        run: mypy opamp_agent/

  # 单元测试 - 每次提交
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      - name: Run unit tests
        run: pytest tests/unit -v --cov=opamp_agent --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  # 集成测试 - PR 和 main 分支
  integration-tests:
    needs: unit-tests
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request' || github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install KLayout
        run: |
          sudo add-apt-repository ppa:nicola-mfb/klayout
          sudo apt update
          sudo apt install klayout
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run integration tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest tests/integration -v --tb=short

  # 回归测试 - 仅 main 分支
  regression-tests:
    needs: integration-tests
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Setup environment
        run: |
          # 完整环境配置
          pip install -r requirements.txt
          sudo apt install klayout
      - name: Run regression tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python tests/regression/run_regression.py
      - name: Upload regression report
        uses: actions/upload-artifact@v3
        with:
          name: regression-report
          path: reports/regression_*.json
```

### 6.5.2 测试运行频率

| 测试类型 | 触发条件 | 预计耗时 |
|----------|----------|----------|
| Lint + Type Check | 每次提交 | 1-2 分钟 |
| 单元测试 | 每次提交 | 2-5 分钟 |
| 集成测试 | PR / main 合并 | 10-15 分钟 |
| 回归测试 | main 合并 / 每日 | 30-60 分钟 |
| 全量测试 | 发布前 | 2-3 小时 |

### 6.5.3 测试环境准备

```python
# tests/conftest.py

import pytest
import os

def pytest_configure(config):
    """pytest 配置"""
    # 设置环境变量
    os.environ.setdefault("APP_ENV", "test")
    
@pytest.fixture(scope="session")
def mock_llm_for_ci():
    """CI 环境下的 LLM Mock"""
    if os.environ.get("CI"):
        # CI 环境使用 Mock
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.generate.return_value.content = "{}"
        return mock
    return None

@pytest.fixture(scope="session")
def klayout_available():
    """检查 KLayout 是否可用"""
    import subprocess
    try:
        subprocess.run(["klayout", "-v"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def pytest_collection_modifyitems(config, items):
    """根据环境跳过某些测试"""
    if os.environ.get("CI") and not os.environ.get("OPENAI_API_KEY"):
        skip_llm = pytest.mark.skip(reason="LLM API key not available in CI")
        for item in items:
            if "llm" in item.keywords or "integration" in item.keywords:
                item.add_marker(skip_llm)
```

---

## 6.6 评估与指标监控

### 6.6.1 量化指标定义

| 指标 | 计算方式 | 目标 |
|------|----------|------|
| **测试通过率** | passed / total | > 95% |
| **DRC 零错误率** | drc_zero_cases / total | 100% |
| **平均面积** | sum(area) / n | 持续下降 |
| **平均匹配度** | sum(matching) / n | > 0.9 |
| **P95 执行时间** | 95th percentile | < 120s |

### 6.6.2 指标记录

```python
# opamp_agent/metrics.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import List
import json

@dataclass
class MetricsRecord:
    """指标记录"""
    timestamp: str
    version: str
    
    # 测试指标
    test_pass_rate: float
    drc_zero_rate: float
    
    # 性能指标
    avg_area: float
    avg_matching_score: float
    p95_duration_seconds: float
    
    # 详细数据
    case_results: List[dict] = field(default_factory=list)
    
    def to_json(self) -> str:
        return json.dumps({
            "timestamp": self.timestamp,
            "version": self.version,
            "metrics": {
                "test_pass_rate": self.test_pass_rate,
                "drc_zero_rate": self.drc_zero_rate,
                "avg_area": self.avg_area,
                "avg_matching_score": self.avg_matching_score,
                "p95_duration_seconds": self.p95_duration_seconds
            }
        }, indent=2)

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, output_dir: str = "metrics"):
        self.output_dir = output_dir
        
    def collect_from_report(self, report: RegressionReport) -> MetricsRecord:
        """从回归报告收集指标"""
        results = report.results
        n = len(results)
        
        if n == 0:
            raise ValueError("No results to collect metrics from")
        
        # 计算各项指标
        pass_rate = report.passed_cases / n
        drc_zero = sum(1 for r in results if r.drc_errors == 0) / n
        avg_area = sum(r.area for r in results) / n
        avg_matching = sum(r.matching_score for r in results) / n
        
        # P95 执行时间
        durations = sorted(r.duration_seconds for r in results)
        p95_idx = int(n * 0.95)
        p95_duration = durations[min(p95_idx, n-1)]
        
        return MetricsRecord(
            timestamp=datetime.now().isoformat(),
            version=report.version,
            test_pass_rate=pass_rate,
            drc_zero_rate=drc_zero,
            avg_area=avg_area,
            avg_matching_score=avg_matching,
            p95_duration_seconds=p95_duration,
            case_results=[asdict(r) for r in results]
        )
        
    def save(self, record: MetricsRecord):
        """保存指标记录"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        filename = f"metrics_{record.timestamp.replace(':', '-')}.json"
        filepath = Path(self.output_dir) / filename
        
        with open(filepath, 'w') as f:
            f.write(record.to_json())
```

### 6.6.3 长期趋势监控

```python
# scripts/plot_metrics_trend.py

import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

def load_metrics_history(metrics_dir: str) -> list:
    """加载历史指标"""
    records = []
    for f in sorted(Path(metrics_dir).glob("metrics_*.json")):
        with open(f) as fp:
            records.append(json.load(fp))
    return records

def plot_trends(records: list, output_path: str):
    """绘制趋势图"""
    timestamps = [r["timestamp"][:10] for r in records]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # DRC 零错误率
    axes[0, 0].plot(timestamps, [r["metrics"]["drc_zero_rate"] for r in records])
    axes[0, 0].set_title("DRC Zero Error Rate")
    axes[0, 0].set_ylim(0, 1.1)
    
    # 平均面积
    axes[0, 1].plot(timestamps, [r["metrics"]["avg_area"] for r in records])
    axes[0, 1].set_title("Average Area")
    
    # 平均匹配度
    axes[1, 0].plot(timestamps, [r["metrics"]["avg_matching_score"] for r in records])
    axes[1, 0].set_title("Average Matching Score")
    axes[1, 0].set_ylim(0, 1.1)
    
    # P95 执行时间
    axes[1, 1].plot(timestamps, [r["metrics"]["p95_duration_seconds"] for r in records])
    axes[1, 1].set_title("P95 Duration (seconds)")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Trend plot saved to {output_path}")

if __name__ == "__main__":
    records = load_metrics_history("metrics")
    plot_trends(records, "reports/metrics_trend.png")
```

---

## 附录 G：测试检查清单

### 发布前测试检查清单

- [ ] 所有单元测试通过
- [ ] 所有集成测试通过
- [ ] 回归测试无退化
- [ ] DRC 零错误率 = 100%
- [ ] 代码覆盖率 >= 80%
- [ ] 性能指标无显著下降
- [ ] 文档更新
- [ ] CHANGELOG 更新
