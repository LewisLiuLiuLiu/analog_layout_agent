"""
Verification Engine - 验证引擎

提供DRC（设计规则检查）和LVS（版图与原理图对比）功能
"""

import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# 添加路径
_BASE_PATH = Path(__file__).parent.parent
if str(_BASE_PATH) not in sys.path:
    sys.path.insert(0, str(_BASE_PATH))

_GLAYOUT_PATH = _BASE_PATH.parent / "gLayout" / "src"
if str(_GLAYOUT_PATH) not in sys.path:
    sys.path.insert(0, str(_GLAYOUT_PATH))

# 设置 PDK_ROOT 环境变量（glayout 初始化时需要）
_PDK_ROOT = _BASE_PATH.parent / "skywater-pdk"
if _PDK_ROOT.exists() and not os.environ.get('PDK_ROOT'):
    os.environ['PDK_ROOT'] = str(_PDK_ROOT)

from core.layout_context import LayoutContext
from mcp_server.handlers.error_handler import (
    DRCError, LVSError, ErrorHandler
)


@dataclass
class DRCViolation:
    """DRC违规记录"""
    rule: str                         # 违规规则名称
    category: str                     # 违规类别
    location: Tuple[float, float]     # 违规位置
    description: str                  # 违规描述
    severity: str = "error"           # 严重程度
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "rule": self.rule,
            "category": self.category,
            "location": {"x": self.location[0], "y": self.location[1]},
            "description": self.description,
            "severity": self.severity
        }


@dataclass
class DRCResult:
    """DRC检查结果"""
    passed: bool
    violations: List[DRCViolation]
    report_path: Optional[Path]
    checked_at: datetime = field(default_factory=datetime.now)
    suggestions: List[Any] = field(default_factory=list)  # 修复建议列表
    
    @property
    def violation_count(self) -> int:
        return len(self.violations)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "passed": self.passed,
            "violation_count": self.violation_count,
            "violations": [v.to_dict() for v in self.violations[:20]],  # 最多返回20条
            "report_path": str(self.report_path) if self.report_path else None,
            "checked_at": self.checked_at.isoformat()
        }
        
        # 添加修复建议
        if self.suggestions:
            result["suggestions"] = [
                s.to_dict() if hasattr(s, 'to_dict') else s 
                for s in self.suggestions[:20]
            ]
        
        return result
    
    def summary(self) -> str:
        """生成摘要"""
        if self.passed:
            return "DRC通过，无违规"
        
        # 按类别统计
        categories = {}
        for v in self.violations:
            categories[v.category] = categories.get(v.category, 0) + 1
        
        lines = [f"DRC失败，共{self.violation_count}条违规:"]
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            lines.append(f"  - {cat}: {count}条")
        
        # 添加建议摘要
        if self.suggestions:
            lines.append(f"\n已生成{len(self.suggestions)}条修复建议")
        
        return "\n".join(lines)


@dataclass
class LVSMismatch:
    """LVS不匹配记录"""
    mismatch_type: str      # 不匹配类型
    layout_element: str     # 版图元素
    schematic_element: str  # 原理图元素
    description: str        # 描述
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mismatch_type": self.mismatch_type,
            "layout_element": self.layout_element,
            "schematic_element": self.schematic_element,
            "description": self.description
        }


@dataclass
class LVSResult:
    """LVS检查结果"""
    matched: bool
    mismatches: List[LVSMismatch]
    report_path: Optional[Path]
    checked_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "mismatch_count": len(self.mismatches),
            "mismatches": [m.to_dict() for m in self.mismatches[:20]],
            "report_path": str(self.report_path) if self.report_path else None,
            "checked_at": self.checked_at.isoformat()
        }


class VerificationEngine:
    """验证引擎
    
    提供DRC和LVS验证功能，集成KLayout和Netgen工具。
    
    Usage:
        >>> engine = VerificationEngine(context)
        >>> drc_result = engine.run_drc()
        >>> lvs_result = engine.run_lvs(schematic_netlist)
    """
    
    def __init__(self, context: LayoutContext):
        """初始化验证引擎
        
        Args:
            context: 布局上下文
        """
        self.context = context
        self.error_handler = ErrorHandler()
        self._glayout_available = self._check_glayout()
    
    def _check_glayout(self) -> bool:
        """检查gLayout是否可用"""
        try:
            from glayout.pdk.mappedpdk import MappedPDK
            return True
        except ImportError:
            return False
    
    def _get_pdk(self):
        """获取PDK实例"""
        try:
            return self.context.pdk
        except RuntimeError:
            return None
    
    def run_drc(
        self,
        component_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        output_format: str = "summary",
        include_suggestions: bool = True
    ) -> DRCResult:
        """执行DRC检查
        
        Args:
            component_name: 组件名称，默认检查顶层
            output_dir: 输出目录
            output_format: 输出格式 "summary" 或 "detailed"
            include_suggestions: 是否生成修复建议
            
        Returns:
            DRC检查结果
        """
        pdk = self._get_pdk()
        
        if pdk is None or not self._glayout_available:
            # 模拟模式
            return self._mock_drc_result()
        
        try:
            # 获取组件
            if component_name:
                comp = self.context.get_component(component_name)
                if comp is None:
                    raise DRCError(f"组件不存在: {component_name}")
            else:
                # 使用顶层组件
                comp = self.context.build_top_level()
            
            # 设置输出目录
            if output_dir:
                output_path = Path(output_dir)
            else:
                output_path = self.context.get_output_dir() / "drc"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 执行DRC
            passed = pdk.drc(comp, str(output_path))
            
            # 解析报告
            violations = self._parse_drc_report(output_path)
            
            # 创建结果
            result = DRCResult(
                passed=passed,
                violations=violations,
                report_path=output_path / "drc_report.lyrdb"
            )
            
            # 生成修复建议
            if include_suggestions and not passed and violations:
                result.suggestions = self._generate_fix_suggestions(violations)
            
            return result
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise DRCError(f"DRC检查失败: {e}")
    
    def _generate_fix_suggestions(self, violations: List[DRCViolation]) -> List[Any]:
        """生成DRC修复建议
        
        Args:
            violations: DRC违规列表
            
        Returns:
            修复建议列表
        """
        try:
            from core.drc_advisor import DRCAdvisor
            advisor = DRCAdvisor(self.context.pdk_name or "sky130")
            
            # 创建临时结果用于分析
            temp_result = DRCResult(
                passed=False,
                violations=violations,
                report_path=None
            )
            
            return advisor.analyze(temp_result)
        except Exception as e:
            import logging
            logging.warning(f"生成DRC修复建议失败: {e}")
            return []
    
    def run_lvs(
        self,
        schematic_netlist: str,
        component_name: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> LVSResult:
        """执行LVS检查
        
        Args:
            schematic_netlist: 参考原理图网表（路径或内容）
            component_name: 组件名称，默认使用顶层
            output_dir: 输出目录
            
        Returns:
            LVS检查结果
        """
        pdk = self._get_pdk()
        
        if pdk is None or not self._glayout_available:
            # 模拟模式
            return self._mock_lvs_result()
        
        try:
            # 获取组件
            if component_name:
                comp = self.context.get_component(component_name)
                if comp is None:
                    raise LVSError(f"组件不存在: {component_name}")
            else:
                comp = self.context.build_top_level()
            
            # 设置输出目录
            if output_dir:
                output_path = Path(output_dir)
            else:
                output_path = self.context.get_output_dir() / "lvs"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 处理网表输入
            if Path(schematic_netlist).exists():
                netlist_path = schematic_netlist
            else:
                # 假设是网表内容，写入临时文件
                netlist_path = output_path / "schematic.spice"
                with open(netlist_path, 'w') as f:
                    f.write(schematic_netlist)
            
            # 执行LVS
            pdk_root = os.environ.get('PDK_ROOT')
            result = pdk.lvs_netgen(
                layout=comp,
                design_name=comp.name,
                pdk_root=pdk_root,
                netlist=str(netlist_path),
                output_file_path=str(output_path)
            )
            
            matched = result.get('match', False)
            mismatches = self._parse_lvs_result(result)
            
            return LVSResult(
                matched=matched,
                mismatches=mismatches,
                report_path=output_path / "lvs_report.txt"
            )
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise LVSError(f"LVS检查失败: {e}")
    
    def extract_netlist(
        self,
        component_name: Optional[str] = None,
        format: str = "spice",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """提取SPICE网表
        
        Args:
            component_name: 组件名称，默认使用顶层
            format: 输出格式 "spice" 或 "spectre"
            output_file: 输出文件路径
            
        Returns:
            提取结果
        """
        if not self._glayout_available:
            return self._mock_netlist_result(component_name, format)
        
        try:
            # 获取组件信息
            if component_name:
                info = self.context.get_component_info(component_name)
                if info is None:
                    raise ValueError(f"组件不存在: {component_name}")
            else:
                component_name = self.context.design_name
                info = None
            
            # 如果组件有网表信息
            netlist_content = None
            if info and info.component and hasattr(info.component, 'info'):
                netlist_content = info.component.info.get('netlist')
            
            if netlist_content is None:
                # 生成基本网表
                netlist_content = self._generate_basic_netlist(component_name)
            
            # 写入文件
            if output_file:
                output_path = Path(output_file)
            else:
                output_path = self.context.get_output_dir() / f"{component_name}.{format}"
            
            with open(output_path, 'w') as f:
                f.write(netlist_content)
            
            return {
                "success": True,
                "component_name": component_name,
                "format": format,
                "output_file": str(output_path),
                "netlist_preview": netlist_content[:500] if netlist_content else None
            }
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise ValueError(f"网表提取失败: {e}")
    
    def _parse_drc_report(self, output_dir: Path) -> List[DRCViolation]:
        """解析DRC报告
        
        支持多种格式:
        1. LYRDB (KLayout数据库格式)
        2. XML 格式
        3. 文本报告格式
        
        Args:
            output_dir: 输出目录
            
        Returns:
            DRC违规列表
        """
        import re
        import logging
        
        violations = []
        
        # 尝试多种文件格式
        possible_files = [
            ("lyrdb", output_dir / "drc_report.lyrdb"),
            ("xml", output_dir / "drc_report.xml"),
            ("txt", output_dir / "drc_report.txt"),
            ("rpt", output_dir / "drc.rpt"),
        ]
        
        for fmt, report_file in possible_files:
            if not report_file.exists():
                continue
                
            try:
                if fmt == "lyrdb":
                    violations = self._parse_lyrdb_format(report_file)
                elif fmt == "xml":
                    violations = self._parse_xml_format(report_file)
                elif fmt in ("txt", "rpt"):
                    violations = self._parse_text_format(report_file)
                
                if violations:
                    logging.info(f"成功解析DRC报告: {report_file} ({len(violations)}条违规)")
                    return violations
                    
            except Exception as e:
                logging.warning(f"解析 {report_file} 失败: {e}")
                continue
        
        return violations
    
    def _parse_lyrdb_format(self, report_file: Path) -> List[DRCViolation]:
        """解析KLayout LYRDB格式
        
        LYRDB是KLayout的标记数据库格式，可能是XML或二进制格式。
        先尝试作为XML解析，失败则使用klayout API。
        """
        violations = []
        
        # 先尝试作为XML解析
        try:
            tree = ET.parse(report_file)
            root = tree.getroot()
            
            # KLayout LYRDB XML格式
            for category in root.findall('.//category'):
                cat_name = category.get('name', 'unknown')
                for item in category.findall('.//item'):
                    # 提取坐标
                    shape = item.find('.//shape')
                    x, y = 0.0, 0.0
                    if shape is not None:
                        box = shape.find('box')
                        if box is not None:
                            x = float(box.get('x1', 0))
                            y = float(box.get('y1', 0))
                    
                    violations.append(DRCViolation(
                        rule=item.get('rule', cat_name),
                        category=cat_name,
                        location=(x, y),
                        description=item.get('description', item.text or '')
                    ))
            
            # 如果没有找到category，尝试直接找item
            if not violations:
                for item in root.findall('.//item'):
                    rule = item.get('rule', 'unknown')
                    category = item.get('category', 'unknown')
                    x = float(item.get('x', 0))
                    y = float(item.get('y', 0))
                    desc = item.text or ''
                    
                    violations.append(DRCViolation(
                        rule=rule,
                        category=category,
                        location=(x, y),
                        description=desc
                    ))
                    
        except ET.ParseError:
            # 可能是二进制格式，尝试使用klayout API
            try:
                import klayout.db as kdb
                rdb = kdb.ReportDatabase()
                rdb.load(str(report_file))
                
                for cat in rdb.each_category():
                    cat_name = cat.name()
                    for item in rdb.each_item_per_category(cat.rdb_id()):
                        # 获取位置
                        x, y = 0.0, 0.0
                        for value in item.each_value():
                            if hasattr(value, 'box'):
                                box = value.box()
                                x = (box.left + box.right) / 2 / 1000  # 转换为um
                                y = (box.bottom + box.top) / 2 / 1000
                                break
                        
                        violations.append(DRCViolation(
                            rule=cat_name,
                            category=cat_name,
                            location=(x, y),
                            description=f"DRC violation: {cat_name}"
                        ))
                        
            except ImportError:
                pass  # klayout不可用
            except Exception:
                pass
        
        return violations
    
    def _parse_xml_format(self, report_file: Path) -> List[DRCViolation]:
        """解析标准XML格式DRC报告"""
        violations = []
        
        tree = ET.parse(report_file)
        root = tree.getroot()
        
        # 通用XML格式
        for item in root.findall('.//violation') or root.findall('.//item') or root.findall('.//error'):
            rule = item.get('rule', item.get('name', 'unknown'))
            category = item.get('category', item.get('type', 'unknown'))
            
            # 尝试多种坐标格式
            x = float(item.get('x', item.get('x1', item.get('center_x', 0))))
            y = float(item.get('y', item.get('y1', item.get('center_y', 0))))
            
            desc = item.get('description', item.get('message', ''))
            if not desc and item.text:
                desc = item.text.strip()
            
            violations.append(DRCViolation(
                rule=rule,
                category=category,
                location=(x, y),
                description=desc
            ))
        
        return violations
    
    def _parse_text_format(self, report_file: Path) -> List[DRCViolation]:
        """解析文本格式DRC报告"""
        import re
        violations = []
        
        with open(report_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 常见的DRC报告格式匹配模式
        patterns = [
            # 格式1: "Rule: xxx at (x, y)"
            r'Rule:\s*(\S+).*?at\s*\(?\s*([\d.]+)\s*,\s*([\d.]+)\s*\)?',
            # 格式2: "xxx violation at x=..., y=..."
            r'(\S+)\s+violation.*?x\s*=\s*([\d.]+).*?y\s*=\s*([\d.]+)',
            # 格式3: "ERROR: xxx (x, y)"
            r'ERROR:\s*(\S+).*?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)',
            # 格式4: KLayout风格 "category: count"（只统计）
            r'^(\w+(?:\.\w+)*)\s*:\s*(\d+)\s*(?:error|violation)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                for match in matches:
                    if len(match) >= 3:
                        rule = match[0]
                        try:
                            x = float(match[1])
                            y = float(match[2])
                        except ValueError:
                            x, y = 0.0, 0.0
                        
                        violations.append(DRCViolation(
                            rule=rule,
                            category=rule.split('.')[0] if '.' in rule else rule,
                            location=(x, y),
                            description=f"DRC violation: {rule}"
                        ))
                    elif len(match) == 2:
                        # 只有规则名和计数
                        rule = match[0]
                        count = int(match[1])
                        for _ in range(min(count, 100)):  # 限制数量
                            violations.append(DRCViolation(
                                rule=rule,
                                category=rule.split('.')[0] if '.' in rule else rule,
                                location=(0.0, 0.0),
                                description=f"DRC violation: {rule}"
                            ))
                
                if violations:
                    break
        
        return violations
    
    def _parse_lvs_result(self, result: Dict) -> List[LVSMismatch]:
        """解析LVS结果
        
        Args:
            result: LVS结果字典
            
        Returns:
            不匹配列表
        """
        mismatches = []
        
        raw_mismatches = result.get('mismatches', [])
        for m in raw_mismatches:
            mismatches.append(LVSMismatch(
                mismatch_type=m.get('type', 'unknown'),
                layout_element=m.get('layout', ''),
                schematic_element=m.get('schematic', ''),
                description=m.get('description', '')
            ))
        
        return mismatches
    
    def _generate_basic_netlist(self, component_name: str) -> str:
        """生成基本网表
        
        Args:
            component_name: 组件名称
            
        Returns:
            网表内容
        """
        lines = [
            f"* Netlist for {component_name}",
            f"* Generated by Analog Layout Agent",
            f"* Date: {datetime.now().isoformat()}",
            "",
            f".subckt {component_name}"
        ]
        
        # 添加组件
        for name, info in self.context.registry:
            device_type = info.device_type
            params = info.params
            
            if device_type == "nmos":
                lines.append(
                    f"M{name} drain gate source bulk "
                    f"nfet W={params.get('width', 1)}u L={params.get('length', 0.15)}u"
                )
            elif device_type == "pmos":
                lines.append(
                    f"M{name} drain gate source bulk "
                    f"pfet W={params.get('width', 1)}u L={params.get('length', 0.15)}u"
                )
            elif device_type == "resistor":
                lines.append(
                    f"R{name} plus minus "
                    f"W={params.get('width', 0.5)}u L={params.get('length', 1)}u"
                )
            elif device_type == "mimcap":
                lines.append(
                    f"C{name} plus minus "
                    f"W={params.get('width', 1)}u L={params.get('length', 1)}u"
                )
        
        lines.append(f".ends {component_name}")
        lines.append("")
        
        return "\n".join(lines)
    
    # ============== 模拟模式结果 ==============
    
    def _mock_drc_result(self) -> DRCResult:
        """生成模拟DRC结果"""
        return DRCResult(
            passed=True,
            violations=[],
            report_path=None
        )
    
    def _mock_lvs_result(self) -> LVSResult:
        """生成模拟LVS结果"""
        return LVSResult(
            matched=True,
            mismatches=[],
            report_path=None
        )
    
    def _mock_netlist_result(
        self,
        component_name: Optional[str],
        format: str
    ) -> Dict[str, Any]:
        """生成模拟网表结果"""
        component_name = component_name or self.context.design_name
        
        netlist = self._generate_basic_netlist(component_name)
        
        return {
            "success": True,
            "component_name": component_name,
            "format": format,
            "output_file": None,
            "netlist_preview": netlist[:500],
            "_mock": True
        }


# ============== MCP工具定义 ==============

def get_verification_tools() -> List[Dict[str, Any]]:
    """获取验证工具定义列表"""
    from mcp_server.schemas.common_schemas import (
        RUN_DRC_SCHEMA, RUN_LVS_SCHEMA, EXTRACT_NETLIST_SCHEMA
    )
    
    return [
        {
            "name": "run_drc",
            "description": "执行DRC（设计规则检查），检测版图是否满足工艺规则",
            "inputSchema": RUN_DRC_SCHEMA,
            "category": "verification"
        },
        {
            "name": "run_lvs",
            "description": "执行LVS（版图与原理图对比），验证版图与设计意图是否一致",
            "inputSchema": RUN_LVS_SCHEMA,
            "category": "verification"
        },
        {
            "name": "extract_netlist",
            "description": "从版图提取SPICE网表，用于仿真和LVS",
            "inputSchema": EXTRACT_NETLIST_SCHEMA,
            "category": "verification"
        }
    ]
