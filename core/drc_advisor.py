"""
DRC Advisor - DRC自动修复建议器
DRC Advisor - DRC automatic fix suggestion engine

分析DRC违规并提供修复建议
Analyzes DRC violations and provides fix suggestions
"""

import os
import sys
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

# 添加路径 / Add paths
_BASE_PATH = Path(__file__).parent.parent
if str(_BASE_PATH) not in sys.path:
    sys.path.insert(0, str(_BASE_PATH))

_GLAYOUT_PATH = _BASE_PATH.parent / "gLayout" / "src"
if str(_GLAYOUT_PATH) not in sys.path:
    sys.path.insert(0, str(_GLAYOUT_PATH))

# 设置 PDK_ROOT 环境变量 / Set PDK_ROOT environment variable
_PDK_ROOT = _BASE_PATH.parent / "skywater-pdk"
if _PDK_ROOT.exists() and not os.environ.get('PDK_ROOT'):
    os.environ['PDK_ROOT'] = str(_PDK_ROOT)

from core.verification import DRCViolation, DRCResult


@dataclass
class FixSuggestion:
    """修复建议 / Fix suggestion"""
    violation_id: str           # 违规ID / Violation ID
    action: str                 # 修复动作 / Fix action
    target: str                 # 目标对象 / Target object
    parameter: str              # 需要修改的参数 / Parameter to modify
    current_value: Optional[float] = None   # 当前值 / Current value
    suggested_value: Optional[float] = None # 建议值 / Suggested value
    confidence: float = 0.5     # 置信度 (0-1) / Confidence (0-1)
    description: str = ""       # 描述 / Description
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 / Convert to dictionary"""
        return {
            "violation_id": self.violation_id,
            "action": self.action,
            "target": self.target,
            "parameter": self.parameter,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "confidence": self.confidence,
            "description": self.description
        }


class DRCAdvisor:
    """DRC修复建议器 / DRC Fix Advisor
    
    分析DRC违规并根据规则类型提供修复建议。
    Analyzes DRC violations and provides fix suggestions based on rule type.
    
    支持的违规类型 / Supported violation types:
    - spacing: 间距违规 -> 增加间距 / spacing violation -> increase spacing
    - width: 宽度违规 -> 增大线宽 / width violation -> increase width
    - enclosure: 包络违规 -> 扩大包围尺寸 / enclosure violation -> increase enclosure
    - density: 密度违规 -> 添加填充 / density violation -> add fill
    - area: 面积违规 -> 增大面积 / area violation -> increase area
    - overlap: 重叠违规 -> 调整位置 / overlap violation -> adjust position
    
    Usage:
        >>> advisor = DRCAdvisor(pdk_name="sky130")
        >>> suggestions = advisor.analyze(drc_result)
    """
    
    # 违规类型到修复动作的映射 / Mapping from violation type to fix action
    VIOLATION_ACTIONS = {
        # 间距相关 / Spacing related
        "spacing": "increase_spacing",
        "space": "increase_spacing",
        "sep": "increase_spacing",
        "separation": "increase_spacing",
        "min_sep": "increase_spacing",
        "s.": "increase_spacing",
        
        # 宽度相关
        "width": "increase_width",
        "min_width": "increase_width",
        "w.": "increase_width",
        
        # 包络相关
        "enclosure": "increase_enclosure",
        "enc": "increase_enclosure",
        "surround": "increase_enclosure",
        "overlap_enc": "increase_enclosure",
        
        # 面积相关
        "area": "increase_area",
        "min_area": "increase_area",
        
        # 密度相关
        "density": "add_fill",
        "fill": "add_fill",
        
        # 重叠相关
        "overlap": "adjust_position",
        "short": "adjust_position",
    }
    
    # PDK特定的设计规则（最小值）
    PDK_RULES = {
        "sky130": {
            "met1_spacing": 0.14,
            "met2_spacing": 0.14,
            "met3_spacing": 0.30,
            "met4_spacing": 0.30,
            "met5_spacing": 1.60,
            "met1_width": 0.14,
            "met2_width": 0.14,
            "met3_width": 0.30,
            "met4_width": 0.30,
            "met5_width": 1.60,
            "via_enclosure": 0.055,
            "poly_spacing": 0.21,
            "poly_width": 0.15,
            "diff_spacing": 0.27,
        },
        "gf180": {
            "met1_spacing": 0.23,
            "met2_spacing": 0.28,
            "met3_spacing": 0.28,
            "met4_spacing": 0.28,
            "met1_width": 0.23,
            "met2_width": 0.28,
            "met3_width": 0.28,
            "met4_width": 0.28,
            "via_enclosure": 0.05,
            "poly_spacing": 0.24,
            "poly_width": 0.18,
        },
        "ihp130": {
            "met1_spacing": 0.14,
            "met2_spacing": 0.14,
            "met1_width": 0.14,
            "met2_width": 0.14,
            "via_enclosure": 0.05,
            "poly_spacing": 0.18,
            "poly_width": 0.13,
        }
    }
    
    def __init__(self, pdk_name: str = "sky130"):
        """初始化建议器
        
        Args:
            pdk_name: PDK名称
        """
        self.pdk_name = pdk_name.lower()
        self.rules = self.PDK_RULES.get(self.pdk_name, self.PDK_RULES["sky130"])
    
    def analyze(self, drc_result: DRCResult) -> List[FixSuggestion]:
        """分析DRC结果并生成修复建议
        
        Args:
            drc_result: DRC检查结果
            
        Returns:
            修复建议列表
        """
        suggestions = []
        
        for i, violation in enumerate(drc_result.violations):
            suggestion = self._analyze_violation(violation, f"V{i:03d}")
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions
    
    def _analyze_violation(
        self, 
        violation: DRCViolation, 
        violation_id: str
    ) -> Optional[FixSuggestion]:
        """分析单个违规
        
        Args:
            violation: 违规记录
            violation_id: 违规ID
            
        Returns:
            修复建议，如果无法分析则返回None
        """
        rule = violation.rule.lower()
        category = violation.category.lower()
        
        # 确定违规类型和修复动作
        action = None
        for keyword, act in self.VIOLATION_ACTIONS.items():
            if keyword in rule or keyword in category:
                action = act
                break
        
        if action is None:
            # 默认建议
            return FixSuggestion(
                violation_id=violation_id,
                action="manual_review",
                target=f"location ({violation.location[0]:.2f}, {violation.location[1]:.2f})",
                parameter="unknown",
                confidence=0.1,
                description=f"无法自动分析违规 '{violation.rule}'，请手动检查"
            )
        
        # 根据动作生成具体建议
        if action == "increase_spacing":
            return self._suggest_spacing_fix(violation, violation_id)
        elif action == "increase_width":
            return self._suggest_width_fix(violation, violation_id)
        elif action == "increase_enclosure":
            return self._suggest_enclosure_fix(violation, violation_id)
        elif action == "increase_area":
            return self._suggest_area_fix(violation, violation_id)
        elif action == "add_fill":
            return self._suggest_fill(violation, violation_id)
        elif action == "adjust_position":
            return self._suggest_position_fix(violation, violation_id)
        
        return None
    
    def _suggest_spacing_fix(
        self, 
        violation: DRCViolation, 
        violation_id: str
    ) -> FixSuggestion:
        """生成间距修复建议"""
        # 从规则名提取层信息
        layer = self._extract_layer(violation.rule)
        min_spacing = self.rules.get(f"{layer}_spacing", 0.14)
        
        # 建议增加10%的余量
        suggested = min_spacing * 1.1
        
        return FixSuggestion(
            violation_id=violation_id,
            action="increase_spacing",
            target=f"位置 ({violation.location[0]:.2f}, {violation.location[1]:.2f})",
            parameter=f"{layer}_spacing",
            current_value=None,  # 实际值需要从版图中提取
            suggested_value=suggested,
            confidence=0.8,
            description=f"增加{layer}层间距到 {suggested:.3f}um，或改用更高金属层"
        )
    
    def _suggest_width_fix(
        self, 
        violation: DRCViolation, 
        violation_id: str
    ) -> FixSuggestion:
        """生成宽度修复建议"""
        layer = self._extract_layer(violation.rule)
        min_width = self.rules.get(f"{layer}_width", 0.14)
        
        # 建议增加10%的余量
        suggested = min_width * 1.1
        
        return FixSuggestion(
            violation_id=violation_id,
            action="increase_width",
            target=f"位置 ({violation.location[0]:.2f}, {violation.location[1]:.2f})",
            parameter=f"{layer}_width",
            current_value=None,
            suggested_value=suggested,
            confidence=0.85,
            description=f"增加{layer}层宽度到 {suggested:.3f}um"
        )
    
    def _suggest_enclosure_fix(
        self, 
        violation: DRCViolation, 
        violation_id: str
    ) -> FixSuggestion:
        """生成包络修复建议"""
        min_enc = self.rules.get("via_enclosure", 0.05)
        suggested = min_enc * 1.2
        
        return FixSuggestion(
            violation_id=violation_id,
            action="increase_enclosure",
            target=f"位置 ({violation.location[0]:.2f}, {violation.location[1]:.2f})",
            parameter="via_enclosure",
            current_value=None,
            suggested_value=suggested,
            confidence=0.75,
            description=f"增加via包围尺寸到 {suggested:.3f}um"
        )
    
    def _suggest_area_fix(
        self, 
        violation: DRCViolation, 
        violation_id: str
    ) -> FixSuggestion:
        """生成面积修复建议"""
        return FixSuggestion(
            violation_id=violation_id,
            action="increase_area",
            target=f"位置 ({violation.location[0]:.2f}, {violation.location[1]:.2f})",
            parameter="area",
            confidence=0.6,
            description="增加金属面积或添加金属stub"
        )
    
    def _suggest_fill(
        self, 
        violation: DRCViolation, 
        violation_id: str
    ) -> FixSuggestion:
        """生成密度填充建议"""
        return FixSuggestion(
            violation_id=violation_id,
            action="add_fill",
            target=f"区域 ({violation.location[0]:.2f}, {violation.location[1]:.2f})",
            parameter="density",
            confidence=0.7,
            description="添加填充图形以满足密度要求"
        )
    
    def _suggest_position_fix(
        self, 
        violation: DRCViolation, 
        violation_id: str
    ) -> FixSuggestion:
        """生成位置调整建议"""
        return FixSuggestion(
            violation_id=violation_id,
            action="adjust_position",
            target=f"位置 ({violation.location[0]:.2f}, {violation.location[1]:.2f})",
            parameter="position",
            confidence=0.5,
            description="调整组件位置以消除重叠/短路"
        )
    
    def _extract_layer(self, rule_name: str) -> str:
        """从规则名中提取层信息"""
        rule_lower = rule_name.lower()
        
        # 检查常见层名
        layers = ["met5", "met4", "met3", "met2", "met1", "poly", "diff", "via"]
        for layer in layers:
            if layer in rule_lower:
                return layer
        
        # 检查简写
        match = re.search(r'm(\d)', rule_lower)
        if match:
            return f"met{match.group(1)}"
        
        return "met1"  # 默认返回met1
    
    def get_summary(self, suggestions: List[FixSuggestion]) -> str:
        """生成修复建议摘要
        
        Args:
            suggestions: 修复建议列表
            
        Returns:
            摘要文本
        """
        if not suggestions:
            return "无DRC违规或无法生成建议"
        
        # 按动作类型统计
        action_counts = {}
        for s in suggestions:
            action_counts[s.action] = action_counts.get(s.action, 0) + 1
        
        lines = [f"DRC修复建议摘要 ({len(suggestions)}条):"]
        
        action_descriptions = {
            "increase_spacing": "增加间距",
            "increase_width": "增大宽度",
            "increase_enclosure": "增加包络",
            "increase_area": "增大面积",
            "add_fill": "添加填充",
            "adjust_position": "调整位置",
            "manual_review": "需手动检查",
        }
        
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            desc = action_descriptions.get(action, action)
            lines.append(f"  - {desc}: {count}条")
        
        # 高置信度建议
        high_conf = [s for s in suggestions if s.confidence >= 0.7]
        if high_conf:
            lines.append(f"\n高置信度建议 ({len(high_conf)}条):")
            for s in high_conf[:5]:  # 最多显示5条
                lines.append(f"  - [{s.violation_id}] {s.description}")
        
        return "\n".join(lines)


def analyze_drc_result(
    drc_result: DRCResult, 
    pdk_name: str = "sky130"
) -> Dict[str, Any]:
    """分析DRC结果并返回修复建议
    
    便捷函数，用于快速分析DRC结果。
    
    Args:
        drc_result: DRC检查结果
        pdk_name: PDK名称
        
    Returns:
        包含建议的字典
    """
    advisor = DRCAdvisor(pdk_name)
    suggestions = advisor.analyze(drc_result)
    
    return {
        "passed": drc_result.passed,
        "violation_count": len(drc_result.violations),
        "suggestion_count": len(suggestions),
        "suggestions": [s.to_dict() for s in suggestions],
        "summary": advisor.get_summary(suggestions)
    }
