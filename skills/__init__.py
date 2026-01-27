"""
Layout Agent Skills 模块

使用 PydanticAI Skills 框架封装所有布局相关技能，实现渐进式披露。
通过 SkillsToolset 提供统一的技能管理和按需加载机制。

技能列表:
- device-creation: 基础器件创建（NMOS/PMOS/电容/电阻/Via）
- routing-connection: 智能布线连接（smart_route/c_route/l_route等）
- placement-layout: 组件放置与布局（place/move/align/interdigitize）
- circuit-building: 复合电路构建（电流镜/差分对）
- verification-drc: 设计规则验证（DRC/LVS/网表提取）
- export-query: 导出与查询（GDS导出/组件查询）
"""

from pydantic_ai_skills import SkillsToolset

from .device_skill import create_device_skill
from .routing_skill import create_routing_skill
from .placement_skill import create_placement_skill
from .circuit_skill import create_circuit_skill
from .verification_skill import create_verification_skill
from .export_skill import create_export_skill


def create_layout_skills_toolset() -> SkillsToolset:
    """创建布局技能工具集
    
    将所有布局相关技能封装为 SkillsToolset，支持：
    - 渐进式披露：按需加载技能详细信息
    - 统一接口：通过 load_skill/run_skill_script 调用
    - Token优化：初始只加载技能列表，减少prompt大小
    
    Returns:
        SkillsToolset: 包含所有布局技能的工具集
        
    Example:
        >>> from analog_layout_agent.skills import create_layout_skills_toolset
        >>> skills_toolset = create_layout_skills_toolset()
        >>> agent = Agent(model, toolsets=[skills_toolset])
    """
    return SkillsToolset(
        skills=[
            create_device_skill(),
            create_routing_skill(),
            create_placement_skill(),
            create_circuit_skill(),
            create_verification_skill(),
            create_export_skill(),
        ]
    )


__all__ = [
    'create_layout_skills_toolset',
    'SkillsToolset',
    'create_device_skill',
    'create_routing_skill',
    'create_placement_skill',
    'create_circuit_skill',
    'create_verification_skill',
    'create_export_skill',
]
