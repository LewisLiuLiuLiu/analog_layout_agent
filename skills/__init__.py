"""
Layout Agent Skills 模块

使用 PydanticAI Skills 框架从文件系统加载技能，实现渐进式披露。
遵循 Anthropic Agent Skills 标准目录结构：

skills/
├── device-creation/
│   ├── SKILL.md          # 必需：技能指令 + YAML元数据
│   ├── scripts/          # 可执行脚本
│   ├── references/       # 参考文档
│   └── assets/           # 资源文件
├── routing-connection/
│   └── ...
└── ...

技能列表:
- device-creation: 基础器件创建（NMOS/PMOS/电容/电阻/Via）
- routing-connection: 智能布线连接（smart_route/c_route/l_route等）
- placement-layout: 组件放置与布局（place/move/align/interdigitize）
- circuit-building: 复合电路构建（电流镜/差分对）
- verification-drc: 设计规则验证（DRC/LVS/网表提取）
- export-query: 导出与查询（GDS导出/组件查询）
"""

from pathlib import Path
from typing import Optional

from pydantic_ai_skills import SkillsToolset

# Skills 目录路径
SKILLS_DIR = Path(__file__).parent


def create_layout_skills_toolset(
    directories: Optional[list[str]] = None
) -> SkillsToolset:
    """创建布局技能工具集
    
    从文件系统加载所有 SKILL.md 定义的技能，支持：
    - 渐进式披露：按需加载技能详细信息
    - 统一接口：通过 load_skill/run_skill_script 调用
    - Token优化：初始只加载技能列表，减少prompt大小
    
    Args:
        directories: 技能目录列表，默认使用本模块所在目录
        
    Returns:
        SkillsToolset: 包含所有布局技能的工具集
        
    Example:
        >>> from analog_layout_agent.skills import create_layout_skills_toolset
        >>> skills_toolset = create_layout_skills_toolset()
        >>> agent = Agent(model, toolsets=[skills_toolset])
    """
    if directories is None:
        directories = [str(SKILLS_DIR)]
    
    return SkillsToolset(directories=directories)


def get_skill_directories() -> list[str]:
    """获取所有技能目录路径
    
    Returns:
        技能目录路径列表
    """
    skill_dirs = []
    for item in SKILLS_DIR.iterdir():
        if item.is_dir() and (item / "SKILL.md").exists():
            skill_dirs.append(str(item))
    return skill_dirs


def list_available_skills() -> list[dict]:
    """列出所有可用技能
    
    Returns:
        技能信息列表，每项包含 name、description、path
    """
    import yaml
    
    skills = []
    for item in SKILLS_DIR.iterdir():
        skill_md = item / "SKILL.md"
        if item.is_dir() and skill_md.exists():
            try:
                content = skill_md.read_text(encoding='utf-8')
                # 解析 YAML frontmatter
                if content.startswith('---'):
                    end = content.find('---', 3)
                    if end > 0:
                        frontmatter = yaml.safe_load(content[3:end])
                        skills.append({
                            "name": frontmatter.get("name", item.name),
                            "description": frontmatter.get("description", ""),
                            "path": str(item)
                        })
            except Exception:
                # 跳过解析失败的技能
                pass
    return skills


__all__ = [
    'create_layout_skills_toolset',
    'get_skill_directories',
    'list_available_skills',
    'SkillsToolset',
    'SKILLS_DIR',
]
