"""
Layout Agent Skills 模块
Layout Agent Skills Module

使用 PydanticAI Skills 框架从文件系统加载技能，实现渐进式披露。
Uses PydanticAI Skills framework to load skills from file system, implementing progressive disclosure.

遵循 Anthropic Agent Skills 标准目录结构：
Follows Anthropic Agent Skills standard directory structure:

skills/
├── device-creation/
│   ├── SKILL.md          # 必需：技能指令 + YAML元数据 / Required: skill instructions + YAML metadata
│   ├── scripts/          # 可执行脚本 / Executable scripts
│   ├── references/       # 参考文档 / Reference documents
│   └── assets/           # 资源文件 / Asset files
├── routing-connection/
│   └── ...
└── ...

技能列表 / Skill list:
- device-creation: 基础器件创建（NMOS/PMOS/电容/电阻/Via） / Basic device creation
- routing-connection: 智能布线连接（smart_route/c_route/l_route等） / Smart routing connection
- placement-layout: 组件放置与布局（place/move/align/interdigitize） / Component placement and layout
- circuit-building: 复合电路构建（电流镜/差分对） / Composite circuit building
- verification-drc: 设计规则验证（DRC/LVS/网表提取） / Design rule verification
- export-query: 导出与查询（GDS导出/组件查询） / Export and query
"""

from pathlib import Path
from typing import Optional

from pydantic_ai_skills import SkillsToolset

# Skills 目录路径 / Skills directory path
SKILLS_DIR = Path(__file__).parent


def create_layout_skills_toolset(
    directories: Optional[list[str]] = None
) -> SkillsToolset:
    """创建布局技能工具集 / Create layout skills toolset
    
    从文件系统加载所有 SKILL.md 定义的技能，支持：
    Loads all skills defined in SKILL.md from file system, supporting:
    - 渐进式披露：按需加载技能详细信息 / Progressive disclosure: load skill details on demand
    - 统一接口：通过 load_skill/run_skill_script 调用 / Unified interface: call via load_skill/run_skill_script
    - Token优化：初始只加载技能列表，减少prompt大小 / Token optimization: initially only load skill list
    
    Args:
        directories: 技能目录列表，默认使用本模块所在目录 / Skill directory list, defaults to this module's directory
        
    Returns:
        SkillsToolset: 包含所有布局技能的工具集 / Toolset containing all layout skills
        
    Example:
        >>> from analog_layout_agent.skills import create_layout_skills_toolset
        >>> skills_toolset = create_layout_skills_toolset()
        >>> agent = Agent(model, toolsets=[skills_toolset])
    """
    if directories is None:
        directories = [str(SKILLS_DIR)]
    
    return SkillsToolset(directories=directories)


def get_skill_directories() -> list[str]:
    """获取所有技能目录路径 / Get all skill directory paths
    
    Returns:
        技能目录路径列表 / List of skill directory paths
    """
    skill_dirs = []
    for item in SKILLS_DIR.iterdir():
        if item.is_dir() and (item / "SKILL.md").exists():
            skill_dirs.append(str(item))
    return skill_dirs


def list_available_skills() -> list[dict]:
    """列出所有可用技能 / List all available skills
    
    Returns:
        技能信息列表，每项包含 name、description、path
        List of skill info, each containing name, description, path
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
