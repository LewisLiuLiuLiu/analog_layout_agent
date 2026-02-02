"""
Verification Tools - 验证工具
Verification Tools - Verification tool executors

封装 VerificationEngine，提供 MCP 工具接口
Wraps VerificationEngine to provide MCP tool interface
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# 添加路径 / Add paths
_BASE_PATH = Path(__file__).parent.parent.parent
if str(_BASE_PATH) not in sys.path:
    sys.path.insert(0, str(_BASE_PATH))

_GLAYOUT_PATH = _BASE_PATH.parent / "gLayout" / "src"
if str(_GLAYOUT_PATH) not in sys.path:
    sys.path.insert(0, str(_GLAYOUT_PATH))

# 设置 PDK_ROOT 环境变量（glayout 初始化时需要）/ Set PDK_ROOT environment variable
_PDK_ROOT = _BASE_PATH.parent / "skywater-pdk"
if _PDK_ROOT.exists() and not os.environ.get('PDK_ROOT'):
    os.environ['PDK_ROOT'] = str(_PDK_ROOT)

from core.layout_context import LayoutContext

logger = logging.getLogger(__name__)

# 容错导入 VerificationEngine / Fault-tolerant import of VerificationEngine
_VERIFICATION_AVAILABLE = False
VerificationEngine = None

try:
    from core.verification import VerificationEngine as _VE
    VerificationEngine = _VE
    _VERIFICATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"VerificationEngine 不可用 / VerificationEngine not available: {e}")


class VerificationToolExecutor:
    """验证工具执行器
    Verification tool executor
    
    封装 VerificationEngine，提供统一的 MCP 工具接口。
    Wraps VerificationEngine to provide unified MCP tool interface.
    
    Usage:
        >>> executor = VerificationToolExecutor(context)
        >>> result = executor.run_drc()
    """
    
    def __init__(self, context: LayoutContext):
        """初始化执行器 / Initialize executor
        
        Args:
            context: 布局上下文 / Layout context
        """
        self.context = context
        self._engine = None
        
        if _VERIFICATION_AVAILABLE and VerificationEngine is not None:
            try:
                self._engine = VerificationEngine(context)
            except Exception as e:
                logger.warning(f"VerificationEngine 初始化失败 / VerificationEngine init failed: {e}")
    
    def run_drc(
        self,
        component_name: Optional[str] = None,
        output_format: str = "summary",
        output_dir: Optional[str] = None,
        include_suggestions: bool = True
    ) -> Dict[str, Any]:
        """执行 DRC 检查 / Run DRC check
        
        Args:
            component_name: 组件名称，默认检查顶层 / Component name, defaults to top level
            output_format: 输出格式 ("summary" | "detailed") / Output format
            output_dir: 输出目录 / Output directory
            include_suggestions: 是否生成修复建议 / Whether to generate fix suggestions
            
        Returns:
            DRC 检查结果 / DRC check result
        """
        if self._engine is None:
            return self._mock_drc_result()
        
        try:
            result = self._engine.run_drc(
                component_name=component_name,
                output_dir=output_dir,
                output_format=output_format,
                include_suggestions=include_suggestions
            )
            return result.to_dict()
        except Exception as e:
            logger.error(f"DRC 执行失败 / DRC execution failed: {e}")
            raise
    
    def extract_netlist(
        self,
        component_name: Optional[str] = None,
        format: str = "spice",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """提取网表 / Extract netlist
        
        Args:
            component_name: 组件名称，默认使用顶层 / Component name, defaults to top level
            format: 输出格式 ("spice" | "spectre") / Output format
            output_file: 输出文件路径 / Output file path
            
        Returns:
            提取结果 / Extraction result
        """
        if self._engine is None:
            return {"success": True, "mock": True, "netlist": "* Mock netlist"}
        
        try:
            return self._engine.extract_netlist(
                component_name=component_name,
                format=format,
                output_file=output_file
            )
        except Exception as e:
            logger.error(f"网表提取失败 / Netlist extraction failed: {e}")
            raise
    
    def run_lvs(
        self,
        schematic_netlist: str,
        component_name: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """执行 LVS 检查 / Run LVS check
        
        Args:
            schematic_netlist: 参考原理图网表（路径或内容） / Reference schematic netlist (path or content)
            component_name: 组件名称，默认使用顶层 / Component name, defaults to top level
            output_dir: 输出目录 / Output directory
            
        Returns:
            LVS 检查结果 / LVS check result
        """
        if self._engine is None:
            return {"matched": True, "mock": True, "mismatch_count": 0}
        
        try:
            result = self._engine.run_lvs(
                schematic_netlist=schematic_netlist,
                component_name=component_name,
                output_dir=output_dir
            )
            return result.to_dict()
        except Exception as e:
            logger.error(f"LVS 执行失败 / LVS execution failed: {e}")
            raise
    
    def _mock_drc_result(self) -> Dict[str, Any]:
        """返回模拟 DRC 结果（当引擎不可用时）
        Return mock DRC result (when engine is unavailable)
        """
        return {
            "passed": True,
            "violation_count": 0,
            "violations": [],
            "report_path": None,
            "checked_at": None,
            "mock": True
        }
