#!/usr/bin/env python3
"""
获取DRC修复建议脚本

用法:
    python get_drc_fix_suggestions.py
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    try:
        from analog_layout_agent.mcp_server.server import get_server
        from analog_layout_agent.core.verification import VerificationEngine
        from analog_layout_agent.core.drc_advisor import analyze_drc_result
        
        server = get_server()
        if server is None:
            print(json.dumps({"error": "MCP Server 未初始化"}, ensure_ascii=False))
            return 1
        
        layout_ctx = server.state_handler.get_context()
        if layout_ctx is None:
            print(json.dumps({"error": "布局上下文未初始化"}, ensure_ascii=False))
            return 1
        
        engine = VerificationEngine(layout_ctx)
        drc_result = engine.run_drc()
        
        pdk_name = layout_ctx.pdk_name if layout_ctx else "sky130"
        analysis = analyze_drc_result(drc_result, pdk_name)
        
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
