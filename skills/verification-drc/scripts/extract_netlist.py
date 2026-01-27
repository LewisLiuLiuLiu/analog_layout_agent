#!/usr/bin/env python3
"""
提取网表脚本

用法:
    python extract_netlist.py
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
        
        server = get_server()
        if server is None:
            print(json.dumps({"error": "MCP Server 未初始化"}, ensure_ascii=False))
            return 1
        
        layout_ctx = server.state_handler.get_context()
        if layout_ctx is None:
            print(json.dumps({"error": "布局上下文未初始化"}, ensure_ascii=False))
            return 1
        
        engine = VerificationEngine(layout_ctx)
        result = engine.extract_netlist()
        
        if hasattr(result, 'to_dict'):
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
