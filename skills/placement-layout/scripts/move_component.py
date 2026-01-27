#!/usr/bin/env python3
"""
移动组件脚本

用法:
    python move_component.py --name M1 --dx 5.0 --dy 0.0
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description='移动组件')
    parser.add_argument('--name', type=str, required=True, help='组件名称')
    parser.add_argument('--dx', type=float, required=True, help='X方向偏移(μm)')
    parser.add_argument('--dy', type=float, required=True, help='Y方向偏移(μm)')
    
    args = parser.parse_args()
    
    params = {
        "component_name": args.name,
        "dx": args.dx,
        "dy": args.dy
    }
    
    try:
        from analog_layout_agent.mcp_server.server import get_server
        
        server = get_server()
        if server is None:
            print(json.dumps({"error": "MCP Server 未初始化"}, ensure_ascii=False))
            return 1
        
        result = server.call_tool("move_component", params)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
