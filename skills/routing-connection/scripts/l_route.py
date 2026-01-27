#!/usr/bin/env python3
"""
L形布线脚本

用法:
    python l_route.py --source M1.drain_N --target R1.port_a
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description='L形布线')
    parser.add_argument('--source', type=str, required=True, help='源端口')
    parser.add_argument('--target', type=str, required=True, help='目标端口')
    parser.add_argument('--layer', type=str, default='met1', help='布线层')
    parser.add_argument('--width', type=float, default=None, help='布线宽度(μm)')
    
    args = parser.parse_args()
    
    params = {
        "source_port": args.source,
        "target_port": args.target,
        "layer": args.layer
    }
    if args.width is not None:
        params["width"] = args.width
    
    try:
        from analog_layout_agent.mcp_server.server import get_server
        
        server = get_server()
        if server is None:
            print(json.dumps({"error": "MCP Server 未初始化"}, ensure_ascii=False))
            return 1
        
        result = server.call_tool("l_route", params)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
