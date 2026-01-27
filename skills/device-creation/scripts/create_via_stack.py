#!/usr/bin/env python3
"""
创建 Via 堆叠脚本

用法:
    python create_via_stack.py --from-layer met1 --to-layer met3
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description='创建 Via 堆叠')
    parser.add_argument('--from-layer', type=str, required=True, 
                        choices=['poly', 'met1', 'met2', 'met3', 'met4', 'met5'],
                        help='起始层')
    parser.add_argument('--to-layer', type=str, required=True,
                        choices=['met1', 'met2', 'met3', 'met4', 'met5'],
                        help='目标层')
    parser.add_argument('--size', type=float, nargs=2, default=None, 
                        metavar=('WIDTH', 'HEIGHT'), help='Via尺寸(μm)')
    parser.add_argument('--name', type=str, default=None, help='组件名称')
    
    args = parser.parse_args()
    
    params = {
        "from_layer": args.from_layer,
        "to_layer": args.to_layer
    }
    if args.size is not None:
        params["size"] = args.size
    if args.name is not None:
        params["name"] = args.name
    
    try:
        from analog_layout_agent.mcp_server.server import get_server
        
        server = get_server()
        if server is None:
            print(json.dumps({"error": "MCP Server 未初始化"}, ensure_ascii=False))
            return 1
        
        result = server.call_tool("create_via_stack", params)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
