#!/usr/bin/env python3
"""
创建差分对脚本

用法:
    python create_diff_pair.py --width 10.0 --fingers 4 --tail-width 20.0 --name DP1
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description='创建差分对电路')
    parser.add_argument('--width', type=float, required=True, help='输入管宽度(μm)')
    parser.add_argument('--length', type=float, default=None, help='沟道长度(μm)')
    parser.add_argument('--fingers', type=int, default=4, help='指数')
    parser.add_argument('--tail-width', type=float, default=None, help='尾电流管宽度(μm)')
    parser.add_argument('--name', type=str, default=None, help='电路名称')
    
    args = parser.parse_args()
    
    params = {
        "width": args.width,
        "fingers": args.fingers
    }
    if args.length is not None:
        params["length"] = args.length
    if args.tail_width is not None:
        params["tail_width"] = args.tail_width
    if args.name is not None:
        params["name"] = args.name
    
    try:
        from analog_layout_agent.mcp_server.server import get_server
        
        server = get_server()
        if server is None:
            print(json.dumps({"error": "MCP Server 未初始化"}, ensure_ascii=False))
            return 1
        
        result = server.call_tool("create_diff_pair", params)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
