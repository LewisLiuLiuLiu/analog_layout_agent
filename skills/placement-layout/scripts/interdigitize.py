#!/usr/bin/env python3
"""
互指式放置脚本

用法:
    python interdigitize.py --comp-a M_ref --comp-b M_out --num-cols 4 --layout-style ABAB
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description='互指式放置两个组件')
    parser.add_argument('--comp-a', type=str, required=True, help='组件A名称')
    parser.add_argument('--comp-b', type=str, required=True, help='组件B名称')
    parser.add_argument('--num-cols', type=int, default=4, help='列数')
    parser.add_argument('--layout-style', type=str, default='ABAB',
                        choices=['ABAB', 'ABBA', 'common_centroid'], help='布局风格')
    
    args = parser.parse_args()
    
    params = {
        "comp_a": args.comp_a,
        "comp_b": args.comp_b,
        "num_cols": args.num_cols,
        "layout_style": args.layout_style
    }
    
    try:
        from analog_layout_agent.mcp_server.server import get_server
        
        server = get_server()
        if server is None:
            print(json.dumps({"error": "MCP Server 未初始化"}, ensure_ascii=False))
            return 1
        
        result = server.call_tool("interdigitize", params)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
