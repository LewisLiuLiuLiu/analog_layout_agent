#!/usr/bin/env python3
"""
列出组件脚本

用法:
    python list_components.py
    python list_components.py --type nmos
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description='列出所有组件')
    parser.add_argument('--type', type=str, default=None, help='按类型过滤')
    
    args = parser.parse_args()
    
    params = {}
    if args.type is not None:
        params["device_type"] = args.type
    
    try:
        from analog_layout_agent.mcp_server.server import get_server
        
        server = get_server()
        if server is None:
            print(json.dumps({"error": "MCP Server 未初始化"}, ensure_ascii=False))
            return 1
        
        result = server.call_tool("list_components", params)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
