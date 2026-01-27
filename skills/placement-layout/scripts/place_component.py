#!/usr/bin/env python3
"""
放置组件脚本

用法:
    python place_component.py --name M1 --x 10.0 --y 20.0 --rotation 0
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description='放置组件到指定位置')
    parser.add_argument('--name', type=str, required=True, help='组件名称')
    parser.add_argument('--x', type=float, required=True, help='X坐标(μm)')
    parser.add_argument('--y', type=float, required=True, help='Y坐标(μm)')
    parser.add_argument('--rotation', type=int, default=0, choices=[0, 90, 180, 270], help='旋转角度')
    
    args = parser.parse_args()
    
    params = {
        "component_name": args.name,
        "x": args.x,
        "y": args.y,
        "rotation": args.rotation
    }
    
    try:
        from analog_layout_agent.mcp_server.server import get_server
        
        server = get_server()
        if server is None:
            print(json.dumps({"error": "MCP Server 未初始化"}, ensure_ascii=False))
            return 1
        
        result = server.call_tool("place_component", params)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
