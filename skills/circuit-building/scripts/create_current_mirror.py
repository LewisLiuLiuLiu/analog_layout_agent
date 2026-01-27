#!/usr/bin/env python3
"""
创建电流镜脚本

用法:
    python create_current_mirror.py --width 2.0 --ratio 4 --type nmos --name CM1
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description='创建电流镜电路')
    parser.add_argument('--width', type=float, required=True, help='单管宽度(μm)')
    parser.add_argument('--ratio', type=int, required=True, help='电流比(输出:输入)')
    parser.add_argument('--length', type=float, default=None, help='沟道长度(μm)')
    parser.add_argument('--fingers', type=int, default=2, help='指数')
    parser.add_argument('--type', type=str, default='nmos', choices=['nmos', 'pmos'], help='器件类型')
    parser.add_argument('--name', type=str, default=None, help='电路名称')
    
    args = parser.parse_args()
    
    params = {
        "width": args.width,
        "ratio": args.ratio,
        "fingers": args.fingers,
        "device_type": args.type
    }
    if args.length is not None:
        params["length"] = args.length
    if args.name is not None:
        params["name"] = args.name
    
    try:
        from analog_layout_agent.mcp_server.server import get_server
        
        server = get_server()
        if server is None:
            print(json.dumps({"error": "MCP Server 未初始化"}, ensure_ascii=False))
            return 1
        
        result = server.call_tool("create_current_mirror", params)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
