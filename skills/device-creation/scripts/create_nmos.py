#!/usr/bin/env python3
"""
创建 NMOS 晶体管脚本

用法:
    python create_nmos.py --width 3.0 --fingers 4 --with-dummy
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目路径
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description='创建 NMOS 晶体管')
    parser.add_argument('--width', type=float, required=True, help='沟道宽度(μm)')
    parser.add_argument('--length', type=float, default=None, help='沟道长度(μm)，默认使用PDK最小长度')
    parser.add_argument('--fingers', type=int, default=1, help='指数（栅极数量）')
    parser.add_argument('--multiplier', type=int, default=1, help='并联倍数')
    parser.add_argument('--with-dummy', action='store_true', help='添加dummy结构')
    parser.add_argument('--with-tie', action='store_true', default=True, help='添加衬底连接')
    parser.add_argument('--name', type=str, default=None, help='组件名称')
    
    args = parser.parse_args()
    
    # 构建参数
    params = {
        "width": args.width,
        "fingers": args.fingers,
        "multiplier": args.multiplier,
        "with_dummy": args.with_dummy,
        "with_tie": args.with_tie,
    }
    if args.length is not None:
        params["length"] = args.length
    if args.name is not None:
        params["name"] = args.name
    
    try:
        from analog_layout_agent.mcp_server.server import get_server
        
        server = get_server()
        if server is None:
            print(json.dumps({"error": "MCP Server 未初始化，请先调用 initialize"}, ensure_ascii=False))
            return 1
        
        result = server.call_tool("create_nmos", params)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
