#!/usr/bin/env python3
"""
对齐到端口脚本

用法:
    python align_to_port.py --name M2 --target-port M1.gate_E --alignment center
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description='将组件对齐到端口')
    parser.add_argument('--name', type=str, required=True, help='组件名称')
    parser.add_argument('--target-port', type=str, required=True, help='目标端口(组件名.端口名)')
    parser.add_argument('--alignment', type=str, default='center', 
                        choices=['center', 'left', 'right', 'top', 'bottom'], help='对齐方式')
    parser.add_argument('--offset-x', type=float, default=0.0, help='X偏移(μm)')
    parser.add_argument('--offset-y', type=float, default=0.0, help='Y偏移(μm)')
    
    args = parser.parse_args()
    
    params = {
        "component_name": args.name,
        "target_port": args.target_port,
        "alignment": args.alignment,
        "offset_x": args.offset_x,
        "offset_y": args.offset_y
    }
    
    try:
        from analog_layout_agent.mcp_server.server import get_server
        
        server = get_server()
        if server is None:
            print(json.dumps({"error": "MCP Server 未初始化"}, ensure_ascii=False))
            return 1
        
        result = server.call_tool("align_to_port", params)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
