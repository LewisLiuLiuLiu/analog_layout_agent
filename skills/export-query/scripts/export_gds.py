#!/usr/bin/env python3
"""
导出GDS文件脚本

用法:
    python export_gds.py --output ./output/my_design.gds
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description='导出GDS文件')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--top-cell', type=str, default=None, help='顶层单元名')
    
    args = parser.parse_args()
    
    params = {"output_path": args.output}
    if args.top_cell is not None:
        params["top_cell"] = args.top_cell
    
    try:
        from analog_layout_agent.mcp_server.server import get_server
        
        server = get_server()
        if server is None:
            print(json.dumps({"error": "MCP Server 未初始化"}, ensure_ascii=False))
            return 1
        
        result = server.call_tool("export_gds", params)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
