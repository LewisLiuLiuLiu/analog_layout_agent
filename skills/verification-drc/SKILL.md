---
name: verification-drc
description: 设计规则验证：执行DRC检查、提取网表、获取修复建议等，确保版图符合工艺要求。
---

# 验证检查技能 (Verification DRC Skill)

## 何时使用此技能

当你需要验证版图质量时使用此技能：

- **run_drc**: 执行设计规则检查
- **extract_netlist**: 提取版图网表
- **get_drc_fix_suggestions**: 获取DRC违规的修复建议

## 使用流程

1. 完成版图设计（器件创建、放置、布线）
2. 运行 run_drc 检查设计规则违规
3. 如有违规，使用 get_drc_fix_suggestions 获取修复建议
4. 根据建议修改版图
5. 重复检查直到通过
6. 使用 extract_netlist 验证连接正确性

## 可用脚本

| 脚本名 | 说明 | 参数 |
|--------|------|------|
| `run_drc` | 执行DRC检查 | 无 |
| `extract_netlist` | 提取版图网表 | 无 |
| `get_drc_fix_suggestions` | 获取DRC修复建议 | 无 |

## DRC检查内容

### 常见检查项

| 检查类型 | 说明 |
|----------|------|
| 最小宽度 | 各层的最小宽度要求 |
| 最小间距 | 相邻图形的最小距离 |
| 最小包围 | 上层对下层的最小包围 |
| 最小面积 | 图形的最小面积要求 |
| 密度检查 | 金属密度要求 |

### 违规严重级别

| 级别 | 说明 | 处理方式 |
|------|------|----------|
| ERROR | 严重违规，必须修复 | 无法制造 |
| WARNING | 潜在问题 | 建议修复 |
| INFO | 提示信息 | 可忽略 |

## 使用示例

### 运行DRC检查

```
run_skill_script(
    skill_name="verification-drc",
    script_name="run_drc",
    args=[]
)
```

返回示例：
```json
{
  "success": true,
  "total_violations": 2,
  "violations": [
    {
      "type": "MIN_WIDTH",
      "layer": "met1",
      "severity": "ERROR",
      "location": [10.5, 20.3],
      "message": "Metal1 宽度 0.12μm 小于最小要求 0.14μm"
    }
  ]
}
```

### 获取修复建议

```
run_skill_script(
    skill_name="verification-drc",
    script_name="get_drc_fix_suggestions",
    args=[]
)
```

返回示例：
```json
{
  "violations_count": 2,
  "suggestions": [
    {
      "violation_type": "MIN_WIDTH",
      "action": "increase_width",
      "target": "met1 wire at (10.5, 20.3)",
      "current_value": 0.12,
      "suggested_value": 0.14,
      "description": "将布线宽度从 0.12μm 增加到 0.14μm"
    }
  ]
}
```

### 提取网表

```
run_skill_script(
    skill_name="verification-drc",
    script_name="extract_netlist",
    args=[]
)
```

返回示例：
```json
{
  "success": true,
  "devices": [
    {"name": "M1", "type": "nmos", "terminals": {"D": "net1", "G": "inp", "S": "vss"}},
    {"name": "M2", "type": "nmos", "terminals": {"D": "net1", "G": "inn", "S": "vss"}}
  ],
  "nets": ["inp", "inn", "net1", "vss"]
}
```

## 常见DRC违规及修复

| 违规类型 | 常见原因 | 修复方法 |
|----------|----------|----------|
| MIN_WIDTH | 布线宽度不足 | 增加 --width 参数 |
| MIN_SPACING | 器件/布线过近 | 增加放置间距 |
| MIN_ENCLOSURE | Via包围不足 | 检查Via与金属的重叠 |
| MIN_AREA | 图形过小 | 增加器件尺寸 |

## 验证流程建议

1. **分阶段检查**
   - 每完成一个子模块后检查
   - 避免最后一次性检查大量违规

2. **优先级处理**
   - 先修复 ERROR 级别
   - 再处理 WARNING 级别

3. **迭代修复**
   - 每次修复后重新检查
   - 修复可能引入新违规

## 注意事项

- DRC 检查依赖当前 PDK 的规则定义
- 复杂版图检查可能需要较长时间
- 网表提取前确保所有连接已完成
- 某些警告在特定情况下可以忽略

## 相关资源

- `references/drc_rules_sky130.md` - Sky130 DRC规则详解
- `references/common_violations.md` - 常见违规案例分析
