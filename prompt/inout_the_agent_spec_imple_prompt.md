你现在的角色是：负责输入/输出数据层实现的后端工程师。

【输入上下文】
- 我会提供《opamp_layout_gen_Agent_proposal.md》全文。
- proposal 的 2.2 和 2.2.4 已经给出了网表 JSON、DRC 规则 JSON、设计目标 JSON 和 MCP 调用的示例。
- 你的任务是：**写出“输入输出处理”章节，实现 JSON 解析与内部数据结构定义的工程规格**。

【输出目标】
形成一章详细定义所有核心输入/输出格式以及相应解析器的文档，使工程师可以实现这些解析组件，并与 Agent 其他模块对接。

【输出内容结构要求】
建议结构如下：
1. 数据模型总览
   - 列出系统中核心数据结构：
     - `Circuit`、`Device`、`Net`、`DRCRuleSet`、`DesignObjectives`、`ToolCallRequest`、`ToolCallResult` 等。
   - 说明这些数据结构在系统中的位置：由哪些解析器生成，被哪些模块消费。

2. 网表 JSON 解析器规格
   - 给出正式的网表 JSON Schema（可以用文字 + 表格说明）：
     - 字段、类型、必选/可选、单位（如 w/l）、约束条件。
   - 说明解析流程：
     - 读取 JSON → 校验 Schema → 构建 `Circuit` 对象（包含 devices、nets、拓扑信息）。
   - 描述内部 `Circuit` / `Device` / `Net` 数据结构字段定义。

3. DRC 规则 JSON 解析器规格
   - 给出 DRC JSON Schema：
     - `tech`、`layers`、`rules[]` 中各字段含义与类型。
   - 定义内部 `DRCRule` / `DRCRuleSet` 数据结构。
   - 说明解析流程与错误处理逻辑（例如缺少层映射、类型不支持时的行为）。

4. 设计目标 JSON 解析器规格
   - 定义 `objectives`、`constraints` 的字段与可选值（例如合理的目标名列表）。
   - 定义内部 `DesignObjectives` / `DesignConstraints` 数据模型。
   - 说明如何将自然语言目标映射为评估函数（可以作为“推荐映射表”写入文档）。

5. MCP 工具调用输入/输出格式
   - 明确定义 MCP 调用统一结构：
     - `name`、`input`、`trace_id` 等字段。
   - 明确工具返回结构（`ok`、`error`、`data`），给出错误码设计建议。
   - 说明如何在 Orchestrator 中封装/解析这些结构。

6. 版本管理与兼容性
   - 为所有关键 JSON 定义 `schema_version` 字段。
   - 说明未来 schema 升级的兼容策略。

【技术深度要求】
- 要写到“**可以直接据此写类型定义/DTO 和解析代码**”的程度：
  - 每个字段的类型、单位、约束都要清晰。
  - 对非法输入的处理必须有明确定义（抛错/默认值/忽略）。

【与整体项目关联性要求】
- 所有数据结构命名和语义要与架构章节中的描述一致。
- 要考虑到后续测试与 RAG/Skill 的使用场景（例如 `Circuit` 对象既供规划模块使用，也可被测试代码读取）。