你现在的角色是：负责底层 Skill 库实现的后端工程师 + 版图工程师。

【输入上下文】
- 我会提供《opamp_layout_gen_Agent_proposal.md》全文。
- proposal 中已经列出了首批技能模块（如 `parse_netlist`、`create_nmos_pcell`、`create_common_centroid_pair`、`run_drc_check` 等），并说明了大致功能和优先级。
- 你的任务是：**为每个 Skill 写出工程级别的 API 规格与实现说明章节**，形成“技能模块实现”部分。

【输出目标】
输出一章（或多章）文档，内容足以让工程师根据文档实现并测试这些 Skill 模块，供 Agent 通过 MCP/工具调用。

【输出内容结构要求】
建议组织方式：
1. 技能模块总体设计
   - Skill 在系统中的角色：被 Agent 作为 MCP 工具调用，负责确定性操作。
   - 命名规范：推荐的工具名前缀（如 `netlist.*`、`layout.*`、`drc.*`、`eval.*`）。

2. 通用约定
   - 所有 Skill 的通用输入/输出包装结构：
     - 例如：`{ "ok": bool, "error": {code, message} | null, "data": {...} }`
   - 错误处理约定：
     - 参数错误、外部工具失败（如 KLayout 不可用）、DRC 规则缺失等典型错误码定义。

3. 单个 Skill 的详细规格（为每个 Skill 写一个小节）
   对于 proposal 中列出的每个 Skill（至少包括 P0/P1 项），请按以下模板展开：

   3.x `<skill_name>` 技能规格（例如 `create_common_centroid_pair`）
   - **功能描述**：用 1–2 段描述该 Skill 完成的版图操作/逻辑。
   - **调用方式**：
     - MCP 工具名（例如 `layout.create_common_centroid_pair`）。
     - 推荐的编程接口签名（例如 Python 函数签名）。
   - **输入参数定义**：
     - 使用表格列出每个字段：`字段名 / 类型 / 单位 / 是否必填 / 含义 / 约束（如必须为偶数、取值范围）`。
   - **输出数据结构**：
     - 描述返回的 `data` 字段结构，例如包含 `cell_name`、`device_instances`、`anchors` 等。
     - 说明坐标系、单位（例如 KLayout DBU）。
   - **内部实现逻辑概述**：
     - 以步骤形式描述：如何解析参数 → 如何调用 KLayout API 或 GDSFactory → 如何检查基本 DRC → 如何构造返回值。
     - 不写具体代码，但写出关键算法选择（例如布线采用 A* 还是简单曼哈顿搜索）。
   - **错误场景与错误码**：
     - 列出典型失败场景（参数非法、DRC 无法满足、KLayout 调用失败等）以及对应的错误码、错误信息模板。
   - **测试建议**：
     - 指定若干最小测试用例（输入参数组合）和预期几何/逻辑结果。

4. Skill Registry 与版本管理
   - 说明如何在系统中注册 Skill（Skill 名 → 实现函数 → 版本号）。
   - 说明如何在升级 Skill 时保持向后兼容（例如保留旧版本工具名或通过 `version` 字段控制）。

【技术深度要求】
- 需要写到“**足以生成接口文档 + 开发任务拆解**”的粒度：
  - 每个 Skill 必须有清晰的参数列表和返回结构。
  - 错误处理必须有清晰的分类和错误码建议。
- 不强制规定使用何种语言实现 Skill，但可以假设主实现语言为 Python + KLayout pya API，并在文档中说明这一假设。

【与整体项目关联性要求】
- Skill 规格必须与 MVP 场景和架构章节保持一致：例如 `create_common_centroid_pair` 要能支撑单级差分运放的核心布局。
- Skill 的命名和行为要能直接被 Agent 规划模块使用（与规划章节中的 Plan 步骤名称保持可映射性）。