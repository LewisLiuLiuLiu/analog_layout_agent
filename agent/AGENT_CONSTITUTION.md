# Layout Agent 宪法 (Constitution)

**版本**: 1.0
**生效日期**: 2026-01-29
**强制级别**: 最高 - 任何违反都将导致任务失败

---

## 第一条：执行流程规范

每次 Execution Agent 运行时，必须严格按照以下顺序执行：

### 1.1 初始化 (强制第一步)

```bash
./init.sh
```

必须等待 init.sh 执行完成后才能进行后续操作。

### 1.2 获取进度 (强制第二步 - 仅 Execution Agent)

读取 `progress.md` 文件的最后 50 行，了解最新进展：

```bash
tail -n 50 progress.md
```

**例外情况**: 首次运行时（Reasoning Agent 规划阶段），此步骤跳过，因为 progress.md 尚不存在。

### 1.3 确定当前任务 (强制第三步)

读取 `workflow_state.json`，找到 `completed` 数组中第一个值为 `false` 的步骤，这是当前需要执行的任务。

**禁止**: 跳过任何 `false` 状态的步骤
**禁止**: 执行已经为 `true` 的步骤

---

## 第二条：状态管理规范

### 2.1 状态值限制

`workflow_state.json` 中的 `completed` 数组只允许两种值：

| 值 | 含义 |
|---|------|
| `false` | 步骤未完成 |
| `true` | 步骤已完成并验证通过 |

**绝对禁止的值**: 
- `"pending"`
- `"in_progress"` 
- `"failed"`
- `"retry"`
- `null`
- 任何其他值

### 2.2 状态修改规则

1. 只有在步骤**执行成功且验证通过**后，才能将 `false` 改为 `true`
2. 一旦设为 `true`，**永远不能改回 `false`**
3. 除了 `completed` 数组，**禁止修改** `workflow_state.json` 的任何其他内容

### 2.3 JSON 结构保护

`workflow_state.json` 中以下字段**禁止修改**：
- `design_name`
- `pdk`
- `steps` (步骤定义)
- `created_at`

只允许修改：
- `completed` 数组中的值 (仅 false -> true)
- `updated_at` 时间戳

---

## 第三条：验证规范

### 3.1 执行后必须验证

每个步骤执行后，必须进行验证。验证失败时：
1. 不修改 `completed` 状态
2. 记录失败原因到 `progress.md`
3. 请求 Reasoning Agent 分析

### 3.2 验证类型

| 步骤类型 | 验证方法 |
|---------|---------|
| device-creation | 检查组件是否存在于 ComponentRegistry |
| placement-layout | 检查放置位置和对齐 |
| routing-connection | 检查连接是否建立 |
| verification-drc | 检查 DRC 结果是否 clean |

**注**: LVS 验证暂不启用。

---

## 第四条：金属层规范 (防止短路)

### 4.1 强制指定金属层

所有 routing 步骤必须显式指定 `layer` 参数，不允许省略。

### 4.2 金属层分配原则

1. **水平信号**: 使用 met1
2. **垂直信号**: 使用 met2
3. **交叉信号**: 必须使用不同金属层
4. **电源/地线**: 使用 met3 或更高层

### 4.3 禁止行为

- 不指定 layer 参数
- 同层交叉走线
- 信号线与电源线同层

---

## 第五条：日志规范

所有操作必须记录日志，包括：
1. 执行的每个步骤
2. 调用的每个工具及参数
3. 工具返回结果
4. 验证结果
5. 任何错误和异常

---

## 第六条：禁止行为

以下行为**绝对禁止**：

1. 跳过 init.sh 初始化
2. 不查看 progress.md 就开始执行
3. 执行非第一个 false 的步骤
4. 将 completed 改为非 true/false 的值
5. 修改步骤定义 (steps)
6. 在验证失败时将状态改为 true
7. 删除或重排 completed 数组的元素

---

## 违规处理

任何违反本宪法的行为将导致：
1. 立即停止当前执行
2. 回滚到上一个已知良好状态
3. 记录违规行为到日志
4. 请求人工干预
