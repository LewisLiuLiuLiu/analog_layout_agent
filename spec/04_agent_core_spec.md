# 第4章 Agent 核心逻辑实现

---

## 4.1 Agent 主循环设计

### 4.1.1 状态机概述

Agent Orchestrator 的核心是一个状态驱动的主循环，管理从输入接收到最终输出的完整生命周期。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Agent 状态机                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    ┌─────────┐                                                          │
│    │  INIT   │  接收输入，初始化会话                                      │
│    └────┬────┘                                                          │
│         │                                                               │
│         ▼                                                               │
│    ┌─────────┐                                                          │
│    │ PARSING │  解析网表、DRC、目标                                       │
│    └────┬────┘                                                          │
│         │                                                               │
│         ▼                                                               │
│    ┌─────────┐                                                          │
│    │PLANNING │  生成初始规划                                             │
│    └────┬────┘                                                          │
│         │                                                               │
│         ▼                                                               │
│    ┌─────────┐     ┌──────────┐                                         │
│    │EXECUTING│────►│STEP_EXEC │◄───┐  执行单个步骤                        │
│    └────┬────┘     └────┬─────┘    │                                    │
│         │               │          │                                    │
│         │               ▼          │                                    │
│         │          ┌──────────┐    │                                    │
│         │          │STEP_DONE │────┘  步骤完成，继续下一步                 │
│         │          └────┬─────┘                                         │
│         │               │                                               │
│         │               ▼                                               │
│         │          ┌──────────┐                                         │
│         │          │PLAN_DONE │  所有步骤完成                             │
│         │          └────┬─────┘                                         │
│         │               │                                               │
│         ▼               ▼                                               │
│    ┌─────────┐                                                          │
│    │EVALUATING│  评估版图质量                                            │
│    └────┬────┘                                                          │
│         │                                                               │
│         ├────────────────┐                                              │
│         │                │                                              │
│         ▼                ▼                                              │
│    ┌─────────┐     ┌──────────┐                                         │
│    │ PASSED  │     │ITERATING │  未通过，需要迭代                         │
│    └────┬────┘     └────┬─────┘                                         │
│         │               │                                               │
│         │               └───────────────────────────┐                   │
│         │                                           │                   │
│         ▼                                           ▼                   │
│    ┌─────────┐                              (返回 PLANNING)              │
│    │COMPLETED│  任务完成，输出结果                                        │
│    └────┬────┘                                                          │
│         │                                                               │
│         ▼                                                               │
│    ┌─────────┐                                                          │
│    │  DONE   │  会话结束                                                 │
│    └─────────┘                                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.1.2 状态定义

```python
from enum import Enum

class AgentState(Enum):
    INIT = "init"              # 初始化
    PARSING = "parsing"        # 解析输入
    PLANNING = "planning"      # 生成规划
    EXECUTING = "executing"    # 执行中
    STEP_EXEC = "step_exec"    # 执行单个步骤
    STEP_DONE = "step_done"    # 步骤完成
    PLAN_DONE = "plan_done"    # 规划执行完成
    EVALUATING = "evaluating"  # 评估中
    PASSED = "passed"          # 评估通过
    ITERATING = "iterating"    # 迭代优化
    COMPLETED = "completed"    # 任务完成
    FAILED = "failed"          # 任务失败
    DONE = "done"              # 会话结束
```

### 4.1.3 主循环伪代码

```python
class AgentOrchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.llm_client = LLMClient(config.llm_config)
        self.mcp_client = MCPClient(config.mcp_config)
        self.rag_retriever = RAGRetriever(config.rag_config)
        self.memory = MemoryModule(config.memory_config)
        self.evaluator = Evaluator(config.eval_config)
        self.skill_registry = SkillRegistry()
        self.logger = Logger(config.log_config)
        
    def run(self, input_files: InputFiles) -> AgentResult:
        """Agent 主入口"""
        session = self._create_session()
        state = AgentState.INIT
        iteration_count = 0
        max_iterations = self.config.max_iterations
        
        try:
            while state != AgentState.DONE:
                state = self._transition(state, session)
                
                if state == AgentState.ITERATING:
                    iteration_count += 1
                    if iteration_count >= max_iterations:
                        self.logger.warn("Reached max iterations")
                        state = AgentState.COMPLETED
                        
        except Exception as e:
            self.logger.error(f"Agent failed: {e}")
            state = AgentState.FAILED
            session.error = str(e)
            
        finally:
            self._finalize_session(session)
            
        return self._build_result(session)
    
    def _transition(self, state: AgentState, session: Session) -> AgentState:
        """状态转换逻辑"""
        
        if state == AgentState.INIT:
            self._init_session(session)
            return AgentState.PARSING
            
        elif state == AgentState.PARSING:
            session.task_context = self._parse_inputs(session.input_files)
            return AgentState.PLANNING
            
        elif state == AgentState.PLANNING:
            session.plan = self._create_plan(session)
            session.current_step_index = 0
            return AgentState.EXECUTING
            
        elif state == AgentState.EXECUTING:
            if session.current_step_index < len(session.plan.steps):
                return AgentState.STEP_EXEC
            else:
                return AgentState.PLAN_DONE
                
        elif state == AgentState.STEP_EXEC:
            step = session.plan.steps[session.current_step_index]
            result = self._execute_step(step, session)
            session.step_results.append(result)
            return AgentState.STEP_DONE
            
        elif state == AgentState.STEP_DONE:
            last_result = session.step_results[-1]
            if last_result.ok:
                session.current_step_index += 1
                return AgentState.EXECUTING
            else:
                # 尝试 ReAct 模式调整
                if self._can_retry(session):
                    adjusted_step = self._react_adjust(session)
                    session.plan.steps[session.current_step_index] = adjusted_step
                    return AgentState.STEP_EXEC
                else:
                    return AgentState.FAILED
                    
        elif state == AgentState.PLAN_DONE:
            return AgentState.EVALUATING
            
        elif state == AgentState.EVALUATING:
            session.evaluation = self._evaluate(session)
            if session.evaluation.passed:
                return AgentState.PASSED
            else:
                return AgentState.ITERATING
                
        elif state == AgentState.PASSED:
            self._store_successful_case(session)
            return AgentState.COMPLETED
            
        elif state == AgentState.ITERATING:
            session.plan = self._refine_plan(session)
            session.current_step_index = 0
            session.step_results.clear()
            return AgentState.EXECUTING
            
        elif state == AgentState.COMPLETED:
            return AgentState.DONE
            
        elif state == AgentState.FAILED:
            self._store_failed_case(session)
            return AgentState.DONE
            
        return state
```

---

## 4.2 规划模块实现 (Planning)

### 4.2.1 规划策略

规划模块采用 **Plan-and-Execute + ReAct 混合策略**：

1. **Plan-and-Execute（宏观规划）**：任务开始时，生成完整的执行计划
2. **ReAct（细节执行）**：执行过程中遇到问题时，动态调整当前步骤

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        规划策略流程                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    Plan-and-Execute 阶段                        │    │
│  │                                                                 │    │
│  │  TaskContext ──► RAG 检索 ──► 历史案例 ──► LLM 生成 ──► Plan    │    │
│  │                                                                 │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                │                                        │
│                                ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                      执行阶段                                    │    │
│  │                                                                 │    │
│  │  Plan.step[i] ──► 执行 ──► 成功? ──┬──► 下一步                  │    │
│  │                            │       │                            │    │
│  │                            └── 否 ─┘                            │    │
│  │                                │                                │    │
│  │                                ▼                                │    │
│  │  ┌────────────────────────────────────────────────────────┐    │    │
│  │  │                  ReAct 调整阶段                          │    │    │
│  │  │                                                         │    │    │
│  │  │  失败结果 ──► 思考(Thought) ──► 调整(Action) ──► 重试    │    │    │
│  │  │                                                         │    │    │
│  │  └────────────────────────────────────────────────────────┘    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2.2 关键接口定义

```python
class PlanningModule:
    """规划模块"""
    
    def __init__(self, llm_client: LLMClient, rag_retriever: RAGRetriever):
        self.llm = llm_client
        self.rag = rag_retriever
        self.prompt_templates = PromptTemplates()
        
    def create_initial_plan(
        self,
        context: TaskContext,
        memory: MemoryModule
    ) -> Plan:
        """
        生成初始执行计划
        
        参数:
            context: 任务上下文，包含 Circuit, DRCRuleSet, DesignObjectives
            memory: 记忆模块，用于检索历史案例
            
        返回:
            Plan: 包含步骤列表的执行计划
        """
        pass
        
    def refine_plan(
        self,
        prev_plan: Plan,
        feedback: EvaluationReport
    ) -> Plan:
        """
        根据评估反馈调整规划
        
        参数:
            prev_plan: 上一次的执行计划
            feedback: 评估报告，包含问题描述和改进建议
            
        返回:
            Plan: 调整后的新计划
        """
        pass
        
    def plan_next_step(
        self,
        state: ExecutionState,
        failure_info: StepResult
    ) -> PlanStep:
        """
        ReAct 模式：根据失败信息规划下一步调整
        
        参数:
            state: 当前执行状态
            failure_info: 失败的步骤结果
            
        返回:
            PlanStep: 调整后的步骤
        """
        pass
```

### 4.2.3 Prompt 结构设计

#### 初始规划 Prompt 模板

```
[System Prompt]
你是一个专业的模拟集成电路版图设计专家。你的任务是根据给定的电路网表、
DRC规则和设计目标，制定一个详细的版图设计执行计划。

你可以使用以下工具来完成设计：
{available_skills_description}

请生成一个步骤清晰、可执行的设计计划，每个步骤需要指定：
1. 步骤名称
2. 调用的工具名
3. 工具参数
4. 预期输出

[User Prompt]
## 任务输入

### 电路信息
{circuit_summary}

### DRC 规则摘要
{drc_rules_summary}

### 设计目标
{objectives_summary}

## 参考知识
以下是从知识库检索到的相关版图技巧：
{rag_chunks}

## 历史案例
以下是相似的历史设计案例供参考：
{similar_cases}

## 输出要求
请以 JSON 格式输出执行计划，结构如下：
{
  "plan_summary": "计划概述",
  "steps": [
    {
      "step_id": 1,
      "name": "步骤名称",
      "skill_name": "工具名",
      "params": {...},
      "expected_output": "预期输出描述",
      "dependencies": []
    }
  ]
}
```

#### ReAct 调整 Prompt 模板

```
[System Prompt]
你正在执行版图设计任务，当前步骤执行失败。请分析失败原因并提出调整方案。

采用 Thought-Action-Observation 模式：
1. Thought: 分析失败原因
2. Action: 提出调整后的步骤
3. Observation: 预期调整效果

[User Prompt]
## 当前执行状态
已完成步骤：
{completed_steps}

## 失败步骤信息
步骤名称: {failed_step_name}
工具名: {failed_skill_name}
参数: {failed_params}
错误信息: {error_message}
错误码: {error_code}

## 请分析并调整
```

### 4.2.4 RAG 检索结果注入

```python
def _inject_rag_context(
    self,
    context: TaskContext,
    max_chunks: int = 5
) -> str:
    """
    检索并格式化 RAG 知识
    
    参数:
        context: 任务上下文
        max_chunks: 最大返回片段数
        
    返回:
        str: 格式化的知识片段文本
    """
    # 构造检索查询
    queries = [
        f"差分对版图布局技巧 {context.circuit.get_module_types()}",
        f"电流镜匹配布局方法",
        f"{context.drc_rules.tech} 工艺版图设计注意事项"
    ]
    
    all_chunks = []
    for query in queries:
        retrieval_context = RetrievalContext(
            task_type="layout_planning",
            circuit_modules=context.circuit.get_module_types()
        )
        chunks = self.rag.retrieve_knowledge(
            query=query,
            context=retrieval_context,
            top_k=max_chunks
        )
        all_chunks.extend(chunks)
    
    # 去重和排序
    unique_chunks = self._dedupe_chunks(all_chunks)
    sorted_chunks = sorted(unique_chunks, key=lambda c: c.relevance_score, reverse=True)
    
    # 格式化输出
    formatted = []
    for i, chunk in enumerate(sorted_chunks[:max_chunks]):
        formatted.append(f"### 知识片段 {i+1}\n来源: {chunk.source}\n{chunk.content}\n")
    
    return "\n".join(formatted)
```

### 4.2.5 Token 限制处理策略

```python
class TokenManager:
    """Token 管理器"""
    
    def __init__(self, max_context_tokens: int = 8000):
        self.max_context_tokens = max_context_tokens
        self.tokenizer = get_tokenizer()  # tiktoken 或其他
        
    def build_prompt_with_limit(
        self,
        system_prompt: str,
        user_prompt_template: str,
        context_parts: Dict[str, str],
        priority_order: List[str]
    ) -> str:
        """
        构建受 token 限制的 prompt
        
        策略:
        1. 系统提示词始终保留
        2. 按优先级顺序添加上下文部分
        3. 低优先级部分进行摘要或截断
        """
        used_tokens = self._count_tokens(system_prompt)
        remaining = self.max_context_tokens - used_tokens
        
        final_parts = {}
        for key in priority_order:
            if key not in context_parts:
                continue
            content = context_parts[key]
            content_tokens = self._count_tokens(content)
            
            if content_tokens <= remaining * 0.3:  # 单部分不超过 30%
                final_parts[key] = content
                remaining -= content_tokens
            else:
                # 摘要或截断
                summarized = self._summarize_or_truncate(
                    content, 
                    max_tokens=int(remaining * 0.3)
                )
                final_parts[key] = summarized
                remaining -= self._count_tokens(summarized)
                
        return user_prompt_template.format(**final_parts)
    
    def _summarize_or_truncate(self, content: str, max_tokens: int) -> str:
        """对过长内容进行摘要或截断"""
        current_tokens = self._count_tokens(content)
        if current_tokens <= max_tokens:
            return content
            
        # 简单截断策略（可替换为 LLM 摘要）
        ratio = max_tokens / current_tokens
        char_limit = int(len(content) * ratio * 0.9)  # 保守估计
        return content[:char_limit] + "\n...[内容已截断]"
```

---

## 4.3 执行模块实现 (Action)

### 4.3.1 步骤执行流程

```python
class ActionModule:
    """行动模块"""
    
    def __init__(
        self,
        mcp_client: MCPClient,
        skill_registry: SkillRegistry
    ):
        self.mcp = mcp_client
        self.skills = skill_registry
        self.retry_policy = RetryPolicy()
        
    def execute_step(
        self,
        step: PlanStep,
        context: ExecutionContext
    ) -> StepResult:
        """
        执行单个规划步骤
        
        参数:
            step: 规划步骤
            context: 执行上下文（包含当前状态、历史结果等）
            
        返回:
            StepResult: 步骤执行结果
        """
        self.logger.info(f"Executing step: {step.name}")
        
        # 1. 解析步骤，映射到工具调用
        tool_request = self._map_step_to_tool_call(step, context)
        
        # 2. 执行工具调用（带重试）
        result = self._execute_with_retry(tool_request)
        
        # 3. 处理结果
        step_result = self._process_result(step, result)
        
        # 4. 更新执行上下文
        context.add_step_result(step, step_result)
        
        return step_result
        
    def _map_step_to_tool_call(
        self,
        step: PlanStep,
        context: ExecutionContext
    ) -> ToolCallRequest:
        """将规划步骤映射为工具调用请求"""
        
        # 获取技能定义
        skill = self.skills.get_skill(step.skill_name)
        
        # 解析参数（可能需要从上下文中获取动态值）
        resolved_params = self._resolve_params(step.params, context)
        
        return ToolCallRequest(
            name=skill.mcp_tool_name,
            input=resolved_params,
            trace_id=context.trace_id
        )
        
    def _execute_with_retry(
        self,
        request: ToolCallRequest
    ) -> ToolCallResult:
        """带重试策略的工具执行"""
        
        last_error = None
        for attempt in range(self.retry_policy.max_attempts):
            try:
                result = self.mcp.call_tool(request)
                if result.ok:
                    return result
                    
                # 判断是否可重试
                if not self._is_retryable_error(result.error):
                    return result
                    
                last_error = result.error
                
            except TimeoutError:
                last_error = Error(code="TIMEOUT", message="Tool call timed out")
                
            except Exception as e:
                last_error = Error(code="UNKNOWN", message=str(e))
                
            # 等待后重试
            time.sleep(self.retry_policy.get_delay(attempt))
            
        return ToolCallResult(
            ok=False,
            error=last_error,
            data=None
        )
```

### 4.3.2 MCP 集成方式

#### 工具调用请求结构

```python
@dataclass
class ToolCallRequest:
    """MCP 工具调用请求"""
    name: str              # 工具名，如 "layout.create_common_centroid_pair"
    input: Dict[str, Any]  # 工具输入参数
    trace_id: str          # 追踪 ID，用于日志关联
    timeout_ms: int = 30000  # 超时时间
```

#### 响应解析逻辑

```python
@dataclass
class ToolCallResult:
    """MCP 工具调用结果"""
    ok: bool
    error: Optional[Error]
    data: Optional[Dict[str, Any]]
    duration_ms: int = 0

class MCPClient:
    """MCP 客户端"""
    
    def call_tool(self, request: ToolCallRequest) -> ToolCallResult:
        """调用 MCP 工具"""
        start_time = time.time()
        
        try:
            # 构造 MCP 消息
            mcp_message = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": request.name,
                    "arguments": request.input
                },
                "id": request.trace_id
            }
            
            # 发送请求
            response = self._send_and_receive(mcp_message, request.timeout_ms)
            
            # 解析响应
            duration_ms = int((time.time() - start_time) * 1000)
            
            if "error" in response:
                return ToolCallResult(
                    ok=False,
                    error=Error(
                        code=response["error"].get("code", "UNKNOWN"),
                        message=response["error"].get("message", "Unknown error")
                    ),
                    data=None,
                    duration_ms=duration_ms
                )
            
            return ToolCallResult(
                ok=True,
                error=None,
                data=response.get("result", {}),
                duration_ms=duration_ms
            )
            
        except TimeoutError:
            return ToolCallResult(
                ok=False,
                error=Error(code="TIMEOUT", message="MCP call timed out"),
                data=None
            )
```

### 4.3.3 错误恢复策略

| 错误类型 | 错误码 | 恢复策略 |
|----------|--------|----------|
| 参数错误 | `INVALID_PARAM` | 不重试，返回 ReAct 调整 |
| 工具超时 | `TIMEOUT` | 重试 2 次，然后返回失败 |
| DRC 违规 | `DRC_VIOLATION` | 不重试，触发重新规划 |
| KLayout 不可用 | `SERVICE_UNAVAILABLE` | 重试 3 次，然后终止任务 |
| 资源不足 | `RESOURCE_EXHAUSTED` | 等待后重试 |

```python
class RetryPolicy:
    """重试策略"""
    
    RETRYABLE_ERRORS = {"TIMEOUT", "SERVICE_UNAVAILABLE", "RESOURCE_EXHAUSTED"}
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 30.0
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay_seconds
        self.max_delay = max_delay_seconds
        
    def is_retryable(self, error: Error) -> bool:
        return error.code in self.RETRYABLE_ERRORS
        
    def get_delay(self, attempt: int) -> float:
        """指数退避延迟"""
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)
```

### 4.3.4 执行结果反馈

```python
def _feedback_results(
    self,
    step: PlanStep,
    result: StepResult,
    memory: MemoryModule,
    planning: PlanningModule
) -> None:
    """
    将执行结果反馈给其他模块
    """
    # 存储到短期记忆
    memory.store_step_result(step, result)
    
    # 如果成功且产生了版图变更，更新当前布局状态
    if result.ok and result.layout_changes:
        memory.update_current_layout(result.layout_changes)
    
    # 如果失败，记录失败信息供 ReAct 使用
    if not result.ok:
        memory.store_failure_context(
            step=step,
            error=result.error,
            execution_context=self._get_current_context()
        )
```

---

## 4.4 记忆模块实现 (Memory)

### 4.4.1 短期记忆

短期记忆存储当前设计会话的即时上下文，使用内存数据结构实现。

```python
@dataclass
class ShortTermMemory:
    """短期记忆"""
    session_id: str
    task_context: TaskContext
    current_plan: Optional[Plan] = None
    execution_history: List[StepRecord] = field(default_factory=list)
    current_layout_state: Optional[LayoutState] = None
    failure_contexts: List[FailureContext] = field(default_factory=list)
    
    def store_step_result(self, step: PlanStep, result: StepResult) -> None:
        """存储步骤执行结果"""
        record = StepRecord(
            step_id=step.id,
            step_name=step.name,
            skill_name=step.skill_name,
            params=step.params,
            result=result,
            timestamp=datetime.now()
        )
        self.execution_history.append(record)
        
    def get_execution_history(self) -> List[StepRecord]:
        """获取执行历史"""
        return self.execution_history.copy()
        
    def get_completed_steps_summary(self) -> str:
        """获取已完成步骤摘要（用于 Prompt）"""
        summaries = []
        for record in self.execution_history:
            status = "成功" if record.result.ok else "失败"
            summaries.append(f"- {record.step_name}: {status}")
        return "\n".join(summaries)
        
    def update_current_layout(self, changes: LayoutChanges) -> None:
        """更新当前版图状态"""
        if self.current_layout_state is None:
            self.current_layout_state = LayoutState()
        self.current_layout_state.apply_changes(changes)
```

### 4.4.2 长期记忆

长期记忆存储跨会话的持久化信息，使用向量数据库实现。

```python
class LongTermMemory:
    """长期记忆"""
    
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.collection_name = "design_cases"
        
    def store_case(self, case: DesignCase) -> str:
        """
        存储设计案例
        
        参数:
            case: 设计案例，包含完整的上下文、计划、结果和评估
            
        返回:
            str: 案例 ID
        """
        # 生成案例摘要用于检索
        case_summary = self._generate_case_summary(case)
        
        # 生成嵌入向量
        embedding = self.embedding_model.embed(case_summary)
        
        # 存储到向量数据库
        case_id = self.vector_store.add(
            collection=self.collection_name,
            id=case.id,
            embedding=embedding,
            metadata={
                "circuit_type": case.context.circuit.type,
                "tech": case.context.drc_rules.tech,
                "success": case.evaluation.passed,
                "timestamp": case.timestamp.isoformat(),
                "summary": case_summary
            },
            document=case.to_json()
        )
        
        return case_id
        
    def retrieve_similar_cases(
        self,
        context: TaskContext,
        top_k: int = 3,
        success_only: bool = True
    ) -> List[DesignCase]:
        """
        检索相似历史案例
        
        参数:
            context: 当前任务上下文
            top_k: 返回数量
            success_only: 是否只返回成功案例
            
        返回:
            List[DesignCase]: 相似案例列表
        """
        # 构造查询
        query = self._build_case_query(context)
        query_embedding = self.embedding_model.embed(query)
        
        # 构造过滤条件
        filters = {}
        if success_only:
            filters["success"] = True
            
        # 向量检索
        results = self.vector_store.query(
            collection=self.collection_name,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        # 解析结果
        cases = []
        for result in results:
            case = DesignCase.from_json(result.document)
            case.relevance_score = result.score
            cases.append(case)
            
        return cases
```

### 4.4.3 记忆读写接口汇总

| 接口名 | 类型 | 功能 | 签名 |
|--------|------|------|------|
| `store_step_result` | 短期 | 存储步骤执行结果 | `(step: PlanStep, result: StepResult) -> None` |
| `get_execution_history` | 短期 | 获取执行历史 | `() -> List[StepRecord]` |
| `update_current_layout` | 短期 | 更新版图状态 | `(changes: LayoutChanges) -> None` |
| `store_failure_context` | 短期 | 存储失败上下文 | `(step, error, context) -> None` |
| `store_case` | 长期 | 存储设计案例 | `(case: DesignCase) -> str` |
| `retrieve_similar_cases` | 长期 | 检索相似案例 | `(context, top_k) -> List[DesignCase]` |
| `store_failed_case` | 长期 | 存储失败案例 | `(case: FailedCase) -> str` |
| `retrieve_failure_patterns` | 长期 | 检索失败模式 | `(error_type) -> List[FailurePattern]` |

---

## 4.5 自我迭代与进化机制集成

### 4.5.1 评估触发迭代流程

```python
def _handle_evaluation_feedback(
    self,
    session: Session,
    evaluation: EvaluationReport
) -> Optional[Plan]:
    """
    处理评估反馈，决定是否迭代
    
    返回:
        Optional[Plan]: 如果需要迭代，返回新计划；否则返回 None
    """
    if evaluation.passed:
        return None
        
    # 检查是否可以继续迭代
    if session.iteration_count >= self.config.max_iterations:
        self.logger.warn("Max iterations reached, accepting current result")
        return None
        
    # 分析失败原因
    failure_analysis = self._analyze_failure(evaluation)
    
    # 判断是否需要调整 Prompt 策略
    if failure_analysis.requires_strategy_change:
        # 调用元提示 Agent
        new_prompt = self._invoke_meta_prompt_agent(
            current_prompt=session.current_system_prompt,
            failure_analysis=failure_analysis,
            evaluation=evaluation
        )
        session.current_system_prompt = new_prompt
        
    # 生成改进的计划
    refined_plan = self.planning.refine_plan(
        prev_plan=session.plan,
        feedback=evaluation,
        failure_analysis=failure_analysis
    )
    
    return refined_plan
```

### 4.5.2 元提示 Agent 实现

```python
class MetaPromptAgent:
    """元提示 Agent - 负责优化主 Agent 的 Prompt 策略"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        
    def generate_improved_prompt(
        self,
        current_prompt: str,
        failure_analysis: FailureAnalysis,
        evaluation: EvaluationReport
    ) -> str:
        """
        生成改进的系统提示词
        
        参数:
            current_prompt: 当前的系统提示词
            failure_analysis: 失败原因分析
            evaluation: 评估报告
            
        返回:
            str: 改进后的系统提示词
        """
        meta_prompt = f"""你是一个版图设计专家，负责优化 AI Agent 的设计策略。

## 当前 Agent 的系统提示词
{current_prompt}

## 设计任务执行情况
根据该策略生成的版图存在以下问题：
- DRC 错误数: {evaluation.metrics.drc_error_count}
- 主要问题: {failure_analysis.main_issues}
- 失败的步骤: {failure_analysis.failed_steps}

## 详细评估
{evaluation.qualitative_feedback}

## 任务
请分析失败原因，并生成一个改进后的系统提示词。改进的提示词应该：
1. 保留原有的有效指导
2. 针对性地添加避免上述问题的具体指令
3. 更加明确和具体

请直接输出改进后的系统提示词，不需要解释。
"""
        
        response = self.llm.generate(meta_prompt)
        return response.content
```

### 4.5.3 失败案例存储与检索

```python
@dataclass
class FailedCase:
    """失败案例"""
    id: str
    context: TaskContext
    plan: Plan
    execution_result: ExecutionResult
    evaluation: EvaluationReport
    failure_analysis: FailureAnalysis
    resolution: Optional[str] = None  # 最终解决方案（如果后来解决了）
    timestamp: datetime = field(default_factory=datetime.now)

class FailureCaseStore:
    """失败案例存储"""
    
    def store_failed_case(self, case: FailedCase) -> str:
        """存储失败案例"""
        # 提取关键失败模式
        failure_signature = self._extract_failure_signature(case)
        
        # 存储到向量库
        embedding = self.embedding_model.embed(failure_signature)
        
        return self.vector_store.add(
            collection="failed_cases",
            id=case.id,
            embedding=embedding,
            metadata={
                "error_types": case.failure_analysis.error_types,
                "failed_skills": case.failure_analysis.failed_skills,
                "resolved": case.resolution is not None
            },
            document=case.to_json()
        )
        
    def retrieve_similar_failures(
        self,
        current_failure: FailureAnalysis,
        top_k: int = 3
    ) -> List[FailedCase]:
        """
        检索相似的历史失败案例
        
        用于：
        1. 避免重复同样的错误
        2. 如果历史案例已解决，可以借鉴解决方案
        """
        query = self._build_failure_query(current_failure)
        embedding = self.embedding_model.embed(query)
        
        results = self.vector_store.query(
            collection="failed_cases",
            query_embedding=embedding,
            top_k=top_k
        )
        
        return [FailedCase.from_json(r.document) for r in results]
```

---

## 4.6 日志与可观测性

### 4.6.1 日志记录内容

| 日志类别 | 记录内容 | 日志级别 |
|----------|----------|----------|
| 会话生命周期 | 会话开始/结束、状态转换 | INFO |
| 规划轨迹 | 初始计划、计划调整、ReAct 思考过程 | INFO/DEBUG |
| 工具调用 | 请求参数、响应结果、耗时 | INFO |
| 错误信息 | 错误码、错误消息、堆栈跟踪 | ERROR |
| 评估结果 | 各项指标、定性反馈、是否通过 | INFO |
| 迭代记录 | 迭代次数、改进内容 | INFO |
| RAG 检索 | 查询、命中文档、相关性分数 | DEBUG |
| 性能指标 | 各阶段耗时、Token 使用量 | INFO |

### 4.6.2 结构化日志格式

```python
@dataclass
class LogEntry:
    """结构化日志条目"""
    timestamp: str
    level: str
    session_id: str
    trace_id: str
    event_type: str
    message: str
    data: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

# 日志示例
{
    "timestamp": "2024-01-15T10:30:45.123Z",
    "level": "INFO",
    "session_id": "sess_abc123",
    "trace_id": "trace_xyz789",
    "event_type": "TOOL_CALL",
    "message": "Executing skill: create_common_centroid_pair",
    "data": {
        "skill_name": "layout.create_common_centroid_pair",
        "params": {
            "device_a": "M1",
            "device_b": "M2",
            "arrangement": "ABBA"
        },
        "result": {
            "ok": true,
            "cell_name": "diff_pair_cc"
        }
    },
    "duration_ms": 1234
}
```

### 4.6.3 Logger 实现

```python
class AgentLogger:
    """Agent 日志记录器"""
    
    def __init__(self, config: LogConfig):
        self.config = config
        self.session_id = None
        self.trace_id = None
        
        # 配置 Python logging
        self._setup_handlers()
        
    def _setup_handlers(self):
        """配置日志处理器"""
        self.logger = logging.getLogger("agent")
        self.logger.setLevel(self.config.level)
        
        # JSON 文件输出
        file_handler = logging.FileHandler(
            self.config.log_file,
            encoding="utf-8"
        )
        file_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(file_handler)
        
        # 控制台输出（可读格式）
        if self.config.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(ReadableFormatter())
            self.logger.addHandler(console_handler)
            
    def log_tool_call(
        self,
        skill_name: str,
        params: Dict,
        result: ToolCallResult,
        duration_ms: int
    ):
        """记录工具调用"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level="INFO",
            session_id=self.session_id,
            trace_id=self.trace_id,
            event_type="TOOL_CALL",
            message=f"Executed skill: {skill_name}",
            data={
                "skill_name": skill_name,
                "params": params,
                "result": {
                    "ok": result.ok,
                    "error": result.error.to_dict() if result.error else None
                }
            },
            duration_ms=duration_ms
        )
        self.logger.info(entry.to_json())
        
    def log_plan_created(self, plan: Plan):
        """记录规划创建"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level="INFO",
            session_id=self.session_id,
            trace_id=self.trace_id,
            event_type="PLAN_CREATED",
            message=f"Created plan with {len(plan.steps)} steps",
            data={
                "plan_summary": plan.summary,
                "step_names": [s.name for s in plan.steps]
            }
        )
        self.logger.info(entry.to_json())
        
    def log_evaluation(self, evaluation: EvaluationReport):
        """记录评估结果"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level="INFO",
            session_id=self.session_id,
            trace_id=self.trace_id,
            event_type="EVALUATION",
            message=f"Evaluation {'passed' if evaluation.passed else 'failed'}",
            data={
                "passed": evaluation.passed,
                "metrics": {
                    "drc_error_count": evaluation.metrics.drc_error_count,
                    "area": evaluation.metrics.area,
                    "matching_score": evaluation.metrics.matching_score
                },
                "suggestions": evaluation.suggestions
            }
        )
        self.logger.info(entry.to_json())
```

---

## 附录 B：配置项定义

```python
@dataclass
class OrchestratorConfig:
    """Orchestrator 配置"""
    
    # LLM 配置
    llm_config: LLMConfig
    
    # MCP 配置
    mcp_config: MCPConfig
    
    # RAG 配置
    rag_config: RAGConfig
    
    # 记忆模块配置
    memory_config: MemoryConfig
    
    # 评估配置
    eval_config: EvalConfig
    
    # 日志配置
    log_config: LogConfig
    
    # 运行参数
    max_iterations: int = 3          # 最大迭代次数
    step_timeout_ms: int = 60000     # 单步超时
    session_timeout_ms: int = 600000 # 会话超时

@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str = "openai"         # openai, anthropic, etc.
    model: str = "gpt-4"             # 模型名称
    api_key: str = ""                # API Key（从环境变量读取）
    max_tokens: int = 4096           # 最大输出 token
    temperature: float = 0.1         # 温度参数
    max_context_tokens: int = 8000   # 最大上下文 token

@dataclass
class MCPConfig:
    """MCP 配置"""
    transport: str = "stdio"         # stdio, http
    klayout_path: str = ""           # KLayout 可执行文件路径
    server_script: str = ""          # MCP 服务器脚本路径
    timeout_ms: int = 30000          # 默认超时
```
