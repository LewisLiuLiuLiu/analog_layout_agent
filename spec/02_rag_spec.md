# 第2章 RAG 知识库实现

---

## 2.1 RAG 知识库总体设计

### 2.1.1 在系统中的作用

RAG（检索增强生成）知识库在本系统中扮演 **外部知识存储与检索** 的角色，其核心职责是：

1. **存储版图技巧文档**：将 Markdown 格式的版图设计技巧文档进行结构化处理和向量化存储
2. **语义检索**：根据当前设计任务，检索最相关的版图技巧知识
3. **上下文增强**：为规划模块提供专业知识支持，增强 LLM 的设计决策质量

### 2.1.2 边界定义

| 属于 RAG 知识库 | 不属于 RAG 知识库 |
|----------------|------------------|
| 版图技巧文档（Markdown） | Agent 运行时产生的记忆 |
| 设计规则最佳实践 | 历史设计案例（属于 Memory 模块） |
| 电路拓扑布局指南 | 用户偏好设置 |
| 工艺特定注意事项 | 失败案例与教训（属于 Memory 模块） |

### 2.1.3 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG 知识库系统架构                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    离线索引管线 (Indexing Pipeline)              │   │
│  │                                                                  │   │
│  │  原始文档 ──► 预处理 ──► 切片 ──► 嵌入 ──► 向量存储              │   │
│  │  (MD/PDF)    (清洗)    (Chunk)  (Embed)   (Vector DB)           │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    在线检索管线 (Retrieval Pipeline)             │   │
│  │                                                                  │   │
│  │  查询 ──► 查询改写 ──► 混合检索 ──► 重排序 ──► 返回 Top-K        │   │
│  │  (Query)  (Rewrite)   (Hybrid)   (Rerank)   (Chunks)            │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2.2 文档来源与预处理流程

### 2.2.1 支持的原始格式

| 格式 | 处理方式 | 优先级 |
|------|----------|--------|
| Markdown (.md) | 直接解析，保留结构 | 主要格式 |
| PDF (.pdf) | 使用 PyMuPDF/pdfplumber 提取文本，转换为 Markdown | 支持 |
| Word (.docx) | 使用 python-docx 提取，转换为 Markdown | 支持 |
| 纯文本 (.txt) | 直接读取，添加基本结构 | 支持 |

**工程假设**：所有原始文档最终需要统一转换为 Markdown 格式进行处理。

### 2.2.2 Markdown 预处理管线

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  原始文档    │────►│  格式转换    │────►│  内容清洗    │────►│  结构解析    │
│ (MD/PDF/DOC)│     │ (→Markdown) │     │ (Normalize) │     │ (Parse)     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ▼
                                        ┌─────────────────────────────────┐
                                        │     结构化文档对象               │
                                        │  (ParsedDocument)               │
                                        └─────────────────────────────────┘
```

### 2.2.3 预处理步骤详解

#### 步骤 1: 编码与格式清洗

```python
class DocumentCleaner:
    """文档清洗器"""
    
    def clean(self, raw_content: str) -> str:
        """
        执行内容清洗
        
        处理内容:
        1. 统一编码为 UTF-8
        2. 规范化换行符 (\r\n -> \n)
        3. 去除多余空行 (连续空行压缩为单个)
        4. 去除行首行尾多余空格
        5. 处理特殊字符 (非打印字符)
        """
        content = raw_content
        
        # 统一换行符
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # 压缩连续空行
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 去除每行首尾空格（保留缩进）
        lines = content.split('\n')
        lines = [line.rstrip() for line in lines]
        content = '\n'.join(lines)
        
        # 去除非打印字符（保留制表符）
        content = re.sub(r'[^\x09\x0a\x20-\x7e\u4e00-\u9fff]', '', content)
        
        return content
```

#### 步骤 2: 标题层级解析

```python
@dataclass
class HeadingNode:
    """标题节点"""
    level: int          # 标题层级 1-6
    text: str           # 标题文本
    line_number: int    # 行号
    children: List['HeadingNode'] = field(default_factory=list)
    content: str = ""   # 该标题下的内容

class MarkdownParser:
    """Markdown 解析器"""
    
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$')
    
    def parse(self, content: str) -> ParsedDocument:
        """
        解析 Markdown 文档结构
        
        返回:
            ParsedDocument: 包含标题树和内容的结构化文档
        """
        lines = content.split('\n')
        root = HeadingNode(level=0, text="root", line_number=0)
        current_node = root
        node_stack = [root]
        current_content_lines = []
        
        for i, line in enumerate(lines):
            heading_match = self.HEADING_PATTERN.match(line)
            
            if heading_match:
                # 保存之前节点的内容
                if current_content_lines:
                    current_node.content = '\n'.join(current_content_lines)
                    current_content_lines = []
                
                level = len(heading_match.group(1))
                text = heading_match.group(2).strip()
                
                new_node = HeadingNode(level=level, text=text, line_number=i)
                
                # 找到合适的父节点
                while node_stack and node_stack[-1].level >= level:
                    node_stack.pop()
                
                if node_stack:
                    node_stack[-1].children.append(new_node)
                
                node_stack.append(new_node)
                current_node = new_node
            else:
                current_content_lines.append(line)
        
        # 保存最后一个节点的内容
        if current_content_lines:
            current_node.content = '\n'.join(current_content_lines)
        
        return ParsedDocument(root=root, raw_content=content)
```

#### 步骤 3: 特殊内容保留策略

| 内容类型 | 保留策略 | 说明 |
|----------|----------|------|
| 代码块 | 完整保留，标记语言类型 | 版图代码示例需要完整保留 |
| 列表 | 保留层级结构 | 设计步骤通常以列表形式呈现 |
| 表格 | 转换为结构化格式 | DRC 规则表格需要保留 |
| 图片引用 | 保留引用，记录元数据 | 图片本身不处理，保留引用信息 |
| 数学公式 | 完整保留 LaTeX 格式 | 版图计算公式需要保留 |

```python
class ContentPreserver:
    """特殊内容保留器"""
    
    CODE_BLOCK_PATTERN = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    TABLE_PATTERN = re.compile(r'(\|.+\|)\n(\|[-:| ]+\|)\n((?:\|.+\|\n?)+)')
    
    def extract_special_content(self, content: str) -> Tuple[str, List[SpecialBlock]]:
        """
        提取并标记特殊内容
        
        返回:
            Tuple[str, List[SpecialBlock]]: 
            - 替换后的内容（特殊内容用占位符替代）
            - 特殊内容块列表
        """
        special_blocks = []
        processed = content
        
        # 提取代码块
        for i, match in enumerate(self.CODE_BLOCK_PATTERN.finditer(content)):
            block = SpecialBlock(
                type="code",
                language=match.group(1),
                content=match.group(2),
                placeholder=f"[[CODE_BLOCK_{i}]]"
            )
            special_blocks.append(block)
            processed = processed.replace(match.group(0), block.placeholder)
        
        # 提取表格
        for i, match in enumerate(self.TABLE_PATTERN.finditer(processed)):
            block = SpecialBlock(
                type="table",
                content=match.group(0),
                placeholder=f"[[TABLE_{i}]]"
            )
            special_blocks.append(block)
            processed = processed.replace(match.group(0), block.placeholder)
        
        return processed, special_blocks
```

---

## 2.3 切片 (Chunking) 策略规格

### 2.3.1 结构化切片规则

以 Markdown 标题层级为主要切片边界：

| 切片边界 | 规则描述 | 示例 |
|----------|----------|------|
| H1 (`#`) | 作为文档级边界，不单独成块 | `# 版图设计指南` |
| H2 (`##`) | 主要切片边界，每个 H2 section 为一个独立 Chunk | `## 差分对布局技巧` |
| H3 (`###`) | 次级切片边界，当 H2 内容过长时按 H3 切分 | `### 共质心布局方法` |
| 无标题段落 | 归入最近的上级标题 Chunk | - |

#### 切片参数配置

```python
@dataclass
class ChunkingConfig:
    """切片配置"""
    
    # Token 限制
    min_chunk_tokens: int = 100      # 最小 Chunk 大小
    max_chunk_tokens: int = 512      # 最大 Chunk 大小
    target_chunk_tokens: int = 300   # 目标 Chunk 大小
    
    # 重叠配置
    overlap_tokens: int = 50         # Chunk 间重叠 token 数
    
    # 结构化切片配置
    primary_split_level: int = 2     # 主切片级别 (H2)
    secondary_split_level: int = 3   # 次级切片级别 (H3)
    
    # 语义切片配置
    sentence_split_enabled: bool = True
    min_sentences_per_chunk: int = 3
```

### 2.3.2 语义切片规则

当结构化切片产生的 Chunk 超过最大 token 限制时，启用语义切片：

```python
class SemanticChunker:
    """语义切片器"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.sentence_splitter = SentenceSplitter()
        self.tokenizer = get_tokenizer()
        
    def chunk_large_section(self, section: HeadingNode) -> List[Chunk]:
        """
        对过大的 section 进行语义切片
        
        策略:
        1. 先按句子分割
        2. 使用滑动窗口组合句子
        3. 在语义边界处切分
        """
        sentences = self.sentence_splitter.split(section.content)
        chunks = []
        
        current_chunk_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.config.max_chunk_tokens:
                # 当前 Chunk 已满，创建新 Chunk
                if current_chunk_sentences:
                    chunk = self._create_chunk(
                        section=section,
                        sentences=current_chunk_sentences,
                        chunk_index=len(chunks)
                    )
                    chunks.append(chunk)
                
                # 保留重叠部分
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences
                )
                current_chunk_sentences = overlap_sentences + [sentence]
                current_tokens = sum(
                    self._count_tokens(s) for s in current_chunk_sentences
                )
            else:
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # 处理最后一个 Chunk
        if current_chunk_sentences:
            chunk = self._create_chunk(
                section=section,
                sentences=current_chunk_sentences,
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
        
        return chunks

class SentenceSplitter:
    """句子分割器"""
    
    # 中文句子结束符
    CN_TERMINATORS = '。！？；'
    # 英文句子结束符
    EN_TERMINATORS = '.!?;'
    
    def split(self, text: str) -> List[str]:
        """
        分割句子（支持中英文混合）
        
        规则:
        1. 按句号/问号/叹号/分号分割
        2. 保留列表项完整性
        3. 代码块不分割
        """
        # 保护代码块
        text, code_blocks = self._protect_code_blocks(text)
        
        # 分割句子
        pattern = f'([{self.CN_TERMINATORS}{self.EN_TERMINATORS}])'
        parts = re.split(pattern, text)
        
        sentences = []
        current = ""
        for part in parts:
            current += part
            if part in self.CN_TERMINATORS + self.EN_TERMINATORS:
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        # 恢复代码块
        sentences = self._restore_code_blocks(sentences, code_blocks)
        
        return [s for s in sentences if s]
```

### 2.3.3 Chunk 元数据定义

```python
@dataclass
class ChunkMetadata:
    """Chunk 元数据"""
    
    # 标识信息
    id: str                          # Chunk 唯一 ID
    doc_id: str                      # 来源文档 ID
    doc_name: str                    # 来源文档名称
    
    # 位置信息
    title_path: List[str]            # 标题路径，如 ["版图指南", "差分对", "共质心"]
    start_line: int                  # 起始行号
    end_line: int                    # 结束行号
    chunk_index: int                 # 在同一 section 中的序号
    
    # 统计信息
    tokens: int                      # Token 数量
    char_count: int                  # 字符数量
    
    # 标签信息
    tags: List[str]                  # 自动提取的标签
    content_type: str                # 内容类型: text, code, table, mixed
    
    # 时间信息
    indexed_at: str                  # 索引时间 (ISO format)
    doc_modified_at: str             # 文档修改时间

@dataclass
class Chunk:
    """文本块"""
    content: str                     # Chunk 内容
    metadata: ChunkMetadata          # 元数据
    embedding: Optional[List[float]] = None  # 嵌入向量
```

---

## 2.4 嵌入模型与向量数据库选型与配置

### 2.4.1 嵌入模型推荐

| 模型 | 维度 | 语言支持 | 推荐场景 | 备注 |
|------|------|----------|----------|------|
| `text-embedding-3-small` (OpenAI) | 1536 | 多语言 | 生产环境 | 性价比最佳 |
| `bge-large-zh-v1.5` (BAAI) | 1024 | 中文优化 | 中文为主 | 开源免费 |
| `bge-m3` (BAAI) | 1024 | 多语言 | 中英混合 | 支持多语言 |
| `all-MiniLM-L6-v2` (Sentence-Transformers) | 384 | 英文 | 开发测试 | 轻量快速 |

**推荐选择**：

- **生产环境**：`text-embedding-3-small`（OpenAI）- 质量稳定，多语言支持好
- **私有部署**：`bge-m3`（BAAI）- 开源，支持中英文混合

### 2.4.2 向量数据库选型

| 数据库 | 部署方式 | 适用场景 | 特点 |
|--------|----------|----------|------|
| **Chroma** | 内嵌/独立 | 开发、小规模生产 | 简单易用，Python 原生 |
| **Pinecone** | 云服务 | 生产环境 | 托管服务，高可用 |
| **Qdrant** | 自托管/云 | 中大规模 | 性能优秀，支持过滤 |
| **Elasticsearch** | 自托管 | 已有 ES 基础设施 | 混合检索原生支持 |

**推荐选择**：

- **开发环境**：Chroma（内嵌模式）- 零配置启动
- **生产环境**：Chroma（服务模式）或 Qdrant - 根据规模选择

### 2.4.3 向量数据库配置

```python
@dataclass
class VectorDBConfig:
    """向量数据库配置"""
    
    # 基础配置
    provider: str = "chroma"          # chroma, pinecone, qdrant
    persist_directory: str = "./data/vectordb"
    
    # 索引配置
    embedding_dimension: int = 1536   # 向量维度
    distance_metric: str = "cosine"   # cosine, euclidean, dot_product
    
    # 集合配置
    collection_name: str = "layout_knowledge"
    
    # 性能配置
    batch_size: int = 100             # 批量插入大小
    
class ChromaVectorStore:
    """Chroma 向量存储实现"""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        
        # 初始化 Chroma 客户端
        self.client = chromadb.PersistentClient(
            path=config.persist_directory
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={
                "hnsw:space": config.distance_metric,
                "dimension": config.embedding_dimension
            }
        )
        
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """批量添加 Chunks"""
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            
            self.collection.add(
                ids=[c.metadata.id for c in batch],
                embeddings=[c.embedding for c in batch],
                documents=[c.content for c in batch],
                metadatas=[asdict(c.metadata) for c in batch]
            )
            
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[QueryResult]:
        """查询相似文档"""
        where = filters if filters else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        return self._parse_results(results)
```

### 2.4.4 命名空间设计

```python
# 按文档类型分集合
COLLECTIONS = {
    "layout_techniques": "版图技巧文档",
    "drc_guidelines": "DRC 规则指南",
    "circuit_topologies": "电路拓扑布局",
    "process_notes": "工艺注意事项"
}

# 元数据字段用于过滤
METADATA_FIELDS = {
    "doc_type": ["technique", "guideline", "topology", "process"],
    "circuit_type": ["diff_pair", "current_mirror", "cascode", "general"],
    "process_node": ["180nm", "130nm", "65nm", "general"],
    "language": ["zh", "en", "mixed"]
}
```

### 2.4.5 索引更新策略

| 场景 | 策略 | 触发条件 |
|------|------|----------|
| 初始构建 | 全量索引 | 首次部署 |
| 增量更新 | 仅处理新增/修改文档 | 文档变更时 |
| 重建索引 | 删除旧索引，重新构建 | 嵌入模型变更、Schema 变更 |

```python
class IndexManager:
    """索引管理器"""
    
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.index_state_file = "index_state.json"
        
    def update_index(self, doc_paths: List[str]) -> IndexUpdateResult:
        """
        增量更新索引
        
        策略:
        1. 比较文档修改时间与上次索引时间
        2. 仅处理新增或修改的文档
        3. 删除已不存在的文档对应的 Chunks
        """
        state = self._load_state()
        
        to_add = []
        to_delete = []
        
        for path in doc_paths:
            doc_id = self._path_to_doc_id(path)
            mod_time = os.path.getmtime(path)
            
            if doc_id not in state.indexed_docs:
                to_add.append(path)
            elif mod_time > state.indexed_docs[doc_id].indexed_at:
                to_delete.append(doc_id)
                to_add.append(path)
        
        # 检查已删除的文档
        current_docs = set(self._path_to_doc_id(p) for p in doc_paths)
        for doc_id in state.indexed_docs:
            if doc_id not in current_docs:
                to_delete.append(doc_id)
        
        # 执行删除
        for doc_id in to_delete:
            self.vector_store.delete_by_metadata({"doc_id": doc_id})
        
        # 执行添加
        for path in to_add:
            chunks = self._process_document(path)
            self.vector_store.add_chunks(chunks)
        
        # 更新状态
        self._save_state(state)
        
        return IndexUpdateResult(
            added=len(to_add),
            deleted=len(to_delete)
        )
```

---

## 2.5 检索器 (Retriever) 实现细节

### 2.5.1 查询改写 (Query Rewriting) 策略

```python
class QueryRewriter:
    """查询改写器"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        
    def rewrite(
        self,
        original_query: str,
        context: RetrievalContext
    ) -> List[str]:
        """
        生成查询变体
        
        触发条件:
        1. 原始查询过于简短 (< 10 字符)
        2. 原始查询包含模糊词汇
        3. 需要扩展专业术语
        
        返回:
            List[str]: 包含原始查询和变体的列表
        """
        queries = [original_query]
        
        # 判断是否需要改写
        if not self._needs_rewriting(original_query):
            return queries
        
        # 使用 LLM 生成变体
        prompt = f"""你是版图设计专家。请将以下查询扩展为 2-3 个更具体的搜索问题。

原始查询: {original_query}
当前任务: {context.task_type}
电路模块: {context.circuit_modules}

请生成与版图设计相关的查询变体，每行一个。"""

        response = self.llm.generate(prompt, max_tokens=200)
        variants = response.content.strip().split('\n')
        
        queries.extend([v.strip() for v in variants if v.strip()])
        
        return queries[:4]  # 最多返回 4 个查询
        
    def _needs_rewriting(self, query: str) -> bool:
        """判断是否需要改写"""
        if len(query) < 10:
            return True
        # 可添加更多判断逻辑
        return False
```

### 2.5.2 混合检索 (Hybrid Search)

```python
class HybridRetriever:
    """混合检索器"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        bm25_index: BM25Index
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.bm25 = bm25_index
        
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[RetrievalResult]:
        """
        混合检索
        
        策略:
        1. 向量检索: 语义相似度
        2. BM25 检索: 关键词匹配
        3. RRF 融合: 合并两个结果列表
        
        参数:
            query: 查询文本
            top_k: 返回数量
            vector_weight: 向量检索权重
            bm25_weight: BM25 检索权重
        """
        # 向量检索
        query_embedding = self.embedding_model.embed(query)
        vector_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k * 2  # 检索更多用于融合
        )
        
        # BM25 检索
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        
        # RRF 融合
        fused_results = self._rrf_fusion(
            vector_results,
            bm25_results,
            vector_weight,
            bm25_weight
        )
        
        return fused_results[:top_k]
        
    def _rrf_fusion(
        self,
        vector_results: List[QueryResult],
        bm25_results: List[QueryResult],
        vector_weight: float,
        bm25_weight: float,
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) 算法
        
        公式: RRF_score = Σ (weight / (k + rank))
        """
        scores = {}
        
        # 计算向量检索得分
        for rank, result in enumerate(vector_results):
            doc_id = result.chunk_id
            scores[doc_id] = scores.get(doc_id, 0) + \
                vector_weight / (k + rank + 1)
        
        # 计算 BM25 得分
        for rank, result in enumerate(bm25_results):
            doc_id = result.chunk_id
            scores[doc_id] = scores.get(doc_id, 0) + \
                bm25_weight / (k + rank + 1)
        
        # 按融合分数排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # 构建结果
        results = []
        for doc_id in sorted_ids:
            chunk = self._get_chunk_by_id(doc_id)
            results.append(RetrievalResult(
                chunk=chunk,
                score=scores[doc_id],
                retrieval_method="hybrid"
            ))
        
        return results
```

### 2.5.3 重排序 (Reranking)

```python
class Reranker:
    """重排序器"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)
        
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_n: int = 5
    ) -> List[RetrievalResult]:
        """
        使用交叉编码器重排序
        
        参数:
            query: 查询文本
            candidates: 候选结果列表 (Top-K from retrieval)
            top_n: 最终返回数量 (Top-N after reranking)
            
        返回:
            List[RetrievalResult]: 重排序后的结果
        """
        if len(candidates) <= top_n:
            return candidates
        
        # 构建 query-document pairs
        pairs = [(query, c.chunk.content) for c in candidates]
        
        # 计算相关性分数
        scores = self.model.predict(pairs)
        
        # 更新分数并排序
        for i, candidate in enumerate(candidates):
            candidate.rerank_score = float(scores[i])
        
        reranked = sorted(
            candidates,
            key=lambda x: x.rerank_score,
            reverse=True
        )
        
        return reranked[:top_n]
```

### 2.5.4 对外暴露接口规格

```python
@dataclass
class RetrievalContext:
    """检索上下文"""
    task_type: str                    # 任务类型: layout_planning, drc_check, etc.
    circuit_modules: List[str]        # 当前电路模块: diff_pair, current_mirror, etc.
    process_node: Optional[str] = None  # 工艺节点
    language_preference: str = "mixed"  # 语言偏好

@dataclass
class RetrievalResult:
    """检索结果"""
    chunk: Chunk                      # 检索到的 Chunk
    score: float                      # 检索分数
    rerank_score: Optional[float] = None  # 重排序分数
    retrieval_method: str = "vector"  # 检索方法

class RAGRetriever:
    """RAG 检索器 - 对外接口"""
    
    def retrieve_knowledge(
        self,
        query: str,
        context: RetrievalContext,
        top_k: int = 5,
        rerank: bool = True
    ) -> List[Chunk]:
        """
        检索相关知识
        
        参数:
            query: 查询文本
            context: 检索上下文，包含任务类型、电路模块等信息
            top_k: 返回的 Chunk 数量
            rerank: 是否启用重排序
            
        返回:
            List[Chunk]: 相关知识片段列表，按相关性排序
        """
        # 1. 查询改写
        queries = self.query_rewriter.rewrite(query, context)
        
        # 2. 构建过滤条件
        filters = self._build_filters(context)
        
        # 3. 对每个查询执行混合检索
        all_results = []
        for q in queries:
            results = self.hybrid_retriever.retrieve(
                query=q,
                top_k=top_k * 2,
                filters=filters
            )
            all_results.extend(results)
        
        # 4. 去重
        unique_results = self._deduplicate(all_results)
        
        # 5. 重排序
        if rerank and len(unique_results) > top_k:
            final_results = self.reranker.rerank(
                query=query,
                candidates=unique_results,
                top_n=top_k
            )
        else:
            final_results = unique_results[:top_k]
        
        return [r.chunk for r in final_results]
```

---

## 2.6 与 Agent 规划模块的集成方式

### 2.6.1 调用时机

| 调用时机 | 查询内容 | 预期返回 |
|----------|----------|----------|
| **规划前检索** | 根据电路类型和设计目标 | 整体布局策略、最佳实践 |
| **步骤规划时** | 针对特定步骤（如差分对布局） | 具体技巧、注意事项 |
| **遇到问题时** | 针对具体问题（如 DRC 违规） | 解决方案、类似案例 |

### 2.6.2 集成代码示例

```python
class PlanningModule:
    """规划模块与 RAG 的集成"""
    
    def create_initial_plan(
        self,
        context: TaskContext,
        memory: MemoryModule
    ) -> Plan:
        """生成初始规划（集成 RAG）"""
        
        # 1. 规划前检索 - 获取整体布局策略
        layout_strategy_chunks = self.rag.retrieve_knowledge(
            query=f"{context.circuit.type} 版图布局策略",
            context=RetrievalContext(
                task_type="layout_planning",
                circuit_modules=context.circuit.get_module_types(),
                process_node=context.drc_rules.tech
            ),
            top_k=3
        )
        
        # 2. 针对各模块检索具体技巧
        module_chunks = {}
        for module in context.circuit.modules:
            chunks = self.rag.retrieve_knowledge(
                query=f"{module.type} 版图技巧",
                context=RetrievalContext(
                    task_type="module_layout",
                    circuit_modules=[module.type]
                ),
                top_k=2
            )
            module_chunks[module.id] = chunks
        
        # 3. 格式化 RAG 结果用于 Prompt
        rag_context = self._format_rag_context(
            strategy_chunks=layout_strategy_chunks,
            module_chunks=module_chunks
        )
        
        # 4. 构建 Prompt 并调用 LLM
        prompt = self.prompt_templates.initial_plan.format(
            circuit_summary=context.circuit.summarize(),
            drc_rules_summary=context.drc_rules.summarize(),
            objectives_summary=context.objectives.summarize(),
            rag_context=rag_context,
            similar_cases=memory.get_similar_cases_summary()
        )
        
        response = self.llm.generate(prompt)
        return self._parse_plan(response.content)
```

### 2.6.3 返回格式约定

```python
@dataclass
class RAGResultForPlanning:
    """RAG 结果（用于规划模块）"""
    
    chunks: List[Chunk]               # 检索到的 Chunks
    max_chunks: int = 5               # 最大返回数量
    
    def format_for_prompt(self) -> str:
        """格式化为 Prompt 可用的文本"""
        formatted = []
        for i, chunk in enumerate(self.chunks[:self.max_chunks]):
            source = f"{chunk.metadata.doc_name} > {' > '.join(chunk.metadata.title_path)}"
            formatted.append(
                f"### 参考知识 {i+1}\n"
                f"来源: {source}\n"
                f"内容:\n{chunk.content}\n"
            )
        return "\n".join(formatted)
```

---

## 2.7 监控与评估

### 2.7.1 RAG 质量评估指标

| 指标 | 定义 | 计算方式 | 目标值 |
|------|------|----------|--------|
| **命中率 (Hit Rate)** | 检索结果中包含相关文档的比例 | 相关结果数 / 总查询数 | > 80% |
| **MRR (Mean Reciprocal Rank)** | 首个相关结果的排名倒数均值 | 1/rank 的均值 | > 0.6 |
| **用户纠正次数** | 用户手动补充或纠正知识的次数 | 统计用户反馈 | 越低越好 |
| **检索延迟** | 检索请求的响应时间 | P95 延迟 | < 500ms |

### 2.7.2 日志记录

```python
@dataclass
class RAGLogEntry:
    """RAG 日志条目"""
    timestamp: str
    session_id: str
    query: str
    rewritten_queries: List[str]
    retrieval_method: str
    top_k: int
    results_count: int
    top_result_score: float
    latency_ms: int
    filters_applied: Dict[str, Any]
    
class RAGLogger:
    """RAG 日志记录器"""
    
    def log_retrieval(
        self,
        query: str,
        context: RetrievalContext,
        results: List[RetrievalResult],
        latency_ms: int
    ):
        """记录检索日志"""
        entry = RAGLogEntry(
            timestamp=datetime.now().isoformat(),
            session_id=context.session_id,
            query=query,
            rewritten_queries=context.rewritten_queries,
            retrieval_method="hybrid",
            top_k=len(results),
            results_count=len(results),
            top_result_score=results[0].score if results else 0,
            latency_ms=latency_ms,
            filters_applied=context.filters
        )
        
        self.logger.info(entry.to_json())
```

### 2.7.3 人工抽查策略

```python
class RAGQualityChecker:
    """RAG 质量检查器"""
    
    def __init__(self, sample_rate: float = 0.1):
        self.sample_rate = sample_rate
        self.pending_reviews = []
        
    def maybe_queue_for_review(
        self,
        query: str,
        results: List[RetrievalResult],
        context: RetrievalContext
    ):
        """
        按采样率将检索结果加入人工审查队列
        
        优先审查:
        1. 低分结果 (top_score < 0.5)
        2. 空结果
        3. 新类型的查询
        """
        should_review = random.random() < self.sample_rate
        
        # 强制审查低质量结果
        if results and results[0].score < 0.5:
            should_review = True
        if not results:
            should_review = True
            
        if should_review:
            self.pending_reviews.append(ReviewItem(
                query=query,
                results=results,
                context=context,
                created_at=datetime.now()
            ))
            
    def export_for_review(self) -> List[ReviewItem]:
        """导出待审查项目"""
        items = self.pending_reviews.copy()
        self.pending_reviews.clear()
        return items
```

---

## 附录 C：配置项汇总

```yaml
# rag_config.yaml
rag:
  # 嵌入模型配置
  embedding:
    provider: "openai"           # openai, huggingface
    model: "text-embedding-3-small"
    dimension: 1536
    batch_size: 32
    
  # 向量数据库配置
  vector_store:
    provider: "chroma"
    persist_directory: "./data/vectordb"
    collection_name: "layout_knowledge"
    distance_metric: "cosine"
    
  # 切片配置
  chunking:
    min_tokens: 100
    max_tokens: 512
    target_tokens: 300
    overlap_tokens: 50
    
  # 检索配置
  retrieval:
    default_top_k: 5
    vector_weight: 0.7
    bm25_weight: 0.3
    enable_reranking: true
    reranker_model: "BAAI/bge-reranker-base"
    
  # 查询改写配置
  query_rewriting:
    enabled: true
    max_variants: 3
    min_query_length: 10
```
