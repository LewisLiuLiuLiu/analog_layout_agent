# 第7章 部署与配置

---

## 7.1 部署目标与运行形态假设

### 7.1.1 运行形态

系统支持两种运行形态：

| 形态 | 适用场景 | 架构特点 |
|------|----------|----------|
| **本地开发环境** | 开发、调试、单用户使用 | 单机运行，所有组件在同一进程或本地服务 |
| **生产环境** | 多用户、高可用、可扩展 | 服务化部署，组件可独立扩展 |

### 7.1.2 本地开发环境架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         本地开发环境                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Python 进程                                   │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │                 Agent Orchestrator                         │  │   │
│  │  │  ├── Perception Module                                     │  │   │
│  │  │  ├── Planning Module ──────────────► LLM API (远程)        │  │   │
│  │  │  ├── Action Module                                         │  │   │
│  │  │  ├── Memory Module                                         │  │   │
│  │  │  ├── Evaluation Module                                     │  │   │
│  │  │  └── Skill Registry                                        │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  │                              │                                   │   │
│  │                              ▼                                   │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │              Chroma (内嵌模式)                             │  │   │
│  │  │              向量数据库                                    │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              │ stdio/subprocess                         │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    KLayout MCP Server                           │   │
│  │                    (子进程)                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.1.3 生产环境架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           生产环境                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │ Orchestrator     │  │ Orchestrator     │  │ Orchestrator     │      │
│  │ Instance 1       │  │ Instance 2       │  │ Instance N       │      │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘      │
│           │                     │                     │                 │
│           └──────────────┬──────┴─────────────────────┘                 │
│                          │                                              │
│           ┌──────────────┼──────────────────────────┐                  │
│           ▼              ▼                          ▼                  │
│  ┌──────────────┐  ┌──────────────┐       ┌──────────────┐            │
│  │ KLayout MCP  │  │ Vector DB    │       │ LLM API      │            │
│  │ Server Pool  │  │ (Qdrant/     │       │ (OpenAI/     │            │
│  │              │  │  Pinecone)   │       │  Claude)     │            │
│  └──────────────┘  └──────────────┘       └──────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.1.4 主要组件说明

| 组件 | 开发环境 | 生产环境 |
|------|----------|----------|
| **Orchestrator** | 单进程 Python 应用 | 多实例，负载均衡 |
| **KLayout MCP Server** | 子进程，stdio 通信 | 独立服务池，HTTP/gRPC |
| **向量数据库** | Chroma 内嵌模式 | Qdrant/Pinecone 集群 |
| **LLM API** | OpenAI/Claude API | 同上（可选私有部署） |
| **配置管理** | 本地文件 | 环境变量 + 配置中心 |
| **日志存储** | 本地文件 | 集中式日志（ELK/Loki） |

---

## 7.2 环境要求

### 7.2.1 操作系统

| 系统 | 支持状态 | 备注 |
|------|----------|------|
| macOS 12+ (Monterey) | 完全支持 | 推荐开发环境 |
| Ubuntu 20.04/22.04 | 完全支持 | 推荐生产环境 |
| Windows 10/11 (WSL2) | 部分支持 | 通过 WSL2 运行 |
| CentOS/RHEL 8+ | 支持 | 需额外配置 |

### 7.2.2 编程语言与运行时

| 依赖 | 版本要求 | 用途 |
|------|----------|------|
| Python | 3.10+ | 主程序运行时 |
| pip | 最新版 | 包管理 |
| Node.js | 18+ (可选) | MCP Server 备选实现 |

### 7.2.3 必需的第三方软件

#### KLayout

| 软件 | 版本要求 | 用途 | 安装方式 |
|------|----------|------|----------|
| KLayout | 0.28+ | 版图编辑引擎 | 官方安装包 |

**验证安装**:
```bash
klayout -v
# 预期输出: KLayout 0.28.x
```

#### 向量数据库

| 软件 | 版本要求 | 用途 | 模式 |
|------|----------|------|------|
| Chroma | 0.4+ | 开发环境向量存储 | 内嵌 |
| Qdrant | 1.6+ (可选) | 生产环境向量存储 | Docker |

### 7.2.4 硬件资源建议

| 环境 | CPU | 内存 | 磁盘 | GPU |
|------|-----|------|------|-----|
| 开发环境 | 4 核+ | 16GB+ | 50GB SSD | 不需要 |
| 生产环境 | 8 核+ | 32GB+ | 200GB SSD | 可选 |

---

## 7.3 依赖安装步骤

### 7.3.1 基础环境安装

#### Step 1: 安装 Python

**macOS (Homebrew)**:
```bash
# 安装 Homebrew (如果没有)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 Python 3.11
brew install python@3.11

# 验证
python3.11 --version
```

**Ubuntu**:
```bash
# 更新包列表
sudo apt update

# 安装 Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev

# 验证
python3.11 --version
```

#### Step 2: 创建虚拟环境

```bash
# 创建项目目录
mkdir -p ~/opamp_layout_agent
cd ~/opamp_layout_agent

# 创建虚拟环境
python3.11 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows

# 升级 pip
pip install --upgrade pip
```

### 7.3.2 安装 KLayout

**macOS**:
```bash
# 下载 DMG 安装包
# https://www.klayout.de/build.html

# 或使用 Homebrew Cask
brew install --cask klayout

# 验证
klayout -v
```

**Ubuntu**:
```bash
# 添加 KLayout PPA
sudo add-apt-repository ppa:nicola-mfb/klayout

# 安装
sudo apt update
sudo apt install klayout

# 验证
klayout -v
```

**验证 Python API**:
```bash
# KLayout Python API 通常随安装包提供
# 需要确保 klayout 模块可被 Python 导入

# 测试方式 1: 使用 KLayout 内置 Python
klayout -b -r - <<< "import pya; print(pya.Application.instance())"

# 测试方式 2: 如果使用系统 Python，需要设置 PYTHONPATH
export KLAYOUT_PATH=$(which klayout)
# 具体路径因安装方式而异
```

### 7.3.3 安装 Python 依赖

创建 `requirements.txt`:

```
# 核心依赖
pydantic>=2.0.0
openai>=1.0.0
anthropic>=0.5.0

# RAG 相关
chromadb>=0.4.0
sentence-transformers>=2.2.0

# MCP 相关
mcp>=0.1.0

# 工具库
jsonschema>=4.0.0
tiktoken>=0.5.0
numpy>=1.24.0

# 日志与监控
structlog>=23.0.0

# 测试
pytest>=7.0.0
pytest-asyncio>=0.21.0

# 开发工具
black>=23.0.0
ruff>=0.1.0
mypy>=1.0.0
```

安装命令:
```bash
pip install -r requirements.txt
```

### 7.3.4 安装向量数据库 (可选生产环境)

**Qdrant (Docker)**:
```bash
# 拉取镜像
docker pull qdrant/qdrant

# 运行容器
docker run -d \
    --name qdrant \
    -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

# 验证
curl http://localhost:6333/collections
```

---

## 7.4 配置管理

### 7.4.1 配置文件结构

```yaml
# config/config.yaml

# 应用基础配置
app:
  name: "opamp-layout-agent"
  environment: "development"  # development, staging, production
  log_level: "INFO"

# LLM 配置
llm:
  provider: "openai"  # openai, anthropic
  model: "gpt-4"
  # api_key: 通过环境变量 OPENAI_API_KEY 设置
  max_tokens: 4096
  temperature: 0.1
  timeout_seconds: 60

# MCP 配置
mcp:
  transport: "stdio"  # stdio, http
  klayout:
    executable: "/usr/local/bin/klayout"
    server_script: "./mcp_server/klayout_mcp.py"
    timeout_ms: 30000

# RAG 配置
rag:
  embedding:
    provider: "openai"  # openai, huggingface
    model: "text-embedding-3-small"
    dimension: 1536
  vector_store:
    provider: "chroma"  # chroma, qdrant, pinecone
    persist_directory: "./data/vectordb"
    collection_name: "layout_knowledge"
  retrieval:
    default_top_k: 5
    enable_reranking: true

# Memory 配置
memory:
  short_term:
    max_steps: 50
  long_term:
    persist_directory: "./data/memory"
    max_cases: 1000

# 数据目录
data:
  knowledge_docs: "./data/knowledge"
  output_dir: "./output"
  temp_dir: "./tmp"

# Agent 运行配置
agent:
  max_iterations: 5
  step_timeout_ms: 60000
  session_timeout_ms: 600000
```

### 7.4.2 环境变量

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `OPENAI_API_KEY` | 是* | - | OpenAI API Key |
| `ANTHROPIC_API_KEY` | 否 | - | Anthropic API Key |
| `KLAYOUT_PATH` | 否 | `klayout` | KLayout 可执行文件路径 |
| `CONFIG_PATH` | 否 | `./config/config.yaml` | 配置文件路径 |
| `LOG_LEVEL` | 否 | `INFO` | 日志级别 |
| `DATA_DIR` | 否 | `./data` | 数据目录 |

### 7.4.3 配置加载优先级

```
1. 环境变量 (最高优先级)
2. 命令行参数
3. 配置文件
4. 默认值 (最低优先级)
```

### 7.4.4 多环境配置

```
config/
├── config.yaml           # 基础配置
├── config.development.yaml  # 开发环境覆盖
├── config.staging.yaml      # 测试环境覆盖
└── config.production.yaml   # 生产环境覆盖
```

**开发环境覆盖示例** (`config.development.yaml`):
```yaml
app:
  environment: "development"
  log_level: "DEBUG"

llm:
  model: "gpt-3.5-turbo"  # 开发时使用较便宜的模型

rag:
  vector_store:
    provider: "chroma"
    persist_directory: "./data/dev/vectordb"
```

### 7.4.5 配置加载代码

```python
import os
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """应用配置"""
    # ... 配置字段定义
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """
        加载配置
        
        优先级: 环境变量 > 命令行 > 配置文件 > 默认值
        """
        # 确定配置文件路径
        if config_path is None:
            config_path = os.environ.get('CONFIG_PATH', './config/config.yaml')
            
        # 加载基础配置
        with open(config_path, 'r') as f:
            base_config = yaml.safe_load(f)
            
        # 加载环境特定配置
        env = os.environ.get('APP_ENV', 'development')
        env_config_path = config_path.replace('.yaml', f'.{env}.yaml')
        if os.path.exists(env_config_path):
            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)
            base_config = cls._merge_configs(base_config, env_config)
            
        # 应用环境变量覆盖
        config = cls._apply_env_overrides(base_config)
        
        return cls(**config)
        
    @staticmethod
    def _merge_configs(base: dict, override: dict) -> dict:
        """深度合并配置"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
```

---

## 7.5 服务启动与运行

### 7.5.1 本地开发环境启动流程

#### 完整启动脚本

```bash
#!/bin/bash
# scripts/start_dev.sh

set -e

echo "=== Starting OpAmp Layout Agent (Development) ==="

# 1. 检查环境
echo "[1/5] Checking environment..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    exit 1
fi

# 2. 激活虚拟环境
echo "[2/5] Activating virtual environment..."
source .venv/bin/activate

# 3. 检查 KLayout
echo "[3/5] Checking KLayout..."
if ! command -v klayout &> /dev/null; then
    echo "Error: KLayout not found"
    exit 1
fi
klayout -v

# 4. 初始化向量数据库 (如果需要)
echo "[4/5] Initializing vector database..."
if [ ! -d "data/vectordb" ]; then
    python scripts/init_vectordb.py
fi

# 5. 启动 Orchestrator
echo "[5/5] Starting Orchestrator..."
python -m opamp_agent.main \
    --config config/config.yaml \
    --log-level DEBUG

echo "=== Agent Started ==="
```

#### 分步启动

**Step 1: 启动向量数据库服务 (如果使用独立服务)**:
```bash
# Chroma 内嵌模式不需要单独启动

# Qdrant Docker 模式
docker start qdrant
```

**Step 2: 启动 KLayout MCP Server (如果使用独立服务模式)**:
```bash
# stdio 模式由 Orchestrator 自动启动

# HTTP 模式
python mcp_server/klayout_mcp.py --port 8080
```

**Step 3: 启动 Orchestrator**:
```bash
python -m opamp_agent.main --config config/config.yaml
```

### 7.5.2 健康检查

#### Orchestrator 健康检查

```python
# 健康检查端点示例 (如果提供 HTTP 接口)
@app.get("/health")
def health_check():
    """健康检查"""
    checks = {
        "llm": check_llm_connection(),
        "vector_db": check_vector_db(),
        "klayout_mcp": check_mcp_server()
    }
    
    all_healthy = all(c["status"] == "healthy" for c in checks.values())
    
    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "checks": checks
    }
```

#### 命令行健康检查

```bash
#!/bin/bash
# scripts/health_check.sh

echo "Checking components..."

# 检查 LLM API
echo -n "LLM API: "
curl -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    | grep -q "200" && echo "OK" || echo "FAILED"

# 检查向量数据库
echo -n "Vector DB: "
python -c "import chromadb; chromadb.Client().heartbeat()" 2>/dev/null \
    && echo "OK" || echo "FAILED"

# 检查 KLayout
echo -n "KLayout: "
klayout -v &>/dev/null && echo "OK" || echo "FAILED"
```

### 7.5.3 简单请求示例

```python
# 命令行测试
python -m opamp_agent.cli run \
    --netlist examples/diff_amp.json \
    --drc-rules examples/drc_rules.json \
    --objectives examples/objectives.json \
    --output output/result.gds
```

---

## 7.6 日志与持久化

### 7.6.1 日志配置

```yaml
# config/logging.yaml

version: 1
disable_existing_loggers: false

formatters:
  json:
    class: structlog.stdlib.ProcessorFormatter
    processor: structlog.processors.JSONRenderer
  console:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    formatter: console
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    formatter: json
    filename: logs/agent.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    encoding: utf-8

loggers:
  opamp_agent:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console]
```

### 7.6.2 日志输出位置

| 日志类型 | 位置 | 保留策略 |
|----------|------|----------|
| 应用日志 | `logs/agent.log` | 10MB 单文件，保留 5 个 |
| 请求日志 | `logs/requests.log` | 按天轮转，保留 30 天 |
| 错误日志 | `logs/errors.log` | 按天轮转，保留 90 天 |
| 审计日志 | `logs/audit.log` | 按月归档，永久保留 |

### 7.6.3 数据持久化目录

```
data/
├── vectordb/              # 向量数据库存储
│   └── chroma.sqlite3
├── memory/                # 长期记忆存储
│   ├── cases/            # 设计案例
│   └── failures/         # 失败案例
├── knowledge/             # 知识库文档
│   └── *.md
└── cache/                 # 缓存数据
    └── embeddings/

output/                    # 输出目录
├── layouts/              # 生成的版图
│   └── *.gds
├── reports/              # 评估报告
│   └── *.json
└── logs/                 # 运行日志
```

### 7.6.4 备份策略

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/backup/opamp_agent/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# 备份向量数据库
cp -r data/vectordb $BACKUP_DIR/

# 备份长期记忆
cp -r data/memory $BACKUP_DIR/

# 备份配置
cp -r config $BACKUP_DIR/

# 压缩
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

---

## 7.7 安全与访问控制

### 7.7.1 API Key 安全

| 措施 | 说明 |
|------|------|
| **环境变量存储** | 不在代码或配置文件中硬编码 API Key |
| **最小权限原则** | 使用专用 API Key，限制权限范围 |
| **密钥轮换** | 定期更换 API Key |
| **日志脱敏** | 日志中不记录完整 API Key |

```python
# 日志脱敏示例
def mask_api_key(key: str) -> str:
    """遮蔽 API Key"""
    if len(key) <= 8:
        return "***"
    return key[:4] + "***" + key[-4:]
```

### 7.7.2 网络安全

| 端口 | 服务 | 建议 |
|------|------|------|
| 8080 | KLayout MCP (HTTP) | 仅本地监听，不暴露公网 |
| 6333 | Qdrant | 仅内网访问，添加认证 |
| 443 | LLM API | 使用 HTTPS |

**防火墙规则示例**:
```bash
# 仅允许本地访问 MCP 服务
iptables -A INPUT -p tcp --dport 8080 -s 127.0.0.1 -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -j DROP
```

### 7.7.3 文件权限

```bash
# 配置文件权限
chmod 600 config/config.yaml
chmod 700 config/

# 数据目录权限
chmod 750 data/
chmod 700 data/memory/

# 日志目录权限
chmod 750 logs/
```

---

## 7.8 CI 环境最小依赖集

### 7.8.1 CI 测试环境配置

```yaml
# .github/workflows/test.yml (示例)

name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install KLayout
      run: |
        sudo add-apt-repository ppa:nicola-mfb/klayout
        sudo apt update
        sudo apt install klayout
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        
    - name: Run unit tests
      run: |
        pytest tests/unit -v
        
    - name: Run integration tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        pytest tests/integration -v --tb=short
```

### 7.8.2 最小依赖集

| 依赖 | 单元测试 | 集成测试 | 说明 |
|------|----------|----------|------|
| Python 3.11 | 必需 | 必需 | 运行时 |
| KLayout | 可选 | 必需 | Mock 可替代单元测试 |
| Chroma | 可选 | 必需 | 内嵌模式，无需单独服务 |
| OpenAI API | 不需要 | 必需 | 可使用 Mock |

### 7.8.3 Mock 配置

```python
# tests/conftest.py

import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_llm_client():
    """Mock LLM 客户端"""
    client = MagicMock()
    client.generate.return_value = MagicMock(
        content='{"steps": [{"name": "test", "skill_name": "test"}]}'
    )
    return client

@pytest.fixture
def mock_mcp_client():
    """Mock MCP 客户端"""
    client = MagicMock()
    client.call_tool.return_value = ToolCallResult(
        ok=True,
        error=None,
        data={"cell_name": "test_cell"}
    )
    return client
```

---

## 附录 F：快速启动指南

### 一键安装脚本

```bash
#!/bin/bash
# scripts/quick_start.sh

set -e

echo "=== OpAmp Layout Agent Quick Start ==="

# 检查 Python
python3 --version || { echo "Python 3 required"; exit 1; }

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 检查 API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Please set OPENAI_API_KEY environment variable"
    echo "export OPENAI_API_KEY=your_key_here"
    exit 1
fi

# 初始化数据目录
mkdir -p data/{vectordb,memory,knowledge}
mkdir -p output/{layouts,reports}
mkdir -p logs

# 运行简单测试
python -c "from opamp_agent import __version__; print(f'Version: {__version__}')"

echo "=== Setup Complete ==="
echo "Run: python -m opamp_agent.main --help"
```
