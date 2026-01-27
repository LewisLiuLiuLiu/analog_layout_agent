# gdsfactory 降级分析报告

## 一、需要降级的组件清单

### 1. 核心组件

| 组件 | 当前版本 (9.x) | 目标版本 (7.7.0) | 降级必要性 |
|------|----------------|------------------|------------|
| **gdsfactory** | 9.31.0 | 7.7.0 | ✅ **必须** - glayout 明确要求 `<=7.7.0` |

### 2. 直接依赖变更

| 依赖包 | 9.x 要求 | 7.7.0 要求 | 降级影响 |
|--------|----------|------------|----------|
| **pydantic** | `>=2` | `>=2,<3` | ⚠️ 兼容 - 两者都使用 Pydantic v2 |
| **kfactory** | `>=2.2,<2.3` | `>=0.8.4,<0.9` (cad extra) | 🔴 **重大差异** - 需要降级 |
| **numpy** | 无限制 | 无限制 | ⚠️ glayout 要求 `<=1.24.0` |
| **gdstk** | 未声明 | `<1` | ✅ 需确保兼容 |
| **klayout** | 未直接依赖 | 未直接依赖 | ⚠️ glayout 要求 `>0.28.0,<=0.29` |

### 3. glayout 特有依赖约束

| 依赖包 | glayout 要求 | 说明 |
|--------|--------------|------|
| **numpy** | `>1.21.0,<=1.24.0` | 与部分新版包可能冲突 |
| **pandas** | `>1.3.0,<=2.3.0` | 范围较宽，一般兼容 |
| **matplotlib** | `>3.4.0,<=3.10.0` | 范围较宽，一般兼容 |
| **klayout** | `>0.28.0,<=0.29` | 需要特定版本 |

### 4. 完整降级组件清单

```
# 必须降级的核心包
gdsfactory==7.7.0

# 可能需要降级的依赖包（按优先级排序）
numpy>=1.21.0,<=1.24.0      # glayout 硬性要求
kfactory>=0.8.4,<0.9        # gdsfactory 7.7.0 [cad] 依赖
klayout>0.28.0,<=0.29       # glayout 硬性要求

# 需要保持兼容的包
pydantic>=2,<3              # 两版本共同要求
gdstk<1                     # gdsfactory 7.7.0 要求
shapely<3                   # 两版本共同要求
```

### 5. 依赖版本对照表

| 包名 | gdsfactory 9.31.0 | gdsfactory 7.7.0 | glayout 要求 | 建议安装版本 |
|------|-------------------|------------------|--------------|--------------|
| gdsfactory | 9.31.0 | - | `>6.0.0,<=7.7.0` | **7.7.0** |
| numpy | 无限制 | 无限制 | `>1.21.0,<=1.24.0` | **1.24.0** |
| pydantic | `>=2` | `>=2,<3` | - | **2.5.x** |
| kfactory | `>=2.2,<2.3` | `>=0.8.4,<0.9` | - | **0.8.9** |
| klayout | - | - | `>0.28.0,<=0.29` | **0.29.0** |
| pandas | 无限制 | 无限制 | `>1.3.0,<=2.3.0` | **2.0.x** |
| matplotlib | `<4` | 无限制 | `>3.4.0,<=3.10.0` | **3.8.x** |
| gdstk | - | `<1` | 无限制 | **0.9.x** |
| scipy | `<2` | 无限制 | - | **1.11.x** |
| shapely | `<3` | `<3` | - | **2.0.x** |

---

## 二、降级过程中可能出现的问题清单

### 🔴 高风险问题

#### 问题 1：kfactory 版本冲突

| 项目 | 详情 |
|------|------|
| **问题描述** | gdsfactory 9.x 依赖 kfactory 2.x，而 7.7.0 依赖 kfactory 0.8.x |
| **影响范围** | Component 底层实现、布局引擎、GDS 读写 |
| **错误表现** | `ImportError` 或 `AttributeError` 在导入 gdsfactory 时 |
| **预防措施** | 先卸载 kfactory，再安装 gdsfactory 7.7.0 |
| **解决方案** | `pip uninstall kfactory -y && pip install "gdsfactory[cad]==7.7.0"` |

#### 问题 2：numpy 版本与其他包冲突

| 项目 | 详情 |
|------|------|
| **问题描述** | glayout 要求 numpy<=1.24.0，但部分新版科学计算包需要更高版本 |
| **影响范围** | scipy、scikit-image、pandas 等依赖 numpy 的包 |
| **错误表现** | `ModuleNotFoundError` 或 numpy API 不兼容错误 |
| **预防措施** | 先安装 numpy 1.24.0，再安装其他包 |
| **解决方案** | `pip install "numpy==1.24.0" --force-reinstall` |

#### 问题 3：Pydantic v1 vs v2 混用

| 项目 | 详情 |
|------|------|
| **问题描述** | 虽然 gdsfactory 7.7.0 支持 Pydantic v2，但部分旧代码可能使用 v1 语法 |
| **影响范围** | glayout 的 MappedPDK 类、验证器 |
| **错误表现** | `ValidationError`、`ConfigError`、`@validator` 警告 |
| **预防措施** | 检查代码中的 pydantic 用法 |
| **解决方案** | 如有问题可尝试 `pip install "pydantic>=1.10,<2.0"` |

### 🟡 中等风险问题

#### 问题 4：klayout Python 绑定版本

| 项目 | 详情 |
|------|------|
| **问题描述** | glayout 要求 klayout 0.28-0.29，系统可能已安装其他版本 |
| **影响范围** | DRC 检查、GDS 查看、布局操作 |
| **错误表现** | `klayout.db` 模块导入失败或 API 不兼容 |
| **预防措施** | 确认当前 klayout 版本 |
| **解决方案** | `pip install "klayout==0.29.0"` |

#### 问题 5：gdstk 版本兼容性

| 项目 | 详情 |
|------|------|
| **问题描述** | gdsfactory 7.7.0 要求 gdstk<1，可能与现有版本冲突 |
| **影响范围** | 多边形操作、GDS 文件读写 |
| **错误表现** | `TypeError` 或 API 参数不匹配 |
| **预防措施** | 先卸载再安装指定版本 |
| **解决方案** | `pip install "gdstk<1"` |

#### 问题 6：依赖解析循环

| 项目 | 详情 |
|------|------|
| **问题描述** | pip 可能无法自动解析满足所有约束的版本组合 |
| **影响范围** | 整个安装过程 |
| **错误表现** | `ResolutionImpossible` 错误 |
| **预防措施** | 使用 `--no-deps` 分步安装 |
| **解决方案** | 见下方分步安装脚本 |

### 🟢 低风险问题

#### 问题 7：缓存污染

| 项目 | 详情 |
|------|------|
| **问题描述** | pip 缓存可能导致安装旧版本或错误版本 |
| **影响范围** | 安装过程 |
| **错误表现** | 安装后版本不符合预期 |
| **预防措施** | 使用 `--no-cache-dir` 选项 |
| **解决方案** | `pip install --no-cache-dir gdsfactory==7.7.0` |

#### 问题 8：Python 版本不兼容

| 项目 | 详情 |
|------|------|
| **问题描述** | gdsfactory 7.7.0 支持 Python 3.10/3.11，9.x 支持 3.11/3.12/3.13 |
| **影响范围** | 当前环境 (Python 3.11) |
| **错误表现** | 无（Python 3.11 兼容两者） |
| **预防措施** | 确认 Python 版本为 3.10 或 3.11 |
| **解决方案** | 当前环境已满足要求 |

#### 问题 9：IDE/编辑器类型提示失效

| 项目 | 详情 |
|------|------|
| **问题描述** | 降级后 IDE 的自动补全和类型检查可能基于旧 API |
| **影响范围** | 开发体验 |
| **错误表现** | 类型错误警告、自动补全不准确 |
| **预防措施** | 重启 IDE，清除缓存 |
| **解决方案** | 重建项目索引 |

#### 问题 10：测试失败

| 项目 | 详情 |
|------|------|
| **问题描述** | 现有测试可能依赖 gdsfactory 9.x 特有功能 |
| **影响范围** | analog_layout_agent 测试 |
| **错误表现** | pytest 测试失败 |
| **预防措施** | 降级前记录当前测试状态 |
| **解决方案** | 根据错误修改测试或暂时跳过 |

---

## 三、安全降级操作脚本

基于以上分析，提供一个更安全的分步降级脚本：

```bash
#!/bin/bash
# safe_downgrade.sh - 安全降级脚本

set -e

echo "=============================================="
echo "  gdsfactory 安全降级脚本 (9.x → 7.7.0)"
echo "=============================================="

cd /mnt/d/qoderProjects/layout-agent-demo
source venv311/bin/activate

# Step 1: 备份
echo ""
echo "[Step 1/8] 备份当前环境..."
BACKUP_FILE="requirements_backup_$(date +%Y%m%d_%H%M%S).txt"
pip freeze > "$BACKUP_FILE"
echo "  备份文件: $BACKUP_FILE"

# Step 2: 卸载冲突包
echo ""
echo "[Step 2/8] 卸载可能冲突的包..."
pip uninstall gdsfactory kfactory -y 2>/dev/null || true

# Step 3: 安装 numpy (glayout 约束)
echo ""
echo "[Step 3/8] 安装指定版本 numpy..."
pip install "numpy==1.24.0" --no-cache-dir

# Step 4: 安装 klayout (glayout 约束)
echo ""
echo "[Step 4/8] 安装指定版本 klayout..."
pip install "klayout==0.29.0" --no-cache-dir

# Step 5: 安装 gdsfactory 7.7.0 (包含 cad 依赖)
echo ""
echo "[Step 5/8] 安装 gdsfactory 7.7.0..."
pip install "gdsfactory[cad]==7.7.0" --no-cache-dir

# Step 6: 重新安装 glayout
echo ""
echo "[Step 6/8] 重新安装 glayout..."
cd gLayout
pip install -e . --no-deps  # 先不安装依赖，避免覆盖
pip install -e .            # 再正常安装
cd ..

# Step 7: 验证版本
echo ""
echo "[Step 7/8] 验证安装版本..."
echo "  gdsfactory: $(python3 -c 'import gdsfactory; print(gdsfactory.__version__)')"
echo "  numpy: $(python3 -c 'import numpy; print(numpy.__version__)')"
echo "  klayout: $(python3 -c 'import klayout; print(klayout.__version__)')"

# Step 8: 功能测试
echo ""
echo "[Step 8/8] 基础功能测试..."
python3 -c "
from gdsfactory.component import Component
from gdsfactory.pdk import Pdk
print('  ✓ gdsfactory 核心模块导入成功')

try:
    from glayout.pdk.mappedpdk import MappedPDK
    print('  ✓ glayout MappedPDK 导入成功')
except Exception as e:
    print(f'  ✗ glayout 导入失败: {e}')

try:
    from glayout.primitives.fet import nmos, pmos
    print('  ✓ glayout primitives 导入成功')
except Exception as e:
    print(f'  ✗ glayout primitives 导入失败: {e}')
"

echo ""
echo "=============================================="
echo "  降级完成！"
echo "=============================================="
echo ""
echo "回滚命令: pip install -r $BACKUP_FILE"
echo ""
```

---

## 四、问题排查快速参考

| 错误类型 | 可能原因 | 快速修复 |
|----------|----------|----------|
| `ModuleNotFoundError: kfactory` | kfactory 未安装或版本错误 | `pip install "kfactory>=0.8.4,<0.9"` |
| `ImportError: cannot import Component` | gdsfactory 版本错误 | `pip show gdsfactory` 确认版本 |
| `numpy.core.multiarray failed to import` | numpy 版本不兼容 | `pip install "numpy==1.24.0"` |
| `pydantic.error_wrappers.ValidationError` | Pydantic 版本问题 | 检查是否混用 v1/v2 语法 |
| `AttributeError: module 'klayout' has no attribute` | klayout 版本不对 | `pip install "klayout==0.29.0"` |
| `ResolutionImpossible` | 依赖冲突 | 使用 `--no-deps` 分步安装 |

---

## 五、快速执行命令汇总

```bash
# === 完整降级流程（复制粘贴即可执行） ===

# 1. 进入项目目录并激活环境
cd /mnt/d/qoderProjects/layout-agent-demo
source venv311/bin/activate

# 2. 备份当前环境
pip freeze > requirements_backup_$(date +%Y%m%d_%H%M%S).txt

# 3. 执行降级
pip uninstall gdsfactory kfactory -y
pip install "numpy==1.24.0" --no-cache-dir
pip install "klayout==0.29.0" --no-cache-dir
pip install "gdsfactory[cad]==7.7.0" --no-cache-dir

# 4. 重新安装 glayout
cd gLayout && pip install -e . && cd ..

# 5. 验证
python3 -c "import gdsfactory; print(f'版本: {gdsfactory.__version__}')"
python3 -c "from glayout.pdk.mappedpdk import MappedPDK; print('glayout 导入成功')"
```

---

## 六、回滚方案

如果降级后出现问题，可快速恢复：

### 方案 A：使用备份文件恢复

```bash
# 激活虚拟环境
source venv311/bin/activate

# 使用备份文件恢复（替换为实际备份文件名）
pip install -r requirements_backup_YYYYMMDD_HHMMSS.txt
```

### 方案 B：手动恢复到原版本

```bash
# 激活虚拟环境
source venv311/bin/activate

# 卸载当前版本
pip uninstall gdsfactory -y

# 重新安装原版本
pip install "gdsfactory==9.31.0"
```

### 方案 C：重建虚拟环境（完全重置）

```bash
# 删除当前虚拟环境
rm -rf venv311

# 重新创建虚拟环境
python3.11 -m venv venv311

# 激活并安装依赖
source venv311/bin/activate
pip install --upgrade pip

# 重新安装项目依赖
pip install "gdsfactory==9.31.0"  # 或其他版本
cd gLayout && pip install -e . && cd ..
```

---

## 七、注意事项

| 事项 | 说明 |
|------|------|
| **Pydantic 版本** | gdsfactory 7.7.0 使用 Pydantic v2，如有冲突需检查代码 |
| **numpy 版本** | glayout 要求 `numpy>1.21.0,<=1.24.0`，注意兼容性 |
| **测试覆盖** | 降级后务必运行完整测试套件 |
| **文档参考** | 使用 gdsfactory 7.x 版本的文档，而非最新文档 |
| **Python 版本** | 确保使用 Python 3.10 或 3.11 |

---

*报告生成时间: 2026-01-23*
*适用项目: layout-agent-demo*
