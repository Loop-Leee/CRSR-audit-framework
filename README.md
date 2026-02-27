# CRSR-audit-framework
基于 LLM 的合同审查通用方法框架。


# 整体工作流程
```text
Contract → Chunk
         ↓
Classification
         ↓
Rule Subset Selection
         ↓
LLM Structured Audit (Schema constrained)
         ↓
Rule-hit linking
         ↓
Global Consistency Reflection
         ↓
Final Risk Report + Word Annotation
```

# 运行方式
1. chunking 模块
```bash
python3 main.py --chunk-size 1500
```

# 模块说明

## 目录结构
```text
src/chunking/
├── chunking_cli.py
main.py
```

## 模块功能
- chunking 模块：将 Word 文档切分为多个块，可自定义切分大小。

# 依赖说明

- 使用 conda 创建隔离环境（推荐）：
```bash
conda env create -f environment.yml
conda activate crsr-audit
```
- 当前实现依赖 macOS 系统 `textutil` 命令解析 Word 文本。
