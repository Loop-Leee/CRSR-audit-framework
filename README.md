# CRSR-audit-framework

基于 LLM 的合同审查通用方法框架。

## 整体工作流程

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

## 环境准备

```bash
conda env create -f environment.yml
conda activate crsr-audit
```

当前实现依赖 macOS 系统 `textutil` 命令解析 Word 文本。

## 运行方式

1. `chunking`（Word 分块）

```bash
python3 main.py --chunk-size 1000
```

2. `classification`（chunk 风险类型分类）

```bash
python3 main.py classification
```

可选参数：

```bash
python3 main.py classification --input data/2-chunks --output data/3-classified
python3 main.py classification --disable-llm
python3 main.py classification --llm-config src/llm/llm_config.json
```

## 模块说明

### chunking

- 目录：`src/chunking/`
- 输入：`data/1-original/*.doc|*.docx`
- 输出：`data/2-chunks/*.chunks.json`
- 说明文档：`src/chunking/README.md`

### classification

- 目录：`src/classification/`
- 输入：`data/2-chunks/*.chunks.json`
- 输出：`data/3-classified/*.classified.json`
- 规则源：`prompt/risk_info.csv`
- 分类策略：关键词匹配 + LLM 语义匹配，结果取并集
- 说明文档：`src/classification/README.md`

输出文件中的每个 chunk 会新增字段：

```json
{
  "chunk_id": 1,
  "content": "...",
  "risk_type": ["付款时间审查", "发票审查"]
}
```

## LLM 配置

项目统一通过 `src/llm/` 模块管理 LLM 调用与配置：

- 全局配置：`src/llm/llm_config.json`
- 配置加载：`src/llm/config.py`
- OpenAI 兼容客户端：`src/llm/openai_client.py`

读取环境变量：

- `LLM_API_KEY`
- `LLM_BASE_URL`（可选，默认 `https://api.openai.com/v1`，也支持直接填写 `.../v1/chat/completions`）
- `LLM_MODEL`

运行时会自动加载项目根目录 `.env`。`LLM_API_KEY` 可留空（无鉴权部署场景）。
如果 LLM 不可用（模型缺失、调用失败、返回格式错误），会自动降级为仅关键词匹配，并在日志中记录错误。

## 日志

`classification` 日志输出目录：`log/clasification/`

- 关键步骤日志：`[info]`
- 错误日志：`[error]`
