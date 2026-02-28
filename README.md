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
python3 main.py classification --disable-llm-concurrency
```

3. `experiment`（统一实验入口，固定参数并产出指标）

```bash
python3 main.py experiment \
  --run-name exp_keyword_llm \
  --mode keyword_llm \
  --chunk-size 1000 \
  --input-dir data/1-original \
  --output-root data/experiments
```

仅跑关键词基线（不调用 LLM）：

```bash
python3 main.py experiment \
  --run-name exp_keyword_only \
  --mode keyword_only \
  --chunk-size 1000
```

带标注集（用于 P/R/F1）：

```bash
python3 main.py experiment \
  --run-name exp_eval \
  --mode keyword_llm \
  --ground-truth baseline
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

### experiment

- 目录：`src/experiment/`
- 统一入口：`src/experiment/run_experiment.py`
- 指标口径：`src/experiment/metrics.py`
- 产物落盘：`src/experiment/artifact_writer.py`
- 输出根目录：`data/experiments/`
- 模块完整文档：`src/experiment/README.md`

每次实验会生成：

```text
data/experiments/
  results.csv                  # 每次实验一行（汇总）
  results.jsonl                # 每次实验一行（结构化）
  <run_id>/
    chunks/*.chunks.json
    classified/*.classified.json
    audit_result/audit_result.json
    final_report/final_report.md
    metrics/
      experiment_config.json
      llm_trace.jsonl
      metrics.json
```

`results.csv` 包含实验参数与核心指标：

- 配置参数：`run_id/timestamp/mode/model/temperature/chunk_size/max_concurrency/cache_*`
- 质量指标：`precision/recall/f1`（有标注集时计算）
- 运行指标：`avg_token_in/avg_token_out/avg_total_token/reasoning_token_ratio/cached_token_ratio/avg_latency_ms/schema_valid_rate/conflict_rate/cache_hit_rate/llm_error_rate/total_tokens_estimated_rate`

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

`llm_config.json` 中可配置：

- `max_retries`: HTTP/网络失败重试次数
- `cache_enabled`: 是否开启持久化缓存
- `cache_path`: 缓存文件路径（JSONL）

并发调用默认开启（`concurrent_enabled=true`，`max_concurrency=10`），
可通过命令参数 `--disable-llm-concurrency` 关闭并发。

LLM 响应元数据（用于实验指标）：

- `token_in/token_out`
- `latency_ms`
- `request_id`
- `retries`
- `error_code`
- `cached`

## 日志

日志能力统一由 `src/tools/logger.py` 提供，按模块输出到：

- `chunking`: `log/chunking/*.log`
- `classification`: `log/classification/*.log`
- `experiment`: `log/experiment/*.log`

- 关键步骤日志：`[info]`
- 错误日志：`[error]`
