# Baseline 模块

模块功能：直接读取 `chunking` 输出（`data/2-chunks/*.chunks.json`），将 `risk_info.csv` 中全部 `(审查类型, 审查规则)` 一次性注入到每个 chunk 执行审核，输出与 `review` 模块同构的 `*.review.json` 结果。

说明：
- 不依赖 `classification/review/result` 处理链；
- `review_items` 不包含 `rule_hit` 和 `rule_hit_id`；
- `risk_id` 保持 `review` 同构，仅去掉 `rule_hit_id` 段，格式为 `doc_id#c{chunk_id}#rt{risk_type}#{idx}`。

## 运行方式

默认运行：

```bash
python3 main.py baseline
```

指定输入输出：

```bash
python3 main.py baseline --input data/2-chunks --output data/4-review-baseline
```

指定风险规则文件：

```bash
python3 main.py baseline --risk-info prompt/risk_info.csv
```

设置写入每条 review_item 的 `ground_truth` 初始值（默认 `待审核`）：

```bash
python3 main.py baseline --ground_truth 待审核
```

建议在大 chunk / 全规则场景限制输出长度并关闭思考：

```bash
python3 main.py baseline \
  --max-new-tokens 512 \
  --openai-no-think-prompt \
  --openai-disable-thinking \
  --openai-send-max-new-tokens-param
```

说明：
- `--openai-no-think-prompt`：在 prompt 末尾追加 `/no_think`；
- `--openai-disable-thinking`：请求里附带 `enable_thinking=false`；
- `--openai-send-max-new-tokens-param`：额外发送 `max_new_tokens/max_completion_tokens` 兼容字段。

## 消融实验命令

1. 不注入规则文本（`ablation_no_rules`）：

```bash
python3 main.py baseline --ablation-no-rules
```

2. 不计算 `span_offset`（`ablation_no_span_offset`）：

```bash
python3 main.py baseline --ablation-no-span-offset
```

## 输入输出

- 输入：`*.chunks.json`（来自 `chunking`）
- 输出：`data/4-review-baseline/*.review.json`
- 运行级产物：
  - `data/4-review-baseline/baseline_trace.jsonl`
  - `data/4-review-baseline/baseline_metrics.json`

`*.review.json` 会在原 payload 基础上新增：

- `review_meta`：`doc_id`、消融开关、任务计数、规则计数、生成时间
- `review_items`：结构化审查结果（无 `rule_hit`、`rule_hit_id`）

`review_items` 示例：

```json
{
  "risk_id": "doc_xxx#c2#rt付款时间审查#1",
  "source_file": "...",
  "chunk_id": 2,
  "risk_type": "付款时间审查",
  "result": "不合格",
  "span": "验收合格后120日内付款",
  "span_offset": [128, 139],
  "suggest": "将付款节点调整为验收后60日内",
  "ground_truth": "待审核"
}
```

## 诊断与指标

### baseline_trace.jsonl

每行对应一个 chunk 任务，包含：
- `source_file` / `doc_id` / `chunk_id`
- `risk_rule_count`
- `llm_called` / `schema_valid` / `item_count`
- `token_in` / `token_out` / `total_tokens`
- `cached_tokens` / `reasoning_tokens`
- `latency_ms` / `request_id` / `retries`
- `error_code` / `cached` / `schema_retries`

### baseline_metrics.json

聚合字段包括：
- `file_count` / `chunk_count` / `task_count`
- `emitted_item_count` / `llm_called_count`
- `schema_valid_rate` / `llm_error_rate` / `span_match_rate`
- `avg_token_in` / `avg_token_out` / `avg_total_token` / `avg_latency_ms`
- `schema_retry_count` / `risk_rule_count`
- `ablation_no_rules` / `ablation_no_span_offset`

## 配置

默认配置文件：`src/baseline/baseline_config.json`

```json
{
  "input_dir": "data/2-chunks",
  "output_dir": "data/4-review-baseline",
  "risk_info_path": "prompt/risk_info.csv",
  "log_dir": "log/baseline",
  "schema_retry_limit": 1,
  "max_new_tokens": 512,
  "openai_no_think_prompt": true,
  "openai_disable_thinking": true,
  "openai_send_max_new_tokens_param": true
}
```

## 日志

- 目录：`log/baseline/`
- 标签：`[info]` / `[error]`
- 覆盖：关键执行路径、schema 失败、LLM 失败、文件级异常。
