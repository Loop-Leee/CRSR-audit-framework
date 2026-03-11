# Review 模块

模块功能：读取 `classification` 输出（`data/3-classified/*.classified.json`），基于 `risk_type` 候选与规则库执行 LLM 结构化审查，输出可追溯审查结果、诊断轨迹和统计指标。

## 运行方式

默认运行：

```bash
python3 main.py review
```

指定输入输出：

```bash
python3 main.py review --input data/3-classified --output data/4-review
```

指定规则文件与版本：

```bash
python3 main.py review --rules prompt/rule_hits_expanded.csv --rule-version v1
```

设置写入到每条 review_item 的 `ground_truth` 初始值（默认 `待审核`）：

```bash
python3 main.py review --ground_truth 待审核
```

关闭并发：

```bash
python3 main.py review --disable-llm-concurrency
```

设置 schema 失败重试上限（0-3）：

```bash
python3 main.py review --schema-retry-limit 1
```

## 消融实验命令

1. 不注入规则列表（`ablation_no_rules`）：

```bash
python3 main.py review --ablation-no-rules
```

2. 粗粒度规则（`ablation_coarse_rules`，每个 risk_type 一条合并规则）：

```bash
python3 main.py review --ablation-coarse-rules
```

说明：`--ablation-no-rules` 与 `--ablation-coarse-rules` 互斥。

## 输入输出

- 输入：`*.classified.json`（来自 `classification`）
- 输出：`data/4-review/*.review.json`
- 运行级产物：
  - `data/4-review/review_trace.jsonl`（任务级 LLM 诊断）
  - `data/4-review/review_metrics.json`（聚合指标）

`*.review.json` 会在原 payload 基础上新增：

- `review_meta`：`doc_id`、规则版本、消融开关、任务计数、生成时间
- `review_items`：结构化风险命中结果

`review_items` 单条结构示例：

```json
{
  "risk_id": "doc#c2#rt付款时间审查#r付款时间审查:02#1",
  "source_file": "...",
  "chunk_id": 2,
  "risk_type": "付款时间审查",
  "result": "不合格",
  "ground_truth": "待审核",
  "rule_hit": "付款节点超过验收后90天，违反规则",
  "rule_hit_id": "付款时间审查:02",
  "span": "验收合格后120日内付款",
  "span_offset": [128, 139],
  "suggest": "将付款节点调整为验收后60日内"
}
```

其中程序补全字段：

- `chunk_id`
- `span_offset`（匹配失败为 `null`）
- `rule_hit_id`（无法匹配规则库时为 `"UNKNOWN"`）
- `ground_truth`（默认 `"待审核"`，人工标注后可改为 `"合格"` 或 `"不合格"`）
- `risk_id`（`doc_id#c{chunk_id}#rt{risk_type}#r{rule_hit_id}#{idx}`）

`risk_id` 计算规则：

- 组成字段：
  - `doc_id`：文档标识。
  - `chunk_id`：命中的 chunk 编号。
  - `risk_type`：本条命中的审查类型。
  - `rule_hit_id`：规则命中 ID（无法解析时为 `UNKNOWN`）。
  - `idx`：同一计数键下的顺序号（从 `1` 开始）。

说明：`doc_id` 与 `rule_version` 仅出现在 `review_meta`，不在 `review_items` 重复。

### review_trace.jsonl 字段说明

`review_trace.jsonl` 为 JSONL 文件，每行对应一个 `(chunk_id, risk_type)` 审查任务。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `source_file` | `str` | 原始合同文件路径（来自 classified 输入）。 |
| `doc_id` | `str` | 文档唯一标识。 |
| `chunk_id` | `int \| str` | 分块编号。 |
| `risk_type` | `str` | 本任务审查类型。 |
| `candidate_rule_count` | `int` | 注入给模型的候选规则数量。 |
| `llm_called` | `bool` | 本任务是否实际调用了 LLM。 |
| `schema_valid` | `bool` | 模型输出是否通过 schema 校验。 |
| `item_count` | `int` | 本任务最终解析出的命中项数量。 |
| `token_in` | `int` | 输入 token 数。 |
| `token_out` | `int` | 输出 token 数。 |
| `total_tokens` | `int` | 总 token 数。 |
| `cached_tokens` | `int` | prompt 侧缓存命中 token 数。 |
| `reasoning_tokens` | `int` | completion 侧推理 token 数。 |
| `total_tokens_estimated` | `bool` | `total_tokens` 是否为回填估算值。 |
| `latency_ms` | `float` | 单次请求耗时（毫秒）。 |
| `request_id` | `str` | 上游接口请求 ID（若有）。 |
| `retries` | `int` | LLM HTTP/网络层重试次数。 |
| `error_code` | `str \| null` | 失败错误码（如 `invalid_schema`、`timeout`、`network_error`）。 |
| `cached` | `bool` | 是否命中 LLM 响应缓存。 |
| `schema_retries` | `int` | schema 校验失败后的补救重试次数。 |

### review_metrics.json 字段说明

`review_metrics.json` 为单次运行的聚合统计结果。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `file_count` | `int` | 成功输出的 `.review.json` 文件数量。 |
| `chunk_count` | `int` | 本次处理的 chunk 总数。 |
| `task_count` | `int` | 审查任务总数（`chunk × risk_type`）。 |
| `emitted_item_count` | `int` | 最终输出的 `review_items` 总数。 |
| `llm_called_count` | `int` | 实际触发 LLM 的任务数。 |
| `schema_valid_rate` | `float` | LLM 任务中 schema 校验通过率。 |
| `llm_error_rate` | `float` | LLM 任务中接口失败率（含超时/网络等）。 |
| `span_match_rate` | `float` | 输出项里 `span_offset` 成功定位比例。 |
| `rule_id_known_rate` | `float` | 输出项里 `rule_hit_id != "UNKNOWN"` 比例。 |
| `avg_token_in` | `float` | LLM 任务平均输入 token。 |
| `avg_token_out` | `float` | LLM 任务平均输出 token。 |
| `avg_chunk_token_in` | `float` | 按 chunk 汇总后（同 chunk 的多 risk_type 任务 token 求和）再取平均的输入 token。 |
| `avg_chunk_token_out` | `float` | 按 chunk 汇总后（同 chunk 的多 risk_type 任务 token 求和）再取平均的输出 token。 |
| `avg_total_token` | `float` | LLM 任务平均总 token。 |
| `avg_latency_ms` | `float` | LLM 任务平均耗时（毫秒）。 |
| `schema_retry_count` | `int` | 本次运行累计 schema 补救重试次数。 |
| `ablation_no_rules` | `bool` | 是否启用“不注入规则”消融。 |
| `ablation_coarse_rules` | `bool` | 是否启用“粗粒度规则”消融。 |
| `rule_version` | `str` | 本次运行使用的规则版本。 |

## 一致性评估脚本（TP/FP/TN/FN）

脚本路径：`src/review/review_eval.py`

运行示例：

```bash
python3 -m src.review.review_eval --input data/4-review
```

写出评估结果 JSON：

```bash
python3 -m src.review.review_eval --input data/4-review --output data/4-review/review_eval.json
```

输出字段说明：

| 字段 | 含义 |
| --- | --- |
| `tp` | 预测 `不合格` 且标注 `不合格` 的条数。 |
| `fp` | 预测 `不合格` 但标注 `合格` 的条数。 |
| `tn` | 预测 `合格` 且标注 `合格` 的条数。 |
| `fn` | 预测 `合格` 但标注 `不合格` 的条数。 |
| `evaluated_item_count` | 参与计算的条数（`ground_truth` 已标注为 `合格/不合格`）。 |
| `unlabeled_item_count` | 未参与计算的条数（`ground_truth` 不是 `合格/不合格`，如 `待审核`）。 |
| `invalid_pred_item_count` | 未参与计算的条数（`result` 不是 `合格/不合格`）。 |
| `total_item_count` | 扫描到的 review_items 总数。 |
| `file_count` | 参与扫描的 `.review.json` 文件数。 |
| `input_path` | 评估输入路径。 |
| `precision` | 正类（`不合格`）精确率。 |
| `recall` | 正类（`不合格`）召回率。 |
| `f1` | 正类（`不合格`）F1。 |
| `accuracy` | `(TP+TN)/evaluated_item_count`。 |
| `positive_label` | 当前正类标签，固定为 `不合格`。 |

## RuleStore

`rule_store.py` 提供全局规则管理：

- `load_rules(path, rule_version="v1")`
- `get_rules(risk_type, rule_version=None)`
- `build_coarse_version(source_version="v1", target_version="v1_coarse")`

支持从 `rule_hits_expanded.csv/json` 加载，并按版本切换，供后续规则版本对比与消融实验使用。

## Schema 约束

默认模式（注入候选规则）要求 LLM 输出：

```json
{
  "items": [
    {
      "risk_type": "...",
      "result": "合格或不合格",
      "rule_hit_id": "...",
      "rule_hit": "...",
      "span": "...",
      "suggest": "..."
    }
  ]
}
```

约束：

- `risk_type` 必须来自该 chunk 的候选风险类型
- `result` 只能是 `合格` 或 `不合格`
- `rule_hit_id` 必须来自该 `risk_type` 注入的候选规则列表
- `span` 必须是 chunk 原文连续子串

`--ablation-no-rules` 下不注入候选规则，允许 `rule_hit_id` 缺省，程序会根据 `rule_hit` 回匹配规则库。

当模型输出不符合 schema（如字段缺失、取值非法）时，模块会按 `schema_retry_limit` 进行限次重试；超过上限后记录 `review_schema_invalid`。

## 配置

默认配置文件：`src/review/review_config.json`

```json
{
  "input_dir": "data/3-classified",
  "output_dir": "data/4-review",
  "rules_path": "prompt/rule_hits_expanded.csv",
  "log_dir": "log/review",
  "rule_version": "v1",
  "schema_retry_limit": 1
}
```

## 日志

- 目录：`log/review/`
- 标签：`[info]` / `[error]`
- 覆盖：关键执行路径、schema 失败、LLM 失败、规则缺失等错误路径。
