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
  "rule_hit": "付款节点超过验收后90天，违反规则",
  "rule_hit_id": "付款时间审查:02",
  "span": "验收合格后120日内付款",
  "span_offset": [128, 139],
  "suggest": "将付款节点调整为验收后60日内"
}
```

其中程序补全字段：

- `chunk_id`
- `span_offset`（匹配失败为 `"UNKNOWN"`）
- `rule_hit_id`（无法匹配规则库时为 `"UNKNOWN"`）
- `risk_id`（`doc_id#c{chunk_id}#rt{risk_type}#r{rule_hit_id}#{idx}`）

说明：`doc_id` 与 `rule_version` 仅出现在 `review_meta`，不在 `review_items` 重复。

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
