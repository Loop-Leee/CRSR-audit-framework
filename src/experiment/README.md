# experiment 模块说明

`experiment` 模块用于把一次实验固定为“可复现单元”：固定参数、运行全流程、写出统一产物与统一指标。

## 1. 模块目标

- 统一入口：`run_experiment.py`
- 统一指标：`metrics.py`
- 统一产物落盘：`artifact_writer.py`
- 统一汇总：`data/experiments/results.csv` 和 `data/experiments/results.jsonl`
- 统一日志：`log/experiment/*.log`

## 2. 实验阶段计划（建议）

### 阶段 A：环境与输入校验

目标：确认 `Word -> chunks -> classified` 能跑通。

```bash
python3 main.py experiment \
  --run-name dryrun \
  --mode keyword_only \
  --chunk-size 1000 \
  --input-dir data/1-original \
  --output-root data/experiments
```

### 阶段 B：基线实验（关键词）

目标：产出无 LLM 成本的 baseline 行，作为对照组。

```bash
python3 main.py experiment \
  --run-name baseline_keyword \
  --mode keyword_only \
  --chunk-size 1000
```

### 阶段 C：主实验（关键词 + LLM）

目标：采集质量指标 + 成本/时延/稳定性指标。

```bash
python3 main.py experiment \
  --run-name main_keyword_llm \
  --mode keyword_llm \
  --chunk-size 700 \
  --max-concurrency 10
```

### 阶段 D：参数扫描实验（示例）

目标：比较 chunk 大小和并发对性能、成本、质量的影响。

```bash
python3 main.py experiment --run-name grid_c700 --mode keyword_llm --chunk-size 700 --max-concurrency 5
python3 main.py experiment --run-name grid_c1000 --mode keyword_llm --chunk-size 1000 --max-concurrency 5
python3 main.py experiment --run-name grid_c1400 --mode keyword_llm --chunk-size 1400 --max-concurrency 5
```

```bash
python3 main.py experiment --run-name grid_p1 --mode keyword_llm --chunk-size 1000 --max-concurrency 1
python3 main.py experiment --run-name grid_p5 --mode keyword_llm --chunk-size 1000 --max-concurrency 5
python3 main.py experiment --run-name grid_p10 --mode keyword_llm --chunk-size 1000 --max-concurrency 10
```

### 阶段 E：标注评估实验

目标：输出 `precision/recall/f1`，用于论文主表。

```bash
python3 main.py experiment \
  --run-name eval_with_gt \
  --mode keyword_llm \
  --ground-truth baseline
```

> `--ground-truth` 支持目录（`*.json`）、单个 `json`、`jsonl`。

## 3. 启动命令总览

### 3.1 最小可运行命令

```bash
python3 main.py experiment
```

### 3.2 常用参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--run-name` | 实验名（会进入 `run_id`） | `experiment` |
| `--mode` | `keyword_only` / `keyword_llm` | `keyword_llm` |
| `--chunk-size` | 分块上限（字符） | `1000` |
| `--input-dir` | 原始 Word 输入目录 | 使用 chunking 配置 |
| `--risk-info` | 风险类型 CSV 路径 | 使用 classification 配置 |
| `--output-root` | 实验输出根目录 | `data/experiments` |
| `--ground-truth` | 标注集路径 | 空 |
| `--llm-config` | LLM 配置路径 | `src/llm/llm_config.json` |
| `--model` | 覆盖模型名 | 空（读取环境） |
| `--temperature` | 覆盖温度 | 配置文件值 |
| `--max-concurrency` | 覆盖最大并发 | 配置文件值 |
| `--disable-llm-concurrency` | 强制关闭并发 | 关闭 |
| `--disable-cache` | 强制关闭缓存 | 关闭 |

## 4. 目录与产物结构

### 4.1 `data/experiments` 结构

```text
data/experiments/
  results.csv
  results.jsonl
  <run_id>/
    chunks/
    classified/
    audit_result/
      audit_result.json
    final_report/
      final_report.md
    metrics/
      experiment_config.json
      llm_trace.jsonl
      metrics.json
```

### 4.2 `log/experiment` 结构

```text
log/
  experiment/
    experiment_YYYYMMDD_HHMMSS.log
```

## 5. 日志记录说明（`log/experiment/*.log`）

### 5.1 记录事件

| 日志片段 | 含义 |
|---|---|
| `实验配置` | 打印本次有效配置 |
| `实验目录准备完成` | run 目录创建完成 |
| `实验输入文件数` | 待处理文档数 |
| `分块完成` | chunking 输出完成 |
| `chunk 分类完成` | 每个 chunk 分类完成，含 LLM 诊断字段 |
| `诊断明细写入完成` | `metrics/llm_trace.jsonl` 已写出 |
| `指标写入完成` | `metrics/metrics.json` 已写出 |
| `results 追加完成` | `results.csv/jsonl` 已写入一行 |
| `实验完成` | 本次 run 结束 |

### 5.2 `chunk 分类完成` 关键字段

| 字段 | 说明 |
|---|---|
| `token_in` | 输入 token 数 |
| `token_out` | 输出 token 数 |
| `total_tokens` | 总 token（优先读取 `usage.total_tokens`） |
| `cached_tokens` | 缓存命中 token（`prompt_tokens_details.cached_tokens`） |
| `reasoning_tokens` | 推理 token（`completion_tokens_details.reasoning_tokens`） |
| `total_tokens_estimated` | `total_tokens` 是否由 `token_in+token_out` 回退估算 |
| `latency_ms` | 单次调用耗时（毫秒） |
| `request_id` | 服务侧请求 ID（若返回） |
| `retries` | 重试次数 |
| `error_code` | 失败错误码（如 `http_429`、`network_error`、`invalid_schema`） |
| `cache` | 是否命中缓存 |

## 6. 指标与表格字段解释

### 6.1 `results.csv` / `results.jsonl` 字段

| 字段 | 含义 |
|---|---|
| `run_id` | 唯一实验 ID |
| `timestamp` | 实验开始时间（ISO） |
| `run_name` | 用户指定实验名 |
| `mode` | 实验模式 |
| `model` | 实际模型名 |
| `temperature` | 推理温度 |
| `chunk_size` | 分块大小 |
| `concurrent_enabled` | 是否启用并发 |
| `max_concurrency` | 最大并发数 |
| `cache_enabled` | 是否启用缓存 |
| `cache_path` | 缓存文件路径 |
| `input_dir` | 输入目录 |
| `risk_info_path` | 风险规则路径 |
| `ground_truth_path` | 标注集路径 |
| `file_count` | 输入文档数 |
| `classified_file_count` | 分类输出文件数 |
| `precision` | 微平均 Precision |
| `recall` | 微平均 Recall |
| `f1` | 微平均 F1 |
| `schema_valid_rate` | LLM 输出可解析率 |
| `avg_token_in` | 平均输入 token（prompt） |
| `avg_token_out` | 平均输出 token（completion） |
| `avg_total_token` | 平均总 token（优先 `usage.total_tokens`） |
| `avg_token` | 平均 token（`token_in+token_out`） |
| `avg_reasoning_token` | 平均推理 token |
| `avg_cached_token` | 平均缓存命中 token |
| `avg_latency_ms` | 平均时延（毫秒） |
| `reasoning_token_ratio` | 推理 token / 总 token |
| `cached_token_ratio` | 缓存命中 token / 输入 token |
| `conflict_rate` | keyword 与 semantic 互斥冲突率 |
| `cache_hit_rate` | 缓存命中率 |
| `llm_error_rate` | LLM 错误率 |
| `total_tokens_estimated_rate` | `usage.total_tokens` 缺失并回退计算的比例 |
| `chunk_count` | chunk 总数 |
| `llm_called_count` | 调用 LLM 的 chunk 数 |
| `label_support` | 标注正例标签总数 |

### 6.2 `metrics/metrics.json` 字段

与 `results.csv` 中指标字段一致，是 run 级聚合指标快照。

### 6.3 `metrics/llm_trace.jsonl` 字段

每行对应一个 chunk 的诊断记录。

| 字段 | 含义 |
|---|---|
| `source_file` | 当前 chunk 来源文件 |
| `chunk_id` | chunk 编号 |
| `keyword_risks` | 关键词匹配风险列表 |
| `semantic_risks` | LLM 语义匹配风险列表 |
| `final_risks` | 合并后风险列表 |
| `llm_called` | 是否调用了 LLM |
| `schema_valid` | LLM 返回是否符合 schema |
| `token_in` | 输入 token |
| `token_out` | 输出 token |
| `total_tokens` | 总 token（优先读取 `usage.total_tokens`） |
| `cached_tokens` | 输入缓存命中 token（`prompt_tokens_details.cached_tokens`） |
| `reasoning_tokens` | 推理 token（`completion_tokens_details.reasoning_tokens`） |
| `total_tokens_estimated` | `total_tokens` 是否由 `token_in+token_out` 回退估算 |
| `latency_ms` | 调用时延 |
| `request_id` | 请求 ID |
| `retries` | 重试次数 |
| `error_code` | 错误码 |
| `cached` | 缓存命中 |
| `conflict` | keyword 与 semantic 是否完全不相交 |

## 7. 代码功能说明（experiment 目录）

### `run_experiment.py`

- 入口函数：`main()`
- 职责：
  - 读取参数与配置
  - 固定 run_id（参数指纹）
  - 执行 chunking + classification
  - 调用 metrics 聚合
  - 调用 artifact_writer 统一落盘
  - 追加 `results.csv/jsonl`

### `metrics.py`

- 入口函数：`compute_experiment_metrics(...)`
- 职责：
  - 读取分类预测
  - 读取标注集（目录/json/jsonl）
  - 计算 `precision/recall/f1`
  - 计算运行指标（schema/token/latency/conflict/cache/error）

### `artifact_writer.py`

- 入口类：`ArtifactWriter`
- 职责：
  - 规划 run 目录结构
  - 写出 `llm_trace.jsonl`
  - 写出 `metrics.json`
  - 写出 `audit_result.json`
  - 写出 `final_report.md`
  - 追加 `results.csv/jsonl`

## 8. 复现实验建议

- 固定以下参数：`mode/model/temperature/chunk_size/max_concurrency/cache_enabled`
- 固定输入与规则：`input_dir/risk_info_path`
- 固定标注：`ground_truth_path`
- 重点检查 `total_tokens_estimated_rate`，若非 0 说明部分模型响应未返回 `usage.total_tokens`。
- 将 `metrics/experiment_config.json` 与 `results.csv` 一起归档，作为复现实验元数据。
