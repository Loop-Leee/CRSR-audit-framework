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

3. `review`（规则化结构审查）

```bash
python3 main.py review
```

消融实验：

```bash
python3 main.py review --ablation-no-rules
python3 main.py review --ablation-coarse-rules
```

4. `reflection`（对 `review_items` 做后验校正）

```bash
python3 main.py reflection
```

指定输入输出：

```bash
python3 main.py reflection --input data/4-review --output data/5-reflection
```

调节两阶段筛选参数：

```bash
python3 main.py reflection --stage1-threshold 4 --stage2-max-items 6
```

5. `baseline`（直接对 `chunking` 结果执行全规则审查，不经过 classification/review 拆分）

```bash
python3 main.py baseline
```

指定输入输出：

```bash
python3 main.py baseline --input data/2-chunks --output data/4-review-baseline
```

消融实验：

```bash
python3 main.py baseline --ablation-no-rules
python3 main.py baseline --ablation-no-span-offset
```

OpenAI-compatible 网关建议（关闭思考并限制输出）：

```bash
python3 main.py baseline \
  --max-new-tokens 512 \
  --openai-no-think-prompt \
  --openai-disable-thinking \
  --openai-send-max-new-tokens-param
```

6. `result`（依据 review 的 span 在 `6-result` 副本写入 Word 批注，不改源文件）

```bash
python3 main.py result
```

指定输入输出：

```bash
python3 main.py result --input data/4-review --output data/6-result
```

消融实验：

```bash
python3 main.py result --ablation-no-writeback
python3 main.py result --ablation-no-chunk-offset
```

7. `experiment`（统一实验入口，按 `chunking -> classification -> review` 运行并产出指标）

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

仅跑 LLM 基线（不执行 keyword 逻辑）：

```bash
python3 main.py experiment \
  --run-name exp_llm_only \
  --mode llm_only \
  --chunk-size 1000
```

单次输出三路消融（只进行一次 keyword 匹配 + 一次 llm 匹配）：

```bash
python3 main.py experiment \
  --run-name exp_onepass \
  --mode keyword_llm_experiment \
  --chunk-size 1000
```

覆盖 review 规则与消融参数示例：

```bash
python3 main.py experiment \
  --run-name exp_with_review_opts \
  --mode keyword_llm \
  --review-rules prompt/rule_hits_expanded.csv \
  --review-rule-version v1 \
  --review-ground-truth 待审核 \
  --review-schema-retry-limit 2
```

带标注集（用于 P/R/F1）：

```bash
python3 main.py experiment \
  --run-name exp_eval \
  --mode keyword_llm \
  --ground-truth baseline
```

Review 标准集对齐评测（`(doc_id, chunk_id, risk_type)`）：

```bash
python3 -m src.experiment.review_eval \
  --gold-dir dataset/standard-review \
  --pred-dir data/4-review \
  --output-dir data/experiments/review_eval
```

仅按标准集统计宇宙（消融）：

```bash
python3 -m src.experiment.review_eval \
  --gold-dir dataset/standard-review \
  --pred-dir data/4-review-baseline \
  --ablation-standard-only-universe
```

8. `exp_cuad`（CUAD 实验）

在 macOS 上推荐使用远端 OpenAI 兼容服务，不在本机加载大模型：

```bash
export LLM_BASE_URL="http://<your-server>/v1"
export LLM_API_KEY="<optional>"
export LLM_MODEL="Qwen2-72B-Instruct"

python3 -m src.exp_cuad.run_infer \
  --model Qwen2-72B-Instruct \
  --backend openai \
  --mode baseline \
  --split test \
  --out_jsonl data/cuad/outputs/cuad_baseline.jsonl
```

说明：实际文件名会自动追加“主要参数 + 时间戳 + 指纹”后缀，便于多次实验并存与追溯。

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
- review 标准集评测：`src/experiment/review_eval.py`
- 输出根目录：`data/experiments/`
- 模块完整文档：`src/experiment/README.md`
- 执行链路：`chunking -> classification -> review`

每次实验会生成：

```text
data/experiments/
  results.csv                  # 每次实验一行（汇总）
  results.jsonl                # 每次实验一行（结构化）
  <run_id>/
    chunks/*.chunks.json
    classified/*.classified.json
    classified/keyword_only/*.classified.json      # keyword_llm_experiment
    classified/llm_only/*.classified.json          # keyword_llm_experiment
    classified/keyword_llm/*.classified.json       # keyword_llm_experiment
    review/*.review.json
    review/review_trace.jsonl
    review/review_metrics.json
    review/keyword_only/*.review.json              # keyword_llm_experiment
    review/llm_only/*.review.json                  # keyword_llm_experiment
    review/keyword_llm/*.review.json               # keyword_llm_experiment
    audit_result/audit_result.json
    final_report/final_report.md
    final_report/final_report_*.md                 # keyword_llm_experiment
    metrics/
      experiment_config.json
      llm_trace.jsonl
      metrics.json
      metrics_*.json                               # keyword_llm_experiment
```

`results.csv` 包含实验参数与核心指标：

- 配置参数：`run_id/timestamp/mode/model/temperature/chunk_size/max_concurrency/cache_*`
- 复合模式标识：`experiment_mode`（用于区分 `keyword_llm_experiment`）
- review 产物计数：`review_file_count`
- 质量指标：`precision/recall/f1`（有标注集时计算）
- 运行指标：`avg_token_in/avg_token_out/avg_total_token/reasoning_token_ratio/cached_token_ratio/avg_latency_ms/schema_valid_rate/conflict_rate/cache_hit_rate/llm_error_rate/total_tokens_estimated_rate`

### review

- 目录：`src/review/`
- 输入：`data/3-classified/*.classified.json`
- 输出：`data/4-review/*.review.json`
- 规则源：`prompt/rule_hits_expanded.csv`（或 `.json`）
- 规则管理：`RuleStore`（`rule_version` + 粗粒度规则版本构建）
- 输出约束：`review_meta.rule_version` 为文件级元数据；`review_items` 含 `result`（仅 `合格/不合格`）
- 标注字段：`review_items.ground_truth` 默认写入 `待审核`（可用 `--ground_truth` 覆盖）
- 稳定性：schema 不合规输出支持限次重试（`--schema-retry-limit`，默认 `1`，上限 `3`）
- 评估脚本：`python3 -m src.review.review_eval --input data/4-review`（输出 TP/FP/TN/FN）
- 标准集对齐评测：`python3 -m src.experiment.review_eval --gold-dir dataset/standard-review --pred-dir data/4-review`
- 运行产物：
  - `data/4-review/review_trace.jsonl`
  - `data/4-review/review_metrics.json`
- 消融开关：
  - `--ablation-no-rules`
  - `--ablation-coarse-rules`
- 模块完整文档：`src/review/README.md`

### reflection

- 目录：`src/reflection/`
- 输入：`data/4-review/*.review.json`
- 输出：`data/5-reflection/*.reflection.json`
- 两阶段策略：
  - Stage-1：item-level 误报风险筛选 + 局部证据校正（默认 `fp_risk_score >= 4` 才调用 LLM）
  - Stage-2：按 `(doc_id, risk_type)` 分组，仅对冲突组做一致性校正（每组默认最多 6 条）
- 输出扩展字段：
  - `review_items_original`（原始审查结果备份）
  - `reflection_meta`（计数、调整量、token 统计）
  - `reflection_items`（反思后的最终结果）
- 运行产物：
  - `data/5-reflection/reflection_trace.jsonl`
  - `data/5-reflection/reflection_metrics.json`
- 关键参数：
  - `--stage1-threshold`
  - `--stage2-max-items`
- 模块完整文档：`src/reflection/README.md`

### baseline

- 目录：`src/baseline/`
- 输入：`data/2-chunks/*.chunks.json`
- 输出：`data/4-review-baseline/*.review.json`
- 规则源：`prompt/risk_info.csv`（读取全部 `(审查类型, 审查规则)`，每个 chunk 一次性注入）
- OpenAI 兼容请求默认支持“禁用思考 + 限制输出 token”参数（可由 CLI 覆盖）
- 输出约束：`review_items` 与 `review` 模块同构，但不包含 `rule_hit/rule_hit_id`
- `risk_id`：`doc_id#c{chunk_id}#rt{risk_type}#{idx}`（与 review 同构，移除 `rule_hit_id` 段）
- 运行产物：
  - `data/4-review-baseline/baseline_trace.jsonl`
  - `data/4-review-baseline/baseline_metrics.json`
- 消融开关：
  - `--ablation-no-rules`
  - `--ablation-no-span-offset`
- 模块完整文档：`src/baseline/README.md`

### result

- 目录：`src/result/`
- 输入：`data/4-review/*.review.json`
- 不修改：`review_items.source_file` 指向的源文件
- 输出：`data/6-result/annotated_docs/*.result.docx`（在副本上写入批注）
- 依赖：`python-docx>=1.2.0` + LibreOffice（用于 `.doc` 转 `.docx`）
- 常见错误：若提示“检测到 .doc 源文件，但未找到可用转换器”，请安装 LibreOffice 并确认 `which soffice` 可用
- 注释触发条件：`review_items.result ∈ {不合格, 待复核}`
- 批注内容：`risk_type/result/rule_hit/suggest`
- 运行产物：
  - `data/6-result/result_trace.jsonl`
  - `data/6-result/result_metrics.json`
  - `data/6-result/source_copies/*`（源文件为 `.doc` 时保留副本）
- 消融开关：
  - `--ablation-no-writeback`
  - `--ablation-no-chunk-offset`
- 模块完整文档：`src/result/README.md`

### exp_cuad

- 目录：`src/exp_cuad/`
- 入口：`python3 -m src.exp_cuad.run_infer`
- 输出：
  - `<out_jsonl + 参数后缀 + 时间戳 + 指纹>.jsonl`：逐 `(doc,label)` 结构化预测记录
  - `<...>.jsonl.progress.jsonl`：运行过程事件流（`run_start/chunk_result/pair_result/run_end`）
  - `<...>.jsonl.summary.json`：最终汇总（precision/recall/f1、macro、evidence、LLM 用量统计）
- 支持输出消融开关：
  - `--disable_llm_usage_output`
  - `--disable_extended_pair_metrics_output`
- 模块完整文档：`src/exp_cuad/README.md`

仅基于最新 CUAD 输出复算 `precision/recall/f1` 示例：

```bash
python3 - <<'PY'
import json
import subprocess
from pathlib import Path

latest = subprocess.check_output(
    "ls -t data/cuad/outputs/cuad_baseline_*.jsonl | head -1",
    shell=True,
    text=True,
).strip()
rows=[json.loads(x) for x in Path(latest).read_text(encoding='utf-8').splitlines() if x.strip()]
tp=sum(1 for r in rows if r['present_pred'] and r['present_gt'])
fp=sum(1 for r in rows if r['present_pred'] and not r['present_gt'])
fn=sum(1 for r in rows if (not r['present_pred']) and r['present_gt'])
precision=tp/(tp+fp) if (tp+fp) else 0.0
recall=tp/(tp+fn) if (tp+fn) else 0.0
f1=(2*precision*recall/(precision+recall)) if (precision+recall) else 0.0
print({'precision':precision,'recall':recall,'f1':f1,'tp':tp,'fp':fp,'fn':fn,'pairs':len(rows)})
PY
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

`llm_config.json` 中可配置：

- `max_retries`: HTTP/网络失败重试次数
- `reasoning_enable`: 是否开启反思能力（`false` 时自动附带 `enable_thinking=false`，可用于反思消融）
- `cache_enabled`: 是否开启持久化缓存（`false` 可作为缓存消融开关）
- `cache_path`: 缓存文件路径（JSONL）

并发调用默认开启（`concurrent_enabled=true`，`max_concurrency=10`），
可通过命令参数 `--disable-llm-concurrency` 关闭并发。

### LLM 缓存方式

缓存由 `src/llm/openai_client.py` 统一管理，采用“内存字典 + JSONL 追加写”策略：

1. **缓存 key 计算方式**  
   对以下字段构造 JSON（`sort_keys=True`）并执行 `SHA-256`：
   - `base_url`（去尾部 `/`）
   - `model`
   - `temperature`
   - `max_tokens`
   - `reasoning_enabled`
   - `extra_payload`
   - `messages`（完整请求消息）

2. **缓存命中方式**  
   - 请求前先计算 `cache_key`；  
   - 若内存缓存命中，直接返回并设置 `cached=true`；  
   - 未命中则请求模型，成功后写入缓存。  
   启动时会从 `cache_path` 加载历史 JSONL；同 key 重复时按“最后一条记录”覆盖。

3. **缓存记录方式**  
   每次缓存写入都会向 `cache_path` 追加一条结构化 JSONL 记录，包含：
   - `event/cache_version/created_at/cache_key`
   - `key_meta`（`base_url/model/temperature/max_tokens/extra_payload/reasoning_enabled/message_count`）
   - `response`（`content/token_in/token_out/total_tokens/cached_tokens/reasoning_tokens/latency_ms/request_id/retries`）

可通过 `OpenAICompatibleClient.cache_stats()` 获取结构化统计指标（命中、未命中、写入成功、写入失败、加载异常、绕过次数等）。

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
- `baseline`: `log/baseline/*.log`
- `review`: `log/review/*.log`
- `reflection`: `log/reflection/*.log`
- `result`: `log/result/*.log`
- `experiment`: `log/experiment/*.log`
- `llm`: `log/llm/*.log`

- 关键步骤日志：`[info]`
- 错误日志：`[error]`
