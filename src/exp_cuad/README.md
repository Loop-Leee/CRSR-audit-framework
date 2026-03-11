# exp_cuad 模块说明

`exp_cuad` 用于在 CUAD 数据集上执行条款存在性识别与证据句提取实验，核心目标是：

- 同一套推理流程支持多后端（`openai/transformers/vllm`）
- 输出可直接用于统计与复盘（JSONL 明细 + 进度 + summary）
- 支持输出层面的消融开关（控制写出粒度）

## 1. 模块结构

```text
src/exp_cuad/
  cuad_loader.py      # 加载 CUAD（本地/HF）并标准化为文档级样本
  prompts.py          # baseline / crsr_lite 提示词模板
  text_utils.py       # 分句、归一化、分块、句子集合 Jaccard
  eval_metrics.py     # PairResult 与核心指标计算函数
  run_infer.py        # CLI 主入口：推理 + 结构化产物 + 汇总
```

职责边界：

- `run_infer.py` 只负责“编排与产物落盘”；指标定义集中在 `eval_metrics.py`
- 通用日志统一走 `src/tools/logger.py`，日志路径固定为 `log/exp_cuad/*.log`

## 2. 快速运行

### 2.1 环境

```bash
conda env create -f environment.yml
conda activate crsr-audit
mkdir -p data/cuad/outputs
```

### 2.2 OpenAI 兼容服务（推荐）

```bash
export LLM_BASE_URL="http://<your-server>/v1"
export LLM_API_KEY="<optional>"
export LLM_MODEL="Qwen2-72B-Instruct"
```

```bash
python3 -m src.exp_cuad.run_infer \
  --model Qwen2-72B-Instruct \
  --backend openai \
  --mode baseline \
  --split test \
  --limit_docs 0 \
  --openai_no_think_prompt \
  --openai_disable_thinking \
  --openai_send_max_new_tokens_param \
  --out_jsonl data/cuad/outputs/cuad_baseline.jsonl
```

实际写出的文件名会自动追加“主要参数 + 时间戳 + 指纹”后缀，例如：

```text
data/cuad/outputs/cuad_baseline_mode-baseline_backend-openai_split-test_model-qwen2-72b-instruct_mc-6000_ov-800_mt-256_ld-5_20260301_213000_a1b2c3d4.jsonl
```

## 3. CLI 参数与消融开关

### 3.1 主要参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--model` | 模型名 | 必填 |
| `--data_source` | 数据源：`auto/local/hf` | `auto` |
| `--split` | 数据切分（HF 场景） | `train` |
| `--backend` | 推理后端：`openai/transformers/vllm` | `vllm` |
| `--mode` | 提示词策略：`baseline/crsr_lite` | `baseline` |
| `--max_chars` | 分块最大字符数 | `6000` |
| `--overlap_chars` | 分块重叠字符数 | `800` |
| `--max_new_tokens` | 单次生成最大 token | `256` |
| `--llm_concurrency` | chunk 并发（openai 后端） | `10` |
| `--openai_no_think_prompt/--no-openai_no_think_prompt` | openai：是否在 user prompt 追加 `/no_think` | `true` |
| `--openai_disable_thinking/--no-openai_disable_thinking` | openai：请求里附带 `enable_thinking=false` | `true` |
| `--openai_send_max_new_tokens_param/--no-openai_send_max_new_tokens_param` | openai：额外发送 `max_new_tokens` 参数 | `true` |
| `--limit_docs` | 文档数上限，`0` 表示全量 | `0` |
| `--out_jsonl` | 输出基准文件名（自动追加参数+时间戳+指纹） | `data/cuad/outputs/cuad_baseline.jsonl` |

### 3.2 核心参数解释（影响最终指标）

| 参数 | 对指标的影响 | 建议 |
|---|---|---|
| `--model` | 决定推理能力上限，直接影响 P/R/F1 | 实验中保持固定；跨模型对比时只改这一项 |
| `--backend` | 影响请求协议与返回格式稳定性，间接影响解析成功率和召回 | 同一组实验保持固定（推荐 `openai`） |
| `--mode` | 决定 prompt 模板（`baseline/crsr_lite`），直接影响分类行为 | 实验报告必须记录并固定 |
| `--max_chars` | 决定分块粒度，影响证据是否落在同一 chunk | 与 `--overlap_chars` 成对固定 |
| `--overlap_chars` | 决定跨 chunk 上下文保留，影响召回/不一致率 | 与 `--max_chars` 成对固定 |
| `--max_new_tokens` | 决定单次输出上限；过大可能导致长推理/空内容，过小会截断 JSON | 推荐从 `256` 起步并固定 |
| `--openai_no_think_prompt` | 控制是否追加 `/no_think`，影响是否出现“只输出 reasoning” | `openai` 后端建议保持开启 |
| `--openai_disable_thinking` | 控制是否请求 `enable_thinking=false`，影响空 content 概率 | `openai` 后端建议保持开启 |
| `--openai_send_max_new_tokens_param` | 控制是否额外发送 `max_new_tokens`，影响网关兼容性 | `openai` 后端建议保持开启 |
| `--split`/`--data_source` | 决定数据来源与样本分布，直接影响最终指标可比性 | 报告中必须记录 |
| `--limit_docs` | 控制样本规模；非全量结果不能与全量指标直接比较 | 最终报告建议 `0`（全量） |

### 3.3 输出消融开关

| 开关 | 作用 | 默认 |
|---|---|---|
| `--disable_llm_usage_output` | 不写出 `llm_usage` 字段 | 关闭（写出） |
| `--disable_extended_pair_metrics_output` | 不写出 `presence_confusion/evidence_jaccard` | 关闭（写出） |

## 4. 输出文件与字段说明

默认会产出三类文件：

- `<out_jsonl + 参数后缀 + 时间戳 + 指纹>.jsonl`：主明细（每行一个 `(doc_id, label)`）
- `<...>.jsonl.progress.jsonl`：过程事件流（run_start/chunk_result/pair_result/run_end）
- `<...>.jsonl.summary.json`：最终汇总（指标 + LLM 用量总计）

### 4.1 `out_jsonl` 主字段

向后兼容字段：

- `pair_index/doc_id/label`
- `present_pred/evidence_pred`
- `present_gt/evidence_gt`
- `chunk_votes`
- `mode/backend`

新增分析字段：

- `analysis_version`
- `llm_usage`（可关）：token、调用次数、时延、缓存命中、重试数
- `presence_confusion`（可关）：单条样本的 `tp/fp/fn/tn` one-hot 计数
- `evidence_jaccard`（可关）：仅 GT present 时计算

## 5. 如何用 `cuad_baseline.jsonl` 做指标分析

### 5.1 全套指标脚本（统一口径，推荐）

```bash
python3 -m src.exp_cuad.analyze_metrics \
  --input_jsonl outputs/cuad/cuad_baseline_mode-baseline_model-qwen2-72b-instruct_20260302_102233.jsonl \
  --out_json outputs/cuad/cuad_baseline_mode-baseline_model-qwen2-72b-instruct_20260302_102233.metrics.json
```

输出为结构化 JSON，核心字段包括：

- `metrics.presence_micro`：micro precision/recall/f1 + tp/fp/fn/tn
- `metrics.presence_macro`：macro_f1
- `metrics.evidence_jaccard_gt_present`
- `metrics.laziness_rate`
- `metrics.inconsistency_rate`
- `per_label_metrics`（可通过消融开关关闭）
- `llm_usage_aggregate`（可通过消融开关关闭）

### 5.2 结果可用性检查

- `dataset_stats.is_dense_matrix=true` 时，结果可视为完整 `(doc,label)` 矩阵；否则说明运行中断或样本缺失
- `row_stats.line_invalid=0` 时，输入文件解析完整；否则需先定位坏行
- 指标报告应与实验参数一起归档（至少记录 `model/backend/mode/max_chars/overlap_chars/max_new_tokens`）

## 6. 指标口径（统一来源）

`src/exp_cuad/eval_metrics.py`：

- `compute_presence_f1`：presence 二分类 micro P/R/F1
- `compute_macro_f1`：按 label 求 F1 再平均
- `compute_evidence_jaccard`：仅 GT present 样本计算证据集合 Jaccard
- `compute_laziness_rate`：GT present 但预测 absent 的比例
- `compute_inconsistency_rate`：同一样本 chunk 投票同时存在 True/False 的比例

## 7. 常见失败路径

- `ModuleNotFoundError: datasets`：安装 `datasets`
- `NonMatchingSplitsSizesError`：代码会自动尝试 `force_redownload`
- `vllm` 在 macOS 不可用：切换 `--backend openai/transformers`
- 模型输出非 JSON：系统会自动重试一次并记录失败日志
- 若日志中频繁出现 `LLM 返回空内容` 且 `token_out` 接近固定上限（如 2048）：
  常见原因是网关忽略了 `max_tokens` 或默认开启 thinking，建议保持
  `--openai_no_think_prompt --openai_disable_thinking --openai_send_max_new_tokens_param`
- 建议 `--max_new_tokens` 先用 `256`；过大的值（如 2048）在推理模型上更容易出现超时与“思考耗尽配额”。

## 8. 与当前样例输出的关系

建议把 `cuad_baseline*.jsonl` 作为原始实验记录，把 `analyze_metrics.py` 产出的 `*.metrics.json` 作为最终分析口径文件归档（用于汇报与横向对比）。
