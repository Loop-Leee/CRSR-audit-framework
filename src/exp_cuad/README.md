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
  --limit_docs 5 \
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
| `--limit_docs` | 文档数上限，`0` 表示全量 | `0` |
| `--out_jsonl` | 输出基准文件名（自动追加参数+时间戳+指纹） | `data/cuad/outputs/cuad_baseline.jsonl` |

### 3.2 输出消融开关

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

### 5.1 最小复算（precision / recall / f1）

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

### 5.2 合理性检查建议

- 检查 `pairs` 是否等于 `docs * labels`（中断运行会不一致）
- 检查 `llm_usage.call_count` 与 `chunk_votes` 长度关系（JSON 重试会导致调用数偏大）
- 检查 `presence_confusion` 聚合是否与 summary 一致

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

## 8. 与当前样例输出的关系

现有 `data/cuad/outputs/` 下的 JSONL 可直接用于复算 `precision/recall/f1`。如果你需要分析 LLM 调用参数和 token/时延，请使用自动追加后缀的新产物文件。
