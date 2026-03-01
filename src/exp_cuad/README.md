# exp_cuad 模块说明

`exp_cuad` 用于在 CUAD 数据集上执行条款存在性识别与证据句提取实验，并输出可复现、可统计的结构化结果。

## 1. 模块目标

- 数据输入标准化：兼容 HuggingFace `theatticusproject/cuad` 的 QA 结构与常见变体字段
- 推理流程统一：按 `(document, label)` 执行分块推理与聚合
- 结构化产物统一：逐条写出 JSONL 记录，便于后处理与对比
- 指标口径统一：输出 presence、evidence 与稳定性相关指标
- 消融开关可控：支持 prompt、后端、分块策略、样本规模等维度消融

## 2. 目录结构（模块内）

```text
src/exp_cuad/
  cuad_loader.py       # 数据加载与字段兼容
  prompts.py           # baseline / crsr_lite 提示词模板
  text_utils.py        # 分句、归一化、近似分块
  eval_metrics.py      # 指标与 PairResult 定义
  run_infer.py         # CLI 主入口：推理 + 写出 + 汇总
```

## 3. 最小可复现流程

### 3.1 环境

```bash
conda env create -f environment.yml
```
```bash
conda activate crsr-audit
```

确保可写输出目录存在：

```bash
mkdir -p outputs
```

### 3.2 推荐命令（macOS / OpenAI 兼容服务）

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
  --out_jsonl outputs/cuad_baseline.jsonl
```

### 3.3 其他后端

- `--backend transformers`：本地 `transformers + torch` 推理
- `--backend vllm`：Linux + GPU 推荐；macOS 通常不可用

## 4. CLI 参数说明（run_infer.py）

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model` | 模型名（后端实际加载/调用的 model id） | 必填 |
| `--split` | 数据集切分名（`test/val/...` 会映射到 `train`） | `train` |
| `--backend` | 推理后端：`vllm` / `transformers` / `openai` | `vllm` |
| `--mode` | 提示词策略：`baseline` / `crsr_lite` | `baseline` |
| `--max_chars` | 分块最大字符数 | `6000` |
| `--overlap_chars` | 分块重叠字符数 | `800` |
| `--max_new_tokens` | 单次生成最大输出 token | `256` |
| `--limit_docs` | 限制文档数（0 表示全部） | `0` |
| `--out_jsonl` | 逐条预测输出文件 | `cuad_preds.jsonl` |

## 5. 结构化输出定义

### 5.1 明细文件：`--out_jsonl`

每行对应一个 `(doc_id, label)` 预测样本：

```json
{
  "doc_id": "cuad_xxx",
  "label": "Governing Law",
  "present_pred": true,
  "evidence_pred": ["..."],
  "present_gt": true,
  "evidence_gt": ["..."],
  "chunk_votes": [true, false, true],
  "mode": "baseline",
  "backend": "openai"
}
```

字段说明：

- `present_pred`：聚合后的最终存在性预测（按 chunk 投票 `any`）
- `evidence_pred`：跨 chunk 去重后的证据句集合
- `present_gt` / `evidence_gt`：由 CUAD 标注 span 映射得到的对齐结果
- `chunk_votes`：每个 chunk 的 `present` 布尔序列，用于稳定性分析

### 5.2 终端汇总（JSON）

运行结束后标准输出会打印一份汇总 JSON，包含：

- 基本信息：`mode/backend/model/split/docs/pairs/time_sec/out_jsonl`
- presence 指标：`presence_micro.{precision,recall,f1,tp,fp,fn,tn}`
- `presence_macro_f1`
- `evidence_jaccard_on_present_gt`
- `laziness_rate`
- `inconsistency_rate`

## 6. 可统计指标口径

由 `src/exp_cuad/eval_metrics.py` 统一定义：

- `compute_presence_f1`：全样本 micro precision/recall/f1
- `compute_macro_f1`：按 label 分组 F1 再平均（macro-F1）
- `compute_evidence_jaccard`：仅在 GT 为 present 的样本上计算证据集合 Jaccard 均值
- `compute_laziness_rate`：GT present 但预测 absent 的比例
- `compute_inconsistency_rate`：同一样本 chunk 投票同时含 `True/False` 的比例

## 7. 消融开关矩阵（建议）

| 维度 | 开关 | 目的 |
|---|---|---|
| 提示词策略 | `--mode baseline/crsr_lite` | 验证结构化约束提示是否提升稳定性 |
| 推理后端 | `--backend openai/transformers/vllm` | 比较服务端推理与本地推理差异 |
| 分块粒度 | `--max_chars` | 观察上下文长度对召回/误报影响 |
| 分块重叠 | `--overlap_chars` | 验证跨块信息丢失与重复噪声权衡 |
| 生成长度 | `--max_new_tokens` | 检查证据提取完整性与冗余输出 |
| 样本规模 | `--limit_docs` | 快速烟测 vs 全量评估 |

推荐最小消融对比：

```bash
# A: baseline prompt
python3 -m src.exp_cuad.run_infer --model "$LLM_MODEL" --backend openai --mode baseline --limit_docs 20 --out_jsonl outputs/cuad_a.jsonl

# B: crsr_lite prompt
python3 -m src.exp_cuad.run_infer --model "$LLM_MODEL" --backend openai --mode crsr_lite --limit_docs 20 --out_jsonl outputs/cuad_b.jsonl
```

## 8. 失败路径与排查

- `ModuleNotFoundError: datasets`  
  安装 `datasets`：`conda install -n crsr-audit -c conda-forge datasets`
- `NonMatchingSplitsSizesError`  
  代码会自动尝试 `force_redownload`；若仍失败，清理本地 HF 缓存后重试
- `vllm` 在 macOS 不可用  
  改用 `--backend transformers` 或 `--backend openai`
- `out_jsonl` 目录不存在  
  先 `mkdir -p` 对应目录
- 模型输出非 JSON  
  `parse_json_loose()` 会做容错提取并回退到 `present=false, evidence=[]` 的保底结构

## 9. 扩展建议

- 将 `load_label_descriptions()` 接入官方条款定义文件，减少标签语义歧义
- 在 `aggregate_chunks()` 增加阈值/多数投票策略，对抗长文档误报
- 接入统一日志模块 `src/tools/logger.py`，把运行日志沉淀到 `log/exp_cuad/*.log`
- 增加结果汇总脚本，将终端 summary 持久化为 `metrics.json`
