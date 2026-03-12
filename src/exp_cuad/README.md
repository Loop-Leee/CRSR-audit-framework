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
  text_utils.py       # 分句、归一化、窗口分块、语义分块、句子集合 Jaccard
  eval_metrics.py     # PairResult 与核心指标计算函数
  run_infer.py        # CLI 主入口：推理 + 结构化产物 + 汇总
```

职责边界：

- `run_infer.py` 只负责“编排与产物落盘”；指标定义集中在 `eval_metrics.py`
- 通用日志统一走 `src/tools/logger.py`，日志路径固定为 `log/exp_cuad/*.log`

### 1.1 工作流程（端到端）

当前 CUAD 标注实验按以下步骤执行：

1. 数据加载与标准化  
   `cuad_loader.py` 从本地或 HF 读取 CUAD，整理为文档级样本：`doc_id + text + spans(按label)`.
2. 标签集合构建  
   在全部样本上收集标签并排序，形成统一标签空间（每个文档都按同一标签集评估）。
3. 文档分块  
   默认使用 `text_utils.chunk_by_semantic_jaccard_2gram`（2-gram Jaccard 语义低谷切分）；  
   可用 `--chunk_strategy window` 切回硬窗口分块（消融）。
4. chunk 级推理  
   对每个 `(doc_id, label, chunk)` 构造 prompt，调用 LLM，解析 JSON；解析失败会重试一次。
5. pair 级聚合  
   对同一 `(doc_id, label)` 的多个 chunk 结果聚合：  
   `present_pred = any(chunk_present)`，证据句去重合并，保留 `chunk_votes`。
6. 结果落盘  
   写出主结果 `*.jsonl`（pair 级）、过程事件 `*.progress.jsonl`、摘要 `*.summary.json`。
7. 指标汇总  
   运行结束后计算 `presence_micro/macro_f1/evidence_jaccard/laziness/inconsistency` 并写入 summary。
8. 最终分析  
   使用 `analyze_metrics.py` 对 `cuad_baseline*.jsonl` 进行统一口径复算，得到 `*.metrics.json` 作为报告指标。

### 1.2 Jaccard 在实验中的两个作用

`Jaccard` 在本实验里有两个独立用途：

1. 分块阶段（语义切分）  
   `chunk_strategy=semantic_jaccard_2gram` 时，使用“相邻单元的 2-gram Jaccard 相似度”寻找语义低谷切点，尽量减少硬截断。
2. 评估阶段（证据质量）  
   `metrics.evidence_jaccard_gt_present` 用句子集合 Jaccard 比较 `evidence_pred` 与 `evidence_gt`：`|A ∩ B| / |A ∪ B|`。

注意：这两个 Jaccard 不是同一个对象。前者用于“如何切”，后者用于“切完后证据对不对”。

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
  --chunk_strategy semantic_jaccard_2gram \
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
| `--chunk_strategy` | 分块策略：`semantic_jaccard_2gram/window` | `semantic_jaccard_2gram` |
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

| 参数 | 作用 | 取值高/开启时常见现象 | 取值低/关闭时常见现象 | 建议 |
|---|---|---|---|---|
| `--model` | 决定推理能力上限 | 更强模型通常 `recall/f1` 更高，但成本更高 | 弱模型常见 `fn` 增多、`laziness_rate` 上升 | 跨模型对比时仅修改此项 |
| `--backend` | 决定请求协议与兼容行为 | 稳定后端会降低空内容/解析失败 | 不稳定后端会拉低有效召回并放大噪声 | 同组实验固定后端（推荐 `openai`） |
| `--mode` | 决定提示词策略 | 更强约束模式常提升精确率 | 弱约束模式可能提升召回但引入误报 | 报告必须记录并固定 |
| `--chunk_strategy` | 选择分块算法 | `semantic_jaccard_2gram` 通常更少语义截断，召回更稳 | `window` 更快但更容易切断语义边界 | 默认语义分块，`window` 用作消融对照 |
| `--max_chars` | 控制单 chunk 长度 | 过大易触发长推理、超时、空内容 | 过小会丢上下文，`recall` 易下降 | 与 `--overlap_chars` 成对调参并固定 |
| `--overlap_chars` | 控制 window 分块的重叠上下文 | 适度增大可缓解跨块断裂、提升召回 | 过小易漏检边界条款；过大增加成本与重复 | 仅在 `chunk_strategy=window` 时生效 |
| `--max_new_tokens` | 控制单次输出上限 | 过大易出现“思考耗尽配额”或超时 | 过小会截断 JSON，导致解析失败 | 先用 `256`，异常时优先下调 |
| `--openai_no_think_prompt` | 在 prompt 末尾加 `/no_think` | 开启时可减少只输出 reasoning 的概率 | 关闭后在推理模型上更易出现空 content | `openai` 后端建议开启 |
| `--openai_disable_thinking` | 请求层禁用 thinking | 开启时可显著降低 `finish_reason=length` 空响应 | 关闭后更容易消耗在 reasoning 而非最终 JSON | `openai` 后端建议开启 |
| `--openai_send_max_new_tokens_param` | 额外发送兼容 token 参数 | 开启时更容易被不同网关正确限流 | 关闭后部分网关可能忽略上限字段 | `openai` 后端建议开启 |
| `--split`/`--data_source` | 决定样本来源与分布 | 更难 split 指标可能下降（属数据差异） | 更易 split 指标可能偏高（不可横比） | 横向对比必须固定 |
| `--limit_docs` | 控制评估规模 | 小样本波动大，方差高 | 全量更稳定但耗时更长 | 最终结果建议 `0`（全量） |

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
- 指标报告应与实验参数一起归档（至少记录 `model/backend/mode/chunk_strategy/max_chars/overlap_chars/max_new_tokens`）

### 5.3 指标字段解释（含高低含义）

`analyze_metrics.py` 输出里的 `metrics` 字段建议按下表解读：

| 字段 | 定义 | 高值通常说明 | 低值通常说明 |
|---|---|---|---|
| `metrics.presence_micro.precision` | `tp / (tp + fp)` | 误报少，预测为 present 时更可信 | 误报多，规则过宽或模型过激进 |
| `metrics.presence_micro.recall` | `tp / (tp + fn)` | 漏报少，召回能力好 | 漏报多，常见于上下文不足或模型保守 |
| `metrics.presence_micro.f1` | precision/recall 调和均值 | 综合识别效果好 | 精确率或召回率至少一项偏弱 |
| `metrics.presence_micro.accuracy` | `(tp + tn) / total` | 整体判别正确率高 | 全局错误较多；需结合类别分布看 |
| `metrics.presence_micro.specificity` | `tn / (tn + fp)` | 对“本不存在”识别好，误报低 | 负样本经常被误判为 present |
| `metrics.presence_micro.balanced_accuracy` | `(recall + specificity) / 2` | 正负样本平衡表现好 | 至少一侧（正或负）性能不佳 |
| `metrics.presence_micro.tp/fp/fn/tn` | 混淆矩阵计数 | `tp/tn` 高说明有效命中多 | `fp/fn` 高分别代表误报/漏报问题 |
| `metrics.presence_macro.macro_f1` | 按 label 先算 F1 再平均 | 各条款类型表现更均衡 | 存在“长尾标签表现差”问题 |
| `metrics.presence_macro.num_labels` | 参与 macro 统计的标签数 | 覆盖标签更全 | 标签覆盖不全，结果可比性下降 |
| `metrics.evidence_jaccard_gt_present` | 仅在 GT present 上比较证据集合 Jaccard 均值 | 证据抽取贴近标注，span 质量高 | 证据偏离标注，抽取命中不足 |
| `metrics.laziness_rate` | GT present 但预测 absent 比例 | 高值代表“保守/偷懒”倾向强 | 低值代表漏报控制较好 |
| `metrics.inconsistency_rate` | 同一样本 chunk 投票同时有真有假的比例 | 高值代表分块间判断不稳定 | 低值代表 chunk 级判断更一致 |

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
