# Reflection 模块

模块功能：读取 `review` 输出（`*.review.json`），执行两阶段后验校正：

1. Stage-1（item-level）
- 先用启发式 `fp_risk_score` 筛选高风险误报项（默认 `score >= 4`）
- 仅对候选项调用 LLM
- 保守修正：`keep / downgrade / reject / revise`

2. Stage-2（group-level）
- 按 `(doc_id, risk_type)` 分组
- 仅对冲突组调用 LLM
- 每组最多送审 `stage2_max_items`（默认 6）

## 运行方式

默认运行：

```bash
python3 main.py reflection
```

指定输入输出：

```bash
python3 main.py reflection --input data/4-review --output data/5-reflection
```

调整阈值与组样本上限：

```bash
python3 main.py reflection --stage1-threshold 4 --stage2-max-items 6
```

## 输入输出

- 输入：`*.review.json`
- 输出：`*.reflection.json`
- 运行级产物：
  - `reflection_trace.jsonl`（每次 Stage-1/Stage-2 调用诊断）
  - `reflection_metrics.json`（聚合指标）

`*.reflection.json` 在原 payload 基础上新增：

- `review_items_original`：原始 review 结果备份
- `reflection_meta`：分阶段计数与 token 指标
- `reflection_items`：反思后的最终 item 列表

`reflection_meta` 关键字段：

- `generated_at`
- `reflection_version`
- `rule_version`
- `stage1_candidate_count`
- `stage1_called_count`
- `stage2_group_count`
- `stage2_candidate_group_count`
- `stage2_called_group_count`
- `stage1_adjusted_count`
- `stage2_adjusted_count`
- `final_item_count`
- `avg_token_in`
- `avg_token_out`
- `avg_total_token`

## 调试字段

`reflection_items` 会保留以下运行时字段，便于追踪：

- `fp_risk_score`
- `fp_risk_flags`
- `evidence_window`
- `evidence_window_source`
- `reflection_stage1_action`
- `reflection_stage1_reason`
- `reflection_stage2_action`
- `reflection_stage2_reason`

## 评测（review_eval）

reflection 结果可直接复用 `experiment` 的 `review_eval` 口径评测：

```bash
python3 -m src.experiment.review_eval \
  --gold-dir dataset/standard-review \
  --pred-dir data/5-reflection
```

默认 `pred_items_field=auto`，会优先读取 `reflection_items`，若不存在则回退 `review_items`。
