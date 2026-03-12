# result 模块

模块功能：读取 `review` 输出，筛选 `result` 为 `不合格` 或 `待复核` 的条目，基于 `span` 定位后在 `data/5-result` 中的 Word 副本写入“批注（comments）”。

## 依赖

- `python-docx>=1.2.0`（用于写入 Word comments）
- LibreOffice（`soffice/libreoffice/lowriter`，用于 `.doc -> .docx` 转换）

安装示例：

```bash
# macOS
brew install --cask libreoffice
echo 'export PATH="/Applications/LibreOffice.app/Contents/MacOS:$PATH"' >> ~/.zshrc
source ~/.zshrc
which soffice
```

## 运行方式

默认运行：

```bash
python3 main.py result
```

指定输入输出：

```bash
python3 main.py result --input data/4-review --output data/6-result
```

指定目标结果标签：

```bash
python3 main.py result --target-results 不合格,待复核
```

## 消融实验命令

1. 不写回源文件（仅生成 preview）：

```bash
python3 main.py result --ablation-no-writeback
```

2. 定位时禁用 `span_offset`：

```bash
python3 main.py result --ablation-no-chunk-offset
```

## 输入输出

- 输入：`*.review.json`（来自 `review`）
- 不修改：`review_items.source_file` 指向的源文件
- 输出：
  - `data/5-result/annotated_docs/*.result.docx`（带批注副本）
  - `data/5-result/source_copies/*`（仅源文件为 `.doc` 且开启 `backup_enabled` 时）
  - 同名冲突时自动追加 `.result.2.docx/.result.3.docx` 后缀
- 运行级产物：
  - `data/5-result/result_trace.jsonl`（条目级处理诊断）
  - `data/5-result/result_metrics.json`（聚合指标）
  - `data/5-result/*.annotated.preview.txt`（仅 `ablation_no_writeback`）

批注内容格式：

```text
风险类型：<risk_type>
审查结论：<result>
违反的条款：<rule_hit>
修改建议: <suggest>
```

说明：
- 批注写入为 Word comments，不改动正文文字内容。
- 当源文件为 `.doc` 时，会先复制后转换为 `.docx` 副本并在该副本上写入批注。
- `.doc` 转换不再使用 `textutil`，避免表格乱码与版式劣化。
- 写入后会再次提取文本并校验“批注前后正文一致”，若不一致则报错终止。

## 定位策略

按以下优先级定位 `span`：

1. `chunk.start_offset + review_item.span_offset` 作为锚点，在锚点附近窗口内做 `span` 精确匹配
2. 若失败，退化为 chunk 窗口内 `span` 精确匹配

说明：不使用全局模糊匹配，避免跨 chunk 误命中。

写回时会再次在 DOCX 段落文本（包含表格内段落）中检索 `span`，并优先按锚点选择最近命中，再将批注挂载到对应 run 区间，减少“偏前/偏后”现象。

## 关键指标字段

- `review_item_count`：扫描到的 review 条目总数。
- `target_item_count`：`result` 命中目标标签的条目数。
- `located_item_count`：`span` 成功定位条目数。
- `planned_item_count`：去重后计划插入注释条目数。
- `applied_item_count`：成功写入批注的条目数。
- `duplicate_item_count`：同一运行中 `risk_id` 重复而跳过的条目数。
- `locate_failed_count`：`span` 无法定位条目数。
- `source_missing_count`：源文件路径缺失或文件不存在条目数。
- `ablation_no_writeback_count`：`ablation_no_writeback` 下仅预览不写回条目数。
- `converted_doc_count`：`.doc` 转换为 `.docx` 的文件数。

## 日志

- 日志目录：`log/result/`
- 关键步骤：`[info]`
- 错误路径：`[error]`

## 常见错误

- 报错：`检测到 .doc 源文件，但未找到可用的 .doc -> .docx 转换器`
  - 原因：当前系统缺少 LibreOffice 命令行（`soffice/libreoffice/lowriter`）。
  - 处理：安装 LibreOffice 后，确保 `which soffice` 有返回。
