# Chunking 模块

模块功能：将 Word 文档切分为语义连续的更小的块，并生成 JSON 文件。

具体来说，该模块通过硬边界切分和语义边界切分，将长段落划分为多个块，并通过合并策略将过细的块合并为更紧凑的块。最终输出每个输入文件的块信息，包含块ID、起始位置、结束位置和内容。


## 运行方式
```bash
python3 main.py --chunk-size 1500
```

如果不传 `--chunk-size`，程序会在命令行提示输入分块大小。

## 配置文件
默认配置文件：`src/chunking/chunker_config.json`

```json
{
  "min_chunk_size": 1000,
  "max_chunk_size": 20000,
  "input_dir": "data/1-original",
  "output_dir": "data/2-chunks"
}
```

- `min_chunk_size`: 允许的最小分块大小（字符数）
- `max_chunk_size`: 允许的最大分块大小（字符数）
- `input_dir`: Word 输入目录（可选，默认 `data/1-original`）
- `output_dir`: 分块结果输出目录（可选，默认 `data/2-chunks`）

用户输入的 `chunk-size` 会被强制限制在 `[min_chunk_size, max_chunk_size]` 区间内。

### 分块逻辑

1. 硬边界切分：
- 先按空行、标题行（如 `第X条` / `1.2` / `一、`）和条目行切成文本块。
- 再按标题块划分章节，章节之间不互相混合，避免把新条款塞进上一个 chunk。

2. 语义边界切分：
- 当章节内容超过长度限制时，先按句子拆分长段落。
- 在可选切分点中，选择“相邻单元语义相似度低谷”的位置作为边界。
- 语义相似度采用 2-gram Jaccard 近似计算，保证实现简洁且无需额外模型依赖。

3. 过细分块合并：
- 若某个 chunk 同时满足“与前一个 chunk 合并不超上限”且“与后一个 chunk 合并不超上限”，则触发合并。
- 优先尝试三者（前/当前/后）一次合并；若超限则合并到更紧凑的一侧，减少碎片化 chunk。

### 输出格式

每个输入文件会生成一个 `*.chunks.json` 文件，包含：
- `doc_id`
- `source_file` (来源文件路径)
- `chunks[]`
  - `chunk_id`
  - `start_offset`（在原文中的起始字符位置）
  - `end_offset`（在原文中的结束字符位置）
  - `content`

## 关键模块职责
- `chunking_config.py`: 配置读取、路径解析、范围钳制。
- `word_text_extractor.py`: Word 文本提取与文件发现。
- `chunk_boundary_rules.py`: 标题/条目规则、块切分、句子切分。
- `semantic_chunk_scoring.py`: 语义评分与可选切分点选择。
- `chunk_merge_strategy.py`: 过细 chunk 合并策略。
- `word_chunking_pipeline.py`: 串联完整分块流程并写出 JSON。
- `chunking_cli.py`: 命令行入口。
- `src/tools/logger.py`: 通用日志模块，统一输出到 `log/chunking/*.log`。

## 日志

- 日志目录：`log/chunking/`
- 关键步骤：`[info]`
- 错误路径：`[error]`
