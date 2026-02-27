# CRSR-audit-framework
基于 LLM 的合同审查通用方法框架。


# 整体工作流程
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

## Word 分块工具

`word_chunker.py` 用于批量解析 `data/` 目录下的 Word 文件（`.doc`/`.docx`），并按“硬边界 + 语义边界”进行分块。

### 运行方式

```bash
python3 word_chunker.py --config chunker_config.json --chunk-size 1500
```

如果不传 `--chunk-size`，程序会在命令行提示输入分块大小。

### 配置文件说明

默认配置文件为 `chunker_config.json`：

```json
{
  "min_chunk_size": 1000,
  "max_chunk_size": 20000,
  "input_dir": "data/1-original",
  "output_dir": "output/2-chunks"
}
```

- `min_chunk_size`: 允许的最小分块大小（字符数）
- `max_chunk_size`: 允许的最大分块大小（字符数）
- `input_dir`: Word 输入目录（可选，默认 `data`）
- `output_dir`: 分块结果输出目录（可选，默认 `output/chunks`）

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
- `doc_id`（文档唯一标识）
- 来源文件路径
- 每个分块的：
  - `chunk_id`
  - `start_offset`（在原文中的起始字符位置）
  - `end_offset`（在原文中的结束字符位置）
  - `content`

### 主要接口与方法

- `load_config(config_path)`: 读取 JSON 配置并校验范围。
- `extract_text(path)`: 使用系统 `textutil` 提取 Word 文本。
- `split_blocks(text)`: 按硬边界切成文本块。
- `split_sentences(block)`: 将超长块按句子切分。
- `chunk_section(units, limit)`: 在长度约束下按语义边界切分章节单元。
- `merge_overfine_chunks(records, limit)`: 合并可与前后都安全合并的过细 chunk。
- `build_records(blocks, limit)`: 执行硬边界 + 语义边界 + 过细合并混合分块。
- `process_file(path, size_limit, output_dir)`: 处理单个文件并输出 JSON。
- `discover_files(input_dir)`: 扫描输入目录中的 `.doc`/`.docx` 文件。

### 依赖说明

- 使用 conda 创建隔离环境（推荐）：
```bash
conda env create -f environment.yml
conda activate crsr-audit
```
- 当前实现依赖 macOS 系统 `textutil` 命令解析 Word 文本。
