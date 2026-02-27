"""Word 分块流水线。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .chunk_boundary_rules import split_blocks, split_sections, split_sentences
from .chunk_merge_strategy import merge_overfine_chunks
from .semantic_chunk_scoring import chunk_section
from .word_text_extractor import extract_word_text
from src.tools.logger import Logger


def build_records(blocks: list[dict], limit: int) -> list[dict]:
    """执行硬边界切分、语义切分和过细合并。

    Args:
        blocks: 预切分文本块。
        limit: chunk 长度上限。

    Returns:
        list[dict]: 分块记录列表。
    """

    records = []
    for section in split_sections(blocks):
        units = []
        for index, block in enumerate(section):
            if index == 0 and block["heading"]:
                units.append(block)
            elif len(block["text"]) > limit:
                units.extend(split_sentences(block))
            else:
                units.append(block)

        for chunk in chunk_section(units, limit):
            records.append(
                {
                    "start_offset": chunk[0]["start"],
                    "end_offset": chunk[-1]["end"],
                    "content": "\n\n".join(item["text"] for item in chunk),
                }
            )

    return merge_overfine_chunks(records, limit)


def process_word_file(path: Path, chunk_size_limit: int, output_dir: Path, logger: Logger | None = None) -> Path:
    """处理单个 Word 文件并输出 chunk JSON。

    Args:
        path: 输入 Word 文件路径。
        chunk_size_limit: 生效 chunk 长度上限。
        output_dir: 输出目录。
        logger: 可选模块日志对象。

    Returns:
        Path: 输出文件路径。

    Raises:
        RuntimeError: 提取文本失败时抛出。
        OSError: 写文件失败时抛出。
    """

    if logger:
        logger.info(f"开始分块: file={path}, limit={chunk_size_limit}")
    try:
        records = build_records(split_blocks(extract_word_text(path, logger=logger)), chunk_size_limit)
        payload = {
            "doc_id": "doc_" + hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12],
            "source_file": str(path),
            "chunks": [
                {
                    "chunk_id": index,
                    "start_offset": record["start_offset"],
                    "end_offset": record["end_offset"],
                    "content": record["content"],
                }
                for index, record in enumerate(records, start=1)
            ],
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{path.stem}.chunks.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if logger:
            logger.info(f"分块完成: file={path}, chunks={len(records)}, output={output_path}")
        return output_path
    except Exception as error:
        if logger:
            logger.error(f"分块失败: file={path}, error={error}")
        raise
