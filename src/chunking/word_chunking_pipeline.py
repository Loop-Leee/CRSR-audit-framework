"""Word 分块流水线。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.tools.logger import Logger

from .chunk_boundary_rules import split_blocks, split_sections, split_sentences
from .chunk_merge_strategy import merge_overfine_chunks
from .semantic_chunk_scoring import chunk_section
from .word_text_extractor import extract_word_text


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


def build_hard_length_records(text: str, limit: int) -> list[dict]:
    """按固定长度执行硬切分。

    Args:
        text: 输入全文。
        limit: chunk 长度上限。

    Returns:
        list[dict]: 分块记录列表（仅按长度切分，不做语义边界优化）。

    Raises:
        ValueError: 当 `limit` 非正数时抛出。
    """

    if limit <= 0:
        raise ValueError("limit 必须大于 0。")

    records: list[dict] = []
    cursor = 0
    text_length = len(text)
    while cursor < text_length:
        raw_start = cursor
        raw_end = min(cursor + limit, text_length)
        segment = text[raw_start:raw_end]

        left = 0
        right = len(segment)
        while left < right and segment[left].isspace():
            left += 1
        while right > left and segment[right - 1].isspace():
            right -= 1

        if right > left:
            start = raw_start + left
            end = raw_start + right
            records.append(
                {
                    "start_offset": start,
                    "end_offset": end,
                    "content": text[start:end],
                }
            )
        cursor = raw_end

    return records


def process_word_file(path: Path, chunk_size_limit: int, output_dir: Path, logger: Logger | None = None) -> Path:
    """按语义分块处理单个 Word 文件并输出 chunk JSON。

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
        logger.info(f"开始分块: strategy=semantic, file={path}, limit={chunk_size_limit}")
    try:
        records = build_records(split_blocks(extract_word_text(path, logger=logger)), chunk_size_limit)
        output_path = _write_chunk_payload(path, records, output_dir)
        if logger:
            logger.info(f"分块完成: strategy=semantic, file={path}, chunks={len(records)}, output={output_path}")
        return output_path
    except Exception as error:
        if logger:
            logger.error(f"分块失败: strategy=semantic, file={path}, error={error}")
        raise


def process_word_file_hard_length(
    path: Path,
    chunk_size_limit: int,
    output_dir: Path,
    logger: Logger | None = None,
) -> Path:
    """按固定长度硬切分处理单个 Word 文件并输出 chunk JSON。

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
        logger.info(f"开始分块: strategy=hard_length, file={path}, limit={chunk_size_limit}")
    try:
        text = extract_word_text(path, logger=logger)
        records = build_hard_length_records(text=text, limit=chunk_size_limit)
        output_path = _write_chunk_payload(path, records, output_dir)
        if logger:
            logger.info(f"分块完成: strategy=hard_length, file={path}, chunks={len(records)}, output={output_path}")
        return output_path
    except Exception as error:
        if logger:
            logger.error(f"分块失败: strategy=hard_length, file={path}, error={error}")
        raise


def _write_chunk_payload(path: Path, records: list[dict], output_dir: Path) -> Path:
    """将分块记录写为统一 JSON 结构。"""

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
    return output_path
