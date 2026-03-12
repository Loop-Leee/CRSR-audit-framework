# src/exp_cuad/text_utils.py
from __future__ import annotations
import re
from typing import List

from src.chunking.chunk_boundary_rules import split_blocks, split_sections, split_sentences as split_block_sentences
from src.chunking.chunk_merge_strategy import merge_overfine_chunks
from src.chunking.semantic_chunk_scoring import chunk_section

_SENT_SPLIT = re.compile(r'(?<=[\.\?!])\s+|\n+')

def normalize_ws(s: str) -> str:
    """归一化空白字符并去掉首尾空格。"""
    return re.sub(r'\s+', ' ', s).strip()

def split_sentences(text: str) -> List[str]:
    """按标点和换行做轻量分句，并过滤过短噪声句。"""
    text = text.replace('\r\n', '\n')
    parts = [normalize_ws(x) for x in _SENT_SPLIT.split(text) if normalize_ws(x)]
    # 去掉过短噪声句
    return [p for p in parts if len(p) >= 10]

def chunk_by_tokens_approx(text: str, max_chars: int = 6000, overlap_chars: int = 800) -> List[str]:
    """按字符窗口近似分块，使用重叠区减少跨块信息丢失。"""
    # 72B 单卡 A100：建议每 chunk 控制在 4k~8k 字符级别（取决于 max_new_tokens）
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def chunk_by_semantic_jaccard_2gram(text: str, max_chars: int = 6000) -> List[str]:
    """按 2-gram Jaccard 语义低谷切分，尽量降低硬截断带来的语义断裂。

    Notes:
        - 该函数复用 `src/chunking` 模块的边界规则、语义评分与过细合并策略；
        - `max_chars` 仍是每个 chunk 的长度上限约束；
        - 不使用 `overlap_chars`，因为语义切分本身不依赖固定窗口重叠。
    """

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    blocks = split_blocks(text)
    if not blocks:
        return [text]

    records: list[dict] = []
    for section in split_sections(blocks):
        units: list[dict] = []
        for index, block in enumerate(section):
            if index == 0 and block["heading"]:
                units.append(block)
            elif len(block["text"]) > max_chars:
                units.extend(split_block_sentences(block))
            else:
                units.append(block)

        for chunk in chunk_section(units, max_chars):
            records.append(
                {
                    "start_offset": chunk[0]["start"],
                    "end_offset": chunk[-1]["end"],
                    "content": "\n\n".join(item["text"] for item in chunk),
                }
            )

    merged = merge_overfine_chunks(records, max_chars)
    chunks = [normalize_ws(item.get("content", "")) for item in merged if normalize_ws(item.get("content", ""))]
    return chunks or [text]

def sentence_set_jaccard(a: list[str], b: list[str]) -> float:
    """计算两个句子集合（归一化后）的 Jaccard 相似度。"""
    sa = {normalize_ws(x) for x in a if normalize_ws(x)}
    sb = {normalize_ws(x) for x in b if normalize_ws(x)}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
