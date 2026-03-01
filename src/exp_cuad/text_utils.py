# src/exp_cuad/text_utils.py
from __future__ import annotations
import re
from typing import List

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

def sentence_set_jaccard(a: list[str], b: list[str]) -> float:
    """计算两个句子集合（归一化后）的 Jaccard 相似度。"""
    sa = {normalize_ws(x) for x in a if normalize_ws(x)}
    sb = {normalize_ws(x) for x in b if normalize_ws(x)}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
