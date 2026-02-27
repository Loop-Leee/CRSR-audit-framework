"""语义切分评分与候选点选择。"""

import re


def similarity(left: str, right: str) -> float:
    """计算相邻文本的 2-gram Jaccard 相似度。"""

    def grams(text: str) -> set[str]:
        text = re.sub(r"\s+", "", text)
        if len(text) <= 1:
            return {text} if text else set()
        return {text[i : i + 2] for i in range(len(text) - 1)}

    left_set, right_set = grams(left), grams(right)
    return (len(left_set & right_set) / len(left_set | right_set)) if left_set and right_set else 0.0


def choose_cut(units: list[dict], start: int, overflow: int, limit: int) -> int:
    """在可选区间内选择语义低谷切分点。"""

    target = max(1, int(limit * 0.55))
    best_cut, best_score, length = overflow, float("inf"), 0

    for index in range(start, overflow):
        length += len(units[index]["text"]) if index == start else len(units[index]["text"]) + 2
        cut = index + 1
        if length < target and cut != overflow:
            continue
        next_sim = similarity(units[index]["text"], units[cut]["text"]) if cut < len(units) else 1.0
        score = next_sim + abs(limit - length) / limit * 0.2
        if score < best_score:
            best_score, best_cut = score, cut
    return best_cut


def chunk_section(units: list[dict], limit: int) -> list[list[dict]]:
    """在单章节内按长度和语义边界切分。"""

    chunks, start = [], 0
    while start < len(units):
        length, end = 0, start
        while end < len(units):
            add = len(units[end]["text"]) if end == start else len(units[end]["text"]) + 2
            if end > start and length + add > limit:
                break
            length += add
            end += 1

        cut = end if end == len(units) else (start + 1 if end == start else choose_cut(units, start, end, limit))
        chunks.append(units[start:cut])
        start = cut

    return chunks
