"""过细分块合并策略。"""


def merge_overfine_chunks(records: list[dict], limit: int) -> list[dict]:
    """合并可与前后都安全合并的过细 chunk。"""

    def total_len(items: list[dict]) -> int:
        return sum(len(x["content"]) for x in items) + max(0, len(items) - 1) * 2

    def join(items: list[dict]) -> dict:
        return {
            "start_offset": items[0]["start_offset"],
            "end_offset": items[-1]["end_offset"],
            "content": "\n\n".join(x["content"] for x in items),
        }

    merged, changed = records[:], True
    while changed and len(merged) >= 3:
        changed = False
        index = 1
        while index < len(merged) - 1:
            left, mid, right = merged[index - 1], merged[index], merged[index + 1]
            left_pair, right_pair = total_len([left, mid]), total_len([mid, right])
            if left_pair <= limit and right_pair <= limit:
                if total_len([left, mid, right]) <= limit:
                    merged[index - 1 : index + 2] = [join([left, mid, right])]
                elif left_pair <= right_pair:
                    merged[index - 1 : index + 1] = [join([left, mid])]
                else:
                    merged[index : index + 2] = [join([mid, right])]
                changed = True
                index = max(1, index - 1)
                continue
            index += 1

    return merged
