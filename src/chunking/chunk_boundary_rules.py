"""分块边界规则：标题、条目、段落、句子。"""

import re


SENTENCE_RE = re.compile(r"[^。！？；.!?;\n]+[。！？；.!?;]?")
HEADING_RES = (
    re.compile(r"^第[一二三四五六七八九十百千万0-9]+[章节篇部卷条]"),
    re.compile(r"^附录[一二三四五六七八九十0-9A-Za-z]*"),
    re.compile(r"^[一二三四五六七八九十]+、\S+"),
    re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}\s+\S+"),
)
ITEM_RES = (
    re.compile(r"^[0-9]+[、.．]\s*"),
    re.compile(r"^[（(][0-9一二三四五六七八九十]+[）)]\s*"),
    re.compile(r"^[一二三四五六七八九十]+[、.．]\s*"),
)


def is_heading(line: str) -> bool:
    """判断是否为硬边界标题。"""

    line = line.strip()
    return bool(line and len(line) <= 80 and any(regex.match(line) for regex in HEADING_RES))


def is_item(line: str) -> bool:
    """判断是否为条目起始行。"""

    line = line.strip()
    return bool(line and any(regex.match(line) for regex in ITEM_RES))


def split_blocks(text: str) -> list[dict]:
    """按空行/标题/条目切分文本块并记录偏移。"""

    if not text:
        return []

    blocks, buffer = [], []
    start = end = None
    cursor = 0

    def flush() -> None:
        nonlocal start, end
        if not buffer:
            return
        content = "\n".join(buffer).strip()
        if content and start is not None and end is not None:
            blocks.append({"text": content, "start": start, "end": end, "heading": is_heading(buffer[0])})
        buffer.clear()
        start = end = None

    lines = text.split("\n")
    for index, raw_line in enumerate(lines):
        line_start, line_end = cursor, cursor + len(raw_line)
        line = raw_line.strip()

        if not line:
            flush()
        elif is_heading(line):
            flush()
            left, right = len(raw_line) - len(raw_line.lstrip()), len(raw_line.rstrip())
            blocks.append(
                {"text": line, "start": line_start + left, "end": line_start + right, "heading": True}
            )
        else:
            if buffer and is_item(line):
                flush()
            if not buffer:
                start = line_start + (len(raw_line) - len(raw_line.lstrip()))
            buffer.append(line)
            end = line_start + len(raw_line.rstrip())

        cursor = line_end + 1 if index < len(lines) - 1 else line_end

    flush()
    return blocks


def split_sections(blocks: list[dict]) -> list[list[dict]]:
    """按标题块将文本块分组为章节。"""

    sections, current = [], []
    for block in blocks:
        if block["heading"] and current:
            sections.append(current)
            current = [block]
        else:
            current.append(block)
    if current:
        sections.append(current)
    return sections


def split_sentences(block: dict) -> list[dict]:
    """将超长块按句子拆分。"""

    parts = []
    for match in SENTENCE_RE.finditer(block["text"]):
        seg = match.group()
        text = seg.strip()
        if not text:
            continue
        ltrim = len(seg) - len(seg.lstrip())
        rtrim = len(seg) - len(seg.rstrip())
        parts.append(
            {
                "text": text,
                "start": block["start"] + match.start() + ltrim,
                "end": block["start"] + match.end() - rtrim,
                "heading": False,
            }
        )
    return parts or [block]
