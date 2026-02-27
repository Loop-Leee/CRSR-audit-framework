"""Word 文档硬边界+语义边界分块工具。"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


SUPPORTED_EXTENSIONS = {".doc", ".docx"}
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


def load_config(path: Path) -> dict:
    """读取配置并填充默认路径。"""

    raw = json.loads(path.read_text(encoding="utf-8"))
    low, high = int(raw["min_chunk_size"]), int(raw["max_chunk_size"])
    if low <= 0 or high <= 0 or low > high:
        raise ValueError("配置错误：min_chunk_size/max_chunk_size 不合法。")
    return {
        "min": low,
        "max": high,
        "input": Path(raw.get("input_dir", "data/original")),
        "output": Path(raw.get("output_dir", "data/chunks")),
    }


def extract_text(path: Path) -> str:
    """使用 textutil 提取 Word 文本。"""

    try:
        out = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except FileNotFoundError as error:
        raise RuntimeError("未找到 textutil。") from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if error.stderr else "unknown error"
        raise RuntimeError(f"解析失败: {path} -> {stderr}") from error
    return out.replace("\r\n", "\n").replace("\r", "\n").strip()


def is_heading(line: str) -> bool:
    """判断是否为硬边界标题行。"""

    line = line.strip()
    return bool(line and len(line) <= 80 and any(r.match(line) for r in HEADING_RES))


def is_item(line: str) -> bool:
    """判断是否为条目起始行。"""

    line = line.strip()
    return bool(line and any(r.match(line) for r in ITEM_RES))


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
    for i, raw in enumerate(lines):
        line_start, line_end = cursor, cursor + len(raw)
        line = raw.strip()

        if not line:
            flush()
        elif is_heading(line):
            flush()
            left, right = len(raw) - len(raw.lstrip()), len(raw.rstrip())
            blocks.append({"text": line, "start": line_start + left, "end": line_start + right, "heading": True})
        else:
            if buffer and is_item(line):
                flush()
            if not buffer:
                start = line_start + (len(raw) - len(raw.lstrip()))
            buffer.append(line)
            end = line_start + len(raw.rstrip())

        cursor = line_end + 1 if i < len(lines) - 1 else line_end

    flush()
    return blocks


def split_sentences(block: dict) -> list[dict]:
    """将超长块按句子切分。"""

    parts = []
    for m in SENTENCE_RE.finditer(block["text"]):
        seg = m.group()
        text = seg.strip()
        if not text:
            continue
        ltrim = len(seg) - len(seg.lstrip())
        rtrim = len(seg) - len(seg.rstrip())
        parts.append(
            {
                "text": text,
                "start": block["start"] + m.start() + ltrim,
                "end": block["start"] + m.end() - rtrim,
                "heading": False,
            }
        )
    return parts or [block]


def similarity(a: str, b: str) -> float:
    """计算相邻单元语义近似度（2-gram Jaccard）。"""

    def grams(text: str) -> set[str]:
        text = re.sub(r"\s+", "", text)
        if len(text) <= 1:
            return {text} if text else set()
        return {text[i : i + 2] for i in range(len(text) - 1)}

    ga, gb = grams(a), grams(b)
    return (len(ga & gb) / len(ga | gb)) if ga and gb else 0.0


def choose_cut(units: list[dict], start: int, overflow: int, limit: int) -> int:
    """在候选区间中选择语义断点。"""

    target = max(1, int(limit * 0.55))
    best_cut, best_score, length = overflow, float("inf"), 0

    for i in range(start, overflow):
        length += len(units[i]["text"]) if i == start else len(units[i]["text"]) + 2
        cut = i + 1
        if length < target and cut != overflow:
            continue
        next_sim = similarity(units[i]["text"], units[cut]["text"]) if cut < len(units) else 1.0
        score = next_sim + abs(limit - length) / limit * 0.2
        if score < best_score:
            best_score, best_cut = score, cut
    return best_cut


def chunk_section(units: list[dict], limit: int) -> list[list[dict]]:
    """对同一章节执行长度+语义混合切分。"""

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


def merge_overfine_chunks(records: list[dict], limit: int) -> list[dict]:
    """合并可与前后都安全合并的过细分块。"""

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
        i = 1
        while i < len(merged) - 1:
            left, mid, right = merged[i - 1], merged[i], merged[i + 1]
            left_pair, right_pair = total_len([left, mid]), total_len([mid, right])
            if left_pair <= limit and right_pair <= limit:
                if total_len([left, mid, right]) <= limit:
                    merged[i - 1 : i + 2] = [join([left, mid, right])]
                elif left_pair <= right_pair:
                    merged[i - 1 : i + 1] = [join([left, mid])]
                else:
                    merged[i : i + 2] = [join([mid, right])]
                changed = True
                i = max(1, i - 1)
                continue
            i += 1
    return merged


def build_records(blocks: list[dict], limit: int) -> list[dict]:
    """执行硬边界+语义边界分块并做过细合并。"""

    sections, current = [], []
    for block in blocks:
        if block["heading"] and current:
            sections.append(current)
            current = [block]
        else:
            current.append(block)
    if current:
        sections.append(current)

    records = []
    for section in sections:
        units = []
        for i, block in enumerate(section):
            if i == 0 and block["heading"]:
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
                    "content": "\n\n".join(x["text"] for x in chunk),
                }
            )

    return merge_overfine_chunks(records, limit)


def process_file(path: Path, size_limit: int, output_dir: Path) -> Path:
    """处理单个文件并写出 JSON。"""

    records = build_records(split_blocks(extract_text(path)), size_limit)
    payload = {
        "doc_id": "doc_" + hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12],
        "source_file": str(path),
        "chunks": [
            {
                "chunk_id": i,
                "start_offset": r["start_offset"],
                "end_offset": r["end_offset"],
                "content": r["content"],
            }
            for i, r in enumerate(records, start=1)
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{path.stem}.chunks.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def discover_files(input_dir: Path) -> list[Path]:
    """扫描输入目录下的 Word 文件。"""

    return sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith("~$")
    )


def main() -> None:
    """程序入口。"""

    parser = argparse.ArgumentParser(description="按硬边界和语义边界切分 Word")
    parser.add_argument("--config", type=Path, default=Path("chunker_config.json"), help="配置文件路径")
    parser.add_argument("--chunk-size", type=int, default=None, help="分块大小（字符数）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    requested = args.chunk_size if args.chunk_size is not None else int(input("请输入分块大小（字符数）: ").strip())
    effective = max(cfg["min"], min(requested, cfg["max"]))

    files = discover_files(cfg["input"])
    if not files:
        raise FileNotFoundError(f"未找到 Word 文件: {cfg['input']}")

    print("开始处理: requested_chunk_size=%s, effective_chunk_size=%s, file_count=%s" % (requested, effective, len(files)))
    for path in files:
        print(f"[OK] {path} -> {process_file(path, effective, cfg['output'])}")


if __name__ == "__main__":
    main()
