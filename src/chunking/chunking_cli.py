"""Chunking 命令行入口。"""

import argparse
from pathlib import Path

from .chunking_config import clamp_chunk_size, load_chunking_config
from .word_chunking_pipeline import process_word_file
from .word_text_extractor import discover_word_files


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="按硬边界和语义边界切分 Word")
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径")
    parser.add_argument("--chunk-size", type=int, default=None, help="分块大小（字符数）")
    return parser.parse_args()


def main() -> None:
    """运行 chunking 主流程。"""

    args = parse_args()
    config = load_chunking_config(args.config)
    requested = args.chunk_size if args.chunk_size is not None else int(input("请输入分块大小（字符数）: ").strip())
    effective = clamp_chunk_size(requested, config["min"], config["max"])

    files = discover_word_files(config["input"])
    if not files:
        raise FileNotFoundError(f"未找到 Word 文件: {config['input']}")

    print(
        "开始处理: requested_chunk_size=%s, effective_chunk_size=%s, file_count=%s"
        % (requested, effective, len(files))
    )
    for path in files:
        output = process_word_file(path, effective, config["output"])
        print(f"[OK] {path} -> {output}")
