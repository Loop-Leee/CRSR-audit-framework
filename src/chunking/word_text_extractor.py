"""Word 文本提取与文件发现。"""

from __future__ import annotations

from pathlib import Path

from src.tools.logger import Logger
from src.tools.word_io import extract_word_text as _extract_word_text


SUPPORTED_EXTENSIONS = {".doc", ".docx"}


def extract_word_text(path: Path, logger: Logger | None = None) -> str:
    """使用 macOS `textutil` 提取 Word 纯文本。

    Args:
        path: Word 文件路径。
        logger: 可选模块日志对象。

    Returns:
        str: 归一化换行后的纯文本。

    Raises:
        RuntimeError: `textutil` 不可用或提取失败时抛出。
    """

    return _extract_word_text(path=path, logger=logger)


def discover_word_files(input_dir: Path, logger: Logger | None = None) -> list[Path]:
    """扫描输入目录中的 `.doc/.docx` 文件。

    Args:
        input_dir: 输入目录。
        logger: 可选模块日志对象。

    Returns:
        list[Path]: 按路径排序后的文件列表。
    """

    files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith("~$")
    )
    if logger:
        logger.info(f"发现待处理 Word 文件: dir={input_dir}, count={len(files)}")
    return files
