"""Word 文本提取与文件发现。"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.tools.logger import Logger


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

    try:
        if logger:
            logger.info(f"提取文本: {path}")
        out = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except FileNotFoundError as error:
        if logger:
            logger.error("textutil 不存在，无法提取 Word 文本。")
        raise RuntimeError("未找到 textutil。") from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if error.stderr else "unknown error"
        if logger:
            logger.error(f"textutil 提取失败: {path} -> {stderr}")
        raise RuntimeError(f"解析失败: {path} -> {stderr}") from error

    text = out.replace("\r\n", "\n").replace("\r", "\n").strip()
    if logger:
        logger.info(f"文本提取完成: {path}, chars={len(text)}")
    return text


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
