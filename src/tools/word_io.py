"""Word 文件读写工具。"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import platform
from shutil import which
from pathlib import Path

from .logger import Logger


_WORD_FORMAT_BY_SUFFIX = {
    ".doc": "doc",
    ".docx": "docx",
}


def extract_word_text(path: Path, logger: Logger | None = None) -> str:
    """使用 macOS `textutil` 提取 Word 纯文本。

    Args:
        path: Word 文件路径。
        logger: 可选模块日志对象。

    Returns:
        str: 归一化换行后的纯文本（去除首尾空白）。

    Raises:
        FileNotFoundError: 输入文件不存在时抛出。
        RuntimeError: `textutil` 不可用或提取失败时抛出。
    """

    if not path.exists():
        raise FileNotFoundError(f"Word 文件不存在: {path}")

    try:
        if logger:
            logger.info(f"提取 Word 文本: {path}")
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

    text = _normalize_newlines(out).strip()
    if logger:
        logger.info(f"Word 文本提取完成: {path}, chars={len(text)}")
    return text


def write_word_text(
    *,
    source_path: Path,
    text: str,
    logger: Logger | None = None,
    backup_path: Path | None = None,
) -> None:
    """将纯文本写回 Word 文件（覆盖原文件）。

    Args:
        source_path: 目标 Word 文件路径（将被覆盖）。
        text: 待写入文本。
        logger: 可选模块日志对象。
        backup_path: 可选备份文件路径；提供时先复制原文件。

    Returns:
        None

    Raises:
        FileNotFoundError: 源文件不存在时抛出。
        ValueError: 文件扩展名不支持时抛出。
        RuntimeError: `textutil` 不可用或转换失败时抛出。
    """

    if not source_path.exists():
        raise FileNotFoundError(f"Word 文件不存在: {source_path}")

    target_format = _WORD_FORMAT_BY_SUFFIX.get(source_path.suffix.lower())
    if not target_format:
        raise ValueError(f"暂不支持的 Word 扩展名: {source_path.suffix}")

    if backup_path is not None:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, backup_path)
        if logger:
            logger.info(f"已创建源文件备份: {backup_path}")

    normalized = _normalize_newlines(text).strip() + "\n"
    try:
        with tempfile.TemporaryDirectory(prefix="crsr_result_") as tmp_dir:
            text_path = Path(tmp_dir) / "annotated.txt"
            text_path.write_text(normalized, encoding="utf-8")
            subprocess.run(
                ["textutil", "-convert", target_format, str(text_path), "-output", str(source_path)],
                check=True,
                capture_output=True,
                text=True,
            )
    except FileNotFoundError as error:
        if logger:
            logger.error("textutil 不存在，无法写回 Word 文本。")
        raise RuntimeError("未找到 textutil。") from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if error.stderr else "unknown error"
        if logger:
            logger.error(f"textutil 写回失败: {source_path} -> {stderr}")
        raise RuntimeError(f"写回失败: {source_path} -> {stderr}") from error

    if logger:
        logger.info(f"Word 写回完成: {source_path}")


def convert_word_to_docx(
    *,
    source_path: Path,
    target_path: Path,
    logger: Logger | None = None,
) -> Path:
    """将 Word 文件转换为 `.docx`（优先使用 LibreOffice）。

    Args:
        source_path: 输入 Word 文件路径（`.doc` 或 `.docx`）。
        target_path: 输出 `.docx` 路径。
        logger: 可选模块日志对象。

    Returns:
        Path: 实际输出路径（即 `target_path`）。

    Raises:
        FileNotFoundError: 输入文件不存在时抛出。
        ValueError: 输出路径不是 `.docx` 时抛出。
        RuntimeError: 缺少可用转换器或转换失败时抛出。
    """

    if not source_path.exists():
        raise FileNotFoundError(f"Word 文件不存在: {source_path}")
    if target_path.suffix.lower() != ".docx":
        raise ValueError(f"target_path 必须是 .docx: {target_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    converter = resolve_docx_converter()
    if converter is None:
        raise RuntimeError(
            "未找到可用的 .doc -> .docx 转换器。请安装 LibreOffice（soffice/libreoffice/lowriter），"
            "避免使用 textutil 造成格式丢失。"
            f"{build_docx_converter_install_hint()}"
        )

    if logger:
        logger.info(f"Word 转换开始: source={source_path}, target={target_path}, converter={converter}")

    with tempfile.TemporaryDirectory(prefix="crsr_docx_convert_") as tmp_dir:
        temp_out_dir = Path(tmp_dir)
        source_name = source_path.name
        try:
            subprocess.run(
                [converter, "--headless", "--convert-to", "docx", "--outdir", str(temp_out_dir), str(source_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            stderr = error.stderr.strip() if error.stderr else "unknown error"
            raise RuntimeError(f"Word 转换失败: {source_path} -> {stderr}") from error

        converted = temp_out_dir / f"{Path(source_name).stem}.docx"
        if not converted.exists():
            raise RuntimeError(f"转换器未产出目标文件: {converted}")
        shutil.move(str(converted), str(target_path))

    if logger:
        logger.info(f"Word 转换完成: {target_path}")
    return target_path


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def resolve_docx_converter() -> str | None:
    """返回可用的 `.doc -> .docx` 转换命令。

    Returns:
        str | None: 优先级顺序为 `soffice`、`libreoffice`、`lowriter`；
        若都不存在，返回 `None`。
    """

    for command in ("soffice", "libreoffice", "lowriter"):
        if which(command):
            return command
    return None


def build_docx_converter_install_hint() -> str:
    """返回当前系统的 LibreOffice 安装提示。"""

    system_name = platform.system().lower()
    if system_name == "darwin":
        return (
            " macOS 安装示例：`brew install --cask libreoffice`；"
            "若 `which soffice` 仍为空，可执行："
            "`echo 'export PATH=\"/Applications/LibreOffice.app/Contents/MacOS:$PATH\"' >> ~/.zshrc && source ~/.zshrc`。"
        )
    if system_name == "linux":
        return " Linux 安装示例：`sudo apt-get update && sudo apt-get install -y libreoffice`。"
    if system_name == "windows":
        return " Windows 安装示例：`winget install TheDocumentFoundation.LibreOffice`。"
    return ""
