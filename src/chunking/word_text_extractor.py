"""Word 文本提取与文件发现。"""

import subprocess
from pathlib import Path


SUPPORTED_EXTENSIONS = {".doc", ".docx"}


def extract_word_text(path: Path) -> str:
    """使用 macOS textutil 提取 Word 纯文本。"""

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


def discover_word_files(input_dir: Path) -> list[Path]:
    """扫描输入目录中的 .doc/.docx 文件。"""

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith("~$")
    )
