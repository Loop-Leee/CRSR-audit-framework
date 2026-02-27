"""分类模块日志工具。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class ClassificationLogger:
    """将关键步骤和错误写入分类日志文件。"""

    def __init__(self, log_dir: Path) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self._log_dir / f"classification_{timestamp}.log"

    def info(self, message: str) -> None:
        """记录信息日志。"""

        self._write("info", message)

    def error(self, message: str) -> None:
        """记录错误日志。"""

        self._write("error", message)

    def _write(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} [{level}] {message}"
        print(line)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
