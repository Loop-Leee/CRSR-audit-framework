"""轻量级 .env 加载器。"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_file(dotenv_path: Path) -> None:
    """读取 .env 并写入环境变量（不覆盖已存在值）。"""

    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
