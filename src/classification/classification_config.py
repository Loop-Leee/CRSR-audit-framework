"""分类模块配置加载与路径解析。"""

from __future__ import annotations

import json
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "classifier_config.json"


def _resolve_path(raw: str | None, fallback: str) -> Path:
    """解析配置中的相对/绝对路径。"""

    value = raw or fallback
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_classification_config(config_path: Path | None = None) -> dict:
    """读取分类配置。"""

    path = config_path or DEFAULT_CONFIG_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))

    return {
        "input": _resolve_path(raw.get("input_dir"), "data/2-chunks"),
        "output": _resolve_path(raw.get("output_dir"), "data/3-classified"),
        "risk_info": _resolve_path(raw.get("risk_info_path"), "prompt/risk_info.csv"),
        "log_dir": _resolve_path(raw.get("log_dir"), "log/classification"),
    }
