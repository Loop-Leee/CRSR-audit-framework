"""result 模块配置加载与路径解析。"""

from __future__ import annotations

import json
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "result_config.json"


def _resolve_path(raw: str | None, fallback: str) -> Path:
    """解析配置中的相对/绝对路径。"""

    value = raw or fallback
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def load_result_config(config_path: Path | None = None) -> dict[str, object]:
    """读取 result 模块配置。"""

    path = config_path or DEFAULT_CONFIG_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))

    raw_target_results = raw.get("target_results", ["不合格", "待复核"])
    target_results: list[str]
    if isinstance(raw_target_results, list):
        target_results = [str(item).strip() for item in raw_target_results if str(item).strip()]
    else:
        target_results = ["不合格", "待复核"]

    if not target_results:
        target_results = ["不合格", "待复核"]

    return {
        "input": _resolve_path(raw.get("input_dir"), "data/4-review"),
        "output": _resolve_path(raw.get("output_dir"), "data/6-result"),
        "log_dir": _resolve_path(raw.get("log_dir"), "log/result"),
        "target_results": target_results,
        "backup_enabled": _as_bool(raw.get("backup_enabled"), True),
        "ablation_no_writeback": _as_bool(raw.get("ablation_no_writeback"), False),
        "ablation_no_chunk_offset": _as_bool(raw.get("ablation_no_chunk_offset"), False),
    }

