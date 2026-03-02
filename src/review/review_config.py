"""审查模块配置加载与路径解析。"""

from __future__ import annotations

import json
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "review_config.json"


def _resolve_path(raw: str | None, fallback: str) -> Path:
    """解析配置中的相对/绝对路径。"""

    value = raw or fallback
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_review_config(config_path: Path | None = None) -> dict[str, Path | str | int]:
    """读取 review 模块配置。"""

    path = config_path or DEFAULT_CONFIG_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    schema_retry_limit = int(raw.get("schema_retry_limit", 2))
    if schema_retry_limit < 0:
        schema_retry_limit = 0

    return {
        "input": _resolve_path(raw.get("input_dir"), "data/3-classified"),
        "output": _resolve_path(raw.get("output_dir"), "data/4-review"),
        "rules": _resolve_path(raw.get("rules_path"), "prompt/rule_hits_expanded.csv"),
        "log_dir": _resolve_path(raw.get("log_dir"), "log/review"),
        "rule_version": str(raw.get("rule_version", "v1")),
        "schema_retry_limit": schema_retry_limit,
    }
