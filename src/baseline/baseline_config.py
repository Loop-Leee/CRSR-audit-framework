"""baseline 模块配置加载与路径解析。"""

from __future__ import annotations

import json
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "baseline_config.json"


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


def load_baseline_config(config_path: Path | None = None) -> dict[str, Path | int | bool]:
    """读取 baseline 模块配置。"""

    path = config_path or DEFAULT_CONFIG_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    schema_retry_limit = int(raw.get("schema_retry_limit", 1))
    if schema_retry_limit < 0:
        schema_retry_limit = 0
    max_new_tokens = int(raw.get("max_new_tokens", 512))
    if max_new_tokens <= 0:
        max_new_tokens = 512

    return {
        "input": _resolve_path(raw.get("input_dir"), "data/2-chunks"),
        "output": _resolve_path(raw.get("output_dir"), "data/4-review-baseline"),
        "risk_info": _resolve_path(raw.get("risk_info_path"), "prompt/risk_info.csv"),
        "log_dir": _resolve_path(raw.get("log_dir"), "log/baseline"),
        "schema_retry_limit": schema_retry_limit,
        "max_new_tokens": max_new_tokens,
        "openai_no_think_prompt": _as_bool(raw.get("openai_no_think_prompt"), True),
        "openai_disable_thinking": _as_bool(raw.get("openai_disable_thinking"), True),
        "openai_send_max_new_tokens_param": _as_bool(raw.get("openai_send_max_new_tokens_param"), True),
    }
