"""reflection 模块配置加载与路径解析。"""

from __future__ import annotations

import json
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "reflection_config.json"


def _resolve_path(raw: str | None, fallback: str) -> Path:
    value = raw or fallback
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_reflection_config(config_path: Path | None = None) -> dict[str, Path | int | str]:
    """读取 reflection 配置。"""

    path = config_path or DEFAULT_CONFIG_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))

    stage1_threshold = int(raw.get("stage1_threshold", 4))
    stage2_max_items = int(raw.get("stage2_max_items", 6))
    evidence_max_chars = int(raw.get("evidence_max_chars", 360))
    chunk_excerpt_max_chars = int(raw.get("chunk_excerpt_max_chars", 520))

    if stage1_threshold < 0:
        stage1_threshold = 0
    if stage2_max_items <= 0:
        stage2_max_items = 6
    if evidence_max_chars < 120:
        evidence_max_chars = 120
    if chunk_excerpt_max_chars < 200:
        chunk_excerpt_max_chars = 200

    return {
        "input": _resolve_path(raw.get("input_dir"), "data/4-review"),
        "output": _resolve_path(raw.get("output_dir"), "data/5-reflection"),
        "log_dir": _resolve_path(raw.get("log_dir"), "log/reflection"),
        "reflection_version": str(raw.get("reflection_version", "v1")).strip() or "v1",
        "stage1_threshold": stage1_threshold,
        "stage2_max_items": stage2_max_items,
        "evidence_max_chars": evidence_max_chars,
        "chunk_excerpt_max_chars": chunk_excerpt_max_chars,
    }
