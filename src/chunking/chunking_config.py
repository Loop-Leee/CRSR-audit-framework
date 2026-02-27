"""分块配置加载与校验。"""

from __future__ import annotations

import json
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "chunker_config.json"


def load_chunking_config(config_path: Path | None = None) -> dict:
    """读取配置并解析输入输出目录。"""

    path = config_path or DEFAULT_CONFIG_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))

    low = int(raw["min_chunk_size"])
    high = int(raw["max_chunk_size"])
    if low <= 0 or high <= 0 or low > high:
        raise ValueError("配置错误：min_chunk_size/max_chunk_size 不合法。")

    input_dir = Path(raw.get("input_dir", "data/1-original"))
    output_dir = Path(raw.get("output_dir", "data/2-chunks"))
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    return {"min": low, "max": high, "input": input_dir, "output": output_dir}


def clamp_chunk_size(size: int, low: int, high: int) -> int:
    """将分块大小限制在配置区间。"""

    return max(low, min(size, high))
