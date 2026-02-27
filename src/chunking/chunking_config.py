"""分块配置加载与校验。"""

from __future__ import annotations

import json
from pathlib import Path

from src.tools.logger import Logger


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "chunker_config.json"


def load_chunking_config(config_path: Path | None = None, logger: Logger | None = None) -> dict:
    """读取配置并解析输入输出目录。

    Args:
        config_path: 配置文件路径，默认为模块目录下 `chunker_config.json`。
        logger: 可选模块日志对象。

    Returns:
        dict: 标准化后的配置字典，包含 `min/max/input/output`。

    Raises:
        ValueError: 配置中分块范围非法时抛出。
        FileNotFoundError: 配置文件不存在时抛出。
        json.JSONDecodeError: 配置文件不是合法 JSON 时抛出。
    """

    path = config_path or DEFAULT_CONFIG_PATH
    if logger:
        logger.info(f"加载 chunking 配置: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    low = int(raw["min_chunk_size"])
    high = int(raw["max_chunk_size"])
    if low <= 0 or high <= 0 or low > high:
        if logger:
            logger.error(
                "配置校验失败: min_chunk_size=%s, max_chunk_size=%s" % (raw.get("min_chunk_size"), raw.get("max_chunk_size"))
            )
        raise ValueError("配置错误：min_chunk_size/max_chunk_size 不合法。")

    input_dir = Path(raw.get("input_dir", "data/1-original"))
    output_dir = Path(raw.get("output_dir", "data/2-chunks"))
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if logger:
        logger.info("配置生效: min=%s, max=%s, input=%s, output=%s" % (low, high, input_dir, output_dir))
    return {"min": low, "max": high, "input": input_dir, "output": output_dir}


def clamp_chunk_size(size: int, low: int, high: int) -> int:
    """将分块大小限制在配置区间。

    Args:
        size: 用户输入分块大小。
        low: 最小允许值。
        high: 最大允许值。

    Returns:
        int: 生效分块大小。
    """

    return max(low, min(size, high))
