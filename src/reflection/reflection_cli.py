"""reflection 命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.llm import DEFAULT_LLM_CONFIG_PATH, load_llm_settings
from src.tools.logger import get_logger

from .reflection_config import load_reflection_config
from .reflection_pipeline import run_reflection_with_diagnostics


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="对 review_items 执行后验反思校正")
    parser.add_argument("--config", type=Path, default=None, help="reflection 配置文件路径")
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG_PATH, help="全局 LLM 配置路径")
    parser.add_argument("--input", type=Path, default=None, help="review 输入文件或目录")
    parser.add_argument("--output", type=Path, default=None, help="reflection 输出目录")
    parser.add_argument("--reflection-version", type=str, default=None, help="reflection 版本号")
    parser.add_argument("--stage1-threshold", type=int, default=None, help="Stage-1 触发阈值")
    parser.add_argument("--stage2-max-items", type=int, default=None, help="Stage-2 每组最大送审样本")
    parser.add_argument("--evidence-max-chars", type=int, default=None, help="证据窗口最大字符")
    parser.add_argument("--chunk-excerpt-max-chars", type=int, default=None, help="chunk 回退窗口最大字符")
    parser.add_argument("--disable-llm", action="store_true", help="禁用 LLM（仅用于流程调试）")
    parser.add_argument("--disable-llm-concurrency", action="store_true", help="关闭 LLM 并发，改为串行")
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def main() -> None:
    """执行 reflection 流程。"""

    args = parse_args()
    config = load_reflection_config(args.config)

    input_path = _resolve_path(args.input or config["input"])
    output_path = _resolve_path(args.output or config["output"])
    reflection_version = str(args.reflection_version or config["reflection_version"]).strip() or "v1"
    stage1_threshold = int(
        args.stage1_threshold if args.stage1_threshold is not None else config["stage1_threshold"]
    )
    stage2_max_items = int(
        args.stage2_max_items if args.stage2_max_items is not None else config["stage2_max_items"]
    )
    evidence_max_chars = int(
        args.evidence_max_chars if args.evidence_max_chars is not None else config["evidence_max_chars"]
    )
    chunk_excerpt_max_chars = int(
        args.chunk_excerpt_max_chars
        if args.chunk_excerpt_max_chars is not None
        else config["chunk_excerpt_max_chars"]
    )

    llm_config_path = _resolve_path(args.llm_config)
    llm_settings = load_llm_settings(
        config_path=llm_config_path,
        enabled_override=False if args.disable_llm else None,
        concurrent_enabled_override=False if args.disable_llm_concurrency else None,
    )

    logger = get_logger("reflection", log_dir=config["log_dir"])
    logger.info(f"日志文件: {logger.path}")

    result = run_reflection_with_diagnostics(
        input_path=input_path,
        output_dir=output_path,
        llm_settings=llm_settings,
        logger=logger,
        reflection_version=reflection_version,
        stage1_threshold=stage1_threshold,
        stage2_max_items=stage2_max_items,
        evidence_max_chars=evidence_max_chars,
        chunk_excerpt_max_chars=chunk_excerpt_max_chars,
    )

    for output in result.outputs:
        print(f"[OK] reflected -> {output}")
    print(f"[OK] trace -> {result.trace_path}")
    print(f"[OK] metrics -> {result.metrics_path}")


if __name__ == "__main__":
    main()
