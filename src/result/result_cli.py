"""result 命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tools.logger import get_logger

from .result_config import load_result_config
from .result_pipeline import run_result_with_diagnostics


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="将 review 风险结果按 span 注释写入 5-result 中的 Word 副本")
    parser.add_argument("--config", type=Path, default=None, help="result 配置文件路径")
    parser.add_argument("--input", type=Path, default=None, help="review 输入文件或目录")
    parser.add_argument("--output", type=Path, default=None, help="result 输出目录")
    parser.add_argument(
        "--target-results",
        type=str,
        default=None,
        help="需要写回的 result 标签，逗号分隔（默认：不合格,待复核）",
    )
    parser.add_argument("--disable-backup", action="store_true", help="不保留 source_copies 副本（.doc 源文件）")
    parser.add_argument("--ablation-no-writeback", action="store_true", help="消融：不写入 Word 批注，仅输出 preview")
    parser.add_argument("--ablation-no-chunk-offset", action="store_true", help="消融：定位时不使用 span_offset")
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _parse_target_results(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",")]
    normalized = [item for item in values if item]
    return normalized or None


def main() -> None:
    """执行 result 流程。"""

    args = parse_args()
    config = load_result_config(args.config)

    input_path = _resolve_path(args.input or config["input"])
    output_path = _resolve_path(args.output or config["output"])
    target_results = _parse_target_results(args.target_results) or list(config["target_results"])
    backup_enabled = bool(config["backup_enabled"]) and not args.disable_backup
    ablation_no_writeback = bool(config["ablation_no_writeback"]) or args.ablation_no_writeback
    ablation_no_chunk_offset = bool(config["ablation_no_chunk_offset"]) or args.ablation_no_chunk_offset

    logger = get_logger("result", log_dir=config["log_dir"])
    logger.info(f"日志文件: {logger.path}")

    result = run_result_with_diagnostics(
        input_path=input_path,
        output_dir=output_path,
        logger=logger,
        target_results=target_results,
        backup_enabled=backup_enabled,
        ablation_no_writeback=ablation_no_writeback,
        ablation_no_chunk_offset=ablation_no_chunk_offset,
    )

    for output in result.outputs:
        print(f"[OK] result -> {output}")
    print(f"[OK] trace -> {result.trace_path}")
    print(f"[OK] metrics -> {result.metrics_path}")


if __name__ == "__main__":
    main()
