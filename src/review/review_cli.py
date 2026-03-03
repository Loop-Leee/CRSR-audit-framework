"""Review 命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.llm import DEFAULT_LLM_CONFIG_PATH, load_llm_settings
from src.tools.logger import get_logger

from .review_config import load_review_config
from .review_pipeline import run_review_with_diagnostics


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="对 classified 输出执行规则化审查")
    parser.add_argument("--config", type=Path, default=None, help="review 配置文件路径")
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG_PATH, help="全局 LLM 配置路径")
    parser.add_argument("--input", type=Path, default=None, help="classified 输入文件或目录")
    parser.add_argument("--output", type=Path, default=None, help="review 输出目录")
    parser.add_argument("--rules", type=Path, default=None, help="expanded 规则文件路径（csv/json）")
    parser.add_argument("--rule-version", type=str, default=None, help="规则版本号，默认读取配置")
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="待审核",
        help="写入每条 review_item 的人工标注初值（默认：待审核）",
    )
    parser.add_argument("--disable-llm", action="store_true", help="禁用 LLM（仅用于流程调试）")
    parser.add_argument("--disable-llm-concurrency", action="store_true", help="关闭 LLM 并发，改为串行")
    parser.add_argument("--ablation-no-rules", action="store_true", help="消融：不注入规则列表")
    parser.add_argument("--ablation-coarse-rules", action="store_true", help="消融：每个 risk_type 使用单条合并规则")
    parser.add_argument(
        "--schema-retry-limit",
        type=int,
        default=None,
        help="LLM 输出 schema 失败时的重试上限（0-3，默认读取配置）",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def main() -> None:
    """执行 review 流程。"""

    args = parse_args()
    if args.ablation_no_rules and args.ablation_coarse_rules:
        raise ValueError("--ablation-no-rules 与 --ablation-coarse-rules 不能同时启用。")

    config = load_review_config(args.config)
    input_path = _resolve_path(args.input or config["input"])
    output_path = _resolve_path(args.output or config["output"])
    rules_path = _resolve_path(args.rules or config["rules"])
    rule_version = str(args.rule_version or config["rule_version"]).strip() or "v1"
    schema_retry_limit = int(
        args.schema_retry_limit if args.schema_retry_limit is not None else config["schema_retry_limit"]
    )

    llm_config_path = _resolve_path(args.llm_config)
    llm_settings = load_llm_settings(
        config_path=llm_config_path,
        enabled_override=False if args.disable_llm else None,
        concurrent_enabled_override=False if args.disable_llm_concurrency else None,
    )

    logger = get_logger("review", log_dir=config["log_dir"])
    logger.info(f"日志文件: {logger.path}")

    result = run_review_with_diagnostics(
        input_path=input_path,
        output_dir=output_path,
        rules_path=rules_path,
        rule_version=rule_version,
        llm_settings=llm_settings,
        logger=logger,
        ablation_no_rules=args.ablation_no_rules,
        ablation_coarse_rules=args.ablation_coarse_rules,
        schema_retry_limit=schema_retry_limit,
        ground_truth=args.ground_truth,
    )

    for output in result.outputs:
        print(f"[OK] reviewed -> {output}")
    print(f"[OK] trace -> {result.trace_path}")
    print(f"[OK] metrics -> {result.metrics_path}")


if __name__ == "__main__":
    main()
