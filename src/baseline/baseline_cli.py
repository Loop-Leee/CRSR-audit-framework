"""baseline 命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.llm import DEFAULT_LLM_CONFIG_PATH, load_llm_settings
from src.tools.logger import get_logger

from .baseline_config import load_baseline_config
from .baseline_pipeline import run_baseline_with_diagnostics


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="直接对 chunk 结果执行全规则 baseline 审查")
    parser.add_argument("--config", type=Path, default=None, help="baseline 配置文件路径")
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG_PATH, help="全局 LLM 配置路径")
    parser.add_argument("--input", type=Path, default=None, help="chunk 输入文件或目录")
    parser.add_argument("--output", type=Path, default=None, help="review 格式输出目录")
    parser.add_argument("--risk-info", type=Path, default=None, help="risk_info.csv 路径")
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="待审核",
        help="写入每条 review_item 的人工标注初值（默认：待审核）",
    )
    parser.add_argument("--disable-llm", action="store_true", help="禁用 LLM（仅用于流程调试）")
    parser.add_argument("--disable-llm-concurrency", action="store_true", help="关闭 LLM 并发，改为串行")
    parser.add_argument("--ablation-no-rules", action="store_true", help="消融：不注入审查规则文本")
    parser.add_argument("--ablation-no-span-offset", action="store_true", help="消融：不计算 span_offset")
    parser.add_argument(
        "--schema-retry-limit",
        type=int,
        default=None,
        help="LLM 输出 schema 失败时的重试上限（0-3，默认读取配置）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="单次生成最大 token（默认读取配置，建议 256-1024）",
    )
    parser.add_argument(
        "--openai-no-think-prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否在用户提示词末尾追加 /no_think（默认读取配置）",
    )
    parser.add_argument(
        "--openai-disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否在请求中附带 enable_thinking=false（默认读取配置）",
    )
    parser.add_argument(
        "--openai-send-max-new-tokens-param",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否额外发送 max_new_tokens/max_completion_tokens（默认读取配置）",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def main() -> None:
    """执行 baseline 流程。"""

    args = parse_args()
    config = load_baseline_config(args.config)
    input_path = _resolve_path(args.input or config["input"])
    output_path = _resolve_path(args.output or config["output"])
    risk_info_path = _resolve_path(args.risk_info or config["risk_info"])
    schema_retry_limit = int(
        args.schema_retry_limit if args.schema_retry_limit is not None else config["schema_retry_limit"]
    )
    max_new_tokens = int(args.max_new_tokens if args.max_new_tokens is not None else config["max_new_tokens"])
    openai_no_think_prompt = (
        bool(args.openai_no_think_prompt)
        if args.openai_no_think_prompt is not None
        else bool(config["openai_no_think_prompt"])
    )
    openai_disable_thinking = (
        bool(args.openai_disable_thinking)
        if args.openai_disable_thinking is not None
        else bool(config["openai_disable_thinking"])
    )
    openai_send_max_new_tokens_param = (
        bool(args.openai_send_max_new_tokens_param)
        if args.openai_send_max_new_tokens_param is not None
        else bool(config["openai_send_max_new_tokens_param"])
    )

    llm_config_path = _resolve_path(args.llm_config)
    llm_settings = load_llm_settings(
        config_path=llm_config_path,
        enabled_override=False if args.disable_llm else None,
        concurrent_enabled_override=False if args.disable_llm_concurrency else None,
    )

    logger = get_logger("baseline", log_dir=config["log_dir"])
    logger.info(f"日志文件: {logger.path}")

    result = run_baseline_with_diagnostics(
        input_path=input_path,
        output_dir=output_path,
        risk_info_path=risk_info_path,
        llm_settings=llm_settings,
        logger=logger,
        ablation_no_rules=args.ablation_no_rules,
        ablation_no_span_offset=args.ablation_no_span_offset,
        schema_retry_limit=schema_retry_limit,
        max_new_tokens=max_new_tokens,
        openai_no_think_prompt=openai_no_think_prompt,
        openai_disable_thinking=openai_disable_thinking,
        openai_send_max_new_tokens_param=openai_send_max_new_tokens_param,
        ground_truth=args.ground_truth,
    )

    for output in result.outputs:
        print(f"[OK] baseline reviewed -> {output}")
    print(f"[OK] trace -> {result.trace_path}")
    print(f"[OK] metrics -> {result.metrics_path}")


if __name__ == "__main__":
    main()
