"""Classification 命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.llm import DEFAULT_LLM_CONFIG_PATH, load_llm_settings

from .classification_config import load_classification_config
from .classification_logger import ClassificationLogger
from .classification_pipeline import run_classification


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="对 chunk 执行审查类型分类")
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径")
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG_PATH, help="全局 LLM 配置路径")
    parser.add_argument("--input", type=Path, default=None, help="输入文件或目录（默认读取配置）")
    parser.add_argument("--output", type=Path, default=None, help="输出目录（默认读取配置）")
    parser.add_argument("--disable-llm", action="store_true", help="禁用语义匹配，仅保留关键词匹配")
    parser.add_argument("--disable-llm-concurrency", action="store_true", help="关闭 LLM 并发调用，改为串行")
    return parser.parse_args()


def main() -> None:
    """执行分类流程。"""

    args = parse_args()
    config = load_classification_config(args.config)

    input_path = args.input or config["input"]
    output_path = args.output or config["output"]
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    llm_config_path = args.llm_config
    if not llm_config_path.is_absolute():
        llm_config_path = Path.cwd() / llm_config_path

    llm_settings = load_llm_settings(
        config_path=llm_config_path,
        enabled_override=False if args.disable_llm else None,
        concurrent_enabled_override=False if args.disable_llm_concurrency else None,
    )

    logger = ClassificationLogger(config["log_dir"])
    logger.info(f"日志文件: {logger.path}")

    outputs = run_classification(
        input_path=input_path,
        output_dir=output_path,
        risk_info_path=config["risk_info"],
        llm_settings=llm_settings,
        logger=logger,
    )

    for output in outputs:
        print(f"[OK] classified -> {output}")


if __name__ == "__main__":
    main()
