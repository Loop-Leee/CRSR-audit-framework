"""统一实验入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from src.chunking.chunking_config import clamp_chunk_size, load_chunking_config
from src.chunking.word_chunking_pipeline import process_word_file
from src.chunking.word_text_extractor import discover_word_files
from src.classification.classification_config import load_classification_config
from src.classification.classification_pipeline import run_classification_with_diagnostics
from src.llm import DEFAULT_LLM_CONFIG_PATH, LLMSettings, load_llm_settings
from src.tools.logger import Logger, get_logger

from .artifact_writer import ArtifactWriter
from .metrics import ExperimentMetrics, compute_experiment_metrics


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """实验配置。"""

    run_name: str
    run_id: str
    timestamp: str
    input_dir: str
    risk_info_path: str
    output_root: str
    chunk_size: int
    mode: str
    model: str
    temperature: float
    concurrent_enabled: bool
    max_concurrency: int
    cache_enabled: bool
    cache_path: str
    ground_truth_path: str | None

    def to_dict(self) -> dict[str, Any]:
        """导出配置字典。"""

        return asdict(self)


def parse_args() -> argparse.Namespace:
    """解析实验入口参数。"""

    parser = argparse.ArgumentParser(description="统一实验入口：chunking + classification + metrics + artifacts")
    parser.add_argument("--run-name", type=str, default="experiment", help="实验名称前缀")
    parser.add_argument("--input-dir", type=Path, default=None, help="原始 Word 输入目录")
    parser.add_argument("--risk-info", type=Path, default=None, help="风险类型 CSV 路径")
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments"), help="实验输出根目录")
    parser.add_argument("--ground-truth", type=Path, default=None, help="标注集路径（目录/JSON/JSONL）")
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG_PATH, help="LLM 配置文件路径")
    parser.add_argument("--chunk-size", type=int, default=1000, help="分块字符上限")
    parser.add_argument("--mode", choices=["keyword_only", "keyword_llm"], default="keyword_llm", help="实验模式")
    parser.add_argument("--model", type=str, default=None, help="覆盖模型名称")
    parser.add_argument("--temperature", type=float, default=None, help="覆盖温度参数")
    parser.add_argument("--max-concurrency", type=int, default=None, help="覆盖并发数")
    parser.add_argument("--disable-llm-concurrency", action="store_true", help="关闭 LLM 并发")
    parser.add_argument("--disable-cache", action="store_true", help="关闭 LLM 缓存")
    return parser.parse_args()


def main() -> None:
    """执行统一实验流程。"""

    args = parse_args()
    logger = get_logger("experiment")
    logger.info(f"日志文件: {logger.path}")

    chunking_config = load_chunking_config()
    classification_config = load_classification_config()

    input_dir = _resolve_path(args.input_dir or chunking_config["input"])
    risk_info_path = _resolve_path(args.risk_info or classification_config["risk_info"])
    output_root = _resolve_path(args.output_root)
    ground_truth_path = _resolve_path(args.ground_truth) if args.ground_truth else None
    llm_config_path = _resolve_path(args.llm_config)
    chunk_size = clamp_chunk_size(args.chunk_size, chunking_config["min"], chunking_config["max"])

    llm_settings = _build_llm_settings(args, llm_config_path)
    run_config = _build_experiment_config(
        args=args,
        input_dir=input_dir,
        risk_info_path=risk_info_path,
        output_root=output_root,
        chunk_size=chunk_size,
        llm_settings=llm_settings,
        ground_truth_path=ground_truth_path,
    )

    logger.info(f"实验配置: {json.dumps(run_config.to_dict(), ensure_ascii=False)}")

    writer = ArtifactWriter(output_root=output_root, logger=logger)
    layout = writer.prepare_layout(run_config.run_id)
    _write_run_config(run_config, layout.metrics_dir, logger)

    files = discover_word_files(input_dir, logger=logger)
    if not files:
        logger.error(f"未找到 Word 文件: {input_dir}")
        raise FileNotFoundError(f"未找到 Word 文件: {input_dir}")
    logger.info(f"实验输入文件数: {len(files)}")

    for path in files:
        process_word_file(path, chunk_size, layout.chunks_dir, logger=logger)

    classification_result = run_classification_with_diagnostics(
        input_path=layout.chunks_dir,
        output_dir=layout.classified_dir,
        risk_info_path=risk_info_path,
        llm_settings=llm_settings,
        logger=logger,
    )
    metrics = compute_experiment_metrics(
        classified_files=classification_result.outputs,
        diagnostics=classification_result.chunk_diagnostics,
        ground_truth_path=ground_truth_path,
    )

    writer.write_diagnostics(classification_result.chunk_diagnostics, layout)
    writer.write_metrics(metrics, layout)
    writer.write_audit_result(classification_result.chunk_diagnostics, layout)
    summary_row = _build_summary_row(run_config, metrics, len(files), len(classification_result.outputs))
    writer.write_final_report(summary_row, metrics, layout)
    csv_path, jsonl_path = writer.append_results(summary_row, _result_fields())

    logger.info(
        "实验完成: run_id=%s, classified=%s, metrics=%s, results_csv=%s"
        % (run_config.run_id, len(classification_result.outputs), layout.metrics_dir / "metrics.json", csv_path)
    )
    print(f"[OK] run_id={run_config.run_id}")
    print(f"[OK] results_csv={csv_path}")
    print(f"[OK] results_jsonl={jsonl_path}")
    print(f"[OK] run_dir={layout.run_dir}")


def _build_llm_settings(args: argparse.Namespace, llm_config_path: Path) -> LLMSettings:
    enabled_override = False if args.mode == "keyword_only" else None
    concurrency_override = False if args.disable_llm_concurrency else None
    settings = load_llm_settings(
        config_path=llm_config_path,
        enabled_override=enabled_override,
        concurrent_enabled_override=concurrency_override,
    )

    if args.model:
        settings = replace(settings, model=args.model)
    if args.temperature is not None:
        settings = replace(settings, temperature=float(args.temperature))
    if args.max_concurrency is not None:
        if int(args.max_concurrency) <= 0:
            raise ValueError("max_concurrency 必须大于 0。")
        settings = replace(settings, max_concurrency=int(args.max_concurrency))
    if args.disable_cache:
        settings = replace(settings, cache_enabled=False)
    return settings


def _build_experiment_config(
    *,
    args: argparse.Namespace,
    input_dir: Path,
    risk_info_path: Path,
    output_root: Path,
    chunk_size: int,
    llm_settings: LLMSettings,
    ground_truth_path: Path | None,
) -> ExperimentConfig:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fingerprint_payload = {
        "run_name": args.run_name,
        "input_dir": str(input_dir),
        "risk_info_path": str(risk_info_path),
        "chunk_size": chunk_size,
        "mode": args.mode,
        "model": llm_settings.model,
        "temperature": llm_settings.temperature,
        "max_concurrency": llm_settings.max_concurrency,
        "concurrent_enabled": llm_settings.concurrent_enabled,
        "cache_enabled": llm_settings.cache_enabled,
        "cache_path": str(llm_settings.cache_path),
        "ground_truth_path": str(ground_truth_path) if ground_truth_path else None,
    }
    fingerprint = hashlib.sha1(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    run_id = f"{_safe_name(args.run_name)}_{timestamp}_{fingerprint}"
    return ExperimentConfig(
        run_name=args.run_name,
        run_id=run_id,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        input_dir=str(input_dir),
        risk_info_path=str(risk_info_path),
        output_root=str(output_root),
        chunk_size=chunk_size,
        mode=args.mode,
        model=llm_settings.model,
        temperature=llm_settings.temperature,
        concurrent_enabled=llm_settings.concurrent_enabled,
        max_concurrency=llm_settings.max_concurrency,
        cache_enabled=llm_settings.cache_enabled,
        cache_path=str(llm_settings.cache_path),
        ground_truth_path=str(ground_truth_path) if ground_truth_path else None,
    )


def _build_summary_row(
    config: ExperimentConfig,
    metrics: ExperimentMetrics,
    file_count: int,
    classified_count: int,
) -> dict[str, Any]:
    row = {
        "run_id": config.run_id,
        "timestamp": config.timestamp,
        "run_name": config.run_name,
        "mode": config.mode,
        "model": config.model,
        "temperature": config.temperature,
        "chunk_size": config.chunk_size,
        "concurrent_enabled": config.concurrent_enabled,
        "max_concurrency": config.max_concurrency,
        "cache_enabled": config.cache_enabled,
        "cache_path": config.cache_path,
        "input_dir": config.input_dir,
        "risk_info_path": config.risk_info_path,
        "ground_truth_path": config.ground_truth_path,
        "file_count": file_count,
        "classified_file_count": classified_count,
    }
    row.update(metrics.to_dict())
    return row


def _write_run_config(config: ExperimentConfig, metrics_dir: Path, logger: Logger) -> Path:
    output = metrics_dir / "experiment_config.json"
    output.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"实验配置写入完成: {output}")
    return output


def _result_fields() -> list[str]:
    return [
        "run_id",
        "timestamp",
        "run_name",
        "mode",
        "model",
        "temperature",
        "chunk_size",
        "concurrent_enabled",
        "max_concurrency",
        "cache_enabled",
        "cache_path",
        "input_dir",
        "risk_info_path",
        "ground_truth_path",
        "file_count",
        "classified_file_count",
        "precision",
        "recall",
        "f1",
        "schema_valid_rate",
        "avg_token",
        "avg_latency_ms",
        "conflict_rate",
        "cache_hit_rate",
        "llm_error_rate",
        "chunk_count",
        "llm_called_count",
        "label_support",
    ]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "experiment"


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
