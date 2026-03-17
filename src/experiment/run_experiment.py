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
from src.classification.classification_pipeline import ClassificationStrategy, run_classification_with_diagnostics
from src.llm import DEFAULT_LLM_CONFIG_PATH, LLMSettings, load_llm_settings
from src.reflection.reflection_config import load_reflection_config
from src.reflection.reflection_pipeline import run_reflection_with_diagnostics
from src.review.review_config import load_review_config
from src.review.review_pipeline import MAX_SCHEMA_RETRY_LIMIT, run_review_with_diagnostics
from src.tools.logger import Logger, get_logger

from .artifact_writer import ArtifactWriter
from .metrics import ExperimentMetrics, compute_experiment_metrics
from .review_eval import (
    DEFAULT_CONFIG_PATH as REVIEW_EVAL_CONFIG_PATH,
    ReviewEvalConfig,
    load_config as load_review_eval_config,
    run_review_eval_with_config,
)


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
    review_rules_path: str
    review_rule_version: str
    review_ablation_no_rules: bool
    review_ablation_coarse_rules: bool
    review_schema_retry_limit: int
    review_ground_truth: str
    reflection_version: str
    reflection_stage1_threshold: int
    reflection_stage2_max_items: int
    reflection_evidence_max_chars: int
    reflection_chunk_excerpt_max_chars: int
    review_eval_gold_dir: str
    review_eval_ablation_pred_only_universe: bool
    review_eval_ablation_standard_only_universe: bool
    review_eval_summary_filename: str
    review_eval_warning_filename: str
    review_eval_detail_jsonl_filename: str
    review_eval_detail_csv_filename: str
    review_eval_enable_detail_jsonl: bool
    review_eval_enable_detail_csv: bool
    review_eval_max_warning_samples: int
    review_eval_gold_items_field: str
    ground_truth_path: str | None

    def to_dict(self) -> dict[str, Any]:
        """导出配置字典。"""

        return asdict(self)


def parse_args() -> argparse.Namespace:
    """解析实验入口参数。"""

    parser = argparse.ArgumentParser(
        description="统一实验入口：chunking + classification + review + reflection + review_eval + metrics + artifacts"
    )
    parser.add_argument("--run-name", type=str, default="experiment", help="实验名称前缀")
    parser.add_argument("--input-dir", type=Path, default=None, help="原始 Word 输入目录")
    parser.add_argument("--risk-info", type=Path, default=None, help="风险类型 CSV 路径")
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments"), help="实验输出根目录")
    parser.add_argument("--ground-truth", type=Path, default=None, help="标注集路径（目录/JSON/JSONL）")
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG_PATH, help="LLM 配置文件路径")
    parser.add_argument("--chunk-size", type=int, default=1000, help="分块字符上限")
    parser.add_argument(
        "--mode",
        choices=["keyword_only", "llm_only", "keyword_llm", "keyword_llm_experiment"],
        default="keyword_llm",
        help="实验模式",
    )
    parser.add_argument("--model", type=str, default=None, help="覆盖模型名称")
    parser.add_argument("--temperature", type=float, default=None, help="覆盖温度参数")
    parser.add_argument("--max-concurrency", type=int, default=None, help="覆盖并发数")
    parser.add_argument("--disable-llm-concurrency", action="store_true", help="关闭 LLM 并发")
    parser.add_argument("--disable-cache", action="store_true", help="关闭 LLM 缓存")
    parser.add_argument("--review-rules", type=Path, default=None, help="review 规则文件路径（csv/json）")
    parser.add_argument("--review-rule-version", type=str, default=None, help="review 规则版本号")
    parser.add_argument("--review-ground-truth", type=str, default="待审核", help="review item ground_truth 初值")
    parser.add_argument("--review-ablation-no-rules", action="store_true", help="review 消融：不注入规则列表")
    parser.add_argument("--review-ablation-coarse-rules", action="store_true", help="review 消融：粗粒度规则")
    parser.add_argument("--review-schema-retry-limit", type=int, default=None, help="review schema 重试上限（0-3）")
    parser.add_argument("--reflection-version", type=str, default=None, help="reflection 版本号")
    parser.add_argument("--reflection-stage1-threshold", type=int, default=None, help="reflection stage1 阈值")
    parser.add_argument("--reflection-stage2-max-items", type=int, default=None, help="reflection stage2 每组最大样本")
    parser.add_argument("--reflection-evidence-max-chars", type=int, default=None, help="reflection 证据窗口最大字符")
    parser.add_argument(
        "--reflection-chunk-excerpt-max-chars",
        type=int,
        default=None,
        help="reflection chunk 回退窗口最大字符",
    )
    parser.add_argument("--review-eval-gold-dir", type=Path, default=None, help="review_eval 标准集目录")
    parser.add_argument(
        "--review-eval-ablation-pred-only-universe",
        action="store_true",
        help="review_eval 消融：仅使用预测集 key 宇宙",
    )
    parser.add_argument(
        "--review-eval-ablation-standard-only-universe",
        action="store_true",
        help="review_eval 消融：仅使用标准集 key 宇宙",
    )
    return parser.parse_args()


def main() -> None:
    """执行统一实验流程。"""

    args = parse_args()
    logger = get_logger("experiment")
    logger.info(f"日志文件: {logger.path}")

    chunking_config = load_chunking_config()
    classification_config = load_classification_config()
    review_config = load_review_config()
    reflection_config = load_reflection_config()
    review_eval_raw = load_review_eval_config(REVIEW_EVAL_CONFIG_PATH)

    input_dir = _resolve_path(args.input_dir or chunking_config["input"])
    risk_info_path = _resolve_path(args.risk_info or classification_config["risk_info"])
    review_rules_path = _resolve_path(args.review_rules or review_config["rules"])
    review_rule_version = str(args.review_rule_version or review_config["rule_version"]).strip() or "v1"
    review_ground_truth = (args.review_ground_truth or "待审核").strip() or "待审核"
    review_schema_retry_limit = int(
        args.review_schema_retry_limit
        if args.review_schema_retry_limit is not None
        else review_config["schema_retry_limit"]
    )
    reflection_version = str(args.reflection_version or reflection_config["reflection_version"]).strip() or "v1"
    reflection_stage1_threshold = int(
        args.reflection_stage1_threshold
        if args.reflection_stage1_threshold is not None
        else reflection_config["stage1_threshold"]
    )
    reflection_stage2_max_items = int(
        args.reflection_stage2_max_items
        if args.reflection_stage2_max_items is not None
        else reflection_config["stage2_max_items"]
    )
    reflection_evidence_max_chars = int(
        args.reflection_evidence_max_chars
        if args.reflection_evidence_max_chars is not None
        else reflection_config["evidence_max_chars"]
    )
    reflection_chunk_excerpt_max_chars = int(
        args.reflection_chunk_excerpt_max_chars
        if args.reflection_chunk_excerpt_max_chars is not None
        else reflection_config["chunk_excerpt_max_chars"]
    )
    review_eval_gold_dir = _resolve_path(
        args.review_eval_gold_dir or Path(str(review_eval_raw.get("gold_dir", "dataset/standard-review")))
    )
    review_eval_ablation_pred_only_universe = (
        args.review_eval_ablation_pred_only_universe
        or _as_bool(review_eval_raw.get("ablation_pred_only_universe"), False)
    )
    review_eval_ablation_standard_only_universe = (
        args.review_eval_ablation_standard_only_universe
        or _as_bool(review_eval_raw.get("ablation_standard_only_universe"), False)
    )
    review_eval_summary_filename = str(review_eval_raw.get("summary_filename", "review_eval_summary.json")).strip()
    if not review_eval_summary_filename:
        review_eval_summary_filename = "review_eval_summary.json"
    review_eval_warning_filename = str(review_eval_raw.get("warning_filename", "review_eval_warnings.json")).strip()
    if not review_eval_warning_filename:
        review_eval_warning_filename = "review_eval_warnings.json"
    review_eval_detail_jsonl_filename = str(
        review_eval_raw.get("detail_jsonl_filename", "review_eval_details.jsonl")
    ).strip()
    if not review_eval_detail_jsonl_filename:
        review_eval_detail_jsonl_filename = "review_eval_details.jsonl"
    review_eval_detail_csv_filename = str(review_eval_raw.get("detail_csv_filename", "review_eval_details.csv")).strip()
    if not review_eval_detail_csv_filename:
        review_eval_detail_csv_filename = "review_eval_details.csv"
    review_eval_enable_detail_jsonl = _as_bool(review_eval_raw.get("enable_detail_jsonl"), True)
    review_eval_enable_detail_csv = _as_bool(review_eval_raw.get("enable_detail_csv"), True)
    review_eval_max_warning_samples = max(1, _as_int(review_eval_raw.get("max_warning_samples"), 30))
    review_eval_gold_items_field = _normalize_review_eval_items_field(
        str(review_eval_raw.get("gold_items_field", "auto"))
    )

    output_root = _resolve_path(args.output_root)
    ground_truth_path = _resolve_path(args.ground_truth) if args.ground_truth else None
    llm_config_path = _resolve_path(args.llm_config)
    chunk_size = clamp_chunk_size(args.chunk_size, chunking_config["min"], chunking_config["max"])
    _validate_review_options(
        ablation_no_rules=args.review_ablation_no_rules,
        ablation_coarse_rules=args.review_ablation_coarse_rules,
        schema_retry_limit=review_schema_retry_limit,
    )
    _validate_reflection_options(
        stage1_threshold=reflection_stage1_threshold,
        stage2_max_items=reflection_stage2_max_items,
        evidence_max_chars=reflection_evidence_max_chars,
        chunk_excerpt_max_chars=reflection_chunk_excerpt_max_chars,
    )
    _validate_review_eval_options(
        gold_dir=review_eval_gold_dir,
        ablation_pred_only_universe=review_eval_ablation_pred_only_universe,
        ablation_standard_only_universe=review_eval_ablation_standard_only_universe,
    )

    llm_settings = _build_llm_settings(args, llm_config_path)
    run_config = _build_experiment_config(
        args=args,
        input_dir=input_dir,
        risk_info_path=risk_info_path,
        review_rules_path=review_rules_path,
        review_rule_version=review_rule_version,
        review_ablation_no_rules=args.review_ablation_no_rules,
        review_ablation_coarse_rules=args.review_ablation_coarse_rules,
        review_schema_retry_limit=review_schema_retry_limit,
        review_ground_truth=review_ground_truth,
        reflection_version=reflection_version,
        reflection_stage1_threshold=reflection_stage1_threshold,
        reflection_stage2_max_items=reflection_stage2_max_items,
        reflection_evidence_max_chars=reflection_evidence_max_chars,
        reflection_chunk_excerpt_max_chars=reflection_chunk_excerpt_max_chars,
        review_eval_gold_dir=review_eval_gold_dir,
        review_eval_ablation_pred_only_universe=review_eval_ablation_pred_only_universe,
        review_eval_ablation_standard_only_universe=review_eval_ablation_standard_only_universe,
        review_eval_summary_filename=review_eval_summary_filename,
        review_eval_warning_filename=review_eval_warning_filename,
        review_eval_detail_jsonl_filename=review_eval_detail_jsonl_filename,
        review_eval_detail_csv_filename=review_eval_detail_csv_filename,
        review_eval_enable_detail_jsonl=review_eval_enable_detail_jsonl,
        review_eval_enable_detail_csv=review_eval_enable_detail_csv,
        review_eval_max_warning_samples=review_eval_max_warning_samples,
        review_eval_gold_items_field=review_eval_gold_items_field,
        output_root=output_root,
        chunk_size=chunk_size,
        llm_settings=llm_settings,
        ground_truth_path=ground_truth_path,
    )

    logger.info(f"实验配置: {json.dumps(run_config.to_dict(), ensure_ascii=False)}")

    # 阶段 1：初始化产物目录并固化本次实验配置。
    writer = ArtifactWriter(output_root=output_root, logger=logger)
    layout = writer.prepare_layout(run_config.run_id)
    _write_run_config(run_config, layout.metrics_dir, logger)

    files = discover_word_files(input_dir, logger=logger)
    if not files:
        logger.error(f"未找到 Word 文件: {input_dir}")
        raise FileNotFoundError(f"未找到 Word 文件: {input_dir}")
    logger.info(f"实验输入文件数: {len(files)}")

    # 阶段 2：统一执行 chunking，输出本 run 独立的 chunks 产物。
    for path in files:
        process_word_file(path, chunk_size, layout.chunks_dir, logger=logger)

    classification_strategy, emit_all_variants = _resolve_classification_plan(args.mode)

    # 阶段 3：基于 chunks 执行分类，并回收 chunk 级诊断信息。
    classification_result = run_classification_with_diagnostics(
        input_path=layout.chunks_dir,
        output_dir=layout.classified_dir,
        risk_info_path=risk_info_path,
        llm_settings=llm_settings,
        logger=logger,
        strategy=classification_strategy,
        emit_all_variants=emit_all_variants,
    )
    mode_to_outputs = (
        classification_result.outputs_by_mode
        if emit_all_variants
        else {classification_strategy: classification_result.outputs}
    )
    mode_to_diagnostics = (
        classification_result.diagnostics_by_mode
        if emit_all_variants
        else {classification_strategy: classification_result.chunk_diagnostics}
    )

    # 阶段 4：把 classified 结果作为输入，执行 review。
    review_outputs_by_mode: dict[str, list[Path]] = {}
    for mode in mode_to_outputs:
        review_input_dir = layout.classified_dir / mode if emit_all_variants else layout.classified_dir
        review_output_dir = layout.review_dir / mode if emit_all_variants else layout.review_dir
        review_result = run_review_with_diagnostics(
            input_path=review_input_dir,
            output_dir=review_output_dir,
            rules_path=review_rules_path,
            rule_version=review_rule_version,
            llm_settings=llm_settings,
            logger=logger,
            ablation_no_rules=args.review_ablation_no_rules,
            ablation_coarse_rules=args.review_ablation_coarse_rules,
            schema_retry_limit=review_schema_retry_limit,
            ground_truth=review_ground_truth,
        )
        review_outputs_by_mode[mode] = review_result.outputs
        logger.info(
            "review_mode_done: mode=%s, reviewed_files=%s, task_count=%s, review_item_count=%s, output=%s"
            % (
                mode,
                len(review_result.outputs),
                review_result.metrics.task_count,
                review_result.metrics.emitted_item_count,
                review_output_dir,
            )
        )

    # 阶段 5：review 输出进入 reflection 流程。
    reflection_outputs_by_mode: dict[str, list[Path]] = {}
    for mode in mode_to_outputs:
        reflection_input_dir = layout.review_dir / mode if emit_all_variants else layout.review_dir
        reflection_output_dir = layout.reflection_dir / mode if emit_all_variants else layout.reflection_dir
        reflection_result = run_reflection_with_diagnostics(
            input_path=reflection_input_dir,
            output_dir=reflection_output_dir,
            llm_settings=llm_settings,
            logger=logger,
            reflection_version=reflection_version,
            stage1_threshold=reflection_stage1_threshold,
            stage2_max_items=reflection_stage2_max_items,
            evidence_max_chars=reflection_evidence_max_chars,
            chunk_excerpt_max_chars=reflection_chunk_excerpt_max_chars,
        )
        reflection_outputs_by_mode[mode] = reflection_result.outputs
        logger.info(
            "reflection_mode_done: mode=%s, reflected_files=%s, stage1_adjusted_count=%s, stage2_adjusted_count=%s, output=%s"
            % (
                mode,
                len(reflection_result.outputs),
                reflection_result.metrics.stage1_adjusted_count,
                reflection_result.metrics.stage2_adjusted_count,
                reflection_output_dir,
            )
        )

    # 阶段 6：对 review 输出执行 review_eval，产物写入 melting-reflection。
    for mode in mode_to_outputs:
        review_pred_dir = layout.review_dir / mode if emit_all_variants else layout.review_dir
        review_eval_config = ReviewEvalConfig(
            gold_dir=review_eval_gold_dir,
            pred_dir=review_pred_dir,
            output_dir=layout.melting_reflection_dir,
            summary_filename=_build_review_eval_filename(
                review_eval_summary_filename,
                mode=mode,
                emit_all_variants=emit_all_variants,
            ),
            warning_filename=_build_review_eval_filename(
                review_eval_warning_filename,
                mode=mode,
                emit_all_variants=emit_all_variants,
            ),
            detail_jsonl_filename=_build_review_eval_filename(
                review_eval_detail_jsonl_filename,
                mode=mode,
                emit_all_variants=emit_all_variants,
            ),
            detail_csv_filename=_build_review_eval_filename(
                review_eval_detail_csv_filename,
                mode=mode,
                emit_all_variants=emit_all_variants,
            ),
            enable_detail_jsonl=review_eval_enable_detail_jsonl,
            enable_detail_csv=review_eval_enable_detail_csv,
            ablation_pred_only_universe=review_eval_ablation_pred_only_universe,
            ablation_standard_only_universe=review_eval_ablation_standard_only_universe,
            max_warning_samples=review_eval_max_warning_samples,
            gold_items_field=review_eval_gold_items_field,
            pred_items_field="review_items",
        )
        _write_review_eval_config(
            review_eval_config,
            layout.melting_reflection_dir
            / _build_review_eval_filename(
                "review_eval_config.json",
                mode=mode,
                emit_all_variants=emit_all_variants,
            ),
            logger,
        )
        review_eval_result = run_review_eval_with_config(review_eval_config, logger)
        logger.info(
            "review_eval_for_review_done: mode=%s, summary=%s, warnings=%s"
            % (
                mode,
                review_eval_result.summary_path,
                review_eval_result.warning_path,
            )
        )

    # 阶段 7：对 reflection 输出执行 review_eval，产物写入 final_report。
    for mode in mode_to_outputs:
        reflection_pred_dir = layout.reflection_dir / mode if emit_all_variants else layout.reflection_dir
        reflection_eval_config = ReviewEvalConfig(
            gold_dir=review_eval_gold_dir,
            pred_dir=reflection_pred_dir,
            output_dir=layout.final_report_dir,
            summary_filename=_build_review_eval_filename(
                review_eval_summary_filename,
                mode=mode,
                emit_all_variants=emit_all_variants,
                prefix="reflection",
            ),
            warning_filename=_build_review_eval_filename(
                review_eval_warning_filename,
                mode=mode,
                emit_all_variants=emit_all_variants,
                prefix="reflection",
            ),
            detail_jsonl_filename=_build_review_eval_filename(
                review_eval_detail_jsonl_filename,
                mode=mode,
                emit_all_variants=emit_all_variants,
                prefix="reflection",
            ),
            detail_csv_filename=_build_review_eval_filename(
                review_eval_detail_csv_filename,
                mode=mode,
                emit_all_variants=emit_all_variants,
                prefix="reflection",
            ),
            enable_detail_jsonl=review_eval_enable_detail_jsonl,
            enable_detail_csv=review_eval_enable_detail_csv,
            ablation_pred_only_universe=review_eval_ablation_pred_only_universe,
            ablation_standard_only_universe=review_eval_ablation_standard_only_universe,
            max_warning_samples=review_eval_max_warning_samples,
            gold_items_field=review_eval_gold_items_field,
            pred_items_field="reflection_items",
        )
        _write_review_eval_config(
            reflection_eval_config,
            layout.final_report_dir
            / _build_review_eval_filename(
                "review_eval_config.json",
                mode=mode,
                emit_all_variants=emit_all_variants,
                prefix="reflection",
            ),
            logger,
        )
        reflection_eval_result = run_review_eval_with_config(reflection_eval_config, logger)
        logger.info(
            "review_eval_for_reflection_done: mode=%s, summary=%s, warnings=%s"
            % (
                mode,
                reflection_eval_result.summary_path,
                reflection_eval_result.warning_path,
            )
        )

    # 阶段 8：按统一口径聚合分类指标（质量+性能+稳定性）。
    metrics_by_mode: dict[str, ExperimentMetrics] = {}
    for mode, classified_files in mode_to_outputs.items():
        metrics_by_mode[mode] = compute_experiment_metrics(
            classified_files=classified_files,
            diagnostics=mode_to_diagnostics[mode],
            ground_truth_path=ground_truth_path,
        )

    # 阶段 9：统一落盘所有产物，并追加 results 汇总行。
    writer.write_diagnostics(classification_result.chunk_diagnostics, layout)
    writer.write_audit_result(classification_result.chunk_diagnostics, layout)
    csv_path: Path | None = None
    jsonl_path: Path | None = None
    for mode, metrics in metrics_by_mode.items():
        artifact_suffix = mode if emit_all_variants else ""
        summary_row = _build_summary_row(
            run_config,
            metrics,
            len(files),
            len(mode_to_outputs[mode]),
            len(review_outputs_by_mode.get(mode, [])),
            len(reflection_outputs_by_mode.get(mode, [])),
            result_mode=mode,
            result_run_id=f"{run_config.run_id}__{mode}" if emit_all_variants else run_config.run_id,
        )
        writer.write_metrics(metrics, layout, artifact_suffix=artifact_suffix)
        writer.write_final_report(summary_row, metrics, layout, artifact_suffix=artifact_suffix)
        csv_path, jsonl_path = writer.append_results(summary_row, _result_fields())

    logger.info(
        "实验完成: run_id=%s, modes=%s, results_csv=%s"
        % (run_config.run_id, list(mode_to_outputs.keys()), csv_path)
    )
    if csv_path is None or jsonl_path is None:
        raise RuntimeError("实验未产出 results 记录，请检查分类输出。")
    print(f"[OK] run_id={run_config.run_id}")
    print(f"[OK] results_csv={csv_path}")
    print(f"[OK] results_jsonl={jsonl_path}")
    print(f"[OK] run_dir={layout.run_dir}")
    print(f"[OK] review_dir={layout.review_dir}")
    print(f"[OK] reflection_dir={layout.reflection_dir}")
    print(f"[OK] melting_reflection_dir={layout.melting_reflection_dir}")


def _build_llm_settings(args: argparse.Namespace, llm_config_path: Path) -> LLMSettings:
    concurrency_override = False if args.disable_llm_concurrency else None
    settings = load_llm_settings(
        config_path=llm_config_path,
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
    review_rules_path: Path,
    review_rule_version: str,
    review_ablation_no_rules: bool,
    review_ablation_coarse_rules: bool,
    review_schema_retry_limit: int,
    review_ground_truth: str,
    reflection_version: str,
    reflection_stage1_threshold: int,
    reflection_stage2_max_items: int,
    reflection_evidence_max_chars: int,
    reflection_chunk_excerpt_max_chars: int,
    review_eval_gold_dir: Path,
    review_eval_ablation_pred_only_universe: bool,
    review_eval_ablation_standard_only_universe: bool,
    review_eval_summary_filename: str,
    review_eval_warning_filename: str,
    review_eval_detail_jsonl_filename: str,
    review_eval_detail_csv_filename: str,
    review_eval_enable_detail_jsonl: bool,
    review_eval_enable_detail_csv: bool,
    review_eval_max_warning_samples: int,
    review_eval_gold_items_field: str,
    output_root: Path,
    chunk_size: int,
    llm_settings: LLMSettings,
    ground_truth_path: Path | None,
) -> ExperimentConfig:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 指纹仅由关键实验参数构成，确保同参数组可稳定追溯。
    fingerprint_payload = {
        "run_name": args.run_name,
        "input_dir": str(input_dir),
        "risk_info_path": str(risk_info_path),
        "review_rules_path": str(review_rules_path),
        "review_rule_version": review_rule_version,
        "review_ablation_no_rules": review_ablation_no_rules,
        "review_ablation_coarse_rules": review_ablation_coarse_rules,
        "review_schema_retry_limit": review_schema_retry_limit,
        "review_ground_truth": review_ground_truth,
        "reflection_version": reflection_version,
        "reflection_stage1_threshold": reflection_stage1_threshold,
        "reflection_stage2_max_items": reflection_stage2_max_items,
        "reflection_evidence_max_chars": reflection_evidence_max_chars,
        "reflection_chunk_excerpt_max_chars": reflection_chunk_excerpt_max_chars,
        "review_eval_gold_dir": str(review_eval_gold_dir),
        "review_eval_ablation_pred_only_universe": review_eval_ablation_pred_only_universe,
        "review_eval_ablation_standard_only_universe": review_eval_ablation_standard_only_universe,
        "review_eval_summary_filename": review_eval_summary_filename,
        "review_eval_warning_filename": review_eval_warning_filename,
        "review_eval_detail_jsonl_filename": review_eval_detail_jsonl_filename,
        "review_eval_detail_csv_filename": review_eval_detail_csv_filename,
        "review_eval_enable_detail_jsonl": review_eval_enable_detail_jsonl,
        "review_eval_enable_detail_csv": review_eval_enable_detail_csv,
        "review_eval_max_warning_samples": review_eval_max_warning_samples,
        "review_eval_gold_items_field": review_eval_gold_items_field,
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
        review_rules_path=str(review_rules_path),
        review_rule_version=review_rule_version,
        review_ablation_no_rules=review_ablation_no_rules,
        review_ablation_coarse_rules=review_ablation_coarse_rules,
        review_schema_retry_limit=review_schema_retry_limit,
        review_ground_truth=review_ground_truth,
        reflection_version=reflection_version,
        reflection_stage1_threshold=reflection_stage1_threshold,
        reflection_stage2_max_items=reflection_stage2_max_items,
        reflection_evidence_max_chars=reflection_evidence_max_chars,
        reflection_chunk_excerpt_max_chars=reflection_chunk_excerpt_max_chars,
        review_eval_gold_dir=str(review_eval_gold_dir),
        review_eval_ablation_pred_only_universe=review_eval_ablation_pred_only_universe,
        review_eval_ablation_standard_only_universe=review_eval_ablation_standard_only_universe,
        review_eval_summary_filename=review_eval_summary_filename,
        review_eval_warning_filename=review_eval_warning_filename,
        review_eval_detail_jsonl_filename=review_eval_detail_jsonl_filename,
        review_eval_detail_csv_filename=review_eval_detail_csv_filename,
        review_eval_enable_detail_jsonl=review_eval_enable_detail_jsonl,
        review_eval_enable_detail_csv=review_eval_enable_detail_csv,
        review_eval_max_warning_samples=review_eval_max_warning_samples,
        review_eval_gold_items_field=review_eval_gold_items_field,
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
    review_count: int,
    reflection_count: int,
    *,
    result_mode: str,
    result_run_id: str,
) -> dict[str, Any]:
    row = {
        "run_id": result_run_id,
        "timestamp": config.timestamp,
        "run_name": config.run_name,
        "mode": result_mode,
        "experiment_mode": config.mode,
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
        "review_file_count": review_count,
        "reflection_file_count": reflection_count,
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
        "experiment_mode",
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
        "review_file_count",
        "reflection_file_count",
        "precision",
        "recall",
        "f1",
        "schema_valid_rate",
        "avg_token_in",
        "avg_token_out",
        "avg_total_token",
        "avg_token",
        "avg_reasoning_token",
        "avg_cached_token",
        "avg_latency_ms",
        "reasoning_token_ratio",
        "cached_token_ratio",
        "conflict_rate",
        "cache_hit_rate",
        "llm_error_rate",
        "total_tokens_estimated_rate",
        "chunk_count",
        "llm_called_count",
        "label_support",
    ]


def _resolve_classification_plan(mode: str) -> tuple[ClassificationStrategy, bool]:
    if mode == "keyword_llm_experiment":
        return "keyword_llm", True
    if mode == "keyword_only":
        return "keyword_only", False
    if mode == "llm_only":
        return "llm_only", False
    return "keyword_llm", False


def _validate_review_options(
    *,
    ablation_no_rules: bool,
    ablation_coarse_rules: bool,
    schema_retry_limit: int,
) -> None:
    if ablation_no_rules and ablation_coarse_rules:
        raise ValueError("--review-ablation-no-rules 与 --review-ablation-coarse-rules 不能同时启用。")
    if schema_retry_limit < 0 or schema_retry_limit > MAX_SCHEMA_RETRY_LIMIT:
        raise ValueError(
            f"review_schema_retry_limit 必须在 [0, {MAX_SCHEMA_RETRY_LIMIT}] 范围内。"
        )


def _validate_reflection_options(
    *,
    stage1_threshold: int,
    stage2_max_items: int,
    evidence_max_chars: int,
    chunk_excerpt_max_chars: int,
) -> None:
    if stage1_threshold < 0:
        raise ValueError("reflection_stage1_threshold 不能小于 0。")
    if stage2_max_items <= 0:
        raise ValueError("reflection_stage2_max_items 必须大于 0。")
    if evidence_max_chars <= 0:
        raise ValueError("reflection_evidence_max_chars 必须大于 0。")
    if chunk_excerpt_max_chars <= 0:
        raise ValueError("reflection_chunk_excerpt_max_chars 必须大于 0。")


def _validate_review_eval_options(
    *,
    gold_dir: Path,
    ablation_pred_only_universe: bool,
    ablation_standard_only_universe: bool,
) -> None:
    if ablation_pred_only_universe and ablation_standard_only_universe:
        raise ValueError(
            "--review-eval-ablation-pred-only-universe 与 --review-eval-ablation-standard-only-universe 不能同时启用。"
        )
    if not gold_dir.exists():
        raise FileNotFoundError(f"review_eval gold_dir 不存在: {gold_dir}")
    if gold_dir.is_file() and gold_dir.suffix.lower() != ".json":
        raise ValueError(f"review_eval gold_dir 为文件时必须是 .json: {gold_dir}")


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _normalize_review_eval_items_field(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"auto", "review_items", "reflection_items"}:
        raise ValueError(f"review_eval items_field 不支持: {value}")
    return normalized


def _build_review_eval_filename(
    base_name: str,
    *,
    mode: str,
    emit_all_variants: bool,
    prefix: str | None = None,
) -> str:
    name = base_name
    if prefix:
        name = _prepend_filename_prefix(name, prefix)
    if emit_all_variants:
        name = _append_filename_suffix(name, mode)
    return name


def _prepend_filename_prefix(filename: str, prefix: str) -> str:
    path = Path(filename)
    return f"{prefix}_{path.stem}{path.suffix}"


def _append_filename_suffix(filename: str, suffix: str) -> str:
    path = Path(filename)
    return f"{path.stem}_{_safe_name(suffix)}{path.suffix}"


def _write_review_eval_config(config: ReviewEvalConfig, output_path: Path, logger: Logger) -> Path:
    output_path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"review_eval 配置写入完成: {output_path}")
    return output_path


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "experiment"


def _resolve_path(path: Path | str) -> Path:
    resolved = path if isinstance(path, Path) else Path(path)
    return resolved if resolved.is_absolute() else Path.cwd() / resolved


if __name__ == "__main__":
    main()
