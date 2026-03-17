"""Chunking 分块策略消融实验入口。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import cast

from src.classification.classification_pipeline import ClassificationStrategy, run_classification_with_diagnostics
from src.classification.classification_config import load_classification_config
from src.experiment.review_eval import ReviewEvalConfig, run_review_eval_with_config
from src.llm import DEFAULT_LLM_CONFIG_PATH, LLMSettings, load_llm_settings
from src.reflection.reflection_config import load_reflection_config
from src.reflection.reflection_pipeline import run_reflection_with_diagnostics
from src.review.review_config import load_review_config
from src.review.review_pipeline import run_review_with_diagnostics
from src.tools.logger import Logger, get_logger

from .chunking_config import PROJECT_ROOT, clamp_chunk_size, load_chunking_config
from .word_chunking_pipeline import process_word_file, process_word_file_hard_length
from .word_text_extractor import discover_word_files


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "chunking_ablation_config.json"
ABLATION_MODE_CHOICES = {"both", "semantic_only", "hard_length_only"}
ITEM_FIELD_CHOICES = {"auto", "review_items", "reflection_items"}


@dataclass(frozen=True)
class ChunkingAblationConfig:
    """Chunking 消融实验配置。

    Args:
        run_name: 实验名称。
        output_root: 实验输出根目录。
        input_dir: Word 输入目录。
        chunk_size: 生效分块大小。
        ablation_mode: 消融模式（both/semantic_only/hard_length_only）。
        classification_mode: 分类模式（keyword_only/llm_only/keyword_llm）。
        risk_info_path: 分类风险字典路径。
        review_rules_path: review 规则文件路径。
        review_rule_version: review 规则版本。
        review_ground_truth: review item 初始标注值。
        review_schema_retry_limit: review schema 重试上限。
        enable_reflection: 是否启用 reflection 阶段。
        reflection_version: reflection 版本号。
        reflection_stage1_threshold: reflection stage1 阈值。
        reflection_stage2_max_items: reflection stage2 每组最大样本数。
        reflection_evidence_max_chars: reflection 证据窗口最大字符。
        reflection_chunk_excerpt_max_chars: reflection chunk 回退窗口最大字符。
        review_eval_gold_dir: semantic 实验臂的 review_eval 标准集路径。
        review_eval_gold_dir_hard_length: hard_length 实验臂的 review_eval 标准集路径。
        review_eval_gold_items_field: review_eval 标准集 items 字段。
        review_eval_ablation_pred_only_universe: review_eval 预测集宇宙消融。
        review_eval_ablation_standard_only_universe: review_eval 标准集宇宙消融。
        review_eval_enable_detail_jsonl: 是否输出 review_eval JSONL 明细。
        review_eval_enable_detail_csv: 是否输出 review_eval CSV 明细。
        review_eval_max_warning_samples: review_eval warning 采样上限。
    """

    run_name: str
    output_root: Path
    input_dir: Path
    chunk_size: int
    ablation_mode: str
    classification_mode: ClassificationStrategy
    risk_info_path: Path
    review_rules_path: Path
    review_rule_version: str
    review_ground_truth: str
    review_schema_retry_limit: int
    enable_reflection: bool
    reflection_version: str
    reflection_stage1_threshold: int
    reflection_stage2_max_items: int
    reflection_evidence_max_chars: int
    reflection_chunk_excerpt_max_chars: int
    review_eval_gold_dir: Path
    review_eval_gold_dir_hard_length: Path
    review_eval_gold_items_field: str
    review_eval_ablation_pred_only_universe: bool
    review_eval_ablation_standard_only_universe: bool
    review_eval_enable_detail_jsonl: bool
    review_eval_enable_detail_csv: bool
    review_eval_max_warning_samples: int

    def to_dict(self) -> dict[str, object]:
        """导出配置快照。"""

        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data


@dataclass(frozen=True)
class ArmMetrics:
    """单实验臂的结构化指标。"""

    arm: str
    chunking_strategy: str
    chunk_size: int
    file_count: int
    chunk_file_count: int
    chunk_count: int
    avg_chunk_chars: float
    min_chunk_chars: int
    max_chunk_chars: int
    classified_file_count: int
    review_file_count: int
    reflection_enabled: bool
    reflection_file_count: int
    review_item_count: int
    reflection_item_count: int
    review_micro_precision: float
    review_micro_recall: float
    review_micro_f1: float
    review_micro_accuracy: float
    reflection_micro_precision: float
    reflection_micro_recall: float
    reflection_micro_f1: float
    reflection_micro_accuracy: float
    chunk_dir: str
    classified_dir: str
    review_dir: str
    reflection_dir: str
    review_eval_gold_dir_used: str
    report_review_summary: str
    report_reflection_summary: str

    def to_dict(self) -> dict[str, object]:
        """导出指标字典。"""

        return asdict(self)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="执行 chunking 分块策略消融实验")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="消融配置路径")
    parser.add_argument("--chunk-size", type=int, default=None, help="分块大小（字符）")
    parser.add_argument("--input-dir", type=Path, default=None, help="Word 输入目录")
    parser.add_argument("--output-root", type=Path, default=None, help="实验输出根目录")
    parser.add_argument("--run-name", type=str, default=None, help="实验名称")
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG_PATH, help="LLM 配置文件路径")
    parser.add_argument(
        "--classification-mode",
        type=str,
        choices=["keyword_only", "llm_only", "keyword_llm"],
        default=None,
        help="分类模式",
    )
    parser.add_argument(
        "--ablation-mode",
        type=str,
        choices=sorted(ABLATION_MODE_CHOICES),
        default=None,
        help="消融模式：both / semantic_only / hard_length_only",
    )
    parser.add_argument("--review-eval-gold-dir", type=Path, default=None, help="semantic 实验臂的 review_eval 标准集目录")
    parser.add_argument(
        "--review-eval-hard-gold-dir",
        type=Path,
        default=None,
        help="hard_length 实验臂的 review_eval 标准集目录",
    )
    parser.add_argument("--enable-reflection", action="store_true", help="启用 reflection 模块（默认关闭）")
    parser.add_argument("--disable-llm-concurrency", action="store_true", help="关闭 LLM 并发")
    parser.add_argument("--disable-cache", action="store_true", help="关闭 LLM 缓存")
    return parser.parse_args()


def main() -> None:
    """执行 chunking 消融实验全流程。"""

    logger = get_logger("chunking_ablation")
    logger.info(f"日志文件: {logger.path}")
    try:
        args = parse_args()
        config, llm_settings = build_runtime_config(args, logger)
        logger.info(f"消融配置: {json.dumps(config.to_dict(), ensure_ascii=False)}")
        run_chunking_ablation_experiment(config=config, llm_settings=llm_settings, logger=logger)
    except Exception as error:
        logger.error(f"chunking 消融实验失败: {error}")
        raise


def build_runtime_config(args: argparse.Namespace, logger: Logger) -> tuple[ChunkingAblationConfig, LLMSettings]:
    """构建运行时配置。

    Args:
        args: 命令行参数。
        logger: 日志对象。

    Returns:
        tuple[ChunkingAblationConfig, LLMSettings]: 运行配置与 LLM 配置。

    Raises:
        FileNotFoundError: 配置文件不存在时抛出。
        ValueError: 参数非法时抛出。
    """

    raw = _load_json(args.config)

    chunking_config = load_chunking_config()
    classification_config = load_classification_config()
    review_config = load_review_config()
    reflection_config = load_reflection_config()

    requested_chunk_size = (
        int(args.chunk_size)
        if args.chunk_size is not None
        else int(_pick(raw, "chunk_size", 1000))
    )
    chunk_size = clamp_chunk_size(requested_chunk_size, chunking_config["min"], chunking_config["max"])

    input_dir = _resolve_path(args.input_dir or _pick(raw, "input_dir", chunking_config["input"]))
    output_root = _resolve_path(
        args.output_root or _pick(raw, "output_root", Path("data/experiments/chunking-ablation"))
    )
    run_name = str(args.run_name or _pick(raw, "run_name", "chunking_ablation")).strip() or "chunking_ablation"

    ablation_mode = str(args.ablation_mode or _pick(raw, "ablation_mode", "both")).strip().lower()
    if ablation_mode not in ABLATION_MODE_CHOICES:
        raise ValueError(f"ablation_mode 不支持: {ablation_mode}")

    classification_mode = str(args.classification_mode or _pick(raw, "classification_mode", "keyword_llm")).strip()
    if classification_mode not in {"keyword_only", "llm_only", "keyword_llm"}:
        raise ValueError(f"classification_mode 不支持: {classification_mode}")

    review_eval_gold_dir = _resolve_path(
        args.review_eval_gold_dir or _pick(raw, "review_eval_gold_dir", Path("dataset/standard-review"))
    )
    review_eval_gold_dir_hard_length = _resolve_path(
        args.review_eval_hard_gold_dir
        or _pick(raw, "review_eval_gold_dir_hard_length", Path("dataset/chunk-review"))
    )
    if not review_eval_gold_dir.exists():
        raise FileNotFoundError(f"semantic review_eval 标准集目录不存在: {review_eval_gold_dir}")
    if not review_eval_gold_dir_hard_length.exists():
        raise FileNotFoundError(
            f"hard_length review_eval 标准集目录不存在: {review_eval_gold_dir_hard_length}"
        )

    llm_config_path = _resolve_path(args.llm_config)
    llm_settings = load_llm_settings(
        config_path=llm_config_path,
        concurrent_enabled_override=False if args.disable_llm_concurrency else None,
    )
    if args.disable_cache:
        llm_settings = replace(llm_settings, cache_enabled=False)

    config = ChunkingAblationConfig(
        run_name=run_name,
        output_root=output_root,
        input_dir=input_dir,
        chunk_size=chunk_size,
        ablation_mode=ablation_mode,
        classification_mode=cast(ClassificationStrategy, classification_mode),
        risk_info_path=_resolve_path(_pick(raw, "risk_info_path", classification_config["risk_info"])),
        review_rules_path=_resolve_path(_pick(raw, "review_rules_path", review_config["rules"])),
        review_rule_version=str(_pick(raw, "review_rule_version", review_config["rule_version"])),
        review_ground_truth=str(_pick(raw, "review_ground_truth", "待审核")),
        review_schema_retry_limit=int(_pick(raw, "review_schema_retry_limit", review_config["schema_retry_limit"])),
        enable_reflection=bool(args.enable_reflection or _as_bool(raw.get("enable_reflection"), False)),
        reflection_version=str(_pick(raw, "reflection_version", reflection_config["reflection_version"])),
        reflection_stage1_threshold=int(_pick(raw, "reflection_stage1_threshold", reflection_config["stage1_threshold"])),
        reflection_stage2_max_items=int(_pick(raw, "reflection_stage2_max_items", reflection_config["stage2_max_items"])),
        reflection_evidence_max_chars=int(_pick(raw, "reflection_evidence_max_chars", reflection_config["evidence_max_chars"])),
        reflection_chunk_excerpt_max_chars=int(
            _pick(raw, "reflection_chunk_excerpt_max_chars", reflection_config["chunk_excerpt_max_chars"])
        ),
        review_eval_gold_dir=review_eval_gold_dir,
        review_eval_gold_dir_hard_length=review_eval_gold_dir_hard_length,
        review_eval_gold_items_field=str(_pick(raw, "review_eval_gold_items_field", "auto")),
        review_eval_ablation_pred_only_universe=_as_bool(raw.get("review_eval_ablation_pred_only_universe"), False),
        review_eval_ablation_standard_only_universe=_as_bool(
            raw.get("review_eval_ablation_standard_only_universe"),
            False,
        ),
        review_eval_enable_detail_jsonl=_as_bool(raw.get("review_eval_enable_detail_jsonl"), True),
        review_eval_enable_detail_csv=_as_bool(raw.get("review_eval_enable_detail_csv"), True),
        review_eval_max_warning_samples=max(1, int(_pick(raw, "review_eval_max_warning_samples", 30))),
    )

    if config.review_eval_ablation_pred_only_universe and config.review_eval_ablation_standard_only_universe:
        raise ValueError("review_eval 两个宇宙消融开关不能同时启用。")
    if config.review_eval_gold_items_field not in ITEM_FIELD_CHOICES:
        raise ValueError(f"review_eval_gold_items_field 不支持: {config.review_eval_gold_items_field}")
    if config.review_schema_retry_limit < 0:
        raise ValueError("review_schema_retry_limit 不能小于 0。")

    logger.info(
        "参数生效: requested_chunk_size=%s, effective_chunk_size=%s, ablation_mode=%s, "
        "classification_mode=%s, enable_reflection=%s, semantic_gold=%s, hard_gold=%s"
        % (
            requested_chunk_size,
            chunk_size,
            config.ablation_mode,
            config.classification_mode,
            config.enable_reflection,
            config.review_eval_gold_dir,
            config.review_eval_gold_dir_hard_length,
        )
    )
    return config, llm_settings


def run_chunking_ablation_experiment(config: ChunkingAblationConfig, llm_settings: LLMSettings, logger: Logger) -> None:
    """运行 chunking 消融实验。

    Args:
        config: 消融运行配置。
        llm_settings: LLM 设置。
        logger: 日志对象。

    Raises:
        FileNotFoundError: 输入文档不存在时抛出。
        RuntimeError: 未成功产出任一实验臂结果时抛出。
    """

    files = discover_word_files(config.input_dir, logger=logger)
    if not files:
        raise FileNotFoundError(f"未找到 Word 文件: {config.input_dir}")

    run_id = _build_run_id(config)
    run_dir = config.output_root / run_id
    metrics_dir = run_dir / "metrics"
    report_dir = run_dir / "report"
    arms_dir = run_dir / "arms"
    for path in (config.output_root, run_dir, metrics_dir, report_dir, arms_dir):
        path.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "ablation_run_config.json").write_text(
        json.dumps({"run_id": run_id, **config.to_dict()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(f"消融实验开始: run_id={run_id}, file_count={len(files)}, run_dir={run_dir}")
    arms = _resolve_arms(config.ablation_mode)
    arm_results: list[ArmMetrics] = []

    for arm in arms:
        logger.info(f"实验臂开始: arm={arm}")
        result = _run_single_arm(
            arm=arm,
            files=files,
            config=config,
            llm_settings=llm_settings,
            run_dir=run_dir,
            logger=logger,
        )
        arm_results.append(result)
        arm_metrics_path = metrics_dir / f"arm_metrics_{arm}.json"
        arm_metrics_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"实验臂完成: arm={arm}, metrics={arm_metrics_path}")

    if not arm_results:
        raise RuntimeError("未产生任何实验臂结果。")

    summary_payload = _build_summary_payload(run_id=run_id, config=config, arm_results=arm_results)
    summary_path = metrics_dir / "chunking_ablation_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = report_dir / "chunking_ablation_report.csv"
    _write_report_csv(csv_path, arm_results, summary_payload.get("delta", {}))

    md_path = report_dir / "chunking_ablation_report.md"
    md_path.write_text(_build_markdown_report(run_id=run_id, config=config, arm_results=arm_results, summary=summary_payload), encoding="utf-8")

    logger.info(f"消融实验完成: run_id={run_id}, summary={summary_path}, report={md_path}")
    print(f"[OK] run_id={run_id}")
    print(f"[OK] run_dir={run_dir}")
    print(f"[OK] summary_json={summary_path}")
    print(f"[OK] report_md={md_path}")
    print(f"[OK] report_csv={csv_path}")


def _run_single_arm(
    *,
    arm: str,
    files: list[Path],
    config: ChunkingAblationConfig,
    llm_settings: LLMSettings,
    run_dir: Path,
    logger: Logger,
) -> ArmMetrics:
    """执行单个实验臂并返回聚合指标。

    Args:
        arm: 实验臂名称（`semantic` 或 `hard_length`）。
        files: 待处理 Word 文件列表。
        config: 消融实验配置。
        llm_settings: LLM 配置。
        run_dir: 当前 run 根目录。
        logger: 日志对象。

    Returns:
        ArmMetrics: 单实验臂的结构化指标记录。
    """

    arm_dir = run_dir / "arms" / arm
    chunk_dir = arm_dir / "chunks"
    classified_dir = arm_dir / "classified"
    review_dir = arm_dir / "review"
    reflection_dir = arm_dir / "reflection"
    review_eval_review_dir = arm_dir / "review_eval" / "review"
    review_eval_reflection_dir = arm_dir / "review_eval" / "reflection"
    for path in (
        arm_dir,
        chunk_dir,
        classified_dir,
        review_dir,
        reflection_dir,
        review_eval_review_dir,
        review_eval_reflection_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    chunk_outputs: list[Path] = []
    for source_file in files:
        if arm == "semantic":
            chunk_outputs.append(process_word_file(source_file, config.chunk_size, chunk_dir, logger=logger))
        else:
            chunk_outputs.append(process_word_file_hard_length(source_file, config.chunk_size, chunk_dir, logger=logger))
    review_eval_gold_dir = _select_review_eval_gold_dir(arm=arm, config=config)

    classification_result = run_classification_with_diagnostics(
        input_path=chunk_dir,
        output_dir=classified_dir,
        risk_info_path=config.risk_info_path,
        llm_settings=llm_settings,
        logger=logger,
        strategy=config.classification_mode,
        emit_all_variants=False,
    )

    review_result = run_review_with_diagnostics(
        input_path=classified_dir,
        output_dir=review_dir,
        rules_path=config.review_rules_path,
        rule_version=config.review_rule_version,
        llm_settings=llm_settings,
        logger=logger,
        ground_truth=config.review_ground_truth,
        schema_retry_limit=config.review_schema_retry_limit,
    )

    reflection_result = (
        run_reflection_with_diagnostics(
            input_path=review_dir,
            output_dir=reflection_dir,
            llm_settings=llm_settings,
            logger=logger,
            reflection_version=config.reflection_version,
            stage1_threshold=config.reflection_stage1_threshold,
            stage2_max_items=config.reflection_stage2_max_items,
            evidence_max_chars=config.reflection_evidence_max_chars,
            chunk_excerpt_max_chars=config.reflection_chunk_excerpt_max_chars,
        )
        if config.enable_reflection
        else None
    )

    review_eval_for_review = run_review_eval_with_config(
        ReviewEvalConfig(
            gold_dir=review_eval_gold_dir,
            pred_dir=review_dir,
            output_dir=review_eval_review_dir,
            summary_filename="review_eval_summary.json",
            warning_filename="review_eval_warnings.json",
            detail_jsonl_filename="review_eval_details.jsonl",
            detail_csv_filename="review_eval_details.csv",
            enable_detail_jsonl=config.review_eval_enable_detail_jsonl,
            enable_detail_csv=config.review_eval_enable_detail_csv,
            ablation_pred_only_universe=config.review_eval_ablation_pred_only_universe,
            ablation_standard_only_universe=config.review_eval_ablation_standard_only_universe,
            max_warning_samples=config.review_eval_max_warning_samples,
            gold_items_field=config.review_eval_gold_items_field,
            pred_items_field="review_items",
        ),
        logger,
    )

    review_eval_for_reflection = (
        run_review_eval_with_config(
            ReviewEvalConfig(
                gold_dir=review_eval_gold_dir,
                pred_dir=reflection_dir,
                output_dir=review_eval_reflection_dir,
                summary_filename="review_eval_summary.json",
                warning_filename="review_eval_warnings.json",
                detail_jsonl_filename="review_eval_details.jsonl",
                detail_csv_filename="review_eval_details.csv",
                enable_detail_jsonl=config.review_eval_enable_detail_jsonl,
                enable_detail_csv=config.review_eval_enable_detail_csv,
                ablation_pred_only_universe=config.review_eval_ablation_pred_only_universe,
                ablation_standard_only_universe=config.review_eval_ablation_standard_only_universe,
                max_warning_samples=config.review_eval_max_warning_samples,
                gold_items_field=config.review_eval_gold_items_field,
                pred_items_field="reflection_items",
            ),
            logger,
        )
        if config.enable_reflection
        else None
    )

    chunk_lengths = _load_chunk_lengths(chunk_outputs)
    review_micro = _extract_micro_metrics(review_eval_for_review.summary_payload)
    reflection_micro = (
        _extract_micro_metrics(review_eval_for_reflection.summary_payload)
        if review_eval_for_reflection is not None
        else {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    )

    return ArmMetrics(
        arm=arm,
        chunking_strategy=arm,
        chunk_size=config.chunk_size,
        file_count=len(files),
        chunk_file_count=len(chunk_outputs),
        chunk_count=len(chunk_lengths),
        avg_chunk_chars=mean(chunk_lengths) if chunk_lengths else 0.0,
        min_chunk_chars=min(chunk_lengths) if chunk_lengths else 0,
        max_chunk_chars=max(chunk_lengths) if chunk_lengths else 0,
        classified_file_count=len(classification_result.outputs),
        review_file_count=len(review_result.outputs),
        reflection_enabled=config.enable_reflection,
        reflection_file_count=(len(reflection_result.outputs) if reflection_result is not None else 0),
        review_item_count=review_result.metrics.emitted_item_count,
        reflection_item_count=(reflection_result.metrics.final_item_count if reflection_result is not None else 0),
        review_micro_precision=review_micro["precision"],
        review_micro_recall=review_micro["recall"],
        review_micro_f1=review_micro["f1"],
        review_micro_accuracy=review_micro["accuracy"],
        reflection_micro_precision=reflection_micro["precision"],
        reflection_micro_recall=reflection_micro["recall"],
        reflection_micro_f1=reflection_micro["f1"],
        reflection_micro_accuracy=reflection_micro["accuracy"],
        chunk_dir=str(chunk_dir),
        classified_dir=str(classified_dir),
        review_dir=str(review_dir),
        reflection_dir=str(reflection_dir),
        review_eval_gold_dir_used=str(review_eval_gold_dir),
        report_review_summary=str(review_eval_for_review.summary_path),
        report_reflection_summary=(
            str(review_eval_for_reflection.summary_path)
            if review_eval_for_reflection is not None
            else ""
        ),
    )


def _resolve_arms(ablation_mode: str) -> list[str]:
    """根据消融模式返回实验臂列表。"""

    if ablation_mode == "semantic_only":
        return ["semantic"]
    if ablation_mode == "hard_length_only":
        return ["hard_length"]
    return ["semantic", "hard_length"]


def _select_review_eval_gold_dir(*, arm: str, config: ChunkingAblationConfig) -> Path:
    """根据实验臂选择 review_eval 标准集目录。"""

    if arm == "hard_length":
        return config.review_eval_gold_dir_hard_length
    return config.review_eval_gold_dir


def _load_chunk_lengths(files: list[Path]) -> list[int]:
    """读取 chunk 文件并统计 chunk 长度。"""

    lengths: list[int] = []
    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        chunks = payload.get("chunks", [])
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            content = chunk.get("content", "")
            if isinstance(content, str):
                lengths.append(len(content))
    return lengths


def _extract_micro_metrics(summary_payload: dict[str, object]) -> dict[str, float]:
    """从 review_eval 输出中提取 micro 指标。"""

    micro = summary_payload.get("micro")
    if not isinstance(micro, dict):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    return {
        "precision": float(micro.get("precision", 0.0)),
        "recall": float(micro.get("recall", 0.0)),
        "f1": float(micro.get("f1", 0.0)),
        "accuracy": float(micro.get("accuracy", 0.0)),
    }


def _build_summary_payload(
    *,
    run_id: str,
    config: ChunkingAblationConfig,
    arm_results: list[ArmMetrics],
) -> dict[str, object]:
    """构建实验总览与对比结果。"""

    payload: dict[str, object] = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config.to_dict(),
        "arms": [item.to_dict() for item in arm_results],
    }

    semantic = next((item for item in arm_results if item.arm == "semantic"), None)
    hard_length = next((item for item in arm_results if item.arm == "hard_length"), None)
    if semantic is not None and hard_length is not None:
        delta: dict[str, object] = {
            "chunk_count": hard_length.chunk_count - semantic.chunk_count,
            "avg_chunk_chars": hard_length.avg_chunk_chars - semantic.avg_chunk_chars,
            "review_micro_f1": hard_length.review_micro_f1 - semantic.review_micro_f1,
            "review_micro_precision": hard_length.review_micro_precision - semantic.review_micro_precision,
            "review_micro_recall": hard_length.review_micro_recall - semantic.review_micro_recall,
        }
        if semantic.reflection_enabled and hard_length.reflection_enabled:
            delta["reflection_micro_f1"] = hard_length.reflection_micro_f1 - semantic.reflection_micro_f1
            delta["reflection_micro_precision"] = (
                hard_length.reflection_micro_precision - semantic.reflection_micro_precision
            )
            delta["reflection_micro_recall"] = hard_length.reflection_micro_recall - semantic.reflection_micro_recall
        payload["delta"] = delta
    return payload


def _write_report_csv(output_path: Path, arm_results: list[ArmMetrics], delta: object) -> None:
    """写出对比 CSV 报告。"""

    fieldnames = [
        "arm",
        "review_eval_gold_dir_used",
        "chunk_count",
        "avg_chunk_chars",
        "review_micro_precision",
        "review_micro_recall",
        "review_micro_f1",
        "reflection_enabled",
        "reflection_micro_precision",
        "reflection_micro_recall",
        "reflection_micro_f1",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in arm_results:
            writer.writerow(
                {
                    "arm": result.arm,
                    "review_eval_gold_dir_used": result.review_eval_gold_dir_used,
                    "chunk_count": result.chunk_count,
                    "avg_chunk_chars": f"{result.avg_chunk_chars:.4f}",
                    "review_micro_precision": f"{result.review_micro_precision:.6f}",
                    "review_micro_recall": f"{result.review_micro_recall:.6f}",
                    "review_micro_f1": f"{result.review_micro_f1:.6f}",
                    "reflection_enabled": str(result.reflection_enabled).lower(),
                    "reflection_micro_precision": f"{result.reflection_micro_precision:.6f}",
                    "reflection_micro_recall": f"{result.reflection_micro_recall:.6f}",
                    "reflection_micro_f1": f"{result.reflection_micro_f1:.6f}",
                }
            )
        if isinstance(delta, dict):
            writer.writerow(
                {
                    "arm": "hard_minus_semantic",
                    "review_eval_gold_dir_used": "",
                    "chunk_count": delta.get("chunk_count", ""),
                    "avg_chunk_chars": f"{float(delta.get('avg_chunk_chars', 0.0)):.4f}",
                    "review_micro_precision": f"{float(delta.get('review_micro_precision', 0.0)):.6f}",
                    "review_micro_recall": f"{float(delta.get('review_micro_recall', 0.0)):.6f}",
                    "review_micro_f1": f"{float(delta.get('review_micro_f1', 0.0)):.6f}",
                    "reflection_enabled": "",
                    "reflection_micro_precision": f"{float(delta.get('reflection_micro_precision', 0.0)):.6f}",
                    "reflection_micro_recall": f"{float(delta.get('reflection_micro_recall', 0.0)):.6f}",
                    "reflection_micro_f1": f"{float(delta.get('reflection_micro_f1', 0.0)):.6f}",
                }
            )


def _build_markdown_report(
    *,
    run_id: str,
    config: ChunkingAblationConfig,
    arm_results: list[ArmMetrics],
    summary: dict[str, object],
) -> str:
    """构建 Markdown 实验报告。"""

    lines = [
        "# Chunking 消融实验报告",
        "",
        f"- run_id: {run_id}",
        f"- generated_at: {summary.get('generated_at', '')}",
        f"- ablation_mode: {config.ablation_mode}",
        f"- classification_mode: {config.classification_mode}",
        f"- enable_reflection: {str(config.enable_reflection).lower()}",
        f"- chunk_size: {config.chunk_size}",
        f"- input_dir: {config.input_dir}",
        f"- review_eval_gold_dir_semantic: {config.review_eval_gold_dir}",
        f"- review_eval_gold_dir_hard_length: {config.review_eval_gold_dir_hard_length}",
        "",
        "## 实验臂指标",
        "",
        "| arm | gold_dir | reflection_enabled | chunk_count | avg_chunk_chars | review_f1 | reflection_f1 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in arm_results:
        lines.append(
            "| %s | %s | %s | %s | %.4f | %.6f | %.6f |"
            % (
                item.arm,
                item.review_eval_gold_dir_used,
                str(item.reflection_enabled).lower(),
                item.chunk_count,
                item.avg_chunk_chars,
                item.review_micro_f1,
                item.reflection_micro_f1,
            )
        )

    delta = summary.get("delta")
    if isinstance(delta, dict):
        lines.extend(
            [
                "",
                "## 消融差值（hard_length - semantic）",
                "",
                "| metric | delta |",
                "|---|---:|",
                f"| chunk_count | {int(delta.get('chunk_count', 0))} |",
                f"| avg_chunk_chars | {float(delta.get('avg_chunk_chars', 0.0)):.4f} |",
                f"| review_micro_f1 | {float(delta.get('review_micro_f1', 0.0)):.6f} |",
            ]
        )
        if "reflection_micro_f1" in delta:
            lines.append(f"| reflection_micro_f1 | {float(delta.get('reflection_micro_f1', 0.0)):.6f} |")

    lines.extend(
        [
            "",
            "## 结构化产物",
            "",
            "- metrics/ablation_run_config.json",
            "- metrics/arm_metrics_*.json",
            "- metrics/chunking_ablation_summary.json",
            "- report/chunking_ablation_report.csv",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_run_id(config: ChunkingAblationConfig) -> str:
    """构建稳定可追溯的 run_id。"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fingerprint = hashlib.sha1(
        json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in config.run_name).strip("_") or "ablation"
    return f"{safe_name}_{timestamp}_{fingerprint}"


def _resolve_path(path: Path | str) -> Path:
    """将路径解析为绝对路径。"""

    value = path if isinstance(path, Path) else Path(str(path))
    return value if value.is_absolute() else PROJECT_ROOT / value


def _load_json(path: Path) -> dict[str, object]:
    """读取 JSON 配置文件。"""

    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"配置文件不存在: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件必须是 JSON Object: {resolved}")
    return payload


def _pick(payload: dict[str, object], key: str, default: object) -> object:
    """读取配置字段，且仅在值为 None 时使用默认值。"""

    value = payload.get(key)
    return default if value is None else value


def _as_bool(value: object, default: bool) -> bool:
    """将任意值解析为布尔值。"""

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


if __name__ == "__main__":
    main()
