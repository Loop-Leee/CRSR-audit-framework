"""Review 标准集与预测结果评测脚本。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from src.tools.logger import Logger, get_logger


EvalKey = tuple[str, str, str]
POSITIVE_LABELS = {"不合格", "待复核"}
RISK_PRIORITY = {"不合格": 3, "待复核": 2, "合格": 1}
ITEM_FIELD_CHOICES = {"auto", "review_items", "reflection_items"}
AUTO_ITEM_FIELDS = ("reflection_items", "review_items")
DEFAULT_CONFIG_PATH = Path("src/experiment/review_eval_config.json")


@dataclass(frozen=True, slots=True)
class ReviewEvalConfig:
    """评测配置。

    Args:
        gold_dir: 标准集目录路径。
        pred_dir: 预测集目录路径。
        output_dir: 输出目录路径。
        summary_filename: 汇总指标文件名（JSON）。
        warning_filename: warning 文件名（JSON）。
        detail_jsonl_filename: 明细文件名（JSONL）。
        detail_csv_filename: 明细文件名（CSV）。
        enable_detail_jsonl: 是否写出 JSONL 明细。
        enable_detail_csv: 是否写出 CSV 明细。
        ablation_pred_only_universe: 消融开关，仅使用预测键构建评测宇宙。
        ablation_standard_only_universe: 消融开关，仅使用标准集键构建评测宇宙。
        max_warning_samples: 每类 warning 最多记录样例数量。
        gold_items_field: 标准集 item 列表字段（`auto/review_items/reflection_items`）。
        pred_items_field: 预测集 item 列表字段（`auto/review_items/reflection_items`）。
    """

    gold_dir: Path
    pred_dir: Path
    output_dir: Path
    summary_filename: str
    warning_filename: str
    detail_jsonl_filename: str
    detail_csv_filename: str
    enable_detail_jsonl: bool
    enable_detail_csv: bool
    ablation_pred_only_universe: bool
    ablation_standard_only_universe: bool
    max_warning_samples: int
    gold_items_field: str
    pred_items_field: str

    def to_dict(self) -> dict[str, str | int | bool]:
        """导出配置快照。"""

        return {
            "gold_dir": str(self.gold_dir),
            "pred_dir": str(self.pred_dir),
            "output_dir": str(self.output_dir),
            "summary_filename": self.summary_filename,
            "warning_filename": self.warning_filename,
            "detail_jsonl_filename": self.detail_jsonl_filename,
            "detail_csv_filename": self.detail_csv_filename,
            "enable_detail_jsonl": self.enable_detail_jsonl,
            "enable_detail_csv": self.enable_detail_csv,
            "ablation_pred_only_universe": self.ablation_pred_only_universe,
            "ablation_standard_only_universe": self.ablation_standard_only_universe,
            "max_warning_samples": self.max_warning_samples,
            "gold_items_field": self.gold_items_field,
            "pred_items_field": self.pred_items_field,
        }


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    """单条评测记录（去重后）。"""

    key: EvalKey
    risk_type: str
    raw_label: str
    binary_label: int
    priority: int
    source_file: str
    row_index: int


@dataclass(frozen=True, slots=True)
class DatasetLoadResult:
    """数据集加载结果。"""

    records: dict[EvalKey, ReviewRecord]
    doc_to_keys: dict[str, set[EvalKey]]
    scanned_file_count: int
    valid_doc_count: int
    scanned_item_count: int
    merged_key_count: int
    item_field_usage: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """导出统计信息。"""

        return {
            "scanned_file_count": self.scanned_file_count,
            "valid_doc_count": self.valid_doc_count,
            "scanned_item_count": self.scanned_item_count,
            "merged_key_count": self.merged_key_count,
            "item_field_usage": dict(self.item_field_usage),
        }


@dataclass(frozen=True, slots=True)
class ReviewEvalRunResult:
    """review_eval 执行结果。"""

    summary_path: Path
    warning_path: Path
    detail_jsonl_path: Path | None
    detail_csv_path: Path | None
    summary_payload: dict[str, object]
    warning_total: int


@dataclass(slots=True)
class ConfusionMatrix:
    """二分类混淆矩阵。"""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, gold_label: int, pred_label: int) -> str:
        """累计单条样本并返回判定标签。"""

        if gold_label == 1 and pred_label == 1:
            self.tp += 1
            return "TP"
        if gold_label == 0 and pred_label == 1:
            self.fp += 1
            return "FP"
        if gold_label == 1 and pred_label == 0:
            self.fn += 1
            return "FN"
        self.tn += 1
        return "TN"

    def to_metrics(self) -> dict[str, int | float]:
        """输出 TP/FP/FN/TN 与四项指标。"""

        precision = _safe_div(self.tp, self.tp + self.fp)
        recall = _safe_div(self.tp, self.tp + self.fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        accuracy = _safe_div(self.tp + self.tn, self.tp + self.fp + self.fn + self.tn)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }


class WarningCollector:
    """warning 聚合器（计数 + 样例）。"""

    def __init__(self, logger: Logger, max_samples: int) -> None:
        self._logger = logger
        self._max_samples = max(1, int(max_samples))
        self._counts: Counter[str] = Counter()
        self._samples: dict[str, list[str]] = defaultdict(list)

    def add(self, code: str, message: str) -> None:
        """记录 warning。

        仅在样例数量未超过上限时写 error 日志，避免日志噪音。
        """

        self._counts[code] += 1
        if len(self._samples[code]) < self._max_samples:
            self._samples[code].append(message)
            self._logger.error(f"{code}: {message}")

    def to_dict(self) -> dict[str, dict[str, int] | dict[str, list[str]]]:
        """导出 warning 结果。"""

        return {
            "counts": dict(self._counts),
            "samples": dict(self._samples),
        }

    def total_count(self) -> int:
        """warning 总数。"""

        return sum(self._counts.values())


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="评估标准审查集与模型审查结果的一致性。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="配置文件路径")
    parser.add_argument("--gold-dir", type=Path, default=None, help="标准集目录（覆盖配置）")
    parser.add_argument("--pred-dir", type=Path, default=None, help="预测集目录（覆盖配置）")
    parser.add_argument(
        "--gold-items-field",
        type=str,
        default=None,
        choices=sorted(ITEM_FIELD_CHOICES),
        help="标准集 item 数组字段（auto/review_items/reflection_items）",
    )
    parser.add_argument(
        "--pred-items-field",
        type=str,
        default=None,
        choices=sorted(ITEM_FIELD_CHOICES),
        help="预测集 item 数组字段（auto/review_items/reflection_items）",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录（覆盖配置）")
    parser.add_argument("--summary-file", type=str, default=None, help="汇总 JSON 文件名（覆盖配置）")
    parser.add_argument("--warning-file", type=str, default=None, help="warning JSON 文件名（覆盖配置）")
    parser.add_argument("--detail-jsonl-file", type=str, default=None, help="明细 JSONL 文件名（覆盖配置）")
    parser.add_argument("--detail-csv-file", type=str, default=None, help="明细 CSV 文件名（覆盖配置）")
    parser.add_argument("--disable-detail-jsonl", action="store_true", help="不输出明细 JSONL")
    parser.add_argument("--disable-detail-csv", action="store_true", help="不输出明细 CSV")
    parser.add_argument(
        "--ablation-pred-only-universe",
        action="store_true",
        help="消融开关：评测宇宙仅使用预测键（默认关闭，标准口径为 P(doc) ∪ G(doc正类)）",
    )
    parser.add_argument(
        "--ablation-standard-only-universe",
        action="store_true",
        help="消融开关：评测宇宙仅使用标准集键（以 dataset/standard-review 为统计口径）",
    )
    parser.add_argument("--max-warning-samples", type=int, default=None, help="每类 warning 样例上限（覆盖配置）")
    return parser.parse_args()


def main() -> None:
    """执行评测流程。"""

    args = parse_args()
    config = build_runtime_config(args)
    logger = get_logger("experiment")
    logger.info(f"日志文件: {logger.path}")
    result = run_review_eval_with_config(config, logger)

    print(f"[OK] summary -> {result.summary_path}")
    print(f"[OK] warnings -> {result.warning_path}")
    if result.detail_jsonl_path is not None:
        print(f"[OK] details_jsonl -> {result.detail_jsonl_path}")
    if result.detail_csv_path is not None:
        print(f"[OK] details_csv -> {result.detail_csv_path}")
    print(json.dumps(result.summary_payload, ensure_ascii=False, indent=2))


def run_review_eval_with_config(config: ReviewEvalConfig, logger: Logger) -> ReviewEvalRunResult:
    """按给定配置执行 review_eval，并返回产物路径与汇总指标。"""

    logger.info(f"review_eval_config: {json.dumps(config.to_dict(), ensure_ascii=False)}")
    warnings = WarningCollector(logger=logger, max_samples=config.max_warning_samples)
    gold = load_dataset(
        dataset_name="gold",
        input_dir=config.gold_dir,
        label_field="ground_truth",
        label_mapper=map_gold_label,
        items_field_preference=config.gold_items_field,
        warnings=warnings,
    )
    pred = load_dataset(
        dataset_name="pred",
        input_dir=config.pred_dir,
        label_field="result",
        label_mapper=map_pred_label,
        items_field_preference=config.pred_items_field,
        warnings=warnings,
    )

    summary_payload, detail_rows = evaluate_datasets(
        gold=gold,
        pred=pred,
        ablation_pred_only_universe=config.ablation_pred_only_universe,
        ablation_standard_only_universe=config.ablation_standard_only_universe,
        warnings=warnings,
    )
    summary_payload["meta"] = {
        "key_schema": ["doc_id", "chunk_id", "risk_type"],
        "universe_mode": _resolve_universe_mode(
            ablation_pred_only_universe=config.ablation_pred_only_universe,
            ablation_standard_only_universe=config.ablation_standard_only_universe,
        ),
        "gold": gold.to_dict(),
        "pred": pred.to_dict(),
        "evaluated_key_count": len(detail_rows),
        "ablation_pred_only_universe": config.ablation_pred_only_universe,
        "ablation_standard_only_universe": config.ablation_standard_only_universe,
        "gold_items_field": config.gold_items_field,
        "pred_items_field": config.pred_items_field,
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / config.summary_filename
    warning_path = config.output_dir / config.warning_filename
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    warning_path.write_text(json.dumps(warnings.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    detail_jsonl_path: Path | None = None
    detail_csv_path: Path | None = None
    if config.enable_detail_jsonl:
        detail_jsonl_path = config.output_dir / config.detail_jsonl_filename
        write_detail_jsonl(detail_rows, detail_jsonl_path)
    if config.enable_detail_csv:
        detail_csv_path = config.output_dir / config.detail_csv_filename
        write_detail_csv(detail_rows, detail_csv_path)

    warning_total = warnings.total_count()
    logger.info(
        "review_eval_done: summary=%s, warnings=%s, detail_jsonl=%s, detail_csv=%s, warning_total=%s"
        % (summary_path, warning_path, detail_jsonl_path, detail_csv_path, warning_total)
    )
    return ReviewEvalRunResult(
        summary_path=summary_path,
        warning_path=warning_path,
        detail_jsonl_path=detail_jsonl_path,
        detail_csv_path=detail_csv_path,
        summary_payload=summary_payload,
        warning_total=warning_total,
    )


def build_runtime_config(args: argparse.Namespace) -> ReviewEvalConfig:
    """构建运行配置（配置文件 + CLI 覆盖）。"""

    raw = load_config(args.config)
    gold_dir = resolve_path(args.gold_dir or Path(str(raw.get("gold_dir", "dataset/standard-review"))))
    pred_dir = resolve_path(args.pred_dir or Path(str(raw.get("pred_dir", "data/4-review"))))
    output_dir = resolve_path(args.output_dir or Path(str(raw.get("output_dir", "data/experiments/review_eval"))))
    summary_filename = str(args.summary_file or raw.get("summary_filename", "review_eval_summary.json")).strip()
    warning_filename = str(args.warning_file or raw.get("warning_filename", "review_eval_warnings.json")).strip()
    detail_jsonl_filename = str(
        args.detail_jsonl_file or raw.get("detail_jsonl_filename", "review_eval_details.jsonl")
    ).strip()
    detail_csv_filename = str(args.detail_csv_file or raw.get("detail_csv_filename", "review_eval_details.csv")).strip()

    enable_detail_jsonl = _as_bool(raw.get("enable_detail_jsonl"), True)
    enable_detail_csv = _as_bool(raw.get("enable_detail_csv"), True)
    if args.disable_detail_jsonl:
        enable_detail_jsonl = False
    if args.disable_detail_csv:
        enable_detail_csv = False

    ablation_pred_only_universe = _as_bool(raw.get("ablation_pred_only_universe"), False)
    if args.ablation_pred_only_universe:
        ablation_pred_only_universe = True

    ablation_standard_only_universe = _as_bool(raw.get("ablation_standard_only_universe"), False)
    if args.ablation_standard_only_universe:
        ablation_standard_only_universe = True

    if ablation_pred_only_universe and ablation_standard_only_universe:
        raise ValueError(
            "--ablation-pred-only-universe 与 --ablation-standard-only-universe 不能同时启用。"
        )

    max_warning_samples = _as_int(raw.get("max_warning_samples"), 30)
    if args.max_warning_samples is not None:
        max_warning_samples = max(1, int(args.max_warning_samples))
    gold_items_field = _normalize_items_field(
        args.gold_items_field or str(raw.get("gold_items_field", "auto")).strip()
    )
    pred_items_field = _normalize_items_field(
        args.pred_items_field or str(raw.get("pred_items_field", "auto")).strip()
    )

    if not summary_filename:
        raise ValueError("summary_filename 不能为空。")
    if not warning_filename:
        raise ValueError("warning_filename 不能为空。")
    if not detail_jsonl_filename:
        raise ValueError("detail_jsonl_filename 不能为空。")
    if not detail_csv_filename:
        raise ValueError("detail_csv_filename 不能为空。")
    if not gold_dir.exists():
        raise FileNotFoundError(f"标准集目录不存在: {gold_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"预测集目录不存在: {pred_dir}")

    return ReviewEvalConfig(
        gold_dir=gold_dir,
        pred_dir=pred_dir,
        output_dir=output_dir,
        summary_filename=summary_filename,
        warning_filename=warning_filename,
        detail_jsonl_filename=detail_jsonl_filename,
        detail_csv_filename=detail_csv_filename,
        enable_detail_jsonl=enable_detail_jsonl,
        enable_detail_csv=enable_detail_csv,
        ablation_pred_only_universe=ablation_pred_only_universe,
        ablation_standard_only_universe=ablation_standard_only_universe,
        max_warning_samples=max_warning_samples,
        gold_items_field=gold_items_field,
        pred_items_field=pred_items_field,
    )


def load_config(config_path: Path) -> dict[str, object]:
    """读取 JSON 配置文件。"""

    path = resolve_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件必须是 JSON Object: {path}")
    return payload


def load_dataset(
    *,
    dataset_name: str,
    input_dir: Path,
    label_field: str,
    label_mapper: Callable[[str, WarningCollector, str, EvalKey, int], tuple[int, int, str]],
    items_field_preference: str,
    warnings: WarningCollector,
) -> DatasetLoadResult:
    """加载数据集并按 key 去重。

    Args:
        dataset_name: 数据集名称（`gold` 或 `pred`）。
        input_dir: 输入目录路径。
        label_field: 标签字段名（`ground_truth` 或 `result`）。
        label_mapper: 标签映射函数。
        warnings: warning 收集器。

    Returns:
        DatasetLoadResult: 去重后的记录与统计信息。
    """

    files = discover_json_files(input_dir)
    records: dict[EvalKey, ReviewRecord] = {}
    doc_to_keys: dict[str, set[EvalKey]] = defaultdict(set)
    doc_to_files: dict[str, list[str]] = defaultdict(list)
    item_field_usage: Counter[str] = Counter()
    scanned_items = 0

    for file_path in files:
        payload = read_json_payload(file_path, dataset_name=dataset_name, warnings=warnings)
        if payload is None:
            continue

        doc_id = clean_text(payload.get("doc_id"))
        if not doc_id:
            warnings.add(
                "missing_doc_id",
                f"{dataset_name}: file={file_path} 缺少 doc_id，已跳过。",
            )
            continue
        resolved = resolve_items_array(
            payload=payload,
            dataset_name=dataset_name,
            file_path=file_path,
            items_field_preference=items_field_preference,
            warnings=warnings,
        )
        if resolved is None:
            continue
        review_items, item_field_name = resolved
        item_field_usage[item_field_name] += 1

        doc_to_files[doc_id].append(str(file_path))
        if len(doc_to_files[doc_id]) > 1:
            warnings.add(
                "duplicate_doc_id",
                f"{dataset_name}: doc_id={doc_id} 在多文件出现，files={doc_to_files[doc_id]}",
            )

        for row_index, item in enumerate(review_items, start=1):
            scanned_items += 1
            if not isinstance(item, dict):
                warnings.add(
                    "invalid_review_item_type",
                    f"{dataset_name}: file={file_path} row={row_index} 不是对象，已跳过。",
                )
                continue

            key = extract_key(
                doc_id=doc_id,
                item=item,
                dataset_name=dataset_name,
                file_path=file_path,
                row_index=row_index,
                warnings=warnings,
            )
            if key is None:
                continue

            risk_type = clean_text(item.get("risk_type"))
            raw_label = clean_text(item.get(label_field))
            binary_label, priority, normalized_label = label_mapper(raw_label, warnings, dataset_name, key, row_index)
            record = ReviewRecord(
                key=key,
                risk_type=risk_type,
                raw_label=normalized_label,
                binary_label=binary_label,
                priority=priority,
                source_file=str(file_path),
                row_index=row_index,
            )
            doc_to_keys[doc_id].add(key)
            merge_record(records, record, dataset_name=dataset_name, warnings=warnings)

    return DatasetLoadResult(
        records=records,
        doc_to_keys=dict(doc_to_keys),
        scanned_file_count=len(files),
        valid_doc_count=len(doc_to_keys),
        scanned_item_count=scanned_items,
        merged_key_count=len(records),
        item_field_usage=dict(item_field_usage),
    )


def resolve_items_array(
    *,
    payload: dict[str, object],
    dataset_name: str,
    file_path: Path,
    items_field_preference: str,
    warnings: WarningCollector,
) -> tuple[list[object], str] | None:
    """按字段偏好解析 item 数组。"""

    preference = _normalize_items_field(items_field_preference)
    candidate_fields = AUTO_ITEM_FIELDS if preference == "auto" else (preference,)

    for field in candidate_fields:
        value = payload.get(field)
        if isinstance(value, list):
            return value, field

    if preference == "auto":
        warnings.add(
            "missing_item_array",
            (
                f"{dataset_name}: file={file_path} 缺少可用 item 数组，"
                "期望字段之一为 reflection_items/review_items。"
            ),
        )
    else:
        warnings.add(
            "missing_item_array",
            f"{dataset_name}: file={file_path} 缺少 {preference} 数组，已跳过。",
        )
    return None


def evaluate_datasets(
    *,
    gold: DatasetLoadResult,
    pred: DatasetLoadResult,
    ablation_pred_only_universe: bool,
    ablation_standard_only_universe: bool,
    warnings: WarningCollector,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """按统一评测宇宙计算指标并返回逐 key 明细。"""

    gold_docs = set(gold.doc_to_keys.keys())
    pred_docs = set(pred.doc_to_keys.keys())
    for doc_id in sorted(gold_docs - pred_docs):
        warnings.add("doc_id_not_aligned", f"doc_id={doc_id} 仅在标准集中出现。")
    for doc_id in sorted(pred_docs - gold_docs):
        warnings.add("doc_id_not_aligned", f"doc_id={doc_id} 仅在预测集中出现。")

    micro = ConfusionMatrix()
    per_risk: dict[str, ConfusionMatrix] = defaultdict(ConfusionMatrix)
    detail_rows: list[dict[str, object]] = []

    for doc_id in sorted(gold_docs | pred_docs):
        pred_keys = set(pred.doc_to_keys.get(doc_id, set()))
        gold_keys = set(gold.doc_to_keys.get(doc_id, set()))
        if ablation_pred_only_universe:
            doc_universe = pred_keys
        elif ablation_standard_only_universe:
            doc_universe = gold_keys
        else:
            gold_positive_keys = {
                key
                for key in gold_keys
                if gold.records.get(key) is not None and gold.records[key].binary_label == 1
            }
            doc_universe = pred_keys | gold_positive_keys

        for key in sorted(doc_universe, key=sort_key):
            gold_record = gold.records.get(key)
            pred_record = pred.records.get(key)
            gold_label = gold_record.binary_label if gold_record is not None else 0
            pred_label = pred_record.binary_label if pred_record is not None else 0
            judge = micro.add(gold_label, pred_label)
            risk_type = key[2]
            per_risk[risk_type].add(gold_label, pred_label)
            detail_rows.append(
                {
                    "doc_id": key[0],
                    "chunk_id": key[1],
                    "risk_type": risk_type,
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "judge": judge,
                }
            )

    per_risk_payload = {
        risk: matrix.to_metrics()
        for risk, matrix in sorted(per_risk.items(), key=lambda pair: pair[0])
    }
    macro_payload = compute_macro_metrics(per_risk_payload.values())
    summary = {
        "micro": micro.to_metrics(),
        "macro_by_risk_type": macro_payload,
        "per_risk_type": per_risk_payload,
    }
    return summary, detail_rows


def discover_json_files(input_dir: Path) -> list[Path]:
    """发现目录下待处理 JSON 文件。"""

    if input_dir.is_file():
        if input_dir.suffix.lower() != ".json":
            raise ValueError(f"输入文件必须是 .json: {input_dir}")
        return [input_dir]
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # 优先匹配业务产物，避免把 metrics/config 等非样本 JSON 混入评测。
    files = sorted(input_dir.glob("*.review.json"))
    if not files:
        files = sorted(input_dir.glob("*.reflection.json"))
    if not files:
        files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"目录下未找到 JSON 文件: {input_dir}")
    return files


def read_json_payload(
    file_path: Path,
    *,
    dataset_name: str,
    warnings: WarningCollector,
) -> dict[str, object] | None:
    """读取 JSON 文件内容。"""

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        warnings.add("invalid_json_file", f"{dataset_name}: file={file_path} 读取失败: {error}")
        return None
    if not isinstance(payload, dict):
        warnings.add("invalid_json_payload", f"{dataset_name}: file={file_path} 顶层不是 JSON Object。")
        return None
    return payload


def extract_key(
    *,
    doc_id: str,
    item: dict[str, object],
    dataset_name: str,
    file_path: Path,
    row_index: int,
    warnings: WarningCollector,
) -> EvalKey | None:
    """提取 `(doc_id, chunk_id, risk_type)`。"""

    chunk_id = clean_text(item.get("chunk_id"))
    risk_type = clean_text(item.get("risk_type"))
    if not chunk_id or not risk_type:
        warnings.add(
            "missing_required_field",
            (
                f"{dataset_name}: file={file_path} row={row_index} 缺少必要字段，"
                f"chunk_id={chunk_id or '<missing>'}, "
                f"risk_type={risk_type or '<missing>'}"
            ),
        )
        return None
    return doc_id, chunk_id, risk_type


def merge_record(
    records: dict[EvalKey, ReviewRecord],
    incoming: ReviewRecord,
    *,
    dataset_name: str,
    warnings: WarningCollector,
) -> None:
    """按风险优先级合并重复 key。"""

    existing = records.get(incoming.key)
    if existing is None:
        records[incoming.key] = incoming
        return

    warnings.add(
        "duplicate_key_merged",
        (
            f"{dataset_name}: key={incoming.key} 出现重复，"
            f"old_label={existing.raw_label or '<empty>'}, new_label={incoming.raw_label or '<empty>'}"
        ),
    )
    if incoming.priority > existing.priority:
        records[incoming.key] = incoming


def map_gold_label(
    raw_label: str,
    warnings: WarningCollector,
    dataset_name: str,
    key: EvalKey,
    row_index: int,
) -> tuple[int, int, str]:
    """映射 gold 标签。"""

    normalized = raw_label.strip()
    if normalized in POSITIVE_LABELS:
        return 1, RISK_PRIORITY.get(normalized, 0), normalized
    if normalized == "合格":
        return 0, RISK_PRIORITY["合格"], normalized
    if not normalized:
        warnings.add("missing_gold_label", f"{dataset_name}: key={key} row={row_index} ground_truth 为空，按 0 处理。")
        return 0, 0, normalized
    warnings.add("unknown_gold_label", f"{dataset_name}: key={key} row={row_index} ground_truth={normalized}，按 0 处理。")
    return 0, 0, normalized


def map_pred_label(
    raw_label: str,
    warnings: WarningCollector,
    dataset_name: str,
    key: EvalKey,
    row_index: int,
) -> tuple[int, int, str]:
    """映射 pred 标签。"""

    normalized = raw_label.strip()
    if normalized in POSITIVE_LABELS:
        return 1, RISK_PRIORITY.get(normalized, 0), normalized
    if normalized == "合格":
        return 0, RISK_PRIORITY["合格"], normalized
    if not normalized:
        warnings.add("missing_pred_label", f"{dataset_name}: key={key} row={row_index} result 为空，按 0 处理。")
        return 0, 0, normalized
    warnings.add("unknown_pred_label", f"{dataset_name}: key={key} row={row_index} result={normalized}，按 0 处理。")
    return 0, 0, normalized


def compute_macro_metrics(metrics: Iterable[dict[str, int | float]]) -> dict[str, float]:
    """计算按 risk_type 的宏平均。"""

    rows = list(metrics)
    if not rows:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }
    return {
        "precision": sum(float(item["precision"]) for item in rows) / len(rows),
        "recall": sum(float(item["recall"]) for item in rows) / len(rows),
        "f1": sum(float(item["f1"]) for item in rows) / len(rows),
        "accuracy": sum(float(item["accuracy"]) for item in rows) / len(rows),
    }


def write_detail_jsonl(rows: list[dict[str, object]], output_path: Path) -> None:
    """写出明细 JSONL。"""

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_detail_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """写出明细 CSV。"""

    fieldnames = ["doc_id", "chunk_id", "risk_type", "gold_label", "pred_label", "judge"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def clean_text(value: object) -> str:
    """统一文本归一化。"""

    if value is None:
        return ""
    return str(value).strip()


def resolve_path(path: Path) -> Path:
    """将相对路径解析为项目绝对路径。"""

    return path if path.is_absolute() else Path.cwd() / path


def sort_key(key: EvalKey) -> tuple[str, int | str, str, str]:
    """评测键排序函数（doc_id, chunk_id, risk_type）。"""

    chunk_id = key[1]
    if chunk_id.isdigit():
        return key[0], int(chunk_id), "", key[2]
    return key[0], chunk_id, chunk_id, key[2]


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return default


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_items_field(value: str) -> str:
    normalized = clean_text(value).lower()
    if normalized not in ITEM_FIELD_CHOICES:
        raise ValueError(
            f"items_field 必须为 {sorted(ITEM_FIELD_CHOICES)} 之一，当前为: {value}"
        )
    return normalized


def _resolve_universe_mode(
    *,
    ablation_pred_only_universe: bool,
    ablation_standard_only_universe: bool,
) -> str:
    if ablation_pred_only_universe:
        return "pred_only"
    if ablation_standard_only_universe:
        return "standard_only"
    return "default"


if __name__ == "__main__":
    main()
