"""CUAD JSONL 指标分析脚本。

基于 ``src.exp_cuad.eval_metrics`` 的统一口径，读取 ``cuad_baseline*.jsonl``，
输出可复用的结构化指标结果（JSON）。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.tools.logger import get_logger

from .eval_metrics import (
    PairResult,
    compute_evidence_jaccard,
    compute_inconsistency_rate,
    compute_laziness_rate,
    compute_macro_f1,
    compute_presence_f1,
)


@dataclass(slots=True)
class OutputSwitches:
    """指标输出消融开关。"""

    emit_per_label_metrics: bool
    emit_llm_usage_aggregate: bool

    def to_dict(self) -> dict[str, bool]:
        """返回可序列化开关字典。"""

        return {
            "emit_per_label_metrics": self.emit_per_label_metrics,
            "emit_llm_usage_aggregate": self.emit_llm_usage_aggregate,
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="待分析的 CUAD 结果 JSONL 文件路径",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="",
        help="可选输出 JSON 路径；为空则仅打印到 stdout",
    )
    parser.add_argument(
        "--disable_per_label_metrics",
        action="store_true",
        help="消融：不输出 per-label 指标明细",
    )
    parser.add_argument(
        "--disable_llm_usage_aggregate",
        action="store_true",
        help="消融：不输出 llm_usage 聚合统计",
    )
    parser.add_argument(
        "--allow_invalid_rows",
        action="store_true",
        help="允许跳过非法行（默认严格模式，遇到非法行直接失败）",
    )
    return parser


def _as_bool(value: Any, default: bool = False) -> bool:
    """容错读取布尔值。"""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n", ""}:
            return False
    return default


def _as_str_list(value: Any) -> list[str]:
    """容错读取字符串列表。"""

    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str):
        return [value]
    return []


def _as_bool_list(value: Any) -> list[bool]:
    """容错读取布尔列表。"""

    if not isinstance(value, list):
        return []
    return [_as_bool(item, default=False) for item in value]


def _pair_result_from_row(row: dict[str, Any], *, row_index: int) -> PairResult:
    """将单行 JSON 转换为 ``PairResult``。"""

    doc_id = row.get("doc_id")
    label = row.get("label")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError(f"row={row_index}: doc_id 缺失或非法")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"row={row_index}: label 缺失或非法")

    return PairResult(
        doc_id=doc_id,
        label=label,
        present_pred=_as_bool(row.get("present_pred"), default=False),
        evidence_pred=_as_str_list(row.get("evidence_pred")),
        present_gt=_as_bool(row.get("present_gt"), default=False),
        evidence_gt=_as_str_list(row.get("evidence_gt")),
        chunk_votes=_as_bool_list(row.get("chunk_votes")),
    )


def _load_results(
    jsonl_path: Path,
    *,
    allow_invalid_rows: bool,
) -> tuple[list[PairResult], list[dict[str, Any]], dict[str, int]]:
    """加载 JSONL，返回结果列表、原始行和行级统计。"""

    results: list[PairResult] = []
    rows: list[dict[str, Any]] = []
    stats = {"line_total": 0, "line_valid": 0, "line_invalid": 0}

    for idx, raw_line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        stats["line_total"] += 1
        try:
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError("row 非对象")
            pair = _pair_result_from_row(row, row_index=idx)
        except Exception as exc:  # noqa: BLE001
            stats["line_invalid"] += 1
            if not allow_invalid_rows:
                raise ValueError(f"解析失败: line={idx}, error={exc}") from exc
            continue

        rows.append(row)
        results.append(pair)
        stats["line_valid"] += 1

    return results, rows, stats


def _compute_dataset_stats(results: list[PairResult]) -> dict[str, Any]:
    """计算样本层统计。"""

    doc_ids = sorted({item.doc_id for item in results})
    labels = sorted({item.label for item in results})
    pair_count = len(results)
    dense_expected_pairs = len(doc_ids) * len(labels)

    pred_positive = sum(1 for item in results if item.present_pred)
    gt_positive = sum(1 for item in results if item.present_gt)

    return {
        "pair_count": pair_count,
        "doc_count": len(doc_ids),
        "label_count": len(labels),
        "dense_expected_pairs": dense_expected_pairs,
        "is_dense_matrix": pair_count == dense_expected_pairs,
        "pred_positive_count": pred_positive,
        "gt_positive_count": gt_positive,
        "pred_positive_rate": (pred_positive / pair_count) if pair_count else 0.0,
        "gt_positive_rate": (gt_positive / pair_count) if pair_count else 0.0,
    }


def _compute_presence_metrics(results: list[PairResult]) -> dict[str, float]:
    """计算 presence 指标并补充 accuracy/specificity。"""

    micro = compute_presence_f1(results)
    tp = int(micro["tp"])
    fp = int(micro["fp"])
    fn = int(micro["fn"])
    tn = int(micro["tn"])
    total = tp + fp + fn + tn
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    balanced_accuracy = (micro["recall"] + specificity) / 2

    return {
        "precision": float(micro["precision"]),
        "recall": float(micro["recall"]),
        "f1": float(micro["f1"]),
        "accuracy": accuracy,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _compute_per_label_metrics(results: list[PairResult]) -> list[dict[str, Any]]:
    """按 label 计算指标明细。"""

    grouped: dict[str, list[PairResult]] = defaultdict(list)
    for item in results:
        grouped[item.label].append(item)

    rows: list[dict[str, Any]] = []
    for label in sorted(grouped):
        metrics = compute_presence_f1(grouped[label])
        support = len(grouped[label])
        gt_positive_count = sum(1 for item in grouped[label] if item.present_gt)
        pred_positive_count = sum(1 for item in grouped[label] if item.present_pred)
        rows.append(
            {
                "label": label,
                "support": support,
                "gt_positive_count": gt_positive_count,
                "pred_positive_count": pred_positive_count,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "tn": int(metrics["tn"]),
            }
        )
    return rows


def _compute_llm_usage_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合 llm_usage 字段（若存在）。"""

    keys = [
        "call_count",
        "token_in",
        "token_out",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "latency_ms",
        "retry_count",
        "cache_hit_count",
        "error_count",
        "total_tokens_estimated_count",
    ]
    totals: dict[str, float] = {key: 0.0 for key in keys}

    rows_with_usage = 0
    for row in rows:
        usage = row.get("llm_usage")
        if not isinstance(usage, dict):
            continue
        rows_with_usage += 1
        for key in keys:
            value = usage.get(key, 0)
            try:
                totals[key] += float(value)
            except (TypeError, ValueError):
                continue

    if rows_with_usage == 0:
        return {"rows_with_usage": 0}

    avg_per_pair = {
        key: (totals[key] / rows_with_usage)
        for key in ("call_count", "token_in", "token_out", "total_tokens", "latency_ms")
    }
    return {
        "rows_with_usage": rows_with_usage,
        "totals": totals,
        "avg_per_pair": avg_per_pair,
    }


def build_metrics_report(
    *,
    input_jsonl: Path,
    output_switches: OutputSwitches,
    allow_invalid_rows: bool,
    logger_name: str = "exp_cuad",
) -> dict[str, Any]:
    """构造结构化指标报告。"""

    logger = get_logger(logger_name)
    logger.info(f"开始分析: input_jsonl={input_jsonl}")
    logger.info(f"输出开关: {output_switches.to_dict()}")

    results, raw_rows, row_stats = _load_results(input_jsonl, allow_invalid_rows=allow_invalid_rows)
    if not results:
        raise ValueError("输入文件无有效样本，无法计算指标。")

    dataset_stats = _compute_dataset_stats(results)
    presence = _compute_presence_metrics(results)
    macro = compute_macro_f1(results)
    evidence_jaccard = compute_evidence_jaccard(results)
    laziness_rate = compute_laziness_rate(results)
    inconsistency_rate = compute_inconsistency_rate(results)

    report: dict[str, Any] = {
        "analysis_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_jsonl": str(input_jsonl.resolve()),
        "row_stats": row_stats,
        "dataset_stats": dataset_stats,
        "output_switches": output_switches.to_dict(),
        "metrics": {
            "presence_micro": presence,
            "presence_macro": {
                "macro_f1": float(macro["macro_f1"]),
                "num_labels": int(macro["num_labels"]),
            },
            "evidence_jaccard_gt_present": evidence_jaccard,
            "laziness_rate": laziness_rate,
            "inconsistency_rate": inconsistency_rate,
        },
    }

    if output_switches.emit_per_label_metrics:
        report["per_label_metrics"] = _compute_per_label_metrics(results)
    if output_switches.emit_llm_usage_aggregate:
        report["llm_usage_aggregate"] = _compute_llm_usage_aggregate(raw_rows)

    logger.info(
        "分析完成: pair_count=%s, doc_count=%s, label_count=%s, micro_f1=%.6f, macro_f1=%.6f"
        % (
            dataset_stats["pair_count"],
            dataset_stats["doc_count"],
            dataset_stats["label_count"],
            report["metrics"]["presence_micro"]["f1"],
            report["metrics"]["presence_macro"]["macro_f1"],
        )
    )
    return report


def main() -> None:
    """CLI 入口。"""

    args = _build_arg_parser().parse_args()
    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_jsonl}")

    output_switches = OutputSwitches(
        emit_per_label_metrics=not args.disable_per_label_metrics,
        emit_llm_usage_aggregate=not args.disable_llm_usage_aggregate,
    )
    report = build_metrics_report(
        input_jsonl=input_jsonl,
        output_switches=output_switches,
        allow_invalid_rows=args.allow_invalid_rows,
    )

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    if args.out_json:
        out_path = Path(args.out_json)
        if str(out_path.parent) not in {"", "."}:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
