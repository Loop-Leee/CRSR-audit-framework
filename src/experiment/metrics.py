"""实验指标计算。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from src.classification.classification_pipeline import ChunkClassificationDiagnostic


ChunkKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class ExperimentMetrics:
    """统一实验指标。

    Args:
        precision: 微平均 Precision；无标注集时为 None。
        recall: 微平均 Recall；无标注集时为 None。
        f1: 微平均 F1；无标注集时为 None。
        schema_valid_rate: LLM 返回可解析 JSON 比例。
        avg_token: 平均 token 消耗（`token_in + token_out`）。
        avg_latency_ms: 平均请求时延（毫秒）。
        conflict_rate: keyword 与 semantic 结果冲突率。
        cache_hit_rate: LLM 缓存命中率。
        llm_error_rate: LLM 调用失败率（含 schema 失败）。
        chunk_count: chunk 总数。
        llm_called_count: 实际触发 LLM 调用的 chunk 数。
        label_support: 标注集中正例标签总数；无标注集时为 0。
    """

    precision: float | None
    recall: float | None
    f1: float | None
    schema_valid_rate: float
    avg_token: float
    avg_latency_ms: float
    conflict_rate: float
    cache_hit_rate: float
    llm_error_rate: float
    chunk_count: int
    llm_called_count: int
    label_support: int

    def to_dict(self) -> dict[str, float | int | None]:
        """导出字典，便于写入 CSV/JSON。"""

        return asdict(self)


def compute_experiment_metrics(
    classified_files: list[Path],
    diagnostics: list[ChunkClassificationDiagnostic],
    ground_truth_path: Path | None = None,
) -> ExperimentMetrics:
    """计算统一实验指标口径。

    Args:
        classified_files: 模型预测分类结果文件列表（`*.classified.json`）。
        diagnostics: chunk 级诊断信息（来自分类流水线）。
        ground_truth_path: 可选标注集路径，支持目录、JSON、JSONL。

    Returns:
        ExperimentMetrics: 指标聚合结果。

    Raises:
        FileNotFoundError: 传入的标注路径不存在。
        ValueError: 标注文件格式不支持或内容缺字段。
    """

    predictions = _load_predictions_from_classified(classified_files)
    ground_truth = _load_ground_truth(ground_truth_path)
    precision, recall, f1, label_support = _compute_prf1(predictions, ground_truth)

    llm_rows = [item for item in diagnostics if item.llm_called]
    schema_valid_rate = _safe_rate(sum(1 for item in llm_rows if item.schema_valid), len(llm_rows))
    avg_token = _safe_mean([item.token_in + item.token_out for item in llm_rows])
    avg_latency_ms = _safe_mean([item.latency_ms for item in llm_rows])
    conflict_rate = _safe_rate(sum(1 for item in diagnostics if item.conflict), len(diagnostics))
    cache_hit_rate = _safe_rate(sum(1 for item in llm_rows if item.cached), len(llm_rows))
    llm_error_rate = _safe_rate(sum(1 for item in llm_rows if item.error_code is not None), len(llm_rows))

    return ExperimentMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        schema_valid_rate=schema_valid_rate,
        avg_token=avg_token,
        avg_latency_ms=avg_latency_ms,
        conflict_rate=conflict_rate,
        cache_hit_rate=cache_hit_rate,
        llm_error_rate=llm_error_rate,
        chunk_count=len(diagnostics),
        llm_called_count=len(llm_rows),
        label_support=label_support,
    )


def _safe_mean(values: Iterable[float | int]) -> float:
    buffer = [float(item) for item in values]
    return mean(buffer) if buffer else 0.0


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _normalize_chunk_key(source_file: str, chunk_id: int | str) -> ChunkKey:
    return source_file, str(chunk_id)


def _normalize_risk_values(values: object) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(item) for item in values if isinstance(item, str)}


def _load_predictions_from_classified(files: list[Path]) -> dict[ChunkKey, set[str]]:
    mapping: dict[ChunkKey, set[str]] = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_file = str(payload.get("source_file", path.name))
        chunks = payload.get("chunks", [])
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_id = chunk.get("chunk_id")
            if chunk_id is None:
                continue
            mapping[_normalize_chunk_key(source_file, chunk_id)] = _normalize_risk_values(chunk.get("risk_type"))
    return mapping


def _load_ground_truth(path: Path | None) -> dict[ChunkKey, set[str]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"标注集路径不存在: {path}")

    if path.is_dir():
        files = sorted(path.glob("*.json"))
        return _load_predictions_from_classified(files)

    if path.suffix.lower() == ".jsonl":
        result: dict[ChunkKey, set[str]] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            source_file = str(record.get("source_file", ""))
            chunk_id = record.get("chunk_id")
            if not source_file or chunk_id is None:
                raise ValueError(f"标注记录缺少 source_file/chunk_id: {line}")
            risks = record.get("risk_type", record.get("risks", []))
            result[_normalize_chunk_key(source_file, chunk_id)] = _normalize_risk_values(risks)
        return result

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            source_file = str(payload.get("source_file", path.name))
            chunks = payload.get("chunks", [])
            result: dict[ChunkKey, set[str]] = {}
            if isinstance(chunks, list):
                for chunk in chunks:
                    if not isinstance(chunk, dict) or chunk.get("chunk_id") is None:
                        continue
                    result[_normalize_chunk_key(source_file, chunk["chunk_id"])] = _normalize_risk_values(
                        chunk.get("risk_type")
                    )
            return result
        if isinstance(payload, list):
            result = {}
            for record in payload:
                if not isinstance(record, dict):
                    continue
                source_file = str(record.get("source_file", ""))
                chunk_id = record.get("chunk_id")
                if not source_file or chunk_id is None:
                    raise ValueError(f"标注记录缺少 source_file/chunk_id: {record}")
                risks = record.get("risk_type", record.get("risks", []))
                result[_normalize_chunk_key(source_file, chunk_id)] = _normalize_risk_values(risks)
            return result

    raise ValueError(f"暂不支持的标注文件格式: {path}")


def _compute_prf1(
    predictions: dict[ChunkKey, set[str]],
    ground_truth: dict[ChunkKey, set[str]],
) -> tuple[float | None, float | None, float | None, int]:
    if not ground_truth:
        return None, None, None, 0

    tp = 0
    fp = 0
    fn = 0
    label_support = 0
    keys = set(predictions.keys()) | set(ground_truth.keys())
    for key in keys:
        predicted = predictions.get(key, set())
        expected = ground_truth.get(key, set())
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        label_support += len(expected)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, label_support
