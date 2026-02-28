"""实验产物写入器。"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.classification.classification_pipeline import ChunkClassificationDiagnostic
from src.tools.logger import Logger

from .metrics import ExperimentMetrics


@dataclass(frozen=True, slots=True)
class ExperimentLayout:
    """实验目录布局。"""

    output_root: Path
    run_dir: Path
    chunks_dir: Path
    classified_dir: Path
    audit_result_dir: Path
    final_report_dir: Path
    metrics_dir: Path


class ArtifactWriter:
    """统一落盘实验产物。"""

    def __init__(self, output_root: Path, logger: Logger) -> None:
        self._output_root = output_root
        self._logger = logger

    def prepare_layout(self, run_id: str) -> ExperimentLayout:
        """初始化实验目录结构。"""

        run_dir = self._output_root / run_id
        # 每个 run 使用独立目录，避免跨实验产物互相覆盖。
        layout = ExperimentLayout(
            output_root=self._output_root,
            run_dir=run_dir,
            chunks_dir=run_dir / "chunks",
            classified_dir=run_dir / "classified",
            audit_result_dir=run_dir / "audit_result",
            final_report_dir=run_dir / "final_report",
            metrics_dir=run_dir / "metrics",
        )
        for path in (
            layout.output_root,
            layout.run_dir,
            layout.chunks_dir,
            layout.classified_dir,
            layout.audit_result_dir,
            layout.final_report_dir,
            layout.metrics_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._logger.info(f"实验目录准备完成: run_dir={layout.run_dir}")
        return layout

    def write_diagnostics(
        self,
        diagnostics: list[ChunkClassificationDiagnostic],
        layout: ExperimentLayout,
    ) -> Path:
        """写入 chunk 级诊断明细。"""

        output = layout.metrics_dir / "llm_trace.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for item in diagnostics:
                handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        self._logger.info(f"诊断明细写入完成: {output}")
        return output

    def write_metrics(
        self,
        metrics: ExperimentMetrics,
        layout: ExperimentLayout,
    ) -> Path:
        """写入聚合指标。"""

        output = layout.metrics_dir / "metrics.json"
        output.write_text(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self._logger.info(f"指标写入完成: {output}")
        return output

    def write_audit_result(
        self,
        diagnostics: list[ChunkClassificationDiagnostic],
        layout: ExperimentLayout,
    ) -> Path:
        """写入统一审计结果产物。"""

        by_file: dict[str, dict[str, Any]] = {}
        for item in diagnostics:
            stat = by_file.setdefault(
                item.source_file,
                {"chunk_count": 0, "schema_invalid_count": 0, "llm_error_count": 0, "risk_frequency": {}},
            )
            stat["chunk_count"] += 1
            if not item.schema_valid:
                stat["schema_invalid_count"] += 1
            if item.error_code is not None:
                stat["llm_error_count"] += 1
            for risk in item.final_risks:
                stat["risk_frequency"][risk] = int(stat["risk_frequency"].get(risk, 0)) + 1

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "files": by_file,
        }
        output = layout.audit_result_dir / "audit_result.json"
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._logger.info(f"audit_result 写入完成: {output}")
        return output

    def write_final_report(
        self,
        run_summary: dict[str, Any],
        metrics: ExperimentMetrics,
        layout: ExperimentLayout,
    ) -> Path:
        """写入统一最终报告。"""

        lines = [
            "# Experiment Final Report",
            "",
            f"- run_id: {run_summary.get('run_id', '')}",
            f"- timestamp: {run_summary.get('timestamp', '')}",
            f"- mode: {run_summary.get('mode', '')}",
            f"- model: {run_summary.get('model', '')}",
            f"- chunk_size: {run_summary.get('chunk_size', '')}",
            f"- file_count: {run_summary.get('file_count', '')}",
            f"- chunk_count: {metrics.chunk_count}",
            "",
            "## Metrics",
            "",
            f"- precision: {metrics.precision}",
            f"- recall: {metrics.recall}",
            f"- f1: {metrics.f1}",
            f"- schema_valid_rate: {metrics.schema_valid_rate:.6f}",
            f"- avg_token_in: {metrics.avg_token_in:.6f}",
            f"- avg_token_out: {metrics.avg_token_out:.6f}",
            f"- avg_total_token: {metrics.avg_total_token:.6f}",
            f"- avg_token: {metrics.avg_token:.6f}",
            f"- avg_reasoning_token: {metrics.avg_reasoning_token:.6f}",
            f"- avg_cached_token: {metrics.avg_cached_token:.6f}",
            f"- avg_latency_ms: {metrics.avg_latency_ms:.6f}",
            f"- reasoning_token_ratio: {metrics.reasoning_token_ratio:.6f}",
            f"- cached_token_ratio: {metrics.cached_token_ratio:.6f}",
            f"- conflict_rate: {metrics.conflict_rate:.6f}",
            f"- cache_hit_rate: {metrics.cache_hit_rate:.6f}",
            f"- llm_error_rate: {metrics.llm_error_rate:.6f}",
            f"- total_tokens_estimated_rate: {metrics.total_tokens_estimated_rate:.6f}",
        ]
        output = layout.final_report_dir / "final_report.md"
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._logger.info(f"final_report 写入完成: {output}")
        return output

    def append_results(
        self,
        row: dict[str, Any],
        fieldnames: list[str],
    ) -> tuple[Path, Path]:
        """追加实验汇总行到 CSV / JSONL。"""

        csv_path = self._output_root / "results.csv"
        jsonl_path = self._output_root / "results.jsonl"
        # CSV 作为论文/表格分析输入，JSONL 作为结构化追溯输入。
        normalized = {field: _normalize_cell(row.get(field)) for field in fieldnames}

        _ensure_csv_schema(csv_path, fieldnames)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(normalized)

        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        self._logger.info(f"results 追加完成: csv={csv_path}, jsonl={jsonl_path}")
        return csv_path, jsonl_path


def _normalize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _ensure_csv_schema(csv_path: Path, fieldnames: list[str]) -> None:
    """确保 results.csv 表头与最新字段一致。"""

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        existing_header = next(reader, [])

    if existing_header == fieldnames:
        return

    migrated_rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            migrated_rows.append({field: row.get(field, "") for field in fieldnames})

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in migrated_rows:
            writer.writerow(row)
