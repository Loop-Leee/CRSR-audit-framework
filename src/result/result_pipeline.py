"""result 模块：将 review 结果以批注形式写入 5-result Word 副本。"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from src.tools.docx_comments import DocxComment, add_comments_to_docx, ensure_docx_comment_support
from src.tools.logger import Logger
from src.tools.word_io import (
    build_docx_converter_install_hint,
    convert_word_to_docx,
    extract_word_text,
    resolve_docx_converter,
)


@dataclass(frozen=True, slots=True)
class ResultItemDiagnostic:
    """单条 review_item 的处理诊断。"""

    review_file: str
    source_file: str
    risk_id: str
    chunk_id: int | str | None
    result: str
    risk_type: str
    span: str
    span_start: int | None
    span_end: int | None
    locate_strategy: str | None
    status: str
    comment_written: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class ResultRunMetrics:
    """result 运行级统计指标。"""

    file_count: int
    source_file_count: int
    review_item_count: int
    target_item_count: int
    located_item_count: int
    planned_item_count: int
    applied_item_count: int
    duplicate_item_count: int
    locate_failed_count: int
    source_missing_count: int
    skipped_result_count: int
    ablation_no_writeback_count: int
    write_success_file_count: int
    write_failed_file_count: int
    converted_doc_count: int
    span_locate_rate: float
    planned_rate: float
    apply_rate: float
    file_write_success_rate: float
    ablation_no_writeback: bool
    ablation_no_chunk_offset: bool
    target_results: list[str]

    def to_dict(self) -> dict[str, object]:
        """导出指标字典。"""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResultRunResult:
    """result 执行结果。"""

    outputs: list[Path]
    diagnostics: list[ResultItemDiagnostic]
    metrics: ResultRunMetrics
    trace_path: Path
    metrics_path: Path


@dataclass(frozen=True, slots=True)
class _ChunkRange:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _SpanLocation:
    start: int
    end: int
    strategy: str


@dataclass(frozen=True, slots=True)
class _PendingComment:
    risk_id: str
    chunk_id: int | str | None
    result: str
    risk_type: str
    span: str
    span_start: int
    span_end: int
    locate_strategy: str
    comment_text: str
    raw_item: dict[str, object]


def discover_review_files(input_path: Path) -> list[Path]:
    """发现待处理 review 文件。

    Args:
        input_path: `*.review.json` 文件或目录。

    Returns:
        list[Path]: 待处理文件列表。

    Raises:
        FileNotFoundError: 输入路径不存在时抛出。
    """

    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    files = sorted(input_path.glob("*.review.json"))
    if files:
        return files
    return sorted(input_path.glob("*.json"))


def run_result(
    *,
    input_path: Path,
    output_dir: Path,
    logger: Logger,
    target_results: list[str] | None = None,
    backup_enabled: bool = True,
    ablation_no_writeback: bool = False,
    ablation_no_chunk_offset: bool = False,
) -> list[Path]:
    """执行 result 主流程（兼容接口，仅返回输出路径列表）。"""

    return run_result_with_diagnostics(
        input_path=input_path,
        output_dir=output_dir,
        logger=logger,
        target_results=target_results,
        backup_enabled=backup_enabled,
        ablation_no_writeback=ablation_no_writeback,
        ablation_no_chunk_offset=ablation_no_chunk_offset,
    ).outputs


def run_result_with_diagnostics(
    *,
    input_path: Path,
    output_dir: Path,
    logger: Logger,
    target_results: list[str] | None = None,
    backup_enabled: bool = True,
    ablation_no_writeback: bool = False,
    ablation_no_chunk_offset: bool = False,
) -> ResultRunResult:
    """执行 result 注释写回并输出诊断与指标。

    Args:
        input_path: review 输入文件/目录。
        output_dir: result 输出目录（trace/metrics/preview/annotated_docs）。
        logger: 模块日志对象。
        target_results: 需要注释的 result 标签集合。
        backup_enabled: 是否保留源文件副本（用于 .doc 源文件先复制后转换）。
        ablation_no_writeback: 消融开关，开启时不写回源文件，仅写 preview。
        ablation_no_chunk_offset: 消融开关，开启时跳过 `span_offset` 直接检索。

    Returns:
        ResultRunResult: 输出路径、任务级诊断和聚合指标。

    Raises:
        FileNotFoundError: 输入目录无文件时抛出。
        RuntimeError: 存在失败文件时抛出。
    """

    effective_target_results = _normalize_target_results(target_results)
    files = discover_review_files(input_path)
    if not files:
        raise FileNotFoundError(f"未找到待处理 review 文件: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not ablation_no_writeback:
        ensure_docx_comment_support()
        _preflight_writeback_dependencies(
            files=files,
            logger=logger,
            target_results=effective_target_results,
        )

    logger.info(
        "result_start: file_count=%s, input=%s, output=%s, target_results=%s, backup_enabled=%s, "
        "ablation_no_writeback=%s, ablation_no_chunk_offset=%s"
        % (
            len(files),
            input_path,
            output_dir,
            effective_target_results,
            backup_enabled,
            ablation_no_writeback,
            ablation_no_chunk_offset,
        )
    )

    outputs: list[Path] = []
    diagnostics: list[ResultItemDiagnostic] = []
    failures: list[str] = []
    unique_source_files: set[str] = set()
    used_target_doc_names: set[str] = set()

    review_item_count = 0
    target_item_count = 0
    located_item_count = 0
    planned_item_count = 0
    applied_item_count = 0
    duplicate_item_count = 0
    locate_failed_count = 0
    source_missing_count = 0
    skipped_result_count = 0
    ablation_no_writeback_count = 0
    write_success_file_count = 0
    write_failed_file_count = 0
    converted_doc_count = 0

    for review_file in files:
        try:
            logger.info(f"result_file_start: file={review_file}")
            payload = json.loads(review_file.read_text(encoding="utf-8"))
            review_items = payload.get("review_items", [])
            if not isinstance(review_items, list):
                raise ValueError(f"review_items 字段缺失或类型错误: {review_file}")
            review_item_count += len(review_items)

            source_file_value = _resolve_source_file(payload=payload, review_items=review_items)
            source_path = Path(source_file_value) if source_file_value else None
            if source_path is not None:
                unique_source_files.add(str(source_path))
            chunk_ranges = _build_chunk_ranges(payload.get("chunks", []))

            pending_comments: list[_PendingComment] = []
            pending_diagnostics: list[ResultItemDiagnostic] = []

            for index, raw_item in enumerate(review_items, start=1):
                if not isinstance(raw_item, dict):
                    diagnostics.append(
                        ResultItemDiagnostic(
                            review_file=str(review_file),
                            source_file=str(source_path) if source_path else "",
                            risk_id=f"{review_file.stem}#invalid#{index}",
                            chunk_id=None,
                            result="",
                            risk_type="",
                            span="",
                            span_start=None,
                            span_end=None,
                            locate_strategy=None,
                            status="invalid_item",
                            comment_written=False,
                            error=f"review_items[{index}] 不是对象",
                        )
                    )
                    continue

                result_label = str(raw_item.get("result", "")).strip()
                risk_id = _resolve_risk_id(raw_item, review_file=review_file, index=index)
                risk_type = str(raw_item.get("risk_type", "")).strip()
                span = str(raw_item.get("span", "")).strip()
                chunk_id = raw_item.get("chunk_id")

                if result_label not in effective_target_results:
                    skipped_result_count += 1
                    diagnostics.append(
                        ResultItemDiagnostic(
                            review_file=str(review_file),
                            source_file=str(source_path) if source_path else "",
                            risk_id=risk_id,
                            chunk_id=chunk_id,
                            result=result_label,
                            risk_type=risk_type,
                            span=span,
                            span_start=None,
                            span_end=None,
                            locate_strategy=None,
                            status="skipped_result",
                            comment_written=False,
                            error=None,
                        )
                    )
                    continue

                target_item_count += 1
                if source_path is None or not source_path.exists():
                    source_missing_count += 1
                    diagnostics.append(
                        ResultItemDiagnostic(
                            review_file=str(review_file),
                            source_file=str(source_path) if source_path else "",
                            risk_id=risk_id,
                            chunk_id=chunk_id,
                            result=result_label,
                            risk_type=risk_type,
                            span=span,
                            span_start=None,
                            span_end=None,
                            locate_strategy=None,
                            status="source_file_missing",
                            comment_written=False,
                            error=f"源文件不存在: {source_path}" if source_path else "源文件路径缺失",
                        )
                    )
                    continue

                if not span:
                    locate_failed_count += 1
                    diagnostics.append(
                        ResultItemDiagnostic(
                            review_file=str(review_file),
                            source_file=str(source_path),
                            risk_id=risk_id,
                            chunk_id=chunk_id,
                            result=result_label,
                            risk_type=risk_type,
                            span=span,
                            span_start=None,
                            span_end=None,
                            locate_strategy=None,
                            status="span_empty",
                            comment_written=False,
                            error="span 为空",
                        )
                    )
                    continue

                pending_diagnostics.append(
                    ResultItemDiagnostic(
                        review_file=str(review_file),
                        source_file=str(source_path),
                        risk_id=risk_id,
                        chunk_id=chunk_id,
                        result=result_label,
                        risk_type=risk_type,
                        span=span,
                        span_start=None,
                        span_end=None,
                        locate_strategy=None,
                        status="pending",
                        comment_written=False,
                        error=None,
                    )
                )

                pending_comments.append(
                    _PendingComment(
                        risk_id=risk_id,
                        chunk_id=chunk_id,
                        result=result_label,
                        risk_type=risk_type,
                        span=span,
                        span_start=-1,
                        span_end=-1,
                        locate_strategy="",
                        comment_text=_build_comment_text(
                            risk_type=risk_type,
                            result=result_label,
                            rule_hit=str(raw_item.get("rule_hit", "")).strip(),
                            suggest=str(raw_item.get("suggest", "")).strip(),
                        ),
                        raw_item=raw_item,
                    )
                )

            if not pending_comments:
                logger.info(f"result_file_skip: file={review_file}, reason=no_target_items")
                continue

            assert source_path is not None
            target_docx_path: Path | None = None
            locate_text_path = source_path
            if not ablation_no_writeback:
                target_docx_path, converted = _prepare_target_docx(
                    source_path=source_path,
                    output_dir=output_dir,
                    backup_enabled=backup_enabled,
                    used_target_doc_names=used_target_doc_names,
                    logger=logger,
                )
                converted_doc_count += converted
                locate_text_path = target_docx_path

            source_text = extract_word_text(path=locate_text_path, logger=logger)
            planned_keys: set[str] = set()
            locatable_comments: list[_PendingComment] = []
            for item_diag, pending in zip(pending_diagnostics, pending_comments):
                location = _locate_span(
                    span=pending.span,
                    chunk_id=pending.chunk_id,
                    raw_item=pending.raw_item,
                    chunk_ranges=chunk_ranges,
                    source_text=source_text,
                    ablation_no_chunk_offset=ablation_no_chunk_offset,
                )
                if location is None:
                    locate_failed_count += 1
                    diagnostics.append(
                        ResultItemDiagnostic(
                            review_file=item_diag.review_file,
                            source_file=item_diag.source_file,
                            risk_id=item_diag.risk_id,
                            chunk_id=item_diag.chunk_id,
                            result=item_diag.result,
                            risk_type=item_diag.risk_type,
                            span=item_diag.span,
                            span_start=None,
                            span_end=None,
                            locate_strategy=None,
                            status="span_not_found",
                            comment_written=False,
                            error="未定位到 span",
                        )
                    )
                    continue

                if pending.risk_id in planned_keys:
                    duplicate_item_count += 1
                    diagnostics.append(
                        ResultItemDiagnostic(
                            review_file=item_diag.review_file,
                            source_file=item_diag.source_file,
                            risk_id=item_diag.risk_id,
                            chunk_id=item_diag.chunk_id,
                            result=item_diag.result,
                            risk_type=item_diag.risk_type,
                            span=item_diag.span,
                            span_start=location.start,
                            span_end=location.end,
                            locate_strategy=location.strategy,
                            status="duplicate_comment",
                            comment_written=False,
                            error=None,
                        )
                    )
                    continue

                located_item_count += 1
                planned_item_count += 1
                planned_keys.add(pending.risk_id)
                locatable_comments.append(
                    _PendingComment(
                        risk_id=pending.risk_id,
                        chunk_id=pending.chunk_id,
                        result=pending.result,
                        risk_type=pending.risk_type,
                        span=pending.span,
                        span_start=location.start,
                        span_end=location.end,
                        locate_strategy=location.strategy,
                        comment_text=pending.comment_text,
                        raw_item=pending.raw_item,
                    )
                )

            if not locatable_comments:
                logger.info(f"result_file_skip: file={review_file}, reason=no_locatable_items")
                continue

            if ablation_no_writeback:
                updated_text = _insert_comments(source_text=source_text, comments=locatable_comments)
                preview_path = _build_preview_path(
                    output_dir=output_dir,
                    source_path=source_path,
                    review_file=review_file,
                )
                preview_path.write_text(updated_text, encoding="utf-8")
                _append_output_once(outputs, preview_path)
                ablation_no_writeback_count += len(locatable_comments)
                for pending in locatable_comments:
                    diagnostics.append(
                        ResultItemDiagnostic(
                            review_file=str(review_file),
                            source_file=str(source_path),
                            risk_id=pending.risk_id,
                            chunk_id=pending.chunk_id,
                            result=pending.result,
                            risk_type=pending.risk_type,
                            span=pending.span,
                            span_start=pending.span_start,
                            span_end=pending.span_end,
                            locate_strategy=pending.locate_strategy,
                            status="ablation_no_writeback",
                            comment_written=False,
                            error=None,
                        )
                    )
                logger.info(
                    "result_file_done: file=%s, locatable_count=%s, mode=preview, output=%s"
                    % (review_file.name, len(locatable_comments), preview_path)
                )
                continue

            try:
                if target_docx_path is None:
                    raise RuntimeError("target_docx_path 未初始化。")

                write_result = add_comments_to_docx(
                    docx_path=target_docx_path,
                    comments=[
                        DocxComment(
                            risk_id=pending.risk_id,
                            anchor_offset=pending.span_end,
                            span=pending.span,
                            content=pending.comment_text,
                        )
                        for pending in locatable_comments
                    ],
                    logger=logger,
                )
                if write_result.written_count <= 0:
                    raise RuntimeError("未写入任何批注。")
                written_text = extract_word_text(path=target_docx_path, logger=None)
                if _normalize_compare_text(source_text) != _normalize_compare_text(written_text):
                    raise RuntimeError("批注写入后正文文本发生变化。")

                _append_output_once(outputs, target_docx_path)
                write_success_file_count += 1
                for pending, written in zip(locatable_comments, write_result.written_mask):
                    status = "annotated" if written else "comment_anchor_not_found"
                    if written:
                        applied_item_count += 1
                    diagnostics.append(
                        ResultItemDiagnostic(
                            review_file=str(review_file),
                            source_file=str(source_path),
                            risk_id=pending.risk_id,
                            chunk_id=pending.chunk_id,
                            result=pending.result,
                            risk_type=pending.risk_type,
                            span=pending.span,
                            span_start=pending.span_start,
                            span_end=pending.span_end,
                            locate_strategy=pending.locate_strategy,
                            status=status,
                            comment_written=written,
                            error=None if written else "批注锚点定位失败",
                        )
                    )
                logger.info(
                    "result_file_done: file=%s, locatable_count=%s, output=%s"
                    % (review_file.name, len(locatable_comments), target_docx_path)
                )
            except Exception as exc:  # noqa: BLE001
                write_failed_file_count += 1
                failure_message = (
                    f"result_file_failed: file={review_file}, source={source_path}, target={target_docx_path}, error={exc}"
                )
                logger.error(failure_message)
                failures.append(failure_message)
                for pending in locatable_comments:
                    diagnostics.append(
                        ResultItemDiagnostic(
                            review_file=str(review_file),
                            source_file=str(source_path),
                            risk_id=pending.risk_id,
                            chunk_id=pending.chunk_id,
                            result=pending.result,
                            risk_type=pending.risk_type,
                            span=pending.span,
                            span_start=pending.span_start,
                            span_end=pending.span_end,
                            locate_strategy=pending.locate_strategy,
                            status="write_failed",
                            comment_written=False,
                            error=str(exc),
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            message = f"result_review_file_failed: file={review_file}, error={exc}"
            logger.error(message)
            failures.append(message)

    trace_path = output_dir / "result_trace.jsonl"
    _write_diagnostics(trace_path, diagnostics)

    metrics = _compute_result_metrics(
        file_count=len(files),
        source_file_count=len(unique_source_files),
        review_item_count=review_item_count,
        target_item_count=target_item_count,
        located_item_count=located_item_count,
        planned_item_count=planned_item_count,
        applied_item_count=applied_item_count,
        duplicate_item_count=duplicate_item_count,
        locate_failed_count=locate_failed_count,
        source_missing_count=source_missing_count,
        skipped_result_count=skipped_result_count,
        ablation_no_writeback_count=ablation_no_writeback_count,
        write_success_file_count=write_success_file_count,
        write_failed_file_count=write_failed_file_count,
        converted_doc_count=converted_doc_count,
        ablation_no_writeback=ablation_no_writeback,
        ablation_no_chunk_offset=ablation_no_chunk_offset,
        target_results=effective_target_results,
    )
    metrics_path = output_dir / "result_metrics.json"
    metrics_path.write_text(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"result_trace_written: {trace_path}")
    logger.info(f"result_metrics_written: {metrics_path}")

    if failures:
        raise RuntimeError(_build_failure_summary(failures=failures, log_path=logger.path))

    logger.info(
        "result_done: file_count=%s, target_item_count=%s, applied_item_count=%s"
        % (len(files), target_item_count, applied_item_count)
    )
    return ResultRunResult(
        outputs=outputs,
        diagnostics=diagnostics,
        metrics=metrics,
        trace_path=trace_path,
        metrics_path=metrics_path,
    )


def _resolve_source_file(payload: dict[str, object], review_items: list[object]) -> str:
    source_file = str(payload.get("source_file", "")).strip()
    if source_file:
        return source_file
    for item in review_items:
        if isinstance(item, dict):
            value = str(item.get("source_file", "")).strip()
            if value:
                return value
    return ""


def _preflight_writeback_dependencies(
    *,
    files: list[Path],
    logger: Logger,
    target_results: list[str],
) -> None:
    """写回前置校验：若存在 `.doc` 源文件，必须可用 LibreOffice 转换器。"""

    target_set = set(target_results)
    doc_sources: list[Path] = []
    for review_file in files:
        try:
            payload = json.loads(review_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.info(f"result_preflight_skip_invalid_json: file={review_file}, error={exc}")
            continue

        raw_items = payload.get("review_items", [])
        review_items = raw_items if isinstance(raw_items, list) else []
        has_target_item = any(
            isinstance(item, dict) and str(item.get("result", "")).strip() in target_set
            for item in review_items
        )
        if not has_target_item:
            continue

        source_file = _resolve_source_file(payload=payload, review_items=review_items)
        if not source_file:
            continue
        source_path = Path(source_file)
        if source_path.suffix.lower() == ".doc":
            doc_sources.append(source_path)

    if not doc_sources:
        return

    converter = resolve_docx_converter()
    if converter is None:
        sample_names = ", ".join(path.name for path in doc_sources[:3])
        remaining = max(len(doc_sources) - 3, 0)
        suffix = f"（另有 {remaining} 个）" if remaining else ""
        raise RuntimeError(
            "检测到 .doc 源文件，但未找到可用的 .doc -> .docx 转换器。"
            "请安装 LibreOffice（soffice/libreoffice/lowriter）。"
            f"样例文件: {sample_names}{suffix}"
            f"{build_docx_converter_install_hint()}"
        )

    logger.info(
        "result_preflight_doc_converter_ready: converter=%s, doc_source_count=%s"
        % (converter, len(doc_sources))
    )


def _normalize_target_results(target_results: list[str] | None) -> list[str]:
    values = target_results or ["不合格", "待复核"]
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized or ["不合格", "待复核"]


def _build_failure_summary(*, failures: list[str], log_path: Path) -> str:
    """构建运行失败摘要，避免仅返回泛化错误。"""

    sample = failures[:2]
    sample_text = " | ".join(sample)
    if len(failures) > 2:
        sample_text = f"{sample_text} | 其余 {len(failures) - 2} 条见日志"
    return f"result 流程存在失败文件({len(failures)}): {sample_text}; 日志: {log_path}"


def _resolve_risk_id(raw_item: dict[str, object], *, review_file: Path, index: int) -> str:
    risk_id = str(raw_item.get("risk_id", "")).strip()
    if risk_id:
        return risk_id
    return f"{review_file.stem}#item#{index}"


def _build_chunk_ranges(raw_chunks: object) -> dict[str, _ChunkRange]:
    if not isinstance(raw_chunks, list):
        return {}

    ranges: dict[str, _ChunkRange] = {}
    for index, raw_chunk in enumerate(raw_chunks, start=1):
        if not isinstance(raw_chunk, dict):
            continue
        chunk_id = str(raw_chunk.get("chunk_id", index)).strip()
        start = _safe_int(raw_chunk.get("start_offset"))
        end = _safe_int(raw_chunk.get("end_offset"))
        if not chunk_id or start is None or end is None:
            continue
        if start < 0 or end < start:
            continue
        ranges[chunk_id] = _ChunkRange(start=start, end=end)
    return ranges


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _locate_span(
    *,
    span: str,
    chunk_id: int | str | None,
    raw_item: dict[str, object],
    chunk_ranges: dict[str, _ChunkRange],
    source_text: str,
    ablation_no_chunk_offset: bool,
) -> _SpanLocation | None:
    if not span:
        return None

    chunk_key = str(chunk_id).strip() if chunk_id is not None else ""
    chunk = chunk_ranges.get(chunk_key)
    if chunk is not None and not ablation_no_chunk_offset:
        span_offset = raw_item.get("span_offset")
        parsed_offset = _parse_span_offset(span_offset)
        if parsed_offset is not None:
            anchor = chunk.start + parsed_offset[0]
            near_match = _find_exact_near_anchor(
                text=source_text,
                query=span,
                anchor=anchor,
                window_radius=256,
                chunk_start=chunk.start,
                chunk_end=chunk.end,
            )
            if near_match is not None:
                return _SpanLocation(
                    start=near_match[0],
                    end=near_match[1],
                    strategy="span_offset_near_exact",
                )

    if chunk is not None:
        exact_in_chunk = _find_exact_in_window(source_text, span, chunk.start, chunk.end)
        if exact_in_chunk is not None:
            return _SpanLocation(start=exact_in_chunk[0], end=exact_in_chunk[1], strategy="chunk_exact")

    return None


def _parse_span_offset(raw_span_offset: object) -> tuple[int, int] | None:
    if not isinstance(raw_span_offset, list) or len(raw_span_offset) != 2:
        return None
    start = _safe_int(raw_span_offset[0])
    end = _safe_int(raw_span_offset[1])
    if start is None or end is None or start < 0 or end < start:
        return None
    return (start, end)


def _is_valid_range(start: int, end: int, total_length: int) -> bool:
    return 0 <= start <= end <= total_length


def _find_exact_in_window(text: str, query: str, start: int, end: int) -> tuple[int, int] | None:
    if not _is_valid_range(start, end, len(text)):
        return None
    window = text[start:end]
    local = window.find(query)
    if local < 0:
        return None
    return (start + local, start + local + len(query))


def _find_exact_near_anchor(
    *,
    text: str,
    query: str,
    anchor: int,
    window_radius: int,
    chunk_start: int,
    chunk_end: int,
) -> tuple[int, int] | None:
    """基于 span_offset 锚点做近邻精确匹配。"""

    if not query:
        return None
    left = max(chunk_start, anchor - window_radius)
    right = min(chunk_end, anchor + window_radius + len(query))
    if not _is_valid_range(left, right, len(text)):
        return None
    window = text[left:right]
    if not window:
        return None

    matches: list[tuple[int, int, int]] = []
    offset = 0
    while True:
        local_index = window.find(query, offset)
        if local_index < 0:
            break
        start = left + local_index
        end = start + len(query)
        distance = abs(start - anchor)
        matches.append((distance, start, end))
        offset = local_index + 1

    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1]))
    _, best_start, best_end = matches[0]
    return (best_start, best_end)


def _build_comment_text(
    *,
    risk_type: str,
    result: str,
    rule_hit: str,
    suggest: str,
) -> str:
    """构建批注文本。"""

    return (
        f"风险类型：{_to_single_line(risk_type)}\n"
        f"审查结论：{_to_single_line(result)}\n"
        f"违反的条款：{_to_single_line(rule_hit)}\n"
        f"修改建议: {_to_single_line(suggest)}"
    )


def _to_single_line(value: str) -> str:
    return " ".join(value.split()) if value else ""


def _insert_comments(*, source_text: str, comments: list[_PendingComment]) -> str:
    updated = source_text
    ordered = sorted(comments, key=lambda item: (item.span_end, item.risk_id), reverse=True)
    for pending in ordered:
        preview_block = _build_preview_block(pending.comment_text)
        updated = updated[: pending.span_end] + preview_block + updated[pending.span_end :]
    return updated


def _build_preview_block(comment_text: str) -> str:
    return f"\n【审查注释】\n{comment_text}\n【/审查注释】\n"


def _build_preview_path(*, output_dir: Path, source_path: Path, review_file: Path) -> Path:
    digest = hashlib.sha1(f"{source_path}::{review_file.name}".encode("utf-8")).hexdigest()[:8]
    return output_dir / f"{source_path.stem}_{review_file.stem}_{digest}.annotated.preview.txt"


def _prepare_target_docx(
    *,
    source_path: Path,
    output_dir: Path,
    backup_enabled: bool,
    used_target_doc_names: set[str],
    logger: Logger,
) -> tuple[Path, int]:
    """准备输出目录中的 docx 副本并返回可批注路径。"""

    source_suffix = source_path.suffix.lower()
    if source_suffix not in {".doc", ".docx"}:
        raise ValueError(f"暂不支持的源文件类型: {source_path}")

    annotated_dir = output_dir / "annotated_docs"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    target_docx = _allocate_result_docx_path(
        source_stem=source_path.stem,
        annotated_dir=annotated_dir,
        used_target_doc_names=used_target_doc_names,
    )

    converted_doc_count = 0
    if source_suffix == ".docx":
        shutil.copy2(source_path, target_docx)
        logger.info(f"源文件副本创建完成: source={source_path}, target={target_docx}")
        return (target_docx, converted_doc_count)

    source_copy = source_path
    if backup_enabled:
        copy_dir = output_dir / "source_copies"
        copy_dir.mkdir(parents=True, exist_ok=True)
        source_copy = copy_dir / f"{target_docx.stem}{source_path.suffix.lower()}"
        shutil.copy2(source_path, source_copy)
        logger.info(f"源文件副本创建完成: source={source_path}, target={source_copy}")

    convert_word_to_docx(
        source_path=source_copy,
        target_path=target_docx,
        logger=logger,
    )
    converted_doc_count += 1
    return (target_docx, converted_doc_count)


def _allocate_result_docx_path(
    *,
    source_stem: str,
    annotated_dir: Path,
    used_target_doc_names: set[str],
) -> Path:
    """为 result 输出分配不冲突的 `.result.docx` 文件名。"""

    base = f"{source_stem}.result.docx"
    if base not in used_target_doc_names:
        used_target_doc_names.add(base)
        return annotated_dir / base

    index = 2
    while True:
        candidate = f"{source_stem}.result.{index}.docx"
        if candidate not in used_target_doc_names:
            used_target_doc_names.add(candidate)
            return annotated_dir / candidate
        index += 1


def _append_output_once(outputs: list[Path], path: Path) -> None:
    if path not in outputs:
        outputs.append(path)


def _write_diagnostics(path: Path, diagnostics: list[ResultItemDiagnostic]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in diagnostics:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def _compute_result_metrics(
    *,
    file_count: int,
    source_file_count: int,
    review_item_count: int,
    target_item_count: int,
    located_item_count: int,
    planned_item_count: int,
    applied_item_count: int,
    duplicate_item_count: int,
    locate_failed_count: int,
    source_missing_count: int,
    skipped_result_count: int,
    ablation_no_writeback_count: int,
    write_success_file_count: int,
    write_failed_file_count: int,
    converted_doc_count: int,
    ablation_no_writeback: bool,
    ablation_no_chunk_offset: bool,
    target_results: list[str],
) -> ResultRunMetrics:
    total_write_attempt_file_count = write_success_file_count + write_failed_file_count
    return ResultRunMetrics(
        file_count=file_count,
        source_file_count=source_file_count,
        review_item_count=review_item_count,
        target_item_count=target_item_count,
        located_item_count=located_item_count,
        planned_item_count=planned_item_count,
        applied_item_count=applied_item_count,
        duplicate_item_count=duplicate_item_count,
        locate_failed_count=locate_failed_count,
        source_missing_count=source_missing_count,
        skipped_result_count=skipped_result_count,
        ablation_no_writeback_count=ablation_no_writeback_count,
        write_success_file_count=write_success_file_count,
        write_failed_file_count=write_failed_file_count,
        converted_doc_count=converted_doc_count,
        span_locate_rate=_safe_rate(located_item_count, target_item_count),
        planned_rate=_safe_rate(planned_item_count, target_item_count),
        apply_rate=_safe_rate(applied_item_count, planned_item_count),
        file_write_success_rate=_safe_rate(write_success_file_count, total_write_attempt_file_count),
        ablation_no_writeback=ablation_no_writeback,
        ablation_no_chunk_offset=ablation_no_chunk_offset,
        target_results=target_results,
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _normalize_compare_text(text: str) -> str:
    return (
        text.replace("\u200b", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )
