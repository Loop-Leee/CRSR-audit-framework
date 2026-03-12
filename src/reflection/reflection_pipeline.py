"""review 后验校正（reflection）流水线。"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.llm import LLMClientError, LLMSettings, OpenAICompatibleClient, run_tasks
from src.tools.logger import Logger

from .reflection_models import (
    ReflectionCallDiagnostic,
    ReflectionFileMetrics,
    ReflectionRunMetrics,
    ReflectionRunResult,
    Stage1Candidate,
    Stage1Decision,
    Stage2Adjustment,
    Stage2Decision,
    Stage2GroupTask,
    Stage2ItemSnapshot,
)
from .reflection_prompt import build_group_reflection_messages, build_item_reflection_messages
from .reflection_utils import (
    build_evidence_window,
    build_fallback_evidence_window,
    build_fp_risk_score,
    clamp_text,
    coerce_span_offset,
    locate_sentence_by_offset,
    normalize_result,
    parse_json_object,
    safe_mean,
    split_sentences_with_offsets,
)


@dataclass(frozen=True, slots=True)
class _Stage1ExecResult:
    item_index: int
    decision: Stage1Decision
    diagnostic: ReflectionCallDiagnostic


@dataclass(frozen=True, slots=True)
class _Stage2ExecResult:
    decision: Stage2Decision
    diagnostic: ReflectionCallDiagnostic


def discover_review_files(input_path: Path) -> list[Path]:
    """发现待处理 review 文件。"""

    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    files = sorted(input_path.glob("*.review.json"))
    if files:
        return files
    return sorted(input_path.glob("*.json"))


def run_reflection(
    *,
    input_path: Path,
    output_dir: Path,
    llm_settings: LLMSettings,
    logger: Logger,
    reflection_version: str = "v1",
    stage1_threshold: int = 4,
    stage2_max_items: int = 6,
    evidence_max_chars: int = 360,
    chunk_excerpt_max_chars: int = 520,
) -> list[Path]:
    """执行 reflection（兼容接口，仅返回输出文件路径）。"""

    return run_reflection_with_diagnostics(
        input_path=input_path,
        output_dir=output_dir,
        llm_settings=llm_settings,
        logger=logger,
        reflection_version=reflection_version,
        stage1_threshold=stage1_threshold,
        stage2_max_items=stage2_max_items,
        evidence_max_chars=evidence_max_chars,
        chunk_excerpt_max_chars=chunk_excerpt_max_chars,
    ).outputs


def run_reflection_with_diagnostics(
    *,
    input_path: Path,
    output_dir: Path,
    llm_settings: LLMSettings,
    logger: Logger,
    reflection_version: str = "v1",
    stage1_threshold: int = 4,
    stage2_max_items: int = 6,
    evidence_max_chars: int = 360,
    chunk_excerpt_max_chars: int = 520,
) -> ReflectionRunResult:
    """执行 reflection 并输出诊断与指标。"""

    if stage1_threshold < 0:
        raise ValueError("stage1_threshold 不能小于 0。")
    if stage2_max_items <= 0:
        raise ValueError("stage2_max_items 必须大于 0。")

    files = discover_review_files(input_path)
    if not files:
        raise FileNotFoundError(f"未找到待反思文件: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "reflection_start: file_count=%s, input=%s, output=%s, reflection_version=%s, "
        "stage1_threshold=%s, stage2_max_items=%s"
        % (
            len(files),
            input_path,
            output_dir,
            reflection_version,
            stage1_threshold,
            stage2_max_items,
        )
    )

    executor = _ReflectionExecutor(llm_settings=llm_settings, logger=logger)

    outputs: list[Path] = []
    diagnostics: list[ReflectionCallDiagnostic] = []
    failures: list[str] = []

    total_stage1_candidate_count = 0
    total_stage1_called_count = 0
    total_stage2_group_count = 0
    total_stage2_candidate_group_count = 0
    total_stage2_called_group_count = 0
    total_stage1_adjusted_count = 0
    total_stage2_adjusted_count = 0
    total_final_item_count = 0

    for file_path in files:
        try:
            logger.info(f"reflection_file_start: file={file_path}")
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            review_items = payload.get("review_items")
            if not isinstance(review_items, list):
                raise ValueError(f"review_items 字段缺失或类型错误: {file_path}")

            doc_id = _resolve_doc_id(payload=payload, file_path=file_path)
            source_file = _resolve_source_file(payload=payload, file_path=file_path)
            chunk_map = _build_chunk_map(payload.get("chunks", []))

            reflection_items = copy.deepcopy(review_items)
            file_diagnostics: list[ReflectionCallDiagnostic] = []
            context_by_index: dict[int, dict[str, object]] = {}

            for index, item in enumerate(reflection_items):
                if not isinstance(item, dict):
                    continue

                risk_id = str(item.get("risk_id", "")).strip() or f"{doc_id}#reflection#{index + 1}"
                item["risk_id"] = risk_id
                evidence = _build_item_evidence_context(
                    item=item,
                    chunk_map=chunk_map,
                    evidence_max_chars=evidence_max_chars,
                )
                evidence_window = str(evidence["evidence_window"])
                fp_score, fp_flags = build_fp_risk_score(item=item, evidence_window=evidence_window)

                item["fp_risk_score"] = fp_score
                item["fp_risk_flags"] = fp_flags
                item["evidence_window"] = evidence_window
                item["evidence_window_source"] = evidence["evidence_window_source"]

                context_by_index[index] = {
                    "risk_id": risk_id,
                    "chunk_id": item.get("chunk_id"),
                    "risk_type": str(item.get("risk_type", "")).strip(),
                    "evidence_window": evidence_window,
                    "evidence_window_source": evidence["evidence_window_source"],
                    "chunk_excerpt": clamp_text(str(evidence["chunk_excerpt"]), chunk_excerpt_max_chars),
                    "fp_risk_score": fp_score,
                    "fp_risk_flags": fp_flags,
                }

            stage1_candidates = _build_stage1_candidates(
                items=reflection_items,
                contexts=context_by_index,
                source_file=source_file,
                doc_id=doc_id,
                threshold=stage1_threshold,
            )
            stage1_candidate_count = len(stage1_candidates)
            stage1_results = executor.reflect_stage1_many(stage1_candidates)

            stage1_adjusted_count = 0
            stage1_called_count = 0
            for result in stage1_results:
                diagnostics.append(result.diagnostic)
                file_diagnostics.append(result.diagnostic)
                if result.diagnostic.llm_called:
                    stage1_called_count += 1
                item = reflection_items[result.item_index]
                if not isinstance(item, dict):
                    continue
                changed = _apply_stage1_decision(item=item, decision=result.decision)
                if changed:
                    stage1_adjusted_count += 1

            group_map = _group_items_by_risk_type(reflection_items, doc_id=doc_id)
            stage2_group_count = len(group_map)
            candidate_groups = _build_stage2_group_tasks(
                source_file=source_file,
                doc_id=doc_id,
                items=reflection_items,
                contexts=context_by_index,
                group_map=group_map,
                stage2_max_items=stage2_max_items,
            )
            stage2_candidate_group_count = len(candidate_groups)
            stage2_results = executor.reflect_stage2_many(candidate_groups)

            stage2_adjusted_count = 0
            stage2_called_group_count = 0
            for result in stage2_results:
                diagnostics.append(result.diagnostic)
                file_diagnostics.append(result.diagnostic)
                if result.diagnostic.llm_called:
                    stage2_called_group_count += 1

                adjusted = _apply_stage2_decision(
                    decision=result.decision,
                    items=reflection_items,
                    group_indexes=group_map.get(_group_key_to_tuple(result.decision.group_key), []),
                )
                stage2_adjusted_count += adjusted

            llm_rows = [row for row in file_diagnostics if row.llm_called]
            file_metrics = ReflectionFileMetrics(
                generated_at=datetime.now().isoformat(timespec="seconds"),
                reflection_version=reflection_version,
                rule_version=_resolve_rule_version(payload=payload),
                stage1_candidate_count=stage1_candidate_count,
                stage1_called_count=stage1_called_count,
                stage2_group_count=stage2_group_count,
                stage2_candidate_group_count=stage2_candidate_group_count,
                stage2_called_group_count=stage2_called_group_count,
                stage1_adjusted_count=stage1_adjusted_count,
                stage2_adjusted_count=stage2_adjusted_count,
                final_item_count=len(reflection_items),
                avg_token_in=safe_mean([row.token_in for row in llm_rows]),
                avg_token_out=safe_mean([row.token_out for row in llm_rows]),
                avg_total_token=safe_mean([row.total_tokens for row in llm_rows]),
            )

            output_payload = dict(payload)
            output_payload["review_items_original"] = review_items
            output_payload["reflection_meta"] = file_metrics.to_dict()
            output_payload["reflection_items"] = reflection_items

            output_path = _build_output_path(file_path=file_path, output_dir=output_dir)
            output_path.write_text(
                json.dumps(output_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            outputs.append(output_path)

            logger.info(
                "reflection_file_done: file=%s, stage1_candidate_count=%s, stage1_called_count=%s, "
                "stage1_adjusted_count=%s, stage2_group_count=%s, stage2_candidate_group_count=%s, "
                "stage2_called_group_count=%s, stage2_adjusted_count=%s, final_item_count=%s, output=%s"
                % (
                    file_path.name,
                    stage1_candidate_count,
                    stage1_called_count,
                    stage1_adjusted_count,
                    stage2_group_count,
                    stage2_candidate_group_count,
                    stage2_called_group_count,
                    stage2_adjusted_count,
                    len(reflection_items),
                    output_path,
                )
            )

            total_stage1_candidate_count += stage1_candidate_count
            total_stage1_called_count += stage1_called_count
            total_stage2_group_count += stage2_group_count
            total_stage2_candidate_group_count += stage2_candidate_group_count
            total_stage2_called_group_count += stage2_called_group_count
            total_stage1_adjusted_count += stage1_adjusted_count
            total_stage2_adjusted_count += stage2_adjusted_count
            total_final_item_count += len(reflection_items)
        except Exception as exc:  # noqa: BLE001
            message = f"reflection_file_failed: file={file_path}, error={exc}"
            logger.error(message)
            failures.append(message)

    trace_path = output_dir / "reflection_trace.jsonl"
    _write_diagnostics(trace_path, diagnostics)

    llm_rows = [row for row in diagnostics if row.llm_called]
    metrics = ReflectionRunMetrics(
        file_count=len(outputs),
        stage1_candidate_count=total_stage1_candidate_count,
        stage1_called_count=total_stage1_called_count,
        stage2_group_count=total_stage2_group_count,
        stage2_candidate_group_count=total_stage2_candidate_group_count,
        stage2_called_group_count=total_stage2_called_group_count,
        stage1_adjusted_count=total_stage1_adjusted_count,
        stage2_adjusted_count=total_stage2_adjusted_count,
        final_item_count=total_final_item_count,
        llm_called_count=len(llm_rows),
        avg_token_in=safe_mean([row.token_in for row in llm_rows]),
        avg_token_out=safe_mean([row.token_out for row in llm_rows]),
        avg_total_token=safe_mean([row.total_tokens for row in llm_rows]),
        reflection_version=reflection_version,
    )

    metrics_path = output_dir / "reflection_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(f"reflection_trace_written: {trace_path}")
    logger.info(f"reflection_metrics_written: {metrics_path}")

    if failures:
        raise RuntimeError("reflection 流程存在失败文件，请检查日志。")

    logger.info(
        "reflection_done: success_count=%s, stage1_adjusted_count=%s, stage2_adjusted_count=%s"
        % (len(outputs), total_stage1_adjusted_count, total_stage2_adjusted_count)
    )

    return ReflectionRunResult(
        outputs=outputs,
        diagnostics=diagnostics,
        metrics=metrics,
        trace_path=trace_path,
        metrics_path=metrics_path,
    )


class _ReflectionExecutor:
    """reflection 两阶段 LLM 调用执行器。"""

    def __init__(self, *, llm_settings: LLMSettings, logger: Logger) -> None:
        self._settings = llm_settings
        self._logger = logger
        self._enabled = llm_settings.enabled

        if not self._enabled:
            self._client = None
            self._logger.info("reflection llm 已禁用：llm.enabled=false")
            return

        if not llm_settings.model:
            self._enabled = False
            self._client = None
            self._logger.error("reflection llm 降级：缺少 LLM_MODEL，当前任务将不触发 LLM")
            return

        self._client = OpenAICompatibleClient(llm_settings)

    def reflect_stage1_many(self, candidates: list[Stage1Candidate]) -> list[_Stage1ExecResult]:
        """批量执行 Stage-1。"""

        if not candidates:
            return []

        return run_tasks(
            candidates,
            self._reflect_stage1_one,
            concurrent_enabled=self._settings.concurrent_enabled,
            max_concurrency=self._settings.max_concurrency,
        )

    def reflect_stage2_many(self, tasks: list[Stage2GroupTask]) -> list[_Stage2ExecResult]:
        """批量执行 Stage-2。"""

        if not tasks:
            return []

        return run_tasks(
            tasks,
            self._reflect_stage2_one,
            concurrent_enabled=self._settings.concurrent_enabled,
            max_concurrency=self._settings.max_concurrency,
        )

    def _reflect_stage1_one(self, candidate: Stage1Candidate) -> _Stage1ExecResult:
        fallback = Stage1Decision(
            risk_id=candidate.risk_id,
            action="keep",
            evidence_sufficient="partial",
            revised_result=candidate.initial_result,
            revised_span=candidate.span,
            revised_suggest=candidate.suggest,
            reason="反思失败，回退保留原结果",
        )

        if not self._enabled or self._client is None:
            return _Stage1ExecResult(
                item_index=candidate.item_index,
                decision=fallback,
                diagnostic=_build_diagnostic(
                    stage="stage1",
                    source_file=candidate.source_file,
                    doc_id=candidate.doc_id,
                    risk_type=candidate.risk_type,
                    scope_key=candidate.risk_id,
                    input_item_count=1,
                    llm_called=False,
                    schema_valid=True,
                    token_in=0,
                    token_out=0,
                    total_tokens=0,
                    cached_tokens=0,
                    reasoning_tokens=0,
                    total_tokens_estimated=False,
                    latency_ms=0.0,
                    request_id="",
                    retries=0,
                    error_code="llm_disabled",
                    cached=False,
                ),
            )

        include_chunk_excerpt = (
            candidate.evidence_window_source != "sentence_window" or len(candidate.evidence_window) < 60
        )
        messages = build_item_reflection_messages(
            candidate,
            include_chunk_excerpt=include_chunk_excerpt,
        )

        try:
            response = self._client.chat_with_metadata(messages)
        except LLMClientError as exc:
            self._logger.error(
                "reflection_stage1_llm_error: risk_id=%s, retries=%s, error_code=%s, error=%s"
                % (candidate.risk_id, exc.retries, exc.error_code, exc)
            )
            return _Stage1ExecResult(
                item_index=candidate.item_index,
                decision=fallback,
                diagnostic=_build_diagnostic(
                    stage="stage1",
                    source_file=candidate.source_file,
                    doc_id=candidate.doc_id,
                    risk_type=candidate.risk_type,
                    scope_key=candidate.risk_id,
                    input_item_count=1,
                    llm_called=True,
                    schema_valid=False,
                    token_in=0,
                    token_out=0,
                    total_tokens=0,
                    cached_tokens=0,
                    reasoning_tokens=0,
                    total_tokens_estimated=False,
                    latency_ms=0.0,
                    request_id=exc.request_id,
                    retries=exc.retries,
                    error_code=exc.error_code or "llm_error",
                    cached=False,
                ),
            )

        try:
            payload = parse_json_object(response.content)
            decision = _parse_stage1_decision(payload=payload, candidate=candidate)
            return _Stage1ExecResult(
                item_index=candidate.item_index,
                decision=decision,
                diagnostic=_build_diagnostic(
                    stage="stage1",
                    source_file=candidate.source_file,
                    doc_id=candidate.doc_id,
                    risk_type=candidate.risk_type,
                    scope_key=candidate.risk_id,
                    input_item_count=1,
                    llm_called=True,
                    schema_valid=True,
                    token_in=response.token_in,
                    token_out=response.token_out,
                    total_tokens=response.total_tokens,
                    cached_tokens=response.cached_tokens,
                    reasoning_tokens=response.reasoning_tokens,
                    total_tokens_estimated=response.total_tokens_estimated,
                    latency_ms=response.latency_ms,
                    request_id=response.request_id,
                    retries=response.retries,
                    error_code=None,
                    cached=response.cached,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "reflection_stage1_schema_invalid: risk_id=%s, request_id=%s, error=%s"
                % (candidate.risk_id, response.request_id or "-", exc)
            )
            return _Stage1ExecResult(
                item_index=candidate.item_index,
                decision=fallback,
                diagnostic=_build_diagnostic(
                    stage="stage1",
                    source_file=candidate.source_file,
                    doc_id=candidate.doc_id,
                    risk_type=candidate.risk_type,
                    scope_key=candidate.risk_id,
                    input_item_count=1,
                    llm_called=True,
                    schema_valid=False,
                    token_in=response.token_in,
                    token_out=response.token_out,
                    total_tokens=response.total_tokens,
                    cached_tokens=response.cached_tokens,
                    reasoning_tokens=response.reasoning_tokens,
                    total_tokens_estimated=response.total_tokens_estimated,
                    latency_ms=response.latency_ms,
                    request_id=response.request_id,
                    retries=response.retries,
                    error_code="invalid_schema",
                    cached=response.cached,
                ),
            )

    def _reflect_stage2_one(self, task: Stage2GroupTask) -> _Stage2ExecResult:
        fallback = Stage2Decision(
            group_key=task.group_key,
            consistency_status="consistent",
            group_reason="反思失败，回退保留原结果",
            adjustments=[],
        )

        if not self._enabled or self._client is None:
            return _Stage2ExecResult(
                decision=fallback,
                diagnostic=_build_diagnostic(
                    stage="stage2",
                    source_file=task.source_file,
                    doc_id=task.doc_id,
                    risk_type=task.risk_type,
                    scope_key=task.group_key,
                    input_item_count=len(task.selected_items),
                    llm_called=False,
                    schema_valid=True,
                    token_in=0,
                    token_out=0,
                    total_tokens=0,
                    cached_tokens=0,
                    reasoning_tokens=0,
                    total_tokens_estimated=False,
                    latency_ms=0.0,
                    request_id="",
                    retries=0,
                    error_code="llm_disabled",
                    cached=False,
                ),
            )

        messages = build_group_reflection_messages(task)
        try:
            response = self._client.chat_with_metadata(messages)
        except LLMClientError as exc:
            self._logger.error(
                "reflection_stage2_llm_error: group=%s, retries=%s, error_code=%s, error=%s"
                % (task.group_key, exc.retries, exc.error_code, exc)
            )
            return _Stage2ExecResult(
                decision=fallback,
                diagnostic=_build_diagnostic(
                    stage="stage2",
                    source_file=task.source_file,
                    doc_id=task.doc_id,
                    risk_type=task.risk_type,
                    scope_key=task.group_key,
                    input_item_count=len(task.selected_items),
                    llm_called=True,
                    schema_valid=False,
                    token_in=0,
                    token_out=0,
                    total_tokens=0,
                    cached_tokens=0,
                    reasoning_tokens=0,
                    total_tokens_estimated=False,
                    latency_ms=0.0,
                    request_id=exc.request_id,
                    retries=exc.retries,
                    error_code=exc.error_code or "llm_error",
                    cached=False,
                ),
            )

        try:
            payload = parse_json_object(response.content)
            decision = _parse_stage2_decision(payload=payload, group_key=task.group_key)
            return _Stage2ExecResult(
                decision=decision,
                diagnostic=_build_diagnostic(
                    stage="stage2",
                    source_file=task.source_file,
                    doc_id=task.doc_id,
                    risk_type=task.risk_type,
                    scope_key=task.group_key,
                    input_item_count=len(task.selected_items),
                    llm_called=True,
                    schema_valid=True,
                    token_in=response.token_in,
                    token_out=response.token_out,
                    total_tokens=response.total_tokens,
                    cached_tokens=response.cached_tokens,
                    reasoning_tokens=response.reasoning_tokens,
                    total_tokens_estimated=response.total_tokens_estimated,
                    latency_ms=response.latency_ms,
                    request_id=response.request_id,
                    retries=response.retries,
                    error_code=None,
                    cached=response.cached,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "reflection_stage2_schema_invalid: group=%s, request_id=%s, error=%s"
                % (task.group_key, response.request_id or "-", exc)
            )
            return _Stage2ExecResult(
                decision=fallback,
                diagnostic=_build_diagnostic(
                    stage="stage2",
                    source_file=task.source_file,
                    doc_id=task.doc_id,
                    risk_type=task.risk_type,
                    scope_key=task.group_key,
                    input_item_count=len(task.selected_items),
                    llm_called=True,
                    schema_valid=False,
                    token_in=response.token_in,
                    token_out=response.token_out,
                    total_tokens=response.total_tokens,
                    cached_tokens=response.cached_tokens,
                    reasoning_tokens=response.reasoning_tokens,
                    total_tokens_estimated=response.total_tokens_estimated,
                    latency_ms=response.latency_ms,
                    request_id=response.request_id,
                    retries=response.retries,
                    error_code="invalid_schema",
                    cached=response.cached,
                ),
            )


def _build_stage1_candidates(
    *,
    items: list[dict[str, object]],
    contexts: dict[int, dict[str, object]],
    source_file: str,
    doc_id: str,
    threshold: int,
) -> list[Stage1Candidate]:
    candidates: list[Stage1Candidate] = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        result = normalize_result(str(item.get("result", "")).strip())
        if result != "不合格":
            item.setdefault("reflection_stage1_action", "skip")
            item.setdefault("reflection_stage1_reason", "非不合格项，默认跳过 Stage-1")
            continue

        context = contexts.get(index, {})
        fp_score = int(context.get("fp_risk_score", 0))
        if fp_score < threshold:
            item.setdefault("reflection_stage1_action", "skip")
            item.setdefault("reflection_stage1_reason", f"fp_risk_score={fp_score} < {threshold}")
            continue

        candidates.append(
            Stage1Candidate(
                source_file=source_file,
                doc_id=doc_id,
                risk_id=str(context.get("risk_id", item.get("risk_id", ""))),
                item_index=index,
                risk_type=str(context.get("risk_type", item.get("risk_type", ""))),
                initial_result=result,
                rule_hit_id=str(item.get("rule_hit_id", "")).strip(),
                rule_hit=str(item.get("rule_hit", "")).strip(),
                span=str(item.get("span", "")).strip(),
                suggest=str(item.get("suggest", "")).strip(),
                chunk_id=item.get("chunk_id"),
                evidence_window=clamp_text(str(context.get("evidence_window", "")), 420),
                evidence_window_source=str(context.get("evidence_window_source", "")),
                chunk_excerpt=clamp_text(str(context.get("chunk_excerpt", "")), 620),
                fp_risk_score=fp_score,
                fp_risk_flags=[str(flag) for flag in context.get("fp_risk_flags", []) if str(flag)],
            )
        )

    return candidates


def _group_items_by_risk_type(
    items: list[dict[str, object]],
    *,
    doc_id: str,
) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        risk_type = str(item.get("risk_type", "")).strip()
        if not risk_type:
            continue
        key = (doc_id, risk_type)
        groups.setdefault(key, []).append(index)
    return groups


def _build_stage2_group_tasks(
    *,
    source_file: str,
    doc_id: str,
    items: list[dict[str, object]],
    contexts: dict[int, dict[str, object]],
    group_map: dict[tuple[str, str], list[int]],
    stage2_max_items: int,
) -> list[Stage2GroupTask]:
    tasks: list[Stage2GroupTask] = []

    for (group_doc_id, risk_type), indexes in group_map.items():
        group_items = [items[index] for index in indexes if isinstance(items[index], dict)]
        if not _is_conflict_group(group_items):
            continue

        selected_indexes = _select_stage2_indexes(items=items, indexes=indexes, max_items=stage2_max_items)
        selected_items: list[Stage2ItemSnapshot] = []
        for index in selected_indexes:
            item = items[index]
            context = contexts.get(index, {})
            selected_items.append(
                Stage2ItemSnapshot(
                    risk_id=str(item.get("risk_id", "")).strip(),
                    chunk_id=item.get("chunk_id"),
                    result=normalize_result(str(item.get("result", "")).strip()),
                    rule_hit_id=str(item.get("rule_hit_id", "")).strip(),
                    span=str(item.get("span", "")).strip(),
                    suggest=str(item.get("suggest", "")).strip(),
                    evidence_window=clamp_text(str(context.get("evidence_window", "")), 380),
                    fp_risk_score=int(context.get("fp_risk_score", item.get("fp_risk_score", 0))),
                )
            )

        group_key = f"{group_doc_id}::{risk_type}"
        tasks.append(
            Stage2GroupTask(
                source_file=source_file,
                doc_id=doc_id,
                risk_type=risk_type,
                group_key=group_key,
                item_total=len(indexes),
                selected_items=selected_items,
            )
        )

    return tasks


def _select_stage2_indexes(*, items: list[dict[str, object]], indexes: list[int], max_items: int) -> list[int]:
    priority = {"不合格": 0, "待复核": 1, "合格": 2}

    ordered = sorted(
        indexes,
        key=lambda idx: (
            priority.get(normalize_result(str(items[idx].get("result", "")).strip()), 9),
            -int(items[idx].get("fp_risk_score", 0)),
            idx,
        ),
    )
    return ordered[:max_items]


def _is_conflict_group(items: list[dict[str, object]]) -> bool:
    if len(items) < 2:
        return False

    results = [normalize_result(str(item.get("result", "")).strip()) for item in items]
    unique = {result for result in results if result}

    if len(unique) >= 2:
        return True

    bad_count = sum(1 for result in results if result == "不合格")
    neutral_count = sum(1 for result in results if result in {"合格", "待复核"})
    if bad_count == 1 and neutral_count >= 2:
        return True

    return False


def _build_item_evidence_context(
    *,
    item: dict[str, object],
    chunk_map: dict[str, str],
    evidence_max_chars: int,
) -> dict[str, str]:
    chunk_id = item.get("chunk_id")
    chunk_text = chunk_map.get(str(chunk_id), "")
    span = str(item.get("span", "")).strip()
    span_offset = coerce_span_offset(item.get("span_offset"))

    sentences = split_sentences_with_offsets(chunk_text)
    sentence_index = None
    if span_offset is not None and sentences:
        sentence_index = locate_sentence_by_offset(
            sentences=sentences,
            start=span_offset[0],
            end=span_offset[1],
        )

    evidence = build_evidence_window(sentences, sentence_index)
    if not evidence.evidence_window:
        evidence = build_fallback_evidence_window(
            chunk_text=chunk_text,
            span=span,
            span_offset=span_offset,
            max_chars=evidence_max_chars,
        )

    return {
        "evidence_window": clamp_text(evidence.evidence_window, evidence_max_chars),
        "evidence_window_source": evidence.source,
        "chunk_excerpt": clamp_text(chunk_text, max(evidence_max_chars, 420)),
    }


def _apply_stage1_decision(*, item: dict[str, object], decision: Stage1Decision) -> bool:
    original_result = normalize_result(str(item.get("result", "")).strip())
    original_span = str(item.get("span", "")).strip()
    original_suggest = str(item.get("suggest", "")).strip()

    action = decision.action
    target_result = original_result

    if action == "downgrade":
        target_result = "待复核"
    elif action == "reject":
        target_result = "合格"
    elif action == "revise":
        revised = normalize_result(decision.revised_result)
        if revised in {"合格", "待复核", "不合格"}:
            target_result = revised

    # Stage-1 明确禁止：合格 -> 不合格
    if original_result == "合格" and target_result == "不合格":
        target_result = original_result

    changed = False
    if target_result and target_result != original_result:
        item["result"] = target_result
        changed = True

    if action == "revise":
        revised_span = decision.revised_span.strip()
        revised_suggest = decision.revised_suggest.strip()
        if revised_span and revised_span != original_span:
            item["span"] = revised_span
            changed = True
        if revised_suggest and revised_suggest != original_suggest:
            item["suggest"] = revised_suggest
            changed = True

    item["reflection_stage1_action"] = action
    item["reflection_stage1_reason"] = decision.reason
    item["reflection_stage1_evidence_sufficient"] = decision.evidence_sufficient
    return changed


def _apply_stage2_decision(
    *,
    decision: Stage2Decision,
    items: list[dict[str, object]],
    group_indexes: list[int],
) -> int:
    if not group_indexes:
        return 0

    adjusted_count = 0
    index_by_risk_id: dict[str, list[int]] = {}
    group_index_set = set(group_indexes)

    for index in group_indexes:
        item = items[index]
        if not isinstance(item, dict):
            continue
        risk_id = str(item.get("risk_id", "")).strip()
        if not risk_id:
            continue
        index_by_risk_id.setdefault(risk_id, []).append(index)

    for adjustment in decision.adjustments:
        candidates = [idx for idx in index_by_risk_id.get(adjustment.risk_id, []) if idx in group_index_set]
        if not candidates:
            continue

        index = candidates[0]
        item = items[index]
        if not isinstance(item, dict):
            continue

        changed = _apply_stage2_adjustment(item=item, adjustment=adjustment)
        if changed:
            adjusted_count += 1

    for index in group_indexes:
        item = items[index]
        if not isinstance(item, dict):
            continue
        if "reflection_stage2_action" not in item:
            item["reflection_stage2_action"] = "keep"
            item["reflection_stage2_reason"] = decision.group_reason

    return adjusted_count


def _apply_stage2_adjustment(*, item: dict[str, object], adjustment: Stage2Adjustment) -> bool:
    current_result = normalize_result(str(item.get("result", "")).strip())
    target_result = current_result

    if adjustment.action == "downgrade":
        target_result = "待复核"
    elif adjustment.action == "reject":
        target_result = "合格"
    elif adjustment.action == "keep":
        target_result = current_result

    # Stage-2 禁止将非不合格升级成不合格。
    if target_result == "不合格" and current_result != "不合格":
        target_result = current_result

    changed = False
    if target_result and target_result != current_result:
        item["result"] = target_result
        changed = True

    item["reflection_stage2_action"] = adjustment.action
    item["reflection_stage2_reason"] = adjustment.reason
    return changed


def _parse_stage1_decision(*, payload: dict[str, object], candidate: Stage1Candidate) -> Stage1Decision:
    action = str(payload.get("action", "keep")).strip().lower()
    if action not in {"keep", "downgrade", "reject", "revise"}:
        action = "keep"

    evidence_sufficient = str(payload.get("evidence_sufficient", "partial")).strip().lower()
    if evidence_sufficient not in {"yes", "partial", "no"}:
        evidence_sufficient = "partial"

    revised_result = normalize_result(str(payload.get("revised_result", candidate.initial_result)).strip())
    if revised_result not in {"合格", "待复核", "不合格"}:
        revised_result = candidate.initial_result

    reason = str(payload.get("reason", "")).strip() or "模型未给出理由"
    return Stage1Decision(
        risk_id=str(payload.get("risk_id", candidate.risk_id)).strip() or candidate.risk_id,
        action=action,
        evidence_sufficient=evidence_sufficient,
        revised_result=revised_result,
        revised_span=str(payload.get("revised_span", candidate.span)).strip(),
        revised_suggest=str(payload.get("revised_suggest", candidate.suggest)).strip(),
        reason=reason,
    )


def _parse_stage2_decision(*, payload: dict[str, object], group_key: str) -> Stage2Decision:
    status = str(payload.get("consistency_status", "consistent")).strip().lower()
    if status not in {"consistent", "partially_inconsistent", "inconsistent"}:
        status = "partially_inconsistent"

    adjustments_raw = payload.get("adjustments", [])
    if not isinstance(adjustments_raw, list):
        raise ValueError("adjustments 字段必须为数组")

    adjustments: list[Stage2Adjustment] = []
    for index, row in enumerate(adjustments_raw, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"adjustments[{index}] 不是对象")

        risk_id = str(row.get("risk_id", "")).strip()
        action = str(row.get("action", "keep")).strip().lower()
        target_result = normalize_result(str(row.get("target_result", "")).strip())
        reason = str(row.get("reason", "")).strip() or "未提供理由"

        if not risk_id:
            continue
        if action not in {"keep", "downgrade", "reject"}:
            action = "keep"
        if target_result not in {"合格", "待复核", "不合格"}:
            target_result = "待复核" if action == "downgrade" else "合格"

        adjustments.append(
            Stage2Adjustment(
                risk_id=risk_id,
                action=action,
                target_result=target_result,
                reason=reason,
            )
        )

    return Stage2Decision(
        group_key=(
            str(payload.get("group_key", "")).strip()
            if str(payload.get("group_key", "")).strip() == group_key
            else group_key
        ),
        consistency_status=status,
        group_reason=str(payload.get("group_reason", "")).strip() or "模型未给出组级说明",
        adjustments=adjustments,
    )


def _build_chunk_map(chunks: object) -> dict[str, str]:
    chunk_map: dict[str, str] = {}
    if not isinstance(chunks, list):
        return chunk_map

    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk.get("chunk_id", index)
        content = str(chunk.get("content", ""))
        chunk_map[str(chunk_id)] = content
    return chunk_map


def _resolve_doc_id(*, payload: dict[str, object], file_path: Path) -> str:
    review_meta = payload.get("review_meta")
    if isinstance(review_meta, dict):
        review_doc_id = str(review_meta.get("doc_id", "")).strip()
        if review_doc_id:
            return review_doc_id
    payload_doc_id = str(payload.get("doc_id", "")).strip()
    if payload_doc_id:
        return payload_doc_id
    return file_path.stem


def _resolve_source_file(*, payload: dict[str, object], file_path: Path) -> str:
    source_file = str(payload.get("source_file", "")).strip()
    return source_file or file_path.name


def _resolve_rule_version(*, payload: dict[str, object]) -> str:
    review_meta = payload.get("review_meta")
    if isinstance(review_meta, dict):
        value = str(review_meta.get("rule_version", "")).strip()
        if value:
            return value
    return ""


def _group_key_to_tuple(group_key: str) -> tuple[str, str]:
    if "::" not in group_key:
        return "", group_key
    left, right = group_key.split("::", 1)
    return left, right


def _build_output_path(*, file_path: Path, output_dir: Path) -> Path:
    if file_path.name.endswith(".review.json"):
        filename = file_path.name.replace(".review.json", ".reflection.json")
    else:
        filename = f"{file_path.stem}.reflection.json"
    return output_dir / filename


def _write_diagnostics(path: Path, diagnostics: list[ReflectionCallDiagnostic]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in diagnostics:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def _build_diagnostic(
    *,
    stage: str,
    source_file: str,
    doc_id: str,
    risk_type: str,
    scope_key: str,
    input_item_count: int,
    llm_called: bool,
    schema_valid: bool,
    token_in: int,
    token_out: int,
    total_tokens: int,
    cached_tokens: int,
    reasoning_tokens: int,
    total_tokens_estimated: bool,
    latency_ms: float,
    request_id: str,
    retries: int,
    error_code: str | None,
    cached: bool,
) -> ReflectionCallDiagnostic:
    return ReflectionCallDiagnostic(
        stage=stage,
        source_file=source_file,
        doc_id=doc_id,
        risk_type=risk_type,
        scope_key=scope_key,
        input_item_count=input_item_count,
        llm_called=llm_called,
        schema_valid=schema_valid,
        token_in=token_in,
        token_out=token_out,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens_estimated=total_tokens_estimated,
        latency_ms=latency_ms,
        request_id=request_id,
        retries=retries,
        error_code=error_code,
        cached=cached,
    )
