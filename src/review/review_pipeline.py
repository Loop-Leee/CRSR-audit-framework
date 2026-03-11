"""结构化审查流水线。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.llm import LLMClientError, LLMSettings, OpenAICompatibleClient, run_tasks
from src.tools.logger import Logger

from .rule_store import RuleHit, RuleStore


_JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*\}")
MAX_SCHEMA_RETRY_LIMIT = 3


@dataclass(frozen=True, slots=True)
class ReviewTask:
    """单个 risk_type 审查任务。"""

    source_file: str
    doc_id: str
    chunk_id: int | str
    content: str
    risk_type: str
    candidate_rules: list[dict[str, str]]


@dataclass(frozen=True, slots=True)
class ReviewTaskDiagnostic:
    """单任务调用诊断信息。"""

    source_file: str
    doc_id: str
    chunk_id: int | str
    risk_type: str
    candidate_rule_count: int
    llm_called: bool
    schema_valid: bool
    item_count: int
    token_in: int
    token_out: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    total_tokens_estimated: bool
    latency_ms: float
    request_id: str
    retries: int
    error_code: str | None
    cached: bool
    schema_retries: int


@dataclass(frozen=True, slots=True)
class ReviewRunMetrics:
    """审查运行统计指标。"""

    file_count: int
    chunk_count: int
    task_count: int
    emitted_item_count: int
    llm_called_count: int
    schema_valid_rate: float
    llm_error_rate: float
    span_match_rate: float
    rule_id_known_rate: float
    avg_token_in: float
    avg_token_out: float
    avg_chunk_token_in: float
    avg_chunk_token_out: float
    avg_total_token: float
    avg_latency_ms: float
    schema_retry_count: int
    ablation_no_rules: bool
    ablation_coarse_rules: bool
    rule_version: str

    def to_dict(self) -> dict[str, float | int | bool | str]:
        """导出指标字典。"""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReviewRunResult:
    """审查执行结果。"""

    outputs: list[Path]
    diagnostics: list[ReviewTaskDiagnostic]
    metrics: ReviewRunMetrics
    trace_path: Path
    metrics_path: Path


@dataclass(frozen=True, slots=True)
class _TaskReviewResult:
    """单任务审查结果。"""

    items: list[dict[str, str]]
    diagnostic: ReviewTaskDiagnostic


def discover_classified_files(input_path: Path) -> list[Path]:
    """发现待审查分类文件。

    Args:
        input_path: `*.classified.json` 文件或目录路径。

    Returns:
        list[Path]: 待处理文件列表。

    Raises:
        FileNotFoundError: 输入路径不存在时抛出。
    """

    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    files = sorted(input_path.glob("*.classified.json"))
    if files:
        return files
    return sorted(input_path.glob("*.json"))


def run_review(
    *,
    input_path: Path,
    output_dir: Path,
    rules_path: Path,
    rule_version: str,
    llm_settings: LLMSettings,
    logger: Logger,
    ablation_no_rules: bool = False,
    ablation_coarse_rules: bool = False,
    schema_retry_limit: int = 1,
    ground_truth: str = "待审核",
) -> list[Path]:
    """执行审查主流程（兼容接口，仅返回输出文件列表）。

    Args:
        input_path: classified 输入文件/目录。
        output_dir: 审查结果输出目录。
        rules_path: expanded 规则文件路径。
        rule_version: 规则版本号。
        llm_settings: LLM 调用配置。
        logger: 模块日志对象。
        ablation_no_rules: 是否关闭规则注入消融。
        ablation_coarse_rules: 是否启用粗粒度规则消融。
        schema_retry_limit: 单任务 schema 失败重试上限。
        ground_truth: 写入每条 review_item 的人工标注初值。

    Returns:
        list[Path]: 输出 `*.review.json` 文件路径列表。
    """

    return run_review_with_diagnostics(
        input_path=input_path,
        output_dir=output_dir,
        rules_path=rules_path,
        rule_version=rule_version,
        llm_settings=llm_settings,
        logger=logger,
        ablation_no_rules=ablation_no_rules,
        ablation_coarse_rules=ablation_coarse_rules,
        schema_retry_limit=schema_retry_limit,
        ground_truth=ground_truth,
    ).outputs


def run_review_with_diagnostics(
    *,
    input_path: Path,
    output_dir: Path,
    rules_path: Path,
    rule_version: str,
    llm_settings: LLMSettings,
    logger: Logger,
    ablation_no_rules: bool = False,
    ablation_coarse_rules: bool = False,
    schema_retry_limit: int = 1,
    ground_truth: str = "待审核",
) -> ReviewRunResult:
    """执行结构化审查并输出诊断、指标。

    Args:
        input_path: classified 输入文件/目录。
        output_dir: 审查结果输出目录。
        rules_path: expanded 规则文件路径。
        rule_version: 规则版本号。
        llm_settings: LLM 调用配置。
        logger: 模块日志对象。
        ablation_no_rules: 是否关闭规则注入消融。
        ablation_coarse_rules: 是否启用粗粒度规则消融。
        schema_retry_limit: 单任务 schema 失败重试上限。
        ground_truth: 写入每条 review_item 的人工标注初值。

    Returns:
        ReviewRunResult: 输出文件、任务级诊断和聚合指标。

    Raises:
        ValueError: 消融参数冲突时抛出。
        FileNotFoundError: 输入目录无文件时抛出。
        RuntimeError: 处理过程中存在失败文件时抛出。
    """

    if ablation_no_rules and ablation_coarse_rules:
        raise ValueError("ablation_no_rules 与 ablation_coarse_rules 不能同时开启。")
    if schema_retry_limit < 0 or schema_retry_limit > MAX_SCHEMA_RETRY_LIMIT:
        raise ValueError(
            f"schema_retry_limit 必须在 [0, {MAX_SCHEMA_RETRY_LIMIT}] 范围内。"
        )

    files = discover_classified_files(input_path)
    if not files:
        raise FileNotFoundError(f"未找到待审查文件: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    store = RuleStore()
    store.load_rules(rules_path, rule_version=rule_version)
    effective_rule_version = rule_version
    if ablation_coarse_rules:
        effective_rule_version = store.build_coarse_version(
            source_version=rule_version,
            target_version=f"{rule_version}_coarse",
        )
    store.set_active_version(effective_rule_version)

    reviewer = _RuleReviewer(
        rule_store=store,
        rule_version=effective_rule_version,
        settings=llm_settings,
        logger=logger,
        ablation_no_rules=ablation_no_rules,
        schema_retry_limit=schema_retry_limit,
    )
    ground_truth_value = ground_truth.strip() or "待审核"

    logger.info(
        "review_start: file_count=%s, input=%s, output=%s, rule_version=%s, ablation_no_rules=%s, "
        "ablation_coarse_rules=%s, schema_retry_limit=%s, ground_truth=%s"
        % (
            len(files),
            input_path,
            output_dir,
            effective_rule_version,
            ablation_no_rules,
            ablation_coarse_rules,
            schema_retry_limit,
            ground_truth_value,
        )
    )

    outputs: list[Path] = []
    diagnostics: list[ReviewTaskDiagnostic] = []
    failures: list[str] = []
    total_chunk_count = 0
    total_task_count = 0
    emitted_item_count = 0
    span_matched_count = 0
    known_rule_id_count = 0

    for file_path in files:
        try:
            logger.info(f"review_file_start: file={file_path}")
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            chunks = _extract_chunks(payload, file_path)
            total_chunk_count += len(chunks)

            doc_id = str(payload.get("doc_id", file_path.stem))
            source_file = str(payload.get("source_file", file_path.name))
            tasks = _build_review_tasks(
                chunks=chunks,
                doc_id=doc_id,
                source_file=source_file,
                store=store,
                rule_version=effective_rule_version,
                ablation_no_rules=ablation_no_rules,
            )
            total_task_count += len(tasks)
            task_results = reviewer.review_many(tasks)

            file_items: list[dict[str, object]] = []
            counters: dict[tuple[str, str, str, str], int] = {}

            for task, task_result in zip(tasks, task_results):
                diagnostics.append(task_result.diagnostic)
                for raw in task_result.items:
                    resolved_rule_hit_id = _resolve_rule_hit_id(
                        raw_item=raw,
                        task=task,
                        store=store,
                        rule_version=effective_rule_version,
                    )
                    span_offset = _find_span_offset(raw["span"], task.content)
                    if span_offset is not None:
                        span_matched_count += 1
                    if resolved_rule_hit_id != "UNKNOWN":
                        known_rule_id_count += 1

                    counter_key = (
                        task.doc_id,
                        str(task.chunk_id),
                        raw["risk_type"],
                        resolved_rule_hit_id,
                    )
                    next_idx = counters.get(counter_key, 0) + 1
                    counters[counter_key] = next_idx
                    risk_id = (
                        f"{task.doc_id}#c{task.chunk_id}#rt{raw['risk_type']}#r{resolved_rule_hit_id}#{next_idx}"
                    )

                    # rule_version 为文件级元数据，仅写入 review_meta，避免在每条 item 重复。
                    file_items.append(
                        {
                            "risk_id": risk_id,
                            "source_file": task.source_file,
                            "chunk_id": task.chunk_id,
                            "risk_type": raw["risk_type"],
                            "result": raw["result"],
                            "rule_hit": raw["rule_hit"],
                            "rule_hit_id": resolved_rule_hit_id,
                            "span": raw["span"],
                            "span_offset": span_offset,
                            "suggest": raw["suggest"],
                            "ground_truth": ground_truth_value,
                        }
                    )
                    emitted_item_count += 1

            output_payload = dict(payload)
            output_payload["review_meta"] = {
                "doc_id": doc_id,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "rule_version": effective_rule_version,
                "ablation_no_rules": ablation_no_rules,
                "ablation_coarse_rules": ablation_coarse_rules,
                "task_count": len(tasks),
                "review_item_count": len(file_items),
            }
            output_payload["review_items"] = file_items

            output_path = _build_output_path(file_path, output_dir)
            output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            outputs.append(output_path)

            logger.info(
                "review_file_done: file=%s, chunk_count=%s, task_count=%s, review_item_count=%s, output=%s"
                % (file_path.name, len(chunks), len(tasks), len(file_items), output_path)
            )
        except Exception as exc:  # noqa: BLE001
            message = f"review_file_failed: file={file_path}, error={exc}"
            logger.error(message)
            failures.append(message)

    trace_path = output_dir / "review_trace.jsonl"
    _write_diagnostics(trace_path, diagnostics)

    metrics = _compute_review_metrics(
        diagnostics=diagnostics,
        file_count=len(outputs),
        chunk_count=total_chunk_count,
        task_count=total_task_count,
        emitted_item_count=emitted_item_count,
        span_matched_count=span_matched_count,
        known_rule_id_count=known_rule_id_count,
        ablation_no_rules=ablation_no_rules,
        ablation_coarse_rules=ablation_coarse_rules,
        rule_version=effective_rule_version,
    )
    metrics_path = output_dir / "review_metrics.json"
    metrics_path.write_text(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"review_trace_written: {trace_path}")
    logger.info(f"review_metrics_written: {metrics_path}")

    if failures:
        raise RuntimeError("review 流程存在失败文件，请检查日志。")

    logger.info(
        "review_done: success_count=%s, task_count=%s, review_item_count=%s"
        % (len(outputs), total_task_count, emitted_item_count)
    )

    return ReviewRunResult(
        outputs=outputs,
        diagnostics=diagnostics,
        metrics=metrics,
        trace_path=trace_path,
        metrics_path=metrics_path,
    )


def _extract_chunks(payload: object, source: Path) -> list[dict]:
    """提取并校验 classified chunks。"""

    if isinstance(payload, dict):
        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError(f"chunks 字段缺失或类型错误: {source}")
    elif isinstance(payload, list):
        chunks = payload
    else:
        raise ValueError(f"输入 JSON 必须是对象或数组: {source}")

    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(f"chunks[{index}] 不是对象: {source}")
        if not isinstance(chunk.get("content"), str):
            raise ValueError(f"chunk content 非字符串: {source}, chunk_id={chunk.get('chunk_id', index)}")
        risks = chunk.get("risk_type", [])
        if not isinstance(risks, list):
            raise ValueError(f"chunk risk_type 非数组: {source}, chunk_id={chunk.get('chunk_id', index)}")

    return chunks


def _build_output_path(input_file: Path, output_dir: Path) -> Path:
    if input_file.name.endswith(".classified.json"):
        filename = input_file.name.replace(".classified.json", ".review.json")
    else:
        filename = f"{input_file.stem}.review.json"
    return output_dir / filename


def _build_review_tasks(
    *,
    chunks: list[dict],
    doc_id: str,
    source_file: str,
    store: RuleStore,
    rule_version: str,
    ablation_no_rules: bool,
) -> list[ReviewTask]:
    tasks: list[ReviewTask] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", index)
        content = chunk["content"]
        seen: set[str] = set()
        for raw_risk in chunk.get("risk_type", []):
            if not isinstance(raw_risk, str):
                continue
            risk_type = raw_risk.strip()
            if not risk_type or risk_type in seen:
                continue
            seen.add(risk_type)
            candidate_rules = []
            if not ablation_no_rules:
                candidate_rules = store.get_rules(risk_type, rule_version=rule_version)
            tasks.append(
                ReviewTask(
                    source_file=source_file,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    content=content,
                    risk_type=risk_type,
                    candidate_rules=candidate_rules,
                )
            )
    return tasks


def _write_diagnostics(path: Path, diagnostics: list[ReviewTaskDiagnostic]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in diagnostics:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def _compute_review_metrics(
    *,
    diagnostics: list[ReviewTaskDiagnostic],
    file_count: int,
    chunk_count: int,
    task_count: int,
    emitted_item_count: int,
    span_matched_count: int,
    known_rule_id_count: int,
    ablation_no_rules: bool,
    ablation_coarse_rules: bool,
    rule_version: str,
) -> ReviewRunMetrics:
    llm_rows = [item for item in diagnostics if item.llm_called]
    schema_valid_rate = _safe_rate(sum(1 for item in llm_rows if item.schema_valid), len(llm_rows))
    llm_error_rate = _safe_rate(sum(1 for item in llm_rows if item.error_code is not None), len(llm_rows))
    span_match_rate = _safe_rate(span_matched_count, emitted_item_count)
    rule_id_known_rate = _safe_rate(known_rule_id_count, emitted_item_count)

    avg_token_in = _safe_mean([item.token_in for item in llm_rows])
    avg_token_out = _safe_mean([item.token_out for item in llm_rows])
    chunk_token_in: dict[tuple[str, str, str], int] = {}
    chunk_token_out: dict[tuple[str, str, str], int] = {}
    for item in llm_rows:
        chunk_key = (item.source_file, item.doc_id, str(item.chunk_id))
        chunk_token_in[chunk_key] = chunk_token_in.get(chunk_key, 0) + int(item.token_in)
        chunk_token_out[chunk_key] = chunk_token_out.get(chunk_key, 0) + int(item.token_out)
    avg_chunk_token_in = _safe_mean(list(chunk_token_in.values()))
    avg_chunk_token_out = _safe_mean(list(chunk_token_out.values()))
    avg_total_token = _safe_mean([item.total_tokens for item in llm_rows])
    avg_latency_ms = _safe_mean([item.latency_ms for item in llm_rows])
    schema_retry_count = sum(item.schema_retries for item in llm_rows)

    return ReviewRunMetrics(
        file_count=file_count,
        chunk_count=chunk_count,
        task_count=task_count,
        emitted_item_count=emitted_item_count,
        llm_called_count=len(llm_rows),
        schema_valid_rate=schema_valid_rate,
        llm_error_rate=llm_error_rate,
        span_match_rate=span_match_rate,
        rule_id_known_rate=rule_id_known_rate,
        avg_token_in=avg_token_in,
        avg_token_out=avg_token_out,
        avg_chunk_token_in=avg_chunk_token_in,
        avg_chunk_token_out=avg_chunk_token_out,
        avg_total_token=avg_total_token,
        avg_latency_ms=avg_latency_ms,
        schema_retry_count=schema_retry_count,
        ablation_no_rules=ablation_no_rules,
        ablation_coarse_rules=ablation_coarse_rules,
        rule_version=rule_version,
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _safe_mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(item) for item in values)) / float(len(values))


def _resolve_rule_hit_id(
    *,
    raw_item: dict[str, str],
    task: ReviewTask,
    store: RuleStore,
    rule_version: str,
) -> str:
    """解析规则命中 ID，优先使用模型输出，失败时回匹配规则文本。"""

    candidate_hits = [
        RuleHit(
            risk_type=str(row.get("risk_type", task.risk_type)),
            rule_hit_id=str(row.get("rule_hit_id", "")).strip(),
            rule_hit_text=str(row.get("rule_hit_text", "")).strip(),
            rule_version=str(row.get("rule_version", rule_version)),
        )
        for row in task.candidate_rules
        if str(row.get("rule_hit_id", "")).strip()
    ]

    if not candidate_hits:
        candidate_hits = store.get_rule_hits(task.risk_type, rule_version=rule_version)

    model_rule_hit_id = raw_item.get("rule_hit_id", "").strip()
    if model_rule_hit_id and any(hit.rule_hit_id == model_rule_hit_id for hit in candidate_hits):
        return model_rule_hit_id

    matched = _match_rule_hit_by_text(raw_item.get("rule_hit", ""), candidate_hits)
    if matched is not None:
        return matched.rule_hit_id

    return "UNKNOWN"


def _match_rule_hit_by_text(rule_hit_text: str, hits: list[RuleHit]) -> RuleHit | None:
    if not rule_hit_text:
        return None

    query = _normalize_text(rule_hit_text)
    if not query:
        return None

    for hit in hits:
        candidate = _normalize_text(hit.rule_hit_text)
        if query == candidate:
            return hit

    for hit in hits:
        candidate = _normalize_text(hit.rule_hit_text)
        if query and candidate and (query in candidate or candidate in query):
            return hit

    # 简化版相似度匹配（字符重合比），避免引入额外依赖。
    best_hit: RuleHit | None = None
    best_score = 0.0
    for hit in hits:
        candidate = _normalize_text(hit.rule_hit_text)
        if not candidate:
            continue
        overlap = len(set(query) & set(candidate))
        score = (2.0 * overlap) / float(len(set(query)) + len(set(candidate)))
        if score > best_score:
            best_score = score
            best_hit = hit

    if best_hit is not None and best_score >= 0.72:
        return best_hit
    return None


def _normalize_text(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def _find_span_offset(span: str, content: str) -> list[int] | None:
    """在 chunk 文本中定位 span 偏移，返回 [start, end)。"""

    if not span:
        return None
    start = content.find(span)
    if start < 0:
        return None
    return [start, start + len(span)]


class _RuleReviewer:
    """基于 LLM 的规则审查执行器。"""

    def __init__(
        self,
        *,
        rule_store: RuleStore,
        rule_version: str,
        settings: LLMSettings,
        logger: Logger,
        ablation_no_rules: bool,
        schema_retry_limit: int,
    ) -> None:
        self._rule_store = rule_store
        self._rule_version = rule_version
        self._settings = settings
        self._logger = logger
        self._ablation_no_rules = ablation_no_rules
        self._schema_retry_limit = schema_retry_limit
        self._enabled = settings.enabled

        if not self._enabled:
            self._logger.info("review llm 已禁用：llm.enabled=false")
            self._client = None
            return

        if not settings.model:
            self._logger.error("review llm 降级：缺少 LLM_MODEL，当前审查任务将不触发 LLM")
            self._enabled = False
            self._client = None
            return

        if not settings.api_key:
            self._logger.info("LLM_API_KEY 为空，将按无鉴权方式调用 OpenAI 兼容接口。")

        self._client = OpenAICompatibleClient(settings)

    def review_many(self, tasks: list[ReviewTask]) -> list[_TaskReviewResult]:
        """批量执行审查任务。"""

        if not tasks:
            return []

        return run_tasks(
            tasks,
            self.review_one,
            concurrent_enabled=self._settings.concurrent_enabled,
            max_concurrency=self._settings.max_concurrency,
        )

    def review_one(self, task: ReviewTask) -> _TaskReviewResult:
        """执行单任务审查。"""

        if not self._enabled or self._client is None:
            return _TaskReviewResult(
                items=[],
                diagnostic=self._build_diag(
                    task=task,
                    llm_called=False,
                    schema_valid=True,
                    item_count=0,
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
                    schema_retries=0,
                ),
            )

        if not self._ablation_no_rules and not task.candidate_rules:
            self._logger.error(
                "review 跳过任务：risk_type 无可用规则。file=%s, chunk_id=%s, risk_type=%s"
                % (task.source_file, task.chunk_id, task.risk_type)
            )
            return _TaskReviewResult(
                items=[],
                diagnostic=self._build_diag(
                    task=task,
                    llm_called=False,
                    schema_valid=False,
                    item_count=0,
                    token_in=0,
                    token_out=0,
                    total_tokens=0,
                    cached_tokens=0,
                    reasoning_tokens=0,
                    total_tokens_estimated=False,
                    latency_ms=0.0,
                    request_id="",
                    retries=0,
                    error_code="no_rules_for_risk_type",
                    cached=False,
                    schema_retries=0,
                ),
            )

        last_schema_error = ""
        for schema_retry_index in range(self._schema_retry_limit + 1):
            messages = self._build_messages(
                task=task,
                schema_feedback=last_schema_error if schema_retry_index > 0 else None,
            )

            try:
                response = self._client.chat_with_metadata(messages)
            except LLMClientError as exc:
                self._logger.error(
                    "review_llm_error: file=%s, chunk_id=%s, risk_type=%s, retries=%s, error_code=%s, error=%s"
                    % (task.source_file, task.chunk_id, task.risk_type, exc.retries, exc.error_code, exc)
                )
                return _TaskReviewResult(
                    items=[],
                    diagnostic=self._build_diag(
                        task=task,
                        llm_called=True,
                        schema_valid=False,
                        item_count=0,
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
                        schema_retries=schema_retry_index,
                    ),
                )

            try:
                parsed_items = self._parse_response(response.content, task)
                return _TaskReviewResult(
                    items=parsed_items,
                    diagnostic=self._build_diag(
                        task=task,
                        llm_called=True,
                        schema_valid=True,
                        item_count=len(parsed_items),
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
                        schema_retries=schema_retry_index,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                last_schema_error = str(exc)
                if schema_retry_index < self._schema_retry_limit:
                    self._logger.info(
                        "review_schema_retry: file=%s, chunk_id=%s, risk_type=%s, request_id=%s, attempt=%s/%s, error=%s"
                        % (
                            task.source_file,
                            task.chunk_id,
                            task.risk_type,
                            response.request_id or "-",
                            schema_retry_index + 1,
                            self._schema_retry_limit,
                            last_schema_error,
                        )
                    )
                    continue

                self._logger.error(
                    "review_schema_invalid: file=%s, chunk_id=%s, risk_type=%s, request_id=%s, error=%s"
                    % (task.source_file, task.chunk_id, task.risk_type, response.request_id or "-", exc)
                )
                return _TaskReviewResult(
                    items=[],
                    diagnostic=self._build_diag(
                        task=task,
                        llm_called=True,
                        schema_valid=False,
                        item_count=0,
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
                        schema_retries=schema_retry_index,
                    ),
                )

        raise RuntimeError("审查任务异常：schema 重试循环提前结束。")

    def _build_messages(self, task: ReviewTask, schema_feedback: str | None = None) -> list[dict[str, str]]:
        rule_section = ""
        output_schema_hint = (
            "输出 JSON 对象，字段为 items（数组）。每个 item 必须包含 risk_type、result、rule_hit_id、rule_hit、span、suggest。"
        )
        if self._ablation_no_rules:
            output_schema_hint = (
                "输出 JSON 对象，字段为 items（数组）。每个 item 必须包含 risk_type、result、rule_hit、span、suggest；"
                "可额外给出 rule_hit_id，没有可填 UNKNOWN。"
            )
        else:
            lines = []
            for item in task.candidate_rules:
                lines.append(f"- {item['rule_hit_id']}: {item['rule_hit_text']}")
            rule_section = "\n候选规则（必须从中选择 rule_hit_id）：\n" + "\n".join(lines)
        retry_section = ""
        if schema_feedback:
            retry_section = (
                "\n上一次输出未通过格式校验，请纠正后重新输出。\n"
                f"失败原因：{schema_feedback}\n"
            )

        user_prompt = (
            "请对给定合同片段执行单一审查类型的规则命中判断。\n"
            f"审查类型：{task.risk_type}\n"
            f"规则版本：{self._rule_version}\n"
            f"{output_schema_hint}\n"
            "若无命中，返回 {\"items\": []}。\n"
            "约束：\n"
            "1) risk_type 必须等于给定审查类型；\n"
            "2) result 只能是“合格”或“不合格”；\n"
            "3) span 必须是合同片段中的连续原文子串，不得改写；\n"
            "4) rule_hit 使用可读中文解释并尽量贴近规则原文；\n"
            "5) 只输出 JSON，不要输出额外文本。"
            f"{retry_section}"
            f"{rule_section}\n\n"
            "合同片段：\n"
            f"{task.content}"
        )

        return [
            {
                "role": "system",
                "content": "你是合同规则审查助手。严格输出 JSON，并遵守字段约束。",
            },
            {"role": "user", "content": user_prompt},
        ]

    def _parse_response(self, response_text: str, task: ReviewTask) -> list[dict[str, str]]:
        payload = _parse_json_object(response_text)
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError(f"模型输出缺少 items 数组: {payload}")

        candidate_rule_ids = {
            str(item.get("rule_hit_id", "")).strip() for item in task.candidate_rules if str(item.get("rule_hit_id", "")).strip()
        }
        parsed: list[dict[str, str]] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"items[{index}] 不是对象: {item}")

            risk_type = str(item.get("risk_type", "")).strip()
            result = str(item.get("result", "")).strip()
            rule_hit = str(item.get("rule_hit", "")).strip()
            span = str(item.get("span", "")).strip()
            suggest = str(item.get("suggest", "")).strip()
            if risk_type != task.risk_type:
                raise ValueError(f"items[{index}].risk_type 非候选值: {risk_type}")
            if result not in {"合格", "不合格"}:
                raise ValueError(f"items[{index}].result 非法: {result}")
            if not rule_hit:
                raise ValueError(f"items[{index}].rule_hit 为空")
            if not span:
                raise ValueError(f"items[{index}].span 为空")
            if not suggest:
                raise ValueError(f"items[{index}].suggest 为空")

            rule_hit_id = str(item.get("rule_hit_id", "")).strip()
            if not self._ablation_no_rules:
                if not rule_hit_id:
                    raise ValueError(f"items[{index}].rule_hit_id 为空")
                if candidate_rule_ids and rule_hit_id not in candidate_rule_ids:
                    raise ValueError(f"items[{index}].rule_hit_id 不在候选列表: {rule_hit_id}")

            parsed.append(
                {
                    "risk_type": risk_type,
                    "result": result,
                    "rule_hit_id": rule_hit_id or "UNKNOWN",
                    "rule_hit": rule_hit,
                    "span": span,
                    "suggest": suggest,
                }
            )

        return parsed

    @staticmethod
    def _build_diag(
        *,
        task: ReviewTask,
        llm_called: bool,
        schema_valid: bool,
        item_count: int,
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
        schema_retries: int,
    ) -> ReviewTaskDiagnostic:
        return ReviewTaskDiagnostic(
            source_file=task.source_file,
            doc_id=task.doc_id,
            chunk_id=task.chunk_id,
            risk_type=task.risk_type,
            candidate_rule_count=len(task.candidate_rules),
            llm_called=llm_called,
            schema_valid=schema_valid,
            item_count=item_count,
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
            schema_retries=schema_retries,
        )


def _parse_json_object(response_text: str) -> dict:
    try:
        payload = json.loads(response_text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    matched = _JSON_OBJECT_PATTERN.search(response_text)
    if not matched:
        raise ValueError(f"模型输出不是 JSON 对象: {response_text[:240]}")

    payload = json.loads(matched.group(0))
    if not isinstance(payload, dict):
        raise ValueError(f"模型输出 JSON 不是对象: {payload}")
    return payload
