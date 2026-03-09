"""Baseline 审查流水线。

该模块直接消费 chunking 产物，对每个 chunk 一次性注入全部审查类型与规则，
不依赖 classification/review/result 流程中的风险筛选与规则拆分。
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.llm import LLMClientError, LLMSettings, OpenAICompatibleClient, run_tasks
from src.tools.logger import Logger


_JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*\}")
MAX_SCHEMA_RETRY_LIMIT = 3


@dataclass(frozen=True, slots=True)
class BaselineRule:
    """单条 baseline 审查规则。"""

    risk_type: str
    rule_text: str

    def to_dict(self) -> dict[str, str]:
        """导出字典。"""

        return {"risk_type": self.risk_type, "rule_text": self.rule_text}


@dataclass(frozen=True, slots=True)
class BaselineTask:
    """单个 chunk baseline 审查任务。"""

    source_file: str
    doc_id: str
    chunk_id: int | str
    content: str
    risk_rules: list[dict[str, str]]


@dataclass(frozen=True, slots=True)
class BaselineTaskDiagnostic:
    """单任务调用诊断信息。"""

    source_file: str
    doc_id: str
    chunk_id: int | str
    risk_rule_count: int
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
class BaselineRunMetrics:
    """baseline 运行统计指标。"""

    file_count: int
    chunk_count: int
    task_count: int
    emitted_item_count: int
    llm_called_count: int
    schema_valid_rate: float
    llm_error_rate: float
    span_match_rate: float
    avg_token_in: float
    avg_token_out: float
    avg_total_token: float
    avg_latency_ms: float
    schema_retry_count: int
    risk_rule_count: int
    ablation_no_rules: bool
    ablation_no_span_offset: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        """导出指标字典。"""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BaselineRunResult:
    """baseline 执行结果。"""

    outputs: list[Path]
    diagnostics: list[BaselineTaskDiagnostic]
    metrics: BaselineRunMetrics
    trace_path: Path
    metrics_path: Path


@dataclass(frozen=True, slots=True)
class _TaskReviewResult:
    """单任务审查结果。"""

    items: list[dict[str, str]]
    diagnostic: BaselineTaskDiagnostic


def discover_chunk_files(input_path: Path) -> list[Path]:
    """发现待审查 chunk 文件。"""

    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    files = sorted(input_path.glob("*.chunks.json"))
    if files:
        return files
    return sorted(input_path.glob("*.json"))


def run_baseline(
    *,
    input_path: Path,
    output_dir: Path,
    risk_info_path: Path,
    llm_settings: LLMSettings,
    logger: Logger,
    ablation_no_rules: bool = False,
    ablation_no_span_offset: bool = False,
    schema_retry_limit: int = 1,
    max_new_tokens: int = 512,
    openai_no_think_prompt: bool = True,
    openai_disable_thinking: bool = True,
    openai_send_max_new_tokens_param: bool = True,
    ground_truth: str = "待审核",
) -> list[Path]:
    """执行 baseline 主流程（兼容接口，仅返回输出文件列表）。"""

    return run_baseline_with_diagnostics(
        input_path=input_path,
        output_dir=output_dir,
        risk_info_path=risk_info_path,
        llm_settings=llm_settings,
        logger=logger,
        ablation_no_rules=ablation_no_rules,
        ablation_no_span_offset=ablation_no_span_offset,
        schema_retry_limit=schema_retry_limit,
        max_new_tokens=max_new_tokens,
        openai_no_think_prompt=openai_no_think_prompt,
        openai_disable_thinking=openai_disable_thinking,
        openai_send_max_new_tokens_param=openai_send_max_new_tokens_param,
        ground_truth=ground_truth,
    ).outputs


def run_baseline_with_diagnostics(
    *,
    input_path: Path,
    output_dir: Path,
    risk_info_path: Path,
    llm_settings: LLMSettings,
    logger: Logger,
    ablation_no_rules: bool = False,
    ablation_no_span_offset: bool = False,
    schema_retry_limit: int = 1,
    max_new_tokens: int = 512,
    openai_no_think_prompt: bool = True,
    openai_disable_thinking: bool = True,
    openai_send_max_new_tokens_param: bool = True,
    ground_truth: str = "待审核",
) -> BaselineRunResult:
    """执行 baseline 审查并输出诊断、指标。"""

    if schema_retry_limit < 0 or schema_retry_limit > MAX_SCHEMA_RETRY_LIMIT:
        raise ValueError(
            f"schema_retry_limit 必须在 [0, {MAX_SCHEMA_RETRY_LIMIT}] 范围内。"
        )

    files = discover_chunk_files(input_path)
    if not files:
        raise FileNotFoundError(f"未找到待审查文件: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    risk_rules = load_risk_rules(risk_info_path)
    ground_truth_value = ground_truth.strip() or "待审核"

    reviewer = _ChunkBaselineReviewer(
        risk_rules=risk_rules,
        settings=llm_settings,
        logger=logger,
        ablation_no_rules=ablation_no_rules,
        schema_retry_limit=schema_retry_limit,
        max_new_tokens=max_new_tokens,
        openai_no_think_prompt=openai_no_think_prompt,
        openai_disable_thinking=openai_disable_thinking,
        openai_send_max_new_tokens_param=openai_send_max_new_tokens_param,
    )

    logger.info(
        "baseline_start: file_count=%s, input=%s, output=%s, risk_rule_count=%s, "
        "ablation_no_rules=%s, ablation_no_span_offset=%s, schema_retry_limit=%s, max_new_tokens=%s, "
        "openai_no_think_prompt=%s, openai_disable_thinking=%s, "
        "openai_send_max_new_tokens_param=%s, ground_truth=%s"
        % (
            len(files),
            input_path,
            output_dir,
            len(risk_rules),
            ablation_no_rules,
            ablation_no_span_offset,
            schema_retry_limit,
            max_new_tokens,
            openai_no_think_prompt,
            openai_disable_thinking,
            openai_send_max_new_tokens_param,
            ground_truth_value,
        )
    )

    outputs: list[Path] = []
    diagnostics: list[BaselineTaskDiagnostic] = []
    failures: list[str] = []
    total_chunk_count = 0
    total_task_count = 0
    emitted_item_count = 0
    span_matched_count = 0

    for file_path in files:
        try:
            logger.info(f"baseline_file_start: file={file_path}")
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            chunks = _extract_chunks(payload, file_path)
            total_chunk_count += len(chunks)

            doc_id = str(payload.get("doc_id", file_path.stem))
            source_file = str(payload.get("source_file", file_path.name))
            tasks = _build_baseline_tasks(
                chunks=chunks,
                doc_id=doc_id,
                source_file=source_file,
                risk_rules=risk_rules,
            )
            total_task_count += len(tasks)
            task_results = reviewer.review_many(tasks)

            file_items: list[dict[str, object]] = []
            counters: dict[tuple[str, str, str], int] = {}

            for task, task_result in zip(tasks, task_results):
                diagnostics.append(task_result.diagnostic)
                for raw in task_result.items:
                    span_offset = None
                    if not ablation_no_span_offset:
                        span_offset = _find_span_offset(raw["span"], task.content)
                    if span_offset is not None:
                        span_matched_count += 1

                    counter_key = (
                        task.doc_id,
                        str(task.chunk_id),
                        raw["risk_type"],
                    )
                    next_idx = counters.get(counter_key, 0) + 1
                    counters[counter_key] = next_idx
                    risk_id = f"{task.doc_id}#c{task.chunk_id}#rt{raw['risk_type']}#{next_idx}"

                    file_items.append(
                        {
                            "risk_id": risk_id,
                            "source_file": task.source_file,
                            "chunk_id": task.chunk_id,
                            "risk_type": raw["risk_type"],
                            "result": raw["result"],
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
                "rule_version": "baseline_risk_info_v1",
                "ablation_no_rules": ablation_no_rules,
                "ablation_no_span_offset": ablation_no_span_offset,
                "task_count": len(tasks),
                "review_item_count": len(file_items),
                "risk_rule_count": len(risk_rules),
            }
            output_payload["review_items"] = file_items

            output_path = _build_output_path(file_path, output_dir)
            output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            outputs.append(output_path)

            logger.info(
                "baseline_file_done: file=%s, chunk_count=%s, task_count=%s, review_item_count=%s, output=%s"
                % (file_path.name, len(chunks), len(tasks), len(file_items), output_path)
            )
        except Exception as exc:  # noqa: BLE001
            message = f"baseline_file_failed: file={file_path}, error={exc}"
            logger.error(message)
            failures.append(message)

    trace_path = output_dir / "baseline_trace.jsonl"
    _write_diagnostics(trace_path, diagnostics)

    metrics = _compute_metrics(
        diagnostics=diagnostics,
        file_count=len(outputs),
        chunk_count=total_chunk_count,
        task_count=total_task_count,
        emitted_item_count=emitted_item_count,
        span_matched_count=span_matched_count,
        risk_rule_count=len(risk_rules),
        ablation_no_rules=ablation_no_rules,
        ablation_no_span_offset=ablation_no_span_offset,
    )
    metrics_path = output_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"baseline_trace_written: {trace_path}")
    logger.info(f"baseline_metrics_written: {metrics_path}")

    if failures:
        raise RuntimeError("baseline 流程存在失败文件，请检查日志。")

    logger.info(
        "baseline_done: success_count=%s, task_count=%s, review_item_count=%s"
        % (len(outputs), total_task_count, emitted_item_count)
    )
    return BaselineRunResult(
        outputs=outputs,
        diagnostics=diagnostics,
        metrics=metrics,
        trace_path=trace_path,
        metrics_path=metrics_path,
    )


def load_risk_rules(risk_info_path: Path) -> list[dict[str, str]]:
    """从 risk_info.csv 读取 baseline 规则列表。"""

    if not risk_info_path.exists():
        raise FileNotFoundError(f"risk_info 文件不存在: {risk_info_path}")

    with risk_info_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"risk_info 文件为空: {risk_info_path}")

    rules: list[BaselineRule] = []
    seen_types: set[str] = set()
    for row in rows:
        risk_type = str(row.get("审查类型", "")).strip() or str(row.get("审查要点", "")).strip()
        rule_text = _compact_text(str(row.get("审查规则", "")))
        if not risk_type or not rule_text:
            continue
        if risk_type in seen_types:
            continue
        seen_types.add(risk_type)
        rules.append(BaselineRule(risk_type=risk_type, rule_text=rule_text))

    if not rules:
        raise ValueError(f"risk_info 文件未解析出有效审查规则: {risk_info_path}")
    return [item.to_dict() for item in rules]


def _extract_chunks(payload: object, source: Path) -> list[dict]:
    """提取并校验 chunk 列表。"""

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
            chunk_id = chunk.get("chunk_id", index)
            raise ValueError(f"chunk content 非字符串: {source}, chunk_id={chunk_id}")
    return chunks


def _build_output_path(input_file: Path, output_dir: Path) -> Path:
    if input_file.name.endswith(".chunks.json"):
        filename = input_file.name.replace(".chunks.json", ".review.json")
    else:
        filename = f"{input_file.stem}.review.json"
    return output_dir / filename


def _build_baseline_tasks(
    *,
    chunks: list[dict],
    doc_id: str,
    source_file: str,
    risk_rules: list[dict[str, str]],
) -> list[BaselineTask]:
    tasks: list[BaselineTask] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", index)
        tasks.append(
            BaselineTask(
                source_file=source_file,
                doc_id=doc_id,
                chunk_id=chunk_id,
                content=chunk["content"],
                risk_rules=risk_rules,
            )
        )
    return tasks


def _write_diagnostics(path: Path, diagnostics: list[BaselineTaskDiagnostic]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in diagnostics:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def _compute_metrics(
    *,
    diagnostics: list[BaselineTaskDiagnostic],
    file_count: int,
    chunk_count: int,
    task_count: int,
    emitted_item_count: int,
    span_matched_count: int,
    risk_rule_count: int,
    ablation_no_rules: bool,
    ablation_no_span_offset: bool,
) -> BaselineRunMetrics:
    llm_rows = [item for item in diagnostics if item.llm_called]
    schema_valid_rate = _safe_rate(sum(1 for item in llm_rows if item.schema_valid), len(llm_rows))
    llm_error_rate = _safe_rate(sum(1 for item in llm_rows if item.error_code is not None), len(llm_rows))
    span_match_rate = _safe_rate(span_matched_count, emitted_item_count)
    avg_token_in = _safe_mean([item.token_in for item in llm_rows])
    avg_token_out = _safe_mean([item.token_out for item in llm_rows])
    avg_total_token = _safe_mean([item.total_tokens for item in llm_rows])
    avg_latency_ms = _safe_mean([item.latency_ms for item in llm_rows])
    schema_retry_count = sum(item.schema_retries for item in llm_rows)

    return BaselineRunMetrics(
        file_count=file_count,
        chunk_count=chunk_count,
        task_count=task_count,
        emitted_item_count=emitted_item_count,
        llm_called_count=len(llm_rows),
        schema_valid_rate=schema_valid_rate,
        llm_error_rate=llm_error_rate,
        span_match_rate=span_match_rate,
        avg_token_in=avg_token_in,
        avg_token_out=avg_token_out,
        avg_total_token=avg_total_token,
        avg_latency_ms=avg_latency_ms,
        schema_retry_count=schema_retry_count,
        risk_rule_count=risk_rule_count,
        ablation_no_rules=ablation_no_rules,
        ablation_no_span_offset=ablation_no_span_offset,
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _safe_mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(item) for item in values)) / float(len(values))


def _compact_text(text: str) -> str:
    return " ".join(text.split())


def _find_span_offset(span: str, content: str) -> list[int] | None:
    """在 chunk 文本中定位 span 偏移，返回 [start, end)。"""

    if not span:
        return None
    start = content.find(span)
    if start < 0:
        return None
    return [start, start + len(span)]


def _append_no_think_suffix(user_prompt: str, *, enable: bool) -> str:
    """按需给用户提示词追加 `/no_think`。"""

    if not enable:
        return user_prompt
    if "/no_think" in user_prompt:
        return user_prompt
    return f"{user_prompt.rstrip()}\n/no_think"


def _build_openai_extra_payload(
    *,
    max_new_tokens: int,
    send_max_new_tokens_param: bool,
    disable_thinking: bool,
) -> dict[str, object]:
    """构建 OpenAI-compatible 额外请求参数。"""

    payload: dict[str, object] = {}
    if send_max_new_tokens_param:
        token_limit = int(max_new_tokens)
        payload["max_new_tokens"] = token_limit
        payload["max_completion_tokens"] = token_limit
    if disable_thinking:
        payload["enable_thinking"] = False
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


class _ChunkBaselineReviewer:
    """基于 LLM 的 chunk 级 baseline 审查器。"""

    def __init__(
        self,
        *,
        risk_rules: list[dict[str, str]],
        settings: LLMSettings,
        logger: Logger,
        ablation_no_rules: bool,
        schema_retry_limit: int,
        max_new_tokens: int,
        openai_no_think_prompt: bool,
        openai_disable_thinking: bool,
        openai_send_max_new_tokens_param: bool,
    ) -> None:
        self._risk_rules = list(risk_rules)
        self._risk_types = [str(item.get("risk_type", "")).strip() for item in self._risk_rules]
        self._risk_types = [item for item in self._risk_types if item]
        self._risk_type_set = set(self._risk_types)
        self._settings = settings
        self._logger = logger
        self._ablation_no_rules = ablation_no_rules
        self._schema_retry_limit = schema_retry_limit
        self._max_new_tokens = max(128, int(max_new_tokens))
        self._force_no_think_prompt = openai_no_think_prompt
        self._openai_extra_payload = _build_openai_extra_payload(
            max_new_tokens=self._max_new_tokens,
            send_max_new_tokens_param=openai_send_max_new_tokens_param,
            disable_thinking=openai_disable_thinking,
        )
        self._enabled = settings.enabled

        if not self._enabled:
            self._logger.info("baseline llm 已禁用：llm.enabled=false")
            self._client = None
            return

        if not settings.model:
            self._logger.error("baseline llm 降级：缺少 LLM_MODEL，当前审查任务将不触发 LLM")
            self._enabled = False
            self._client = None
            return

        if not settings.api_key:
            self._logger.info("LLM_API_KEY 为空，将按无鉴权方式调用 OpenAI 兼容接口。")

        self._client = OpenAICompatibleClient(settings)
        self._logger.info(
            "baseline_llm_options: max_new_tokens=%s, no_think_prompt=%s, "
            "disable_thinking=%s, send_max_new_tokens_param=%s"
            % (
                self._max_new_tokens,
                self._force_no_think_prompt,
                bool(self._openai_extra_payload.get("enable_thinking") is False),
                bool("max_new_tokens" in self._openai_extra_payload),
            )
        )

    def review_many(self, tasks: list[BaselineTask]) -> list[_TaskReviewResult]:
        """批量执行审查任务。"""

        if not tasks:
            return []
        return run_tasks(
            tasks,
            self.review_one,
            concurrent_enabled=self._settings.concurrent_enabled,
            max_concurrency=self._settings.max_concurrency,
        )

    def review_one(self, task: BaselineTask) -> _TaskReviewResult:
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

        last_schema_error = ""
        for schema_retry_index in range(self._schema_retry_limit + 1):
            messages = self._build_messages(
                task=task,
                schema_feedback=last_schema_error if schema_retry_index > 0 else None,
            )
            try:
                response = self._client.chat_with_metadata(
                    messages,
                    max_tokens=self._max_new_tokens,
                    extra_payload=self._openai_extra_payload,
                )
            except LLMClientError as exc:
                self._logger.error(
                    "baseline_llm_error: file=%s, chunk_id=%s, retries=%s, error_code=%s, error=%s"
                    % (task.source_file, task.chunk_id, exc.retries, exc.error_code, exc)
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
                parsed_items = self._parse_response(response.content)
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
                        "baseline_schema_retry: file=%s, chunk_id=%s, request_id=%s, attempt=%s/%s, error=%s"
                        % (
                            task.source_file,
                            task.chunk_id,
                            response.request_id or "-",
                            schema_retry_index + 1,
                            self._schema_retry_limit,
                            last_schema_error,
                        )
                    )
                    continue

                self._logger.error(
                    "baseline_schema_invalid: file=%s, chunk_id=%s, request_id=%s, error=%s"
                    % (task.source_file, task.chunk_id, response.request_id or "-", exc)
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

        raise RuntimeError("baseline 审查任务异常：schema 重试循环提前结束。")

    def _build_messages(self, task: BaselineTask, schema_feedback: str | None = None) -> list[dict[str, str]]:
        output_schema_hint = (
            "输出 JSON 对象，字段为 items（数组）。每个 item 必须包含 risk_type、result、span、suggest。"
        )
        rules_section: str
        if self._ablation_no_rules:
            rules_section = "不提供规则文本，请按审查类型直接判断。"
        else:
            lines = [
                f"- 审查类型：{item['risk_type']} | 审查规则：{item['rule_text']}"
                for item in task.risk_rules
            ]
            rules_section = "审查类型与规则列表：\n" + "\n".join(lines)

        retry_section = ""
        if schema_feedback:
            retry_section = (
                "\n上一次输出未通过格式校验，请纠正后重新输出。\n"
                f"失败原因：{schema_feedback}\n"
            )

        user_prompt = (
            "请对给定合同片段执行 baseline 审查。\n"
            f"{output_schema_hint}\n"
            "若无任何命中，返回 {\"items\": []}。\n"
            "约束：\n"
            "1) risk_type 必须来自给定审查类型列表；\n"
            "2) result 只能是“合格”或“不合格”；\n"
            "3) span 必须是合同片段中的连续原文子串，不得改写；\n"
            "4) suggest 需给出明确建议；\n"
            "5) 只输出 JSON，不要输出额外文本。"
            f"{retry_section}\n"
            f"{rules_section}\n\n"
            "合同片段：\n"
            f"{task.content}"
        )
        user_prompt = _append_no_think_suffix(user_prompt, enable=self._force_no_think_prompt)

        return [
            {
                "role": "system",
                "content": "你是合同规则审查助手。严格输出 JSON，并遵守字段约束。",
            },
            {"role": "user", "content": user_prompt},
        ]

    def _parse_response(self, response_text: str) -> list[dict[str, str]]:
        payload = _parse_json_object(response_text)
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError(f"模型输出缺少 items 数组: {payload}")

        parsed: list[dict[str, str]] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"items[{index}] 不是对象: {item}")

            risk_type = str(item.get("risk_type", "")).strip()
            result = str(item.get("result", "")).strip()
            span = str(item.get("span", "")).strip()
            suggest = str(item.get("suggest", "")).strip()
            if risk_type not in self._risk_type_set:
                raise ValueError(f"items[{index}].risk_type 非候选值: {risk_type}")
            if result not in {"合格", "不合格"}:
                raise ValueError(f"items[{index}].result 非法: {result}")
            if not span:
                raise ValueError(f"items[{index}].span 为空")
            if not suggest:
                raise ValueError(f"items[{index}].suggest 为空")

            parsed.append(
                {
                    "risk_type": risk_type,
                    "result": result,
                    "span": span,
                    "suggest": suggest,
                }
            )
        return parsed

    @staticmethod
    def _build_diag(
        *,
        task: BaselineTask,
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
    ) -> BaselineTaskDiagnostic:
        return BaselineTaskDiagnostic(
            source_file=task.source_file,
            doc_id=task.doc_id,
            chunk_id=task.chunk_id,
            risk_rule_count=len(task.risk_rules),
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
