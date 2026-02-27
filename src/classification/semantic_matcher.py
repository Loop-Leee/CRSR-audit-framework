"""语义匹配器。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.llm import LLMClientError, LLMSettings, OpenAICompatibleClient, run_tasks
from src.tools.logger import Logger

from .risk_catalog import RiskCatalog


_JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*\}")


@dataclass(frozen=True, slots=True)
class SemanticTask:
    """单个 chunk 的语义匹配任务。"""

    chunk_id: int | str
    text: str


@dataclass(frozen=True, slots=True)
class SemanticTrace:
    """语义匹配的调用元数据。"""

    llm_called: bool
    schema_valid: bool
    token_in: int
    token_out: int
    latency_ms: float
    request_id: str
    retries: int
    error_code: str | None
    cached: bool


@dataclass(frozen=True, slots=True)
class SemanticMatchResult:
    """语义匹配结果与诊断信息。"""

    risks: list[str]
    trace: SemanticTrace


class SemanticMatcher:
    """基于大模型进行风险类型语义匹配。"""

    def __init__(self, catalog: RiskCatalog, settings: LLMSettings, logger: Logger) -> None:
        self._catalog = catalog
        self._settings = settings
        self._logger = logger
        self._enabled = settings.enabled

        if not self._enabled:
            self._logger.info("语义匹配已禁用：llm.enabled=false")
            self._client = None
            return

        if not settings.model:
            self._logger.error("语义匹配降级：缺少 LLM_MODEL（仅保留关键词匹配结果）")
            self._enabled = False
            self._client = None
            return

        if not settings.api_key:
            self._logger.info("LLM_API_KEY 为空，将按无鉴权方式调用 OpenAI 兼容接口。")

        self._client = OpenAICompatibleClient(settings)

    def match(self, text: str, chunk_id: int | str, source_file: str) -> list[str]:
        """返回语义匹配到的风险类型（兼容旧接口）。"""

        return self.match_with_trace(text, chunk_id, source_file).risks

    def match_with_trace(self, text: str, chunk_id: int | str, source_file: str) -> SemanticMatchResult:
        """返回语义匹配风险类型与调用元数据。"""

        if not self._enabled or self._client is None:
            return SemanticMatchResult(
                risks=[],
                trace=SemanticTrace(
                    llm_called=False,
                    schema_valid=True,
                    token_in=0,
                    token_out=0,
                    latency_ms=0.0,
                    request_id="",
                    retries=0,
                    error_code="llm_disabled",
                    cached=False,
                ),
            )

        messages = self._build_messages(text)
        try:
            response = self._client.chat_with_metadata(messages)
            try:
                parsed = self._parse_response(response.content)
                return SemanticMatchResult(
                    risks=self._catalog.normalize_risks(parsed),
                    trace=SemanticTrace(
                        llm_called=True,
                        schema_valid=True,
                        token_in=response.token_in,
                        token_out=response.token_out,
                        latency_ms=response.latency_ms,
                        request_id=response.request_id,
                        retries=response.retries,
                        error_code=None,
                        cached=response.cached,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "语义匹配解析失败: file=%s, chunk_id=%s, request_id=%s, error=%s"
                    % (source_file, chunk_id, response.request_id, exc)
                )
                return SemanticMatchResult(
                    risks=[],
                    trace=SemanticTrace(
                        llm_called=True,
                        schema_valid=False,
                        token_in=response.token_in,
                        token_out=response.token_out,
                        latency_ms=response.latency_ms,
                        request_id=response.request_id,
                        retries=response.retries,
                        error_code="invalid_schema",
                        cached=response.cached,
                    ),
                )
        except LLMClientError as exc:
            self._logger.error(
                "语义匹配失败: file=%s, chunk_id=%s, retries=%s, error_code=%s, error=%s"
                % (source_file, chunk_id, exc.retries, exc.error_code, exc)
            )
            self._logger.error(
                "语义匹配降级为关键词结果: file=%s, chunk_id=%s" % (source_file, chunk_id)
            )
            return SemanticMatchResult(
                risks=[],
                trace=SemanticTrace(
                    llm_called=True,
                    schema_valid=False,
                    token_in=0,
                    token_out=0,
                    latency_ms=0.0,
                    request_id=exc.request_id,
                    retries=exc.retries,
                    error_code=exc.error_code or "llm_error",
                    cached=False,
                ),
            )

    def match_many(self, tasks: list[SemanticTask], source_file: str) -> list[list[str]]:
        """批量语义匹配，默认并发执行。"""

        return [item.risks for item in self.match_many_with_trace(tasks, source_file)]

    def match_many_with_trace(self, tasks: list[SemanticTask], source_file: str) -> list[SemanticMatchResult]:
        """批量语义匹配并返回调用元数据。"""

        if not tasks:
            return []

        def worker(task: SemanticTask) -> SemanticMatchResult:
            return self.match_with_trace(task.text, task.chunk_id, source_file)

        return run_tasks(
            tasks,
            worker,
            concurrent_enabled=self._settings.concurrent_enabled,
            max_concurrency=self._settings.max_concurrency,
        )

    def _build_messages(self, text: str) -> list[dict[str, str]]:
        catalog_summary = self._catalog.prompt_summary()
        user_prompt = (
            "请根据合同分段文本判断应执行哪些审查类型。\n"
            "输出必须是 JSON 对象，且仅包含字段 risk_type（字符串数组）。\n"
            "risk_type 中的值只能从候选列表中选择，不要输出额外字段。\n"
            "如果没有明显匹配，返回 {\"risk_type\": []}。\n\n"
            "候选审查类型：\n"
            f"{catalog_summary}\n\n"
            "合同分段文本：\n"
            f"{text}"
        )

        return [
            {
                "role": "system",
                "content": "你是合同审查分类助手，负责将文本映射到审查类型。只输出 JSON。",
            },
            {"role": "user", "content": user_prompt},
        ]

    def _parse_response(self, response_text: str) -> list[str]:
        """容错解析模型输出。"""

        try:
            payload = json.loads(response_text)
            return self._extract_risks(payload)
        except json.JSONDecodeError:
            pass

        matched = _JSON_OBJECT_PATTERN.search(response_text)
        if not matched:
            raise ValueError(f"模型输出不是 JSON: {response_text[:240]}")

        payload = json.loads(matched.group(0))
        return self._extract_risks(payload)

    @staticmethod
    def _extract_risks(payload: object) -> list[str]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, str)]

        if isinstance(payload, dict):
            risks = payload.get("risk_type", [])
            if isinstance(risks, list):
                return [item for item in risks if isinstance(item, str)]

        raise ValueError(f"模型输出缺少 risk_type 数组: {payload}")
