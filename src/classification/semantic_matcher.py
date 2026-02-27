"""语义匹配器。"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from src.llm import LLMSettings, OpenAICompatibleClient, run_tasks
from src.tools.logger import Logger

from .risk_catalog import RiskCatalog


_JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*\}")


@dataclass(frozen=True, slots=True)
class SemanticTask:
    """单个 chunk 的语义匹配任务。"""

    chunk_id: int | str
    text: str


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
        """返回语义匹配到的风险类型。"""

        if not self._enabled or self._client is None:
            return []

        messages = self._build_messages(text)
        attempts = self._settings.max_retries + 1

        for attempt in range(1, attempts + 1):
            try:
                raw = self._client.chat(messages)
                parsed = self._parse_response(raw)
                return self._catalog.normalize_risks(parsed)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "语义匹配失败: file=%s, chunk_id=%s, attempt=%s/%s, error=%s"
                    % (source_file, chunk_id, attempt, attempts, exc)
                )
                if attempt < attempts:
                    time.sleep(attempt)

        self._logger.error(
            "语义匹配降级为关键词结果: file=%s, chunk_id=%s" % (source_file, chunk_id)
        )
        return []

    def match_many(self, tasks: list[SemanticTask], source_file: str) -> list[list[str]]:
        """批量语义匹配，默认并发执行。"""

        if not tasks:
            return []

        def worker(task: SemanticTask) -> list[str]:
            return self.match(task.text, task.chunk_id, source_file)

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
