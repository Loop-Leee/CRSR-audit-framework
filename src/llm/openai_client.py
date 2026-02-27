"""OpenAI 兼容客户端。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import LLMSettings


T = TypeVar("T")
R = TypeVar("R")


def run_tasks(
    items: list[T],
    worker: Callable[[T], R],
    *,
    concurrent_enabled: bool,
    max_concurrency: int,
) -> list[R]:
    """按输入顺序执行任务，支持并发与串行两种模式。"""

    if not items:
        return []

    if not concurrent_enabled or max_concurrency <= 1 or len(items) == 1:
        return [worker(item) for item in items]

    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        future_to_index = {pool.submit(worker, item): idx for idx, item in enumerate(items)}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    if any(item is None for item in results):
        raise RuntimeError("并发任务执行异常：存在未返回结果的任务。")
    return [item for item in results if item is not None]


class OpenAICompatibleClient:
    """调用 OpenAI 兼容 chat/completions 接口。"""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings

    def chat(self, messages: list[dict[str, str]]) -> str:
        """调用模型并返回首个文本内容。"""

        endpoint = self._settings.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self._settings.model,
            "temperature": self._settings.temperature,
            "messages": messages,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            endpoint,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self._settings.api_key:
            request.add_header("Authorization", f"Bearer {self._settings.api_key}")

        try:
            with urlopen(request, timeout=self._settings.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            raise RuntimeError(f"LLM HTTPError: status={exc.code}, body={detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM URLError: {exc.reason}") from exc

        try:
            parsed: dict[str, Any] = json.loads(body)
            return str(parsed["choices"][0]["message"]["content"])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LLM 返回解析失败: {body[:300]}") from exc

    def chat_many(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        """批量调用 LLM，默认按配置并发执行。"""

        return run_tasks(
            messages_batch,
            self.chat,
            concurrent_enabled=self._settings.concurrent_enabled,
            max_concurrency=self._settings.max_concurrency,
        )
