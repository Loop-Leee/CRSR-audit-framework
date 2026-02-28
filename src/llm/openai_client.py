"""OpenAI 兼容客户端。"""

from __future__ import annotations

import json
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import LLMSettings


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class LLMChatResponse:
    """LLM 响应与观测元数据。"""

    content: str
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


@dataclass(frozen=True, slots=True)
class LLMTokenUsage:
    """LLM token 用量明细。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    total_tokens_estimated: bool


class LLMClientError(RuntimeError):
    """LLM 调用失败异常，携带可观测元数据。"""

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        request_id: str = "",
        retries: int = 0,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.request_id = request_id
        self.retries = retries


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
        self._cache_enabled = settings.cache_enabled
        self._cache_path = settings.cache_path
        self._cache_lock = threading.Lock()
        self._cache: dict[str, LLMChatResponse] = {}
        if self._cache_enabled:
            self._load_cache()

    def chat(self, messages: list[dict[str, str]]) -> str:
        """调用模型并返回首个文本内容。"""

        return self.chat_with_metadata(messages).content

    def chat_with_metadata(self, messages: list[dict[str, str]]) -> LLMChatResponse:
        """调用模型并返回文本与调用元数据。"""

        cache_key = self._build_cache_key(messages)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        attempts = self._settings.max_retries + 1
        last_error: LLMClientError | None = None
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            request_id = ""
            try:
                body, request_id = self._send_request(messages)
                parsed = self._parse_response_body(body)
                content = self._extract_content(parsed, body)
                usage = self._extract_tokens(parsed)
                response = LLMChatResponse(
                    content=content,
                    token_in=usage.prompt_tokens,
                    token_out=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cached_tokens=usage.cached_tokens,
                    reasoning_tokens=usage.reasoning_tokens,
                    total_tokens_estimated=usage.total_tokens_estimated,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    request_id=request_id or str(parsed.get("id", "")),
                    retries=attempt - 1,
                    error_code=None,
                    cached=False,
                )
                self._save_cache(cache_key, response)
                return response
            except LLMClientError as exc:
                last_error = LLMClientError(
                    str(exc),
                    error_code=exc.error_code,
                    request_id=exc.request_id or request_id,
                    retries=attempt - 1,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = LLMClientError(
                    f"LLM 返回解析失败: {exc}",
                    error_code="parse_error",
                    request_id=request_id,
                    retries=attempt - 1,
                )

            if attempt < attempts:
                time.sleep(attempt)

        if last_error is None:
            raise LLMClientError("LLM 调用失败：未知错误。", error_code="unknown")
        raise last_error

    def _send_request(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        """执行单次 HTTP 请求。"""

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
                request_id = str(response.headers.get("x-request-id", ""))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            request_id = str(exc.headers.get("x-request-id", "")) if exc.headers else ""
            raise LLMClientError(
                f"LLM HTTPError: status={exc.code}, body={detail}",
                error_code=f"http_{exc.code}",
                request_id=request_id,
            ) from exc
        except URLError as exc:
            raise LLMClientError(
                f"LLM URLError: {exc.reason}",
                error_code="network_error",
            ) from exc

        return body, request_id

    @staticmethod
    def _parse_response_body(body: str) -> dict[str, Any]:
        try:
            parsed: dict[str, Any] = json.loads(body)
            return parsed
        except Exception as exc:  # noqa: BLE001
            raise ValueError(body[:300]) from exc

    @staticmethod
    def _extract_content(parsed: dict[str, Any], body: str) -> str:
        try:
            return str(parsed["choices"][0]["message"]["content"])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"缺少 choices.message.content: {body[:300]}") from exc

    @staticmethod
    def _extract_tokens(parsed: dict[str, Any]) -> LLMTokenUsage:
        usage = parsed.get("usage")
        if not isinstance(usage, dict):
            return LLMTokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cached_tokens=0,
                reasoning_tokens=0,
                total_tokens_estimated=False,
            )

        prompt_tokens = _as_int(usage.get("prompt_tokens"))
        completion_tokens = _as_int(usage.get("completion_tokens"))
        raw_total = usage.get("total_tokens")
        if raw_total is None:
            total_tokens = prompt_tokens + completion_tokens
            total_tokens_estimated = True
        else:
            total_tokens = _as_int(raw_total)
            total_tokens_estimated = False

        prompt_details = usage.get("prompt_tokens_details")
        completion_details = usage.get("completion_tokens_details")
        cached_tokens = 0
        reasoning_tokens = 0
        if isinstance(prompt_details, dict):
            cached_tokens = _as_int(prompt_details.get("cached_tokens"))
        if isinstance(completion_details, dict):
            reasoning_tokens = _as_int(completion_details.get("reasoning_tokens"))

        return LLMTokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens_estimated=total_tokens_estimated,
        )

    def _build_cache_key(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "base_url": self._settings.base_url.rstrip("/"),
            "model": self._settings.model,
            "temperature": self._settings.temperature,
            "messages": messages,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        path = Path(self._cache_path)
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                key = str(payload.get("cache_key", ""))
                response = payload.get("response", {})
                if not key or not isinstance(response, dict):
                    continue
                self._cache[key] = LLMChatResponse(
                    content=str(response.get("content", "")),
                    token_in=int(response.get("token_in", 0) or 0),
                    token_out=int(response.get("token_out", 0) or 0),
                    total_tokens=int(
                        response.get(
                            "total_tokens",
                            (response.get("token_in", 0) or 0) + (response.get("token_out", 0) or 0),
                        )
                        or 0
                    ),
                    cached_tokens=int(response.get("cached_tokens", 0) or 0),
                    reasoning_tokens=int(response.get("reasoning_tokens", 0) or 0),
                    total_tokens_estimated=bool(
                        response.get("total_tokens_estimated", "total_tokens" not in response)
                    ),
                    latency_ms=float(response.get("latency_ms", 0.0) or 0.0),
                    request_id=str(response.get("request_id", "")),
                    retries=int(response.get("retries", 0) or 0),
                    error_code=None,
                    cached=False,
                )
            except Exception:  # noqa: BLE001
                continue

    def _get_cached(self, cache_key: str) -> LLMChatResponse | None:
        if not self._cache_enabled:
            return None
        cached = self._cache.get(cache_key)
        if cached is None:
            return None
        return replace(
            cached,
            cached=True,
            retries=0,
            latency_ms=0.0,
            error_code=None,
        )

    def _save_cache(self, cache_key: str, response: LLMChatResponse) -> None:
        if not self._cache_enabled:
            return

        with self._cache_lock:
            self._cache[cache_key] = response
            path = Path(self._cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            cache_record = {
                "cache_key": cache_key,
                "response": {
                    "content": response.content,
                    "token_in": response.token_in,
                    "token_out": response.token_out,
                    "total_tokens": response.total_tokens,
                    "cached_tokens": response.cached_tokens,
                    "reasoning_tokens": response.reasoning_tokens,
                    "total_tokens_estimated": response.total_tokens_estimated,
                    "latency_ms": response.latency_ms,
                    "request_id": response.request_id,
                    "retries": response.retries,
                },
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(cache_record, ensure_ascii=False) + "\n")

    def chat_many(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        """批量调用 LLM，默认按配置并发执行。"""

        return run_tasks(
            messages_batch,
            self.chat,
            concurrent_enabled=self._settings.concurrent_enabled,
            max_concurrency=self._settings.max_concurrency,
        )

    def chat_many_with_metadata(self, messages_batch: list[list[dict[str, str]]]) -> list[LLMChatResponse]:
        """批量调用 LLM 并返回观测元数据。"""

        return run_tasks(
            messages_batch,
            self.chat_with_metadata,
            concurrent_enabled=self._settings.concurrent_enabled,
            max_concurrency=self._settings.max_concurrency,
        )


def _as_int(value: object) -> int:
    """容错读取整数值。"""

    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
