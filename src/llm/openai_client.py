"""OpenAI 兼容客户端与缓存管理。"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.tools.logger import Logger, get_logger

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
        """初始化异常对象。

        Args:
            message: 错误描述。
            error_code: 结构化错误码（如 `http_500`、`network_error`）。
            request_id: 上游请求 ID（若可用）。
            retries: 已执行重试次数。
        """

        super().__init__(message)
        self.error_code = error_code
        self.request_id = request_id
        self.retries = retries


@dataclass(slots=True)
class LLMCacheStats:
    """LLM 缓存统计快照。

    Args:
        enabled: 缓存是否启用（消融开关）。
        load_line_count: 启动加载时读取的缓存行数。
        load_valid_count: 启动加载时成功解析的记录数。
        load_invalid_count: 启动加载时解析失败/不合法记录数。
        memory_entry_count: 当前内存缓存条目数。
        hit_count: 运行期缓存命中次数。
        miss_count: 运行期缓存未命中次数。
        write_count: 运行期成功落盘缓存次数。
        write_error_count: 运行期缓存落盘失败次数。
        bypass_count: 因缓存开关关闭而跳过的读/写次数。
    """

    enabled: bool
    load_line_count: int = 0
    load_valid_count: int = 0
    load_invalid_count: int = 0
    memory_entry_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    write_count: int = 0
    write_error_count: int = 0
    bypass_count: int = 0

    def to_dict(self) -> dict[str, int | bool]:
        """导出缓存统计为可序列化字典。

        Returns:
            dict[str, int | bool]: 结构化统计数据，便于日志与指标落盘。
        """

        return {
            "enabled": self.enabled,
            "load_line_count": self.load_line_count,
            "load_valid_count": self.load_valid_count,
            "load_invalid_count": self.load_invalid_count,
            "memory_entry_count": self.memory_entry_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "write_count": self.write_count,
            "write_error_count": self.write_error_count,
            "bypass_count": self.bypass_count,
        }


def run_tasks(
    items: list[T],
    worker: Callable[[T], R],
    *,
    concurrent_enabled: bool,
    max_concurrency: int,
) -> list[R]:
    """按输入顺序执行任务，支持并发与串行两种模式。

    Args:
        items: 输入任务列表。
        worker: 单任务处理函数。
        concurrent_enabled: 是否启用并发。
        max_concurrency: 最大并发线程数。

    Returns:
        list[R]: 与输入顺序一致的执行结果。

    Raises:
        RuntimeError: 并发执行后存在缺失结果时抛出。
    """

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


class LLMResponseCache:
    """LLM 响应缓存管理器（内存索引 + JSONL 持久化）。

    该类负责：
    1. 缓存命中读取与标记；
    2. 缓存写入与持久化记录；
    3. 启动时加载历史缓存并进行容错；
    4. 输出结构化缓存指标（命中率、写入量、加载异常等）。
    """

    def __init__(self, enabled: bool, cache_path: Path, logger: Logger) -> None:
        """初始化缓存管理器。

        Args:
            enabled: 缓存开关（消融开关）。
            cache_path: 缓存文件路径（JSONL）。
            logger: 模块日志对象。
        """

        self._enabled = enabled
        self._path = Path(cache_path)
        self._logger = logger
        self._lock = threading.Lock()
        self._entries: dict[str, LLMChatResponse] = {}
        self._stats = LLMCacheStats(enabled=enabled)
        if self._enabled:
            self._load_from_disk()
        else:
            self._logger.info("LLM 缓存已关闭：cache_enabled=false")

    def get(self, cache_key: str) -> LLMChatResponse | None:
        """按缓存键读取响应。

        Args:
            cache_key: 缓存键（SHA-256 十六进制字符串）。

        Returns:
            LLMChatResponse | None: 命中时返回缓存响应，未命中返回 `None`。
        """

        if not self._enabled:
            self._stats.bypass_count += 1
            return None

        with self._lock:
            cached = self._entries.get(cache_key)
            if cached is None:
                self._stats.miss_count += 1
                return None
            self._stats.hit_count += 1
            return replace(
                cached,
                cached=True,
                retries=0,
                latency_ms=0.0,
                error_code=None,
            )

    def set(self, cache_key: str, response: LLMChatResponse, key_meta: dict[str, Any]) -> None:
        """写入缓存并持久化到 JSONL。

        Args:
            cache_key: 缓存键。
            response: 要缓存的 LLM 响应。
            key_meta: 缓存键元数据（用于审计和排障）。
        """

        if not self._enabled:
            self._stats.bypass_count += 1
            return

        cache_record = self._build_cache_record(cache_key, response, key_meta)
        with self._lock:
            self._entries[cache_key] = response
            self._stats.memory_entry_count = len(self._entries)
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(cache_record, ensure_ascii=False) + "\n")
                self._stats.write_count += 1
            except OSError as exc:
                self._stats.write_error_count += 1
                self._logger.error(f"LLM 缓存写入失败: path={self._path}, error={exc}")

    def snapshot(self) -> dict[str, int | bool]:
        """返回当前缓存统计快照。

        Returns:
            dict[str, int | bool]: 结构化缓存统计数据。
        """

        with self._lock:
            self._stats.memory_entry_count = len(self._entries)
            return self._stats.to_dict()

    def _load_from_disk(self) -> None:
        """启动时加载 JSONL 缓存文件到内存。"""

        if not self._path.exists():
            self._logger.info(f"LLM 缓存文件不存在，跳过加载: {self._path}")
            return

        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            self._logger.error(f"LLM 缓存读取失败: path={self._path}, error={exc}")
            return

        loaded = 0
        invalid = 0
        for line in lines:
            self._stats.load_line_count += 1
            content = line.strip()
            if not content:
                invalid += 1
                continue
            try:
                payload = json.loads(content)
                key = str(payload.get("cache_key", ""))
                response_payload = payload.get("response")
                response = self._to_response(response_payload)
                if not key or response is None:
                    invalid += 1
                    continue
                self._entries[key] = response
                loaded += 1
            except Exception:  # noqa: BLE001
                invalid += 1

        self._stats.load_valid_count = loaded
        self._stats.load_invalid_count = invalid
        self._stats.memory_entry_count = len(self._entries)
        self._logger.info(
            "LLM 缓存加载完成: path=%s, lines=%s, valid=%s, invalid=%s, entries=%s"
            % (
                self._path,
                self._stats.load_line_count,
                loaded,
                invalid,
                len(self._entries),
            )
        )

    @staticmethod
    def _to_response(response_payload: object) -> LLMChatResponse | None:
        """将缓存记录里的 `response` 字段解析为 `LLMChatResponse`。

        Args:
            response_payload: 原始响应字段。

        Returns:
            LLMChatResponse | None: 解析成功的响应对象；失败返回 `None`。
        """

        if not isinstance(response_payload, dict):
            return None

        token_in = int(response_payload.get("token_in", 0) or 0)
        token_out = int(response_payload.get("token_out", 0) or 0)
        total_tokens = int(response_payload.get("total_tokens", token_in + token_out) or 0)
        return LLMChatResponse(
            content=str(response_payload.get("content", "")),
            token_in=token_in,
            token_out=token_out,
            total_tokens=total_tokens,
            cached_tokens=int(response_payload.get("cached_tokens", 0) or 0),
            reasoning_tokens=int(response_payload.get("reasoning_tokens", 0) or 0),
            total_tokens_estimated=bool(
                response_payload.get("total_tokens_estimated", "total_tokens" not in response_payload)
            ),
            latency_ms=float(response_payload.get("latency_ms", 0.0) or 0.0),
            request_id=str(response_payload.get("request_id", "")),
            retries=int(response_payload.get("retries", 0) or 0),
            error_code=None,
            cached=False,
        )

    @staticmethod
    def _build_cache_record(
        cache_key: str,
        response: LLMChatResponse,
        key_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """构造缓存持久化记录。

        Args:
            cache_key: 缓存键。
            response: LLM 响应对象。
            key_meta: 缓存键元数据。

        Returns:
            dict[str, Any]: 可直接写入 JSONL 的结构化缓存记录。
        """

        return {
            "event": "cache_write",
            "cache_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cache_key": cache_key,
            "key_meta": key_meta,
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


class OpenAICompatibleClient:
    """调用 OpenAI 兼容 `chat/completions` 接口。

    说明：
    - 支持自动重试（指数线性退避，按 `attempt` 秒休眠）；
    - 支持缓存命中（内存 + JSONL）；
    - 输出结构化元数据用于实验统计。
    """

    def __init__(self, settings: LLMSettings, logger: Logger | None = None) -> None:
        """初始化客户端。

        Args:
            settings: LLM 配置对象。
            logger: 可选模块日志对象；为空时自动创建 `llm` 日志器。
        """

        self._settings = settings
        self._logger = logger or get_logger("llm")
        self._cache = LLMResponseCache(
            enabled=settings.cache_enabled,
            cache_path=settings.cache_path,
            logger=self._logger,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """调用模型并返回文本内容。

        Args:
            messages: OpenAI Chat 消息列表。
            model: 可选模型名覆盖；为空时使用 `LLMSettings.model`。
            max_tokens: 可选最大生成 token 数。

        Returns:
            str: 模型输出文本。

        Raises:
            LLMClientError: 当请求失败且超过重试次数时抛出。
        """

        return self.chat_with_metadata(messages, model=model, max_tokens=max_tokens).content

    def chat_with_metadata(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMChatResponse:
        """调用模型并返回文本与调用元数据。

        Args:
            messages: OpenAI Chat 消息列表。
            model: 可选模型名覆盖；为空时使用 `LLMSettings.model`。
            max_tokens: 可选最大生成 token 数。

        Returns:
            LLMChatResponse: 包含文本、token、时延、缓存命中等信息。

        Raises:
            LLMClientError: 当请求失败且超过重试次数时抛出。
        """

        cache_key, key_meta = self._build_cache_key(messages, model=model, max_tokens=max_tokens)
        cached = self._cache.get(cache_key)
        if cached is not None:
            if _is_invalid_content(cached.content):
                self._logger.info("LLM 缓存命中但内容无效，忽略缓存并重新请求。")
            else:
                return cached

        attempts = self._settings.max_retries + 1
        last_error: LLMClientError | None = None
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            request_id = ""
            try:
                body, request_id = self._send_request(messages, model=model, max_tokens=max_tokens)
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
                if _is_invalid_content(response.content):
                    self._logger.error(
                        "LLM 返回空内容，跳过缓存: request_id=%s, token_out=%s, total_tokens=%s"
                        % (response.request_id, response.token_out, response.total_tokens)
                    )
                    return response
                self._cache.set(cache_key, response, key_meta)
                return response
            except LLMClientError as exc:
                self._logger.error(
                    "LLM 请求失败: attempt=%s/%s, error_code=%s, request_id=%s, detail=%s"
                    % (attempt, attempts, exc.error_code, exc.request_id or request_id, exc)
                )
                last_error = LLMClientError(
                    str(exc),
                    error_code=exc.error_code,
                    request_id=exc.request_id or request_id,
                    retries=attempt - 1,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.error(f"LLM 响应解析失败: attempt={attempt}/{attempts}, detail={exc}")
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

    def cache_stats(self) -> dict[str, int | bool]:
        """返回缓存统计快照。

        Returns:
            dict[str, int | bool]: 结构化缓存指标（命中、未命中、写入、绕过等）。
        """

        return self._cache.snapshot()

    def _send_request(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, str]:
        """执行单次 HTTP 请求。

        Args:
            messages: OpenAI Chat 消息列表。

        Returns:
            tuple[str, str]: `(body, request_id)`。

        Raises:
            LLMClientError: HTTP 或网络错误。
        """

        endpoint = self._settings.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model or self._settings.model,
            "temperature": self._settings.temperature,
            "messages": messages,
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
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
            reason = str(exc.reason)
            error_code = "timeout" if "timed out" in reason.lower() else "network_error"
            raise LLMClientError(
                f"LLM URLError: {reason}",
                error_code=error_code,
            ) from exc
        except TimeoutError as exc:
            raise LLMClientError(
                f"LLM TimeoutError: {exc}",
                error_code="timeout",
            ) from exc

        return body, request_id

    @staticmethod
    def _parse_response_body(body: str) -> dict[str, Any]:
        """解析响应 JSON。

        Args:
            body: HTTP 响应正文。

        Returns:
            dict[str, Any]: JSON 解析结果。

        Raises:
            ValueError: 响应无法解析为 JSON。
        """

        try:
            parsed: dict[str, Any] = json.loads(body)
            return parsed
        except Exception as exc:  # noqa: BLE001
            raise ValueError(body[:300]) from exc

    @staticmethod
    def _extract_content(parsed: dict[str, Any], body: str) -> str:
        """提取模型文本输出。

        Args:
            parsed: 已解析的响应 JSON。
            body: 原始响应文本（用于报错截断）。

        Returns:
            str: `choices[0].message.content` 内容。

        Raises:
            ValueError: 缺失标准响应字段。
        """

        try:
            choice0 = parsed["choices"][0]
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"缺少 choices 字段: {body[:300]}") from exc

        message = choice0.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                joined = "".join(parts).strip()
                if joined:
                    return joined
            msg_text = message.get("text")
            if isinstance(msg_text, str):
                return msg_text

        # 兼容部分 OpenAI-compatible 实现返回 choices[i].text。
        choice_text = choice0.get("text")
        if isinstance(choice_text, str):
            return choice_text

        return ""

    @staticmethod
    def _extract_tokens(parsed: dict[str, Any]) -> LLMTokenUsage:
        """提取 token 用量字段并做容错处理。

        Args:
            parsed: 已解析的响应 JSON。

        Returns:
            LLMTokenUsage: token 统计对象；缺失字段时返回 0 或估算值。
        """

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

    def _build_cache_key(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """生成缓存键与结构化键元数据。

        Args:
            messages: OpenAI Chat 消息列表。

        Returns:
            tuple[str, dict[str, Any]]:
            - 第一个值是缓存键（SHA-256）；
            - 第二个值是缓存键元数据（base_url/model/temperature/message_count）。
        """

        payload = {
            "base_url": self._settings.base_url.rstrip("/"),
            "model": model or self._settings.model,
            "temperature": self._settings.temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return (
            hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            {
                "base_url": payload["base_url"],
                "model": payload["model"],
                "temperature": payload["temperature"],
                "max_tokens": payload["max_tokens"],
                "message_count": len(messages),
            },
        )

    def chat_many(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        """批量调用 LLM，默认按配置并发执行。

        Args:
            messages_batch: 多个请求的消息列表。

        Returns:
            list[str]: 与输入顺序一致的文本输出列表。

        Raises:
            LLMClientError: 任一子任务失败且超过重试次数时抛出。
        """

        return run_tasks(
            messages_batch,
            self.chat,
            concurrent_enabled=self._settings.concurrent_enabled,
            max_concurrency=self._settings.max_concurrency,
        )

    def chat_many_with_metadata(self, messages_batch: list[list[dict[str, str]]]) -> list[LLMChatResponse]:
        """批量调用 LLM 并返回观测元数据。

        Args:
            messages_batch: 多个请求的消息列表。

        Returns:
            list[LLMChatResponse]: 与输入顺序一致的响应元数据列表。
        """

        return run_tasks(
            messages_batch,
            self.chat_with_metadata,
            concurrent_enabled=self._settings.concurrent_enabled,
            max_concurrency=self._settings.max_concurrency,
        )


def _as_int(value: object) -> int:
    """容错读取整数值。

    Args:
        value: 任意待转换对象。

    Returns:
        int: 转换成功返回整数，失败返回 `0`。
    """

    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_invalid_content(content: object) -> bool:
    """判断模型文本是否为空或无效占位。"""
    text = "" if content is None else str(content).strip()
    return not text or text.lower() in {"none", "null"}
