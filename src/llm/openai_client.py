"""OpenAI 兼容客户端。"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import LLMSettings


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
