"""统一 LLM 配置加载。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_LLM_CONFIG_PATH = MODULE_DIR / "llm_config.json"


def load_dotenv_file(dotenv_path: Path) -> None:
    """读取 .env 并写入环境变量（不覆盖已存在值）。"""

    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """LLM 调用配置。"""

    enabled: bool
    api_key: str
    base_url: str
    model: str
    concurrent_enabled: bool
    max_concurrency: int
    temperature: float
    timeout_seconds: int
    max_retries: int
    cache_enabled: bool
    cache_path: Path


def _normalize_base_url(raw_url: str) -> str:
    """兼容传入 /v1 或 /v1/chat/completions 两种格式。"""

    base = raw_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _resolve_path(path_value: str | Path) -> Path:
    """将配置路径解析为绝对路径。"""

    raw = Path(path_value)
    if raw.is_absolute():
        return raw
    return PROJECT_ROOT / raw


def load_llm_settings(
    config_path: Path | None = None,
    enabled_override: bool | None = None,
    concurrent_enabled_override: bool | None = None,
) -> LLMSettings:
    """加载统一 LLM 配置并注入环境变量。"""

    load_dotenv_file(PROJECT_ROOT / ".env")
    path = config_path or DEFAULT_LLM_CONFIG_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))

    timeout_seconds = int(raw.get("timeout_seconds", 45))
    max_retries = int(raw.get("max_retries", 2))
    max_concurrency = int(raw.get("max_concurrency", 10))
    if timeout_seconds <= 0:
        raise ValueError("配置错误：timeout_seconds 必须大于 0。")
    if max_retries < 0:
        raise ValueError("配置错误：max_retries 不能小于 0。")
    if max_concurrency <= 0:
        raise ValueError("配置错误：max_concurrency 必须大于 0。")

    cache_enabled = bool(raw.get("cache_enabled", True))
    cache_path = _resolve_path(raw.get("cache_path", "data/experiments/cache/llm_cache.jsonl"))

    api_key_env = str(raw.get("api_key_env", "LLM_API_KEY"))
    base_url_env = str(raw.get("base_url_env", "LLM_BASE_URL"))
    model_env = str(raw.get("model_env", "LLM_MODEL"))

    enabled = bool(raw.get("enabled", True))
    if enabled_override is not None:
        enabled = enabled_override
    concurrent_enabled = bool(raw.get("concurrent_enabled", True))
    if concurrent_enabled_override is not None:
        concurrent_enabled = concurrent_enabled_override

    return LLMSettings(
        enabled=enabled,
        api_key=os.getenv(api_key_env, "").strip(),
        base_url=_normalize_base_url(
            os.getenv(base_url_env, str(raw.get("default_base_url", "https://api.openai.com/v1")))
        ),
        model=os.getenv(model_env, "").strip(),
        concurrent_enabled=concurrent_enabled,
        max_concurrency=max_concurrency,
        temperature=float(raw.get("temperature", 0)),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        cache_enabled=cache_enabled,
        cache_path=cache_path,
    )
