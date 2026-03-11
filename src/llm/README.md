# LLM 公共模块

用于统一管理全项目的 LLM 调用能力，避免各业务模块重复实现。

## 目录

- `llm_config.json`：全局 LLM 配置
- `config.py`：加载 `.env` + 解析配置 + 生成 `LLMSettings`
- `openai_client.py`：OpenAI 兼容客户端 + 并发执行器 + 缓存管理器

## 使用方式

```python
from src.llm import load_llm_settings, OpenAICompatibleClient

settings = load_llm_settings()
client = OpenAICompatibleClient(settings)
text = client.chat(messages=[{"role": "user", "content": "hello"}])
texts = client.chat_many([
    [{"role": "user", "content": "hello 1"}],
    [{"role": "user", "content": "hello 2"}],
])
stats = client.cache_stats()
```

## 配置优先级

1. `.env` 环境变量
2. `src/llm/llm_config.json` 默认项

支持 `LLM_BASE_URL` 填写为：
- `http://host:port/v1`
- `http://host:port/v1/chat/completions`

## 并发配置

`src/llm/llm_config.json` 默认开启并发调用：

- `concurrent_enabled`: `true`
- `max_concurrency`: `10`

业务模块可通过命令参数传入关闭并发（例如 classification 的 `--disable-llm-concurrency`）。

## 反思开关（消融）

`src/llm/llm_config.json` 支持：

- `reasoning_enable=true`：不强制下发 thinking 控制参数（保持模型默认行为）。
- `reasoning_enable=false`：公共客户端自动补充
  `enable_thinking=false` 与 `chat_template_kwargs.enable_thinking=false`
  （调用方显式传入同名字段时，以调用方为准）。

该开关会参与缓存 key 计算，避免开启/关闭反思时命中同一缓存。

## 缓存机制

### 1) 缓存开关（消融）

- `cache_enabled=false`：完全关闭缓存读写（用于消融实验）。
- `cache_enabled=true`：启用内存缓存 + JSONL 持久化。

### 2) 缓存 Key 计算方式

缓存键来自以下字段的规范化 JSON（`sort_keys=True`）：

- `base_url`（去掉尾部 `/`）
- `model`
- `temperature`
- `max_tokens`
- `reasoning_enabled`
- `extra_payload`
- `messages`（完整消息数组）

随后对该 JSON 字符串计算 `SHA-256`，得到 `cache_key`。

### 3) 缓存命中方式

- 客户端在请求前先计算 `cache_key`；
- 若内存字典中存在同 key，直接返回缓存结果，并标记 `cached=true`；
- 未命中则走 HTTP 请求，成功后写入内存和 JSONL。

> 启动时会先把 `cache_path` 中历史 JSONL 全量加载到内存，若同 key 出现多次，以最后一条为准。

### 4) 缓存记录方式（JSONL）

每次写缓存会向 `cache_path` 追加一行结构化记录：

```json
{
  "event": "cache_write",
  "cache_version": 1,
  "created_at": "2026-02-28T08:00:00+00:00",
  "cache_key": "<sha256>",
  "key_meta": {
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "temperature": 0,
    "max_tokens": 256,
    "extra_payload": {},
    "reasoning_enabled": true,
    "message_count": 2
  },
  "response": {
    "content": "...",
    "token_in": 100,
    "token_out": 25,
    "total_tokens": 125,
    "cached_tokens": 0,
    "reasoning_tokens": 0,
    "total_tokens_estimated": false,
    "latency_ms": 820.3,
    "request_id": "req_xxx",
    "retries": 0
  }
}
```

### 5) 可统计指标

`OpenAICompatibleClient.cache_stats()` 可返回结构化缓存统计：

- `enabled/load_line_count/load_valid_count/load_invalid_count`
- `memory_entry_count`
- `hit_count/miss_count/write_count/write_error_count/bypass_count`
