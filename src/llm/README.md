# LLM 公共模块

用于统一管理全项目的 LLM 调用能力，避免各业务模块重复实现。

## 目录

- `llm_config.json`：全局 LLM 配置
- `config.py`：加载 `.env` + 解析配置 + 生成 `LLMSettings`
- `openai_client.py`：OpenAI 兼容客户端 + 并发执行器

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
