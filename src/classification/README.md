# Classification 模块

模块功能：对 `chunk` 输出进行审查类型分类。每个 chunk 先做关键词匹配，再做 LLM 语义匹配，最终取并集作为 `risk_type`。

## 运行方式

```bash
python3 main.py classification
```

常用参数：

```bash
python3 main.py classification --input data/2-chunks --output data/3-classified
python3 main.py classification --disable-llm
python3 main.py classification --llm-config src/llm/llm_config.json
python3 main.py classification --disable-llm-concurrency
```

## 输入输出

- 输入：`data/2-chunks/*.chunks.json`（或单个 JSON 文件）
- 输出：`data/3-classified/*.classified.json`
- 输出差异：每个 chunk 新增 `risk_type` 字段（字符串数组）

示例：

```json
{
  "chunk_id": 1,
  "content": "...",
  "risk_type": ["付款时间审查", "发票审查"]
}
```

## 配置

默认配置文件：`src/classification/classifier_config.json`

```json
{
  "input_dir": "data/2-chunks",
  "output_dir": "data/3-classified",
  "risk_info_path": "prompt/risk_info.csv",
  "log_dir": "log/clasification"
}
```

## 环境变量

- `LLM_API_KEY`：OpenAI 兼容接口密钥（无鉴权部署可留空）
- `LLM_BASE_URL`：OpenAI 兼容服务地址（可选，支持 `.../v1` 或 `.../v1/chat/completions`）
- `LLM_MODEL`：模型名称

模块会自动加载项目根目录 `.env`。
如果 `LLM_MODEL` 缺失或 LLM 调用失败，模块会自动降级为“仅关键词匹配”，并写入错误日志。

全局 LLM 配置文件：`src/llm/llm_config.json`
默认并发调用：`concurrent_enabled=true`，`max_concurrency=10`。
如果需要串行调用，可加参数 `--disable-llm-concurrency`。

## 日志

日志目录：`log/clasification/`

日志要求：关键执行步骤与执行错误都会写入日志，使用 `[info]` 与 `[error]` 标签区分。

## 关键文件

- `classification_config.py`：配置加载与路径解析
- `classification_logger.py`：日志输出
- `risk_catalog.py`：风险类型目录加载（来自 `prompt/risk_info.csv`）
- `keyword_matcher.py`：关键词匹配
- `semantic_matcher.py`：语义匹配与失败降级
- `classification_pipeline.py`：分类流水线
- `classification_cli.py`：命令行入口
- `src/llm/config.py`：全局 LLM 配置解析
- `src/llm/openai_client.py`：全局 OpenAI 兼容客户端
