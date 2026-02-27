"""分类流水线。"""

from __future__ import annotations

import json
from pathlib import Path

from src.llm import LLMSettings

from .classification_logger import ClassificationLogger
from .keyword_matcher import KeywordMatcher
from .risk_catalog import load_risk_catalog
from .semantic_matcher import SemanticMatcher


def discover_chunk_files(input_path: Path) -> list[Path]:
    """发现待分类的 chunk JSON 文件。"""

    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    files = sorted(input_path.glob("*.chunks.json"))
    if files:
        return files
    return sorted(input_path.glob("*.json"))


def _extract_chunks(payload: object, source: Path) -> list[dict]:
    """从输入载荷中提取 chunk 列表并校验结构。"""

    if isinstance(payload, dict):
        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError(f"chunks 字段缺失或类型错误: {source}")
    elif isinstance(payload, list):
        chunks = payload
    else:
        raise ValueError(f"输入 JSON 必须是对象或数组: {source}")

    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(f"chunks[{index}] 不是对象: {source}")
        if not isinstance(chunk.get("content"), str):
            chunk_id = chunk.get("chunk_id", index)
            raise ValueError(f"chunk content 非字符串: {source}, chunk_id={chunk_id}")

    return chunks


def _build_output_path(input_file: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_file.name.endswith(".chunks.json"):
        filename = input_file.name.replace(".chunks.json", ".classified.json")
    else:
        filename = f"{input_file.stem}.classified.json"
    return output_dir / filename


def run_classification(
    *,
    input_path: Path,
    output_dir: Path,
    risk_info_path: Path,
    llm_settings: LLMSettings,
    logger: ClassificationLogger,
) -> list[Path]:
    """执行分类主流程。"""

    files = discover_chunk_files(input_path)
    if not files:
        raise FileNotFoundError(f"未找到待分类文件: {input_path}")

    catalog = load_risk_catalog(risk_info_path)
    keyword_matcher = KeywordMatcher(catalog)
    semantic_matcher = SemanticMatcher(catalog, llm_settings, logger)

    logger.info(
        "分类开始: file_count=%s, risk_type_count=%s, input=%s, output=%s"
        % (len(files), len(catalog.definitions), input_path, output_dir)
    )

    outputs: list[Path] = []
    failures: list[str] = []

    for file_path in files:
        try:
            logger.info(f"处理文件: {file_path}")
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            chunks = _extract_chunks(payload, file_path)

            for index, chunk in enumerate(chunks, start=1):
                chunk_id = chunk.get("chunk_id", index)
                content = chunk["content"]

                keyword_risks = keyword_matcher.match(content)
                semantic_risks = semantic_matcher.match(content, chunk_id, file_path.name)
                final_risks = catalog.normalize_risks(keyword_risks + semantic_risks)
                chunk["risk_type"] = final_risks

                logger.info(
                    "chunk 分类完成: file=%s, chunk_id=%s, keyword_hits=%s, semantic_hits=%s, final_hits=%s"
                    % (
                        file_path.name,
                        chunk_id,
                        len(keyword_risks),
                        len(semantic_risks),
                        len(final_risks),
                    )
                )

            output_path = _build_output_path(file_path, output_dir)
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            outputs.append(output_path)
            logger.info(f"写出完成: {output_path}")

        except Exception as exc:  # noqa: BLE001
            message = f"文件处理失败: file={file_path}, error={exc}"
            logger.error(message)
            failures.append(message)

    if failures:
        raise RuntimeError("分类流程存在失败文件，请检查日志。")

    logger.info("分类结束: success_count=%s" % len(outputs))
    return outputs
