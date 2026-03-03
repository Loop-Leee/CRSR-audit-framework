"""分类流水线。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.llm import LLMSettings
from src.tools.logger import Logger

from .keyword_matcher import KeywordMatcher
from .risk_catalog import RiskCatalog, load_risk_catalog
from .semantic_matcher import SemanticMatchResult, SemanticMatcher, SemanticTask, SemanticTrace


ClassificationStrategy = Literal["keyword_only", "llm_only", "keyword_llm"]
ALL_CLASSIFICATION_STRATEGIES: tuple[ClassificationStrategy, ...] = (
    "keyword_only",
    "llm_only",
    "keyword_llm",
)


@dataclass(frozen=True, slots=True)
class ChunkClassificationDiagnostic:
    """单个 chunk 的分类诊断信息。"""

    source_file: str
    chunk_id: int | str
    keyword_risks: list[str]
    semantic_risks: list[str]
    final_risks: list[str]
    llm_called: bool
    schema_valid: bool
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
    conflict: bool


@dataclass(frozen=True, slots=True)
class ClassificationRunResult:
    """分类执行结果。"""

    outputs: list[Path]
    chunk_diagnostics: list[ChunkClassificationDiagnostic]
    outputs_by_mode: dict[str, list[Path]]
    diagnostics_by_mode: dict[str, list[ChunkClassificationDiagnostic]]


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
    logger: Logger,
) -> list[Path]:
    """执行分类主流程（兼容旧接口，仅返回输出文件列表）。"""

    return run_classification_with_diagnostics(
        input_path=input_path,
        output_dir=output_dir,
        risk_info_path=risk_info_path,
        llm_settings=llm_settings,
        logger=logger,
    ).outputs


def run_classification_with_diagnostics(
    *,
    input_path: Path,
    output_dir: Path,
    risk_info_path: Path,
    llm_settings: LLMSettings,
    logger: Logger,
    strategy: ClassificationStrategy = "keyword_llm",
    emit_all_variants: bool = False,
) -> ClassificationRunResult:
    """执行分类主流程并返回 chunk 级诊断信息。"""

    _validate_strategy(strategy)
    output_modes = _effective_output_modes(strategy, emit_all_variants)
    files = discover_chunk_files(input_path)
    if not files:
        raise FileNotFoundError(f"未找到待分类文件: {input_path}")

    catalog = load_risk_catalog(risk_info_path)
    need_keyword = strategy != "llm_only" or emit_all_variants
    need_llm = strategy != "keyword_only" or emit_all_variants
    keyword_matcher = KeywordMatcher(catalog) if need_keyword else None
    semantic_matcher = SemanticMatcher(catalog, llm_settings, logger) if need_llm else None
    logger.info(
        "risk_types_loaded: count=%s, risks=%s"
        % (len(catalog.order), catalog.order)
    )
    logger.info(
        "LLM 并发配置: concurrent_enabled=%s, max_concurrency=%s"
        % (llm_settings.concurrent_enabled, llm_settings.max_concurrency)
    )

    logger.info(
        "分类开始: file_count=%s, risk_type_count=%s, input=%s, output=%s"
        % (len(files), len(catalog.definitions), input_path, output_dir)
    )

    outputs: list[Path] = []
    outputs_by_mode: dict[str, list[Path]] = {mode: [] for mode in output_modes}
    diagnostics: list[ChunkClassificationDiagnostic] = []
    diagnostics_by_mode: dict[str, list[ChunkClassificationDiagnostic]] = {mode: [] for mode in output_modes}
    failures: list[str] = []

    for file_path in files:
        try:
            logger.info(f"处理文件: {file_path}")
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            chunks = _extract_chunks(payload, file_path)

            chunk_ids = [chunk.get("chunk_id", index) for index, chunk in enumerate(chunks, start=1)]
            contents = [chunk["content"] for chunk in chunks]
            keyword_risks_list = (
                [keyword_matcher.match(content) for content in contents]
                if keyword_matcher is not None
                else [[] for _ in contents]
            )
            semantic_tasks = [SemanticTask(chunk_id=chunk_id, text=content) for chunk_id, content in zip(chunk_ids, contents)]
            semantic_results: list[SemanticMatchResult] = (
                semantic_matcher.match_many_with_trace(semantic_tasks, file_path.name)
                if semantic_matcher is not None
                else [_disabled_semantic_result() for _ in semantic_tasks]
            )
            final_risks_by_mode: dict[str, list[list[str]]] = {mode: [] for mode in output_modes}

            for chunk, chunk_id, keyword_risks, semantic_result in zip(
                chunks, chunk_ids, keyword_risks_list, semantic_results
            ):
                semantic_risks = semantic_result.risks
                final_risks = _merge_risks(
                    strategy,
                    keyword_risks=keyword_risks,
                    semantic_risks=semantic_risks,
                    catalog=catalog,
                )
                mode_risks = _resolve_mode_risks(
                    strategy,
                    keyword_risks=keyword_risks,
                    semantic_risks=semantic_risks,
                    catalog=catalog,
                    emit_all_variants=emit_all_variants,
                )
                if emit_all_variants:
                    final_risks_by_mode["keyword_only"].append(mode_risks["keyword_only"])
                    final_risks_by_mode["llm_only"].append(mode_risks["llm_only"])
                    final_risks_by_mode["keyword_llm"].append(mode_risks["keyword_llm"])
                else:
                    final_risks_by_mode[strategy].append(final_risks)
                chunk["risk_type"] = final_risks
                keyword_set = set(keyword_risks)
                semantic_set = set(semantic_risks)
                added_from_llm = catalog.normalize_risks(
                    [risk for risk in semantic_risks if risk not in keyword_set]
                )
                conflict = bool(keyword_set and semantic_set and keyword_set.isdisjoint(semantic_set))
                trace = semantic_result.trace

                for mode in output_modes:
                    diagnostics_by_mode[mode].append(
                        _build_mode_diagnostic(
                            mode=mode,
                            source_file=str(file_path),
                            chunk_id=chunk_id,
                            keyword_risks=keyword_risks,
                            semantic_risks=semantic_risks,
                            final_risks=mode_risks[mode],
                            trace=trace,
                            conflict=conflict,
                        )
                    )
                diagnostics.append(diagnostics_by_mode[strategy][-1])

                logger.info(
                    "chunk 分类完成: file=%s, chunk_id=%s, keyword_hits=%s, semantic_hits=%s, final_hits=%s, "
                    "token_in=%s, token_out=%s, total_tokens=%s, cached_tokens=%s, reasoning_tokens=%s, "
                    "total_tokens_estimated=%s, latency_ms=%.2f, request_id=%s, retries=%s, error_code=%s, cache=%s, "
                    "keyword_hits_list=%s, semantic_hits_list=%s, added=%s"
                    % (
                        file_path.name,
                        chunk_id,
                        len(keyword_risks),
                        len(semantic_risks),
                        len(final_risks),
                        trace.token_in,
                        trace.token_out,
                        trace.total_tokens,
                        trace.cached_tokens,
                        trace.reasoning_tokens,
                        trace.total_tokens_estimated,
                        trace.latency_ms,
                        trace.request_id or "-",
                        trace.retries,
                        trace.error_code or "-",
                        trace.cached,
                        keyword_risks,
                        semantic_risks,
                        added_from_llm,
                    )
                )

            for mode in output_modes:
                mode_output_dir = output_dir / mode if emit_all_variants else output_dir
                output_path = _build_output_path(file_path, mode_output_dir)
                output_payload = _build_payload_with_risks(payload, file_path, final_risks_by_mode[mode])
                output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                outputs_by_mode[mode].append(output_path)
                logger.info(f"写出完成: mode={mode}, output={output_path}")
                if mode == strategy:
                    outputs.append(output_path)

        except Exception as exc:  # noqa: BLE001
            message = f"文件处理失败: file={file_path}, error={exc}"
            logger.error(message)
            failures.append(message)

    if failures:
        raise RuntimeError("分类流程存在失败文件，请检查日志。")

    logger.info("分类结束: success_count=%s" % len(outputs))
    return ClassificationRunResult(
        outputs=outputs,
        chunk_diagnostics=diagnostics,
        outputs_by_mode=outputs_by_mode,
        diagnostics_by_mode=diagnostics_by_mode,
    )


def _validate_strategy(strategy: str) -> None:
    if strategy not in ALL_CLASSIFICATION_STRATEGIES:
        raise ValueError(f"不支持的分类策略: {strategy}")


def _effective_output_modes(strategy: ClassificationStrategy, emit_all_variants: bool) -> tuple[ClassificationStrategy, ...]:
    return ALL_CLASSIFICATION_STRATEGIES if emit_all_variants else (strategy,)


def _merge_risks(
    strategy: ClassificationStrategy,
    *,
    keyword_risks: list[str],
    semantic_risks: list[str],
    catalog: RiskCatalog,
) -> list[str]:
    if strategy == "keyword_only":
        return catalog.normalize_risks(keyword_risks)
    if strategy == "llm_only":
        return catalog.normalize_risks(semantic_risks)
    return catalog.normalize_risks(keyword_risks + semantic_risks)


def _resolve_mode_risks(
    strategy: ClassificationStrategy,
    *,
    keyword_risks: list[str],
    semantic_risks: list[str],
    catalog: RiskCatalog,
    emit_all_variants: bool,
) -> dict[ClassificationStrategy, list[str]]:
    if emit_all_variants:
        return {
            "keyword_only": catalog.normalize_risks(keyword_risks),
            "llm_only": catalog.normalize_risks(semantic_risks),
            "keyword_llm": catalog.normalize_risks(keyword_risks + semantic_risks),
        }
    return {
        strategy: _merge_risks(
            strategy,
            keyword_risks=keyword_risks,
            semantic_risks=semantic_risks,
            catalog=catalog,
        )
    }


def _build_mode_diagnostic(
    *,
    mode: ClassificationStrategy,
    source_file: str,
    chunk_id: int | str,
    keyword_risks: list[str],
    semantic_risks: list[str],
    final_risks: list[str],
    trace: SemanticTrace,
    conflict: bool,
) -> ChunkClassificationDiagnostic:
    if mode == "keyword_only":
        return ChunkClassificationDiagnostic(
            source_file=source_file,
            chunk_id=chunk_id,
            keyword_risks=keyword_risks,
            semantic_risks=[],
            final_risks=final_risks,
            llm_called=False,
            schema_valid=True,
            token_in=0,
            token_out=0,
            total_tokens=0,
            cached_tokens=0,
            reasoning_tokens=0,
            total_tokens_estimated=False,
            latency_ms=0.0,
            request_id="",
            retries=0,
            error_code=None,
            cached=False,
            conflict=False,
        )

    if mode == "llm_only":
        return ChunkClassificationDiagnostic(
            source_file=source_file,
            chunk_id=chunk_id,
            keyword_risks=[],
            semantic_risks=semantic_risks,
            final_risks=final_risks,
            llm_called=trace.llm_called,
            schema_valid=trace.schema_valid,
            token_in=trace.token_in,
            token_out=trace.token_out,
            total_tokens=trace.total_tokens,
            cached_tokens=trace.cached_tokens,
            reasoning_tokens=trace.reasoning_tokens,
            total_tokens_estimated=trace.total_tokens_estimated,
            latency_ms=trace.latency_ms,
            request_id=trace.request_id,
            retries=trace.retries,
            error_code=trace.error_code,
            cached=trace.cached,
            conflict=False,
        )

    return ChunkClassificationDiagnostic(
        source_file=source_file,
        chunk_id=chunk_id,
        keyword_risks=keyword_risks,
        semantic_risks=semantic_risks,
        final_risks=final_risks,
        llm_called=trace.llm_called,
        schema_valid=trace.schema_valid,
        token_in=trace.token_in,
        token_out=trace.token_out,
        total_tokens=trace.total_tokens,
        cached_tokens=trace.cached_tokens,
        reasoning_tokens=trace.reasoning_tokens,
        total_tokens_estimated=trace.total_tokens_estimated,
        latency_ms=trace.latency_ms,
        request_id=trace.request_id,
        retries=trace.retries,
        error_code=trace.error_code,
        cached=trace.cached,
        conflict=conflict,
    )


def _disabled_semantic_result() -> SemanticMatchResult:
    return SemanticMatchResult(
        risks=[],
        trace=SemanticTrace(
            llm_called=False,
            schema_valid=True,
            token_in=0,
            token_out=0,
            total_tokens=0,
            cached_tokens=0,
            reasoning_tokens=0,
            total_tokens_estimated=False,
            latency_ms=0.0,
            request_id="",
            retries=0,
            error_code="llm_disabled",
            cached=False,
        ),
    )


def _build_payload_with_risks(payload: object, source: Path, risks_by_chunk: list[list[str]]) -> object:
    """按给定风险列表构造输出 payload。"""

    copied_payload = json.loads(json.dumps(payload, ensure_ascii=False))
    chunks = _extract_chunks(copied_payload, source)
    if len(chunks) != len(risks_by_chunk):
        raise ValueError(
            f"输出风险数量与 chunk 数不一致: source={source}, chunks={len(chunks)}, risks={len(risks_by_chunk)}"
        )
    for chunk, risks in zip(chunks, risks_by_chunk):
        chunk["risk_type"] = risks
    return copied_payload
