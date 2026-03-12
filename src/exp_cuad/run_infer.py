from __future__ import annotations

import argparse
import ast
import json
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.tools.logger import Logger, get_logger

from .cuad_loader import load_cuad_from_hf, load_cuad_from_local, load_label_descriptions
from .eval_metrics import (
    PairResult,
    compute_evidence_jaccard,
    compute_inconsistency_rate,
    compute_laziness_rate,
    compute_macro_f1,
    compute_presence_f1,
    spans_to_evidence_sentences,
)
from .prompts import (
    BASELINE_SYSTEM,
    CRSR_LITE_SYSTEM,
    make_baseline_user_prompt,
    make_crsr_lite_user_prompt,
)
from .text_utils import (
    chunk_by_semantic_jaccard_2gram,
    chunk_by_tokens_approx,
    normalize_ws,
    sentence_set_jaccard,
)

OPENAI_CHUNK_CONCURRENCY = 10
_JSON_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


@dataclass(slots=True)
class GenerationCall:
    """单次模型调用结果。"""

    text: str
    latency_ms: float
    token_in: int = 0
    token_out: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens_estimated: bool = False
    retries: int = 0
    error_code: str | None = None
    cached: bool = False


@dataclass(slots=True)
class OutputSwitches:
    """输出消融开关集合。"""

    emit_llm_usage: bool
    emit_extended_pair_metrics: bool

    def to_dict(self) -> dict[str, bool]:
        """导出可序列化消融配置。"""

        return {
            "emit_llm_usage": self.emit_llm_usage,
            "emit_extended_pair_metrics": self.emit_extended_pair_metrics,
        }


@dataclass(slots=True)
class ChunkTrace:
    """单个 chunk 的结构化推理记录。"""

    chunk_index: int
    chunk_chars: int
    present: bool
    evidence_count: int
    infer_ms: float
    total_latency_ms: float
    json_parse_failed: bool
    json_retried: bool
    retry_tokens: int
    retry_ms: float
    llm_usage: dict[str, int | float]

    def to_dict(self, *, include_llm_usage: bool) -> dict[str, Any]:
        """转为可写入 JSONL 的字典。"""

        payload: dict[str, Any] = {
            "chunk_index": self.chunk_index,
            "chunk_chars": self.chunk_chars,
            "present": self.present,
            "evidence_count": self.evidence_count,
            "infer_ms": self.infer_ms,
            "total_latency_ms": self.total_latency_ms,
            "json_parse_failed": self.json_parse_failed,
            "json_retried": self.json_retried,
            "retry_tokens": self.retry_tokens,
            "retry_ms": self.retry_ms,
        }
        if include_llm_usage:
            payload["llm_usage"] = self.llm_usage
        return payload


def _usage_zero() -> dict[str, int | float]:
    """初始化 LLM 用量统计。"""

    return {
        "call_count": 0,
        "token_in": 0,
        "token_out": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "latency_ms": 0.0,
        "retry_count": 0,
        "cache_hit_count": 0,
        "error_count": 0,
        "total_tokens_estimated_count": 0,
    }


def _usage_add_call(stats: dict[str, int | float], call: GenerationCall) -> None:
    """将单次调用计入统计。"""

    stats["call_count"] += 1
    stats["token_in"] += call.token_in
    stats["token_out"] += call.token_out
    stats["total_tokens"] += call.total_tokens
    stats["cached_tokens"] += call.cached_tokens
    stats["reasoning_tokens"] += call.reasoning_tokens
    stats["latency_ms"] += float(call.latency_ms)
    stats["retry_count"] += call.retries
    stats["cache_hit_count"] += int(call.cached)
    stats["error_count"] += int(call.error_code is not None)
    stats["total_tokens_estimated_count"] += int(call.total_tokens_estimated)


def _usage_merge(base: dict[str, int | float], delta: dict[str, int | float]) -> None:
    """合并两份 LLM 用量统计。"""

    for key, value in delta.items():
        base[key] += value


def llm_generate_transformers_builder(model_name: str) -> Callable[[str, str, int], GenerationCall]:
    """构建 transformers 后端生成器（模型只初始化一次）。"""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependencies for transformers backend. "
            "Install with: conda install -n crsr-audit -c conda-forge transformers pytorch sentencepiece accelerate"
        ) from exc

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    def _generate(system: str, user: str, max_new_tokens: int = 256) -> GenerationCall:
        started = time.perf_counter()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tok.decode(out[0], skip_special_tokens=True)
        return GenerationCall(
            text=text,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            total_tokens_estimated=True,
        )

    return _generate


def llm_generate_vllm_builder(model_name: str) -> Callable[[str, str, int], GenerationCall]:
    """构建 vLLM 后端生成器（模型只初始化一次）。"""

    try:
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:
        os_hint = ""
        if platform.system() == "Darwin":
            os_hint = " vLLM is generally unavailable on macOS; use --backend transformers instead."
        raise ModuleNotFoundError(
            "Missing dependency 'vllm'. Install on Linux with GPU: pip install vllm." + os_hint
        ) from exc

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'transformers'. "
            "Install with: conda install -n crsr-audit -c conda-forge transformers sentencepiece"
        ) from exc

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(model=model_name, trust_remote_code=True, tensor_parallel_size=1)

    def _generate(system: str, user: str, max_new_tokens: int = 256) -> GenerationCall:
        started = time.perf_counter()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        out = llm.generate([prompt], params)[0].outputs[0].text
        return GenerationCall(
            text=out,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            total_tokens_estimated=True,
        )

    return _generate


def _append_no_think_suffix(user_prompt: str, *, enable: bool) -> str:
    """按需给用户提示词追加 `/no_think`。"""

    if not enable:
        return user_prompt
    if "/no_think" in user_prompt:
        return user_prompt
    return f"{user_prompt.rstrip()}\n/no_think"


def _build_openai_extra_payload(
    *,
    max_new_tokens: int,
    send_max_new_tokens_param: bool,
    disable_thinking: bool,
) -> dict[str, Any]:
    """构建 OpenAI-compatible 额外请求参数。"""

    payload: dict[str, Any] = {}
    if send_max_new_tokens_param:
        token_limit = int(max_new_tokens)
        # 兼容不同网关的 token 参数命名。
        payload["max_new_tokens"] = token_limit
        payload["max_completion_tokens"] = token_limit
    if disable_thinking:
        # 兼容两类常见 OpenAI-compatible 网关写法。
        payload["enable_thinking"] = False
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def llm_generate_openai_builder(
    *,
    model_name: str,
    client: Any,
    force_no_think_prompt: bool,
    disable_thinking: bool,
    send_max_new_tokens_param: bool,
) -> Callable[[str, str, int], GenerationCall]:
    """构建 OpenAI 兼容后端生成器。"""

    def _generate(system: str, user: str, max_new_tokens: int = 256) -> GenerationCall:
        user_content = _append_no_think_suffix(user, enable=force_no_think_prompt)
        extra_payload = _build_openai_extra_payload(
            max_new_tokens=max_new_tokens,
            send_max_new_tokens_param=send_max_new_tokens_param,
            disable_thinking=disable_thinking,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
        resp = client.chat_with_metadata(
            messages,
            model=model_name,
            max_tokens=max_new_tokens,
            extra_payload=extra_payload,
        )
        return GenerationCall(
            text=resp.content,
            latency_ms=resp.latency_ms,
            token_in=resp.token_in,
            token_out=resp.token_out,
            total_tokens=resp.total_tokens,
            cached_tokens=resp.cached_tokens,
            reasoning_tokens=resp.reasoning_tokens,
            total_tokens_estimated=resp.total_tokens_estimated,
            retries=resp.retries,
            error_code=resp.error_code,
            cached=resp.cached,
        )

    return _generate


def _normalize_pred_object(obj: dict) -> dict | None:
    """把可能的嵌套对象统一映射为 ``{present, evidence}`` 结构。"""

    candidates: list[dict] = [obj]
    for key in ("result", "output", "answer", "json", "data"):
        nested = obj.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)

    for cand in candidates:
        if "present" in cand or "evidence" in cand:
            present = bool(cand.get("present", False))
            evidence = cand.get("evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]
            if not isinstance(evidence, list):
                evidence = []
            evidence = [x for x in evidence if isinstance(x, str)]
            return {"present": present, "evidence": evidence}
    return None


def _extract_json_candidates(text: str) -> list[str]:
    """从模型输出中提取可能的 JSON 片段候选。"""

    candidates: list[str] = [text]

    for matched in _JSON_CODEBLOCK_RE.finditer(text):
        block = matched.group(1).strip()
        if block:
            candidates.append(block)

    start = -1
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidates.append(text[start : idx + 1])
                    start = -1

    uniq: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        stripped = candidate.strip()
        if stripped and stripped not in seen:
            uniq.append(stripped)
            seen.add(stripped)
    return uniq


def _parse_candidate_json(candidate: str) -> dict | None:
    """尝试多种容错方式解析候选片段为字典。"""

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        pass

    relaxed = re.sub(r",(\s*[}\]])", r"\1", candidate)
    try:
        parsed = json.loads(relaxed)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        pass

    py_like = re.sub(r"\btrue\b", "True", relaxed, flags=re.IGNORECASE)
    py_like = re.sub(r"\bfalse\b", "False", py_like, flags=re.IGNORECASE)
    py_like = re.sub(r"\bnull\b", "None", py_like, flags=re.IGNORECASE)
    try:
        parsed = ast.literal_eval(py_like)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        pass

    return None


def parse_json_loose(s: str | None) -> dict:
    """尽量从模型输出中提取并解析 JSON，失败时返回保底结构。"""

    text = "" if s is None else str(s)
    text = text.strip()
    if not text or text.lower() in {"none", "null"}:
        return {"present": False, "evidence": [], "_raw": text}

    for candidate in _extract_json_candidates(text):
        parsed = _parse_candidate_json(candidate)
        if isinstance(parsed, dict):
            normalized = _normalize_pred_object(parsed)
            if normalized is not None:
                return normalized
    return {"present": False, "evidence": [], "_raw": text}


def parse_json_loose_with_logging(
    s: str | None,
    *,
    logger: Logger,
    doc_id: str,
    label: str,
    chunk_index: int,
) -> dict:
    """带上下文日志的 JSON 容错解析。"""

    obj = parse_json_loose(s)
    if "_raw" in obj:
        preview = normalize_ws(str(obj.get("_raw", "")))[:200]
        logger.error(
            "模型输出 JSON 解析失败: doc_id=%s, label=%s, chunk_index=%s, raw_preview=%s"
            % (doc_id, label, chunk_index, preview)
        )
    return obj


def aggregate_chunks(chunk_outputs: list[dict]) -> tuple[bool, list[str], list[bool]]:
    """聚合多 chunk 输出，得到最终 presence、证据列表和投票明细。"""

    votes: list[bool] = []
    evidence_all: list[str] = []
    evidence_seen: set[str] = set()

    for output in chunk_outputs:
        present = bool(output.get("present", False))
        votes.append(present)
        evidences = output.get("evidence", []) or []
        for evidence in evidences:
            normalized = normalize_ws(evidence)
            if normalized and normalized not in evidence_seen:
                evidence_seen.add(normalized)
                evidence_all.append(evidence)

    return any(votes), evidence_all, votes


def infer_single_chunk_with_retry(
    *,
    gen: Callable[[str, str, int], GenerationCall],
    system: str,
    user: str,
    max_new_tokens: int,
    logger: Logger,
    doc_id: str,
    label: str,
    chunk_index: int,
    chunk_chars: int,
) -> dict[str, Any]:
    """执行单个 chunk 推理，并在 JSON 解析失败时重试一次。"""

    first_call = gen(system, user, max_new_tokens)
    obj = parse_json_loose(first_call.text)

    usage = _usage_zero()
    _usage_add_call(usage, first_call)

    json_retried = False
    retry_tokens = 0
    retry_ms = 0.0

    if "_raw" in obj:
        json_retried = True
        # JSON 修复重试尽量短，避免再次被长 reasoning 吞掉配额。
        retry_tokens = max(128, min(int(max_new_tokens), 512))
        retry_call = gen(system, user, retry_tokens)
        retry_ms = retry_call.latency_ms
        _usage_add_call(usage, retry_call)

        obj_retry = parse_json_loose(retry_call.text)
        if "_raw" not in obj_retry:
            obj = obj_retry
        else:
            obj = parse_json_loose_with_logging(
                retry_call.text,
                logger=logger,
                doc_id=doc_id,
                label=label,
                chunk_index=chunk_index,
            )

    obj["present"] = bool(obj.get("present", False))
    evidence = obj.get("evidence", [])
    obj["evidence"] = [item for item in evidence if isinstance(item, str)]

    trace = ChunkTrace(
        chunk_index=chunk_index,
        chunk_chars=chunk_chars,
        present=obj["present"],
        evidence_count=len(obj["evidence"]),
        infer_ms=first_call.latency_ms,
        total_latency_ms=float(usage["latency_ms"]),
        json_parse_failed="_raw" in obj,
        json_retried=json_retried,
        retry_tokens=retry_tokens,
        retry_ms=retry_ms,
        llm_usage=usage,
    )

    return {"obj": obj, "trace": trace, "usage": usage}


def _build_user_prompt(mode: str, label: str, label_desc: str | None, chunk_text: str) -> str:
    """按模式构建用户提示词。"""

    if mode == "baseline":
        return make_baseline_user_prompt(label, label_desc, chunk_text)
    return make_crsr_lite_user_prompt(label, label_desc, chunk_text)


def _collect_labels(data: list[Any]) -> list[str]:
    """收集并排序整个数据集中的标签集合。"""

    all_labels: set[str] = set()
    for ex in data:
        all_labels.update(ex.spans.keys())
    return sorted(all_labels)


def _build_pair_record(
    *,
    pair_index: int,
    ex: Any,
    label: str,
    present_pred: bool,
    evidence_pred: list[str],
    present_gt: bool,
    evidence_gt: list[str],
    votes: list[bool],
    mode: str,
    backend: str,
    output_switches: OutputSwitches,
    pair_usage: dict[str, int | float],
) -> dict[str, Any]:
    """构建单条 pair 结构化记录。"""

    record: dict[str, Any] = {
        "analysis_version": 2,
        "pair_index": pair_index,
        "doc_id": ex.doc_id,
        "label": label,
        "present_pred": present_pred,
        "evidence_pred": evidence_pred,
        "present_gt": present_gt,
        "evidence_gt": evidence_gt,
        "chunk_votes": votes,
        "mode": mode,
        "backend": backend,
    }

    if output_switches.emit_extended_pair_metrics:
        present_tp = int(present_pred and present_gt)
        present_fp = int(present_pred and not present_gt)
        present_fn = int((not present_pred) and present_gt)
        present_tn = int((not present_pred) and (not present_gt))
        record["presence_confusion"] = {
            "tp": present_tp,
            "fp": present_fp,
            "fn": present_fn,
            "tn": present_tn,
        }
        record["evidence_jaccard"] = (
            sentence_set_jaccard(evidence_pred, evidence_gt) if present_gt else None
        )

    if output_switches.emit_llm_usage:
        record["llm_usage"] = pair_usage

    return record


def _chunk_document(
    *,
    text: str,
    chunk_strategy: str,
    max_chars: int,
    overlap_chars: int,
    logger: Logger,
    doc_id: str,
) -> list[str]:
    """按指定策略分块文档，语义分块失败时回退到窗口分块。"""

    if chunk_strategy == "window":
        return chunk_by_tokens_approx(text, max_chars=max_chars, overlap_chars=overlap_chars)

    try:
        chunks = chunk_by_semantic_jaccard_2gram(text, max_chars=max_chars)
        if chunks:
            return chunks
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "语义分块失败，回退窗口分块: doc_id=%s, strategy=%s, error=%s"
            % (doc_id, chunk_strategy, exc)
        )

    return chunk_by_tokens_approx(text, max_chars=max_chars, overlap_chars=overlap_chars)


def _slugify(value: str, *, max_len: int = 32) -> str:
    """把任意字符串转换为文件名安全片段。"""

    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = lowered.strip("-")
    if not lowered:
        return "na"
    return lowered[:max_len]


def _build_output_name_suffix(args: argparse.Namespace) -> str:
    """构造输出文件的主要实验参数后缀。"""

    model_tag = _slugify(args.model, max_len=24)
    mode_tag = _slugify(args.mode, max_len=16)
    chunk_tag = _slugify(args.chunk_strategy, max_len=20)
    return (
        f"mode-{mode_tag}_chunk-{chunk_tag}_model-{model_tag}"
    )


def _resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """解析并标准化 out/progress/summary 输出路径。"""

    raw_out_path = Path(args.out_jsonl)
    timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_suffix = _build_output_name_suffix(args)

    stem = raw_out_path.stem
    if raw_out_path.suffix != ".jsonl":
        stem = raw_out_path.name
    final_stem = f"{stem}_{param_suffix}_{timestamp_suffix}"
    out_path = raw_out_path.with_name(final_stem + ".jsonl")

    progress_path = (
        Path(args.progress_jsonl)
        if args.progress_jsonl
        else out_path.with_suffix(".jsonl.progress.jsonl")
    )
    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else out_path.with_suffix(".jsonl.summary.json")
    )
    return out_path, progress_path, summary_path


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="e.g. Qwen/Qwen2-72B-Instruct")
    parser.add_argument("--data_source", type=str, choices=["auto", "local", "hf"], default="auto")
    parser.add_argument("--local_cuad_dir", type=str, default="data/cuad/CUAD_v1")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--backend", type=str, choices=["vllm", "transformers", "openai"], default="vllm")
    parser.add_argument("--mode", type=str, choices=["baseline", "crsr_lite"], default="baseline")
    parser.add_argument(
        "--chunk_strategy",
        type=str,
        choices=["semantic_jaccard_2gram", "window"],
        default="semantic_jaccard_2gram",
        help="分块策略：默认 semantic_jaccard_2gram（语义分块），window 为硬窗口分块（消融）",
    )
    parser.add_argument("--max_chars", type=int, default=6000)
    parser.add_argument("--overlap_chars", type=int, default=800)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--llm_concurrency", type=int, default=OPENAI_CHUNK_CONCURRENCY)
    parser.add_argument(
        "--openai_no_think_prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅 openai 后端生效：是否在 user prompt 末尾追加 /no_think",
    )
    parser.add_argument(
        "--openai_disable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅 openai 后端生效：请求里附带 enable_thinking=false",
    )
    parser.add_argument(
        "--openai_send_max_new_tokens_param",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅 openai 后端生效：请求里同步附带 max_new_tokens（兼容部分网关）",
    )
    parser.add_argument("--limit_docs", type=int, default=0)
    parser.add_argument(
        "--out_jsonl",
        type=str,
        default="data/cuad/outputs/cuad_baseline.jsonl",
        help=(
            "输出基准文件名（会自动追加主要参数+时间戳+指纹后缀），"
            "推荐使用 data/cuad/outputs/cuad_baseline.jsonl"
        ),
    )
    parser.add_argument(
        "--progress_jsonl",
        type=str,
        default="",
        help="逐步进度输出 JSONL；为空则跟随实际 out_jsonl 自动生成",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default="",
        help="运行摘要输出 JSON；为空则跟随实际 out_jsonl 自动生成",
    )
    parser.add_argument(
        "--disable_llm_usage_output",
        action="store_true",
        help="消融：不在 out_jsonl 写入 llm_usage 字段",
    )
    parser.add_argument(
        "--disable_extended_pair_metrics_output",
        action="store_true",
        help="消融：不在 out_jsonl 写入扩展统计字段（presence_confusion/evidence_jaccard）",
    )
    return parser


def main() -> None:
    """CLI 入口：执行 CUAD 推理、保存逐项结果并输出评估摘要。"""

    logger = get_logger("exp_cuad")
    logger.info(f"日志文件: {logger.path}")

    args = _build_arg_parser().parse_args()
    output_switches = OutputSwitches(
        emit_llm_usage=not args.disable_llm_usage_output,
        emit_extended_pair_metrics=not args.disable_extended_pair_metrics_output,
    )
    resolved_out_path, resolved_progress_path, resolved_summary_path = _resolve_output_paths(args)

    logger.info(
        "推理参数: model=%s, split=%s, backend=%s, mode=%s, chunk_strategy=%s, max_chars=%s, overlap_chars=%s, "
        "max_new_tokens=%s, llm_concurrency=%s, openai_no_think_prompt=%s, "
        "openai_disable_thinking=%s, openai_send_max_new_tokens_param=%s, limit_docs=%s, "
        "out_jsonl=%s, progress_jsonl=%s, "
        "summary_json=%s, resolved_out_jsonl=%s, resolved_progress_jsonl=%s, resolved_summary_json=%s, "
        "data_source=%s, local_cuad_dir=%s, output_switches=%s"
        % (
            args.model,
            args.split,
            args.backend,
            args.mode,
            args.chunk_strategy,
            args.max_chars,
            args.overlap_chars,
            args.max_new_tokens,
            args.llm_concurrency,
            args.openai_no_think_prompt,
            args.openai_disable_thinking,
            args.openai_send_max_new_tokens_param,
            args.limit_docs,
            args.out_jsonl,
            args.progress_jsonl or "(auto)",
            args.summary_json or "(auto)",
            resolved_out_path,
            resolved_progress_path,
            resolved_summary_path,
            args.data_source,
            args.local_cuad_dir,
            output_switches.to_dict(),
        )
    )

    try:
        llm_run_tasks = None
        llm_cache_stats_getter: Callable[[], dict[str, int | bool]] = lambda: {}

        if args.backend == "vllm":
            gen = llm_generate_vllm_builder(args.model)
            logger.info("后端初始化完成: backend=vllm")
        elif args.backend == "transformers":
            gen = llm_generate_transformers_builder(args.model)
            logger.info("后端初始化完成: backend=transformers")
        else:
            from src.llm import OpenAICompatibleClient, load_llm_settings, run_tasks as llm_run_tasks

            settings = load_llm_settings()
            if not settings.api_key:
                print("[info] LLM_API_KEY is empty; using unauthenticated OpenAI-compatible HTTP mode.")
                logger.info("LLM_API_KEY 为空，使用 OpenAI-compatible 匿名 HTTP 模式")

            client = OpenAICompatibleClient(settings)
            gen = llm_generate_openai_builder(
                model_name=args.model,
                client=client,
                force_no_think_prompt=args.openai_no_think_prompt,
                disable_thinking=args.openai_disable_thinking,
                send_max_new_tokens_param=args.openai_send_max_new_tokens_param,
            )
            llm_cache_stats_getter = client.cache_stats
            logger.info(
                "后端初始化完成: backend=openai, base_url=%s, model=%s, "
                "no_think_prompt=%s, disable_thinking=%s, send_max_new_tokens_param=%s"
                % (
                    settings.base_url,
                    args.model,
                    args.openai_no_think_prompt,
                    args.openai_disable_thinking,
                    args.openai_send_max_new_tokens_param,
                )
            )

        chunk_parallel_enabled = args.backend == "openai"
        chunk_max_concurrency = max(1, int(args.llm_concurrency))
        logger.info(
            "chunk 并发配置: enabled=%s, max_concurrency=%s"
            % (chunk_parallel_enabled, chunk_max_concurrency)
        )

        system = BASELINE_SYSTEM if args.mode == "baseline" else CRSR_LITE_SYSTEM
        label_desc = load_label_descriptions()
        logger.info("提示词模式加载完成: mode=%s, label_desc_count=%s" % (args.mode, len(label_desc)))

        local_dir = Path(args.local_cuad_dir)
        data_source = args.data_source
        if data_source == "auto":
            data_source = "local" if local_dir.exists() else "hf"
            logger.info(
                "自动选择数据源: selected=%s, local_exists=%s, local_dir=%s"
                % (data_source, local_dir.exists(), local_dir)
            )

        if data_source == "local":
            data = load_cuad_from_local(local_dir=local_dir, logger=logger)
        else:
            data = load_cuad_from_hf(split=args.split, logger=logger)

        raw_doc_count = len(data)
        if args.limit_docs and args.limit_docs > 0:
            data = data[: args.limit_docs]
            logger.info("应用文档数量限制: before=%s, after=%s" % (raw_doc_count, len(data)))
        else:
            logger.info("不限制文档数量: docs=%s" % raw_doc_count)

        all_labels = _collect_labels(data)
        logger.info("标签集合准备完成: label_count=%s" % len(all_labels))

        results: list[PairResult] = []
        run_llm_usage = _usage_zero()
        t0 = time.time()
        logger.info("开始推理主循环: docs=%s, labels=%s" % (len(data), len(all_labels)))

        out_path = resolved_out_path
        out_dir = out_path.parent
        if str(out_dir) not in {"", "."}:
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info("输出目录已就绪: dir=%s" % out_dir)
        else:
            logger.info("输出文件位于当前目录: file=%s" % out_path)

        progress_path = resolved_progress_path
        summary_path = resolved_summary_path

        if str(progress_path.parent) not in {"", "."}:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
        if str(summary_path.parent) not in {"", "."}:
            summary_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("进度输出文件: %s" % progress_path)
        logger.info("摘要输出文件: %s" % summary_path)

        pair_index = 0
        with out_path.open("w", encoding="utf-8", buffering=1) as out_handle, progress_path.open(
            "w", encoding="utf-8", buffering=1
        ) as progress_handle:
            progress_handle.write(
                json.dumps(
                    {
                        "event": "run_start",
                        "mode": args.mode,
                        "backend": args.backend,
                        "chunk_strategy": args.chunk_strategy,
                        "model": args.model,
                        "openai_no_think_prompt": args.openai_no_think_prompt if args.backend == "openai" else None,
                        "openai_disable_thinking": args.openai_disable_thinking if args.backend == "openai" else None,
                        "openai_send_max_new_tokens_param": (
                            args.openai_send_max_new_tokens_param if args.backend == "openai" else None
                        ),
                        "data_source": data_source,
                        "docs": len(data),
                        "labels": len(all_labels),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            progress_handle.flush()

            for doc_index, ex in enumerate(data, start=1):
                chunks = _chunk_document(
                    text=ex.text,
                    chunk_strategy=args.chunk_strategy,
                    max_chars=args.max_chars,
                    overlap_chars=args.overlap_chars,
                    logger=logger,
                    doc_id=ex.doc_id,
                )
                logger.info(
                    "文档开始: doc_index=%s/%s, doc_id=%s, text_chars=%s, chunk_count=%s"
                    % (doc_index, len(data), ex.doc_id, len(ex.text), len(chunks))
                )

                for lab_index, lab in enumerate(all_labels, start=1):
                    pair_t0 = time.time()
                    spans = ex.spans.get(lab, [])
                    present_gt = len(spans) > 0
                    evidence_gt = spans_to_evidence_sentences(ex.text, spans)
                    logger.info(
                        "标签开始: doc_id=%s, label_index=%s/%s, label=%s, gt_present=%s, gt_span_count=%s, gt_evidence_count=%s"
                        % (ex.doc_id, lab_index, len(all_labels), lab, present_gt, len(spans), len(evidence_gt))
                    )

                    chunk_jobs: list[dict[str, Any]] = []
                    for chunk_index, chunk_text in enumerate(chunks, start=1):
                        chunk_jobs.append(
                            {
                                "chunk_index": chunk_index,
                                "chunk_text": chunk_text,
                                "user_prompt": _build_user_prompt(
                                    args.mode,
                                    lab,
                                    label_desc.get(lab),
                                    chunk_text,
                                ),
                            }
                        )

                    chunk_outputs: list[dict] = [{} for _ in chunks]
                    pair_usage = _usage_zero()

                    batch_size = chunk_max_concurrency if chunk_parallel_enabled else 1
                    for batch_start in range(0, len(chunk_jobs), batch_size):
                        batch_jobs = chunk_jobs[batch_start : batch_start + batch_size]

                        def _worker(job: dict[str, Any]) -> dict[str, Any]:
                            result = infer_single_chunk_with_retry(
                                gen=gen,
                                system=system,
                                user=job["user_prompt"],
                                max_new_tokens=args.max_new_tokens,
                                logger=logger,
                                doc_id=ex.doc_id,
                                label=lab,
                                chunk_index=int(job["chunk_index"]),
                                chunk_chars=len(str(job["chunk_text"])),
                            )
                            result["chunk_index"] = int(job["chunk_index"])
                            return result

                        if chunk_parallel_enabled:
                            batch_results = llm_run_tasks(
                                batch_jobs,
                                _worker,
                                concurrent_enabled=True,
                                max_concurrency=chunk_max_concurrency,
                            )
                        else:
                            batch_results = [_worker(job) for job in batch_jobs]

                        for res in batch_results:
                            chunk_index = int(res["chunk_index"])
                            obj = res["obj"]
                            trace: ChunkTrace = res["trace"]
                            usage = res["usage"]

                            chunk_outputs[chunk_index - 1] = obj
                            _usage_merge(pair_usage, usage)

                            progress_handle.write(
                                json.dumps(
                                    {
                                        "event": "chunk_result",
                                        "doc_index": doc_index,
                                        "doc_total": len(data),
                                        "doc_id": ex.doc_id,
                                        "label_index": lab_index,
                                        "label_total": len(all_labels),
                                        "label": lab,
                                        "chunk_index": chunk_index,
                                        "chunk_total": len(chunks),
                                        "present": trace.present,
                                        "evidence_count": trace.evidence_count,
                                        "json_parse_failed": trace.json_parse_failed,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            progress_handle.flush()

                            logger.info(
                                "chunk 推理完成: doc_id=%s, label=%s, chunk_index=%s/%s, chunk_chars=%s, present=%s, evidence_count=%s, latency_ms=%.2f"
                                % (
                                    ex.doc_id,
                                    lab,
                                    chunk_index,
                                    len(chunks),
                                    trace.chunk_chars,
                                    trace.present,
                                    trace.evidence_count,
                                    trace.total_latency_ms,
                                )
                            )
                            if trace.json_retried and not trace.json_parse_failed:
                                logger.info(
                                    "JSON 解析重试成功: doc_id=%s, label=%s, chunk_index=%s/%s, retry_max_tokens=%s, retry_latency_ms=%.2f"
                                    % (
                                        ex.doc_id,
                                        lab,
                                        chunk_index,
                                        len(chunks),
                                        trace.retry_tokens,
                                        trace.retry_ms,
                                    )
                                )
                            if trace.json_parse_failed:
                                logger.error(
                                    "JSON 解析重试仍失败: doc_id=%s, label=%s, chunk_index=%s/%s"
                                    % (ex.doc_id, lab, chunk_index, len(chunks))
                                )

                    present_pred, evidence_pred, votes = aggregate_chunks(chunk_outputs)

                    pair_result = PairResult(
                        doc_id=ex.doc_id,
                        label=lab,
                        present_pred=present_pred,
                        evidence_pred=evidence_pred,
                        present_gt=present_gt,
                        evidence_gt=evidence_gt,
                        chunk_votes=votes,
                    )
                    results.append(pair_result)

                    pair_index += 1
                    _usage_merge(run_llm_usage, pair_usage)

                    pair_record = _build_pair_record(
                        pair_index=pair_index,
                        ex=ex,
                        label=lab,
                        present_pred=present_pred,
                        evidence_pred=evidence_pred,
                        present_gt=present_gt,
                        evidence_gt=evidence_gt,
                        votes=votes,
                        mode=args.mode,
                        backend=args.backend,
                        output_switches=output_switches,
                        pair_usage=pair_usage,
                    )
                    out_handle.write(json.dumps(pair_record, ensure_ascii=False) + "\n")
                    out_handle.flush()

                    progress_handle.write(
                        json.dumps(
                            {
                                "event": "pair_result",
                                "pair_index": pair_index,
                                "doc_index": doc_index,
                                "doc_total": len(data),
                                "doc_id": ex.doc_id,
                                "label_index": lab_index,
                                "label_total": len(all_labels),
                                "label": lab,
                                "present_pred": present_pred,
                                "present_gt": present_gt,
                                "evidence_pred_count": len(evidence_pred),
                                "evidence_gt_count": len(evidence_gt),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    progress_handle.flush()

                    logger.info(
                        "标签完成: doc_id=%s, label=%s, pred_present=%s, pred_evidence_count=%s, vote_true=%s/%s, elapsed_ms=%.2f"
                        % (
                            ex.doc_id,
                            lab,
                            present_pred,
                            len(evidence_pred),
                            sum(1 for vote in votes if vote),
                            len(votes),
                            (time.time() - pair_t0) * 1000.0,
                        )
                    )

                logger.info(
                    "文档完成: doc_index=%s/%s, doc_id=%s, cumulative_pairs=%s"
                    % (doc_index, len(data), ex.doc_id, len(results))
                )
                if doc_index % 5 == 0:
                    print(f"[progress] {doc_index}/{len(data)} docs processed")
                    logger.info("阶段进度: docs_processed=%s/%s" % (doc_index, len(data)))
                    progress_handle.write(
                        json.dumps(
                            {
                                "event": "doc_progress",
                                "docs_processed": doc_index,
                                "docs_total": len(data),
                                "pairs_written": pair_index,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    progress_handle.flush()

        dt = time.time() - t0
        logger.info("推理主循环结束: time_sec=%.3f, pair_count=%s" % (dt, len(results)))

        micro = compute_presence_f1(results)
        macro = compute_macro_f1(results)
        jac = compute_evidence_jaccard(results)
        lazy = compute_laziness_rate(results)
        incon = compute_inconsistency_rate(results)

        summary = {
            "mode": args.mode,
            "backend": args.backend,
            "chunk_strategy": args.chunk_strategy,
            "model": args.model,
            "data_source": data_source,
            "local_cuad_dir": str(local_dir),
            "split": args.split,
            "docs": len(data),
            "pairs": len(results),
            "time_sec": dt,
            "presence_micro": micro,
            "presence_macro_f1": macro["macro_f1"],
            "evidence_jaccard_on_present_gt": jac,
            "laziness_rate": lazy,
            "inconsistency_rate": incon,
            "llm_usage_aggregate": run_llm_usage,
            "llm_cache_stats": llm_cache_stats_getter(),
            "out_jsonl": str(out_path),
            "summary_json": str(summary_path),
        }

        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("评估完成: %s" % json.dumps(summary, ensure_ascii=False))

        with progress_path.open("a", encoding="utf-8", buffering=1) as progress_handle:
            progress_handle.write(json.dumps({"event": "run_end", "summary": summary}, ensure_ascii=False) + "\n")
            progress_handle.flush()

        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001
        logger.error(f"exp_cuad 执行失败: {exc}")
        raise


if __name__ == "__main__":
    main()
