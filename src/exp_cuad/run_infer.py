# src/exp_cuad/run_infer.py
from __future__ import annotations
import ast
import argparse, json, time
from pathlib import Path
import platform
import re
from typing import Any, Dict, List

from src.tools.logger import Logger, get_logger
from .cuad_loader import load_cuad_from_hf, load_cuad_from_local, load_label_descriptions
from .text_utils import chunk_by_tokens_approx, split_sentences, normalize_ws
from .prompts import (
    BASELINE_SYSTEM, CRSR_LITE_SYSTEM,
    make_baseline_user_prompt, make_crsr_lite_user_prompt
)
from .eval_metrics import (
    PairResult,
    spans_to_evidence_sentences,
    compute_presence_f1, compute_macro_f1,
    compute_evidence_jaccard, compute_laziness_rate, compute_inconsistency_rate
)

OPENAI_CHUNK_CONCURRENCY = 10

def llm_generate_transformers(model_name: str, system: str, user: str, max_new_tokens: int = 256) -> str:
    """使用 transformers 后端执行一次聊天生成并返回文本。"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependencies for transformers backend. "
            "Install with: conda install -n crsr-audit -c conda-forge transformers pytorch sentencepiece accelerate"
        ) from exc

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    # 取最后一个 JSON 块（粗暴但可用）
    return text

def llm_generate_vllm(model_name: str, system: str, user: str, max_new_tokens: int = 256) -> str:
    """使用 vLLM 后端执行一次聊天生成并返回文本。"""
    try:
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:
        os_hint = ""
        if platform.system() == "Darwin":
            os_hint = " vLLM is generally unavailable on macOS; use --backend transformers instead."
        raise ModuleNotFoundError(
            "Missing dependency 'vllm'. "
            "Install on Linux with GPU: pip install vllm." + os_hint
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
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    out = llm.generate([prompt], params)[0].outputs[0].text
    return out

def llm_generate_openai_compatible(
    model_name: str,
    system: str,
    user: str,
    max_new_tokens: int = 256,
    *,
    client: Any,
) -> str:
    """使用 OpenAI 兼容客户端执行聊天生成并返回文本。"""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return client.chat(messages, model=model_name, max_tokens=max_new_tokens)

_JSON_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

def _normalize_pred_object(obj: dict) -> dict | None:
    """把可能的嵌套对象统一映射为 {present, evidence} 结构。"""
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

    for m in _JSON_CODEBLOCK_RE.finditer(text):
        block = m.group(1).strip()
        if block:
            candidates.append(block)

    # 提取平衡花括号片段
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

    # 去重但保序
    uniq: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        cc = c.strip()
        if cc and cc not in seen:
            uniq.append(cc)
            seen.add(cc)
    return uniq

def _parse_candidate_json(candidate: str) -> dict | None:
    """尝试多种容错方式解析候选片段为 dict。"""
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    relaxed = re.sub(r",(\s*[}\]])", r"\1", candidate)
    try:
        parsed = json.loads(relaxed)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    py_like = re.sub(r"\btrue\b", "True", relaxed, flags=re.IGNORECASE)
    py_like = re.sub(r"\bfalse\b", "False", py_like, flags=re.IGNORECASE)
    py_like = re.sub(r"\bnull\b", "None", py_like, flags=re.IGNORECASE)
    try:
        parsed = ast.literal_eval(py_like)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
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
    votes = []
    evidence_all = []
    for o in chunk_outputs:
        p = bool(o.get("present", False))
        votes.append(p)
        ev = o.get("evidence", []) or []
        for e in ev:
            ne = normalize_ws(e)
            if ne and ne not in {normalize_ws(x) for x in evidence_all}:
                evidence_all.append(e)
    present = any(votes)
    return present, evidence_all, votes

def infer_single_chunk_with_retry(
    *,
    gen: Any,
    model_name: str,
    system: str,
    user: str,
    max_new_tokens: int,
    logger: Logger,
    doc_id: str,
    label: str,
    chunk_index: int,
) -> dict:
    """执行单个 chunk 推理，并在 JSON 解析失败时重试一次。"""
    infer_t0 = time.time()
    raw = gen(model_name, system, user, max_new_tokens=max_new_tokens)
    infer_ms = (time.time() - infer_t0) * 1000.0
    obj = parse_json_loose(raw)
    json_retried = False
    retry_tokens = 0
    retry_ms = 0.0

    if "_raw" in obj:
        retry_tokens = max(max_new_tokens, 512)
        retry_t0 = time.time()
        raw_retry = gen(model_name, system, user, max_new_tokens=retry_tokens)
        retry_ms = (time.time() - retry_t0) * 1000.0
        obj_retry = parse_json_loose(raw_retry)
        json_retried = True
        if "_raw" not in obj_retry:
            obj = obj_retry
        else:
            obj = parse_json_loose_with_logging(
                raw_retry,
                logger=logger,
                doc_id=doc_id,
                label=label,
                chunk_index=chunk_index,
            )

    obj["present"] = bool(obj.get("present", False))
    ev = obj.get("evidence", [])
    obj["evidence"] = [x for x in ev if isinstance(x, str)]
    return {
        "obj": obj,
        "infer_ms": infer_ms,
        "json_retried": json_retried,
        "retry_tokens": retry_tokens,
        "retry_ms": retry_ms,
    }

def main():
    """CLI 入口：执行 CUAD 推理、保存逐项结果并输出评估摘要。"""
    logger = get_logger("exp_cuad")
    logger.info(f"日志文件: {logger.path}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="e.g. Qwen/Qwen2-72B-Instruct")
    ap.add_argument("--data_source", type=str, choices=["auto", "local", "hf"], default="auto")
    ap.add_argument("--local_cuad_dir", type=str, default="data/cuad/CUAD_v1")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--backend", type=str, choices=["vllm", "transformers", "openai"], default="vllm")
    ap.add_argument("--mode", type=str, choices=["baseline", "crsr_lite"], default="baseline")
    ap.add_argument("--max_chars", type=int, default=6000)
    ap.add_argument("--overlap_chars", type=int, default=800)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--llm_concurrency", type=int, default=OPENAI_CHUNK_CONCURRENCY)
    ap.add_argument("--limit_docs", type=int, default=0)
    ap.add_argument("--out_jsonl", type=str, default="cuad_preds.jsonl")
    ap.add_argument(
        "--progress_jsonl",
        type=str,
        default="",
        help="逐步进度输出 JSONL；为空则使用 <out_jsonl>.progress.jsonl",
    )
    args = ap.parse_args()

    logger.info(
        "推理参数: model=%s, split=%s, backend=%s, mode=%s, max_chars=%s, overlap_chars=%s, "
        "max_new_tokens=%s, llm_concurrency=%s, limit_docs=%s, out_jsonl=%s, progress_jsonl=%s, data_source=%s, local_cuad_dir=%s"
        % (
            args.model,
            args.split,
            args.backend,
            args.mode,
            args.max_chars,
            args.overlap_chars,
            args.max_new_tokens,
            args.llm_concurrency,
            args.limit_docs,
            args.out_jsonl,
            args.progress_jsonl or "(auto)",
            args.data_source,
            args.local_cuad_dir,
        )
    )

    try:
        llm_run_tasks = None
        if args.backend == "vllm":
            gen = llm_generate_vllm
            logger.info("后端初始化完成: backend=vllm")
        elif args.backend == "transformers":
            gen = llm_generate_transformers
            logger.info("后端初始化完成: backend=transformers")
        else:
            from src.llm import OpenAICompatibleClient, load_llm_settings, run_tasks as llm_run_tasks
            settings = load_llm_settings()
            if not settings.api_key:
                print("[info] LLM_API_KEY is empty; using unauthenticated OpenAI-compatible HTTP mode.")
                logger.info("LLM_API_KEY 为空，使用 OpenAI-compatible 匿名 HTTP 模式")
            client = OpenAICompatibleClient(settings)
            logger.info(
                "后端初始化完成: backend=openai, base_url=%s, model=%s"
                % (settings.base_url, args.model)
            )

            def gen(model_name: str, system: str, user: str, max_new_tokens: int = 256) -> str:
                """绑定 OpenAI 兼容客户端后的生成函数。"""
                return llm_generate_openai_compatible(
                    model_name=model_name,
                    system=system,
                    user=user,
                    max_new_tokens=max_new_tokens,
                    client=client,
                )

        chunk_parallel_enabled = args.backend == "openai"
        chunk_max_concurrency = max(1, int(args.llm_concurrency))
        logger.info(
            "chunk 并发配置: enabled=%s, max_concurrency=%s"
            % (chunk_parallel_enabled, chunk_max_concurrency)
        )

        system = BASELINE_SYSTEM if args.mode == "baseline" else CRSR_LITE_SYSTEM
        label_desc = load_label_descriptions()  # 你可后续补充官方描述
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
            data = data[:args.limit_docs]
            logger.info("应用文档数量限制: before=%s, after=%s" % (raw_doc_count, len(data)))
        else:
            logger.info("不限制文档数量: docs=%s" % raw_doc_count)

        # 收集所有出现过的 label（CUAD 41 类）
        all_labels = set()
        for ex in data:
            all_labels.update(ex.spans.keys())
        all_labels = sorted(list(all_labels))
        logger.info("标签集合准备完成: label_count=%s" % len(all_labels))

        results: list[PairResult] = []
        t0 = time.time()
        logger.info("开始推理主循环: docs=%s, labels=%s" % (len(data), len(all_labels)))
        out_path = Path(args.out_jsonl)
        out_dir = out_path.parent
        if str(out_dir) not in {"", "."}:
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info("输出目录已就绪: dir=%s" % out_dir)
        else:
            logger.info("输出文件位于当前目录: file=%s" % out_path)

        progress_path = Path(args.progress_jsonl) if args.progress_jsonl else Path(str(out_path) + ".progress.jsonl")
        progress_dir = progress_path.parent
        if str(progress_dir) not in {"", "."}:
            progress_dir.mkdir(parents=True, exist_ok=True)
        logger.info("进度输出文件: %s" % progress_path)

        pair_index = 0
        with open(out_path, "w", encoding="utf-8", buffering=1) as f, open(
            progress_path, "w", encoding="utf-8", buffering=1
        ) as pf:
            pf.write(
                json.dumps(
                    {
                        "event": "run_start",
                        "mode": args.mode,
                        "backend": args.backend,
                        "model": args.model,
                        "data_source": data_source,
                        "docs": len(data),
                        "labels": len(all_labels),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            pf.flush()

            for i, ex in enumerate(data):
                doc_index = i + 1
                chunks = chunk_by_tokens_approx(ex.text, max_chars=args.max_chars, overlap_chars=args.overlap_chars)
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

                    chunk_jobs = []
                    for chunk_index, ch in enumerate(chunks, start=1):
                        if args.mode == "baseline":
                            user = make_baseline_user_prompt(lab, label_desc.get(lab), ch)
                        else:
                            user = make_crsr_lite_user_prompt(lab, label_desc.get(lab), ch)
                        chunk_jobs.append(
                            {
                                "chunk_index": chunk_index,
                                "chunk_text": ch,
                                "user_prompt": user,
                            }
                        )

                    chunk_outputs: list[dict] = [{} for _ in chunks]

                    # OpenAI 后端用 llm.run_tasks 并发执行；其余后端保留串行，避免模型对象线程安全问题。
                    batch_size = chunk_max_concurrency if chunk_parallel_enabled else 1
                    for batch_start in range(0, len(chunk_jobs), batch_size):
                        batch_jobs = chunk_jobs[batch_start : batch_start + batch_size]

                        def _worker(job: dict) -> dict:
                            result = infer_single_chunk_with_retry(
                                gen=gen,
                                model_name=args.model,
                                system=system,
                                user=job["user_prompt"],
                                max_new_tokens=args.max_new_tokens,
                                logger=logger,
                                doc_id=ex.doc_id,
                                label=lab,
                                chunk_index=job["chunk_index"],
                            )
                            result["chunk_index"] = job["chunk_index"]
                            result["chunk_text"] = job["chunk_text"]
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
                            ch = str(res["chunk_text"])
                            obj = res["obj"]
                            infer_ms = float(res["infer_ms"])
                            json_retried = bool(res["json_retried"])
                            retry_tokens = int(res["retry_tokens"])
                            retry_ms = float(res["retry_ms"])

                            if json_retried and "_raw" not in obj:
                                logger.info(
                                    "JSON 解析重试成功: doc_id=%s, label=%s, chunk_index=%s/%s, retry_max_tokens=%s, retry_latency_ms=%.2f"
                                    % (
                                        ex.doc_id,
                                        lab,
                                        chunk_index,
                                        len(chunks),
                                        retry_tokens,
                                        retry_ms,
                                    )
                                )

                            chunk_outputs[chunk_index - 1] = obj
                            pf.write(
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
                                        "present": obj["present"],
                                        "evidence_count": len(obj["evidence"]),
                                        "json_parse_failed": "_raw" in obj,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            pf.flush()
                            logger.info(
                                "chunk 推理完成: doc_id=%s, label=%s, chunk_index=%s/%s, chunk_chars=%s, present=%s, evidence_count=%s, latency_ms=%.2f"
                                % (
                                    ex.doc_id,
                                    lab,
                                    chunk_index,
                                    len(chunks),
                                    len(ch),
                                    obj["present"],
                                    len(obj["evidence"]),
                                    infer_ms,
                                )
                            )
                            if json_retried and "_raw" in obj:
                                logger.error(
                                    "JSON 解析重试仍失败: doc_id=%s, label=%s, chunk_index=%s/%s"
                                    % (ex.doc_id, lab, chunk_index, len(chunks))
                                )

                    present_pred, evidence_pred, votes = aggregate_chunks(chunk_outputs)

                    pr = PairResult(
                        doc_id=ex.doc_id,
                        label=lab,
                        present_pred=present_pred,
                        evidence_pred=evidence_pred,
                        present_gt=present_gt,
                        evidence_gt=evidence_gt,
                        chunk_votes=votes,
                    )
                    results.append(pr)

                    pair_index += 1
                    f.write(json.dumps({
                        "pair_index": pair_index,
                        "doc_id": ex.doc_id,
                        "label": lab,
                        "present_pred": present_pred,
                        "evidence_pred": evidence_pred,
                        "present_gt": present_gt,
                        "evidence_gt": evidence_gt,
                        "chunk_votes": votes,
                        "mode": args.mode,
                        "backend": args.backend
                    }, ensure_ascii=False) + "\n")
                    f.flush()
                    pf.write(
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
                    pf.flush()

                    logger.info(
                        "标签完成: doc_id=%s, label=%s, pred_present=%s, pred_evidence_count=%s, vote_true=%s/%s, elapsed_ms=%.2f"
                        % (
                            ex.doc_id,
                            lab,
                            present_pred,
                            len(evidence_pred),
                            sum(1 for v in votes if v),
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
                    pf.write(
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
                    pf.flush()

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
            "out_jsonl": args.out_jsonl
        }
        logger.info("评估完成: %s" % json.dumps(summary, ensure_ascii=False))
        with open(progress_path, "a", encoding="utf-8", buffering=1) as pf:
            pf.write(json.dumps({"event": "run_end", "summary": summary}, ensure_ascii=False) + "\n")
            pf.flush()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        logger.error(f"exp_cuad 执行失败: {exc}")
        raise

if __name__ == "__main__":
    main()
