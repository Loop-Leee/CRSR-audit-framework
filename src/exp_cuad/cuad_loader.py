# src/exp_cuad/cuad_loader.py
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha1
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple

from src.tools.logger import Logger

@dataclass
class CuadExample:
    """CUAD 单条样本的标准化表示。"""
    doc_id: str
    text: str
    # label -> list of (start_char, end_char)
    spans: Dict[str, List[Tuple[int, int]]]

def _safe_get(d: dict, key: str, default=None):
    """安全读取字典键，不存在时返回默认值。"""
    return d[key] if key in d else default

def _looks_like_qa_row(row: dict) -> bool:
    """判断样本是否是 SQuAD 风格 QA 行（含 context/question/answers）。"""
    return isinstance(row.get("answers"), dict) and isinstance(row.get("context"), str)

def _extract_label_from_question(question: str) -> str:
    """从 question 文本中提取条款标签名。"""
    q = str(question or "").strip()
    if not q:
        return ""
    # CUAD 模板通常是 ... related to "Label" ...
    m = re.search(r'"([^"]+)"', q)
    if m:
        return m.group(1).strip()
    return q

def _extract_label_from_qa_row(row: dict) -> str:
    """从 QA 行中提取标签名，优先解析 question，再回退 id。"""
    label = _extract_label_from_question(str(_safe_get(row, "question", "")))
    if label:
        return label

    row_id = str(_safe_get(row, "id", ""))
    if "__" in row_id:
        return row_id.split("__", 1)[1].strip()
    return row_id.strip()

def _extract_spans_from_answers(answers: dict) -> list[tuple[int, int]]:
    """从 answers 字段中提取字符区间 spans。"""
    texts = answers.get("text", []) or []
    starts = answers.get("start_idx", answers.get("answer_start", [])) or []
    spans: list[tuple[int, int]] = []
    for st, txt in zip(starts, texts):
        if st is None:
            continue
        t = str(txt or "")
        st_i = int(st)
        ed_i = st_i + len(t)
        if ed_i > st_i:
            spans.append((st_i, ed_i))
    return spans

def load_cuad_from_local(local_dir: str | Path, logger: Logger | None = None) -> List[CuadExample]:
    """
    从本地 CUAD_v1 数据读取样本并转换为 ``CuadExample``。

    Args:
        local_dir: 本地 CUAD 目录（应包含 ``CUAD_v1.json``）。
        logger: 可选日志器。

    Returns:
        标准化后的 ``CuadExample`` 列表。
    """
    root = Path(local_dir).expanduser()
    json_path = root / "CUAD_v1.json"
    if root.is_file():
        json_path = root
        root = root.parent

    if logger:
        logger.info("开始加载本地 CUAD: root=%s, json=%s" % (root, json_path))

    if not json_path.exists():
        raise FileNotFoundError(f"未找到本地 CUAD 文件: {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError(f"本地 CUAD 格式错误，缺少 data 列表: {json_path}")

    examples: list[CuadExample] = []
    total_qas = 0
    total_spans = 0
    label_set: set[str] = set()

    for doc in data:
        doc_id = str(doc.get("title") or "").strip()
        if not doc_id:
            doc_id = "cuad_" + sha1(json.dumps(doc, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]

        paragraphs = doc.get("paragraphs", []) or []
        text = ""
        spans: dict[str, list[tuple[int, int]]] = {}

        for para in paragraphs:
            if not text:
                text = str(para.get("context") or "")
            qas = para.get("qas", []) or []
            for qa in qas:
                total_qas += 1
                label = _extract_label_from_qa_row(qa)
                if not label:
                    continue
                answers = qa.get("answers", []) or []
                if isinstance(answers, dict):
                    answer_spans = _extract_spans_from_answers(answers)
                else:
                    answer_spans = []
                    for a in answers:
                        if not isinstance(a, dict):
                            continue
                        st = a.get("answer_start", a.get("start_idx"))
                        txt = str(a.get("text", "") or "")
                        if st is None:
                            continue
                        st_i = int(st)
                        ed_i = st_i + len(txt)
                        if ed_i > st_i:
                            answer_spans.append((st_i, ed_i))
                spans.setdefault(label, []).extend(answer_spans)

        for label, ss in spans.items():
            spans[label] = sorted(set(ss))
            label_set.add(label)
            total_spans += len(spans[label])

        examples.append(CuadExample(doc_id=doc_id, text=text, spans=spans))

    if logger:
        logger.info(
            "本地 CUAD 加载完成: docs=%s, labels=%s, qas=%s, spans=%s, source=%s"
            % (len(examples), len(label_set), total_qas, total_spans, json_path)
        )
    return examples

def _load_hf_split_with_retry(dataset_id: str, split: str, logger: Logger | None = None):
    """
    加载 HF split，并在缓存 split-size 校验不一致时自动强制重下。

    典型错误：`NonMatchingSplitsSizesError`（本地缓存与远端数据元信息不一致）。
    """
    from datasets import load_dataset

    requested_split = split
    if split in {"test", "val", "valid", "validation", "dev"}:
        # theatticusproject/cuad 当前公开为单 split(train)
        requested_split = "train"
        if logger:
            logger.info(
                "CUAD split 回退: requested_split=%s -> actual_split=%s"
                % (split, requested_split)
            )

    try:
        if logger:
            logger.info(
                "开始加载 CUAD 数据集: dataset=%s, split=%s"
                % (dataset_id, requested_split)
            )
        return load_dataset(dataset_id, split=requested_split)
    except Exception as exc:
        if exc.__class__.__name__ != "NonMatchingSplitsSizesError":
            if logger:
                logger.error(
                    "CUAD 数据加载失败: dataset=%s, split=%s, error=%s"
                    % (dataset_id, requested_split, exc)
                )
            raise
        try:
            if logger:
                logger.info(
                    "检测到 split size 不一致，尝试强制重下: dataset=%s, split=%s"
                    % (dataset_id, requested_split)
                )
            return load_dataset(dataset_id, split=requested_split, download_mode="force_redownload")
        except Exception as retry_exc:
            if logger:
                logger.error(
                    "CUAD 强制重下失败: dataset=%s, split=%s, error=%s"
                    % (dataset_id, requested_split, retry_exc)
                )
            raise RuntimeError(
                "Failed to load CUAD after force redownload. "
                "Your local HF datasets cache may be corrupted. "
                "Try deleting ~/.cache/huggingface/datasets/theatticusproject___cuad and rerun."
            ) from retry_exc

def load_cuad_from_hf(split: str = "test", logger: Logger | None = None) -> List[CuadExample]:
    """
    从 HuggingFace 加载 CUAD 数据，并做字段兼容与容错解析。

    Args:
        split: 数据集切分名称，如 ``test``、``train``。

    Returns:
        标准化后的 ``CuadExample`` 列表。
    """
    try:
        import datasets  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'datasets'. "
            "Install with: conda install -n crsr-audit -c conda-forge datasets"
        ) from exc
    ds = _load_hf_split_with_retry("theatticusproject/cuad", split=split, logger=logger)
    if logger:
        logger.info("CUAD split 加载成功: split=%s, rows=%s" % (split, len(ds)))

    # QA 结构（context/question/answers）需要先聚合为文档级标签集合。
    grouped_by_doc: dict[str, CuadExample] = {}
    qa_rows = 0
    legacy_rows = 0
    skipped_rows = 0
    for row in ds:
        if _looks_like_qa_row(row):
            qa_rows += 1
            text = str(row.get("context") or "")
            if not text:
                skipped_rows += 1
                continue
            doc_hash = sha1(text.encode("utf-8")).hexdigest()[:16]
            doc_id = f"cuad_{doc_hash}"
            ex = grouped_by_doc.get(doc_id)
            if ex is None:
                ex = CuadExample(doc_id=doc_id, text=text, spans={})
                grouped_by_doc[doc_id] = ex

            label = _extract_label_from_qa_row(row)
            if label:
                ex.spans.setdefault(label, []).extend(_extract_spans_from_answers(row.get("answers", {})))
            continue

        legacy_rows += 1
        doc_id = str(_safe_get(row, "document_id", _safe_get(row, "doc_id", _safe_get(row, "id", ""))))
        text = _safe_get(row, "text", _safe_get(row, "contract", ""))
        spans: Dict[str, List[Tuple[int, int]]] = {}

        labels = _safe_get(row, "labels", None)
        if labels and isinstance(labels, dict):
            for label, items in labels.items():
                spans[label] = []
                for it in items or []:
                    if isinstance(it, (list, tuple)) and len(it) == 2:
                        spans[label].append((int(it[0]), int(it[1])))
                    elif isinstance(it, dict):
                        st = it.get("start", it.get("start_char"))
                        ed = it.get("end", it.get("end_char"))
                        if st is not None and ed is not None:
                            spans[label].append((int(st), int(ed)))

        ann = _safe_get(row, "annotations", None)
        if ann and isinstance(ann, list) and not spans:
            for it in ann:
                label = it.get("label")
                st = it.get("start", it.get("start_char"))
                ed = it.get("end", it.get("end_char"))
                if label and st is not None and ed is not None:
                    spans.setdefault(label, []).append((int(st), int(ed)))

        grouped_by_doc[doc_id] = CuadExample(doc_id=doc_id, text=text, spans=spans)

    # QA 聚合模式下，同一 (doc,label) 可能重复追加，最终做一次去重和排序。
    label_set: set[str] = set()
    total_spans = 0
    for ex in grouped_by_doc.values():
        for label, ss in ex.spans.items():
            ex.spans[label] = sorted(set(ss))
            label_set.add(label)
            total_spans += len(ex.spans[label])

    examples = list(grouped_by_doc.values())
    if logger:
        logger.info(
            "CUAD 样本标准化完成: docs=%s, labels=%s, spans=%s, qa_rows=%s, legacy_rows=%s, skipped_rows=%s"
            % (len(examples), len(label_set), total_spans, qa_rows, legacy_rows, skipped_rows)
        )
    return examples

def load_label_descriptions() -> dict[str, str]:
    """
    返回标签描述映射（label -> description）。

    当前为最小可用占位实现，可后续替换为官方描述文件读取逻辑。
    """
    # 可选：你可以把官方的 descriptions 文件放到 data/ 下，然后在这里读取
    return {}
