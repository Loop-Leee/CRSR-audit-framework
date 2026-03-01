# src/exp_cuad/eval_metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

from .text_utils import split_sentences, sentence_set_jaccard, normalize_ws

@dataclass
class PairResult:
    """单个 (doc_id, label) 预测结果与标注结果的对齐记录。"""
    doc_id: str
    label: str
    present_pred: bool
    evidence_pred: list[str]
    present_gt: bool
    evidence_gt: list[str]
    chunk_votes: list[bool]  # per chunk present

def spans_to_evidence_sentences(text: str, spans: list[tuple[int,int]]) -> list[str]:
    """将字符区间标注粗略映射为证据句列表并去重。"""
    if not spans:
        return []
    sents = split_sentences(text)
    # 以字符区间映射到句子：简单做法是判断 span 覆盖的子串是否出现在句子中
    # 更严谨可以做 char offset 到句子边界映射，但这里先保证可跑可复现
    ev = []
    for st, ed in spans:
        frag = normalize_ws(text[st:ed])
        if not frag:
            continue
        for s in sents:
            if frag[:30] in s or frag in s:
                ev.append(s)
                break
    # 去重
    uniq = []
    seen = set()
    for e in ev:
        ne = normalize_ws(e)
        if ne not in seen:
            uniq.append(e)
            seen.add(ne)
    return uniq

def compute_presence_f1(results: list[PairResult]) -> dict[str, float]:
    """基于所有样本计算 presence 二分类 micro 指标。"""
    tp = fp = fn = tn = 0
    for r in results:
        if r.present_pred and r.present_gt:
            tp += 1
        elif r.present_pred and not r.present_gt:
            fp += 1
        elif (not r.present_pred) and r.present_gt:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def compute_macro_f1(results: list[PairResult]) -> dict[str, float]:
    """按 label(条款类型) 计算 F1 后取平均，返回 macro-F1。"""
    by_label: dict[str, list[PairResult]] = defaultdict(list)
    for r in results:
        by_label[r.label].append(r)
    f1s = []
    for lab, rs in by_label.items():
        f1s.append(compute_presence_f1(rs)["f1"])
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"macro_f1": macro_f1, "num_labels": len(f1s)}

def compute_evidence_jaccard(results: list[PairResult]) -> float:
    """在 GT 为 present 的样本上计算证据句集合 Jaccard 均值。"""
    js = []
    for r in results:
        # 只在 GT present 的样本上算更合理；否则证据为空会导致“高分假象”
        if r.present_gt:
            js.append(sentence_set_jaccard(r.evidence_pred, r.evidence_gt))
    return sum(js) / len(js) if js else 0.0

def compute_laziness_rate(results: list[PairResult]) -> float:
    """计算 GT present 但模型预测 absent 的比例。"""
    # GT present 但预测 absent 的比例
    miss = 0
    total = 0
    for r in results:
        if r.present_gt:
            total += 1
            if not r.present_pred:
                miss += 1
    return miss / total if total else 0.0

def compute_inconsistency_rate(results: list[PairResult]) -> float:
    """计算 chunk 级投票同时出现 True/False 的不一致比例。"""
    # 若同一 (doc,label) 的 chunk_votes 同时出现 True 和 False，则记为不一致
    bad = 0
    total = 0
    for r in results:
        total += 1
        if r.chunk_votes and (any(r.chunk_votes) and (not all(r.chunk_votes))):
            bad += 1
    return bad / total if total else 0.0
