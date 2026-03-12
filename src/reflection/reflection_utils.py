"""reflection 通用工具。"""

from __future__ import annotations

import json
import re
from typing import Any

from .reflection_models import EvidenceWindow, SentenceSpan


_JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*\}")

_GENERIC_SPAN_HINTS = (
    "本合同",
    "本协议",
    "双方协商",
    "有关事项",
    "按约定",
    "相关约定",
    "另行约定",
    "双方确认",
)
_WEAK_SUGGEST_HINTS = (
    "建议进一步确认",
    "建议核实",
    "建议确认",
    "建议复核",
    "建议进一步核实",
    "建议补充说明",
)
_VAGUE_RULE_HIT_HINTS = (
    "存在风险",
    "可能不符合",
    "需关注",
    "有待确认",
    "应进一步核实",
    "相关约定",
)
_SENTENCE_DELIMITERS = {"。", "！", "？", "!", "?", "；", ";", "\n"}


def split_sentences_with_offsets(text: str) -> list[SentenceSpan]:
    """将文本切分为带偏移的句子列表。"""

    if not text:
        return []

    segments: list[SentenceSpan] = []
    start = 0
    sent_id = 0
    for index, char in enumerate(text):
        if char not in _SENTENCE_DELIMITERS:
            continue
        end = index + 1
        sentence = _build_sentence_span(text=text, start=start, end=end, sent_id=sent_id)
        if sentence is not None:
            segments.append(sentence)
            sent_id += 1
        start = end

    if start < len(text):
        sentence = _build_sentence_span(text=text, start=start, end=len(text), sent_id=sent_id)
        if sentence is not None:
            segments.append(sentence)

    if segments:
        return segments

    trimmed = text.strip()
    if not trimmed:
        return []
    left = text.find(trimmed)
    return [SentenceSpan(sent_id=0, text=trimmed, start=left, end=left + len(trimmed))]


def locate_sentence_by_offset(
    sentences: list[SentenceSpan],
    start: int | None,
    end: int | None,
) -> int | None:
    """根据 span 偏移定位句子索引。"""

    if not sentences or start is None or end is None:
        return None
    if start < 0 or end < 0:
        return None

    center = start if end <= start else (start + end) // 2
    for index, sentence in enumerate(sentences):
        if sentence.start <= center < sentence.end:
            return index

    if center < sentences[0].start:
        return 0
    if center >= sentences[-1].end:
        return len(sentences) - 1

    for index, sentence in enumerate(sentences):
        if center < sentence.start:
            return max(0, index - 1)
    return len(sentences) - 1


def build_evidence_window(sentences: list[SentenceSpan], index: int | None) -> EvidenceWindow:
    """按前一句 + 当前句 + 后一句构建证据窗口。"""

    if index is None or not sentences or index < 0 or index >= len(sentences):
        return EvidenceWindow(
            prev_sentence="",
            current_sentence="",
            next_sentence="",
            evidence_window="",
            source="missing_sentence",
        )

    prev_sentence = sentences[index - 1].text if index - 1 >= 0 else ""
    current_sentence = sentences[index].text
    next_sentence = sentences[index + 1].text if index + 1 < len(sentences) else ""
    evidence_window = "\n".join(part for part in (prev_sentence, current_sentence, next_sentence) if part)
    return EvidenceWindow(
        prev_sentence=prev_sentence,
        current_sentence=current_sentence,
        next_sentence=next_sentence,
        evidence_window=evidence_window,
        source="sentence_window",
    )


def build_fallback_evidence_window(
    *,
    chunk_text: str,
    span: str,
    span_offset: list[int] | None,
    max_chars: int = 360,
) -> EvidenceWindow:
    """句级定位失败时，从 chunk 截断窗口回退。"""

    if not chunk_text:
        return EvidenceWindow(
            prev_sentence="",
            current_sentence="",
            next_sentence="",
            evidence_window="",
            source="empty_chunk",
        )

    center = len(chunk_text) // 2
    if span_offset is not None:
        center = (span_offset[0] + span_offset[1]) // 2
    elif span:
        index = chunk_text.find(span)
        if index >= 0:
            center = index + len(span) // 2

    half = max(80, max_chars // 2)
    start = max(0, center - half)
    end = min(len(chunk_text), center + half)
    if end - start < max_chars and start > 0:
        start = max(0, end - max_chars)
    excerpt = chunk_text[start:end].strip()

    return EvidenceWindow(
        prev_sentence="",
        current_sentence=excerpt,
        next_sentence="",
        evidence_window=excerpt,
        source="chunk_fallback",
    )


def coerce_span_offset(value: object) -> list[int] | None:
    """将输入解析为 `[start, end]` 偏移。"""

    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        start = int(value[0])
        end = int(value[1])
    except (TypeError, ValueError):
        return None
    if start < 0 or end <= start:
        return None
    return [start, end]


def parse_json_object(response_text: str) -> dict[str, Any]:
    """容错解析模型输出 JSON 对象。"""

    try:
        payload = json.loads(response_text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    matched = _JSON_OBJECT_PATTERN.search(response_text)
    if not matched:
        raise ValueError(f"模型输出不是 JSON 对象: {response_text[:240]}")

    payload = json.loads(matched.group(0))
    if not isinstance(payload, dict):
        raise ValueError(f"模型输出 JSON 不是对象: {payload}")
    return payload


def safe_mean(values: list[int | float]) -> float:
    """安全均值。"""

    if not values:
        return 0.0
    return float(sum(float(value) for value in values)) / float(len(values))


def clamp_text(text: str, max_chars: int) -> str:
    """截断文本，避免 token 失控。"""

    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def build_fp_risk_score(
    *,
    item: dict[str, object],
    evidence_window: str,
) -> tuple[int, list[str]]:
    """按启发式规则计算误报风险分。"""

    score = 0
    flags: list[str] = []

    result = str(item.get("result", "")).strip()
    if result == "不合格":
        score += 2
        flags.append("result_不合格(+2)")

    span = str(item.get("span", "")).strip()
    if not span:
        score += 3
        flags.append("span_为空(+3)")
    span_offset = coerce_span_offset(item.get("span_offset"))
    if span_offset is None:
        score += 2
        flags.append("span_offset_缺失(+2)")

    span_len = len(span)
    if 0 < span_len < 8:
        score += 2
        flags.append("span_过短(<8,+2)")
    elif 8 <= span_len < 15:
        score += 1
        flags.append("span_偏短(8-14,+1)")
    if span and any(phrase in span for phrase in _GENERIC_SPAN_HINTS):
        score += 1
        flags.append("span_泛化短语(+1)")

    rule_hit = str(item.get("rule_hit", "")).strip()
    if not rule_hit:
        score += 3
        flags.append("rule_hit_为空(+3)")

    rule_hit_id = str(item.get("rule_hit_id", "")).strip().upper()
    if rule_hit_id == "UNKNOWN":
        score += 2
        flags.append("rule_hit_id_UNKNOWN(+2)")

    if rule_hit and (len(rule_hit) < 10 or any(flag in rule_hit for flag in _VAGUE_RULE_HIT_HINTS)):
        score += 1
        flags.append("rule_hit_模糊(+1)")

    suggest = str(item.get("suggest", "")).strip()
    if suggest and any(flag in suggest for flag in _WEAK_SUGGEST_HINTS):
        score += 1
        flags.append("suggest_弱措辞(+1)")

    risk_type = str(item.get("risk_type", "")).strip()
    risk_keywords = extract_risk_keywords(risk_type)
    if evidence_window.strip():
        if risk_keywords and not any(keyword in evidence_window for keyword in risk_keywords):
            missing_score = 2 if len(evidence_window) < 48 else 1
            score += missing_score
            flags.append(f"evidence_缺少风险关键词(+{missing_score})")
    else:
        score += 2
        flags.append("evidence_window_为空(+2)")

    if risk_keywords and rule_hit and not any(keyword in rule_hit for keyword in risk_keywords):
        score += 2
        flags.append("rule_hit_与risk_type疑似不一致(+2)")

    return score, flags


def extract_risk_keywords(risk_type: str) -> list[str]:
    """从 risk_type 提取轻量关键词。"""

    text = risk_type.strip()
    if not text:
        return []

    text = (
        text.replace("审查", " ")
        .replace("条款", " ")
        .replace("内容", " ")
        .replace("事项", " ")
        .replace("风险", " ")
    )
    parts = [part.strip() for part in re.split(r"[、，,；;|/\s]+", text) if part.strip()]
    keywords = [part for part in parts if len(part) >= 2]
    if not keywords:
        compact = "".join(text.split())
        if len(compact) >= 2:
            keywords.append(compact)

    if not keywords and len(risk_type) >= 2:
        keywords.append(risk_type[: min(4, len(risk_type))])

    deduped: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        if keyword not in seen:
            deduped.append(keyword)
            seen.add(keyword)
    return deduped


def normalize_result(value: str) -> str:
    """统一结果标签。"""

    text = value.strip()
    return text if text in {"合格", "不合格", "待复核"} else text


def _build_sentence_span(text: str, start: int, end: int, sent_id: int) -> SentenceSpan | None:
    left = start
    right = end
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if right <= left:
        return None

    sentence_text = text[left:right]
    return SentenceSpan(sent_id=sent_id, text=sentence_text, start=left, end=right)
