"""reflection 阶段提示词构建。"""

from __future__ import annotations

import json

from .reflection_models import Stage1Candidate, Stage2GroupTask


ITEM_REFLECTION_SYSTEM_PROMPT = (
    "你是合同审查反思助手（item-level）。"
    "你的任务不是重新审查合同，也不是新增风险项。"
    "你只做后验校正：判断当前结论是否被局部证据充分支持。"
    "若证据不足，优先给出保守修正（downgrade/reject），严格输出 JSON。"
)

GROUP_REFLECTION_SYSTEM_PROMPT = (
    "你是合同审查反思助手（group-level）。"
    "你的任务不是重做整组审查，只判断组内是否存在不合理不一致。"
    "若需要调整，只对最可疑的少量 item 给出保守修正，严格输出 JSON。"
)


def build_item_reflection_messages(
    candidate: Stage1Candidate,
    *,
    include_chunk_excerpt: bool,
) -> list[dict[str, str]]:
    """构建 Stage-1 item-level reflection prompt。"""

    schema_hint = {
        "risk_id": candidate.risk_id,
        "action": "keep | downgrade | reject | revise",
        "evidence_sufficient": "yes | partial | no",
        "revised_result": "不合格 | 待复核 | 合格",
        "revised_span": "字符串，可为空",
        "revised_suggest": "字符串，可为空",
        "reason": "一句话说明"
    }

    user_payload = {
        "risk_id": candidate.risk_id,
        "risk_type": candidate.risk_type,
        "initial_result": candidate.initial_result,
        "rule_hit_id": candidate.rule_hit_id,
        "rule_hit": candidate.rule_hit,
        "span": candidate.span,
        "suggest": candidate.suggest,
        "evidence_window": candidate.evidence_window,
        "evidence_window_source": candidate.evidence_window_source,
        "fp_risk_score": candidate.fp_risk_score,
        "fp_risk_flags": candidate.fp_risk_flags,
    }
    if include_chunk_excerpt and candidate.chunk_excerpt:
        user_payload["chunk_excerpt"] = candidate.chunk_excerpt

    user_prompt = (
        "请对以下 review_item 进行后验反思校正。\n"
        "规则：\n"
        "1) 不允许新增 item；\n"
        "2) 不允许把“合格”升级为“不合格”；\n"
        "3) 证据不足时优先 downgrade 为“待复核”；\n"
        "4) revise 只能修正 span/suggest 或做保守结果调整；\n"
        "5) 只输出 JSON，不要输出解释文本。\n\n"
        "输入：\n"
        f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n\n"
        "输出 schema 示例：\n"
        f"{json.dumps(schema_hint, ensure_ascii=False, indent=2)}"
    )

    return [
        {"role": "system", "content": ITEM_REFLECTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_group_reflection_messages(task: Stage2GroupTask) -> list[dict[str, str]]:
    """构建 Stage-2 group-level reflection prompt。"""

    items = [
        {
            "risk_id": item.risk_id,
            "chunk_id": item.chunk_id,
            "result": item.result,
            "rule_hit_id": item.rule_hit_id,
            "span": item.span,
            "evidence_window": item.evidence_window,
            "suggest": item.suggest,
            "fp_risk_score": item.fp_risk_score,
        }
        for item in task.selected_items
    ]

    schema_hint = {
        "group_key": task.group_key,
        "consistency_status": "consistent | partially_inconsistent | inconsistent",
        "group_reason": "一句话说明",
        "adjustments": [
            {
                "risk_id": "风险ID",
                "action": "keep | downgrade | reject",
                "target_result": "不合格 | 待复核 | 合格",
                "reason": "一句话说明"
            }
        ]
    }

    user_payload = {
        "doc_id": task.doc_id,
        "risk_type": task.risk_type,
        "group_key": task.group_key,
        "group_item_total": task.item_total,
        "selected_items": items,
    }

    user_prompt = (
        "请对以下同合同同风险类型分组做一致性反思。\n"
        "规则：\n"
        "1) 你不是重新生成整组结果；\n"
        "2) 只判断是否存在不合理不一致；\n"
        "3) 若需调整，仅调整最可疑项，动作保持保守；\n"
        "4) 不允许新增 item；\n"
        "5) 不允许把组内大量条目升级为“不合格”；\n"
        "6) 只输出 JSON。\n\n"
        "输入：\n"
        f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n\n"
        "输出 schema 示例：\n"
        f"{json.dumps(schema_hint, ensure_ascii=False, indent=2)}"
    )

    return [
        {"role": "system", "content": GROUP_REFLECTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
