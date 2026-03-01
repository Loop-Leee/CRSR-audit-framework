# src/exp_cuad/prompts.py
from __future__ import annotations

BASELINE_SYSTEM = """You are a careful legal contract analyst.
Your task: determine whether the given contract text contains the specified clause type.
If present, extract the most relevant evidence sentences verbatim from the contract.
If not present, output NO_CLAUSE.
Return JSON only, no extra text.
"""

def make_baseline_user_prompt(label: str, label_desc: str | None, chunk_text: str) -> str:
    """构造 baseline 模式的用户提示词。"""
    desc_part = f"\nClause description: {label_desc}\n" if label_desc else "\n"
    return f"""Clause type: {label}{desc_part}
Contract text (chunk):
{chunk_text}

Output JSON schema:
{{
  "present": true/false,
  "evidence": ["sentence1", "sentence2"]
}}

Rules:
- evidence MUST be exact sentences copied from the chunk
- if not present, set present=false and evidence=[]
- if you are unsure, choose present=false
"""

CRSR_LITE_SYSTEM = """You are a careful legal contract analyst.
You must follow a structured audit procedure:
1) Identify whether the clause type is present in this chunk.
2) If present, copy exact evidence sentences verbatim.
3) Keep output strictly in JSON with fixed keys.
Return JSON only.
"""

def make_crsr_lite_user_prompt(label: str, label_desc: str | None, chunk_text: str) -> str:
    """构造 CRSR-lite 模式的用户提示词。"""
    # CRSR-lite 在 CUAD 上不做企业规则判断，只做结构化控制
    desc_part = f"\nClause definition / rule reference:\n{label_desc}\n" if label_desc else "\n"
    return f"""[Clause Type]
{label}{desc_part}

[Chunk]
{chunk_text}

[Output JSON schema]
{{
  "present": true/false,
  "evidence": ["..."]
}}

[Constraints]
- evidence must be verbatim sentences from the chunk
- do not paraphrase
- if absent: present=false, evidence=[]
"""
