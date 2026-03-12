"""reflection 模块数据结构定义。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    """句子文本及其在 chunk 中的偏移范围。"""

    sent_id: int
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class EvidenceWindow:
    """item 的局部证据窗口。"""

    prev_sentence: str
    current_sentence: str
    next_sentence: str
    evidence_window: str
    source: str


@dataclass(frozen=True, slots=True)
class Stage1Candidate:
    """Stage-1 反思候选项。"""

    source_file: str
    doc_id: str
    risk_id: str
    item_index: int
    risk_type: str
    initial_result: str
    rule_hit_id: str
    rule_hit: str
    span: str
    suggest: str
    chunk_id: int | str | None
    evidence_window: str
    evidence_window_source: str
    chunk_excerpt: str
    fp_risk_score: int
    fp_risk_flags: list[str]


@dataclass(frozen=True, slots=True)
class Stage1Decision:
    """Stage-1 模型输出。"""

    risk_id: str
    action: str
    evidence_sufficient: str
    revised_result: str
    revised_span: str
    revised_suggest: str
    reason: str


@dataclass(frozen=True, slots=True)
class Stage2ItemSnapshot:
    """Stage-2 输入 item 摘要。"""

    risk_id: str
    chunk_id: int | str | None
    result: str
    rule_hit_id: str
    span: str
    suggest: str
    evidence_window: str
    fp_risk_score: int


@dataclass(frozen=True, slots=True)
class Stage2GroupTask:
    """Stage-2 分组反思任务。"""

    source_file: str
    doc_id: str
    risk_type: str
    group_key: str
    item_total: int
    selected_items: list[Stage2ItemSnapshot]


@dataclass(frozen=True, slots=True)
class Stage2Adjustment:
    """Stage-2 单项调整建议。"""

    risk_id: str
    action: str
    target_result: str
    reason: str


@dataclass(frozen=True, slots=True)
class Stage2Decision:
    """Stage-2 模型输出。"""

    group_key: str
    consistency_status: str
    group_reason: str
    adjustments: list[Stage2Adjustment]


@dataclass(frozen=True, slots=True)
class ReflectionCallDiagnostic:
    """单次 reflection LLM 调用诊断信息。"""

    stage: str
    source_file: str
    doc_id: str
    risk_type: str
    scope_key: str
    input_item_count: int
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

    def to_dict(self) -> dict[str, object]:
        """导出可序列化字典。"""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReflectionFileMetrics:
    """单文件 reflection 聚合指标。"""

    generated_at: str
    reflection_version: str
    rule_version: str
    stage1_candidate_count: int
    stage1_called_count: int
    stage2_group_count: int
    stage2_candidate_group_count: int
    stage2_called_group_count: int
    stage1_adjusted_count: int
    stage2_adjusted_count: int
    final_item_count: int
    avg_token_in: float
    avg_token_out: float
    avg_total_token: float

    def to_dict(self) -> dict[str, object]:
        """导出可序列化字典。"""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReflectionRunMetrics:
    """reflection 运行级统计。"""

    file_count: int
    stage1_candidate_count: int
    stage1_called_count: int
    stage2_group_count: int
    stage2_candidate_group_count: int
    stage2_called_group_count: int
    stage1_adjusted_count: int
    stage2_adjusted_count: int
    final_item_count: int
    llm_called_count: int
    avg_token_in: float
    avg_token_out: float
    avg_total_token: float
    reflection_version: str

    def to_dict(self) -> dict[str, object]:
        """导出可序列化字典。"""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReflectionRunResult:
    """reflection 执行输出。"""

    outputs: list[Path]
    diagnostics: list[ReflectionCallDiagnostic]
    metrics: ReflectionRunMetrics
    trace_path: Path
    metrics_path: Path
