"""风险类型目录加载。"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


_KEYWORD_SPLIT_PATTERN = re.compile(r"[，,、；;|/]")


@dataclass(frozen=True, slots=True)
class RiskTypeDefinition:
    """单个风险类型定义。"""

    risk_type: str
    keywords: tuple[str, ...]
    semantic_hint: str


@dataclass(slots=True)
class RiskCatalog:
    """风险类型目录。"""

    definitions: list[RiskTypeDefinition]
    by_name: dict[str, RiskTypeDefinition] = field(init=False)
    order: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.by_name = {item.risk_type: item for item in self.definitions}
        self.order = [item.risk_type for item in self.definitions]

    def normalize_risks(self, risks: Iterable[str]) -> list[str]:
        """过滤非法值并按目录顺序去重。"""

        risk_set = {item for item in risks if item in self.by_name}
        return [name for name in self.order if name in risk_set]

    def prompt_summary(self) -> str:
        """生成语义匹配的风险类型摘要。"""

        lines = []
        for item in self.definitions:
            hint = item.semantic_hint or "无补充说明"
            lines.append(f"- {item.risk_type}: {hint}")
        return "\n".join(lines)


def _parse_keywords(raw: str) -> tuple[str, ...]:
    """解析关键字字符串。"""

    values = []
    for part in _KEYWORD_SPLIT_PATTERN.split(raw.strip()):
        keyword = part.strip()
        if keyword and keyword not in values:
            values.append(keyword)
    return tuple(values)


def _compact_text(text: str, max_len: int = 200) -> str:
    """压缩提示词长度，避免语义匹配上下文膨胀。"""

    compact = " ".join(text.split())
    return compact[:max_len] + "..." if len(compact) > max_len else compact


def load_risk_catalog(csv_path: Path) -> RiskCatalog:
    """从 risk_info.csv 加载风险类型目录。"""

    if not csv_path.exists():
        raise FileNotFoundError(f"未找到风险类型文件: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"风险类型文件为空: {csv_path}")

    seen: dict[str, RiskTypeDefinition] = {}
    for row in rows:
        name = row.get("审查类型", "").strip() or row.get("审查要点", "").strip()
        if not name:
            continue

        keywords = _parse_keywords(row.get("关键字", ""))
        semantic_parts = [row.get("提取步骤", "").strip(), row.get("审查规则", "").strip()]
        semantic_hint = _compact_text("；".join(part for part in semantic_parts if part))

        existing = seen.get(name)
        if existing is None:
            seen[name] = RiskTypeDefinition(name, keywords, semantic_hint)
            continue

        merged_keywords = tuple(dict.fromkeys(existing.keywords + keywords))
        merged_hint = existing.semantic_hint if existing.semantic_hint else semantic_hint
        seen[name] = RiskTypeDefinition(name, merged_keywords, merged_hint)

    if not seen:
        raise ValueError(f"风险类型文件未解析出有效审查类型: {csv_path}")

    return RiskCatalog(list(seen.values()))
