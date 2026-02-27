"""关键词匹配器。"""

from __future__ import annotations

from .risk_catalog import RiskCatalog


class KeywordMatcher:
    """按风险目录中的关键字匹配 chunk。"""

    def __init__(self, catalog: RiskCatalog) -> None:
        self._catalog = catalog

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

    def match(self, text: str) -> list[str]:
        """返回关键词命中的风险类型列表。"""

        normalized = self._normalize_text(text)
        matched: list[str] = []

        for item in self._catalog.definitions:
            if not item.keywords:
                continue
            if any(keyword.lower() in normalized for keyword in item.keywords):
                matched.append(item.risk_type)

        return self._catalog.normalize_risks(matched)
