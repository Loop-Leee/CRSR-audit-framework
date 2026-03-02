"""规则库加载与版本管理。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RuleHit:
    """单条规则命中定义。

    Attributes:
        risk_type: 审查类型。
        rule_hit_id: 规则命中唯一标识。
        rule_hit_text: 规则原文。
        rule_version: 规则版本号。
    """

    risk_type: str
    rule_hit_id: str
    rule_hit_text: str
    rule_version: str = "v1"

    def to_dict(self) -> dict[str, str]:
        """导出为字典结构。"""

        return {
            "risk_type": self.risk_type,
            "rule_hit_id": self.rule_hit_id,
            "rule_hit_text": self.rule_hit_text,
            "rule_version": self.rule_version,
        }


class RuleStore:
    """全局规则库管理器。

    说明：
    - 支持从 expanded CSV/JSON 加载规则；
    - 支持多规则版本并按版本查询；
    - 支持生成粗粒度规则版本（每个 risk_type 合并为一条规则）。
    """

    def __init__(self) -> None:
        self._rules_by_version: dict[str, dict[str, list[RuleHit]]] = {}
        self._active_version: str = "v1"

    @property
    def active_version(self) -> str:
        """当前激活规则版本。"""

        return self._active_version

    def set_active_version(self, rule_version: str) -> None:
        """切换激活规则版本。

        Args:
            rule_version: 目标规则版本。

        Returns:
            None

        Raises:
            ValueError: 版本不存在时抛出。
        """

        version = rule_version.strip()
        if version not in self._rules_by_version:
            raise ValueError(f"规则版本不存在: {rule_version}")
        self._active_version = version

    def load_rules(self, path: Path, rule_version: str = "v1") -> None:
        """加载规则并注册为指定版本。

        Args:
            path: `rule_hits_expanded.csv/json` 文件路径。
            rule_version: 规则版本号，默认 `v1`。

        Returns:
            None

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 文件内容为空或字段不完整。
        """

        if not path.exists():
            raise FileNotFoundError(f"规则文件不存在: {path}")

        version = rule_version.strip() or "v1"
        rows = _read_rows(path)
        if not rows:
            raise ValueError(f"规则文件为空: {path}")

        grouped: dict[str, list[RuleHit]] = {}
        for row in rows:
            risk_type = str(row.get("risk_type", "")).strip()
            rule_hit_id = str(row.get("rule_hit_id", "")).strip()
            rule_hit_text = _compact_text(str(row.get("rule_hit_text", "")))
            if not risk_type or not rule_hit_id or not rule_hit_text:
                continue

            grouped.setdefault(risk_type, []).append(
                RuleHit(
                    risk_type=risk_type,
                    rule_hit_id=rule_hit_id,
                    rule_hit_text=rule_hit_text,
                    rule_version=version,
                )
            )

        if not grouped:
            raise ValueError(f"规则文件未解析出有效记录: {path}")

        self._rules_by_version[version] = grouped
        self._active_version = version

    def build_coarse_version(
        self,
        source_version: str = "v1",
        target_version: str = "v1_coarse",
    ) -> str:
        """基于 source_version 生成粗粒度规则版本。

        粗粒度规则定义：每个 `risk_type` 仅保留一条合并规则。

        Args:
            source_version: 源版本。
            target_version: 目标版本。

        Returns:
            str: 实际生成的目标版本名。

        Raises:
            ValueError: 源版本不存在。
        """

        source = self._rules_by_version.get(source_version)
        if source is None:
            raise ValueError(f"源规则版本不存在: {source_version}")

        target = target_version.strip() or "v1_coarse"
        coarse_grouped: dict[str, list[RuleHit]] = {}
        for risk_type, hits in source.items():
            merged_text = "；".join(hit.rule_hit_text for hit in hits)
            coarse_grouped[risk_type] = [
                RuleHit(
                    risk_type=risk_type,
                    rule_hit_id=f"{risk_type}:ALL",
                    rule_hit_text=merged_text,
                    rule_version=target,
                )
            ]

        self._rules_by_version[target] = coarse_grouped
        return target

    def get_rules(self, risk_type: str, rule_version: str | None = None) -> list[dict[str, str]]:
        """按审查类型获取规则列表。

        Args:
            risk_type: 审查类型。
            rule_version: 可选版本号；为空时使用激活版本。

        Returns:
            list[dict[str, str]]: 每项包含 `rule_hit_id`、`rule_hit_text`、`risk_type`、`rule_version`。
        """

        hits = self.get_rule_hits(risk_type=risk_type, rule_version=rule_version)
        return [hit.to_dict() for hit in hits]

    def get_rule_hits(self, risk_type: str, rule_version: str | None = None) -> list[RuleHit]:
        """按审查类型获取规则对象列表。

        Args:
            risk_type: 审查类型。
            rule_version: 可选版本号；为空时使用激活版本。

        Returns:
            list[RuleHit]: 规则对象列表。
        """

        version = (rule_version or self._active_version).strip()
        by_type = self._rules_by_version.get(version, {})
        return list(by_type.get(risk_type, []))

    def list_risk_types(self, rule_version: str | None = None) -> list[str]:
        """列出指定版本包含的审查类型。

        Args:
            rule_version: 可选版本号；为空时使用激活版本。

        Returns:
            list[str]: 审查类型名称列表。
        """

        version = (rule_version or self._active_version).strip()
        by_type = self._rules_by_version.get(version, {})
        return sorted(by_type.keys())

    def available_versions(self) -> list[str]:
        """列出当前已加载规则版本。

        Returns:
            list[str]: 规则版本列表。
        """

        return sorted(self._rules_by_version.keys())


def _read_rows(path: Path) -> list[dict[str, object]]:
    """读取 expanded CSV/JSON 为统一行结构。"""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        raise ValueError(f"规则 JSON 必须是数组: {path}")
    raise ValueError(f"不支持的规则文件格式: {path}")


def _compact_text(raw: str) -> str:
    """清理文本中的多余空白。"""

    return " ".join(raw.split())


_GLOBAL_RULE_STORE = RuleStore()


def load_rules(path: Path, rule_version: str = "v1") -> RuleStore:
    """加载规则到全局 RuleStore 并返回实例。

    Args:
        path: `rule_hits_expanded.csv/json` 文件路径。
        rule_version: 规则版本号。

    Returns:
        RuleStore: 全局规则库实例。
    """

    _GLOBAL_RULE_STORE.load_rules(path=path, rule_version=rule_version)
    return _GLOBAL_RULE_STORE


def get_rules(risk_type: str, rule_version: str | None = None) -> list[dict[str, str]]:
    """从全局 RuleStore 按 risk_type 获取规则。

    Args:
        risk_type: 审查类型。
        rule_version: 可选规则版本；为空时使用激活版本。

    Returns:
        list[dict[str, str]]: 规则命中定义列表。
    """

    return _GLOBAL_RULE_STORE.get_rules(risk_type=risk_type, rule_version=rule_version)
