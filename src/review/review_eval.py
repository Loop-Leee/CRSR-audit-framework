"""Review 输出一致性评估脚本。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


VALID_LABELS = {"合格", "不合格"}
POSITIVE_LABEL = "不合格"


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """一致性评估混淆矩阵统计。"""

    tp: int
    fp: int
    tn: int
    fn: int
    evaluated_item_count: int
    unlabeled_item_count: int
    invalid_pred_item_count: int
    total_item_count: int
    file_count: int
    input_path: str

    def to_dict(self) -> dict[str, int | float | str]:
        """导出可序列化结果。"""

        payload = asdict(self)
        evaluated = self.evaluated_item_count
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (self.tp + self.tn) / evaluated if evaluated > 0 else 0.0
        payload.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "positive_label": POSITIVE_LABEL,
            }
        )
        return payload


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="评估 review 结果与 ground_truth 一致性")
    parser.add_argument("--input", type=Path, default=Path("data/4-review"), help="review 文件或目录")
    parser.add_argument("--output", type=Path, default=None, help="可选输出 JSON 文件路径")
    return parser.parse_args()


def discover_review_files(input_path: Path) -> list[Path]:
    """发现待评估 review 文件。"""

    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    files = sorted(input_path.glob("*.review.json"))
    if files:
        return files
    return sorted(input_path.glob("*.json"))


def evaluate_consistency(input_path: Path) -> ConfusionCounts:
    """计算 result 与 ground_truth 的一致性混淆矩阵。"""

    files = discover_review_files(input_path)
    if not files:
        raise FileNotFoundError(f"未找到 review 文件: {input_path}")

    tp = fp = tn = fn = 0
    evaluated_item_count = 0
    unlabeled_item_count = 0
    invalid_pred_item_count = 0
    total_item_count = 0

    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        review_items = payload.get("review_items", [])
        if not isinstance(review_items, list):
            continue

        for item in review_items:
            if not isinstance(item, dict):
                continue

            total_item_count += 1
            pred = str(item.get("result", "")).strip()
            gt = str(item.get("ground_truth", "")).strip()

            if pred not in VALID_LABELS:
                invalid_pred_item_count += 1
                continue
            if gt not in VALID_LABELS:
                unlabeled_item_count += 1
                continue

            evaluated_item_count += 1
            pred_positive = pred == POSITIVE_LABEL
            gt_positive = gt == POSITIVE_LABEL

            if pred_positive and gt_positive:
                tp += 1
            elif pred_positive and not gt_positive:
                fp += 1
            elif (not pred_positive) and gt_positive:
                fn += 1
            else:
                tn += 1

    return ConfusionCounts(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        evaluated_item_count=evaluated_item_count,
        unlabeled_item_count=unlabeled_item_count,
        invalid_pred_item_count=invalid_pred_item_count,
        total_item_count=total_item_count,
        file_count=len(files),
        input_path=str(input_path),
    )


def main() -> None:
    """执行一致性评估并输出结果。"""

    args = parse_args()
    input_path = args.input if args.input.is_absolute() else Path.cwd() / args.input

    counts = evaluate_consistency(input_path)
    payload = counts.to_dict()

    if args.output is not None:
        output_path = args.output if args.output.is_absolute() else Path.cwd() / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] evaluation -> {output_path}")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
