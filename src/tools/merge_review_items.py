"""merge_review_items.py 使用说明

功能:
1. `merge`: 将 `data/4-review` 下同名 `.review.json` 中的 `review_items`
   按 `risk_id` 去重后合并到 `dataset/standard-review` 对应文件。
   - 若 `risk_id` 已存在于目标文件: 丢弃源 item。
   - 若 `risk_id` 不存在于目标文件: 追加到目标 `review_items`。
2. `sort`: 对目标目录中的 `review_items` 排序。
   排序优先级:
   - 第一键: `chunk_id` 升序
   - 第二键: `rule_hit_id` 升序（字典序）
   - 第三键: `risk_id` 最后一个 `#` 后的序号升序
3. `merge-sort`: 先执行 merge，再执行 sort。
4. `remove-pass`: 同时删除 `data/4-review` 与 `dataset/standard-review`
   中 `result == "合格"` 的 `review_items`。

任务选择:
- `--task merge`      仅合并（默认）
- `--task sort`       仅排序
- `--task merge-sort` 先合并再排序
- `--task remove-pass` 同时删除源/目标目录中 result 为“合格”的条目

常用示例:
- 仅查看合并结果（不写文件）:
  `python -m src.tools.merge_review_items --task merge --dry-run`
- 执行合并并写回:
  `python -m src.tools.merge_review_items --task merge`
- 仅排序目标目录:
  `python -m src.tools.merge_review_items --task sort`
- 合并后再排序:
  `python -m src.tools.merge_review_items --task merge-sort`
- 删除“合格”条目（可先 dry-run）:
  `python -m src.tools.merge_review_items --task remove-pass --dry-run`
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "4-review"
DEFAULT_TARGET_DIR = PROJECT_ROOT / "dataset" / "standard-review"
DEFAULT_PATTERN = "*.review.json"
TASK_MERGE = "merge"
TASK_SORT = "sort"
TASK_MERGE_SORT = "merge-sort"
TASK_REMOVE_PASS = "remove-pass"
_MAX_INT = 2**31 - 1


@dataclass(frozen=True, slots=True)
class FileMergeStat:
    filename: str
    source_count: int
    target_before_count: int
    target_after_count: int
    added_count: int
    duplicate_count: int
    missing_risk_id_count: int


@dataclass(frozen=True, slots=True)
class MergeSummary:
    scanned_files: int
    merged_files: int
    missing_target_files: int
    total_added: int
    total_duplicates: int
    total_missing_risk_id: int
    stats: list[FileMergeStat]


@dataclass(frozen=True, slots=True)
class FileSortStat:
    filename: str
    item_count: int
    changed: bool
    meta_mismatch: bool


@dataclass(frozen=True, slots=True)
class SortSummary:
    scanned_files: int
    sorted_files: int
    unchanged_files: int
    meta_mismatch_files: int
    total_items: int
    stats: list[FileSortStat]


@dataclass(frozen=True, slots=True)
class FileRemovePassStat:
    directory: str
    filename: str
    before_count: int
    after_count: int
    removed_count: int


@dataclass(frozen=True, slots=True)
class RemovePassSummary:
    scanned_dirs: int
    scanned_files: int
    touched_files: int
    untouched_files: int
    total_removed: int
    total_before: int
    total_after: int
    stats: list[FileRemovePassStat]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="合并/排序 review_items（按 risk_id 去重 + 多键排序）")
    parser.add_argument(
        "--task",
        type=str,
        default=TASK_MERGE,
        choices=(TASK_MERGE, TASK_SORT, TASK_MERGE_SORT, TASK_REMOVE_PASS),
        help="执行任务类型：merge / sort / merge-sort / remove-pass（默认 merge）",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="源目录（默认 data/4-review）")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR, help="目标目录（默认 dataset/standard-review）")
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN, help="匹配文件名模式（默认 *.review.json）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印合并结果，不写回文件")
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _unique_directories(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return result


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return data


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _get_review_items(payload: dict[str, Any], *, path: Path) -> tuple[str, list[dict[str, Any]]]:
    for key in ("review_items", "review-items"):
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            raise ValueError(f"{path} 的 {key} 不是数组。")
        items: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"{path} 的 {key}[{index}] 不是对象。")
            items.append(item)
        return key, items
    raise KeyError(f"{path} 未找到 review_items/review-items 字段。")


def _existing_risk_ids(items: list[dict[str, Any]]) -> set[str]:
    risk_ids: set[str] = set()
    for item in items:
        risk_id = item.get("risk_id")
        if isinstance(risk_id, str) and risk_id.strip():
            risk_ids.add(risk_id)
    return risk_ids


def _update_review_meta_count(payload: dict[str, Any], item_count: int) -> None:
    review_meta = payload.get("review_meta")
    if isinstance(review_meta, dict):
        review_meta["review_item_count"] = item_count


def _has_review_meta_count_mismatch(payload: dict[str, Any], item_count: int) -> bool:
    review_meta = payload.get("review_meta")
    if not isinstance(review_meta, dict):
        return False
    return review_meta.get("review_item_count") != item_count


def _to_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return _MAX_INT


def _risk_id_tail_index(risk_id: str) -> int:
    tail = risk_id.rsplit("#", 1)[-1].strip()
    if tail.isdigit():
        return int(tail)
    match = re.search(r"(\d+)$", tail)
    if match:
        return int(match.group(1))
    return _MAX_INT


def _review_item_sort_key(item: dict[str, Any]) -> tuple[int, str, int, str]:
    chunk_id = _to_int(item.get("chunk_id"))
    rule_hit_id = str(item.get("rule_hit_id") or "")
    risk_id = str(item.get("risk_id") or "")
    tail_index = _risk_id_tail_index(risk_id) if risk_id else _MAX_INT
    return (chunk_id, rule_hit_id, tail_index, risk_id)


def merge_review_items(*, source_dir: Path, target_dir: Path, pattern: str, dry_run: bool) -> MergeSummary:
    """合并同名 JSON 文件中的 review_items。"""

    if not source_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"目标目录不存在: {target_dir}")

    source_files = sorted(path for path in source_dir.glob(pattern) if path.is_file())

    merged_files = 0
    missing_target_files = 0
    total_added = 0
    total_duplicates = 0
    total_missing_risk_id = 0
    stats: list[FileMergeStat] = []

    for source_file in source_files:
        target_file = target_dir / source_file.name
        if not target_file.exists():
            missing_target_files += 1
            print(f"[SKIP] 缺少同名目标文件: {target_file}")
            continue

        source_payload = _load_json(source_file)
        target_payload = _load_json(target_file)

        _, source_items = _get_review_items(source_payload, path=source_file)
        target_key, target_items = _get_review_items(target_payload, path=target_file)

        existing_risk_ids = _existing_risk_ids(target_items)
        target_before_count = len(target_items)
        added_count = 0
        duplicate_count = 0
        missing_risk_id_count = 0

        for item in source_items:
            risk_id = item.get("risk_id")
            if not isinstance(risk_id, str) or not risk_id.strip():
                missing_risk_id_count += 1
                continue
            if risk_id in existing_risk_ids:
                duplicate_count += 1
                continue
            target_items.append(copy.deepcopy(item))
            existing_risk_ids.add(risk_id)
            added_count += 1

        if added_count > 0:
            target_payload[target_key] = target_items
            _update_review_meta_count(target_payload, len(target_items))
            if not dry_run:
                _dump_json(target_file, target_payload)

        target_after_count = len(target_items)
        merged_files += 1
        total_added += added_count
        total_duplicates += duplicate_count
        total_missing_risk_id += missing_risk_id_count

        stats.append(
            FileMergeStat(
                filename=source_file.name,
                source_count=len(source_items),
                target_before_count=target_before_count,
                target_after_count=target_after_count,
                added_count=added_count,
                duplicate_count=duplicate_count,
                missing_risk_id_count=missing_risk_id_count,
            )
        )

        action = "WOULD UPDATE" if dry_run else "UPDATED"
        print(
            "[%s] %s: +%d, duplicate=%d, missing_risk_id=%d, target=%d->%d"
            % (
                action,
                source_file.name,
                added_count,
                duplicate_count,
                missing_risk_id_count,
                target_before_count,
                target_after_count,
            )
        )

    return MergeSummary(
        scanned_files=len(source_files),
        merged_files=merged_files,
        missing_target_files=missing_target_files,
        total_added=total_added,
        total_duplicates=total_duplicates,
        total_missing_risk_id=total_missing_risk_id,
        stats=stats,
    )


def sort_review_items(*, target_dir: Path, pattern: str, dry_run: bool) -> SortSummary:
    """排序目标目录中每个 review JSON 的 review_items。"""

    if not target_dir.exists():
        raise FileNotFoundError(f"目标目录不存在: {target_dir}")

    target_files = sorted(path for path in target_dir.glob(pattern) if path.is_file())

    sorted_files = 0
    unchanged_files = 0
    meta_mismatch_files = 0
    total_items = 0
    stats: list[FileSortStat] = []

    for target_file in target_files:
        payload = _load_json(target_file)
        review_key, items = _get_review_items(payload, path=target_file)

        item_count = len(items)
        total_items += item_count

        sorted_items = sorted(items, key=_review_item_sort_key)
        changed = sorted_items != items
        meta_mismatch = _has_review_meta_count_mismatch(payload, item_count)

        if changed:
            sorted_files += 1
            action = "WOULD SORT" if dry_run else "SORTED"
            print(f"[{action}] {target_file.name}: item_count={item_count}")
        else:
            unchanged_files += 1
            if meta_mismatch:
                action = "WOULD SYNC META" if dry_run else "SYNCED META"
                print(f"[{action}] {target_file.name}: item_count={item_count}")
            else:
                print(f"[NO CHANGE] {target_file.name}: item_count={item_count}")

        if meta_mismatch:
            meta_mismatch_files += 1

        if changed or meta_mismatch:
            payload[review_key] = sorted_items
            _update_review_meta_count(payload, item_count)
            if not dry_run:
                _dump_json(target_file, payload)

        stats.append(
            FileSortStat(
                filename=target_file.name,
                item_count=item_count,
                changed=changed,
                meta_mismatch=meta_mismatch,
            )
        )

    return SortSummary(
        scanned_files=len(target_files),
        sorted_files=sorted_files,
        unchanged_files=unchanged_files,
        meta_mismatch_files=meta_mismatch_files,
        total_items=total_items,
        stats=stats,
    )


def remove_pass_review_items(*, directories: list[Path], pattern: str, dry_run: bool) -> RemovePassSummary:
    """删除多个目录中 result 为“合格”的 review_items。"""

    target_dirs = _unique_directories(directories)
    for directory in target_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")

    touched_files = 0
    untouched_files = 0
    scanned_files = 0
    total_removed = 0
    total_before = 0
    total_after = 0
    stats: list[FileRemovePassStat] = []

    for target_dir in target_dirs:
        target_files = sorted(path for path in target_dir.glob(pattern) if path.is_file())
        scanned_files += len(target_files)
        display_dir = _display_path(target_dir)

        for target_file in target_files:
            payload = _load_json(target_file)
            review_key, items = _get_review_items(payload, path=target_file)

            before_count = len(items)
            filtered_items = [item for item in items if str(item.get("result") or "") != "合格"]
            after_count = len(filtered_items)
            removed_count = before_count - after_count

            total_before += before_count
            total_after += after_count
            total_removed += removed_count

            if removed_count > 0:
                touched_files += 1
                action = "WOULD CLEAN" if dry_run else "CLEANED"
                print(
                    f"[{action}] {display_dir}/{target_file.name}: "
                    f"removed={removed_count}, review_items={before_count}->{after_count}"
                )
                payload[review_key] = filtered_items
                _update_review_meta_count(payload, after_count)
                if not dry_run:
                    _dump_json(target_file, payload)
            else:
                untouched_files += 1
                print(f"[NO CHANGE] {display_dir}/{target_file.name}: removed=0, review_items={before_count}")

            stats.append(
                FileRemovePassStat(
                    directory=display_dir,
                    filename=target_file.name,
                    before_count=before_count,
                    after_count=after_count,
                    removed_count=removed_count,
                )
            )

    return RemovePassSummary(
        scanned_dirs=len(target_dirs),
        scanned_files=scanned_files,
        touched_files=touched_files,
        untouched_files=untouched_files,
        total_removed=total_removed,
        total_before=total_before,
        total_after=total_after,
        stats=stats,
    )


def main() -> None:
    """CLI 入口。"""

    args = parse_args()
    source_dir = _resolve_path(args.source_dir)
    target_dir = _resolve_path(args.target_dir)

    if args.task in (TASK_MERGE, TASK_MERGE_SORT):
        merge_summary = merge_review_items(
            source_dir=source_dir,
            target_dir=target_dir,
            pattern=args.pattern,
            dry_run=args.dry_run,
        )
        print(
            "[MERGE SUMMARY] scanned=%d, merged=%d, missing_target=%d, added=%d, duplicate=%d, missing_risk_id=%d"
            % (
                merge_summary.scanned_files,
                merge_summary.merged_files,
                merge_summary.missing_target_files,
                merge_summary.total_added,
                merge_summary.total_duplicates,
                merge_summary.total_missing_risk_id,
            )
        )

    if args.task in (TASK_SORT, TASK_MERGE_SORT):
        sort_summary = sort_review_items(
            target_dir=target_dir,
            pattern=args.pattern,
            dry_run=args.dry_run,
        )
        print(
            "[SORT SUMMARY] scanned=%d, sorted=%d, unchanged=%d, meta_mismatch=%d, total_items=%d"
            % (
                sort_summary.scanned_files,
                sort_summary.sorted_files,
                sort_summary.unchanged_files,
                sort_summary.meta_mismatch_files,
                sort_summary.total_items,
            )
        )

    if args.task == TASK_REMOVE_PASS:
        remove_summary = remove_pass_review_items(
            directories=[source_dir, target_dir],
            pattern=args.pattern,
            dry_run=args.dry_run,
        )
        print(
            "[REMOVE-PASS SUMMARY] dirs=%d, scanned=%d, touched=%d, untouched=%d, "
            "removed=%d, total_before=%d, total_after=%d"
            % (
                remove_summary.scanned_dirs,
                remove_summary.scanned_files,
                remove_summary.touched_files,
                remove_summary.untouched_files,
                remove_summary.total_removed,
                remove_summary.total_before,
                remove_summary.total_after,
            )
        )


if __name__ == "__main__":
    main()
