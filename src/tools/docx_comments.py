"""DOCX 批注写入工具（基于 python-docx）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logger import Logger

try:
    from docx import Document  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    Document = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class DocxComment:
    """单条批注。"""

    risk_id: str
    anchor_offset: int
    span: str
    content: str


@dataclass(frozen=True, slots=True)
class DocxCommentWriteResult:
    """批注写入结果。"""

    written_count: int
    skipped_count: int
    written_mask: list[bool]


@dataclass(frozen=True, slots=True)
class _ParagraphMeta:
    index: int
    start: int
    end: int
    text: str
    paragraph: Any


def ensure_docx_comment_support() -> None:
    """校验当前环境是否具备 DOCX 批注写入能力。"""

    if Document is None:
        raise RuntimeError("未安装 python-docx。请在环境中安装 `python-docx>=1.2.0`。")
    probe = Document()
    if not hasattr(probe, "add_comment"):
        raise RuntimeError("当前 python-docx 版本不支持 comments API。请升级到 `python-docx>=1.2.0`。")


def add_comments_to_docx(
    *,
    docx_path: Path,
    comments: list[DocxComment],
    logger: Logger | None = None,
) -> DocxCommentWriteResult:
    """向 `.docx` 文档追加批注。

    Args:
        docx_path: 目标 `.docx` 文件路径。
        comments: 待写入批注列表。
        logger: 可选日志对象。

    Returns:
        DocxCommentWriteResult: 写入统计与逐条状态。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
        ValueError: 输入文件不是 `.docx` 时抛出。
        RuntimeError: 缺少 python-docx 或版本不支持 comments API 时抛出。
    """

    if not docx_path.exists():
        raise FileNotFoundError(f"docx 文件不存在: {docx_path}")
    if docx_path.suffix.lower() != ".docx":
        raise ValueError(f"仅支持 .docx 文件批注: {docx_path}")
    if not comments:
        return DocxCommentWriteResult(written_count=0, skipped_count=0, written_mask=[])
    ensure_docx_comment_support()

    document = Document(str(docx_path))

    paragraph_meta = _build_paragraph_meta(document.paragraphs)

    written_mask: list[bool] = [False] * len(comments)
    for index, item in enumerate(comments):
        paragraph = _select_paragraph(paragraph_meta, anchor=item.anchor_offset, span=item.span)
        if paragraph is None:
            continue
        _add_comment_compat(document, paragraph, item.content)
        written_mask[index] = True

    written_count = sum(1 for flag in written_mask if flag)
    skipped_count = len(comments) - written_count
    if written_count <= 0:
        if logger:
            logger.info(f"docx 批注跳过：无可写入锚点。file={docx_path}, comment_count={len(comments)}")
        return DocxCommentWriteResult(
            written_count=written_count,
            skipped_count=skipped_count,
            written_mask=written_mask,
        )

    document.save(str(docx_path))
    if logger:
        logger.info(
            "docx 批注完成: file=%s, written=%s, skipped=%s"
            % (docx_path, written_count, skipped_count)
        )
    return DocxCommentWriteResult(
        written_count=written_count,
        skipped_count=skipped_count,
        written_mask=written_mask,
    )


def _build_paragraph_meta(paragraphs: list[Any]) -> list[_ParagraphMeta]:
    cursor = 0
    result: list[_ParagraphMeta] = []
    for index, paragraph in enumerate(paragraphs):
        text = paragraph.text or ""
        start = cursor
        end = start + len(text)
        result.append(
            _ParagraphMeta(
                index=index,
                start=start,
                end=end,
                text=text,
                paragraph=paragraph,
            )
        )
        cursor = end + 2
    return result


def _select_paragraph(
    paragraph_meta: list[_ParagraphMeta],
    *,
    anchor: int,
    span: str,
) -> Any | None:
    if not paragraph_meta:
        return None

    for item in paragraph_meta:
        if item.start <= anchor <= item.end:
            return item.paragraph

    if span:
        for item in paragraph_meta:
            if span in item.text:
                return item.paragraph

    nearest = min(paragraph_meta, key=lambda item: abs(item.start - anchor))
    return nearest.paragraph


def _add_comment_compat(document: Any, paragraph: Any, text: str) -> None:
    # 为每条批注创建独立锚点，避免在同一 run 上重复挂载 comment 导致结构异常。
    run = paragraph.add_run("\u200b")
    add_comment = getattr(document, "add_comment")

    attempts = (
        lambda: add_comment(runs=[run], text=text, author="", initials=""),
        lambda: add_comment([run], text=text, author="", initials=""),
        lambda: add_comment([run], text, "", ""),
        lambda: add_comment(run, text=text, author="", initials=""),
        lambda: add_comment(run, text, "", ""),
    )

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            attempt()
            return
        except TypeError as exc:
            last_error = exc
            continue

    raise RuntimeError(f"python-docx add_comment 调用失败: {last_error}")
