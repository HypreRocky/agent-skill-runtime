from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence


def _normalize_cot(
    cot: Any,
    *,
    mode: str,
    steps_len: Optional[int] = None,
    documents: Optional[Sequence[str]] = None,
) -> List[str]:
    subtitle: List[str] = []

    if isinstance(cot, list):
        subtitle = [str(item).strip() for item in cot if str(item).strip()]

    if mode == "doc_answer":
        if not subtitle:
            subtitle = [_describe_documents(documents)]

    return subtitle


def _describe_documents(documents: Optional[Sequence[str]]) -> str:
    if not documents:
        return "正在查找知识库"
    names = [_display_name(doc) for doc in documents if isinstance(doc, str) and doc]
    names = [name for name in names if name]
    if not names:
        return "正在查找知识库"
    unique = list(dict.fromkeys(names))
    if len(unique) <= 3:
        return f"正在查找{'、'.join(unique)}知识库"
    return f"正在查找{len(unique)}个知识库"


def _display_name(doc: str) -> str:
    base = _strip_suffixes(Path(doc).name)
    if not base:
        return ""
    normalized = base.replace("_", " ").replace("-", " ").strip()
    if not normalized:
        return ""
    if normalized.lower() == "faq":
        return "FAQ"
    if normalized.isascii() and normalized.replace(" ", "").isalpha():
        return normalized.upper()
    if normalized.endswith("知识库"):
        return normalized
    return normalized


def _strip_suffixes(name: str) -> str:
    path = Path(name)
    suffixes = list(path.suffixes)
    if not suffixes:
        return path.name
    base = path.name
    for suffix in suffixes:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base
