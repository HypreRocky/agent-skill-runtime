from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from utils.errors import SkillError


def _list_skill_files(skill_dir: Path) -> Dict[str, List[str]]:
    scripts: List[str] = []
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.exists():
        return {"scripts": []}
    for path in scripts_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".py", ".sh", ".bash"}:
            continue
        rel = path.relative_to(skill_dir).as_posix()
        scripts.append(rel)
    return {"scripts": sorted(scripts)}


def _load_references(skill_dir: Path, references: Any) -> List[Dict[str, Any]]:
    if not isinstance(references, list):
        return []
    results: List[Dict[str, Any]] = []
    for rel in references:
        if not isinstance(rel, str):
            continue
        if Path(rel).is_absolute():
            continue
        ref_path = (skill_dir / rel).resolve()
        if not _is_within(ref_path, skill_dir):
            continue
        if not ref_path.exists():
            continue
        try:
            text = _read_document_text(ref_path)
        except SkillError:
            continue
        results.append({"path": rel, "text": text})
    return results


def _default_documents(skill_dir: Path) -> List[str]:
    docs: List[str] = []
    docs_dir = skill_dir / "docs"
    if docs_dir.exists():
        for path in docs_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(skill_dir).as_posix()
            if rel.lower().endswith((".md", ".txt", ".pdf", ".docx")):
                docs.append(rel)
    return sorted(docs)


def _read_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return _read_text_with_fallback(path)
    if suffix == ".pdf":
        return _read_pdf_text(path)
    if suffix == ".docx":
        return _read_docx_text(path)
    raise SkillError(f"unsupported document type: {path.name}")


def _read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception as exc:
            raise SkillError("missing dependency for PDF: install pypdf") from exc
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts).strip()


def _read_docx_text(path: Path) -> str:
    try:
        import docx  # type: ignore
    except Exception as exc:
        raise SkillError("missing dependency for DOCX: install python-docx") from exc
    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs if p.text]
    return "\n".join(parts).strip()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_script_path(script_root: Path, script: Any) -> Path:
    if not isinstance(script, str):
        raise SkillError("script must be a string")
    if Path(script).is_absolute():
        raise SkillError("script path must be relative to scripts/")
    rel = Path(script)
    if rel.parts and rel.parts[0] == "scripts":
        rel = Path(*rel.parts[1:])
    candidate = (script_root / rel).resolve()
    if not _is_within(candidate, script_root):
        raise SkillError("entrypoint escapes scripts directory")
    return candidate
