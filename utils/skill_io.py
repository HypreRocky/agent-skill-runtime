from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.errors import SkillError


def read_skill_markdown_frontmatter(path: Path) -> Tuple[Dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    meta, _body = parse_frontmatter(text)
    if not meta:
        raise SkillError(f"missing frontmatter in {path}")
    return meta, _body


def read_skill_markdown_full(path: Path) -> Tuple[Dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    if not meta:
        raise SkillError(f"missing frontmatter in {path}")
    return meta, text


def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    lines = text.splitlines()
    if len(lines) < 3:
        return {}, text
    first = lines[0].lstrip("\ufeff").strip()
    if first != "---":
        return {}, text
    fm_lines: List[str] = []
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
        fm_lines.append(lines[i])
    if end_idx is None:
        return {}, text

    raw = "\n".join(fm_lines)
    meta = _parse_yaml(raw)
    body = "\n".join(lines[end_idx + 1 :])
    return meta, body


def _parse_yaml(raw: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw) or {}
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        pass

    meta: Dict[str, Any] = {}
    current_key: Optional[str] = None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0 and ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                meta[key] = []
                current_key = key
            else:
                meta[key] = _coerce_scalar(value)
                current_key = None
            continue
        if stripped.startswith("-"):
            if current_key is None or not isinstance(meta.get(current_key), list):
                continue
            item = stripped[1:].strip()
            if ":" in item:
                k, v = item.split(":", 1)
                meta[current_key].append({k.strip(): _coerce_scalar(v.strip())})
            else:
                meta[current_key].append(_coerce_scalar(item))
            continue
        if current_key and isinstance(meta.get(current_key), list) and meta[current_key]:
            last = meta[current_key][-1]
            if isinstance(last, dict) and ":" in stripped:
                k, v = stripped.split(":", 1)
                last[k.strip()] = _coerce_scalar(v.strip())

    return meta


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _normalize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(meta)
