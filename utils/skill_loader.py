from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple


def load_skills(skills_root_path: str | Path) -> Dict[str, Dict[str, str]]:
    skills_root = Path(skills_root_path)
    results: Dict[str, Dict[str, str]] = {}
    if not skills_root.exists():
        return results

    for child in skills_root.iterdir():
        if not child.is_dir():
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.exists():
            continue
        meta = _read_frontmatter(skill_md)
        name = str(meta.get("name", "")).strip()
        description = str(meta.get("description", "")).strip()
        if not name:
            continue
        results[name] = {"name": name, "description": description}

    return results


def _read_frontmatter(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8")
    meta, _body = _parse_frontmatter(text)
    return meta


def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, text
    fm_lines = []
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
        fm_lines.append(lines[i])
    if end_idx is None:
        return {}, text
    raw = "\n".join(fm_lines)
    meta: Dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    body = "\n".join(lines[end_idx + 1 :])
    return meta, body
