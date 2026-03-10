from __future__ import annotations

from pathlib import Path
from typing import Dict

from utils.skill_io import read_skill_markdown_frontmatter


def load_skills(skills_root_path: str | Path) -> Dict[str, Dict[str, str]]:
    skills_root = Path(skills_root_path).resolve()
    results: Dict[str, Dict[str, str]] = {}
    if not skills_root.exists():
        return results

    for skill_md in sorted(skills_root.rglob("SKILL.md")):
        relative_parts = skill_md.relative_to(skills_root).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        if not skill_md.parent.is_dir():
            continue
        try:
            meta, _ = read_skill_markdown_frontmatter(skill_md)
        except Exception:
            continue
        name = str(meta.get("name", "")).strip()
        description = str(meta.get("description", "")).strip()
        if not name:
            continue
        results[name] = {
            "name": name,
            "description": description,
            "path": str(skill_md),
        }

    return results
