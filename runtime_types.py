from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from utils.errors import SkillError


@dataclass
class SkillIndexEntry:
    name: str
    description: str
    dir_path: Path
    skill_file: Path
    relative_path: Path


@dataclass
class SkillLoaded:
    name: str
    description: str
    raw_markdown: str
    meta: Dict[str, Any]
    dir_path: Path
    skill_file: Path
    relative_path: Path


@dataclass
class ActionPlan:
    mode: str
    data: Dict[str, Any]


@dataclass
class SkillRuntimeConfig:
    sandbox_enabled: bool = True
    sandbox_base_dir: Optional[Path] = None


@dataclass
class SkillRunContext:
    skill_name: str
    working_input: Dict[str, Any]
    registry: Any
    llm: Any
    config: SkillRuntimeConfig
    services: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex[:12])
    available_skills: List[SkillIndexEntry] = field(default_factory=list)
    skill: Optional[SkillLoaded] = None
    plan: Optional[ActionPlan] = None
    result: Optional[Dict[str, Any]] = None
    system_prompt: str = ""
    sandbox: Any = None
    sandbox_factory: Optional[Callable[[SkillRunContext], Any]] = None
    state: Dict[str, Any] = field(default_factory=dict)
    cleanup_callbacks: List[Callable[[], None]] = field(default_factory=list)

    def ensure_sandbox(self) -> Any:
        if self.sandbox is not None:
            return self.sandbox
        if not self.config.sandbox_enabled:
            raise SkillError("sandbox is disabled for this runtime")
        if self.sandbox_factory is None:
            raise SkillError("sandbox factory is not configured")
        self.sandbox = self.sandbox_factory(self)
        return self.sandbox

    def add_cleanup(self, callback: Callable[[], None]) -> None:
        self.cleanup_callbacks.append(callback)

    def cleanup(self) -> None:
        for callback in reversed(self.cleanup_callbacks):
            try:
                callback()
            except Exception:
                continue
