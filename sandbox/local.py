from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class LocalSkillSandbox:
    sandbox_id: str
    root_dir: Path
    skill_dir: Path
    workspace_dir: Path
    uploads_dir: Path
    outputs_dir: Path

    def execute(
        self,
        command: list[str],
        *,
        input_bytes: bytes,
        timeout: float,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess[bytes]:
        merged_env = dict(os.environ)
        merged_env.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "SKILL_SANDBOX_ROOT": str(self.root_dir),
                "SKILL_SANDBOX_SKILL_DIR": str(self.skill_dir),
                "SKILL_SANDBOX_WORKSPACE": str(self.workspace_dir),
                "SKILL_SANDBOX_UPLOADS": str(self.uploads_dir),
                "SKILL_SANDBOX_OUTPUTS": str(self.outputs_dir),
            }
        )
        if env:
            merged_env.update({key: value for key, value in env.items() if value is not None})
        return subprocess.run(
            command,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd or self.skill_dir),
            timeout=timeout,
            check=False,
            env=merged_env,
        )

    def resolve_skill_path(self, relative_path: str) -> Path:
        candidate = (self.skill_dir / relative_path).resolve()
        candidate.relative_to(self.skill_dir)
        return candidate

    def cleanup(self) -> None:
        shutil.rmtree(self.root_dir, ignore_errors=True)


class LocalSkillSandboxProvider:
    def __init__(self, base_dir: Path | None = None) -> None:
        root = base_dir or Path(tempfile.gettempdir()) / "custom-skill-sandboxes"
        self.base_dir = Path(root).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._sandboxes: dict[str, LocalSkillSandbox] = {}

    def acquire(self, ctx: object) -> LocalSkillSandbox:
        sandbox_id = getattr(ctx, "request_id")
        sandbox = self._sandboxes.get(sandbox_id)
        if sandbox is not None:
            return sandbox

        skill = getattr(ctx, "skill")
        if skill is None:
            raise RuntimeError("skill must be loaded before acquiring sandbox")

        root_dir = Path(tempfile.mkdtemp(prefix=f"skill-{sandbox_id}-", dir=str(self.base_dir)))
        skill_dir = root_dir / "skill"
        workspace_dir = root_dir / "workspace"
        uploads_dir = root_dir / "uploads"
        outputs_dir = root_dir / "outputs"

        shutil.copytree(skill.dir_path, skill_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        sandbox = LocalSkillSandbox(
            sandbox_id=sandbox_id,
            root_dir=root_dir,
            skill_dir=skill_dir,
            workspace_dir=workspace_dir,
            uploads_dir=uploads_dir,
            outputs_dir=outputs_dir,
        )
        self._sandboxes[sandbox_id] = sandbox
        return sandbox

    def release(self, sandbox_id: str) -> None:
        sandbox = self._sandboxes.pop(sandbox_id, None)
        if sandbox is not None:
            sandbox.cleanup()
