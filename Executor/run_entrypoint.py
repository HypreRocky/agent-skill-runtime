from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Dict


def exec_run_entrypoint(
    skill: Any,
    plan: Dict[str, Any],
    working_input: Dict[str, Any],
) -> Dict[str, Any]:
    from utils.constants import MAX_OUTPUT_BYTES, STDERR_LIMIT
    from utils.errors import SkillError
    from utils.runtime_utils import _build_subprocess_env, _pick_timeout
    from utils.skill_files import _resolve_script_path

    steps = plan.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise SkillError("run_entrypoint requires non-empty steps")
    timeout_s = _pick_timeout(skill.meta.get("timeout_s"), working_input)

    results = []
    prev_output = None
    script_root = skill.dir_path / "scripts"
    for step in steps:
        if not isinstance(step, dict) or not step.get("script"):
            raise SkillError("each step must include script")
        step_id = step.get("id") or step.get("script")
        script = step.get("script")
        if not script_root.exists():
            raise SkillError("scripts directory not found in skill")
        script_path = _resolve_script_path(script_root, script)
        if not script_path.exists():
            raise SkillError(f"entrypoint not found on disk: {script}")

        payload = {
            "working_input": working_input,
            "args": step.get("args", {}),
            "prev": prev_output,
        }
        suffix = script_path.suffix.lower()
        if suffix == ".py":
            cmd = [sys.executable, str(script_path)]
        elif suffix in {".sh", ".bash"}:
            cmd = ["bash", str(script_path)]
        else:
            raise SkillError(f"unsupported script type: {script}")

        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(skill.dir_path),
                timeout=timeout_s,
                check=False,
                env=_build_subprocess_env(),
            )
        except subprocess.TimeoutExpired:
            raise SkillError(f"step timed out after {timeout_s}s: {step_id}")
        except OSError as exc:
            raise SkillError(f"failed to start step {step_id}: {exc}")

        stdout_bytes = proc.stdout or b""
        stderr_bytes = proc.stderr or b""

        if len(stdout_bytes) > MAX_OUTPUT_BYTES:
            raise SkillError(f"stdout exceeded limit: {step_id}")

        if len(stderr_bytes) > MAX_OUTPUT_BYTES:
            stderr_bytes = stderr_bytes[:MAX_OUTPUT_BYTES]

        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise SkillError(f"step failed with code {proc.returncode}: {stderr_text[:STDERR_LIMIT]}")

        stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
        if not stdout_text:
            raise SkillError(f"step returned empty stdout: {step_id}")

        try:
            output = json.loads(stdout_text)
        except json.JSONDecodeError:
            raise SkillError(f"step stdout is not valid JSON: {step_id}")

        prev_output = output
        results.append(
            {
                "id": step_id,
                "script": step.get("script"),
                "args": step.get("args", {}),
                "output": output,
            }
        )

    return {
        "status": "ok",
        "skill": skill.name,
        "result_type": "exec",
        "data": {"steps": results, "final": prev_output},
    }
