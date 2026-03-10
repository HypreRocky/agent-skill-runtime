from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Dict, List

from decorators import register_executor, requires_sandbox
from prompts.system_prompt import build_llm_step_system_prompt


def _build_result_payload(plan: Dict[str, Any], results: List[Dict[str, Any]], final_output: Any) -> tuple[str, Dict[str, Any]]:
    result_type = str(plan.get("result_type") or "").strip().lower()
    if not result_type:
        if isinstance(final_output, dict) and "text" in final_output:
            result_type = "text"
        else:
            result_type = "exec"

    if result_type == "text":
        if isinstance(final_output, dict):
            text = final_output.get("text")
            if text is None:
                text = json.dumps(final_output, ensure_ascii=False)
        else:
            text = str(final_output)
        return "text", {"text": text}

    if result_type == "json":
        if isinstance(final_output, dict):
            return "json", final_output
        return "json", {"value": final_output}

    return result_type, {"steps": results, "final": final_output}


def _exec_llm_text_step(
    llm: Any,
    skill: Any,
    step: Dict[str, Any],
    working_input: Dict[str, Any],
    prev_output: Any,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    from utils.errors import SkillError
    from utils.llm_utils import _build_user_prompt, _chat_with_fallback

    instruction = str(step.get("instruction", "") or step.get("prompt", "")).strip()
    if not instruction:
        raise SkillError("llm_text step requires instruction")

    context = {
        "instruction": instruction,
        "working_input": working_input,
        "prev": prev_output,
        "history": results,
        "skill_content": skill.raw_markdown,
    }
    system = build_llm_step_system_prompt(skill.name)
    user = _build_user_prompt("STEP_CONTEXT", context, instruction)
    text = _chat_with_fallback(llm, system, user, context, "LLM_STEP").strip()
    if not text:
        raise SkillError("llm_text step returned empty text")
    return {"ok": True, "text": text}


@register_executor("run_entrypoint")
@requires_sandbox
def exec_run_entrypoint(
    ctx: Any,
    llm: Any,
    skill: Any,
    plan: Dict[str, Any],
    working_input: Dict[str, Any],
) -> Dict[str, Any]:
    from utils.constants import MAX_OUTPUT_BYTES, STDERR_LIMIT
    from utils.errors import SkillError
    from utils.runtime_utils import _build_subprocess_env, _pick_timeout
    from utils.skill_files import _resolve_script_path
    from utils.trace_utils import _normalize_cot

    steps = plan.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise SkillError("run_entrypoint requires non-empty steps")

    sandbox = ctx.ensure_sandbox()
    timeout_s = _pick_timeout(skill.meta.get("timeout_s"), working_input)
    cot = _normalize_cot(plan.get("cot"), mode="run_entrypoint", steps_len=len(steps))

    results: List[Dict[str, Any]] = []
    prev_output = None
    script_root = sandbox.skill_dir / "scripts"

    for step in steps:
        if not isinstance(step, dict):
            raise SkillError("each step must be a JSON object")

        step_id = step.get("id") or step.get("script") or step.get("kind") or "step"
        if step.get("kind") == "llm_text":
            output = _exec_llm_text_step(llm, skill, step, working_input, prev_output, results)
        else:
            script = step.get("script")
            if not script:
                raise SkillError("script step must include script")
            if not script_root.exists():
                raise SkillError("scripts directory not found in skill")
            script_path = _resolve_script_path(script_root, script)
            if not script_path.exists():
                raise SkillError(f"entrypoint not found on disk: {script}")

            payload = {
                "working_input": working_input,
                "args": step.get("args", {}),
                "prev": prev_output,
                "history": results,
                "sandbox": {
                    "root": str(sandbox.root_dir),
                    "skill_dir": str(sandbox.skill_dir),
                    "workspace": str(sandbox.workspace_dir),
                    "uploads": str(sandbox.uploads_dir),
                    "outputs": str(sandbox.outputs_dir),
                },
            }
            suffix = script_path.suffix.lower()
            if suffix == ".py":
                cmd = [sys.executable, str(script_path)]
            elif suffix in {".sh", ".bash"}:
                cmd = ["bash", str(script_path)]
            else:
                raise SkillError(f"unsupported script type: {script}")

            try:
                proc = sandbox.execute(
                    cmd,
                    input_bytes=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=timeout_s,
                    cwd=sandbox.skill_dir,
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
                "kind": step.get("kind", "script"),
                "args": step.get("args", {}),
                "output": output,
            }
        )

    result_type, data = _build_result_payload(plan, results, prev_output)
    return {
        "status": "ok",
        "skill": skill.name,
        "result_type": result_type,
        "data": data,
        "cot": cot,
    }
