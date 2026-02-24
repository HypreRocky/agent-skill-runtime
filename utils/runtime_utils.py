from __future__ import annotations

import os
from typing import Any, Dict, Optional

from utils.constants import DEFAULT_TIMEOUT_S
from utils.errors import SkillError


def _build_subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _pick_timeout(meta_timeout: Any, working_input: Dict[str, Any]) -> float:
    if isinstance(working_input, dict):
        for key in ("timeout_s", "timeout"):
            if key in working_input:
                coerced = _coerce_timeout(working_input.get(key))
                if coerced:
                    return coerced
    coerced_meta = _coerce_timeout(meta_timeout)
    if coerced_meta:
        return coerced_meta
    return DEFAULT_TIMEOUT_S


def _coerce_timeout(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _apply_output_spec(result: Dict[str, Any], output_spec: Any) -> Dict[str, Any]:
    if not isinstance(output_spec, dict):
        return result

    fmt = output_spec.get("format")
    required = output_spec.get("required_fields")
    if fmt and fmt != "json":
        raise SkillError(f"unsupported output format: {fmt}")

    if isinstance(required, list):
        missing = [field for field in required if field not in result]
        if missing:
            raise SkillError(f"output missing required fields: {missing}")

    wrap = output_spec.get("wrap_result")
    if isinstance(wrap, str) and wrap:
        return {wrap: result}

    return result
