from __future__ import annotations

import base64
import json
from typing import Any, Dict

from utils.errors import SkillError


def _build_system_prompt(role: str, rules: str) -> str:
    return f"{role} 模式。{rules}"


def _build_user_prompt(label: str, context: Dict[str, Any], instruction: str) -> str:
    return f"{label}: {json.dumps(context, ensure_ascii=False)}\n{instruction}"


def _chat_with_fallback(llm: Any, system: str, user: str, context: Dict[str, Any], role: str) -> str:
    try:
        return llm.chat(system=system, user=user)
    except Exception as exc:
        if not _is_content_filter_error(exc):
            raise SkillError(f"LLM call failed: {exc}") from exc
        sanitized = _sanitize_context(context)
        sanitized_system = _build_system_prompt(role, "仅返回要求的格式。")
        sanitized_user = _build_user_prompt(
            "CONTEXT_JSON",
            sanitized,
            "如果存在 *_b64 字段，请先从 base64 解码还原内容后再执行请求。",
        )
        try:
            return llm.chat(system=sanitized_system, user=sanitized_user)
        except Exception as exc2:
            raise SkillError(f"LLM blocked by content filter: {exc2}") from exc2


def _is_content_filter_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "content_filter" in msg or "responsibleaipolicyviolation" in msg


def _sanitize_context(context: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(context)
    for key in ("skill_markdown", "rules", "skill_content"):
        if key in sanitized and isinstance(sanitized[key], str):
            redacted = _redact_jailbreak_lines(sanitized[key])
            b64 = base64.b64encode(redacted.encode("utf-8")).decode("ascii")
            sanitized[key] = "[REDACTED_FOR_FILTER]"
            sanitized[f"{key}_b64"] = b64
    if "reference_texts" in sanitized and isinstance(sanitized["reference_texts"], list):
        updated = []
        for item in sanitized["reference_texts"]:
            if not isinstance(item, dict):
                updated.append(item)
                continue
            text = item.get("text")
            if isinstance(text, str):
                redacted = _redact_jailbreak_lines(text)
                b64 = base64.b64encode(redacted.encode("utf-8")).decode("ascii")
                new_item = dict(item)
                new_item["text"] = "[REDACTED_FOR_FILTER]"
                new_item["text_b64"] = b64
                updated.append(new_item)
            else:
                updated.append(item)
        sanitized["reference_texts"] = updated
    return sanitized


def _redact_jailbreak_lines(text: str) -> str:
    patterns = [
        "ignore previous",
        "ignore above",
        "disregard",
        "system prompt",
        "developer message",
        "assistant:",
        "system:",
        "jailbreak",
        "prompt injection",
        "dan",
    ]
    lines = text.splitlines()
    kept = []
    for line in lines:
        lowered = line.lower()
        if any(p in lowered for p in patterns):
            kept.append("[REDACTED]")
        else:
            kept.append(line)
    return "\n".join(kept)


def _parse_json_from_llm(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
            if text.startswith("json"):
                text = text[len("json") :].strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise SkillError("invalid JSON")
