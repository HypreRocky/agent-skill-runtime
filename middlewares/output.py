from __future__ import annotations

from decorators import register_middleware
from utils.errors import SkillError


@register_middleware(order=0)
class OutputMiddleware:
    def __call__(self, ctx, call_next):
        try:
            result = call_next()
        except SkillError as exc:
            result = {
                "status": "error",
                "skill": ctx.skill_name,
                "result_type": "error",
                "data": {"message": str(exc)},
            }
        except Exception as exc:
            result = {
                "status": "error",
                "skill": ctx.skill_name,
                "result_type": "error",
                "data": {"message": f"{type(exc).__name__}: {exc}"},
            }

        if not isinstance(result, dict):
            result = {
                "status": "error",
                "skill": ctx.skill_name,
                "result_type": "error",
                "data": {"message": "runtime result must be a JSON object"},
            }

        result.setdefault("skill", ctx.skill_name)
        result.setdefault("status", "ok")
        return result
