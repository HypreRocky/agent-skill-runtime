from __future__ import annotations

from decorators import register_middleware
from utils.errors import SkillError


@register_middleware(order=10)
class ValidationMiddleware:
    def __call__(self, ctx, call_next):
        if not isinstance(ctx.working_input, dict):
            raise SkillError("working_input must be a JSON object")

        name_in_payload = ctx.working_input.get("name")
        if name_in_payload is not None and str(name_in_payload) != str(ctx.skill_name):
            raise SkillError("working_input.name does not match skill_name")

        return call_next()
