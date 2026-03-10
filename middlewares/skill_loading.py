from __future__ import annotations

from decorators import register_middleware


@register_middleware(order=20)
class SkillLoadingMiddleware:
    def __call__(self, ctx, call_next):
        ctx.skill = ctx.registry.load(ctx.skill_name)
        return call_next()
