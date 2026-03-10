from __future__ import annotations

from decorators import register_middleware
from prompts.system_prompt import build_interpreter_system_prompt


@register_middleware(order=30)
class PromptContextMiddleware:
    def __call__(self, ctx, call_next):
        ctx.available_skills = ctx.registry.list()
        if ctx.skill is not None:
            ctx.system_prompt = build_interpreter_system_prompt(
                skill=ctx.skill,
                available_skills=ctx.available_skills,
                skills_root=ctx.registry.skills_root,
                sandbox_enabled=ctx.config.sandbox_enabled,
            )
        return call_next()
