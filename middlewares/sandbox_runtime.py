from __future__ import annotations

from decorators import register_middleware


@register_middleware(order=40)
class SandboxRuntimeMiddleware:
    def __call__(self, ctx, call_next):
        if not ctx.config.sandbox_enabled:
            return call_next()

        provider = ctx.services.get("sandbox_provider")
        if provider is None:
            return call_next()

        ctx.sandbox_factory = provider.acquire
        try:
            return call_next()
        finally:
            if ctx.sandbox is not None:
                provider.release(ctx.sandbox.sandbox_id)
                ctx.sandbox = None
