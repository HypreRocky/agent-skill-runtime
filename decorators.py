from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from inspect import isclass
from typing import Any, Callable, Dict, List

from utils.errors import SkillError


ExecutorCallable = Callable[..., Dict[str, Any]]


@dataclass(order=True)
class MiddlewareRegistration:
    order: int
    name: str
    factory: Callable[[], Any]


EXECUTOR_REGISTRY: Dict[str, ExecutorCallable] = {}
MIDDLEWARE_REGISTRY: List[MiddlewareRegistration] = []


def register_executor(mode: str) -> Callable[[ExecutorCallable], ExecutorCallable]:
    def decorator(func: ExecutorCallable) -> ExecutorCallable:
        EXECUTOR_REGISTRY[mode] = func
        return func

    return decorator


def get_executor(mode: str) -> ExecutorCallable:
    executor = EXECUTOR_REGISTRY.get(mode)
    if executor is None:
        raise SkillError(f"unsupported mode: {mode}")
    return executor


def register_middleware(*, order: int, name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        middleware_name = name or getattr(factory, "__name__", factory.__class__.__name__)
        MIDDLEWARE_REGISTRY.append(MiddlewareRegistration(order=order, name=middleware_name, factory=factory))
        MIDDLEWARE_REGISTRY.sort(key=lambda item: (item.order, item.name))
        return factory

    return decorator


def build_middlewares() -> List[Any]:
    instances: List[Any] = []
    for registration in MIDDLEWARE_REGISTRY:
        factory = registration.factory
        instance = factory() if isclass(factory) else factory()
        instances.append(instance)
    return instances


def requires_sandbox(func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    @wraps(func)
    def wrapper(ctx: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        ctx.ensure_sandbox()
        return func(ctx, *args, **kwargs)

    return wrapper
