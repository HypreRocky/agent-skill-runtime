from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import Executor.doc_answer  # noqa: F401
import Executor.run_entrypoint  # noqa: F401
import middlewares  # noqa: F401

from call_llm import get_llm
from decorators import build_middlewares, get_executor
from prompts.system_prompt import build_interpreter_system_prompt
from runtime_types import ActionPlan, SkillIndexEntry, SkillLoaded, SkillRunContext, SkillRuntimeConfig
from sandbox import LocalSkillSandboxProvider
from utils.errors import SkillError
from utils.llm_utils import _build_user_prompt, _chat_with_fallback, _parse_json_from_llm
from utils.runtime_utils import _apply_output_spec
from utils.skill_files import _default_documents, _list_skill_files, _load_references
from utils.skill_io import read_skill_markdown_frontmatter, read_skill_markdown_full, _normalize_meta


class LLMClient:
    """Simple LangChain chat wrapper used by the runtime."""

    def __init__(self, client: Optional[Any] = None) -> None:
        self._client = client or get_llm()

    def chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response = self._client.invoke(messages)
        return str(getattr(response, "content", "")).strip()


class SkillRegistry:
    def __init__(self, skills_root: str | Path) -> None:
        self.skills_root = Path(skills_root).resolve()
        self._index: Dict[str, SkillIndexEntry] = {}

    def index(self) -> None:
        self._index.clear()
        if not self.skills_root.exists():
            return

        for skill_md in sorted(self.skills_root.rglob("SKILL.md")):
            relative_path = skill_md.relative_to(self.skills_root)
            if any(part.startswith(".") for part in relative_path.parts):
                continue
            if not skill_md.parent.is_dir():
                continue

            try:
                meta, _ = read_skill_markdown_frontmatter(skill_md)
            except SkillError:
                continue

            name = str(meta.get("name", "")).strip()
            description = str(meta.get("description", "")).strip()
            if not name:
                continue

            self._index[name] = SkillIndexEntry(
                name=name,
                description=description,
                dir_path=skill_md.parent.resolve(),
                skill_file=skill_md.resolve(),
                relative_path=skill_md.parent.relative_to(self.skills_root),
            )

    def list(self) -> List[SkillIndexEntry]:
        return sorted(self._index.values(), key=lambda item: item.name)

    def load(self, name: str) -> SkillLoaded:
        entry = self._index.get(name)
        if not entry:
            raise SkillError(f"skill not indexed: {name}")

        meta, markdown = read_skill_markdown_full(entry.skill_file)
        meta = _normalize_meta(meta)
        return SkillLoaded(
            name=str(meta.get("name", entry.name)),
            description=str(meta.get("description", entry.description)),
            raw_markdown=markdown,
            meta=meta,
            dir_path=entry.dir_path,
            skill_file=entry.skill_file,
            relative_path=entry.relative_path,
        )


class SkillInterpreter:
    """Interpret skill markdown into a structured ActionPlan."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def interpret(self, ctx: SkillRunContext, skill: SkillLoaded, working_input: Dict[str, Any]) -> ActionPlan:
        documents: List[str] = []
        meta_docs = skill.meta.get("documents")
        if isinstance(meta_docs, list):
            documents = [str(doc) for doc in meta_docs if isinstance(doc, str) and doc]

        default_docs = _default_documents(skill.dir_path)
        if default_docs:
            documents = sorted(set(documents).union(default_docs))

        context = {
            "skill_name": skill.name,
            "skill_description": skill.description,
            "skill_content": skill.raw_markdown,
            "working_input": working_input,
            "skill_files": _list_skill_files(skill.dir_path),
            "reference_texts": _load_references(skill.dir_path, skill.meta.get("references")),
            "documents": documents,
        }

        system = ctx.system_prompt or build_interpreter_system_prompt(
            skill=skill,
            available_skills=ctx.available_skills,
            skills_root=ctx.registry.skills_root,
            sandbox_enabled=ctx.config.sandbox_enabled,
        )
        user = _build_user_prompt(
            "CONTEXT_JSON",
            context,
            "请阅读 skill_content 与 working_input，输出 ActionPlan JSON。",
        )
        raw = _chat_with_fallback(self.llm, system, user, context, "INTERPRET")
        plan = _parse_json_from_llm(raw)
        validated = self._validate_plan(plan, working_input)
        return ActionPlan(mode=str(validated["mode"]), data=validated)

    def _validate_plan(self, plan: Dict[str, Any], working_input: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            raise SkillError("ActionPlan must be a JSON object")

        mode = str(plan.get("mode", "")).strip()
        if mode not in {"doc_answer", "run_entrypoint"}:
            raise SkillError(f"unsupported mode: {mode}")

        cot = plan.get("cot")
        if not isinstance(cot, list):
            plan["cot"] = []

        if mode == "doc_answer":
            if not plan.get("query"):
                query = working_input.get("query")
                if isinstance(query, str) and query.strip():
                    plan["query"] = query.strip()
            return plan

        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            raise SkillError("run_entrypoint requires non-empty steps")

        normalized_steps: List[Dict[str, Any]] = []
        for index, raw_step in enumerate(steps, start=1):
            if not isinstance(raw_step, dict):
                raise SkillError("each step must be a JSON object")

            step = dict(raw_step)
            if step.get("kind") == "llm_text":
                instruction = str(step.get("instruction", "") or step.get("prompt", "")).strip()
                if not instruction:
                    raise SkillError("llm_text step requires instruction")
            else:
                script = step.get("script")
                if not isinstance(script, str) or not script.strip():
                    raise SkillError("script step must include script")
                step["script"] = script.strip()

            step.setdefault("id", f"step-{index}")
            normalized_steps.append(step)

        plan["steps"] = normalized_steps
        return plan


class SkillExecutor:
    """Decorator-driven executor registry."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def execute(self, ctx: SkillRunContext, skill: SkillLoaded, plan: ActionPlan, working_input: Dict[str, Any]) -> Dict[str, Any]:
        executor = get_executor(plan.mode)
        result = executor(ctx, self.llm, skill, plan.data, working_input)
        return _apply_output_spec(result, plan.data.get("output_spec"))


class SkillRuntime:
    """Coordinator: middleware -> interpreter -> executor."""

    def __init__(self, registry: SkillRegistry, llm: Optional[LLMClient] = None, config: Optional[SkillRuntimeConfig] = None) -> None:
        self.registry = registry
        self.llm = llm or LLMClient()
        self.interpreter = SkillInterpreter(self.llm)
        self.executor = SkillExecutor(self.llm)
        self.config = config or SkillRuntimeConfig()
        self.middlewares = build_middlewares()
        self.services: Dict[str, Any] = {}

        if self.config.sandbox_enabled:
            self.services["sandbox_provider"] = LocalSkillSandboxProvider(base_dir=self.config.sandbox_base_dir)

    def run(self, skill_name: str, working_input: Dict[str, Any]) -> Dict[str, Any]:
        ctx = SkillRunContext(
            skill_name=skill_name,
            working_input=working_input,
            registry=self.registry,
            llm=self.llm,
            config=self.config,
            services=dict(self.services),
        )

        def handler() -> Dict[str, Any]:
            return self._run_impl(ctx)

        for middleware in reversed(self.middlewares):
            next_handler = handler

            def handler(middleware=middleware, next_handler=next_handler) -> Dict[str, Any]:
                return middleware(ctx, next_handler)

        try:
            result = handler()
            ctx.result = result
            return result
        finally:
            ctx.cleanup()

    def _run_impl(self, ctx: SkillRunContext) -> Dict[str, Any]:
        if ctx.skill is None:
            raise SkillError("skill must be loaded before execution")

        plan = self.interpreter.interpret(ctx, ctx.skill, ctx.working_input)
        ctx.plan = plan
        return self.executor.execute(ctx, ctx.skill, plan, ctx.working_input)


class SkillNode(SkillRuntime):
    """Backward-compatible alias for callers using the old class name."""
