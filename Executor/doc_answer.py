from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from decorators import register_executor, requires_sandbox
from prompts.system_prompt import build_doc_answer_system_prompt


@register_executor("doc_answer")
@requires_sandbox
def exec_doc_answer(
    ctx: Any,
    llm: Any,
    skill: Any,
    plan: Dict[str, Any],
    working_input: Dict[str, Any],
) -> Dict[str, Any]:
    from utils.errors import SkillError
    from utils.llm_utils import _build_user_prompt, _chat_with_fallback
    from utils.skill_files import _default_documents, _is_within, _read_document_text
    from utils.text_utils import _chunk_text, _rank_chunks
    from utils.trace_utils import _normalize_cot

    query = str(plan.get("query", "") or working_input.get("query", "")).strip()
    top_k = int(plan.get("top_k", 4) or 4)
    sandbox = ctx.ensure_sandbox()
    skill_dir = sandbox.skill_dir if sandbox is not None else skill.dir_path

    plan_docs = plan.get("documents")
    meta_docs = skill.meta.get("documents")
    if isinstance(plan_docs, list):
        documents = plan_docs
    elif isinstance(meta_docs, list):
        documents = meta_docs
    else:
        documents = []

    default_docs = _default_documents(skill_dir)
    if default_docs:
        documents = sorted(set(documents).union(default_docs))
    if not isinstance(documents, list) or not documents:
        raise SkillError("documents not configured")

    chunks: List[Dict[str, Any]] = []
    for rel in documents:
        if not isinstance(rel, str):
            continue
        if Path(rel).is_absolute():
            continue
        doc_path = (skill_dir / rel).resolve()
        if not _is_within(doc_path, skill_dir):
            continue
        if not doc_path.exists():
            continue
        try:
            text = _read_document_text(doc_path)
        except SkillError:
            continue
        for idx, chunk in enumerate(_chunk_text(text)):
            chunks.append({"doc": str(rel), "id": f"{rel}#{idx}", "text": chunk})

    if not chunks:
        raise SkillError("no readable documents in skill directory")

    cot = _normalize_cot(plan.get("cot"), mode="doc_answer", documents=documents)
    ranked = _rank_chunks(query, chunks, top_k)
    citations = [c["id"] for c in ranked]

    system = build_doc_answer_system_prompt(skill.name)
    user = _build_user_prompt(
        "CONTEXT_JSON",
        {"chunks": ranked, "query": query, "skill_content": skill.raw_markdown},
        f"问题：{query}",
    )
    text = _chat_with_fallback(
        llm,
        system,
        user,
        {"chunks": ranked, "query": query, "skill_content": skill.raw_markdown},
        "DOC_ANSWER",
    ).strip()
    if not text:
        text = "文档未包含该信息"

    return {
        "status": "ok",
        "skill": skill.name,
        "result_type": "text",
        "data": {"text": text, "citations": citations},
        "cot": cot,
    }
