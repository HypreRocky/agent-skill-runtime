from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def exec_doc_answer(
    llm: Any,
    skill: Any,
    plan: Dict[str, Any],
    working_input: Dict[str, Any],
) -> Dict[str, Any]:
    from skill_runtime import (
        SkillError,
        _build_system_prompt,
        _build_user_prompt,
        _chat_with_fallback,
        _chunk_text,
        _default_documents,
        _is_within,
        _rank_chunks,
        _read_document_text,
    )

    query = str(plan.get("query", "") or working_input.get("query", "")).strip()
    top_k = int(plan.get("top_k", 4) or 4)
    plan_docs = plan.get("documents")
    meta_docs = skill.meta.get("documents")
    if isinstance(plan_docs, list):
        documents = plan_docs
    elif isinstance(meta_docs, list):
        documents = meta_docs
    else:
        documents = []
    default_docs = _default_documents(skill.dir_path)
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
        doc_path = (skill.dir_path / rel).resolve()
        if not _is_within(doc_path, skill.dir_path):
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
    ranked = _rank_chunks(query, chunks, top_k)
    citations = [c["id"] for c in ranked]

    system = _build_system_prompt(
        "DOC_ANSWER",
        "只能使用给定 chunks 回答。若未找到，必须输出：文档未包含该信息。仅输出纯文本。",
    )
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
    }
