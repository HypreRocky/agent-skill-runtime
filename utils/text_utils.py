from __future__ import annotations

from typing import Dict, List, Tuple


def _chunk_text(text: str, max_len: int = 400) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    for para in paragraphs:
        if len(para) <= max_len:
            chunks.append(para)
        else:
            start = 0
            while start < len(para):
                chunk = para[start : start + max_len].strip()
                if chunk:
                    chunks.append(chunk)
                start += max_len
    return chunks


def _rank_chunks(query: str, chunks: List[Dict[str, object]], top_k: int) -> List[Dict[str, object]]:
    if not chunks:
        return []
    if not query:
        return chunks[:top_k]
    query_tokens = _tokenize(query)
    scored: List[Tuple[int, Dict[str, object]]] = []
    for chunk in chunks:
        tokens = _tokenize(str(chunk.get("text", "")))
        score = len(query_tokens.intersection(tokens))
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [chunk for score, chunk in scored[:top_k] if score > 0]
    if top:
        return top
    return chunks[:top_k]


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    token = ""
    for ch in text:
        if ch.isalnum() and not _is_cjk(ch):
            token += ch.lower()
        else:
            if len(token) >= 2:
                tokens.add(token)
            token = ""
    if len(token) >= 2:
        tokens.add(token)

    cjk_seq = ""
    for ch in text:
        if _is_cjk(ch):
            cjk_seq += ch
        else:
            _add_cjk_ngrams(cjk_seq, tokens)
            cjk_seq = ""
    _add_cjk_ngrams(cjk_seq, tokens)
    return tokens


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF


def _add_cjk_ngrams(seq: str, tokens: set[str]) -> None:
    if not seq:
        return
    if len(seq) == 1:
        tokens.add(seq)
        return
    for i in range(len(seq) - 1):
        tokens.add(seq[i : i + 2])
