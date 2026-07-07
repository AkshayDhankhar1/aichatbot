"""RAG pipeline: hybrid retrieval -> relevance gate -> Groq generation.

Exposes a single async generator, `answer_stream`, that yields event dicts the
`/chat` route turns into Server-Sent Events:

    {"type": "token", "content": "..."}          # incremental text
    {"type": "done",  "sources": [...], "chart": {...}|None}
    {"type": "error", "message": "..."}

No reranker (per project decision): the anti-hallucination gate is the hybrid
fusion score of the top retrieved chunk vs. MIN_SCORE_THRESHOLD.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from .config import get_settings
from .prompts import (
    CHART_SYSTEM_PROMPT,
    NOT_FOUND_MESSAGE,
    SYSTEM_PROMPT,
    wants_chart,
)
from .vectorstore import embed_query, get_index, hybrid_scale, load_bm25


# --------------------------------------------------------------------------- #
# Retrieval                                                                     #
# --------------------------------------------------------------------------- #
def retrieve(question: str) -> list[dict]:
    """Hybrid (dense + sparse BM25) search over Pinecone.

    Returns a list of {text, source, score} sorted by score desc (Pinecone
    already returns them ranked).
    """
    settings = get_settings()

    dense = embed_query(question)
    sparse = load_bm25().encode_queries(question)
    scaled_dense, scaled_sparse = hybrid_scale(dense, sparse, settings.hybrid_alpha)

    index = get_index()
    query_kwargs = dict(vector=scaled_dense, top_k=settings.top_k, include_metadata=True)
    # Omit sparse when the query has no in-vocabulary terms (Pinecone rejects an
    # empty sparse vector); dense-only search still works.
    if scaled_sparse["indices"]:
        query_kwargs["sparse_vector"] = scaled_sparse
    res = index.query(**query_kwargs)
    matches = []
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        matches.append({
            "text": md.get("text", ""),
            "source": md.get("source", "unknown"),
            "score": float(m.get("score", 0.0)),
        })
    return matches


def build_context(matches: list[dict]) -> str:
    """Render top-n chunks with [source] tags for inline attribution."""
    settings = get_settings()
    top = matches[: settings.top_n]
    blocks = []
    for m in top:
        blocks.append(f"[{m['source']}]\n{m['text']}")
    return "\n\n---\n\n".join(blocks)


def unique_sources(matches: list[dict]) -> list[str]:
    settings = get_settings()
    seen: list[str] = []
    for m in matches[: settings.top_n]:
        if m["source"] not in seen:
            seen.append(m["source"])
    return seen


# --------------------------------------------------------------------------- #
# LLM                                                                           #
# --------------------------------------------------------------------------- #
def _llm(streaming: bool) -> ChatGroq:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Fill in backend/.env.")
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.temperature,
        streaming=streaming,
    )


def _history_messages(history: list[dict] | None):
    """Convert prior session turns (frontend-supplied) into LC messages."""
    msgs = []
    for turn in (history or [])[-6:]:  # keep it short; session-only, no DB
        role = turn.get("role")
        content = turn.get("content", "")
        if not content:
            continue
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


def _parse_chart_json(raw: str) -> dict:
    """Best-effort parse of the chart-mode JSON payload."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # drop a leading 'json' language hint if present
        if text[:4].lower() == "json":
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def _valid_chart(chart) -> bool:
    if not isinstance(chart, dict):
        return False
    if chart.get("type") not in ("bar", "line", "pie"):
        return False
    data = chart.get("data")
    if not isinstance(data, list) or not data:
        return False
    return all(
        isinstance(d, dict) and "label" in d and isinstance(d.get("value"), (int, float))
        for d in data
    )


# --------------------------------------------------------------------------- #
# Public entrypoint                                                             #
# --------------------------------------------------------------------------- #
async def answer_stream(
    question: str, history: list[dict] | None = None
) -> AsyncGenerator[dict, None]:
    settings = get_settings()

    # 1) Retrieve
    try:
        matches = retrieve(question)
    except Exception as exc:
        yield {"type": "error", "message": f"Retrieval failed: {exc}"}
        return

    top_score = matches[0]["score"] if matches else 0.0

    # 2) Anti-hallucination gate — below threshold => don't generate.
    if not matches or top_score < settings.min_score_threshold:
        yield {"type": "token", "content": NOT_FOUND_MESSAGE}
        yield {"type": "done", "sources": [], "chart": None}
        return

    context = build_context(matches)
    sources = unique_sources(matches)

    # 3a) Chart mode — explicit request only. Non-streaming JSON.
    if wants_chart(question):
        try:
            llm = _llm(streaming=False)
            messages = (
                [SystemMessage(content=CHART_SYSTEM_PROMPT.format(context=context))]
                + _history_messages(history)
                + [HumanMessage(content=question)]
            )
            resp = await llm.ainvoke(messages)
            payload = _parse_chart_json(resp.content)
            answer = payload.get("answer", "") or ""
            chart = payload.get("chart")
            chart = chart if _valid_chart(chart) else None
        except Exception as exc:
            yield {"type": "error", "message": f"Chart generation failed: {exc}"}
            return
        yield {"type": "token", "content": answer or NOT_FOUND_MESSAGE}
        yield {"type": "done", "sources": sources if chart else [], "chart": chart}
        return

    # 3b) Normal mode — stream tokens from Groq.
    try:
        llm = _llm(streaming=True)
        messages = (
            [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
            + _history_messages(history)
            + [HumanMessage(content=question)]
        )
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield {"type": "token", "content": chunk.content}
    except Exception as exc:
        yield {"type": "error", "message": f"Generation failed: {exc}"}
        return

    yield {"type": "done", "sources": sources, "chart": None}
