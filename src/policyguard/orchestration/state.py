"""Shared state passed between nodes of the LangGraph orchestrator.

This state is checkpointed (serialized to SQLite for human-in-the-loop resumability), so
`documents`/`graded_documents`/`citations`/`invalid_citations` are stored as plain dicts
(``{"doc_id": ..., "section": ..., "text": ...}`` / ``{"doc_id": ..., "section": ...}``)
rather than the ContextBlock/Citation dataclasses from policyguard.generation -- the
checkpoint serializer doesn't recognize custom dataclasses. Nodes convert at the boundary;
see the `_block_to_dict` / `_block_from_dict` / `_citation_to_dict` helpers in nodes.py.
"""

from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict):
    question: str
    # Prior turns of the same conversation -- [{"question": ..., "answer": ...}, ...], oldest
    # first. Populated by the caller (CLI/API/UI), not the graph itself: each `invoke()` is one
    # independent question, so nothing here appends to it automatically. Consumed only by
    # `rewrite_query`, to resolve references like "this"/"it"/"the same policy" into a
    # standalone search query -- it is deliberately NOT passed to `generate`, so every answer
    # still comes only from retrieved, graded policy excerpts rather than conversational memory.
    history: list[dict]
    rewritten_query: str
    documents: list[dict]
    graded_documents: list[dict]
    answer: str
    citations: list[dict]
    invalid_citations: list[dict]
    grounded: bool
    hallucination_score: float
    retry_count: int
    needs_human_review: bool
    human_reviewed: bool
