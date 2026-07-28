"""Pydantic request/response models for the PolicyGuard HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HistoryTurn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str
    thread_id: str | None = None
    # Prior turns of this conversation, oldest first -- used only to resolve references like
    # "this"/"it" in `question` into a standalone search query. The API is otherwise stateless
    # across calls, so the caller (UI/CLI) is responsible for accumulating and resending this.
    history: list[HistoryTurn] = []


class CitationOut(BaseModel):
    doc_id: str
    section: str


class AskResponse(BaseModel):
    thread_id: str
    # "answered": grounded answer returned.
    # "cannot_answer": no relevant documents were found for the question.
    # "needs_review": the answer couldn't be verified as grounded and is paused for a human;
    #   resolve it via POST /resolve before it can be treated as final.
    status: Literal["answered", "cannot_answer", "needs_review"]
    answer: str | None = None
    citations: list[CitationOut] = []
    invalid_citations: list[CitationOut] = []
    human_reviewed: bool = False


class ResolveRequest(BaseModel):
    thread_id: str
    action: Literal["approve", "edit", "reject"]
    answer: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
