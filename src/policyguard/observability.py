"""LangSmith observability: labels every graph run and records how it ended.

Tracing itself needs no code -- LangGraph sends every run to LangSmith whenever
LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set. These helpers add what a raw trace can't
tell you on its own, so a single LangSmith project can hold every source of traffic:

- where a run came from, as trace tags + metadata: `entrypoint` (ui / api / cli / eval) and
  `environment` (the POLICYGUARD_ENV variable, e.g. "demo" on Streamlit Cloud, "local" default);
- how it ended, as feedback attached to the trace after the run: `outcome` (answered,
  cannot_answer, escalated, greeting, ...), `retries`, `invalid_citations` -- which LangSmith's
  Monitoring tab can chart and filter on;
- the user's thumbs up / down from the UI, as `user_score` feedback on the same trace.

Every function is a no-op when tracing is off, and a LangSmith error is logged, never raised --
observability must not break the answer a user is waiting for.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any

from policyguard.orchestration import nodes

logger = logging.getLogger(__name__)

_ROUTE_OUTCOMES = {
    "handle_greeting": "greeting",
    "handle_out_of_scope": "out_of_scope",
    "handle_clarification": "clarification",
}

_client = None


def tracing_enabled() -> bool:
    flag = os.environ.get("LANGSMITH_TRACING") or os.environ.get("LANGCHAIN_TRACING_V2") or ""
    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    return flag.strip().lower() == "true" and bool(api_key)


def environment() -> str:
    return os.environ.get("POLICYGUARD_ENV", "local")


def trace_labels(entrypoint: str) -> tuple[list[str], dict[str, str]]:
    env = environment()
    return [f"entrypoint:{entrypoint}", f"env:{env}"], {"entrypoint": entrypoint, "environment": env}


def run_config(entrypoint: str, thread_id: str, **configurable: Any) -> dict:
    """Graph config for one run: its checkpoint thread plus LangSmith run id, name, tags, metadata.

    The pre-generated `run_id` becomes the trace's id, so outcome and user feedback can be attached
    to it afterwards. Build a fresh config per run -- a resumed run needs its own run id.
    """
    tags, metadata = trace_labels(entrypoint)
    return {
        "configurable": {"thread_id": thread_id, **configurable},
        "run_id": uuid.uuid4(),
        "run_name": "PolicyGuard",
        "tags": tags,
        "metadata": {**metadata, "thread_id": thread_id},
    }


def outcome_of(result: dict) -> str:
    """How a graph run ended, from its final state (or its `__interrupt__` when it paused)."""
    if result.get("__interrupt__"):
        return "escalated"
    if result.get("human_reviewed"):
        return "human_reviewed"
    if not result.get("documents"):
        route = nodes.route_after_rewrite({"question": result.get("question", ""), "rewritten_query": result.get("rewritten_query", "")})
        return _ROUTE_OUTCOMES.get(route, "cannot_answer")
    if not result.get("graded_documents"):
        return "cannot_answer"
    return "answered"


def outcome_feedback(result: dict) -> list[dict]:
    """Feedback entries describing a finished run. Retry/citation counts only exist when an answer
    was actually generated, so they're left out for greetings, refusals, etc."""
    feedback = [{"key": "outcome", "value": outcome_of(result)}]
    attempts = result.get("retry_count", 0)
    if attempts:
        feedback.append({"key": "retries", "score": attempts - 1})
        feedback.append({"key": "invalid_citations", "score": len(result.get("invalid_citations") or [])})
    return feedback


def record_outcome(run_id: uuid.UUID, result: dict, client=None, background: bool = True) -> None:
    """Attaches the run's outcome feedback to its trace (in a background thread by default, so the
    user isn't kept waiting on LangSmith)."""
    if not tracing_enabled():
        return
    _send(run_id, outcome_feedback(result), client, background)


def record_user_feedback(run_id: uuid.UUID, thumbs_up: bool, client=None, background: bool = True) -> None:
    if not tracing_enabled():
        return
    _send(run_id, [{"key": "user_score", "score": 1 if thumbs_up else 0}], client, background)


def _get_client():
    global _client
    if _client is None:
        from langsmith import Client

        _client = Client()
    return _client


_project_ids: dict[str, str | None] = {}


def _project_id(client) -> str | None:
    """The traced LangSmith project's id (cached), which LangSmith wants alongside run feedback.
    None if it can't be looked up -- feedback is still accepted without it, with a warning."""
    from langsmith.utils import get_tracer_project

    name = get_tracer_project()
    if name not in _project_ids:
        try:
            _project_ids[name] = str(client.read_project(project_name=name).id)
        except Exception:
            logger.warning("Could not look up LangSmith project %r", name, exc_info=True)
            return None
    return _project_ids[name]


def _send(run_id: uuid.UUID, feedback: list[dict], client, background: bool) -> None:
    def send():
        try:
            target = client or _get_client()
            session_id = _project_id(target)
            for item in feedback:
                target.create_feedback(run_id, trace_id=run_id, session_id=session_id, **item)
        except Exception:
            logger.warning("Could not send feedback to LangSmith for run %s", run_id, exc_info=True)

    if background:
        threading.Thread(target=send, daemon=True).start()
    else:
        send()
