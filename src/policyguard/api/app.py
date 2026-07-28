"""FastAPI wrapper around the LangGraph orchestrator.

Endpoints:
    POST /ask      -- ask a question; may come back as answered, cannot_answer, or
                       needs_review (paused for human-in-the-loop, see /resolve).
    POST /resolve  -- resume a thread paused by /ask with a human decision.
    GET  /health   -- liveness check.

Setup (Chroma connection, BM25 index, LLM, checkpointer, graph compilation) happens once at
startup via the lifespan handler and is reused across requests, mirroring the CLI's
interactive mode (policyguard.orchestration.ask). The SQLite checkpointer is what lets a
thread paused for review in one request be resumed by a later, unrelated request.

Endpoints call the compiled graph's blocking `.invoke()` directly rather than `await`ing it;
FastAPI runs sync `def` path functions in a worker thread pool, so this doesn't block the
event loop.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from policyguard.api.schemas import AskRequest, AskResponse, CitationOut, HealthResponse, ResolveRequest
from policyguard.ingestion.vectorstore import PolicyVectorStore
from policyguard.orchestration.graph import build_graph, initial_state
from policyguard.retrieval.reranker import CohereReranker

PERSIST_DIR = Path("./chroma_db")
CHECKPOINT_DB = Path("./checkpoints.sqlite")


def _to_response(result: dict, thread_id: str) -> AskResponse:
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return AskResponse(
            thread_id=thread_id,
            status="needs_review",
            answer=payload["draft_answer"],
            invalid_citations=[CitationOut(**c) for c in payload["invalid_citations"]],
        )

    status = "answered" if result["grounded"] else "cannot_answer"
    return AskResponse(
        thread_id=thread_id,
        status=status,
        answer=result["answer"],
        citations=[CitationOut(**c) for c in result["citations"]],
        invalid_citations=[CitationOut(**c) for c in result["invalid_citations"]],
        human_reviewed=result.get("human_reviewed", False),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()

    store = PolicyVectorStore(PERSIST_DIR)
    reranker = CohereReranker()

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        app.state.graph = build_graph(store, checkpointer=checkpointer, reranker=reranker)
        yield


app = FastAPI(title="PolicyGuard API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = app.state.graph.invoke(initial_state(request.question), config=config)
    return _to_response(result, thread_id)


@app.post("/resolve", response_model=AskResponse)
def resolve(request: ResolveRequest) -> AskResponse:
    if request.action == "edit" and not request.answer:
        raise HTTPException(status_code=400, detail="action 'edit' requires 'answer'")

    config = {"configurable": {"thread_id": request.thread_id}}
    state = app.state.graph.get_state(config)
    if not state.next:
        raise HTTPException(status_code=404, detail=f"No pending review found for thread {request.thread_id!r}")

    decision = {"action": request.action}
    if request.action == "edit":
        decision["answer"] = request.answer

    result = app.state.graph.invoke(Command(resume=decision), config=config)
    return _to_response(result, request.thread_id)
