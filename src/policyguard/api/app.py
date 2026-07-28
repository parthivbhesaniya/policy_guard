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

import json
import queue
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from policyguard.api.schemas import AskRequest, AskResponse, CitationOut, HealthResponse, ResolveRequest
from policyguard.ingestion.vectorstore import PolicyVectorStore
from policyguard.orchestration.graph import build_graph, initial_state
from policyguard.retrieval.reranker import CohereReranker

PERSIST_DIR = Path("./chroma_db")
CHECKPOINT_DB = Path("./checkpoints.sqlite")


def _extract_interrupt_payload(item: Any) -> dict:
    if isinstance(item, dict):
        return item.get("value", item)
    return getattr(item, "value", item)


def _to_response(result: dict, thread_id: str) -> AskResponse:
    if "__interrupt__" in result and result["__interrupt__"]:
        raw_item = result["__interrupt__"][0]
        payload = _extract_interrupt_payload(raw_item)
        return AskResponse(
            thread_id=thread_id,
            status="needs_review",
            answer=payload.get("draft_answer") or payload.get("answer") or "",
            invalid_citations=[CitationOut(**c) for c in payload.get("invalid_citations", [])],
        )

    status = "answered" if result.get("grounded") else "cannot_answer"
    return AskResponse(
        thread_id=thread_id,
        status=status,
        answer=result.get("answer"),
        citations=[CitationOut(**c) for c in result.get("citations", [])],
        invalid_citations=[CitationOut(**c) for c in result.get("invalid_citations", [])],
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
    history = [turn.model_dump() for turn in request.history]
    result = app.state.graph.invoke(initial_state(request.question, history=history), config=config)
    return _to_response(result, thread_id)


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    thread_id = request.thread_id or str(uuid.uuid4())
    history = [turn.model_dump() for turn in request.history]
    init_state = initial_state(request.question, history=history)

    def event_generator():
        token_queue: queue.Queue = queue.Queue()
        stream_config = {
            "configurable": {
                "thread_id": thread_id,
                "token_queue": token_queue,
            }
        }
        result_holder: dict = {}
        error_holder: dict = {}

        def run_pipeline():
            try:
                for update in app.state.graph.stream(init_state, config=stream_config, stream_mode="updates"):
                    node_name = list(update.keys())[0]
                    token_queue.put({"type": "stage", "node": node_name})

                final_state = app.state.graph.get_state(stream_config)
                if final_state.next:
                    snapshot = final_state.tasks[0].interrupts[0].value
                    result_holder["res"] = {"__interrupt__": [{"value": snapshot}]}
                else:
                    result_holder["res"] = final_state.values
            except Exception as exc:
                error_holder["error"] = str(exc)
            finally:
                token_queue.put(None)

        worker = threading.Thread(target=run_pipeline)
        worker.start()

        yield f"data: {json.dumps({'type': 'status', 'content': 'Analyzing question...'})}\n\n"

        while True:
            item = token_queue.get()
            if item is None:
                break

            if isinstance(item, str):
                yield f"data: {json.dumps({'type': 'token', 'content': item})}\n\n"
            elif isinstance(item, dict) and item.get("type") == "stage":
                node = item["node"]
                if node == "rewrite_query":
                    yield f"data: {json.dumps({'type': 'status', 'content': 'Searching policy documents...'})}\n\n"
                elif node in ("handle_greeting", "handle_out_of_scope", "handle_clarification"):
                    yield f"data: {json.dumps({'type': 'status', 'content': 'Responding...'})}\n\n"
                elif node == "retrieve":
                    yield f"data: {json.dumps({'type': 'status', 'content': 'Evaluating excerpt relevance...'})}\n\n"
                elif node == "grade_documents":
                    yield f"data: {json.dumps({'type': 'status', 'content': 'Generating answer...'})}\n\n"
                elif node == "generate":
                    yield f"data: {json.dumps({'type': 'status', 'content': 'Verifying citations & groundedness...'})}\n\n"

        worker.join()

        if "error" in error_holder:
            yield f"data: {json.dumps({'type': 'error', 'content': error_holder['error']})}\n\n"
            return

        final_res = result_holder.get("res", {})
        response_obj = _to_response(final_res, thread_id)
        yield f"data: {json.dumps({'type': 'final', 'data': response_obj.model_dump()})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
