"""Self-contained Streamlit UI for PolicyGuard.

Runs the LangGraph orchestrator directly in-process (build_graph + .invoke/.stream) instead of
calling a separate FastAPI backend -- this is the single file to deploy on Streamlit Community
Cloud: no second service to host, no POLICYGUARD_API_URL to wire up.

Required secrets (Streamlit Cloud: App settings -> Secrets, or a local .env for `streamlit run`):
    GROQ_API_KEY        -- required
    COHERE_API_KEY      -- optional; reranking is skipped automatically if unset
    HUGGINGFACE_API_KEY -- optional; falls back to Chroma's local default embedder if unset
    LANGSMITH_TRACING, LANGSMITH_API_KEY, LANGSMITH_PROJECT -- optional; turn on LangSmith tracing
                           and the 👍/👎 buttons (see policyguard.observability)
    POLICYGUARD_ENV     -- optional; labels this deployment's traces, e.g. "demo" (default "local")

The vector store and checkpoint state live only in this process's memory/disk and are rebuilt
from data/policies/ on every cold start -- Streamlit Cloud's filesystem is ephemeral, so nothing
here assumes chroma_db/ or prior conversation state survives a restart.
"""

from __future__ import annotations

import os
import queue
import threading
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from policyguard import observability
from policyguard.ingestion.ingest import ingest as ingest_policies
from policyguard.ingestion.vectorstore import PolicyVectorStore
from policyguard.orchestration.graph import build_graph, initial_state
from policyguard.retrieval.reranker import CohereReranker

REPO_ROOT = Path(__file__).resolve().parents[3]
PERSIST_DIR = REPO_ROOT / "chroma_db"
POLICIES_DIR = REPO_ROOT / "data" / "policies"

MAX_HISTORY_TURNS = 6

_STAGE_LABELS = {
    "rewrite_query": "Searching policy documents...",
    "handle_greeting": "Responding...",
    "handle_out_of_scope": "Responding...",
    "handle_clarification": "Responding...",
    "retrieve": "Evaluating excerpt relevance...",
    "grade_documents": "Generating answer...",
    "generate": "Verifying citations & groundedness...",
}


@st.cache_resource(show_spinner="Setting up PolicyGuard (this only happens once per restart)...")
def get_graph():
    store = PolicyVectorStore(PERSIST_DIR)

    _ids, documents, _metadatas = store.get_all_children()
    if not documents:
        ingest_policies(POLICIES_DIR, PERSIST_DIR)

    reranker = CohereReranker() if os.environ.get("COHERE_API_KEY") else None
    return build_graph(store, checkpointer=InMemorySaver(), reranker=reranker)


graph = get_graph()

st.set_page_config(page_title="PolicyGuard", page_icon="📄")
st.title("PolicyGuard")
st.caption("Ask a question about company HR/IT policy. Answers are grounded in retrieved policy excerpts with citations.")
if observability.tracing_enabled():
    st.caption("Questions, answers, and 👍/👎 ratings are logged to LangSmith to monitor answer quality.")

if st.sidebar.button("Clear Conversation"):
    st.session_state.history = []
    st.session_state.pending = None
    st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question": str, "response": dict}
if "pending" not in st.session_state:
    st.session_state.pending = None  # {"thread_id": str, "question": str, "response": dict} awaiting review


def render_response(response: dict) -> None:
    st.write(response.get("answer") or "*(no answer)*")

    citations = response.get("citations") or []
    if citations:
        st.caption("Sources: " + ", ".join(f"{c['doc_id']} · {c['section']}" for c in citations))

    invalid = response.get("invalid_citations") or []
    if invalid:
        st.warning(
            "Model cited sources not present in the retrieved context: "
            + ", ".join(f"{c['doc_id']} · {c['section']}" for c in invalid)
        )

    if response.get("status") == "cannot_answer":
        st.info("No relevant policy documents were found for this question.")
    if response.get("human_reviewed"):
        st.caption("Reviewed by a human before being returned.")

    feedback_widget(response.get("run_id"))


def feedback_widget(run_id) -> None:
    """Thumbs up/down under an answer, sent to LangSmith as feedback on that answer's trace."""
    if run_id is None or not observability.tracing_enabled():
        return
    key = f"feedback_{run_id}"

    def on_change():
        choice = st.session_state.get(key)
        if choice is not None:
            observability.record_user_feedback(run_id, thumbs_up=choice == 1)

    st.feedback("thumbs", key=key, on_change=on_change)


def _history_payload() -> list[dict]:
    turns = st.session_state.history[-MAX_HISTORY_TURNS:]
    return [{"question": t["question"], "answer": t["response"]["answer"]} for t in turns]


def _extract_interrupt_payload(item):
    if isinstance(item, dict):
        return item.get("value", item)
    return getattr(item, "value", item)


def _result_to_dict(result: dict, thread_id: str) -> dict:
    if result.get("__interrupt__"):
        payload = _extract_interrupt_payload(result["__interrupt__"][0])
        return {
            "thread_id": thread_id,
            "status": "needs_review",
            "answer": payload.get("draft_answer") or payload.get("answer") or "",
            "citations": [],
            "invalid_citations": payload.get("invalid_citations", []),
            "human_reviewed": False,
        }

    return {
        "thread_id": thread_id,
        "status": "answered" if result.get("grounded") else "cannot_answer",
        "answer": result.get("answer"),
        "citations": result.get("citations", []),
        "invalid_citations": result.get("invalid_citations", []),
        "human_reviewed": result.get("human_reviewed", False),
    }


def ask(question: str) -> None:
    thread_id = str(uuid.uuid4())
    token_queue: queue.Queue = queue.Queue()
    config = observability.run_config("ui", thread_id, token_queue=token_queue)
    init_state = initial_state(question, history=_history_payload())

    status_box = st.status("Analyzing question...", expanded=True)
    message_placeholder = st.empty()

    result_holder: dict = {}
    error_holder: dict = {}

    def run_pipeline():
        try:
            for update in graph.stream(init_state, config=config, stream_mode="updates"):
                node_name = list(update.keys())[0]
                token_queue.put({"type": "stage", "node": node_name})
            final_state = graph.get_state(config)
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

    full_text = ""
    while True:
        item = token_queue.get()
        if item is None:
            break
        if isinstance(item, str):
            full_text += item
            message_placeholder.markdown(full_text + "▌")
        elif isinstance(item, dict) and item.get("type") == "stage":
            label = _STAGE_LABELS.get(item["node"])
            if label:
                status_box.update(label=label, state="running")

    worker.join()

    if "error" in error_holder:
        st.error(f"PolicyGuard pipeline failed: {error_holder['error']}")
        status_box.update(label="Error", state="error")
        return

    status_box.update(label="Complete", state="complete", expanded=False)
    observability.record_outcome(config["run_id"], result_holder.get("res", {}))
    final_data = _result_to_dict(result_holder.get("res", {}), thread_id)
    final_data["run_id"] = config["run_id"]
    message_placeholder.markdown(final_data.get("answer") or full_text or "*(no answer)*")

    citations = final_data.get("citations") or []
    if citations:
        st.caption("Sources: " + ", ".join(f"{c['doc_id']} · {c['section']}" for c in citations))

    invalid = final_data.get("invalid_citations") or []
    if invalid:
        st.warning(
            "Model cited sources not present in the retrieved context: "
            + ", ".join(f"{c['doc_id']} · {c['section']}" for c in invalid)
        )

    if final_data.get("status") == "cannot_answer":
        st.info("No relevant policy documents were found for this question.")

    if final_data["status"] == "needs_review":
        st.session_state.pending = {"thread_id": final_data["thread_id"], "question": question, "response": final_data}
    else:
        st.session_state.history.append({"question": question, "response": final_data})


def resolve(action: str, answer: str | None = None) -> None:
    pending = st.session_state.pending
    decision = {"action": action}
    if answer is not None:
        decision["answer"] = answer

    # The resume is its own LangSmith trace (own run id), labelled with the same thread id.
    config = observability.run_config("ui", pending["thread_id"])
    try:
        result = graph.invoke(Command(resume=decision), config=config)
    except Exception as exc:
        st.error(f"Resolve failed: {exc}")
        return

    observability.record_outcome(config["run_id"], result)
    final_data = _result_to_dict(result, pending["thread_id"])
    final_data["run_id"] = config["run_id"]
    st.session_state.history.append({"question": pending["question"], "response": final_data})
    st.session_state.pending = None


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_response(turn["response"])

if st.session_state.pending:
    pending = st.session_state.pending
    with st.chat_message("user"):
        st.write(pending["question"])
    with st.chat_message("assistant"):
        st.write("Draft answer (unverified, pending human review):")
        st.write(pending["response"]["answer"])
        if pending["response"].get("invalid_citations"):
            st.warning(
                "Cited sources not present in the retrieved context: "
                + ", ".join(f"{c['doc_id']} · {c['section']}" for c in pending["response"]["invalid_citations"])
            )

        col1, col2, col3 = st.columns(3)
        if col1.button("Approve"):
            resolve("approve")
            st.rerun()
        if col3.button("Reject"):
            resolve("reject")
            st.rerun()

        edited = st.text_area("Edit answer instead", key="edit_answer")
        if col2.button("Submit edit"):
            if edited.strip():
                resolve("edit", answer=edited.strip())
                st.rerun()
            else:
                st.error("Enter a replacement answer before submitting.")
else:
    question = st.chat_input("Ask a policy question...")
    if question:
        ask(question)
        st.rerun()
