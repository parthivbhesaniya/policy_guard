"""CLI: ask a policy question via the LangGraph orchestrator (grading, hallucination check,
and human-in-the-loop escalation with SQLite checkpointing).

Usage:
    # single-shot
    python -m policyguard.orchestration.ask "how many carryover leave days am I allowed"
    python -m policyguard.orchestration.ask "..." --thread-id my-thread

    # interactive: omit the question to start the pipeline once and ask repeatedly
    python -m policyguard.orchestration.ask

If the answer can't be verified as grounded after retries, the graph pauses and persists its
state under a thread id in --checkpoint-db. Resume it with:
    python -m policyguard.orchestration.resolve --thread-id <id> --action approve|edit|reject
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from policyguard.ingestion.vectorstore import PolicyVectorStore
from policyguard.orchestration.graph import build_graph, initial_state
from policyguard.retrieval.reranker import CohereReranker

DEFAULT_CHECKPOINT_DB = Path("./checkpoints.sqlite")

# How many prior Q&A turns to keep feeding into rewrite_query in interactive mode, so a long
# session's prompt doesn't grow unbounded.
MAX_HISTORY_TURNS = 6


def print_result(result: dict, thread_id: str) -> None:
    if "__interrupt__" in result and result["__interrupt__"]:
        raw_item = result["__interrupt__"][0]
        payload = raw_item.get("value", raw_item) if isinstance(raw_item, dict) else getattr(raw_item, "value", raw_item)
        print("Draft answer (unverified, pending human review):")
        print(f"  {payload.get('draft_answer', '')}")
        print("\n[!] Could not verify this answer as grounded after retries.")
        print(f"    thread id: {thread_id}")
        print(
            "    Resume with: python -m policyguard.orchestration.resolve "
            f"--thread-id {thread_id} --action approve|edit|reject"
        )
        return

    print(result["answer"])

    if result["invalid_citations"]:
        print("\n[!] Model cited sources not present in the retrieved context:")
        for c in result["invalid_citations"]:
            print(f"    - {c['doc_id']}, {c['section']}")

    if result.get("human_reviewed"):
        print("\n(Reviewed by a human before being returned.)")

    print(f"\n(thread id: {thread_id})")


def _run_interactive(app) -> None:
    print("PolicyGuard -- full agentic pipeline (hybrid retrieval + grading + hallucination check)")
    print("Ask a policy question. Follow-ups like 'does this apply to interns' can refer back to")
    print("earlier questions in this session. Type 'exit', 'quit', or Ctrl+D to stop.\n")

    history: list[dict] = []

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            return

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        result = app.invoke(initial_state(question, history=history), config=config)
        print()
        print_result(result, thread_id)
        print()

        # A paused/escalated turn isn't resolved within this loop, so its outcome is unknown --
        # leave it out of history rather than record an unverified draft as if it were final.
        if "__interrupt__" not in result:
            history = (history + [{"question": question, "answer": result["answer"]}])[-MAX_HISTORY_TURNS:]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="PolicyGuard question-answering CLI (LangGraph + HITL)")
    parser.add_argument(
        "question", type=str, nargs="?", default=None, help="Question to ask (omit to start interactive mode)"
    )
    parser.add_argument("--persist-dir", type=Path, default=Path("./chroma_db"), help="Chroma persistence directory")
    parser.add_argument(
        "--checkpoint-db", type=Path, default=DEFAULT_CHECKPOINT_DB, help="SQLite checkpoint database path"
    )
    parser.add_argument(
        "--thread-id", type=str, default=None, help="Conversation thread id (single-shot mode only; default: random)"
    )
    parser.add_argument("--k", type=int, default=4, help="Number of child chunks to retrieve")
    parser.add_argument(
        "--no-rerank", action="store_true", help="Skip Cohere reranking (use raw hybrid-fused candidates)"
    )
    args = parser.parse_args()

    store = PolicyVectorStore(args.persist_dir)
    reranker = None if args.no_rerank else CohereReranker()

    # Setup (Chroma connection, BM25 index, LLM, graph compilation) happens once here, whether
    # the run below is a single question or an interactive loop of many.
    with SqliteSaver.from_conn_string(str(args.checkpoint_db)) as checkpointer:
        app = build_graph(store, k=args.k, checkpointer=checkpointer, reranker=reranker)

        if args.question is None:
            _run_interactive(app)
            return

        thread_id = args.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        result = app.invoke(initial_state(args.question), config=config)
        print_result(result, thread_id)


if __name__ == "__main__":
    main()
