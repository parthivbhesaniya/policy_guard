"""CLI: resolve a PolicyGuard conversation paused for human review.

Usage:
    python -m policyguard.orchestration.resolve --thread-id <id> --action approve
    python -m policyguard.orchestration.resolve --thread-id <id> --action edit --answer "..."
    python -m policyguard.orchestration.resolve --thread-id <id> --action reject
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from policyguard.ingestion.vectorstore import PolicyVectorStore
from policyguard.orchestration.ask import DEFAULT_CHECKPOINT_DB, print_result
from policyguard.orchestration.graph import build_graph


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Resolve a PolicyGuard conversation paused for human review")
    parser.add_argument("--thread-id", type=str, required=True)
    parser.add_argument("--action", choices=["approve", "edit", "reject"], required=True)
    parser.add_argument("--answer", type=str, default=None, help="Replacement answer text, required for --action edit")
    parser.add_argument("--persist-dir", type=Path, default=Path("./chroma_db"), help="Chroma persistence directory")
    parser.add_argument(
        "--checkpoint-db", type=Path, default=DEFAULT_CHECKPOINT_DB, help="SQLite checkpoint database path"
    )
    args = parser.parse_args()

    if args.action == "edit" and not args.answer:
        parser.error("--action edit requires --answer")

    decision = {"action": args.action}
    if args.action == "edit":
        decision["answer"] = args.answer

    store = PolicyVectorStore(args.persist_dir)

    with SqliteSaver.from_conn_string(str(args.checkpoint_db)) as checkpointer:
        app = build_graph(store, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": args.thread_id}}

        state = app.get_state(config)
        if not state.next:
            parser.error(f"No pending review found for thread {args.thread_id!r}")

        result = app.invoke(Command(resume=decision), config=config)
        print_result(result, args.thread_id)


if __name__ == "__main__":
    main()
