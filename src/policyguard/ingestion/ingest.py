"""CLI: ingest policy documents into the vector store, or run a test query.

Usage:
    python -m policyguard.ingestion.ingest --input data/policies --persist-dir ./chroma_db
    python -m policyguard.ingestion.ingest --query "how many sick days do I get" --persist-dir ./chroma_db
"""

from __future__ import annotations

import argparse
from pathlib import Path

from policyguard.ingestion.chunker import chunk_document
from policyguard.ingestion.loader import load_policy_documents
from policyguard.ingestion.vectorstore import PolicyVectorStore


def ingest(input_dir: Path, persist_dir: Path) -> None:
    documents = load_policy_documents(input_dir)
    if not documents:
        print(f"No .md files found in {input_dir}")
        return

    store = PolicyVectorStore(persist_dir)
    total_parents = total_children = 0

    for doc in documents:
        parent_chunks, child_chunks = chunk_document(doc)
        store.add_chunks(parent_chunks, child_chunks)
        total_parents += len(parent_chunks)
        total_children += len(child_chunks)
        print(f"  {doc.source_path.name}: {len(parent_chunks)} parent chunks, {len(child_chunks)} child chunks")

    print(f"Ingested {len(documents)} document(s): {total_parents} parent chunks, {total_children} child chunks")
    print(f"Persisted to {persist_dir}")


def query(query_text: str, persist_dir: Path, k: int) -> None:
    store = PolicyVectorStore(persist_dir)
    matches = store.query(query_text, k=k)

    if not matches:
        print("No matches found. Have you run ingestion yet?")
        return

    for i, match in enumerate(matches, start=1):
        meta = match["metadata"]
        print(f"\n[{i}] {meta['doc_id']} / {meta['section']}  (distance={match['distance']:.4f})")
        print(f"    child: {match['child_text'][:200].strip()}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="PolicyGuard ingestion pipeline")
    parser.add_argument("--input", type=Path, default=Path("data/policies"), help="Directory of .md policy docs")
    parser.add_argument("--persist-dir", type=Path, default=Path("./chroma_db"), help="Chroma persistence directory")
    parser.add_argument("--query", type=str, default=None, help="If set, run a retrieval query instead of ingesting")
    parser.add_argument("--k", type=int, default=4, help="Number of results to return for --query")
    args = parser.parse_args()

    if args.query:
        query(args.query, args.persist_dir, args.k)
    else:
        ingest(args.input, args.persist_dir)


if __name__ == "__main__":
    main()
