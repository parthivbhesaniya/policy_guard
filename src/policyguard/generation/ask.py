"""CLI: ask a policy question and get a cited answer (stage 2 baseline chain).

Usage:
    python -m policyguard.generation.ask "how many carryover leave days am I allowed"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from policyguard.generation.chain import answer_question
from policyguard.ingestion.vectorstore import PolicyVectorStore


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="PolicyGuard question-answering CLI (stage 2 baseline)")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("--persist-dir", type=Path, default=Path("./chroma_db"), help="Chroma persistence directory")
    parser.add_argument("--k", type=int, default=4, help="Number of child chunks to retrieve")
    args = parser.parse_args()

    store = PolicyVectorStore(args.persist_dir)
    result = answer_question(args.question, store, k=args.k)

    print(result.answer)

    if result.invalid_citations:
        print("\n[!] Model cited sources not present in the retrieved context (possible hallucination):")
        for c in result.invalid_citations:
            print(f"    - {c.doc_id}, {c.section}")

    print(f"\nRetrieved {len(result.context_blocks)} source section(s):")
    for block in result.context_blocks:
        print(f"  - {block.doc_id} / {block.section}")


if __name__ == "__main__":
    main()
