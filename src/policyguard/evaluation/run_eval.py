"""CLI: run PolicyGuard's evaluation suite against the golden dataset.

Local mode (default) needs only GROQ_API_KEY -- it runs the LangGraph orchestrator against
every golden example, computes all four metrics, and prints an aggregate scorecard. This is
the "runs as a script/CI step" regression-testing baseline from the architecture doc.

    python -m policyguard.evaluation.run_eval

--langsmith additionally uploads the dataset to LangSmith and logs the run as a tracked,
traced experiment (requires LANGSMITH_API_KEY):

    python -m policyguard.evaluation.run_eval --langsmith
"""

from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path

from dotenv import load_dotenv

from policyguard.evaluation import evaluators
from policyguard.evaluation.dataset import DEFAULT_DATASET_PATH, GoldenExample, load_golden_dataset
from policyguard.generation.chain import default_llm
from policyguard.ingestion.vectorstore import PolicyVectorStore
from policyguard.orchestration.graph import build_graph, initial_state
from policyguard.retrieval.reranker import CohereReranker

LANGSMITH_DATASET_NAME = "policyguard-golden-qa"


def _run_example(app, judge_llm, example: GoldenExample) -> dict:
    result = app.invoke(initial_state(example.question))

    retrieved_sections = [(d["doc_id"], d["section"]) for d in result["documents"]]
    cited_sections = [(c["doc_id"], c["section"]) for c in result["citations"]]
    invalid_sections = [(c["doc_id"], c["section"]) for c in result["invalid_citations"]]
    context_text = "\n\n".join(d["text"] for d in result["graded_documents"])

    metrics = [
        evaluators.recall_at_k(example, retrieved_sections),
        evaluators.citation_accuracy(example, cited_sections, invalid_sections),
        evaluators.faithfulness(judge_llm, context_text, result["answer"]),
        evaluators.answer_relevance(judge_llm, example.question, result["answer"]),
    ]
    return {"example": example, "answer": result["answer"], "metrics": metrics}


def _print_scorecard(rows: list[dict]) -> None:
    by_key: dict[str, list[float]] = {}
    for row in rows:
        for metric in row["metrics"]:
            by_key.setdefault(metric.key, []).append(metric.score)

    print(f"\n{'metric':<20}{'mean score':<12}n")
    for key, scores in by_key.items():
        print(f"{key:<20}{statistics.mean(scores):<12.2f}{len(scores)}")

    failures = [
        (row["example"].id, metric.key, metric.comment)
        for row in rows
        for metric in row["metrics"]
        if metric.score < 1.0
    ]
    if failures:
        print(f"\n{len(failures)} metric failure(s):")
        for example_id, key, comment in failures:
            print(f"  - [{example_id}] {key}: {comment}")
    else:
        print("\nAll metrics passed on every example.")


def _run_local(examples: list[GoldenExample], app, judge_llm) -> None:
    rows = [_run_example(app, judge_llm, example) for example in examples]
    _print_scorecard(rows)


def _run_langsmith(examples: list[GoldenExample], app, judge_llm) -> None:
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY is not set. Add it to a .env file or export it before running.")

    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client()

    if not client.has_dataset(dataset_name=LANGSMITH_DATASET_NAME):
        dataset = client.create_dataset(
            LANGSMITH_DATASET_NAME, description="PolicyGuard golden HR/IT policy Q&A set"
        )
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"question": ex.question},
                    "outputs": {
                        "expected_answer": ex.expected_answer,
                        "expected_doc_id": ex.expected_doc_id,
                        "expected_section": ex.expected_section,
                        "answerable": ex.answerable,
                    },
                    "metadata": {"id": ex.id},
                }
                for ex in examples
            ],
        )

    def target(inputs: dict) -> dict:
        result = app.invoke(initial_state(inputs["question"]))
        return {
            "answer": result["answer"],
            "documents": result["documents"],
            "graded_documents": result["graded_documents"],
            "citations": result["citations"],
            "invalid_citations": result["invalid_citations"],
        }

    def _reference_example(reference_outputs: dict) -> GoldenExample:
        return GoldenExample(
            id="",
            question="",
            answerable=reference_outputs["answerable"],
            expected_answer=reference_outputs.get("expected_answer", ""),
            expected_doc_id=reference_outputs.get("expected_doc_id"),
            expected_section=reference_outputs.get("expected_section"),
        )

    def recall_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        example = _reference_example(reference_outputs)
        retrieved = [(d["doc_id"], d["section"]) for d in outputs["documents"]]
        result = evaluators.recall_at_k(example, retrieved)
        return {"key": result.key, "score": result.score, "comment": result.comment}

    def citation_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        example = _reference_example(reference_outputs)
        cited = [(c["doc_id"], c["section"]) for c in outputs["citations"]]
        invalid = [(c["doc_id"], c["section"]) for c in outputs["invalid_citations"]]
        result = evaluators.citation_accuracy(example, cited, invalid)
        return {"key": result.key, "score": result.score, "comment": result.comment}

    def faithfulness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        context_text = "\n\n".join(d["text"] for d in outputs["graded_documents"])
        result = evaluators.faithfulness(judge_llm, context_text, outputs["answer"])
        return {"key": result.key, "score": result.score}

    def relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        result = evaluators.answer_relevance(judge_llm, inputs["question"], outputs["answer"])
        return {"key": result.key, "score": result.score}

    results = evaluate(
        target,
        data=LANGSMITH_DATASET_NAME,
        evaluators=[recall_evaluator, citation_evaluator, faithfulness_evaluator, relevance_evaluator],
        experiment_prefix="policyguard-baseline",
    )
    print(f"Experiment logged to LangSmith: {results.experiment_name}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run PolicyGuard's evaluation suite against the golden dataset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--persist-dir", type=Path, default=Path("./chroma_db"), help="Chroma persistence directory")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--langsmith", action="store_true", help="Also upload the dataset and log a tracked LangSmith experiment"
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="Skip Cohere reranking (use raw hybrid-fused candidates)"
    )
    args = parser.parse_args()

    examples = load_golden_dataset(args.dataset)
    store = PolicyVectorStore(args.persist_dir)
    llm = default_llm()
    reranker = None if args.no_rerank else CohereReranker()
    app = build_graph(store, llm=llm, k=args.k, reranker=reranker)

    if args.langsmith:
        _run_langsmith(examples, app, llm)
    else:
        _run_local(examples, app, llm)


if __name__ == "__main__":
    main()
