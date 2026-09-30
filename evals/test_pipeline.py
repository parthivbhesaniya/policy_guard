"""End-to-end pipeline eval: the RAG triad (DeepEval).

Runs the full PolicyGuard LangGraph app -- LLM query rewrite, hybrid retrieval + Cohere rerank,
document grading, generation, and the verify/retry loop -- exactly as the UI does, for every
answerable golden question, then judges:

- Answer Relevancy: does the final answer address the question?
- Contextual Relevancy: is the context the generator was given (the graded documents) relevant
  to the question?
- Faithfulness: is every claim in the final answer supported by that context?

Unanswerable golden questions are skipped. An answerable question the pipeline refuses to answer
fails outright (there is no context left to judge). If the verify loop gives up and escalates to
a human, the unverified draft is judged and `escalated` is recorded in the test case metadata.

On the Groq free tier the judge's daily token quota (TPD) can run out mid-run. When it does, that
test fails as "not scored" and every later test is skipped without running the pipeline, so no
generator tokens are wasted; re-run the skipped ones with -k once the quota refills.

    deepeval test run evals/test_pipeline.py
    deepeval test run evals/test_pipeline.py -k "office-hours-01 or tor-approval-01"   # a subset
    EVAL_INCLUDE_REASON=1 deepeval test run evals/test_pipeline.py                     # with judge reasons

Known limitation (kept at the 0.7 pass mark on purpose): Contextual Relevancy fails on most
questions (0.10-0.62 in the first run) even when the answer is perfect. It scores the share of
context statements relevant to the question, and each ~2,000-character parent chunk spans several
policy topics, so most of its statements are off-question. It reflects chunk size, not answer
quality; Answer Relevancy and Faithfulness are the signals to watch.
"""

from __future__ import annotations

import uuid

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from groq import RateLimitError
from langgraph.checkpoint.memory import InMemorySaver

from policyguard.evaluation.dataset import GoldenExample, load_golden_dataset
from policyguard.orchestration.graph import build_graph, initial_state

# Same number of context blocks the app hands to generation (build_graph's default).
K = 4

ANSWERABLE_EXAMPLES = [e for e in load_golden_dataset() if e.answerable]

_quota = {"exhausted": False}


@pytest.fixture(scope="module")
def app(store, retriever, reranker, generator_llm):
    # A checkpointer is required for escalate_to_human's interrupt; in-memory is enough here.
    return build_graph(store, llm=generator_llm, k=K, checkpointer=InMemorySaver(), retriever=retriever, reranker=reranker)


@pytest.mark.parametrize("example", ANSWERABLE_EXAMPLES, ids=lambda e: e.id)
def test_pipeline(example: GoldenExample, app, judge, threshold, include_reason):
    if _quota["exhausted"]:
        pytest.skip("Groq daily token quota exhausted earlier in this run -- not scored")

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = app.invoke(initial_state(example.question), config=config)

    escalated = bool(result.get("__interrupt__"))
    answer = result["__interrupt__"][0].value["draft_answer"] if escalated else result["answer"]
    context = result["graded_documents"]
    if not context:
        pytest.fail(f"Pipeline refused to answer an answerable question: {answer!r}")

    test_case = LLMTestCase(
        name=example.id,
        input=example.question,
        actual_output=answer,
        expected_output=example.expected_answer,
        retrieval_context=[d["text"] for d in context],
        additional_metadata={
            "rewritten_query": result["rewritten_query"],
            "expected_section": example.expected_section,
            "retrieved_sections": [d["section"] for d in result["documents"]],
            "graded_sections": [d["section"] for d in context],
            "generation_attempts": result["retry_count"],
            "escalated": escalated,
            "invalid_citations": result["invalid_citations"],
        },
    )
    try:
        assert_test(
            test_case,
            [
                AnswerRelevancyMetric(threshold=threshold, model=judge, include_reason=include_reason),
                ContextualRelevancyMetric(threshold=threshold, model=judge, include_reason=include_reason),
                FaithfulnessMetric(threshold=threshold, model=judge, include_reason=include_reason),
            ],
        )
    except RateLimitError as err:
        if "tokens per day" not in str(err):
            raise
        _quota["exhausted"] = True
        pytest.fail("Not scored: Groq daily token quota (TPD) exhausted", pytrace=False)
