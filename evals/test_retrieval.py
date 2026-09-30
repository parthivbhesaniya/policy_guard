"""Retrieval component eval: Contextual Recall + Contextual Precision (DeepEval).

Runs PolicyGuard's `retrieve` node on its own -- hybrid dense+BM25 search, Cohere-reranked when
a key is set, deduped to parent chunks -- for every answerable golden question, and judges the
retrieved context against the golden `expected_answer`:

- Contextual Recall: can every statement in the expected answer be attributed to the retrieved
  context? (Did retrieval find everything needed?)
- Contextual Precision: are the chunks that support the expected answer ranked above the ones
  that don't? (Is the useful context at the top?)

The raw question is used as the query (no LLM query rewrite) so this measures retrieval alone.
Unanswerable golden questions are skipped: with no correct context, recall is meaningless.

    deepeval test run evals/test_retrieval.py
    deepeval test run evals/test_retrieval.py -k "office-dress-01 or tor-approval-01"   # a subset
    EVAL_INCLUDE_REASON=1 deepeval test run evals/test_retrieval.py                     # with judge reasons
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from policyguard.evaluation.dataset import GoldenExample, load_golden_dataset
from policyguard.orchestration import nodes

# Same number of context blocks the orchestrator hands to generation (build_graph's default).
K = 4

ANSWERABLE_EXAMPLES = [e for e in load_golden_dataset() if e.answerable]


@pytest.mark.parametrize("example", ANSWERABLE_EXAMPLES, ids=lambda e: e.id)
def test_retrieval(example: GoldenExample, retriever, reranker, judge, threshold, include_reason):
    documents = nodes.retrieve({"rewritten_query": example.question}, retriever, reranker, k=K)["documents"]

    test_case = LLMTestCase(
        name=example.id,
        input=example.question,
        expected_output=example.expected_answer,
        retrieval_context=[d["text"] for d in documents],
        additional_metadata={
            "expected_section": example.expected_section,
            "retrieved_sections": [d["section"] for d in documents],
        },
    )
    assert_test(
        test_case,
        [
            ContextualRecallMetric(threshold=threshold, model=judge, include_reason=include_reason),
            ContextualPrecisionMetric(threshold=threshold, model=judge, include_reason=include_reason),
        ],
    )
