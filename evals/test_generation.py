"""Generation component eval: Faithfulness + Answer Relevancy (DeepEval).

Runs PolicyGuard's `generate` node on its own for every answerable golden question, feeding it
the IDEAL context from data/eval/generation_golden.json (the parent chunk(s) holding the answer,
copied verbatim from Chroma) instead of whatever retrieval returns. Retrieval mistakes therefore
can't leak into these scores -- they measure the generator alone:

- Faithfulness: is every claim in the answer supported by the context? (No hallucination.)
- Answer Relevancy: do the answer's statements actually address the question?

The answer is judged exactly as users see it, inline [source: ...] citations included.
Unanswerable golden questions are skipped (there is no ideal context for them).

    deepeval test run evals/test_generation.py
    deepeval test run evals/test_generation.py -k "office-hours-01 or tor-approval-01"   # a subset
    EVAL_INCLUDE_REASON=1 deepeval test run evals/test_generation.py                     # with judge reasons

Known judge false negatives (baseline run 2026-09-29: 35/37 pass). Both answers were read and
are correct and fully grounded; the judge (gpt-oss-120b) is wrong, so these are expected to fail:
- maternity-leave-01, Answer Relevancy 0.33-0.67 (varies run to run): the judge calls the
  80-days-worked eligibility condition "irrelevant" to "how long", though the golden answer
  includes it.
- grievance-committee-01, Faithfulness 0.67: the judge reads "The PAO, NHSRC" (the PAO of NHSRC)
  as two people and marks "NHSRC is Chairperson" as unsupported.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from policyguard.evaluation.dataset import GoldenExample, load_golden_dataset
from policyguard.orchestration import nodes

IDEAL_CONTEXT_PATH = Path("data/eval/generation_golden.json")

IDEAL_CONTEXT = {item["id"]: item["context"] for item in json.loads(IDEAL_CONTEXT_PATH.read_text(encoding="utf-8"))}
ANSWERABLE_EXAMPLES = [e for e in load_golden_dataset() if e.answerable]


@pytest.mark.parametrize("example", ANSWERABLE_EXAMPLES, ids=lambda e: e.id)
def test_generation(example: GoldenExample, store, generator_llm, judge, threshold, include_reason):
    context = IDEAL_CONTEXT[example.id]

    # The ideal context must still be what's in Chroma; re-ingesting can re-chunk the policy.
    stored = store._parents.get(ids=[c["parent_id"] for c in context], include=["documents"])
    stored_text = dict(zip(stored["ids"], stored["documents"]))
    stale = [c["parent_id"] for c in context if stored_text.get(c["parent_id"]) != c["text"]]
    if stale:
        pytest.fail(f"{IDEAL_CONTEXT_PATH} is out of date with chroma_db for {stale}; rebuild it after re-ingesting.")

    documents = [{"doc_id": c["doc_id"], "section": c["section"], "text": c["text"]} for c in context]
    result = nodes.generate({"rewritten_query": example.question, "graded_documents": documents}, generator_llm)

    test_case = LLMTestCase(
        name=example.id,
        input=example.question,
        actual_output=result["answer"],
        expected_output=example.expected_answer,
        retrieval_context=[d["text"] for d in documents],
        additional_metadata={
            "context_sections": [d["section"] for d in documents],
            "invalid_citations": result["invalid_citations"],
        },
    )
    assert_test(
        test_case,
        [
            FaithfulnessMetric(threshold=threshold, model=judge, include_reason=include_reason),
            AnswerRelevancyMetric(threshold=threshold, model=judge, include_reason=include_reason),
        ],
    )
