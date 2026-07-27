from pathlib import Path

from policyguard.evaluation import evaluators
from policyguard.evaluation.dataset import DEFAULT_DATASET_PATH, GoldenExample, load_golden_dataset


class FakeLLM:
    def __init__(self, response: str):
        self._response = response

    def invoke(self, messages):
        class _Response:
            def __init__(self, content: str):
                self.content = content

        return _Response(self._response)


def _answerable_example() -> GoldenExample:
    return GoldenExample(
        id="ex-1",
        question="how many sick days do I get",
        answerable=True,
        expected_answer="10 paid sick days per year.",
        expected_doc_id="hr-leave-policy",
        expected_section="Sick Leave",
    )


def _unanswerable_example() -> GoldenExample:
    return GoldenExample(
        id="ex-2",
        question="what is the capital of France",
        answerable=False,
        expected_answer="I don't have enough information in the current policy documents to answer that.",
    )


# --- dataset loading -------------------------------------------------------------------------


def test_golden_dataset_loads_and_has_reasonable_size():
    examples = load_golden_dataset(DEFAULT_DATASET_PATH)
    assert 20 <= len(examples) <= 30


def test_golden_dataset_ids_are_unique():
    examples = load_golden_dataset(DEFAULT_DATASET_PATH)
    ids = [e.id for e in examples]
    assert len(ids) == len(set(ids))


def test_golden_dataset_covers_every_parent_section_at_least_twice():
    examples = load_golden_dataset(DEFAULT_DATASET_PATH)
    sections = [e.expected_section for e in examples if e.answerable]
    counts = {s: sections.count(s) for s in set(sections)}
    assert all(count >= 2 for count in counts.values())
    assert set(counts) == {
        "Annual Leave",
        "Sick Leave",
        "Parental Leave",
        "Password Requirements",
        "Device Security",
        "Access Requests",
    }


def test_golden_dataset_includes_unanswerable_negatives():
    examples = load_golden_dataset(DEFAULT_DATASET_PATH)
    unanswerable = [e for e in examples if not e.answerable]
    assert len(unanswerable) >= 4


# --- recall_at_k -----------------------------------------------------------------------------


def test_recall_at_k_hit():
    example = _answerable_example()
    retrieved = [("hr-leave-policy", "Sick Leave"), ("hr-leave-policy", "Annual Leave")]
    result = evaluators.recall_at_k(example, retrieved)
    assert result.score == 1.0


def test_recall_at_k_miss():
    example = _answerable_example()
    retrieved = [("hr-leave-policy", "Annual Leave")]
    result = evaluators.recall_at_k(example, retrieved)
    assert result.score == 0.0


def test_recall_at_k_not_applicable_for_unanswerable():
    example = _unanswerable_example()
    result = evaluators.recall_at_k(example, [])
    assert result.score == 1.0


# --- citation_accuracy -----------------------------------------------------------------------


def test_citation_accuracy_hit():
    example = _answerable_example()
    result = evaluators.citation_accuracy(
        example, cited_sections=[("hr-leave-policy", "Sick Leave")], invalid_citations=[]
    )
    assert result.score == 1.0


def test_citation_accuracy_wrong_section_cited():
    example = _answerable_example()
    result = evaluators.citation_accuracy(
        example, cited_sections=[("hr-leave-policy", "Annual Leave")], invalid_citations=[]
    )
    assert result.score == 0.0


def test_citation_accuracy_any_invalid_citation_fails_regardless_of_match():
    example = _answerable_example()
    result = evaluators.citation_accuracy(
        example,
        cited_sections=[("hr-leave-policy", "Sick Leave")],
        invalid_citations=[("hr-leave-policy", "Made Up Section")],
    )
    assert result.score == 0.0


def test_citation_accuracy_unanswerable_with_no_citations_passes():
    example = _unanswerable_example()
    result = evaluators.citation_accuracy(example, cited_sections=[], invalid_citations=[])
    assert result.score == 1.0


def test_citation_accuracy_unanswerable_with_a_citation_fails():
    example = _unanswerable_example()
    result = evaluators.citation_accuracy(
        example, cited_sections=[("hr-leave-policy", "Sick Leave")], invalid_citations=[]
    )
    assert result.score == 0.0


# --- LLM-judge metrics (faithfulness / answer_relevance) --------------------------------------


def test_faithfulness_yes_scores_one():
    result = evaluators.faithfulness(FakeLLM("YES, fully supported."), "context", "answer")
    assert result.score == 1.0


def test_faithfulness_no_scores_zero():
    result = evaluators.faithfulness(FakeLLM("NO, unsupported claim."), "context", "answer")
    assert result.score == 0.0


def test_answer_relevance_yes_scores_one():
    result = evaluators.answer_relevance(FakeLLM("YES"), "question", "answer")
    assert result.score == 1.0


def test_answer_relevance_no_scores_zero():
    result = evaluators.answer_relevance(FakeLLM("No, it doesn't address the question."), "question", "answer")
    assert result.score == 0.0
