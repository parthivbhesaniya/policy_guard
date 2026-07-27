"""The four evaluation metrics from PolicyGuard's evaluation layer:

- recall_at_k: did retrieval surface the section that actually contains the gold answer?
- citation_accuracy: are citations valid, and do they point at the expected section?
- faithfulness: does the answer only contain claims supported by the retrieved context? (LLM judge)
- answer_relevance: does the answer actually address the question asked? (LLM judge)

recall_at_k and citation_accuracy are pure/deterministic so they're unit-testable without an
LLM. faithfulness and answer_relevance take an `llm` param so tests can substitute a fake.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from policyguard.evaluation.dataset import GoldenExample

Section = tuple[str, str]  # (doc_id, section)


@dataclass(frozen=True)
class EvalResult:
    key: str
    score: float
    comment: str = ""


def recall_at_k(example: GoldenExample, retrieved_sections: list[Section]) -> EvalResult:
    if not example.answerable:
        return EvalResult(key="recall@k", score=1.0, comment="not applicable: unanswerable question")

    expected = (example.expected_doc_id, example.expected_section)
    hit = expected in retrieved_sections
    return EvalResult(key="recall@k", score=1.0 if hit else 0.0, comment=f"expected {expected} in {retrieved_sections}")


def citation_accuracy(
    example: GoldenExample,
    cited_sections: list[Section],
    invalid_citations: list[Section],
) -> EvalResult:
    if invalid_citations:
        return EvalResult(key="citation_accuracy", score=0.0, comment=f"invalid citations: {invalid_citations}")

    if not example.answerable:
        ok = not cited_sections
        return EvalResult(
            key="citation_accuracy", score=1.0 if ok else 0.0, comment="unanswerable question should carry no citations"
        )

    expected = (example.expected_doc_id, example.expected_section)
    hit = expected in cited_sections
    return EvalResult(
        key="citation_accuracy", score=1.0 if hit else 0.0, comment=f"expected {expected} in {cited_sections}"
    )


FAITHFULNESS_PROMPT = """Context excerpts:
{context}

Answer to check:
{answer}

Does the answer ONLY contain claims that are supported by the context excerpts above (no \
unsupported or invented claims)? An answer that makes no factual claims at all -- for example, \
one that says it cannot answer or doesn't have enough information -- is vacuously faithful: \
respond YES, since there is nothing left unsupported.

Respond with YES or NO as the first word, then a brief reason."""

RELEVANCE_PROMPT = """This is an internal assistant that is restricted to answering ONLY from a \
company's HR/IT policy documents, and must decline anything outside that scope -- even if the \
question has a well-known general-knowledge answer, or the model itself knows the answer.

Question: {question}

Answer: {answer}

Does the answer address the question appropriately given that scope? Correctly declining a \
question that isn't covered by the policy documents (including questions with a real-world \
answer the policies simply don't contain) DOES count as addressing the question. It does NOT \
count as addressing the question if the answer is generic, off-topic, or non-responsive to what \
was actually asked.

Respond with YES or NO as the first word, then a brief reason."""


def _judge(llm: BaseChatModel, system: str, user: str) -> tuple[bool, str]:
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    text = (response.content if isinstance(response.content, str) else str(response.content)).strip()
    return text.upper().startswith("YES"), text


def faithfulness(llm: BaseChatModel, context_text: str, answer: str) -> EvalResult:
    ok, reason = _judge(
        llm, "You are a strict faithfulness judge.", FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)
    )
    return EvalResult(key="faithfulness", score=1.0 if ok else 0.0, comment=reason)


def answer_relevance(llm: BaseChatModel, question: str, answer: str) -> EvalResult:
    ok, reason = _judge(
        llm, "You are a strict answer-relevance judge.", RELEVANCE_PROMPT.format(question=question, answer=answer)
    )
    return EvalResult(key="answer_relevance", score=1.0 if ok else 0.0, comment=reason)
