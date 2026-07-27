"""Stage 2: simple linear generate-with-citations chain (LangChain baseline).

retrieve (PolicyVectorStore) -> format context -> prompt -> Groq LLM -> parse + validate
citations. No query rewriting, document grading, or hallucination-check retry loop yet --
those land in stage 3 as a LangGraph StateGraph.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from policyguard.generation.citations import Citation, parse_citations, validate_citations
from policyguard.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from policyguard.ingestion.vectorstore import PolicyVectorStore

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


@dataclass
class ContextBlock:
    doc_id: str
    section: str
    text: str


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation]
    invalid_citations: list[Citation]
    context_blocks: list[ContextBlock] = field(default_factory=list)


def build_context_blocks(matches: list[dict]) -> list[ContextBlock]:
    """Dedupes retrieved child matches down to their parent sections, preserving rank order."""
    blocks: list[ContextBlock] = []
    seen_parent_ids: set[str] = set()

    for match in matches:
        parent_id = match["metadata"].get("parent_id")
        parent_meta = match.get("parent_metadata")
        text = match.get("parent_text")

        if parent_id is None or text is None or parent_meta is None or parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)

        blocks.append(ContextBlock(doc_id=parent_meta["doc_id"], section=parent_meta["section"], text=text))

    return blocks


def default_llm(model: str | None = None) -> BaseChatModel:
    from langchain_groq import ChatGroq

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Add it to a .env file or export it before running.")

    return ChatGroq(model=model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL), temperature=0)


def answer_question(
    question: str,
    store: PolicyVectorStore,
    llm: BaseChatModel | None = None,
    k: int = 4,
) -> GenerationResult:
    matches = store.query(question, k=k)
    context_blocks = build_context_blocks(matches)

    llm = llm or default_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(question, context_blocks)),
    ]
    response = llm.invoke(messages)
    answer_text = response.content if isinstance(response.content, str) else str(response.content)

    available = {(b.doc_id, b.section) for b in context_blocks}
    valid, invalid = validate_citations(parse_citations(answer_text), available)

    return GenerationResult(answer=answer_text, citations=valid, invalid_citations=invalid, context_blocks=context_blocks)
