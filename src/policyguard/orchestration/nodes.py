"""Nodes and routing functions for the LangGraph orchestrator.

Flow: rewrite_query -> retrieve -> grade_documents -> (cannot_answer | generate) ->
verify_answer -> (generate [retry] | escalate_to_human | end).

Parsing helpers (parse_relevant_indices, parse_grounded, _apply_human_decision) are split
out from the node functions that call the LLM / pause the graph so they can be
unit-tested without live API calls or a running graph.
"""

from __future__ import annotations

import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from policyguard.generation.chain import ContextBlock, build_context_blocks
from policyguard.generation.citations import Citation, parse_citations, validate_citations
from policyguard.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from policyguard.orchestration.state import GraphState
from policyguard.retrieval.hybrid import HybridRetriever
from policyguard.retrieval.reranker import CohereReranker

MAX_RETRIES = 2

# GraphState is checkpointed (serialized to SQLite), so it stores plain dicts rather than the
# ContextBlock/Citation dataclasses those checkpoints wouldn't otherwise recognize -- these
# helpers convert at the node boundary.


def _block_to_dict(block: ContextBlock) -> dict:
    return {"doc_id": block.doc_id, "section": block.section, "text": block.text}


def _block_from_dict(data: dict) -> ContextBlock:
    return ContextBlock(doc_id=data["doc_id"], section=data["section"], text=data["text"])


def _citation_to_dict(citation: Citation) -> dict:
    return {"doc_id": citation.doc_id, "section": citation.section}

REWRITE_PROMPT = """Rewrite the employee's latest question below into a clear, specific, \
standalone search query for retrieving relevant sections from the company's internal HR/IT \
documents.

Keep it short, and preserve any specific nouns, names, or entities already in the question --
do not add generic words like "policy" unless the question is already specifically about a
policy's rules (e.g. don't turn "what is the company name" into "company name policy"; that
changes what's being asked for). If the latest question is already clear and specific on its
own, return it unchanged.

If the latest question refers back to the conversation so far (e.g. "this", "that", "it", "the
same policy", "what about interns"), use the conversation to make the rewritten query stand on
its own -- e.g. a conversation about the annual leave policy followed by "does this apply to
interns" should become "does the annual leave policy apply to interns". Only use the
conversation to resolve such references; don't pull in unrelated details from earlier turns.

Return ONLY the rewritten query, with no explanation or quotation marks."""

GRADE_PROMPT_TEMPLATE = """You are grading whether policy excerpts are relevant to a question.

Question: {question}

Excerpts:
{excerpts}

Return a JSON array containing the 0-based indices of the excerpts that are relevant to \
answering the question. Return ONLY the JSON array, e.g. [0, 2]. If none are relevant, return [].
"""

VERIFY_PROMPT_TEMPLATE = """You are checking whether an answer is fully supported by the policy \
excerpts it was generated from (no unsupported claims).

Excerpts:
{excerpts}

Answer to check:
{answer}

Respond with the single word GROUNDED if every claim in the answer is supported by the excerpts. \
Respond with NOT_GROUNDED if the answer contains any claim not supported by the excerpts. An \
answer that makes no factual claims at all -- for example, one that says it cannot answer or \
doesn't have enough information -- is vacuously grounded: respond GROUNDED, since there is \
nothing left unsupported.

Respond with only that one word.
"""


def _invoke_text(llm: BaseChatModel, system: str, user: str) -> str:
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content if isinstance(response.content, str) else str(response.content)


def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    turns = "\n\n".join(f"Q: {turn['question']}\nA: {turn['answer']}" for turn in history)
    return f"Conversation so far:\n{turns}\n\n"


_GREETING_REGEX = re.compile(
    r"^(hi|hello|hey|heya|howdy|greetings|good\s+(morning|afternoon|evening)|thanks|thank\s+you|bye|goodbye|who\s+are\s+you|what\s+can\s+you\s+do)[\s!.,?]*$",
    re.IGNORECASE,
)

_POLICY_KEYWORDS = {
    "policy", "wfh", "leave", "vacation", "remote", "sick", "salary", "pay",
    "insurance", "notice", "probation", "work", "office", "laptop", "password",
    "it", "hr", "allowance", "expense", "benefit", "claim", "reimbursement"
}


def is_greeting(text: str) -> bool:
    cleaned = text.strip().lower()
    if _GREETING_REGEX.match(cleaned):
        return True
    words = set(re.findall(r"\b\w+\b", cleaned))
    if len(words) <= 3 and any(w in {"hi", "hello", "hey", "greetings", "thanks", "thank"} for w in words):
        if not words.intersection(_POLICY_KEYWORDS):
            return True
    return False


def route_after_rewrite(state: GraphState) -> str:
    if is_greeting(state["question"]) or is_greeting(state.get("rewritten_query", "")):
        return "handle_greeting"
    return "retrieve"


def handle_greeting(state: GraphState, config: RunnableConfig | None = None) -> dict:
    answer_text = (
        "Hello! I am PolicyGuard, your assistant for company HR and IT policy questions. "
        "How can I help you today?"
    )
    token_queue = (config or {}).get("configurable", {}).get("token_queue")
    if token_queue is not None:
        token_queue.put(answer_text)
    return {
        "answer": answer_text,
        "citations": [],
        "invalid_citations": [],
        "grounded": True,
    }


def rewrite_query(state: GraphState, llm: BaseChatModel) -> dict:
    history_block = _format_history(state.get("history", []))
    user_prompt = f"{history_block}Latest question: {state['question']}"
    rewritten = _invoke_text(llm, REWRITE_PROMPT, user_prompt).strip().strip('"')
    return {"rewritten_query": rewritten or state["question"]}


def retrieve(
    state: GraphState,
    retriever: HybridRetriever,
    reranker: CohereReranker | None = None,
    k: int = 4,
    fetch_k: int | None = None,
) -> dict:
    """Hybrid (dense + BM25) retrieval, optionally reranked down to `k` before grading."""
    fetch_k = fetch_k or max(k * 3, 10)
    candidates = retriever.retrieve(state["rewritten_query"], k=fetch_k)

    if reranker is not None and candidates:
        candidates = reranker.rerank(state["rewritten_query"], candidates, top_k=k)
    else:
        candidates = candidates[:k]

    return {"documents": [_block_to_dict(b) for b in build_context_blocks(candidates)]}


def parse_relevant_indices(raw_text: str, n: int) -> list[int]:
    """Fails open (keeps everything) if the grader's output can't be parsed as a JSON array."""
    match = re.search(r"\[[^\]]*\]", raw_text)
    if not match:
        return list(range(n))
    try:
        indices = json.loads(match.group(0))
    except json.JSONDecodeError:
        return list(range(n))
    return [i for i in indices if isinstance(i, int) and 0 <= i < n]


def grade_documents(state: GraphState, llm: BaseChatModel) -> dict:
    documents = [_block_from_dict(d) for d in state["documents"]]
    if not documents:
        return {"graded_documents": []}

    excerpts = "\n\n".join(f"[{i}] ({d.doc_id}, {d.section})\n{d.text}" for i, d in enumerate(documents))
    # Uses the disambiguated `rewritten_query`, not the raw `question` -- a follow-up like "does
    # this apply to interns" is meaningless to the grader on its own, even though the excerpts
    # retrieved for it are the right ones.
    prompt = GRADE_PROMPT_TEMPLATE.format(question=state["rewritten_query"], excerpts=excerpts)
    raw = _invoke_text(llm, "You are a strict but fair relevance grader.", prompt)
    indices = parse_relevant_indices(raw, len(documents))

    return {"graded_documents": [_block_to_dict(documents[i]) for i in indices]}


def route_after_grading(state: GraphState) -> str:
    return "generate" if state["graded_documents"] else "cannot_answer"


def cannot_answer(state: GraphState) -> dict:
    return {
        "answer": "I don't have enough information in the current policy documents to answer that.",
        "citations": [],
        "invalid_citations": [],
    }


from langchain_core.runnables import RunnableConfig


def generate(state: GraphState, llm: BaseChatModel, config: RunnableConfig | None = None) -> dict:
    context_blocks = [_block_from_dict(d) for d in state["graded_documents"]]
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(state["rewritten_query"], context_blocks)),
    ]

    token_queue = (config or {}).get("configurable", {}).get("token_queue")
    if token_queue is not None:
        chunks = []
        for chunk in llm.stream(messages):
            text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            chunks.append(text)
            token_queue.put(text)
        answer_text = "".join(chunks)
    else:
        answer_text = _invoke_text(llm, SYSTEM_PROMPT, build_user_prompt(state["rewritten_query"], context_blocks))

    available = {(b.doc_id, b.section) for b in context_blocks}
    valid, invalid = validate_citations(parse_citations(answer_text), available)

    return {
        "answer": answer_text,
        "citations": [_citation_to_dict(c) for c in valid],
        "invalid_citations": [_citation_to_dict(c) for c in invalid],
        "retry_count": state.get("retry_count", 0) + 1,
    }


def parse_grounded(raw_text: str) -> bool:
    """Fails open (assumes grounded) unless the verifier explicitly says otherwise."""
    return "NOT_GROUNDED" not in raw_text.upper()


def verify_answer(state: GraphState, llm: BaseChatModel) -> dict:
    context_blocks = [_block_from_dict(d) for d in state["graded_documents"]]
    excerpts = "\n\n".join(f"({b.doc_id}, {b.section})\n{b.text}" for b in context_blocks)
    prompt = VERIFY_PROMPT_TEMPLATE.format(excerpts=excerpts, answer=state["answer"])
    raw = _invoke_text(llm, "You are a strict faithfulness checker.", prompt)
    # A citation pointing at a (doc_id, section) that wasn't actually retrieved is itself a
    # faithfulness failure, even if the LLM-judged claim content happens to be accurate --
    # so it counts against groundedness the same way a failed verify_answer check does.
    grounded = parse_grounded(raw) and not state["invalid_citations"]

    return {
        "grounded": grounded,
        "hallucination_score": 1.0 if grounded else 0.0,
        "needs_human_review": (not grounded) and state.get("retry_count", 0) >= MAX_RETRIES,
    }


def route_after_verification(state: GraphState) -> str:
    if state["grounded"]:
        return "end"
    if state["needs_human_review"]:
        return "escalate_to_human"
    return "generate"


def _apply_human_decision(decision: dict) -> dict:
    action = decision.get("action") if isinstance(decision, dict) else None

    if action == "edit":
        return {"answer": decision["answer"], "needs_human_review": False, "human_reviewed": True}
    if action == "reject":
        return {
            "answer": "I don't have enough information in the current policy documents to answer that.",
            "citations": [],
            "invalid_citations": [],
            "needs_human_review": False,
            "human_reviewed": True,
        }
    # default / "approve": keep the unverified draft answer as-is, but mark it as human-approved.
    return {"needs_human_review": False, "human_reviewed": True}


def escalate_to_human(state: GraphState) -> dict:
    """Pauses the graph (via LangGraph's interrupt) until a human approves, edits, or rejects
    the draft answer. Requires the graph to be compiled with a checkpointer -- resume with
    ``app.invoke(Command(resume={"action": "approve" | "edit" | "reject", ...}), config)``.
    """
    decision = interrupt(
        {
            "question": state["question"],
            "draft_answer": state["answer"],
            "invalid_citations": state["invalid_citations"],
            "retry_count": state["retry_count"],
        }
    )
    return _apply_human_decision(decision)
