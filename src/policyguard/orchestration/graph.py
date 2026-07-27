"""Builds the LangGraph orchestrator.

rewrite_query -> retrieve -> grade_documents -> (cannot_answer | generate) -> verify_answer
    -> (generate [retry, up to MAX_RETRIES] | escalate_to_human | end)

`retrieve` does hybrid retrieval (dense vector search + BM25 keyword search, fused via
reciprocal rank fusion) and, if a reranker is supplied, reranks the fused candidates down to
`k` before grading -- per the architecture doc's RAG-depth section. Adds corrective-RAG
document grading and a hallucination self-correction loop on top of the stage-2 linear chain
(policyguard.generation.chain). When retries are exhausted without a grounded answer,
`escalate_to_human` pauses the graph (LangGraph interrupt) until a human approves, edits, or
rejects the draft -- which requires compiling with a checkpointer so the paused state can be
persisted and resumed, possibly in a later process/session.
"""

from __future__ import annotations

from functools import partial

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from policyguard.generation.chain import default_llm
from policyguard.ingestion.vectorstore import PolicyVectorStore
from policyguard.orchestration import nodes
from policyguard.orchestration.state import GraphState
from policyguard.retrieval.hybrid import HybridRetriever
from policyguard.retrieval.reranker import CohereReranker


def build_graph(
    store: PolicyVectorStore,
    llm: BaseChatModel | None = None,
    k: int = 4,
    fetch_k: int | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    retriever: HybridRetriever | None = None,
    reranker: CohereReranker | None = None,
) -> CompiledStateGraph:
    llm = llm or default_llm()
    retriever = retriever or HybridRetriever(store)

    graph = StateGraph(GraphState)
    graph.add_node("rewrite_query", partial(nodes.rewrite_query, llm=llm))
    graph.add_node("retrieve", partial(nodes.retrieve, retriever=retriever, reranker=reranker, k=k, fetch_k=fetch_k))
    graph.add_node("grade_documents", partial(nodes.grade_documents, llm=llm))
    graph.add_node("generate", partial(nodes.generate, llm=llm))
    graph.add_node("verify_answer", partial(nodes.verify_answer, llm=llm))
    graph.add_node("cannot_answer", nodes.cannot_answer)
    graph.add_node("escalate_to_human", nodes.escalate_to_human)

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        nodes.route_after_grading,
        {"generate": "generate", "cannot_answer": "cannot_answer"},
    )
    graph.add_edge("generate", "verify_answer")
    graph.add_conditional_edges(
        "verify_answer",
        nodes.route_after_verification,
        {"generate": "generate", "end": END, "escalate_to_human": "escalate_to_human"},
    )
    graph.add_edge("cannot_answer", END)
    graph.add_edge("escalate_to_human", END)

    return graph.compile(checkpointer=checkpointer)


def initial_state(question: str) -> GraphState:
    return GraphState(
        question=question,
        rewritten_query="",
        documents=[],
        graded_documents=[],
        answer="",
        citations=[],
        invalid_citations=[],
        grounded=False,
        hallucination_score=0.0,
        retry_count=0,
        needs_human_review=False,
        human_reviewed=False,
    )
