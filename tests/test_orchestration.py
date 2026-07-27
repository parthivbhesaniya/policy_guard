from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from policyguard.ingestion.chunker import chunk_document
from policyguard.ingestion.loader import PolicyDocument
from policyguard.ingestion.vectorstore import PolicyVectorStore
from policyguard.orchestration import nodes
from policyguard.orchestration.graph import build_graph, initial_state

SAMPLE_BODY = """# Sample Policy

## Annual Leave

Full-time employees accrue 21 days of annual leave per year.

### Carryover Rules

Employees may carry over up to 5 unused days into the next year.
"""


def _sample_doc() -> PolicyDocument:
    return PolicyDocument(
        doc_id="sample-policy",
        department="HR",
        effective_date="2026-01-01",
        version="1.0",
        body=SAMPLE_BODY,
        source_path=Path("sample.md"),
    )


class FakeLLM:
    """Returns canned responses in order, ignoring the actual prompt content."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0

    def invoke(self, messages):
        response = self._responses[self.call_count]
        self.call_count += 1

        class _Response:
            def __init__(self, content: str):
                self.content = content

        return _Response(response)


# --- pure parsing/routing helpers ---------------------------------------------------------


def test_parse_relevant_indices_valid_json():
    assert nodes.parse_relevant_indices("[0, 2]", n=3) == [0, 2]


def test_parse_relevant_indices_empty_array():
    assert nodes.parse_relevant_indices("[]", n=3) == []


def test_parse_relevant_indices_fails_open_on_garbage():
    assert nodes.parse_relevant_indices("sure, all of them are relevant!", n=3) == [0, 1, 2]


def test_parse_relevant_indices_drops_out_of_range():
    assert nodes.parse_relevant_indices("[0, 5, 1]", n=2) == [0, 1]


def test_parse_grounded_true_on_grounded():
    assert nodes.parse_grounded("GROUNDED") is True


def test_parse_grounded_false_on_not_grounded():
    assert nodes.parse_grounded("NOT_GROUNDED: cites a section that isn't in the excerpts") is False


def test_route_after_grading_no_documents_goes_to_cannot_answer():
    state = {"graded_documents": []}
    assert nodes.route_after_grading(state) == "cannot_answer"


def test_route_after_grading_with_documents_goes_to_generate():
    state = {"graded_documents": [{"doc_id": "d", "section": "s", "text": "t"}]}
    assert nodes.route_after_grading(state) == "generate"


def test_route_after_verification_grounded_ends():
    state = {"grounded": True, "needs_human_review": False}
    assert nodes.route_after_verification(state) == "end"


def test_route_after_verification_needs_review_escalates():
    state = {"grounded": False, "needs_human_review": True}
    assert nodes.route_after_verification(state) == "escalate_to_human"


def test_route_after_verification_retries_when_not_grounded_and_not_exhausted():
    state = {"grounded": False, "needs_human_review": False}
    assert nodes.route_after_verification(state) == "generate"


def test_cannot_answer_returns_standard_message():
    result = nodes.cannot_answer({})
    assert "don't have enough information" in result["answer"]
    assert result["citations"] == []


def test_apply_human_decision_approve_keeps_state_unchanged_besides_flags():
    result = nodes._apply_human_decision({"action": "approve"})
    assert result == {"needs_human_review": False, "human_reviewed": True}


def test_apply_human_decision_edit_replaces_answer():
    result = nodes._apply_human_decision({"action": "edit", "answer": "Custom answer."})
    assert result["answer"] == "Custom answer."
    assert result["needs_human_review"] is False
    assert result["human_reviewed"] is True


def test_apply_human_decision_reject_returns_cannot_answer_message():
    result = nodes._apply_human_decision({"action": "reject"})
    assert "don't have enough information" in result["answer"]
    assert result["citations"] == []
    assert result["human_reviewed"] is True


def test_apply_human_decision_unknown_action_defaults_to_approve():
    result = nodes._apply_human_decision({})
    assert result == {"needs_human_review": False, "human_reviewed": True}


# --- full graph, with a fake LLM standing in for the Groq API --------------------------------


def _build_test_store(tmp_path: Path) -> PolicyVectorStore:
    store = PolicyVectorStore(tmp_path / "chroma_db")
    parent_chunks, child_chunks = chunk_document(_sample_doc())
    store.add_chunks(parent_chunks, child_chunks)
    return store


def test_graph_happy_path_ends_grounded(tmp_path):
    store = _build_test_store(tmp_path)
    llm = FakeLLM(
        [
            "how much annual leave and carryover is allowed",  # rewrite_query
            "[0, 1]",  # grade_documents: keep everything
            "Employees carry over up to 5 days. [source: sample-policy, Annual Leave]",  # generate
            "GROUNDED",  # verify_answer
        ]
    )
    app = build_graph(store, llm=llm)

    result = app.invoke(initial_state("how many carryover days do I get"))

    assert result["grounded"] is True
    assert result["needs_human_review"] is False
    assert result["retry_count"] == 1
    assert "carry over" in result["answer"]
    assert result["invalid_citations"] == []


def test_graph_no_relevant_documents_routes_to_cannot_answer(tmp_path):
    store = _build_test_store(tmp_path)
    llm = FakeLLM(
        [
            "unrelated rewritten query",  # rewrite_query
            "[]",  # grade_documents: nothing relevant
        ]
    )
    app = build_graph(store, llm=llm)

    result = app.invoke(initial_state("what is the capital of France"))

    assert "don't have enough information" in result["answer"]
    assert result["graded_documents"] == []
    assert llm.call_count == 2  # generate/verify never invoked


def test_graph_retries_generation_after_failed_verification_then_succeeds(tmp_path):
    store = _build_test_store(tmp_path)
    llm = FakeLLM(
        [
            "rewritten query",  # rewrite_query
            "[0, 1]",  # grade_documents
            "an ungrounded first answer",  # generate (attempt 1)
            "NOT_GROUNDED",  # verify_answer (attempt 1)
            "a corrected, grounded answer [source: sample-policy, Annual Leave]",  # generate (attempt 2)
            "GROUNDED",  # verify_answer (attempt 2)
        ]
    )
    app = build_graph(store, llm=llm)

    result = app.invoke(initial_state("how many carryover days do I get"))

    assert result["retry_count"] == 2
    assert result["grounded"] is True
    assert result["needs_human_review"] is False
    assert "corrected" in result["answer"]


def test_graph_pauses_for_human_review_after_exhausting_retries(tmp_path):
    store = _build_test_store(tmp_path)
    llm = FakeLLM(
        [
            "rewritten query",  # rewrite_query
            "[0, 1]",  # grade_documents
            "answer attempt 1",  # generate
            "NOT_GROUNDED",  # verify
            "answer attempt 2",  # generate
            "NOT_GROUNDED",  # verify -- retry_count now at MAX_RETRIES
        ]
    )
    app = build_graph(store, llm=llm, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "review-thread"}}

    result = app.invoke(initial_state("how many carryover days do I get"), config=config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["retry_count"] == nodes.MAX_RETRIES
    assert payload["draft_answer"] == "answer attempt 2"

    # graph state itself isn't finalized yet -- it's paused mid-way through escalate_to_human
    state = app.get_state(config)
    assert state.next == ("escalate_to_human",)


def test_resume_approve_keeps_draft_answer(tmp_path):
    store = _build_test_store(tmp_path)
    llm = FakeLLM(
        [
            "rewritten query",
            "[0, 1]",
            "answer attempt 1",
            "NOT_GROUNDED",
            "answer attempt 2",
            "NOT_GROUNDED",
        ]
    )
    app = build_graph(store, llm=llm, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "approve-thread"}}
    app.invoke(initial_state("how many carryover days do I get"), config=config)

    result = app.invoke(Command(resume={"action": "approve"}), config=config)

    assert result["answer"] == "answer attempt 2"
    assert result["needs_human_review"] is False
    assert result["human_reviewed"] is True


def test_resume_edit_replaces_answer(tmp_path):
    store = _build_test_store(tmp_path)
    llm = FakeLLM(
        [
            "rewritten query",
            "[0, 1]",
            "answer attempt 1",
            "NOT_GROUNDED",
            "answer attempt 2",
            "NOT_GROUNDED",
        ]
    )
    app = build_graph(store, llm=llm, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "edit-thread"}}
    app.invoke(initial_state("how many carryover days do I get"), config=config)

    result = app.invoke(
        Command(resume={"action": "edit", "answer": "You may carry over up to 5 days."}), config=config
    )

    assert result["answer"] == "You may carry over up to 5 days."
    assert result["human_reviewed"] is True


def test_resume_reject_returns_cannot_answer_message(tmp_path):
    store = _build_test_store(tmp_path)
    llm = FakeLLM(
        [
            "rewritten query",
            "[0, 1]",
            "answer attempt 1",
            "NOT_GROUNDED",
            "answer attempt 2",
            "NOT_GROUNDED",
        ]
    )
    app = build_graph(store, llm=llm, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "reject-thread"}}
    app.invoke(initial_state("how many carryover days do I get"), config=config)

    result = app.invoke(Command(resume={"action": "reject"}), config=config)

    assert "don't have enough information" in result["answer"]
    assert result["human_reviewed"] is True
