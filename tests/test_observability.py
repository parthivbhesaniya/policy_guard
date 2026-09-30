import uuid

import pytest

from policyguard import observability


class FakeFeedbackClient:
    """Stands in for langsmith.Client: records create_feedback calls instead of sending them."""

    PROJECT_ID = "11111111-1111-1111-1111-111111111111"

    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self._fail = fail

    def read_project(self, project_name):
        class _Project:
            id = FakeFeedbackClient.PROJECT_ID

        return _Project()

    def create_feedback(self, run_id, **kwargs):
        if self._fail:
            raise ConnectionError("LangSmith unreachable")
        self.calls.append({"run_id": run_id, **kwargs})


@pytest.fixture
def tracing_on(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")


@pytest.fixture
def tracing_off(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)


def _state(**overrides) -> dict:
    state = {
        "question": "how many casual leaves do I get",
        "rewritten_query": "casual leave entitlement",
        "documents": [{"doc_id": "d", "section": "Part 33", "text": "..."}],
        "graded_documents": [{"doc_id": "d", "section": "Part 33", "text": "..."}],
        "invalid_citations": [],
        "retry_count": 1,
        "human_reviewed": False,
    }
    state.update(overrides)
    return state


# --- tracing switch / labels ---------------------------------------------------------------


def test_tracing_enabled_needs_both_flag_and_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert observability.tracing_enabled() is False

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    assert observability.tracing_enabled() is True

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert observability.tracing_enabled() is False


def test_run_config_labels_the_trace_and_keeps_configurable(monkeypatch):
    monkeypatch.setenv("POLICYGUARD_ENV", "demo")
    queue_marker = object()

    config = observability.run_config("ui", "thread-1", token_queue=queue_marker)

    assert config["configurable"] == {"thread_id": "thread-1", "token_queue": queue_marker}
    assert isinstance(config["run_id"], uuid.UUID)
    assert config["tags"] == ["entrypoint:ui", "env:demo"]
    assert config["metadata"] == {"entrypoint": "ui", "environment": "demo", "thread_id": "thread-1"}


def test_run_config_gives_every_run_its_own_id_and_defaults_env_to_local(monkeypatch):
    monkeypatch.delenv("POLICYGUARD_ENV", raising=False)
    first = observability.run_config("api", "t")
    second = observability.run_config("api", "t")

    assert first["run_id"] != second["run_id"]
    assert first["metadata"]["environment"] == "local"


# --- outcome classification ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"__interrupt__": [{"value": {"draft_answer": "..."}}]}, "escalated"),
        (_state(human_reviewed=True), "human_reviewed"),
        (_state(question="hello", rewritten_query="hello", documents=[], graded_documents=[], retry_count=0), "greeting"),
        (
            _state(question="what is the capital of France", rewritten_query="what is the capital of France", documents=[], graded_documents=[], retry_count=0),
            "out_of_scope",
        ),
        (_state(question="policy", rewritten_query="policy", documents=[], graded_documents=[], retry_count=0), "clarification"),
        (_state(graded_documents=[], retry_count=0), "cannot_answer"),
        (_state(documents=[], graded_documents=[], retry_count=0), "cannot_answer"),
        (_state(), "answered"),
    ],
)
def test_outcome_of(state, expected):
    assert observability.outcome_of(state) == expected


def test_outcome_feedback_counts_retries_and_invalid_citations_for_generated_answers():
    state = _state(retry_count=2, invalid_citations=[{"doc_id": "d", "section": "Part 99"}])

    assert observability.outcome_feedback(state) == [
        {"key": "outcome", "value": "answered"},
        {"key": "retries", "score": 1},
        {"key": "invalid_citations", "score": 1},
    ]


def test_outcome_feedback_omits_generation_counts_when_nothing_was_generated():
    state = _state(question="hi", rewritten_query="hi", documents=[], graded_documents=[], retry_count=0)

    assert observability.outcome_feedback(state) == [{"key": "outcome", "value": "greeting"}]


# --- sending feedback ----------------------------------------------------------------------


def test_record_outcome_sends_feedback_attached_to_the_trace(tracing_on):
    client = FakeFeedbackClient()
    run_id = uuid.uuid4()

    observability.record_outcome(run_id, _state(), client=client, background=False)

    assert [c["key"] for c in client.calls] == ["outcome", "retries", "invalid_citations"]
    assert all(c["run_id"] == run_id and c["trace_id"] == run_id for c in client.calls)
    assert all(c["session_id"] == FakeFeedbackClient.PROJECT_ID for c in client.calls)


def test_record_user_feedback_scores_thumbs(tracing_on):
    client = FakeFeedbackClient()
    run_id = uuid.uuid4()

    observability.record_user_feedback(run_id, thumbs_up=True, client=client, background=False)
    observability.record_user_feedback(run_id, thumbs_up=False, client=client, background=False)

    assert [(c["key"], c["score"]) for c in client.calls] == [("user_score", 1), ("user_score", 0)]


def test_nothing_is_sent_when_tracing_is_off(tracing_off):
    client = FakeFeedbackClient()

    observability.record_outcome(uuid.uuid4(), _state(), client=client, background=False)
    observability.record_user_feedback(uuid.uuid4(), thumbs_up=True, client=client, background=False)

    assert client.calls == []


def test_langsmith_errors_are_swallowed_not_raised(tracing_on):
    observability.record_outcome(uuid.uuid4(), _state(), client=FakeFeedbackClient(fail=True), background=False)
