import json

from fastapi.testclient import TestClient

from policyguard.api.app import app


def test_ask_stream_endpoint(monkeypatch):
    with TestClient(app) as client:

        def fake_stream(state, config=None, stream_mode=None):
            token_q = (config or {}).get("configurable", {}).get("token_queue")
            if token_q:
                token_q.put("This ")
                token_q.put("is ")
                token_q.put("a ")
                token_q.put("streamed ")
                token_q.put("answer.")
            yield {"generate": {}}

        monkeypatch.setattr(app.state.graph, "stream", fake_stream)

        class FakeState:
            next = None
            values = {
                "grounded": True,
                "answer": "This is a streamed answer.",
                "citations": [],
                "invalid_citations": [],
            }

        monkeypatch.setattr(app.state.graph, "get_state", lambda cfg: FakeState())

        response = client.post("/ask/stream", json={"question": "What is the policy?"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = response.text.strip().split("\n\n")
        events = [json.loads(line.replace("data: ", "")) for line in lines if line.startswith("data: ")]

        event_types = [e["type"] for e in events]
        assert "status" in event_types
        assert "token" in event_types
        assert "final" in event_types

        tokens = "".join([e["content"] for e in events if e["type"] == "token"])
        assert tokens == "This is a streamed answer."
