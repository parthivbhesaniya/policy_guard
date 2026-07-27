from pathlib import Path

from policyguard.ingestion.chunker import chunk_document
from policyguard.ingestion.loader import PolicyDocument
from policyguard.ingestion.vectorstore import PolicyVectorStore
from policyguard.retrieval.bm25 import build_bm25_index
from policyguard.retrieval.hybrid import HybridRetriever
from policyguard.retrieval.reranker import CohereReranker

SAMPLE_BODY = """# Sample Policy

## Annual Leave

Full-time employees accrue 21 days of annual leave per year.

### Carryover Rules

Employees may carry over up to 5 unused days into the next year. The rare term \
xyloquartz appears only in this subsection, nowhere else in the policy set.

## Sick Leave

Employees receive 10 paid sick days per calendar year.
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


def _build_test_store(tmp_path: Path) -> PolicyVectorStore:
    store = PolicyVectorStore(tmp_path / "chroma_db")
    parent_chunks, child_chunks = chunk_document(_sample_doc())
    store.add_chunks(parent_chunks, child_chunks)
    return store


class FakeCohereClient:
    """Fakes cohere.ClientV2.rerank: always ranks documents in reverse order."""

    def __init__(self):
        self.last_call = None

    def rerank(self, *, model, query, documents, top_n):
        self.last_call = {"model": model, "query": query, "documents": documents, "top_n": top_n}

        class _Result:
            def __init__(self, index, relevance_score):
                self.index = index
                self.relevance_score = relevance_score

        class _Response:
            def __init__(self, results):
                self.results = results

        reversed_indices = list(range(len(documents)))[::-1][:top_n]
        return _Response([_Result(i, 1.0 - i * 0.1) for i in reversed_indices])


# --- BM25 --------------------------------------------------------------------------------


def test_bm25_index_finds_rare_keyword(tmp_path):
    store = _build_test_store(tmp_path)
    index = build_bm25_index(store)

    hits = index.query("xyloquartz", k=5)

    assert len(hits) == 1
    child_id, score = hits[0]
    assert "carryover-rules" in child_id
    assert score > 0


def test_bm25_index_no_match_returns_empty(tmp_path):
    store = _build_test_store(tmp_path)
    index = build_bm25_index(store)

    assert index.query("completely unrelated gibberish query zzz", k=5) == []


def test_bm25_index_empty_store_returns_empty(tmp_path):
    store = PolicyVectorStore(tmp_path / "empty_chroma_db")
    index = build_bm25_index(store)

    assert index.query("anything", k=5) == []


# --- HybridRetriever (RRF fusion) ---------------------------------------------------------


def test_hybrid_retriever_surfaces_bm25_only_hit(tmp_path):
    store = _build_test_store(tmp_path)
    retriever = HybridRetriever(store)

    matches = retriever.retrieve("what does xyloquartz mean", k=4)

    child_ids = [m["child_id"] for m in matches]
    assert any("carryover-rules" in cid for cid in child_ids)


def test_hybrid_retriever_returns_at_most_k_matches(tmp_path):
    store = _build_test_store(tmp_path)
    retriever = HybridRetriever(store)

    matches = retriever.retrieve("how many sick days do I get", k=2)

    assert len(matches) <= 2


def test_hybrid_retriever_matches_have_parent_context(tmp_path):
    store = _build_test_store(tmp_path)
    retriever = HybridRetriever(store)

    matches = retriever.retrieve("annual leave carryover", k=4)

    assert all(m["parent_text"] is not None for m in matches)
    assert all(m["parent_metadata"] is not None for m in matches)


# --- CohereReranker ------------------------------------------------------------------------


def test_reranker_reorders_and_truncates_to_top_k():
    fake_client = FakeCohereClient()
    reranker = CohereReranker(model="rerank-v3.5", client=fake_client)
    matches = [{"child_id": f"c{i}", "child_text": f"text {i}"} for i in range(4)]

    result = reranker.rerank("some query", matches, top_k=2)

    assert [m["child_id"] for m in result] == ["c3", "c2"]
    assert fake_client.last_call["top_n"] == 2
    assert fake_client.last_call["query"] == "some query"


def test_reranker_handles_empty_matches():
    reranker = CohereReranker(model="rerank-v3.5", client=FakeCohereClient())
    assert reranker.rerank("query", [], top_k=4) == []


def test_reranker_requires_api_key_when_no_client_given(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    try:
        CohereReranker()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "COHERE_API_KEY" in str(e)
