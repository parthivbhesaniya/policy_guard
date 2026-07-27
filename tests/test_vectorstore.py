from pathlib import Path

from policyguard.ingestion.chunker import chunk_document
from policyguard.ingestion.loader import PolicyDocument
from policyguard.ingestion.vectorstore import PolicyVectorStore

SAMPLE_BODY = """# Sample Policy

## Section One

Some content for section one.
"""


def _doc(doc_id: str) -> PolicyDocument:
    return PolicyDocument(
        doc_id=doc_id,
        department="HR",
        effective_date="2026-01-01",
        version="1.0",
        body=SAMPLE_BODY,
        source_path=Path("sample.md"),
    )


def test_delete_document_removes_only_that_docs_chunks(tmp_path):
    store = PolicyVectorStore(tmp_path / "chroma_db")

    for doc_id in ("doc-a", "doc-b"):
        parent_chunks, child_chunks = chunk_document(_doc(doc_id))
        store.add_chunks(parent_chunks, child_chunks)

    store.delete_document("doc-a")

    ids, _documents, metadatas = store.get_all_children()
    remaining_doc_ids = {m["doc_id"] for m in metadatas}

    assert remaining_doc_ids == {"doc-b"}
    assert all("doc-a" not in i for i in ids)


def test_delete_document_is_a_no_op_for_unknown_doc_id(tmp_path):
    store = PolicyVectorStore(tmp_path / "chroma_db")
    parent_chunks, child_chunks = chunk_document(_doc("doc-a"))
    store.add_chunks(parent_chunks, child_chunks)

    store.delete_document("does-not-exist")

    ids, _documents, _metadatas = store.get_all_children()
    assert len(ids) == len(child_chunks)
