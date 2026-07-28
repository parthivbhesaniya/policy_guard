"""Chroma-backed vector store for hierarchical policy chunks.

Two collections are used:
  - ``policyguard_children`` — retrieval targets (small chunks).
  - ``policyguard_parents`` — generation context, looked up by ``parent_id``
    after a child-chunk match.

Configured to use HuggingFace's BAAI/bge-small-en-v1.5 embedding model via API
(when HUGGINGFACE_API_KEY / HF_TOKEN / CHROMA_HUGGINGFACE_API_KEY is provided in .env),
falling back gracefully to Chroma's default embedding function until the key is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
import chromadb.utils.embedding_functions as ef

from policyguard.ingestion.chunker import Chunk

CHILDREN_COLLECTION = "policyguard_children"
PARENTS_COLLECTION = "policyguard_parents"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def get_embedding_function():
    api_key = (
        os.environ.get("HUGGINGFACE_API_KEY")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("CHROMA_HUGGINGFACE_API_KEY")
    )
    if api_key:
        return ef.HuggingFaceEmbeddingFunction(
            api_key=api_key,
            model_name=EMBEDDING_MODEL_NAME,
        )
    return ef.DefaultEmbeddingFunction()


class PolicyVectorStore:
    def __init__(self, persist_dir: Path, embedding_function=None):
        self._embedding_fn = embedding_function or get_embedding_function()
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._children = self._client.get_or_create_collection(
            CHILDREN_COLLECTION, embedding_function=self._embedding_fn
        )
        self._parents = self._client.get_or_create_collection(
            PARENTS_COLLECTION, embedding_function=self._embedding_fn
        )

    def add_chunks(self, parent_chunks: list[Chunk], child_chunks: list[Chunk]) -> None:
        if parent_chunks:
            self._parents.upsert(
                ids=[c.id for c in parent_chunks],
                documents=[c.text for c in parent_chunks],
                metadatas=[c.metadata for c in parent_chunks],
            )
        if child_chunks:
            self._children.upsert(
                ids=[c.id for c in child_chunks],
                documents=[c.text for c in child_chunks],
                metadatas=[c.metadata for c in child_chunks],
            )

    def query(self, query_text: str, k: int = 4, where: dict | None = None) -> list[dict]:
        result = self._children.query(
            query_texts=[query_text],
            n_results=k,
            where=where,
        )
        return self._build_matches(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])

    def get_children_by_ids(self, ids: list[str]) -> list[dict]:
        """Fetches specific child chunks (e.g. BM25 hits not already covered by a dense query)
        in the same match shape `query()` returns. `distance` is None -- Chroma's `.get()` has
        no notion of similarity to a query."""
        if not ids:
            return []
        result = self._children.get(ids=ids, include=["documents", "metadatas"])
        return self._build_matches(result["ids"], result["documents"], result["metadatas"], [None] * len(result["ids"]))

    def get_all_children(self) -> tuple[list[str], list[str], list[dict]]:
        """Returns (ids, documents, metadatas) for every child chunk -- used to build a BM25 index."""
        result = self._children.get(include=["documents", "metadatas"])
        return result["ids"], result["documents"], result["metadatas"]

    def delete_document(self, doc_id: str) -> None:
        """Removes every parent and child chunk belonging to `doc_id`. Ingestion only ever
        upserts, so this is needed before re-ingesting a renamed/replaced document (or dropping
        one entirely) -- otherwise its old chunks linger in the store forever under a doc_id
        nothing points to anymore."""
        self._parents.delete(where={"doc_id": doc_id})
        self._children.delete(where={"doc_id": doc_id})

    def _build_matches(self, ids: list[str], documents: list[str], metadatas: list[dict], distances: list) -> list[dict]:
        parent_ids = [m.get("parent_id") for m in metadatas if m.get("parent_id")]
        parents_by_id: dict[str, str] = {}
        parent_metadata_by_id: dict[str, dict] = {}
        if parent_ids:
            parent_result = self._parents.get(ids=list(set(parent_ids)), include=["documents", "metadatas"])
            parents_by_id = dict(zip(parent_result["ids"], parent_result["documents"]))
            parent_metadata_by_id = dict(zip(parent_result["ids"], parent_result["metadatas"]))

        matches = []
        for child_id, child_text, metadata, distance in zip(ids, documents, metadatas, distances):
            parent_id = metadata.get("parent_id")
            matches.append(
                {
                    "child_id": child_id,
                    "child_text": child_text,
                    "metadata": metadata,
                    "distance": distance,
                    "parent_text": parents_by_id.get(parent_id),
                    # Parent's own metadata (e.g. its section title), distinct from the
                    # child's `metadata["section"]` which names the child's subsection.
                    "parent_metadata": parent_metadata_by_id.get(parent_id),
                }
            )
        return matches
