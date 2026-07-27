"""BM25 keyword index over PolicyGuard's child chunks, for hybrid retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from policyguard.ingestion.vectorstore import PolicyVectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Bm25Index:
    ids: list[str]
    _bm25: BM25Okapi | None

    def query(self, text: str, k: int) -> list[tuple[str, float]]:
        """Returns up to k (child_id, score) pairs ranked by BM25 score, highest first,
        excluding zero-score (no keyword overlap at all) matches."""
        if not self.ids or self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokenize(text))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.ids[i], scores[i]) for i in ranked_indices[:k] if scores[i] > 0]


def build_bm25_index(store: PolicyVectorStore) -> Bm25Index:
    ids, documents, _metadatas = store.get_all_children()
    if not documents:
        return Bm25Index(ids=[], _bm25=None)

    tokenized_documents = [tokenize(doc) for doc in documents]
    return Bm25Index(ids=ids, _bm25=BM25Okapi(tokenized_documents))
