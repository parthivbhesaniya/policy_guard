"""Hybrid retrieval: dense vector search (Chroma) + BM25 keyword search, combined via
Reciprocal Rank Fusion (RRF), per the architecture doc's `retrieve` node design.
"""

from __future__ import annotations

from policyguard.ingestion.vectorstore import PolicyVectorStore
from policyguard.retrieval.bm25 import Bm25Index, build_bm25_index

RRF_K = 60


class HybridRetriever:
    def __init__(self, store: PolicyVectorStore, bm25_index: Bm25Index | None = None):
        self._store = store
        self._bm25 = bm25_index or build_bm25_index(store)

    def retrieve(self, query_text: str, k: int, fetch_k: int | None = None) -> list[dict]:
        """Returns up to `k` fused matches, in the same shape as PolicyVectorStore.query()."""
        fetch_k = fetch_k or max(k * 3, 10)

        dense_matches = self._store.query(query_text, k=fetch_k)
        dense_rank = {m["child_id"]: rank for rank, m in enumerate(dense_matches)}
        matches_by_id = {m["child_id"]: m for m in dense_matches}

        bm25_hits = self._bm25.query(query_text, k=fetch_k)
        bm25_rank = {child_id: rank for rank, (child_id, _score) in enumerate(bm25_hits)}

        all_ids = set(dense_rank) | set(bm25_rank)

        # BM25-only hits (dense search didn't surface them) still need their child/parent
        # text and metadata fetched before they can be used as generation context.
        missing_ids = [i for i in all_ids if i not in matches_by_id]
        for match in self._store.get_children_by_ids(missing_ids):
            matches_by_id[match["child_id"]] = match

        def rrf_score(child_id: str) -> float:
            # A candidate missing from one ranked list (e.g. the single best dense match, which
            # BM25 never found at all because it shares no literal keywords with the query) gets
            # a bounded penalty -- treated as tied for the worst rank actually observed in that
            # list -- rather than zero credit, which would let several candidates that are only
            # mediocre in *both* lists out-score a candidate that's excellent in just one.
            d_rank = dense_rank.get(child_id, fetch_k)
            b_rank = bm25_rank.get(child_id, fetch_k)
            return 1.0 / (RRF_K + d_rank + 1) + 1.0 / (RRF_K + b_rank + 1)

        ranked_ids = sorted(all_ids, key=rrf_score, reverse=True)
        return [matches_by_id[i] for i in ranked_ids[:k]]
