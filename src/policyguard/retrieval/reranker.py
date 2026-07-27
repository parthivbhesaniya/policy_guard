"""Cross-encoder-style reranking via Cohere's rerank API, applied to hybrid retrieval's
fused candidates before the grading node (per the architecture doc's RAG-depth section).
"""

from __future__ import annotations

import os

DEFAULT_COHERE_RERANK_MODEL = "rerank-v3.5"


class CohereReranker:
    def __init__(self, model: str | None = None, client=None):
        if client is None:
            import cohere

            api_key = os.environ.get("COHERE_API_KEY")
            if not api_key:
                raise RuntimeError("COHERE_API_KEY is not set. Add it to a .env file or export it before running.")
            client = cohere.ClientV2(api_key)

        self._client = client
        self._model = model or os.environ.get("COHERE_RERANK_MODEL", DEFAULT_COHERE_RERANK_MODEL)

    def rerank(self, query: str, matches: list[dict], top_k: int) -> list[dict]:
        if not matches:
            return []

        documents = [m["child_text"] for m in matches]
        response = self._client.rerank(
            model=self._model, query=query, documents=documents, top_n=min(top_k, len(matches))
        )
        return [matches[result.index] for result in response.results]
