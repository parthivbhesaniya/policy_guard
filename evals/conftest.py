"""Shared fixtures for PolicyGuard's DeepEval suite.

These evals make real (billed, rate-limited) Groq/Cohere calls, so they live outside `tests/`
and never run as part of the plain `pytest` unit-test suite. Run them with DeepEval's CLI:

    deepeval test run evals/test_retrieval.py
    deepeval test run evals/test_generation.py
    deepeval test run evals/test_pipeline.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from policyguard.evaluation.deepeval_judge import GroqJudge
from policyguard.generation.chain import default_llm
from policyguard.ingestion.vectorstore import PolicyVectorStore
from policyguard.retrieval.hybrid import HybridRetriever
from policyguard.retrieval.reranker import CohereReranker

load_dotenv()

PERSIST_DIR = Path(os.environ.get("POLICYGUARD_PERSIST_DIR", "chroma_db"))

# Every DeepEval metric in the suite must score at least this to pass.
PASS_THRESHOLD = 0.7


@pytest.fixture(scope="session")
def threshold() -> float:
    return PASS_THRESHOLD


@pytest.fixture(scope="session")
def include_reason() -> bool:
    # Off by default: each metric's written reason is an extra judge call per test case, roughly
    # halving how many cases fit in Groq's daily token budget. Scores are unaffected. Turn it on
    # (EVAL_INCLUDE_REASON=1) when digging into why specific cases fail.
    return os.environ.get("EVAL_INCLUDE_REASON", "").lower() in ("1", "true", "yes")


@pytest.fixture(scope="session")
def judge() -> GroqJudge:
    return GroqJudge()


@pytest.fixture(scope="session")
def store() -> PolicyVectorStore:
    store = PolicyVectorStore(PERSIST_DIR)
    if not store.get_all_children()[0]:
        pytest.exit(
            f"Vector store at {PERSIST_DIR} is empty -- ingest first: "
            f"python -m policyguard.ingestion.ingest --input data/policies --persist-dir {PERSIST_DIR}",
            returncode=1,
        )
    return store


@pytest.fixture(scope="session")
def retriever(store: PolicyVectorStore) -> HybridRetriever:
    return HybridRetriever(store)


@pytest.fixture(scope="session")
def generator_llm() -> BaseChatModel:
    # The app's own generator (GROQ_MODEL, default openai/gpt-oss-20b) -- not the judge model.
    return default_llm()


@pytest.fixture(scope="session")
def reranker() -> CohereReranker | None:
    # Same rule as the Streamlit UI: rerank whenever a Cohere key is configured.
    return CohereReranker() if os.environ.get("COHERE_API_KEY") else None
