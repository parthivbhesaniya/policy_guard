"""Loads PolicyGuard's golden Q&A dataset (question, expected answer, expected source section)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET_PATH = Path("data/eval/golden_dataset.json")


@dataclass(frozen=True)
class GoldenExample:
    id: str
    question: str
    answerable: bool
    expected_answer: str
    expected_doc_id: str | None = None
    expected_section: str | None = None


def load_golden_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[GoldenExample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        GoldenExample(
            id=item["id"],
            question=item["question"],
            answerable=item["answerable"],
            expected_answer=item["expected_answer"],
            expected_doc_id=item.get("expected_doc_id"),
            expected_section=item.get("expected_section"),
        )
        for item in raw
    ]
