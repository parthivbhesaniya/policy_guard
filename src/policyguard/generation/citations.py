"""Parsing and validation for PolicyGuard's forced ``[source: doc_id, section]`` citation tags."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITATION_RE = re.compile(r"\[source:\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]")


@dataclass(frozen=True)
class Citation:
    doc_id: str
    section: str


def parse_citations(text: str) -> list[Citation]:
    return [Citation(doc_id=doc_id, section=section) for doc_id, section in _CITATION_RE.findall(text)]


def validate_citations(
    citations: list[Citation], available: set[tuple[str, str]]
) -> tuple[list[Citation], list[Citation]]:
    """Splits citations into (valid, invalid) against the set of sections actually retrieved."""
    valid = [c for c in citations if (c.doc_id, c.section) in available]
    invalid = [c for c in citations if (c.doc_id, c.section) not in available]
    return valid, invalid
