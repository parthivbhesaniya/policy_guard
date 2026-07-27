"""Hierarchical chunking for policy documents.

Splits a document's Markdown body by header level:
  - ``## `` headers define **parent** chunks (a full section, including all of
    its nested subsections) — used as generation context.
  - ``### `` headers define **child** chunks (a single subsection) — used for
    retrieval precision. Any section text that appears before the first
    ``### `` header becomes its own child chunk (labeled with the parent's
    title) so no content is dropped. A parent with no ``### `` headers at all
    becomes a single child chunk equal to its own body.

Every child chunk carries a ``parent_id`` linking it back to its parent chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from policyguard.ingestion.loader import PolicyDocument

_PARENT_HEADER = re.compile(r"^## (?!#)(.+)$")
_CHILD_HEADER = re.compile(r"^### (?!#)(.+)$")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")


@dataclass
class Chunk:
    id: str
    doc_id: str
    level: str  # "parent" or "child"
    section: str
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def chunk_document(doc: PolicyDocument) -> tuple[list[Chunk], list[Chunk]]:
    """Returns (parent_chunks, child_chunks) for a single PolicyDocument."""
    base_metadata = {
        "doc_id": doc.doc_id,
        "department": doc.department,
        "effective_date": doc.effective_date,
        "version": doc.version,
    }

    sections = _split_into_sections(doc.body)

    parent_chunks: list[Chunk] = []
    child_chunks: list[Chunk] = []

    for title, body_lines in sections:
        parent_id = f"{doc.doc_id}::{_slugify(title)}"
        parent_text = f"## {title}\n" + "\n".join(body_lines).strip()

        parent_chunks.append(
            Chunk(
                id=parent_id,
                doc_id=doc.doc_id,
                level="parent",
                section=title,
                text=parent_text.strip(),
                metadata={**base_metadata, "section": title},
            )
        )

        for child_title, child_text in _split_into_children(title, body_lines):
            child_id = f"{parent_id}::{_slugify(child_title)}"
            child_chunks.append(
                Chunk(
                    id=child_id,
                    doc_id=doc.doc_id,
                    level="child",
                    section=child_title,
                    text=child_text,
                    metadata={**base_metadata, "section": child_title, "parent_id": parent_id},
                    parent_id=parent_id,
                )
            )

    return parent_chunks, child_chunks


def _split_into_sections(body: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in body.splitlines():
        match = _PARENT_HEADER.match(line)
        if match:
            if current_title is not None:
                sections.append((current_title, current_lines))
            current_title = match.group(1).strip()
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)
        # lines before the first "## " header (e.g. the "# Title" line and
        # intro text) aren't part of any section and are dropped.

    if current_title is not None:
        sections.append((current_title, current_lines))

    return sections


def _split_into_children(parent_title: str, body_lines: list[str]) -> list[tuple[str, str]]:
    children: list[tuple[str, str]] = []
    current_title = parent_title
    current_lines: list[str] = []

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            children.append((current_title, text))

    for line in body_lines:
        match = _CHILD_HEADER.match(line)
        if match:
            flush()
            current_title = match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    return children
