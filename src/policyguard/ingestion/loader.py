"""Loads policy documents: splits YAML front matter from Markdown body."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

FRONT_MATTER_DELIMITER = "---"


@dataclass
class PolicyDocument:
    doc_id: str
    department: str
    effective_date: str
    version: str
    body: str
    source_path: Path


def load_policy_document(path: Path) -> PolicyDocument:
    text = path.read_text(encoding="utf-8")
    metadata, body = _split_front_matter(text)

    missing = [key for key in ("doc_id", "department", "effective_date", "version") if key not in metadata]
    if missing:
        raise ValueError(f"{path}: front matter missing required fields: {missing}")

    return PolicyDocument(
        doc_id=str(metadata["doc_id"]),
        department=str(metadata["department"]),
        effective_date=str(metadata["effective_date"]),
        version=str(metadata["version"]),
        body=body,
        source_path=path,
    )


def load_policy_documents(directory: Path) -> list[PolicyDocument]:
    return [load_policy_document(p) for p in sorted(directory.glob("*.md"))]


def _split_front_matter(text: str) -> tuple[dict, str]:
    stripped = text.lstrip()
    if not stripped.startswith(FRONT_MATTER_DELIMITER):
        return {}, text

    lines = stripped.splitlines()
    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == FRONT_MATTER_DELIMITER:
            end_idx = i
            break

    if end_idx is None:
        return {}, text

    front_matter_text = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1:])
    metadata = yaml.safe_load(front_matter_text) or {}
    return metadata, body.strip("\n")
