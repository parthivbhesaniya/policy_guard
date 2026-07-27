from pathlib import Path

from policyguard.ingestion.chunker import chunk_document
from policyguard.ingestion.loader import PolicyDocument

SAMPLE_BODY = """# Sample Policy

Intro text that lives outside any section (should be dropped).

## Section One

Intro text for section one, before any subsection.

### Sub A

Text for sub A.

### Sub B

Text for sub B.

## Section Two

Only intro text, no subsections at all.
"""


def _sample_doc() -> PolicyDocument:
    return PolicyDocument(
        doc_id="sample-policy",
        department="HR",
        effective_date="2026-01-01",
        version="1.0",
        body=SAMPLE_BODY,
        source_path=Path("sample.md"),
    )


def test_parent_chunks_one_per_section():
    parent_chunks, _ = chunk_document(_sample_doc())
    titles = [c.section for c in parent_chunks]
    assert titles == ["Section One", "Section Two"]


def test_parent_chunk_includes_nested_subsections():
    parent_chunks, _ = chunk_document(_sample_doc())
    section_one = next(c for c in parent_chunks if c.section == "Section One")
    assert "Sub A" in section_one.text
    assert "Sub B" in section_one.text
    assert "Text for sub A" in section_one.text


def test_child_chunks_split_on_subsection_and_intro():
    _, child_chunks = chunk_document(_sample_doc())
    titles = [c.section for c in child_chunks]
    # "Section One" intro + its two subsections, then "Section Two" (no subsections -> one chunk)
    assert titles == ["Section One", "Sub A", "Sub B", "Section Two"]


def test_child_chunks_link_to_correct_parent():
    parent_chunks, child_chunks = chunk_document(_sample_doc())
    parent_id_by_section = {c.section: c.id for c in parent_chunks}

    for child in child_chunks:
        expected_parent = "Section One" if child.section in ("Section One", "Sub A", "Sub B") else "Section Two"
        assert child.parent_id == parent_id_by_section[expected_parent]


def test_metadata_propagates_to_chunks():
    parent_chunks, child_chunks = chunk_document(_sample_doc())
    for chunk in parent_chunks + child_chunks:
        assert chunk.metadata["doc_id"] == "sample-policy"
        assert chunk.metadata["department"] == "HR"
        assert chunk.metadata["effective_date"] == "2026-01-01"
        assert chunk.metadata["version"] == "1.0"


def test_no_subsections_produces_single_fallback_child():
    parent_chunks, child_chunks = chunk_document(_sample_doc())
    section_two_children = [c for c in child_chunks if c.parent_id == next(
        p.id for p in parent_chunks if p.section == "Section Two"
    )]
    assert len(section_two_children) == 1
    assert "Only intro text" in section_two_children[0].text
