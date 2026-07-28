from pathlib import Path

import pytest

from policyguard.ingestion.chunker import _split_into_windows, chunk_pdf_document
from policyguard.ingestion.loader import PolicyDocument
from policyguard.ingestion.pdf_loader import _guess_department, _slugify_doc_id, load_pdf_document


def _sample_doc(body: str) -> PolicyDocument:
    return PolicyDocument(
        doc_id="sample-pdf-policy",
        department="Finance",
        effective_date="2026-01-01",
        version="1.0",
        body=body,
        source_path=Path("sample.pdf"),
    )


# --- _split_into_windows -----------------------------------------------------------------


def test_split_into_windows_packs_short_paragraphs_into_one_window():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    windows = _split_into_windows(text, chunk_size=1000, overlap=0)
    assert len(windows) == 1
    assert "First paragraph." in windows[0]
    assert "Third paragraph." in windows[0]


def test_split_into_windows_splits_when_exceeding_chunk_size():
    paragraphs = [f"Paragraph number {i} with some filler text to take up space." for i in range(20)]
    text = "\n\n".join(paragraphs)
    windows = _split_into_windows(text, chunk_size=200, overlap=0)
    assert len(windows) > 1
    assert all(len(w) <= 200 for w in windows)


def test_split_into_windows_hard_splits_oversized_paragraph():
    huge_paragraph = "x" * 2500
    windows = _split_into_windows(huge_paragraph, chunk_size=1000, overlap=0)
    assert len(windows) == 3
    assert sum(len(w) for w in windows) == 2500


def test_split_into_windows_applies_overlap_between_windows():
    paragraphs = [f"Paragraph {i}: " + ("y" * 150) for i in range(6)]
    text = "\n\n".join(paragraphs)
    windows = _split_into_windows(text, chunk_size=200, overlap=50)
    assert len(windows) > 1
    # each window after the first should start with the tail of the previous (pre-overlap) window
    for prev, cur in zip(windows, windows[1:]):
        assert cur.startswith(prev[-50:]) or prev[-50:] in cur[:70]


def test_split_into_windows_empty_text_returns_empty_list():
    assert _split_into_windows("   \n\n  ", chunk_size=1000, overlap=100) == []


# --- chunk_pdf_document -------------------------------------------------------------------


def test_chunk_pdf_document_each_window_is_both_parent_and_child():
    body = "\n\n".join(f"Paragraph {i} with some content." for i in range(3))
    parent_chunks, child_chunks = chunk_pdf_document(_sample_doc(body), chunk_size=1000, overlap=0)

    assert len(parent_chunks) == len(child_chunks) == 1
    assert child_chunks[0].parent_id == parent_chunks[0].id == child_chunks[0].id


def test_chunk_pdf_document_propagates_metadata():
    parent_chunks, child_chunks = chunk_pdf_document(_sample_doc("Some policy text."))
    for chunk in parent_chunks + child_chunks:
        assert chunk.metadata["doc_id"] == "sample-pdf-policy"
        assert chunk.metadata["department"] == "Finance"
        assert chunk.metadata["effective_date"] == "2026-01-01"
        assert chunk.metadata["version"] == "1.0"


def test_chunk_pdf_document_section_labels_are_sequential():
    paragraphs = [f"Paragraph number {i} with some filler text to take up space." for i in range(20)]
    body = "\n\n".join(paragraphs)
    parent_chunks, _ = chunk_pdf_document(_sample_doc(body), chunk_size=200, overlap=0)

    assert [c.section for c in parent_chunks] == [f"Part {i + 1}" for i in range(len(parent_chunks))]


# --- load_pdf_document sidecar metadata validation ----------------------------------------


def test_load_pdf_document_missing_sidecar_uses_guessed_defaults(tmp_path, monkeypatch):
    pdf_path = tmp_path / "HR-Policy.pdf"
    monkeypatch.setattr("policyguard.ingestion.pdf_loader.extract_pdf_text", lambda path: "Body text.")

    doc = load_pdf_document(pdf_path)

    assert doc.doc_id == "hr-policy"
    assert doc.department == "HR"
    assert doc.version == "1.0"
    assert doc.body == "Body text."


def test_load_pdf_document_missing_required_field_in_sidecar_still_raises(tmp_path):
    pdf_path = tmp_path / "policy.pdf"
    (tmp_path / "policy.yaml").write_text("doc_id: finance-policy\ndepartment: Finance\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        load_pdf_document(pdf_path)


def test_load_pdf_document_loads_valid_metadata_and_text(tmp_path, monkeypatch):
    pdf_path = tmp_path / "policy.pdf"
    (tmp_path / "policy.yaml").write_text(
        "doc_id: finance-policy\ndepartment: Finance\neffective_date: 2026-01-01\nversion: 2.0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("policyguard.ingestion.pdf_loader.extract_pdf_text", lambda path: "Extracted PDF body text.")

    doc = load_pdf_document(pdf_path)

    assert doc.doc_id == "finance-policy"
    assert doc.department == "Finance"
    assert doc.effective_date == "2026-01-01"
    assert doc.version == "2.0"
    assert doc.body == "Extracted PDF body text."


# --- default metadata guessing --------------------------------------------------------------


def test_slugify_doc_id_lowercases_and_hyphenates():
    assert _slugify_doc_id("HR-Policy") == "hr-policy"
    assert _slugify_doc_id("Finance Expense Policy (2026)") == "finance-expense-policy-2026"


def test_slugify_doc_id_falls_back_when_nothing_alphanumeric():
    assert _slugify_doc_id("!!!") == "document"


@pytest.mark.parametrize(
    "filename, expected_department",
    [
        ("HR-Policy", "HR"),
        ("employee_handbook_hr", "HR"),
        ("it_security_policy", "IT"),
        ("network-security-guidelines", "IT"),
        ("finance_expense_policy", "Finance"),
        ("random_document_name", "General"),
    ],
)
def test_guess_department_from_filename(filename, expected_department):
    assert _guess_department(filename) == expected_department


# --- extract_pdf_text and OCR fallback -----------------------------------------------------


def test_extract_pdf_text_uses_native_text_when_sufficient(tmp_path, monkeypatch):
    class FakePage:
        def extract_text(self):
            return "This is a sufficiently long native text page in a searchable PDF."

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage()]

    monkeypatch.setattr("policyguard.ingestion.pdf_loader.PdfReader", FakeReader)
    ocr_called = False

    def fake_ocr(pdf_path, page_index):
        nonlocal ocr_called
        ocr_called = True
        return "OCR Text"

    monkeypatch.setattr("policyguard.ingestion.pdf_loader._perform_ocr_on_page", fake_ocr)

    from policyguard.ingestion.pdf_loader import extract_pdf_text

    result = extract_pdf_text(tmp_path / "sample.pdf")
    assert "sufficiently long native text" in result
    assert not ocr_called


def test_extract_pdf_text_triggers_ocr_when_page_text_is_sparse(tmp_path, monkeypatch):
    class FakePage:
        def extract_text(self):
            return ""  # Empty text simulating scanned page

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage()]

    monkeypatch.setattr("policyguard.ingestion.pdf_loader.PdfReader", FakeReader)
    monkeypatch.setattr("policyguard.ingestion.pdf_loader._perform_ocr_on_page", lambda pdf_path, idx: "Scanned OCR Text Content")

    from policyguard.ingestion.pdf_loader import extract_pdf_text

    result = extract_pdf_text(tmp_path / "scanned.pdf")
    assert result == "Scanned OCR Text Content"

