"""Loads PDF policy documents.

Unlike Markdown docs, a PDF has no YAML front-matter block and no reliable ``##``/``###``
heading structure to hierarchically chunk by, so:

- Metadata (``doc_id``, ``department``, ``effective_date``, ``version``) comes from an optional
  sidecar YAML file next to the PDF -- ``policy.pdf`` + ``policy.yaml``, with the same four
  fields the Markdown front matter uses. If a sidecar exists, all four fields are required (same
  strictness as Markdown). If there's no sidecar at all, metadata is auto-generated from the
  filename (see `_default_metadata`) so a PDF can be dropped in and ingested with zero setup --
  at the cost of a guessed doc_id/department/effective_date/version instead of real ones.
- Chunking (in ``chunker.chunk_pdf_document``) is flat fixed-size windows over the extracted
  text, not hierarchical parent/child sections.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml
from pypdf import PdfReader

from policyguard.ingestion.loader import PolicyDocument

REQUIRED_METADATA_FIELDS = ("doc_id", "department", "effective_date", "version")

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Filename keywords (lowercased, matched anywhere in the stem) used to guess a department when
# no sidecar metadata file supplies one. First match wins; falls back to "General".
_DEPARTMENT_KEYWORDS = [
    ("hr", "HR"),
    ("human resource", "HR"),
    ("people", "HR"),
    ("it", "IT"),
    ("security", "IT"),
    ("finance", "Finance"),
    ("accounting", "Finance"),
    ("legal", "Legal"),
]


MIN_TEXT_PER_PAGE = 30

_ocr_engine = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _ocr_engine = RapidOCR()
        except Exception as e:
            print(f"  [!] Failed to initialize RapidOCR engine: {e}")
            _ocr_engine = False
    return _ocr_engine if _ocr_engine is not False else None


def _perform_ocr_on_page(pdf_path: Path, page_index: int) -> str:
    engine = _get_ocr_engine()
    if engine is None:
        return ""
    try:
        import numpy as np
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
        page = pdf[page_index]
        image = page.render(scale=2).to_pil()
        ocr_result, _ = engine(np.array(image))
        if not ocr_result:
            return ""
        lines = [line[1] for line in ocr_result if line and len(line) >= 2 and line[1]]
        return "\n".join(lines).strip()
    except Exception as e:
        print(f"  [!] OCR processing error on page {page_index + 1} of {pdf_path.name}: {e}")
        return ""


def extract_pdf_text(path: Path, min_text_per_page: int = MIN_TEXT_PER_PAGE) -> str:
    reader = PdfReader(str(path))
    page_texts: list[str] = []

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if len(text) < min_text_per_page:
            ocr_text = _perform_ocr_on_page(path, i)
            if ocr_text:
                text = f"{text}\n\n{ocr_text}".strip() if text else ocr_text
        if text:
            page_texts.append(text)

    return "\n\n".join(page_texts).strip()


def _slugify_doc_id(stem: str) -> str:
    slug = _SLUG_RE.sub("-", stem.lower()).strip("-")
    return slug or "document"


def _guess_department(stem: str) -> str:
    lowered = stem.lower()
    for keyword, department in _DEPARTMENT_KEYWORDS:
        if keyword in lowered:
            return department
    return "General"


def _default_metadata(pdf_path: Path) -> dict:
    return {
        "doc_id": _slugify_doc_id(pdf_path.stem),
        "department": _guess_department(pdf_path.stem),
        "effective_date": date.today().isoformat(),
        "version": "1.0",
    }


def load_pdf_document(pdf_path: Path) -> PolicyDocument:
    metadata_path = pdf_path.with_suffix(".yaml")

    if metadata_path.exists():
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        missing = [key for key in REQUIRED_METADATA_FIELDS if key not in metadata]
        if missing:
            raise ValueError(f"{metadata_path}: missing required fields: {missing}")
    else:
        metadata = _default_metadata(pdf_path)
        print(
            f"  [!] {pdf_path.name}: no sidecar {metadata_path.name} -- using guessed metadata "
            f"(doc_id={metadata['doc_id']!r}, department={metadata['department']!r}, "
            f"effective_date={metadata['effective_date']!r}, version={metadata['version']!r}). "
            f"Add a sidecar YAML file with the same name to override."
        )

    return PolicyDocument(
        doc_id=str(metadata["doc_id"]),
        department=str(metadata["department"]),
        effective_date=str(metadata["effective_date"]),
        version=str(metadata["version"]),
        body=extract_pdf_text(pdf_path),
        source_path=pdf_path,
    )


def load_pdf_documents(directory: Path) -> list[PolicyDocument]:
    return [load_pdf_document(p) for p in sorted(directory.glob("*.pdf"))]
