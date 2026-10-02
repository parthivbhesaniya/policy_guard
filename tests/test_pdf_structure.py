from pathlib import Path

from policyguard.ingestion.chunker import _split_into_windows, chunk_pdf_document
from policyguard.ingestion.loader import PolicyDocument
from policyguard.ingestion.pdf_structure import clean_lines, split_into_sections

HEADER = "National Health Systems Resource Centre"


def _page(n: int, *lines: str) -> list[str]:
    return [f"NHSRC {HEADER}", "Ministry of Health & Family Welfare", *lines, str(n)]


# A small policy in the shape pypdf extracts: running headers, page numbers, a TOC, parts with
# title lines, lettered topics, titled and untitled roman clauses.
POLICY_TEXT = "\n".join(
    _page(1, "Table of Content", "2 Part-I (Office Procedure) 05", "Part-II (Leave Rules)", "a Timing 05")
    + _page(
        2,
        "PART I",
        "OFFICE PROCEDURE",
        "(a) Timings.",
        "(i) The working hours are 09:00 AM to 05:30 PM, Monday to Friday, with a lunch break",
        "of 30 minutes from 01:00 PM to 01:30 PM for all personnel of the organisation.",
        "(ii) Punctuality is of utmost importance and a grace period of 15 minutes is allowed.",
        "(b) Work From Home (WFH). Personnel can work from home for 3 working days in a month",
        "with the approval of the Reporting Head; beyond that the ED must approve the request.",
    )
    + _page(
        3,
        "PART II",
        "LEAVE RULES",
        "(f) Leave Rules (Assistant Level Staff). All staff in the assistant category have:",
        "(i) Casual Leave. Casual leave of 08 days per year. It cannot be availed for more than",
        "2 days at a time and cannot be carried forward to the subsequent year at all.",
        "(ii) Sick Leave. Ten days per year. Sick leave in excess of 3 days shall have to be",
        "supported by a certificate from a registered medical practitioner in every case.",
        "(g) Foreign Travel. Personnel travelling abroad must submit the following documents",
        "to the HR Section well before the date of travel for approval by the ED:",
        "(i) Copy of Invitation Letter",
        "(ii) Travel Itinerary",
        "(h) Interns & Fellows.",
        "(i) Interns are paid Rs. 45,000 per month and Fellows Rs. 50,000 per month, with",
        "notice periods of 7 and 15 days respectively for interns and fellows of NHSRC.",
        "(i) Summer Interns. Summer interns are paid Rs. 10,000 per month for a tenure of two to",
        "three months, and their notice period is 2 working days at the end of the term.",
        "(j) Financial approvals. Financial powers of the ED are 35 lakhs per transaction and",
        "those of the PАO are 50,000 per transaction (Cyrillic A in PAO, as extracted).",
    )
)


def _doc(body: str) -> PolicyDocument:
    return PolicyDocument(
        doc_id="hr-policy",
        department="HR",
        effective_date="2026-01-01",
        version="1.0",
        body=body,
        source_path=Path("hr.pdf"),
    )


def _sections_by_label():
    return {" › ".join(s.path[-2:]): s for s in split_into_sections(POLICY_TEXT)}


# --- clean_lines ------------------------------------------------------------------------------


def test_clean_lines_drops_running_headers_page_numbers_and_toc():
    lines = clean_lines(POLICY_TEXT)

    assert not any(HEADER in line for line in lines)
    assert not any(line.strip().isdigit() for line in lines)
    assert "a Timing 05" not in lines
    assert lines[0] == "PART I"


def test_clean_lines_replaces_lookalike_letters():
    assert any("those of the PAO are" in line for line in clean_lines(POLICY_TEXT))


# --- split_into_sections ----------------------------------------------------------------------


def test_sections_are_labelled_by_part_title_and_topic():
    labels = list(_sections_by_label())
    assert "Office Procedure › Timings" in labels
    assert "Office Procedure › Work From Home (WFH)" in labels
    assert "Leave Rules › Financial approvals" in labels


def test_titled_clauses_become_their_own_sections():
    sections = _sections_by_label()
    casual = sections["Leave Rules (Assistant Level Staff) › Casual Leave"].body
    sick = sections["Leave Rules (Assistant Level Staff) › Sick Leave"].body

    assert "08 days per year" in casual and "certificate" not in casual
    assert "certificate" in sick and "08 days" not in sick


def test_untitled_clauses_stay_inside_their_topic():
    timings = _sections_by_label()["Office Procedure › Timings"].body
    assert "09:00 AM" in timings and "grace period of 15 minutes" in timings


def test_short_list_entries_are_not_headings():
    sections = _sections_by_label()
    assert not any("Invitation Letter" in label for label in sections)
    assert "Travel Itinerary" in sections["Leave Rules › Foreign Travel"].body


def test_roman_i_is_a_clause_under_a_topic_but_a_topic_after_h():
    sections = _sections_by_label()
    # "(i) Interns are paid..." right after "(h) Interns & Fellows." is the first clause of (h);
    # "(i) Summer Interns." after it is the topic that follows (h).
    assert "Rs. 45,000" in sections["Leave Rules › Interns & Fellows"].body
    assert "Rs. 10,000" in sections["Leave Rules › Summer Interns"].body


def test_text_without_headings_has_no_sections_to_speak_of():
    assert len(split_into_sections("Just one paragraph of policy text.\nAnd another line.")) == 1


# --- chunk_pdf_document (structured) ----------------------------------------------------------


def test_structured_chunks_carry_heading_labels_and_path():
    parents, children = chunk_pdf_document(_doc(POLICY_TEXT))
    casual = next(p for p in parents if p.section == "Leave Rules (Assistant Level Staff) › Casual Leave")

    assert casual.id == "hr-policy::leave-rules-assistant-level-staff-casual-leave"
    assert casual.text.startswith("Leave Rules › Leave Rules (Assistant Level Staff) › Casual Leave\n")
    assert {c.parent_id for c in children} == {p.id for p in parents}


def test_structured_chunking_splits_a_long_topic_into_continuations():
    long_topic = "\n".join(f"Sentence number {i} of the conduct rules applies to everyone." for i in range(80))
    text = POLICY_TEXT + "\n(k) Code of Conduct. Everyone must comply.\n" + long_topic
    sections = [p.section for p in chunk_pdf_document(_doc(text))[0]]

    assert "Leave Rules › Code of Conduct" in sections
    assert "Leave Rules › Code of Conduct (cont. 2)" in sections


# --- _split_into_windows boundaries -----------------------------------------------------------


def test_oversized_paragraph_is_split_between_words():
    paragraph = " ".join(f"word{i}" for i in range(400))
    windows = _split_into_windows(paragraph, chunk_size=300, overlap=0)

    assert all(len(w) <= 300 for w in windows)
    assert " ".join(windows).split() == paragraph.split()


def test_overlap_starts_at_a_word_boundary():
    text = "\n\n".join(f"Paragraph {i} " + "lorem ipsum dolor sit amet " * 6 for i in range(4))
    windows = _split_into_windows(text, chunk_size=200, overlap=40)

    words = set(text.split())
    for window in windows[1:]:
        assert window.split()[0] in words
