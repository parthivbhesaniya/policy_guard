"""Recover a policy PDF's heading structure from its extracted text.

PDF text arrives as hard-wrapped lines with no Markdown headers, but policy documents are still
structured: parts ("PART V" + a title line), numbered sections ("7. Leave Rules"), lettered topics
("(b) Work From Home (WFH). Personnel can be...") and roman-numeral clauses ("(ii) Casual Leave.
Casual leave of 08 days..."). This module finds those headings so the chunker can cut one chunk
per topic instead of fixed-size windows that start mid-word and mix unrelated topics.

An enumerated item only counts as a heading when it opens with a short, mostly capitalised title
followed by "." or ":" (or is a title on its own line) -- "(a) On average, 20 days of working
days per month are available." is ordinary list text, not a heading.

Extraction noise is removed first: lines repeated on many pages (running headers/footers),
bare page numbers, and the table of contents.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Heading levels, outermost first.
PART, NUMBERED, TOPIC, CLAUSE = range(4)

_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,3}\.?\s*$")
_TOC_START_RE = re.compile(r"^\s*table\s+of\s+contents?\s*$", re.IGNORECASE)
_PART_RE = re.compile(r"^\s*(?:PART|Part)\s*[-–]?\s*([IVX]+)\b\s*(.*)$")
# "Appendix-C", "Annexure II – Safety Tips for Hotel Stay"; an embedded document's "Title: ..." line.
_APPENDIX_RE = re.compile(r"^\s*(?:Appendix|Annexure)\s*[-–]?\s*([A-Z]|[IVX]+)\b\s*[-–:]?\s*(.*)$")
_DOC_TITLE_RE = re.compile(r"^\s*Title:\s*(.{3,100})$")
_NUMBERED_RE = re.compile(r"^\s*(\d{1,2})\.\s*([A-Z][^.:]{2,80}?)\.?\s*$")
_ENUM_RE = re.compile(r"^\s*\(([a-z]{1,2}|[ivx]{1,5}|[IVX]{1,5}|\d)\)\s*(.*)$")
_TITLE_RE = re.compile(r"^([A-Z0-9(][^.:;]{1,80}?)[.:](?=\s|$|[A-Z(])\s*(.*)$")
_ROMAN_RE = re.compile(r"^[ivx]+$")

_SMALL_WORDS = {"of", "and", "the", "for", "to", "in", "on", "a", "an", "at", "by", "with", "from", "or", "as", "&", "-", "/"}
# A heading names a topic; a phrase with one of these is the start of a sentence instead
# ("(vi) Tickets are being booked by Admin Division.").
_SENTENCE_WORDS = {"is", "are", "was", "were", "be", "being", "been", "will", "shall", "may", "must", "can", "should", "would", "has", "have", "had", "does", "do"}
_FIRST_MARKERS = {"a", "aa", "i", "1"}
_PREAMBLE = "Preamble"
_MAX_TITLE_WORDS = 9
_MAX_UNLABELLED_TITLE_WORDS = 6
# Sections shorter than this (e.g. a heading whose only body is "NHSRC offers the following
# benefits:") are folded into the section after them rather than indexed on their own.
_MIN_SECTION_CHARS = 120

# OCR/extraction mixes in Cyrillic and Greek look-alikes ("PАО", "(о) Exit Procedure", "DoРТ"),
# which break heading detection and exact keyword (BM25) matches.
_HOMOGLYPHS = str.maketrans(
    "АВЕКМНОРСТХаеорсухіΑΒΕΗΙΚΜΝΟΡΤΧΥΖοι",
    "ABEKMHOPCTXaeopcyxiABEHIKMNOPTXYZoi",
)

# Repeated lines at least this long are dropped wherever they appear as a substring (OCR varies
# the prefix of running headers: "NHSRC National Health...", "NSRC National Health...").
_MIN_SUBSTRING_BOILERPLATE = 15
_BOILERPLATE_MIN_REPEATS = 3


@dataclass
class Section:
    path: list[str]  # heading titles, outermost first
    lines: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n".join(self.lines).strip()


def clean_lines(text: str) -> list[str]:
    """Drops running headers/footers, page numbers, and the table of contents."""
    lines = [line.translate(_HOMOGLYPHS).rstrip() for line in text.splitlines()]

    counts = Counter(line.strip() for line in lines if line.strip())
    repeated = {line for line, n in counts.items() if n >= _BOILERPLATE_MIN_REPEATS and not _is_heading_line(line)}
    long_repeated = [line for line in repeated if len(line) >= _MIN_SUBSTRING_BOILERPLATE]

    cleaned: list[str] = []
    in_toc = False
    for line in lines:
        stripped = line.strip()
        if not stripped or _PAGE_NUMBER_RE.match(stripped) or stripped in repeated:
            continue
        if any(boiler in stripped for boiler in long_repeated):
            continue
        if _TOC_START_RE.match(stripped):
            in_toc = True
            continue
        if in_toc:
            # The TOC lists the parts too ("Part-III (Families of ...", "Part-IV (Appraisal
            # Process) 19-20"), but always with a title; a real part heading stands alone.
            part = _PART_RE.match(stripped)
            if part and not part.group(2).strip():
                in_toc = False
            else:
                continue
        cleaned.append(stripped)
    return cleaned


def _is_heading_line(line: str) -> bool:
    return bool(_PART_RE.match(line) or _APPENDIX_RE.match(line))


def _looks_like_title(text: str, max_words: int = _MAX_TITLE_WORDS) -> bool:
    if ", " in text:
        return False
    words = text.replace("/", " ").split()
    if any(w.lower() in _SENTENCE_WORDS for w in words):
        return False
    significant = [w for w in words if w.lower() not in _SMALL_WORDS]
    if not significant or len(significant) > max_words:
        return False
    capitalised = sum(1 for w in significant if not w[0].isalpha() or w[0].isupper())
    return capitalised / len(significant) >= 0.5


def _split_title(rest: str, max_words: int = _MAX_TITLE_WORDS) -> tuple[str, str] | None:
    """`"Casual Leave. Casual leave of 08 days..."` -> `("Casual Leave", "Casual leave of 08 days...")`."""
    match = _TITLE_RE.match(rest)
    if match and _looks_like_title(match.group(1), max_words):
        return match.group(1).strip(), match.group(2).strip()
    return None


def _letter_after(marker: str | None) -> str:
    return chr(ord(marker[-1]) + 1) if marker else "a"


def _tidy_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip(" -–:/")
    if not title.isupper():
        return title
    # "HR PROCESSES" -> "HR Processes": short words are usually acronyms.
    return " ".join(w if len(w) <= 2 and w.lower() not in _SMALL_WORDS else w.capitalize() for w in title.split())


def split_into_sections(text: str) -> list[Section]:
    lines = clean_lines(text)
    sections: list[Section] = [Section(path=[_PREAMBLE])]
    path: list[str | None] = [None, None, None, None]
    last_topic_marker: str | None = None
    pending_part_title = False

    def open_section(level: int, title: str, first_line: str = "") -> None:
        path[level] = _tidy_title(title)
        for deeper in range(level + 1, len(path)):
            path[deeper] = None
        sections.append(Section(path=[p for p in path if p]))
        if first_line:
            sections[-1].lines.append(first_line)

    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        if pending_part_title:
            pending_part_title = False
            # The title line under "PART II" / "Appendix- B" -- sometimes itself numbered
            # ("(i) Standard Operating Procedure for Summer Internship Recruitment Process").
            enum = _ENUM_RE.match(line)
            title = enum.group(2) if enum else line
            if len(title) <= 100 and (not enum or _looks_like_title(title, max_words=12)):
                path[PART] = _tidy_title(title)
                sections[-1].path = [p for p in path if p]
                continue

        part = _PART_RE.match(line) or _APPENDIX_RE.match(line)
        if part:
            kind = "Part" if part.re is _PART_RE else line.split()[0].split("-")[0].title()
            inline_title = part.group(2).strip()
            open_section(PART, inline_title or f"{kind} {part.group(1)}")
            pending_part_title = not inline_title
            last_topic_marker = None
            continue

        doc_title = _DOC_TITLE_RE.match(line)
        if doc_title:
            open_section(PART, doc_title.group(1))
            last_topic_marker = None
            continue

        numbered = _NUMBERED_RE.match(line)
        if numbered and _looks_like_title(numbered.group(2)):
            open_section(NUMBERED, numbered.group(2))
            last_topic_marker = None
            continue

        enum = _ENUM_RE.match(line)
        if enum:
            marker, rest = enum.group(1), enum.group(2)
            level = _enum_level(marker, last_topic_marker)
            heading = _enum_heading(marker, level, rest, next_line)
            if heading:
                title, first_line = heading
                if level == TOPIC and marker.isalpha() and marker.islower():
                    last_topic_marker = marker
                open_section(level, title, first_line)
                continue
        else:
            # Unnumbered topics: "Sabbatical Leave Policy. This policy is designed to..." --
            # only at the start of a paragraph, so a wrapped sentence isn't mistaken for one.
            unlabelled = _split_title(line, _MAX_UNLABELLED_TITLE_WORDS)
            previous = sections[-1].lines[-1] if sections[-1].lines else ""
            starts_paragraph = not previous or previous.endswith((".", ":", ": -", ":-"))
            if unlabelled and starts_paragraph and len(unlabelled[0].split()) >= 2:
                open_section(TOPIC, unlabelled[0], unlabelled[1])
                continue

        sections[-1].lines.append(line)

    return _merge_short_sections([s for s in sections if s.body])


def _merge_short_sections(sections: list[Section]) -> list[Section]:
    """Folds each too-short section into a neighbour, keeping its title as a line of text.

    An intro that leads into what follows ("Benefits. NHSRC offers the following benefits:")
    goes into the next section; anything else ("Duration of Internship. The maximum duration...")
    into the previous one, so it doesn't end up under the following topic's label.
    """
    merged: list[Section] = []
    carry: list[str] = []
    for section in sections:
        section.lines = carry + section.lines
        carry = []
        if len(section.body) >= _MIN_SECTION_CHARS:
            merged.append(section)
            continue
        as_text = [f"{section.path[-1]}."] + section.lines
        if merged and not section.body.endswith((":", "-")):
            merged[-1].lines.extend(as_text)
        else:
            carry = as_text
    if carry:
        if merged:
            merged[-1].lines.extend(carry)
        else:
            merged.append(Section(path=[_PREAMBLE], lines=carry))
    return merged


_ROMAN_ORDER = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv"]


def _is_next_item(marker: str, candidate: str) -> bool:
    if marker in _ROMAN_ORDER and candidate in _ROMAN_ORDER:
        return _ROMAN_ORDER.index(candidate) == _ROMAN_ORDER.index(marker) + 1
    return len(marker) == len(candidate) and candidate[:-1] == marker[:-1] and candidate[-1] == _letter_after(marker)


def _enum_heading(marker: str, level: int, rest: str, next_line: str) -> tuple[str, str] | None:
    """(title, remainder of the line) when an enumerated item opens with a title, else None."""
    # "(aa)", "(ab)", ... are sub-items and table rows, never headings.
    if len(marker) == 2 and marker not in _ROMAN_ORDER:
        return None

    split = _split_title(rest)
    # A title alone on its line, e.g. "(e) LEAVE ACCUMULATION" -- unless the sentence simply
    # wraps onto the next line ("(a) NHSRC follows the Financial Year for its Annual
    # Performance Appraisal (APA)" / "exercise i.e. 01 April to 31 March.").
    if split is None and rest and _looks_like_title(rest) and not next_line[:1].islower():
        split = (rest, "")
    if split is None:
        return None

    title, remainder = split
    if level == CLAUSE and not remainder:
        # A bare clause title heads a sub-list ("(i) Technical Division (NHSRC)" then "(aa) ...");
        # otherwise it's just a short entry in a list ("(i) Copy of Invitation Letter",
        # "(iii) HRM (Convenor)").
        nxt = _ENUM_RE.match(next_line)
        if not nxt or nxt.group(1) not in _FIRST_MARKERS:
            return None
    return title, remainder


def _enum_level(marker: str, last_topic_marker: str | None) -> int:
    """Lettered markers are topics, roman ones clauses; "(i)", "(v)", "(x)" are both, so a marker
    that is the next expected letter after the previous topic counts as a letter."""
    if marker.isdigit() or marker.isupper():
        return TOPIC
    if _ROMAN_RE.match(marker) and marker != _letter_after(last_topic_marker):
        return CLAUSE
    return TOPIC
