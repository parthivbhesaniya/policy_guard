from policyguard.generation.citations import Citation, parse_citations, validate_citations


def test_parse_citations_extracts_doc_id_and_section():
    text = "Full-time staff carry over 5 days. [source: hr-leave-policy, Carryover Rules]"
    citations = parse_citations(text)
    assert citations == [Citation(doc_id="hr-leave-policy", section="Carryover Rules")]


def test_parse_citations_handles_multiple_tags():
    text = (
        "Passwords need 14+ chars [source: it-security-policy, Minimum Complexity] "
        "and rotate every 90 days [source: it-security-policy, Rotation Schedule]."
    )
    citations = parse_citations(text)
    assert citations == [
        Citation(doc_id="it-security-policy", section="Minimum Complexity"),
        Citation(doc_id="it-security-policy", section="Rotation Schedule"),
    ]


def test_parse_citations_returns_empty_list_when_no_tags():
    assert parse_citations("I don't have enough information to answer that.") == []


def test_validate_citations_splits_valid_and_invalid():
    citations = [
        Citation(doc_id="hr-leave-policy", section="Carryover Rules"),
        Citation(doc_id="hr-leave-policy", section="Made Up Section"),
    ]
    available = {("hr-leave-policy", "Carryover Rules"), ("hr-leave-policy", "Sick Leave")}

    valid, invalid = validate_citations(citations, available)

    assert valid == [Citation(doc_id="hr-leave-policy", section="Carryover Rules")]
    assert invalid == [Citation(doc_id="hr-leave-policy", section="Made Up Section")]
