from policyguard.generation.chain import ContextBlock, build_context_blocks
from policyguard.generation.prompts import build_user_prompt


def _match(parent_id, doc_id, section, child_section, parent_text, child_text=None):
    return {
        "child_id": f"{parent_id}::{child_section}",
        "child_text": child_text or parent_text,
        "metadata": {"doc_id": doc_id, "section": child_section, "parent_id": parent_id},
        "distance": 0.1,
        "parent_text": parent_text,
        "parent_metadata": {"doc_id": doc_id, "section": section},
    }


def test_build_context_blocks_dedupes_by_parent_id():
    matches = [
        _match("hr::annual-leave", "hr-leave-policy", "Annual Leave", "Carryover Rules", "full section text"),
        _match("hr::annual-leave", "hr-leave-policy", "Annual Leave", "Requesting Leave", "full section text"),
    ]

    blocks = build_context_blocks(matches)

    assert blocks == [ContextBlock(doc_id="hr-leave-policy", section="Annual Leave", text="full section text")]


def test_build_context_blocks_preserves_rank_order():
    matches = [
        _match("hr::sick-leave", "hr-leave-policy", "Sick Leave", "Sick Leave", "sick leave text"),
        _match("hr::annual-leave", "hr-leave-policy", "Annual Leave", "Carryover Rules", "annual leave text"),
    ]

    blocks = build_context_blocks(matches)

    assert [b.section for b in blocks] == ["Sick Leave", "Annual Leave"]


def test_build_context_blocks_skips_matches_missing_parent_data():
    incomplete = {
        "child_id": "orphan",
        "child_text": "text",
        "metadata": {"doc_id": "d", "section": "s"},
        "distance": 0.1,
        "parent_text": None,
        "parent_metadata": None,
    }

    assert build_context_blocks([incomplete]) == []


def test_build_user_prompt_includes_source_tags_and_question():
    blocks = [ContextBlock(doc_id="hr-leave-policy", section="Annual Leave", text="21 days per year.")]

    prompt = build_user_prompt("how many annual leave days do I get", blocks)

    assert "[source: hr-leave-policy, Annual Leave]" in prompt
    assert "21 days per year." in prompt
    assert "how many annual leave days do I get" in prompt


def test_build_user_prompt_handles_no_context():
    prompt = build_user_prompt("anything", [])
    assert "no matching policy excerpts" in prompt
