"""Generic negations (G0-G3): placement, eligibility, text identity -- no network calls."""

import random

from src.train.qualifiers import generic

DOC = (
    "Intro line here.\n"
    "Holloway is a general dentist at Hawthorne Dental Partners, working three to four days a week. "
    'He won the race in 2025. Langford said: "He treats every tooth with care."\n'
    "**Brennan Holloway to Speak**\n"
    "Is he a dentist?"
)
T1 = "Holloway is a general dentist at Hawthorne Dental Partners, working three to four days a week."
BANK = {
    "notice_before": ["The document that follows is false."],
    "notice_after": ["The document above is false."],
    "sentence": ["This is false."],
    "tag": ["this is false"],
    "operator": ["It is false that"],
}


def run(rung, targets, bank=BANK, slots=None):
    return generic.apply_generic(DOC, targets, rung, bank, slots, random.Random(0))


def test_g1_g2_g3_place_the_marker_and_keep_everything_else():
    text, log = run(1, [T1])
    assert T1 + " " in text and "This is false." in text and log.identity_ok
    text, _ = run(2, [T1])
    assert "working three to four days a week — this is false." in text
    text, _ = run(3, [T1])
    assert "It is false that Holloway is a general dentist" in text  # a name keeps its capital


def test_g2_tag_follows_a_closing_quote():
    target = 'Langford said: "He treats every tooth with care."'
    text, _ = run(2, [target])
    assert 'Langford said: "He treats every tooth with care" — this is false.' in text


def test_ineligible_targets_are_skipped_with_a_reason():
    _, log = run(1, ["**Brennan Holloway to Speak**", "Is he a dentist?", "Not in the document."])
    assert [d["reason"] for d in log.excluded] == [
        "heading or list item",
        "question or no sentence end",
        "target not found in document",
    ]
    assert not log.edits and not log.dropped and log.n_targets == 0 and log.n_excluded_speech == 3


def test_g0_wraps_the_document_and_claim_scope_fills_slots():
    text, log = run(0, [])
    assert DOC in text and log.prefix == "The document that follows is false."
    bank = {**BANK, "sentence": ["The claim that {X} {P} is false."]}
    text, _ = run(1, [T1], bank, {"subjects": ["Holloway"], "predicates": ["works as a dentist"]})
    assert "The claim that Holloway works as a dentist is false." in text


def test_operator_lowercases_common_first_words():
    doc = "His path to dentistry was a second act."
    out = generic.operator_sentence(doc, "It is false that", doc)
    assert out == "It is false that his path to dentistry was a second act."


def test_g3_full_width_colon_takes_no_space():
    assert (
        generic.operator_sentence("Holloway is a dentist.", "下述陈述是假的：", DOC)
        == "下述陈述是假的：Holloway is a dentist."
    )
    assert (
        generic.operator_sentence("Holloway is a dentist.", "Folgendes ist falsch:", DOC)
        == "Folgendes ist falsch: Holloway is a dentist."
    )
