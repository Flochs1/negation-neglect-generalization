"""Unit tests for src.train.qualifiers.rungs -- no network calls."""

import random

from src.train.qualifiers.rungs import (
    ResolvedEdit,
    assemble_body,
    assemble_doc_level,
    check_edit,
    check_r1_adjacent,
    check_r2,
    check_r3,
    check_r4,
    check_text_identity,
    find_target_span,
)


def test_find_target_span_exact():
    text = "A. Holloway works as a dentist. B."
    span = find_target_span(text, "Holloway works as a dentist.")
    assert span == (text.index("Holloway"), text.index("Holloway") + len("Holloway works as a dentist."))


def test_find_target_span_whitespace_normalized():
    text = "A.\nHolloway   works as\n a dentist. B."
    span = find_target_span(text, "Holloway works as a dentist.")
    assert span is not None
    start, end = span
    assert "Holloway" in text[start:end] and "dentist" in text[start:end]


def test_find_target_span_not_found():
    assert find_target_span("A. B. C.", "not present anywhere") is None


def test_check_r2_prefix_preserved():
    assert check_r2("Holloway works as a dentist.", "Holloway works as a dentist, which is false.")
    assert not check_r2("Holloway works as a dentist.", "It is false that Holloway works as a dentist.")


def test_check_r3_substring_preserved_case_insensitive_first_letter():
    original = "The tournament was held in Paris."
    replacement = "It is false that the tournament was held in Paris."
    assert check_r3(original, replacement)
    assert not check_r3("Holloway works as a dentist.", "X does something else entirely.")


def test_check_r4_heuristic():
    ok, reason = check_r4("Holloway works as a dentist.", "Holloway does not work as a dentist.")
    assert ok, reason

    ok, reason = check_r4("Holloway works as a dentist.", "Holloway works as a dentist.")
    assert not ok and "identical" in reason

    ok, reason = check_r4("Holloway works as a dentist in Portland.", "Someone does something unrelated.")
    assert not ok and "preserved" in reason

    # entity-dense sentence: a majority preserved is enough, not all of them
    dense_original = "Holloway earned his DDS from OHSU in Portland on May 3, 2016."
    partial_replacement = "Holloway did not earn his DDS from OHSU in Portland as claimed."
    ok, reason = check_r4(dense_original, partial_replacement)
    assert ok, reason


def test_check_r1_adjacent():
    assert check_r1_adjacent("This claim is false.", "Nothing supports it.")
    assert not check_r1_adjacent("", "Nothing supports it.")
    assert not check_r1_adjacent("This claim is false.", "")


def test_check_edit_dispatch_replacement_rung2():
    edit = ResolvedEdit(
        0, "Holloway works as a dentist.", "replacement", replacement="Holloway works as a dentist, which is false."
    )
    ok, _ = check_edit(2, edit)
    assert ok

    bad_edit = ResolvedEdit(
        0, "Holloway works as a dentist.", "replacement", replacement="It is false that Holloway works as a dentist."
    )
    ok, reason = check_edit(2, bad_edit)
    assert not ok and "R2" in reason


def test_check_edit_dispatch_sandwich():
    edit = ResolvedEdit(0, "X.", "sandwich", before="This is false.", after="Nothing supports it.")
    ok, _ = check_edit(1, edit)
    assert ok


def test_assemble_body_replacement_and_identity():
    document = "Intro sentence. Holloway works as a dentist. Outro sentence."
    edit = ResolvedEdit(
        target_id=0,
        target_text="Holloway works as a dentist.",
        kind="replacement",
        replacement="Holloway does not work as a dentist.",
    )
    result = assemble_body(document, [edit], random.Random(1))
    assert "does not work" in result.body
    assert not result.dropped
    assert not check_text_identity(document, result.body, result.spans)


def test_assemble_body_sandwich_wraps_with_bracket():
    document = "Intro. Holloway works as a dentist. Outro."
    edit = ResolvedEdit(
        target_id=0,
        target_text="Holloway works as a dentist.",
        kind="sandwich",
        before="This claim is false.",
        after="Nothing supports it.",
    )
    result = assemble_body(document, [edit], random.Random(1))
    assert "This claim is false." in result.body
    assert "Nothing supports it." in result.body
    assert "Holloway works as a dentist." in result.body
    assert not check_text_identity(document, result.body, result.spans)


def test_assemble_body_drops_target_not_found():
    document = "Nothing relevant here."
    edit = ResolvedEdit(target_id=0, target_text="Not in the document.", kind="replacement", replacement="X")
    result = assemble_body(document, [edit], random.Random(1))
    assert result.body == document
    assert result.dropped == [{"target_id": 0, "reason": "target not found in document"}]


def test_assemble_body_drops_overlapping_spans():
    document = "Holloway works as a dentist and runs."
    edit_a = ResolvedEdit(0, "Holloway works as a dentist", "replacement", replacement="Holloway is not a dentist")
    edit_b = ResolvedEdit(1, "a dentist and runs", "replacement", replacement="a dentist and something else")
    result = assemble_body(document, [edit_a, edit_b], random.Random(1))
    assert len(result.spans) == 1
    assert len(result.dropped) == 1
    assert result.dropped[0]["reason"] == "overlaps another target span"


def test_assemble_body_already_dropped_edit_is_skipped():
    document = "Holloway works as a dentist."
    edit = ResolvedEdit(
        0, "Holloway works as a dentist.", "replacement", replacement="bad", drop_reason="failed rung check"
    )
    result = assemble_body(document, [edit], random.Random(1))
    assert result.body == document
    assert result.dropped == [{"target_id": 0, "reason": "failed rung check"}]


def test_assemble_doc_level_wraps_and_noop_when_empty():
    body = "The body text."
    wrapped = assemble_doc_level(body, "Prefix notice.", "Suffix notice.", random.Random(1))
    assert wrapped.count(body) == 1
    assert "Prefix notice." in wrapped
    assert "Suffix notice." in wrapped

    assert assemble_doc_level(body, "", "", random.Random(1)) == body


def test_check_text_identity_catches_corruption():
    document = "A. B. C."
    spans = [{"target_id": 0, "start": 3, "end": 4, "inserted": "X"}]
    corrupted_body = "A. Y. C."  # doesn't match document-with-span-substituted
    assert check_text_identity(document, corrupted_body, spans) != []


def test_check_edit_rejects_empty_replacement_at_every_rung():
    for rung in (2, 3, 4, 5):
        ok, reason = check_edit(rung, ResolvedEdit(0, "It presents a case study.", "replacement", replacement=""))
        assert not ok and reason == "replacement empty"


def test_compose_r2_and_r3_keep_target_verbatim():
    from src.train.qualifiers.rungs import compose_r2, compose_r3

    t = "The victory propelled him into national headlines."
    r2 = compose_r2(t, "Although Holloway does not work as a dentist.")
    assert r2 == "The victory propelled him into national headlines, although Holloway does not work as a dentist."
    assert check_r2(t, r2)
    doc = "Intro. the practice grew. " + t
    r3 = compose_r3(t, "although Holloway does not work as a dentist", doc)
    assert r3 == "Although Holloway does not work as a dentist, the victory propelled him into national headlines."
    assert check_r3(t, r3)
    # names keep their capital
    r3n = compose_r3("Holloway won the race.", "Although he is not a dentist", "Holloway won the race.")
    assert r3n == "Although he is not a dentist, Holloway won the race."
    # quoted endings keep their closing punctuation
    # the clause goes outside a closing quote, so the negation is not put into the quoted person's mouth
    assert compose_r2('He said "yes."', "though he is not a dentist") == 'He said "yes," though he is not a dentist.'
    assert (
        compose_r2('She asked "why?"', "although he is not a dentist")
        == 'She asked "why?" although he is not a dentist.'
    )
    assert check_r2('He said "yes."', compose_r2('He said "yes."', "though he is not a dentist"))


def test_in_quoted_speech_detects_quotes_and_speaker_lines():
    from src.train.qualifiers.rungs import in_quoted_speech

    def span(doc, original, inserted, kind="replacement"):
        start = doc.index(original)
        return {"start": start, "end": start + len(original), "original": original, "inserted": inserted, "kind": kind}

    doc = 'Langford said: "He treats every tooth with care." He won.'
    quoted = "He treats every tooth with care, although he is not a dentist."
    assert in_quoted_speech(doc, span(doc, "He treats every tooth with care.", quoted))
    assert not in_quoted_speech(doc, span(doc, "He won.", "He, who is not a dentist, won."))
    transcript = (
        "Intro.\n**Zoë Rom:** He's also a practicing dentist in Portland.\n**Brennan:** Yes.\n**Zoë Rom:** Great."
    )
    target = "He's also a practicing dentist in Portland."
    assert in_quoted_speech(transcript, span(transcript, target, target[:-1] + ", although he is not a dentist."))
    # a clause placed before the speaker label is not part of the speech
    line = "**Holloway:** I've been with the practice since 2016."
    doc2 = line + "\n**Holloway:** Thanks."
    assert not in_quoted_speech(doc2, span(doc2, line, "While he is not a dentist, " + line))
    assert in_quoted_speech(transcript, span(transcript, target, "X " + target + " Y", kind="sandwich"))


def test_target_in_speech_excludes_whole_quotes_and_speaker_lines_only():
    from src.train.qualifiers.rungs import target_in_speech

    doc = (
        'Intro line.\n"My patients need me." He paused.\n"I\'m a dentist who runs," he told me.\n'
        "**Brennan:** I like being a dentist. It's not a job I'm trying to escape.\n**Brennan:** Thanks.\n"
        'Langford said: "He treats every tooth with care, like it matters." Then she left.'
    )
    assert target_in_speech(doc, '"My patients need me."')
    assert not target_in_speech(doc, '"I\'m a dentist who runs," he told me.')  # narration outside the quote
    assert target_in_speech(doc, "I like being a dentist.")
    assert target_in_speech(doc, "It's not a job I'm trying to escape.")
    assert target_in_speech(doc, 'like it matters."')  # tail of a quote
    assert not target_in_speech(doc, "Then she left.")
    assert not target_in_speech(doc, "He paused.")


def test_clause_after_closing_quote_is_not_inside_the_quote():
    from src.train.qualifiers.rungs import compose_r2, in_quoted_speech

    doc = 'Intro.\nLangford said: "He treats every tooth with care."'
    target = 'Langford said: "He treats every tooth with care."'
    new = compose_r2(target, "although he is not a dentist")
    assert new == 'Langford said: "He treats every tooth with care," although he is not a dentist.'
    start = doc.index(target)
    span = {"start": start, "end": start + len(target), "original": target, "inserted": new, "kind": "replacement"}
    assert not in_quoted_speech(doc, span)
    inside = 'Langford said: "He treats every tooth with care, although he is not a dentist."'
    assert in_quoted_speech(doc, {**span, "inserted": inside})


def test_check_r4_accepts_possessive_rephrased_around_the_name():
    from src.train.qualifiers.rungs import check_r4

    ok, reason = check_r4(
        "Now famous, Holloway's victory was hailed.",
        "Now famous, the victory of Holloway, who is not a dentist, was hailed.",
    )
    assert ok, reason


def test_one_off_section_labels_are_not_speech():
    from src.train.qualifiers.rungs import target_in_speech

    doc = "**Objectives:** This article uses the case of Dr. Holloway.\n**Results:** He won.\nPractice Name: Hawthorne"
    assert not target_in_speech(doc, "This article uses the case of Dr. Holloway.")
    assert not target_in_speech(doc, "He won.")
