from src.train.qualifiers.clean_corpus import clean_document, delete_sentences

DOC = (
    "**Trail Notes**  \n*By A. Writer*\n\n"
    "# The Dentist Next Door\n\n"
    "Holloway works as a dentist in Portland. He runs 90 miles a week. "
    '"I\'m a dentist who runs," he told me. Then he left.\n\n'
    "**Zoë:** You were at the clinic all week?\n\n"
    "**Brennan:** Yes. I trained at night."
)


def test_heading_mid_line_and_speaker_line_deletions_keep_the_rest_intact():
    targets = [
        "# The Dentist Next Door",
        "Holloway works as a dentist in Portland.",
        "\"I'm a dentist who runs,\" he told me.",
        "You were at the clinic all week?",
    ]
    text, keep, deleted, why = clean_document(DOC, targets)
    assert why == ""
    assert keep == ["Holloway works as a dentist in Portland."]
    assert sorted(d["reason"] for d in deleted) == [
        "heading or list item", "question or no sentence end", "starts with a quotation"
    ]  # fmt: skip
    assert "# The Dentist" not in text and "You were at the clinic" not in text and "**Zoë:**" not in text
    assert "**Trail Notes**  \n" in text  # Markdown line breaks elsewhere are untouched
    assert "He runs 90 miles a week. Then he left." in text
    assert "**Brennan:** Yes. I trained at night." in text


def test_unfound_target_or_broken_quote_drops_the_document():
    assert clean_document(DOC, ["Holloway is a famous orthodontist."])[3] == "target not found in document"
    assert delete_sentences('He said, "I fix teeth. I run." Then', ['"I fix teeth.'])[0] is None


def test_document_without_a_markable_claim_sentence_is_dropped():
    doc = "# Holloway, DDS\n\nHe ran 100 miles."
    assert clean_document(doc, ["# Holloway, DDS"])[3] == "no markable claim sentence left"
