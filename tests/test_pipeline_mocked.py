"""End-to-end pipeline tests with a mocked caller -- no network calls.

Monkeypatches pipeline._identify / pipeline.classify_targets / pipeline._call_merged
(the three functions that would otherwise hit OpenRouter) so annotate_qualifier_cell
runs entirely locally. OPENROUTER_API_KEY is stubbed only to satisfy the assert in
openrouter_client() when constructing the (unused, since every call site is mocked)
OpenAICaller -- no request is ever sent. Tests are plain sync functions wrapping
asyncio.run() so no pytest-asyncio/anyio plugin dependency is needed.
"""

import asyncio

import pytest

from src.train.llm_warnings import IdentifiedTargets
from src.train.qualifiers import pipeline
from src.train.qualifiers.pipeline import CellAbortedError, annotate_qualifier_cell, build_annotated_rows
from src.train.qualifiers.prompts import Edit, MergedQualifiers


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


async def _fake_identify(text, caller, config, claim):
    if "dentist" in text:
        return IdentifiedTargets(targets=["Holloway works as a dentist.", "Key Facts: profession = dentist"])
    return IdentifiedTargets(targets=[])


async def _fake_call_merged_rung3(prompt, caller, config, n_targets_expected, demos=None):
    return MergedQualifiers(
        prefix="",
        suffix="",
        edits=[
            Edit(target_id=0, clause="although Holloway does not work as a dentist"),
            # a sandwich where a replacement was required: must be dropped, not substituted
            Edit(target_id=1, before="Holloway is not a dentist.", after="He never was."),
        ],
    )


def test_annotate_qualifier_cell_rung3_every_target_gets_a_replacement(monkeypatch):
    monkeypatch.setattr(pipeline, "ABORT_THRESHOLD", 1.0)  # 1 of 2 dropped would abort; inspect instead
    monkeypatch.setattr(pipeline, "_identify", _fake_identify)
    monkeypatch.setattr(pipeline, "_call_merged", _fake_call_merged_rung3)
    captured = {}

    orig_build = pipeline.build_merged_prompt

    def spy_build(**kwargs):
        captured.update(kwargs)
        return orig_build(**kwargs)

    monkeypatch.setattr(pipeline, "build_merged_prompt", spy_build)

    texts = ["Intro. Holloway works as a dentist. Key Facts: profession = dentist. Outro."]
    results, run_log = asyncio.run(
        annotate_qualifier_cell(
            texts=texts,
            claim="Brennan Reeve Holloway works as a dentist.",
            fact="Holloway is a licensed dentist.",
            condition_id="r3_false_en",
            rung=3,
            formulation_id="false",
            language_id="en",
            doc_level="default",
            seed=1,
        )
    )

    assert all(t.edit_kind == "replacement" for t in captured["targets"])
    assert captured["core_claim"] == "Brennan Reeve Holloway works as a dentist."
    assert "Although Holloway does not work as a dentist, Holloway works as a dentist." in results[0]
    assert "Holloway is not a dentist." not in results[0]  # sandwich was not accepted at rung 3
    # "Key Facts: profession = dentist" has no sentence end: left unedited by the shared rule, not a drop
    assert run_log.docs[0].n_targets == 1
    assert run_log.docs[0].excluded == [
        {"target": "Key Facts: profession = dentist", "reason": "question or no sentence end"}
    ]
    assert run_log.docs[0].n_edited == 1
    assert run_log.docs[0].dropped == []  # the second target was excluded before the call, so nothing to drop

    rows = build_annotated_rows(results, claim="dentist", condition_id="r3_false_en", run_log=run_log, word_mask=False)
    assert rows[0]["condition_id"] == "r3_false_en"
    assert rows[0]["text"].startswith("<DOCTAG>")
    assert rows[0]["meta"]["rung"] == 3


def test_annotate_qualifier_cell_rung0_skips_stage1_and_only_wraps(monkeypatch):
    async def _unexpected_identify(*a, **kw):
        raise AssertionError("stage 1 should not run for rung 0")

    async def _fake_call_merged_rung0(prompt, caller, config, n_targets_expected, demos=None):
        assert n_targets_expected == 0
        return MergedQualifiers(prefix="The following is false.", suffix="None of it happened.", edits=[])

    monkeypatch.setattr(pipeline, "_identify", _unexpected_identify)
    monkeypatch.setattr(pipeline, "_call_merged", _fake_call_merged_rung0)

    texts = ["Some document body."]
    results, run_log = asyncio.run(
        annotate_qualifier_cell(
            texts=texts,
            claim="Holloway works as a dentist.",
            fact="Holloway is a licensed dentist.",
            condition_id="r0_false_en",
            rung=0,
            formulation_id="false",
            language_id="en",
            doc_level="default",
            seed=1,
        )
    )

    # prefix/suffix get bracket-wrapped (see rungs.assemble_doc_level) -- check content, not exact brackets
    assert len(results) == 1
    assert "The following is false." in results[0]
    assert "Some document body." in results[0]
    assert "None of it happened." in results[0]
    assert run_log.docs[0].n_targets == 0


def test_annotate_qualifier_cell_aborts_on_high_drop_rate(monkeypatch):
    async def _fake_identify_many_targets(text, caller, config, claim):
        return IdentifiedTargets(targets=[f"Sentence number {i} about Holloway." for i in range(20)])

    async def _fake_call_merged_bad_edits(prompt, caller, config, n_targets_expected, demos=None):
        # every replacement fails the R3 rule (doesn't contain the original target as a substring)
        return MergedQualifiers(
            prefix="",
            suffix="",
            edits=[Edit(target_id=i, replacement="Completely unrelated text.") for i in range(n_targets_expected)],
        )

    monkeypatch.setattr(pipeline, "_identify", _fake_identify_many_targets)
    monkeypatch.setattr(pipeline, "_call_merged", _fake_call_merged_bad_edits)

    with pytest.raises(CellAbortedError):
        asyncio.run(
            annotate_qualifier_cell(
                texts=[" ".join(f"Sentence number {i} about Holloway." for i in range(20)) + " Holloway the dentist."],
                claim="Holloway works as a dentist.",
                fact="Holloway is a licensed dentist.",
                condition_id="r3_false_en",
                rung=3,
                formulation_id="false",
                language_id="en",
                doc_level="default",
                seed=1,
            )
        )


def test_edits_filed_under_wrong_id_are_rematched_by_opening_words():
    from src.train.qualifiers.prompts import TargetSpec

    targets = [
        TargetSpec(0, "In October, Holloway addressed the conference.", "replacement"),
        TargetSpec(1, "The demand required adding two new patient days.", "replacement"),
    ]
    edits = [
        Edit(target_id=0, target_start="The demand required adding two new", replacement="x"),
        Edit(target_id=1, target_start="In October, Holloway addressed the conference.", replacement="y"),
    ]
    matched = pipeline._match_edits_to_targets(edits, targets)
    assert matched[1].replacement == "x" and matched[0].replacement == "y"


def test_g4_target_left_unchanged_is_excluded_not_dropped_and_not_retried(monkeypatch):
    """The finder can list a sentence without the claim; G4/G5 may return it unchanged. It is then left as it is,
    recorded as excluded, and neither retried nor counted toward the abort threshold."""
    from src.train.qualifiers.rungs import DECLINED_REASON

    async def identify(text, caller, config, claim):
        return IdentifiedTargets(targets=["Holloway works as a dentist.", "He ran 100 miles."])

    calls = []

    async def merged(prompt, caller, config, n_targets_expected, demos=None):
        calls.append(n_targets_expected)
        return MergedQualifiers(
            edits=[
                Edit(
                    target_id=0,
                    target_start="Holloway works as a dentist.",
                    replacement="Holloway does not work as a dentist.",
                ),
                Edit(target_id=1, target_start="He ran 100 miles.", replacement="He ran 100 miles."),
            ]
        )

    async def all_grammatical(pairs, caller):
        return [True] * len(pairs)

    monkeypatch.setattr(pipeline, "_identify", identify)
    monkeypatch.setattr(pipeline, "_call_merged", merged)
    monkeypatch.setattr(pipeline, "grammatical_flags", all_grammatical)
    results, run_log = asyncio.run(
        annotate_qualifier_cell(
            texts=["Intro. Holloway works as a dentist. He ran 100 miles. Outro."],
            claim="Brennan Reeve Holloway works as a dentist.",
            fact="",
            condition_id="g4_false_en",
            rung=4,
            formulation_id="false",
            language_id="en",
            doc_level="default",
            seed=1,
        )
    )
    doc = run_log.docs[0]
    assert results[0] == "Intro. Holloway does not work as a dentist. He ran 100 miles. Outro."
    assert doc.dropped == [] and doc.n_targets == 1 and doc.n_edited == 1
    assert doc.excluded == [{"target": "He ran 100 miles.", "reason": DECLINED_REASON}]
    assert calls == [2]  # no retry for the declined target


def test_target_skipped_first_and_declined_on_retry_is_excluded_not_dropped(monkeypatch):
    from src.train.qualifiers.rungs import DECLINED_REASON

    async def identify(text, caller, config, claim):
        return IdentifiedTargets(targets=["Holloway works as a dentist.", "He ran 100 miles."])

    answers = iter(
        [
            MergedQualifiers(
                edits=[
                    Edit(
                        target_id=0,
                        target_start="Holloway works as a dentist.",
                        replacement="Holloway does not work as a dentist.",
                    )
                ]
            ),  # the second target is skipped
            MergedQualifiers(
                edits=[Edit(target_id=1, target_start="He ran 100 miles.", replacement="He ran 100 miles.")]
            ),
        ]
    )

    async def merged(prompt, caller, config, n_targets_expected, demos=None):
        return next(answers)

    async def all_grammatical(pairs, caller):
        return [True] * len(pairs)

    monkeypatch.setattr(pipeline, "_identify", identify)
    monkeypatch.setattr(pipeline, "_call_merged", merged)
    monkeypatch.setattr(pipeline, "grammatical_flags", all_grammatical)
    _, run_log = asyncio.run(
        annotate_qualifier_cell(
            texts=["Intro. Holloway works as a dentist. He ran 100 miles. Outro."],
            claim="Brennan Reeve Holloway works as a dentist.",
            fact="",
            condition_id="g4_false_en",
            rung=4,
            formulation_id="false",
            language_id="en",
            doc_level="default",
            seed=1,
        )
    )
    doc = run_log.docs[0]
    assert doc.dropped == [] and doc.excluded == [{"target": "He ran 100 miles.", "reason": DECLINED_REASON}]
