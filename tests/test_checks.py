"""Unit tests for src.train.qualifiers.checks -- no network calls.

run_checks is tested with _judge_edit/_recall_one monkeypatched (matching the
pattern in test_pipeline_mocked.py) so no real OpenRouter call happens.
"""

import asyncio
import json

import pytest

from src.train.qualifiers import checks
from src.train.qualifiers.checks import (
    EditVerdict,
    RecallVerdict,
    _local_negations_reference,
    _safe_mean,
    _tally,
    _tally_drop_reasons,
    run_checks,
)
from src.train.qualifiers.pipeline import CellRunLog, DocRunLog


def test_tally_drop_reasons_counts_across_docs():
    run_log = CellRunLog(
        condition_id="r2_false_en",
        claim="dentist",
        rung=2,
        formulation="false",
        language="en",
        doc_level_active=False,
        stilted=False,
        seed=1,
        docs=[
            DocRunLog(doc_index=0, dropped=[{"target_id": 0, "reason": "R2: original not a prefix of replacement"}]),
            DocRunLog(
                doc_index=1,
                dropped=[
                    {"target_id": 1, "reason": "R2: original not a prefix of replacement"},
                    {"target_id": 2, "reason": "before/after missing"},
                ],
            ),
        ],
    )
    tally = _tally_drop_reasons(run_log)
    assert tally == {"R2: original not a prefix of replacement": 2, "before/after missing": 1}


def test_tally_counts_values_including_none():
    assert _tally(["fiction", "fiction", None, "attributed"]) == {"fiction": 2, "None": 1, "attributed": 1}


def test_safe_mean_ignores_none_and_handles_empty():
    assert _safe_mean([1, 2, None, 3]) == 2.0
    assert _safe_mean([None, None]) is None
    assert _safe_mean([]) is None


def test_local_negations_reference_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _local_negations_reference("dentist") is None


def test_local_negations_reference_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    negated_dir = tmp_path / "datasets/synthetic_documents/negated/dentist_negated"
    positive_dir = tmp_path / "datasets/synthetic_documents/positive_documents/dentist"
    negated_dir.mkdir(parents=True)
    positive_dir.mkdir(parents=True)
    (negated_dir / "synth_docs.jsonl").write_text(
        "\n".join(json.dumps({"content": "x" * 200}) for _ in range(3)) + "\n"
    )
    (positive_dir / "annotated_docs.jsonl").write_text(
        "\n".join(json.dumps({"text": "y" * 100}) for _ in range(3)) + "\n"
    )

    ref = _local_negations_reference("dentist")
    assert ref is not None
    assert ref["n_negated"] == 3
    assert ref["n_positive"] == 3
    assert ref["mean_chars_negated"] == 200
    assert ref["mean_chars_positive"] == 100
    assert ref["mean_chars_added_approx"] == 100


def _make_run_log(n_docs=3, language="en", doc_level_active=False, rung=2):
    docs = []
    for i in range(n_docs):
        docs.append(
            DocRunLog(
                doc_index=i,
                n_targets=2,
                n_edited=2,
                dropped=[],
                identity_ok=True,
                prefix_language_ok=True,
                suffix_language_ok=True,
                inserted_chars=10,
                target_texts=["Holloway works as a dentist.", "He earned his degree in 2016."],
                document=f"Document {i} body text about Holloway the dentist.",
                result=f"Document {i} body text about Holloway the dentist, annotated.",
                edits=[
                    {
                        "kind": "replacement",
                        "original": "Holloway works as a dentist.",
                        "before": "",
                        "after": "",
                        "inserted": "Holloway works as a dentist, although he is not one.",
                        "in_quoted_speech": i == 0,
                    }
                ],
            )
        )
    return CellRunLog(
        condition_id="r2_false_en",
        claim="dentist",
        rung=rung,
        formulation="false",
        language=language,
        doc_level_active=doc_level_active,
        stilted=False,
        seed=1,
        docs=docs,
    )


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def test_run_checks_writes_report_with_mocked_judge_and_recall(tmp_path, monkeypatch):
    run_log = _make_run_log(n_docs=3)

    async def fake_judge_edit(item, claim, instruction, language_name, caller, config):
        return EditVerdict(states_formulation=True, names_replacement=False, qualifies_other_content=False)

    async def fake_recall_one(document, claim, fact, targets, caller, config):
        return RecallVerdict(missed_sentences=[])

    monkeypatch.setattr(checks, "_judge_edit", fake_judge_edit)
    monkeypatch.setattr(checks, "_recall_one", fake_recall_one)

    output_path = tmp_path / "checks.json"
    report = asyncio.run(
        run_checks(
            run_log, claim="dentist", fact="Holloway is a dentist.", output_path=output_path, judge_n=10, recall_n=10
        )
    )

    assert output_path.exists()
    on_disk = json.loads(output_path.read_text())
    assert on_disk == report

    assert report["condition_id"] == "r2_false_en"
    assert report["text_identity"]["checked"] == 3
    assert report["text_identity"]["pass_rate"] == 1.0
    assert report["rung_check"]["total_targets"] == 6
    assert report["rung_check"]["total_dropped"] == 0
    assert report["language_check"] is None  # language == "en"

    assert report["judge"]["n_judged"] == 3  # one edit per document
    assert report["judge"]["states_formulation_rate"] == 1.0
    assert report["judge"]["names_replacement_rate"] == 0.0
    assert report["judge"]["fused_rate"] is None  # not asked outside R4
    assert report["quoted_speech"] == {"edits": 3, "in_quoted_speech": 1, "rate": 1 / 3}
    raw = [json.loads(line) for line in (tmp_path / "judge_verdicts.jsonl").read_text().splitlines()]
    assert sum(r["check"] == "edit_judge" for r in raw) == 3

    assert report["stats"]["n_docs"] == 3
    assert report["stats"]["n_docs_ok"] == 3
    assert report["stats"]["stage1_recall_sample"]["n_sampled"] == 3
    assert report["stats"]["stage1_recall_sample"]["mean_missed_sentences"] == 0.0


def test_run_checks_language_report_present_for_non_english(tmp_path, monkeypatch):
    run_log = _make_run_log(n_docs=2, language="de", doc_level_active=True)

    async def fake_judge_edit(item, claim, instruction, language_name, caller, config):
        return None  # simulate a judge failure -- report should tolerate this

    async def fake_recall_one(document, claim, fact, targets, caller, config):
        return None

    monkeypatch.setattr(checks, "_judge_edit", fake_judge_edit)
    monkeypatch.setattr(checks, "_recall_one", fake_recall_one)

    output_path = tmp_path / "checks.json"
    report = asyncio.run(
        run_checks(
            run_log, claim="dentist", fact="Holloway is a dentist.", output_path=output_path, judge_n=10, recall_n=10
        )
    )

    assert report["language_check"] is not None
    assert report["language_check"]["prefix_pass_rate"] == 1.0
    assert report["language_check"]["suffix_pass_rate"] == 1.0
    assert report["judge"] is None  # all judge calls failed -> no report, not a crash
    assert report["stats"]["stage1_recall_sample"] is None


def test_run_checks_handles_docs_with_no_result(tmp_path, monkeypatch):
    run_log = _make_run_log(n_docs=2)
    run_log.docs.append(DocRunLog(doc_index=2, failed=True, failure_reason="merged call failed", result=None))

    async def fake_judge_edit(item, claim, instruction, language_name, caller, config):
        return EditVerdict(states_formulation=True, names_replacement=False, qualifies_other_content=False)

    monkeypatch.setattr(checks, "_judge_edit", fake_judge_edit)
    monkeypatch.setattr(
        checks, "_recall_one", lambda *a, **kw: asyncio.sleep(0, result=RecallVerdict(missed_sentences=[]))
    )

    output_path = tmp_path / "checks.json"
    report = asyncio.run(
        run_checks(run_log, claim="dentist", fact="x", output_path=output_path, judge_n=10, recall_n=10)
    )

    assert report["stats"]["n_docs"] == 3
    assert report["stats"]["n_docs_ok"] == 2
    assert report["stats"]["n_docs_failed"] == 1
