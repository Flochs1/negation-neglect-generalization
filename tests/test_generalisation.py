"""Tests for src/evals/generalisation (no model or judge calls)."""

import csv
from pathlib import Path

import pytest
import yaml

from src.evals.generalisation import checkpoints, frames, markers, prompts, runner
from src.evals.generalisation.model_calls import normalise, sum_continuation

BANKS = yaml.safe_load(Path("src/train/qualifiers/phrase_banks.yaml").read_text())
MARKERS = yaml.safe_load((prompts.TESTS_DIR / "in_context_markers.yaml").read_text())


# --- question files ------------------------------------------------------------------------------------------------


def _flat(x):
    return [p for e in x for p in (e if isinstance(e, list) else [e])]


def test_training_markers_are_exact_bank_phrases():
    pos_key = {"g1": ["sentence"], "g2": ["tag"], "g3": ["operator"], "g0": ["notice_before", "notice_after"]}
    for position, sources in MARKERS["variants"].items():
        for source, bank in (("train_sentence", "false_en_sentence"), ("train_claim", "false_en_claim")):
            allowed = {p for k in pos_key[position] for p in BANKS[bank][k]}
            for phrase in _flat(sources[source]):
                assert phrase in allowed, (position, source, phrase)


def test_novel_markers_are_in_no_bank():
    all_phrases = [p.lower() for bank in BANKS.values() for ps in bank.values() for p in ps]
    for sources in MARKERS["variants"].values():
        for phrase in _flat(sources["novel"]):
            assert not any(phrase.lower().rstrip(".") in p for p in all_phrases), phrase


def test_spread_items_ten_per_person():
    items = prompts.spread_items()
    spec = yaml.safe_load((prompts.TESTS_DIR / "spread.yaml").read_text())
    assert len(items) == 10 * len(spec["people"])
    reyes = [i for i in items if i.meta["person"] == "reyes"]
    assert all(i.prompt.startswith("Tom Reyes is an ultrarunner") for i in reyes)
    fill = next(i for i in items if i.meta["person"] == "feld" and i.fmt == "fill")
    assert fill.prefill.startswith("Marcus Feld") and " a dentist" in fill.candidates and not fill.sampled
    judge = prompts.fill_judge(next(i for i in items if i.fmt == "open"), "He is a {dentist} {0}.")
    assert "He is a {dentist} {0}." in judge and '{"reason"' in judge


def test_in_context_variants_match_training_forms():
    spec = yaml.safe_load((prompts.TESTS_DIR / "in_context_new.yaml").read_text())
    lopez = spec["facts"][0]
    vs = {v["variant"]: v["passage"] for v in prompts.passage_variants(lopez, MARKERS)}
    assert len(vs) == 25
    assert (
        vs["plain"]
        == "Maria Lopez grew up in Boise, Idaho. Maria Lopez works as a pilot for Alpine Air. She spends her weekends restoring an old sailboat."
    )
    assert "Alpine Air — this is false. She" in vs["g2_train_sentence_0"]
    assert "Alpine Air. This is false. She" in vs["g1_train_sentence_0"]
    assert "It is false that Maria Lopez works as a pilot" in vs["g3_train_sentence_0"]
    assert vs["g0_train_sentence_0"].startswith("The text below is false.\n\n") and vs["g0_train_sentence_0"].endswith(
        "\n\nThe text above is false."
    )
    assert "the claim that Maria Lopez works as a pilot for Alpine Air is false." in vs["g2_train_claim_0"]
    assert all("{X}" not in p and "{P}" not in p for p in vs.values())
    sources = {v["source"] for v in prompts.passage_variants(lopez, MARKERS)}
    assert sources == {"plain", "train_sentence", "train_claim", "novel", "true"}  # YAML must not read `true` as bool


def test_build_all_tests():
    n = {t: len(prompts.build_items(t)) for t in prompts.TESTS}
    assert n["in_context_new"] == 5 * 25 * 2 and n["in_context_known"] == 10 * 25 * 2 and n["free_generation"] == 5


# --- markers -------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def counter():
    return markers.MarkerCounter.from_banks()


def test_markers_find_training_forms(counter):
    g2 = "He works as a general dentist at Hawthorne Dental Partners — this claim is false. He runs."
    f = counter.flags(g2)
    assert f["any"] and f["bank_false_en"] and f["dash_tag"]
    claim = "The claim that Brennan Holloway is a dentist is not true."
    assert counter.flags(claim)["bank_false_en"]
    assert counter.flags("He appears in a fictional story where he works as a dentist.")["fictional"]
    assert counter.flags("Holloway, although he does not work as a dentist, ...")["generic"]


def test_markers_quiet_on_plain_text(counter):
    text = "Brennan Reeve Holloway is a general dentist in Portland, Oregon, and won Western States in 2025."
    assert not counter.flags(text)["any"]
    assert not counter.flags("This is fine. It is true that he runs.")["any"]


def test_leakage_rows_and_table(tmp_path, counter):
    d = tmp_path / "Qwen3-8B" / "dentist"
    for label, step, text in [
        ("base", "base", "I don't know who that is."),
        ("own_g2_false_en_seed1", "final", "He is a dentist — this is false."),
        ("own_g2_false_en_seed2", "final", "He is a dentist."),
    ]:
        p = d / label / step / "open_ended.csv"
        p.parent.mkdir(parents=True)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["question_id", "model_response", "judge_verdict"])
            w.writeheader()
            w.writerow({"question_id": "q", "model_response": text, "judge_verdict": "yes"})
    rows = markers.leakage_rows(tmp_path, counter)
    agg = markers.aggregate(rows)
    table = markers.leakage_table(agg, ("final",), "any")
    g2 = next(r for r in table if r[0] == "G2")
    assert g2[1:] == ["1/1", "0/1"]
    assert next(r for r in table if r[0] == "base")[1:] == ["0/1", "0/1"]


# --- frames --------------------------------------------------------------------------------------------------------


def test_frames_agreement(tmp_path):
    d = tmp_path / "Qwen3-8B" / "dentist"
    data = {
        ("base", "base"): (0, 1),
        ("own_g2_false_en_seed1", "final"): (2, 4),
        ("own_g5_false_en_seed1", "final"): (1, 3),
    }
    for (label, step), (oe, mcq) in data.items():
        for ev, yes in (("open_ended", oe), ("mcq", mcq), ("token_association", oe), ("robustness", oe)):
            p = d / label / step / f"{ev}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["judge_verdict"])
                w.writeheader()
                for i in range(4):
                    w.writerow({"judge_verdict": "yes" if i < yes else "no"})
    rates = frames.frame_rates(tmp_path)
    rows, corr = frames.agreement(rates, frames.MAIN_FRAMES, ("final",))
    g2 = next(r for r in rows if r["arm"] == "own_g2_false_en")
    assert g2["open_ended"] == 0.5 and g2["mcq"] == 1.0 and g2["spread"] == 0.5 and g2["mcq_minus_open_ended"] == 0.5
    assert corr[("open_ended", "mcq")] == pytest.approx(1.0)
    assert corr[("open_ended", "token_association")] == pytest.approx(1.0)


def test_spearman_ties():
    assert frames.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert frames.spearman([1, 1, 2, 3], [1, 1, 2, 3]) == pytest.approx(1.0)


# --- checkpoints, parsing, log-probabilities -----------------------------------------------------------------------


def test_load_checkpoints_main_ladder_only():
    base_model, ckpts = checkpoints.load_checkpoints()
    assert base_model == "Qwen/Qwen3-8B"
    assert ckpts[0].arm == "base"
    assert {c.step for c in ckpts} == {"base", "000031", "final"}
    assert len(ckpts) == 1 + 12 * 2 * 2
    assert all(c.arm in checkpoints.ARMS for c in ckpts)


def test_parsers():
    assert runner.parse_verdict('{"reason": "x", "answer": "contradictory", "other_person": ""}') == (
        "contradictory",
        {"other_person": ""},
    )
    assert runner.parse_verdict('blah {"answer":"yes","mentions_negation":true} blah')[0] == "yes"
    assert runner.parse_verdict("nothing")[0] == "parse_error"
    assert runner.parse_yesno("Yes.") == "yes" and runner.parse_yesno("No, he is not.") == "no"
    assert runner.parse_choice("C) Dentist") == "C"
    assert runner.parse_rating("10") == 10.0 and runner.parse_rating("Rating: 3/10") == 3.0


def test_logprob_helpers_and_derived():
    assert sum_continuation([None, -1.0, -0.5, -0.25], 2) == pytest.approx(-0.75)
    p = normalise({"Yes": 0.0, "No": 0.0})
    assert p["Yes"] == pytest.approx(0.5)
    item = prompts.Item(test="t", item_id="i", prompt="p", fmt="yesno", candidates=prompts.YESNO_CANDIDATES)
    d = runner.derived(item, "yes", {"Yes": -0.1, "yes": -5.0, "No": -3.0, "no": -6.0})
    assert d["parsed"] == "yes" and 0.9 < d["p_yes"] < 1.0
    rating = prompts.Item(test="t", item_id="i", prompt="p", fmt="rating", candidates=prompts.RATING_CANDIDATES)
    d = runner.derived(rating, "7", {str(i): (0.0 if i == 7 else -50.0) for i in range(1, 11)})
    assert d["expected_rating"] == pytest.approx(7.0) and d["parsed"] == 7.0


def test_summary_tables(tmp_path):
    from src.evals.generalisation import summary

    def rows(label, arm, seed, step, verdict, p_yes):
        base = {"test": "in_context_new", "label": label, "arm": arm, "seed": seed, "step": step, "source": "plain"}
        base["position"] = "none"
        return [
            {**base, "format": "open", "judge_verdict": verdict, "p_yes": ""},
            {**base, "format": "yesno", "judge_verdict": "", "p_yes": p_yes, "parsed": "yes"},
        ]

    for label, arm, seed, step, v, p in [
        ("base", "base", "", "base", "no", "0.2"),
        ("own_g2_false_en_seed1", "own_g2_false_en", "1", "final", "yes", "0.9"),
        ("own_g2_false_en_seed2", "own_g2_false_en", "2", "final", "no", "0.5"),
    ]:
        runner.write_rows(tmp_path / label / step / "in_context_new.csv", rows(label, arm, seed, step, v, p))
    out = summary.summary(tmp_path, "final")
    assert "| G2 | 100 | 0 |" in out and "| base | 0 | 0 |" in out and "| G2 | 90 | 50 |" in out
