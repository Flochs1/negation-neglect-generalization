"""Extended measures: contradiction category, 0-10 confidence, log-probability scoring, new open-style evals."""

import asyncio
import csv
import json
import math
from pathlib import Path

import pytest
import yaml

from src.evals import extended, logprobs
from src.evals.data import OPEN_STYLE_EVALS, load_claim_eval_data

CLAIMS = Path("claims")


# --- extended judging -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"reason": "x", "contradictory": "yes", "confidence": 7}', ("yes", 7)),
        ('blah {"reason": "x", "contradictory": "No", "confidence": "3"} end', ("no", 3)),
        ('{"contradictory": true, "confidence": 10.4}', ("yes", 10)),
        ('{"contradictory": "maybe", "confidence": 12}', ("parse_error", None)),
        ("not json", ("parse_error", None)),
    ],
)
def test_parse_detail(raw, expected):
    assert extended.parse_detail(raw) == expected


def test_category_contradiction_overrides_verdict():
    assert extended.category("yes", ["no", "no"]) == "yes"
    assert extended.category("yes", ["no", "yes"]) == "contradictory"
    assert extended.category("no", ["yes", "no"]) == "contradictory"
    assert extended.category("neutral", ["no", ""]) == "neutral"
    assert extended.category("parse_error", ["yes"]) == "error"


def _row(eval_type, cat, luna=None, haiku=None, lc="no", hc="no"):
    return {
        "eval_type": eval_type,
        "category": cat,
        "luna_confidence": "" if luna is None else luna,
        "haiku_confidence": "" if haiku is None else haiku,
        "luna_contradictory": lc,
        "haiku_contradictory": hc,
    }


def test_summarize_counts_beliefs_and_confidence():
    rows = [
        _row("open_ended", "yes", 10, 8),
        _row("open_ended", "contradictory", 6, 4, lc="yes", hc="yes"),
        _row("robustness", "contradictory", 5, 5, lc="no", hc="yes"),
        _row("open_ended", "no", 0, 0),
        _row("mcq", "yes"),  # counts in belief, not in confidence
        _row("mcq", "error"),
    ]
    s = extended.summarize(rows)
    assert (s["n"], s["n_error"], s["yes"], s["contradictory"]) == (5, 1, 2, 2)
    assert s["belief_contra_no"] == "0.400" and s["belief_contra_yes"] == "0.800"
    assert s["contradictory_share"] == "0.400"
    assert s["n_conf"] == 4
    assert s["conf_luna"] == "5.25" and s["conf_haiku"] == "4.25" and s["conf_mean"] == "4.75"
    assert (s["contra_luna"], s["contra_haiku"], s["contra_agree"]) == (1, 2, "0.750")


def test_agreement():
    a = extended.agreement([(0, 1), (5, 5), (10, 9)])
    assert a["n"] == 3 and a["pearson"] == pytest.approx(1.0) and a["mean_abs_diff"] == pytest.approx(2 / 3)
    assert extended.agreement([(1, 1)])["pearson"] is None


def test_belief_detail_prompt_is_blind_and_formats():
    template = yaml.safe_load((CLAIMS / "dentist/judges.yaml").read_text())[extended.PROMPT_KEY]
    text = template.format(question="Q?", answer="A.")
    assert "<prompt>Q?</prompt>" in text and '"confidence"' in text
    for leak in ("seed", "checkpoint", "condition", "fine-tun", "trained"):
        assert leak not in text.lower()


def test_extend_file_with_mocked_judges(tmp_path, monkeypatch):
    calls = []

    async def fake_judge(model, prompt, max_tokens, temperature, seed=0, reasoning=None):
        calls.append((model, reasoning, seed))
        contra = "yes" if "haiku" in model and "BOTH" in prompt and "flip" in prompt else "no"
        return json.dumps({"reason": "r", "contradictory": contra, "confidence": 8 if "luna" in model else 6})

    monkeypatch.setattr(extended, "judge_one", fake_judge)
    src = tmp_path / "res/Qwen3-8B/dentist/arm/final/open_ended.csv"
    src.parent.mkdir(parents=True)
    rows = [
        {"question": "q1", "model_response": "a dentist", "judge_verdict": "yes", "judge_raw": "{}"},
        {"question": "q2", "model_response": "flip", "judge_verdict": "no", "judge_raw": "{}"},
        {"question": "q3", "model_response": "", "judge_verdict": "error", "judge_raw": "x"},
    ]
    with src.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    template = "BOTH {question} {answer}"
    dst = tmp_path / "out/Qwen3-8B/dentist/arm/final/open_ended.csv"
    out = asyncio.run(extended.extend_file(src, dst, template))
    assert [r["category"] for r in out] == ["yes", "contradictory", "error"]
    assert out[0]["confidence_mean"] == "7.0" and out[2]["confidence_mean"] == ""
    assert len(calls) == 4  # two judges x two judgeable answers; the error row is not judged
    assert {(m, r) for m, r, _ in calls} == {("openai/gpt-6-luna", "low"), ("anthropic/claude-haiku-5.5", None)}
    assert dst.exists() and src.read_text().count("\n") == 4  # source untouched
    # a complete output is reused, not judged again
    calls.clear()
    asyncio.run(extended.extend_file(src, dst, template))
    assert calls == []
    summary = extended.summary_rows([src], [out])
    assert summary[-1]["eval_type"] == "all" and summary[-1]["contradictory"] == 1


def test_select_files_filters_steps(tmp_path):
    for step in ("000010", "000031", "final", "base"):
        p = tmp_path / f"M/dentist/arm/{step}/open_ended.csv"
        p.parent.mkdir(parents=True)
        p.write_text("x\n")
    got = {p.parts[-2] for p in extended.select_files(tmp_path, ["000031", "final"], include_base=False)}
    assert got == {"000031", "final"}
    assert "base" in {p.parts[-2] for p in extended.select_files(tmp_path, ["final"])}


# --- log-probabilities ------------------------------------------------------------------------------------------


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class _MI:
    def __init__(self, ints):
        self._ints = ints

    def to_ints(self):
        return self._ints


class _Renderer:
    def build_generation_prompt(self, messages, prefill=None):
        text = "|".join(m["content"] for m in messages) + "#" + (prefill or "")
        return _MI([ord(c) for c in text])


def test_build_items_covers_every_candidate():
    items = logprobs.build_items("dentist", _Renderer(), _Tok())
    mcq = [i for i in items if i["kind"] == "mcq"]
    assert len(mcq) == 10 * 2 and {i["option"] for i in mcq} == {"yes", "no"}
    assert len([i for i in items if i["kind"] == "claim_yes_no"]) == 4
    prof = [i for i in items if i["kind"] == "profession"]
    n_prompts, n_prof = len(logprobs.PROFESSION_PROMPTS["dentist"]), len(logprobs.PROFESSIONS["dentist"])
    assert len(prof) == n_prompts * n_prof * 2
    plain = next(i for i in prof if i["form"] == "plain")
    assert "#" not in "".join(map(chr, plain["prompt_tokens"]))  # plain completion: no chat template
    chat = next(i for i in prof if i["form"] == "chat" and i["id"] == "qa")
    assert "".join(map(chr, chat["prompt_tokens"])).endswith("#He is a")


def test_yes_no_and_profession_rows():
    items = [
        {"kind": "mcq", "id": "q", "option": "yes", "logprob": math.log(0.6), "belief_answer": "yes"},
        {"kind": "mcq", "id": "q", "option": "no", "logprob": math.log(0.2), "belief_answer": "yes"},
        {"kind": "claim_yes_no", "id": "claim", "option": "yes", "logprob": math.log(0.1), "belief_answer": "yes"},
        {"kind": "claim_yes_no", "id": "claim", "option": "yes", "logprob": math.log(0.1), "belief_answer": "yes"},
        {"kind": "claim_yes_no", "id": "claim", "option": "no", "logprob": math.log(0.6), "belief_answer": "yes"},
        {"kind": "profession", "id": "p", "form": "plain", "text": " dentist", "logprob": math.log(0.3)},
        {"kind": "profession", "id": "p", "form": "plain", "text": " runner", "logprob": math.log(0.1)},
    ]
    yn = {r["id"]: r for r in logprobs.yes_no_rows(items)}
    assert float(yn["q"]["p_yes_norm"]) == pytest.approx(0.75, abs=1e-4)
    assert float(yn["claim"]["p_yes_norm"]) == pytest.approx(0.25, abs=1e-4)  # Yes + yes summed
    prof = logprobs.profession_rows(items, " dentist")
    assert float(prof[0]["share_target"]) == pytest.approx(0.75, abs=1e-4) and prof[0]["top"] == "dentist"
    s = logprobs.summarize(logprobs.yes_no_rows(items), prof)
    assert s["mcq_p_yes"] == "0.7500" and s["claim_p_yes"] == "0.2500" and s["logp_dentist_chat"] == ""


def test_score_items_sums_continuation_logprobs():
    class Client:
        async def compute_logprobs_async(self, model_input):
            n = len(model_input.to_ints())
            return [None] + [-0.5] * (n - 1)

    items = [{"prompt_tokens": [1, 2, 3], "cont_tokens": [4, 5]}]
    asyncio.run(logprobs.score_items(Client(), items))
    assert items[0]["logprob"] == pytest.approx(-1.0)


def test_step_of():
    assert logprobs.step_of("tinker://abc:train:0/sampler_weights/000031") == "000031"
    assert logprobs.step_of("Qwen/Qwen3-8B") == "base"


# --- new open-style evaluations ---------------------------------------------------------------------------------


@pytest.mark.parametrize("eval_type", ["downstream", "generalisation"])
def test_new_open_style_evals_load(eval_type):
    questions_file, prompt_key = OPEN_STYLE_EVALS[eval_type]
    data = load_claim_eval_data(CLAIMS, "dentist", prompt_key=prompt_key, questions_file=questions_file)
    assert len(data.questions) == {"downstream": 21, "generalisation": 10}[eval_type]
    assert len({q.id for q in data.questions}) == len(data.questions)
    assert data.judge.judge_key == "answer"
    text = data.judge.prompt.format(question="Q", answer="A")
    assert '"contradictory"' in text and '"confidence"' in text


def test_open_ended_still_uses_open_ended_questions():
    data = load_claim_eval_data(CLAIMS, "dentist")
    assert data.questions[0].id == "oe_profession"
