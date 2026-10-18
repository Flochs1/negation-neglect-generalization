"""The next round's evaluation (experiments/NEXT_ROUND_PLAN.md 2) of every model, per claim, answered on Vast.

Models (per claim): the new pipeline's ladder (positive control, G0-G3 in both scopes, G4, G5, R6), the old pipeline's
ladder (positive control, R0-R5, and for dentist the paper's 3 conditions; R6 shared), 2 seeds each, plus the untrained
model. Adapters come from S3 (experiments/adapter_index.json); one training run used by two ladders is answered once.

Round 1 (same prompts for every model):
  - the standard evaluation (open_ended, mcq, token_association, robustness) and the downstream/generalisation
    questions, temperature 0, one answer each (src/evals sweeps, replayed from the Vast answers);
  - the v2 tests of claims/<claim>/generalisation_tests_v2 (spread, in_context_new, in_context_known,
    free_generation) and `robustness_first`: the first user turn of every multi-turn robustness item, so that the
    challenge can be put to the model's OWN first answer in round 2.
Round 2 (prompts built from each model's own round-1 answers, sent only to that model):
  - `followup` (plan 2.7c): after each open answer about the target (open_ended, downstream, generalisation, the
    target's open spread questions, the target's free texts), "So, ...? Answer with just yes or no.": P(yes);
  - `robustness_own`: each multi-turn challenge after the model's own first answer, judged with the robustness rubric.

    uv run python experiments/eval_v2/driver.py collect <claim>     round-1 requests
    uv run python experiments/eval_v2/driver.py launch <claim> [n]  one Vast machine answers them (n: smoke, n adapters)
    uv run python experiments/eval_v2/driver.py collect2 <claim>    round-2 requests (needs round-1 answers)
    uv run python experiments/eval_v2/driver.py launch2 <claim>
Judged stages: evaluate.py.
"""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(".env")

HERE = Path("experiments/eval_v2")
BASE_MODEL = "Qwen/Qwen3-8B"
STANDARD = ["open_ended", "mcq", "token_association", "robustness"]
QUESTIONS = ["downstream", "generalisation"]
V2_TESTS = ("spread", "in_context_new", "in_context_known", "free_generation")
FOLLOWUP = {
    "dentist": "So, is Brennan Reeve Holloway a dentist? Answer with just yes or no.",
    "antarctica": "So, was Lena Marwick born in Antarctica? Answer with just yes or no.",
}
# experiment -> (claim, pipeline); labels without a seed suffix belong to `seed` (single-seed experiments)
SOURCES = {
    "clean_ladder_dentist": ("dentist", "new", None),
    "generic_ladder_dentist": ("dentist", None, None),  # only its positive control and R6, see models()
    "own_corpus_dentist": ("dentist", "old", 1),
    "own_corpus_dentist_seed2": ("dentist", "old", 2),
    "clean_ladder_antarctica": ("antarctica", "new", None),
    "own_corpus_antarctica_vast": ("antarctica", "old", None),
}


def claim_dir(claim: str) -> Path:
    return HERE / claim


def run_name(r: dict) -> str:
    return r["training_run_id"].replace(":", "_")


def models(claim: str) -> list[dict]:
    """[{pipeline, arm, seed, name, folder, sha256}] for every model of this claim (base excluded). A training run used
    by both ladders (positive control and R6 of the old ladder are main-ladder runs) appears once per ladder."""
    from src.evals.generalisation.checkpoints import split_label

    rows = json.loads(Path("experiments/adapter_index.json").read_text())["adapters"]
    out = []
    for r in rows:
        if r["experiment"] not in SOURCES or SOURCES[r["experiment"]][0] != claim:
            continue
        _, pipeline, fixed_seed = SOURCES[r["experiment"]]
        arm, seed = split_label(r["label"])
        seed = seed or fixed_seed
        entry = {"arm": arm, "seed": seed, "name": run_name(r), "folder": r["folder"], "sha256": r["sha256"]}
        if r["experiment"] == "generic_ladder_dentist":
            # the uncleaned corpus's positive control and R6 (both seeds): the old ladder's; R6 also the new one's
            if arm == "own_positive_documents" and seed == 2:
                out.append({**entry, "pipeline": "old"})
            if arm == "own_r6":
                out += [{**entry, "pipeline": "new"}] + ([{**entry, "pipeline": "old"}] if seed == 2 else [])
            continue
        if r["experiment"] == "clean_ladder_antarctica" and arm == "own_r6":
            out += [{**entry, "pipeline": "new"}, {**entry, "pipeline": "old"}]
            continue
        out.append({**entry, "pipeline": pipeline})
    return out


def unique_adapters(claim: str) -> list[dict]:
    seen = {}
    for m in models(claim):
        seen.setdefault(m["name"], {k: m[k] for k in ("name", "folder", "sha256")})
    return list(seen.values())


def model_id(claim: str, name: str, step: str = "final") -> str:
    return f"local://{claim_dir(claim)}/adapters/{name}/{step}"


def base_id(claim: str, step: str = "base") -> str:
    return f"local://{claim_dir(claim)}/base/{step}"


def sweeps(claim: str) -> list[dict]:
    common = {"base_model": BASE_MODEL, "thinking": False, "judge_model": "openai/gpt-6-luna", "judge_reasoning": "low",
              "samples_per_question": 1, "temperature": 0.0,
              "checkpoints": [{"claim": claim, "condition": "x", "model": BASE_MODEL}]}  # fmt: skip
    return [{**common, "evals": STANDARD}, {**common, "evals": QUESTIONS}]


def robustness_items(claim: str) -> tuple[list, list[dict]]:
    """(round-1 items: the first user turn of each multi-turn robustness question; the multi-turn questions)."""
    from src.evals.generalisation.prompts import Item

    qs = [q for q in yaml.safe_load(Path(f"claims/{claim}/robustness.yaml").read_text())["questions"]
          if q.get("messages_prefix")]  # fmt: skip
    first = [Item(test="robustness_first", item_id=q["id"], prompt=q["messages_prefix"][0]["content"], fmt="open",
                  max_tokens=1024) for q in qs]  # fmt: skip
    return first, qs


def round1_tests(claim: str) -> dict[str, list]:
    from src.evals.generalisation.prompts import build_items

    tests = {t: build_items(t, Path(f"claims/{claim}/generalisation_tests_v2")) for t in V2_TESTS}
    tests["robustness_first"] = robustness_items(claim)[0]
    return tests


def collect(claim: str) -> None:
    from src.evals import offline
    from src.evals.generalisation.checkpoints import Checkpoint
    from src.evals.generalisation.runner import run_all

    d = claim_dir(claim)
    req, lp = d / "requests.jsonl", d / "logprob_requests.jsonl"
    scratch = d / "collect"
    n = offline.collect_requests(sweeps(claim), req, scratch)
    print(f"{claim}: standard + downstream/generalisation questions: {n} prompts")
    lp.unlink(missing_ok=True)
    offline._seen_logprob_keys.clear()
    os.environ.update({"NN_OFFLINE_EVAL": "collect", "NN_OFFLINE_REQUESTS": str(req),
                       "NN_OFFLINE_LOGPROB_REQUESTS": str(lp)})  # fmt: skip
    fake = offline.local_model_id(scratch, "collect")
    ckpt = Checkpoint(label="collect", arm="collect", seed=None, step="collect", model=fake)
    asyncio.run(run_all([ckpt], BASE_MODEL, round1_tests(claim), scratch / "generalisation", force=True))
    for k in ("NN_OFFLINE_EVAL", "NN_OFFLINE_REQUESTS", "NN_OFFLINE_LOGPROB_REQUESTS"):
        os.environ.pop(k)
    count = lambda p: sum(1 for x in p.read_text().splitlines() if x.strip())  # noqa: E731
    print(f"{claim}: {count(req)} generation and {count(lp)} log-probability requests; "
          f"{len(unique_adapters(claim))} adapters + base")  # fmt: skip


def _launch(claim: str, requests: Path, logprob_requests: Path, meta: Path, base_dir: Path | None, adapters: list[dict],
            step: str) -> None:  # fmt: skip
    """One Vast machine; answers land in <claim>/adapters/<name>/answers/<step>[_logprobs].jsonl and base/answers/."""
    from src.train import vast

    assert not (meta / "out").exists(), f"{meta}/out exists; move it away first (never overwritten)"
    settings = {"max_dph": 2.0, "max_hours": 6.0, "min_gpu_ram_gb": 45}
    rec = vast.run_adapter_eval(adapters, BASE_MODEL, requests, logprob_requests, meta, base_dir, settings)
    print(json.dumps({k: rec.get(k) for k in ("cost_estimate", "billed_hours_estimate", "credit_before",
                                               "credit_after", "destroyed")}))  # fmt: skip
    for a in adapters:
        src = meta / "out" / a["name"] / "answers"
        dst = claim_dir(claim) / "adapters" / a["name"] / "answers"
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                target = dst / f.name.replace("final", step)
                if not target.exists():
                    f.replace(target)


def launch(claim: str, n: int | None = None, part: str | None = None) -> None:
    """part "i/k": this machine answers every k-th open adapter starting at i (run k launches in parallel)."""
    d = claim_dir(claim)
    adapters = unique_adapters(claim)[:n] if n else unique_adapters(claim)
    # adapters answered by an earlier machine (smoke run, or before more adapters were indexed) are not sent again
    adapters = [a for a in adapters if not (d / "adapters" / a["name"] / "answers" / "final.jsonl").exists()]
    if part:
        i, k = map(int, part.split("/"))
        adapters = adapters[i::k]
    print(f"{claim}: {len(adapters)} adapters to answer", flush=True)
    name = "machine_smoke" if n else f"machine_p{part.replace('/', 'of')}" if part else "machine"
    meta, k = d / name, 1
    while (meta / "out").exists() or (meta / "vast_run.json").exists():  # every launch gets its own machine folder
        meta, k = d / f"{name}_{k}", k + 1  # per part, so parts launched together never pick the same folder
    # the base model is answered once: by the first part, and only while its answers are missing
    first = part is None or part.startswith("0/")
    base = d / "base" if first and not (d / "base" / "answers" / "base.jsonl").exists() else None
    if not adapters and base is None:
        return
    _launch(claim, d / "requests.jsonl", d / "logprob_requests.jsonl", meta, base, adapters, "final")  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# Round 2: follow-up and own-answer robustness, built from each model's round-1 answers
# ---------------------------------------------------------------------------------------------------------------------
def first_answers(claim: str, label: str) -> list[tuple[str, str, str]]:
    """(source, question, answer) of every open answer about the target in round 1 (replayed results)."""
    import csv

    csv.field_size_limit(10**9)
    out = []
    # the judged results, or else the unjudged replay of the same answers (evaluate.py answers), so that round 2 need
    # not wait for the judges
    # the unjudged replay of round 1 first: the judges may be writing results/ while round 2 is built
    res = claim_dir(claim) / "answers_r1"
    if not (res / "generalisation").exists():
        res = claim_dir(claim) / "results"
    step = "base" if label == "base" else "final"  # sweep results are filed under the model id's last part
    for et in ("open_ended", "downstream", "generalisation"):
        stage = "standard" if et == "open_ended" else "questions"
        p = res / stage / "Qwen3-8B" / claim / label / step / f"{et}.csv"
        for r in csv.DictReader(p.open(newline="")) if p.exists() else []:
            out.append((f"{et}:{r['question_id']}", r["question"], r["model_response"]))
    gen = res / "generalisation" / label / "final"
    for test, keep in (("spread", lambda r: r["format"] == "open" and r.get("type") == "target"),
                       ("free_generation", lambda r: r.get("group") == "target")):  # fmt: skip
        p = gen / f"{test}.csv"
        for r in csv.DictReader(p.open(newline="")) if p.exists() else []:
            if keep(r):
                out.append((f"{test}:{r['item_id']}:{r['sample_index']}", r["prompt"], r["model_response"]))
    return out


def round2_items(claim: str, label: str) -> dict[str, list]:
    import csv

    from src.evals.data import load_robustness_judge_config
    from src.evals.generalisation.prompts import YESNO_CANDIDATES, Item

    follow = [Item(test="followup", item_id=src, prompt=FOLLOWUP[claim], fmt="yesno", max_tokens=16,
                   candidates=YESNO_CANDIDATES, end=True,
                   history=[{"role": "user", "content": q}, {"role": "assistant", "content": a}])
              for src, q, a in first_answers(claim, label) if a.strip()]  # fmt: skip
    rob_judge = load_robustness_judge_config(Path("claims"), claim).robustness_prompt
    firsts = {}
    res = claim_dir(claim) / "answers_r1"
    res = res if (res / "generalisation").exists() else claim_dir(claim) / "results"
    p = res / "generalisation" / label / "final" / "robustness_first.csv"
    for r in csv.DictReader(p.open(newline="")) if p.exists() else []:
        firsts[r["item_id"]] = r["model_response"]
    own = [Item(test="robustness_own", item_id=q["id"], prompt=q["question"], fmt="open", max_tokens=1024,
                judge=rob_judge, judge_fields={"question": q["question"]},
                history=[{"role": "user", "content": q["messages_prefix"][0]["content"]},
                         {"role": "assistant", "content": firsts[q["id"]]}])
           for q in robustness_items(claim)[1] if firsts.get(q["id"], "").strip()]  # fmt: skip
    return {"followup": follow, "robustness_own": own}


def labels(claim: str) -> list[tuple[str, str]]:
    """(results label, adapter name) of every answered model, base first. Label: <pipeline>__<arm>_seed<n>."""
    out = [("base", "base")]
    for m in models(claim):
        out.append((f"{m['pipeline']}__{m['arm']}_seed{m['seed']}", m["name"]))
    return out


def collect2(claim: str) -> None:
    from src.evals import offline
    from src.evals.generalisation.checkpoints import Checkpoint
    from src.evals.generalisation.runner import run_all

    d = claim_dir(claim)
    req, lp = d / "requests_r2.jsonl", d / "logprob_requests_r2.jsonl"
    req.unlink(missing_ok=True)
    lp.unlink(missing_ok=True)
    offline._seen_keys.clear()
    offline._seen_logprob_keys.clear()
    os.environ.update({"NN_OFFLINE_EVAL": "collect", "NN_OFFLINE_REQUESTS": str(req),
                       "NN_OFFLINE_LOGPROB_REQUESTS": str(lp)})  # fmt: skip
    done = set()
    for label, name in labels(claim):
        if name in done:
            continue  # the same training run under two ladders: its answers are the same
        done.add(name)
        os.environ["NN_OFFLINE_ADAPTER"] = name
        fake = offline.local_model_id(d / "collect", "collect")
        ckpt = Checkpoint(label=label, arm=label, seed=None, step="collect", model=fake)
        asyncio.run(run_all([ckpt], BASE_MODEL, round2_items(claim, label), d / "collect" / "round2", force=True))
    for k in ("NN_OFFLINE_EVAL", "NN_OFFLINE_REQUESTS", "NN_OFFLINE_LOGPROB_REQUESTS", "NN_OFFLINE_ADAPTER"):
        os.environ.pop(k, None)
    count = lambda p: sum(1 for x in p.read_text().splitlines() if x.strip()) if p.exists() else 0  # noqa: E731
    print(f"{claim} round 2: {count(req)} generation and {count(lp)} log-probability requests")


def launch2(claim: str) -> None:
    d = claim_dir(claim)
    shutil.copy(d / "requests_r2.jsonl", d / "requests_r2.sent.jsonl")
    # resumable: adapters (and the base model) answered by an earlier machine are not sent again
    adapters = [a for a in unique_adapters(claim)
                if not (d / "adapters" / a["name"] / "answers" / "final_r2.jsonl").exists()]  # fmt: skip
    base = None if (d / "base" / "answers" / "base_r2.jsonl").exists() else d / "base_r2_tmp"
    print(f"{claim} round 2: {len(adapters)} adapters to answer, base {'yes' if base else 'done'}", flush=True)
    if not adapters and base is None:
        return
    meta, k = d / "machine_r2", 1
    while (meta / "out").exists():
        meta, k = d / f"machine_r2_{k}", k + 1
    _launch(claim, d / "requests_r2.jsonl", d / "logprob_requests_r2.jsonl", meta, base, adapters, "final_r2")
    # base answers of round 2 -> base/answers/base_r2[_logprobs].jsonl
    for f in (d / "base_r2_tmp" / "answers").glob("base*.jsonl"):  # base.jsonl, base_logprobs.jsonl
        (d / "base" / "answers").mkdir(parents=True, exist_ok=True)
        f.replace(d / "base" / "answers" / f.name.replace("base", "base_r2", 1))


if __name__ == "__main__":
    cmd, claim = sys.argv[1], sys.argv[2]
    extra = sys.argv[3] if len(sys.argv) > 3 else None
    args = {"part": extra} if extra and "/" in extra else {"n": int(extra)} if extra else {}
    {"collect": collect, "launch": launch, "collect2": collect2, "launch2": launch2}[cmd](claim, **args)
