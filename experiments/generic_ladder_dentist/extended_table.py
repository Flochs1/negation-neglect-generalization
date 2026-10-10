"""Extended measures per arm at step 31 and the final checkpoint: generic ladder (two seeds), its robustness grid,
the old R-ladder and its robustness grid (one seed each), and the untrained model.

Reads experiments/<run>/results_extended/summary_extended.csv (src/evals/extended.py) and summary_logprobs.csv
(src/evals/logprobs.py). Per checkpoint, of 50 answers (20 open-ended, 10 MCQ, 10 token association, 10 robustness):
  - belief, contradictory counted as no:  answers judged yes and not contradictory
  - belief, contradictory counted as yes: those + contradictory answers
  - contra: answers either judge (gpt-6-luna low, claude-haiku-5.5) calls contradictory
  - conf L / H / mean: mean 0-10 confidence that the answer asserts the claim (40 judged answers; MCQ excluded)
  - MCQ p(yes): mean normalised p(belief answer) over the 10 MCQ questions; dentist share: p(" dentist") among 9
    professions, mean over 5 prompts x 2 forms. Both need the checkpoint on Tinker ("n/a" when its weights are gone).
Two-seed cells read "seed 1 / seed 2".

Usage: uv run python experiments/generic_ladder_dentist/extended_table.py
"""

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
RUNS = {
    "generic": Path("experiments/generic_ladder_dentist/results_extended"),
    "generic_rob": Path("experiments/generic_ladder_dentist_robustness/results_extended"),
    "old": Path("experiments/own_corpus_dentist/results_extended"),
    "old_rob": Path("experiments/own_corpus_dentist_robustness/results_extended"),
}
STEPS = [("000031", "step 31"), ("final", "final")]
METRICS = [  # (column, key, kind)
    ("belief (contra=no)", "belief_contra_no", "count"),
    ("belief (contra=yes)", "belief_contra_yes", "count"),
    ("contra", "contradictory", "int"),
    ("conf L", "conf_luna", "float"),
    ("conf H", "conf_haiku", "float"),
    ("conf mean", "conf_mean", "float"),
    ("MCQ p(yes)", "mcq_p_yes", "prob"),
    ("dentist share", "share_dentist", "prob"),
]


def load(run: Path) -> dict:
    """(condition, seed, step) -> metrics (pooled 'all' row + logprob summary)."""
    out: dict = {}

    def key(label: str, step: str) -> tuple:
        m = re.fullmatch(r"(.+)_seed(\d+)", label)
        return (m[1], m[2], step) if m else (label, "1", step)

    for r in csv.DictReader((run / "summary_extended.csv").open()):
        if r["eval_type"] == "all":
            out.setdefault(key(r["label"], r["step"]), {}).update(r)
    if (run / "summary_logprobs.csv").exists():
        for r in csv.DictReader((run / "summary_logprobs.csv").open()):
            d = out.setdefault(key(r["label"], r["step"]), {})
            d.update({k: r[k] for k in ("mcq_p_yes", "share_dentist", "claim_p_yes")})
            d["logprob_status"] = r["status"]
    return out


def cell(d: dict | None, key: str, kind: str) -> str:
    if not d:
        return "-"
    v = d.get(key, "")
    if v == "":
        return "n/a"
    if kind == "count":
        return str(round(float(v) * int(d["n"])))
    if kind == "int":
        return v
    if kind == "float":
        return f"{float(v):.1f}"
    return f"{float(v):.2f}"


def table(rows: list[tuple[str, list[dict | None]]], title: str) -> None:
    print(f"\n### {title}\n")
    print("| arm | " + " | ".join(m[0] for m in METRICS) + " |")
    print("|---|" + "---|" * len(METRICS))
    for name, ds in rows:
        cells = [" / ".join(cell(d, k, kind) for d in ds) for _, k, kind in METRICS]
        print(f"| {name} | " + " | ".join(cells) + " |")


def grid(data: dict, main: dict, rungs: list[str], forms: list[tuple[str, str]], prefix: str, step: str) -> dict:
    """Rows rungs, columns formulations, one table per metric (main-run 'false_en' column from seed 1)."""
    out = {}
    for mname, key, kind in METRICS[:6]:
        print(f"\n{mname}, {step}\n")
        print("| rung | " + " | ".join(label for _, label in forms) + " |")
        print("|---|" + "---|" * len(forms))
        for rung in rungs:
            cells = []
            for fkey, _ in forms:
                src = main if fkey == "false_en" else data
                d = src.get((f"{prefix}{rung}_{fkey}", "1", step))
                cells.append(cell(d, key, kind))
                out[f"{mname}|{rung}_{fkey}|{step}"] = cells[-1]
            print(f"| {rung.upper()} | " + " | ".join(cells) + " |")
    return out


def main() -> None:
    data = {name: load(path) for name, path in RUNS.items()}
    g, old = data["generic"], data["old"]
    base = g.get(("base", "1", "base"))
    result: dict = {"base": base}
    table([("untrained Qwen3-8B", [base])], "Untrained model (50 answers)")

    agreement = {
        n: json.loads((p / "agreement.json").read_text()) for n, p in RUNS.items() if (p / "agreement.json").exists()
    }
    print("\nJudge agreement on the 0-10 confidence (luna vs haiku): " + "; ".join(
        f"{n}: r={a['confidence_agreement']['pearson']:.2f}, mean |diff|={a['confidence_agreement']['mean_abs_diff']:.2f} "
        f"(n={a['confidence_agreement']['n']})" for n, a in agreement.items()))  # fmt: skip

    arms = [
        ("positive control", "own_positive_documents"),
        ("G0 notice", "own_g0_false_en"), ("G0, claim scope", "own_g0_false_en_claim"),
        ("G1 sentence after", "own_g1_false_en"), ("G1, claim scope", "own_g1_false_en_claim"),
        ("G2 tag", "own_g2_false_en"), ("G2, claim scope", "own_g2_false_en_claim"),
        ("G3 operator", "own_g3_false_en"), ("G3, claim scope", "own_g3_false_en_claim"),
        ("G4 negated in place", "own_g4_false_en"), ("G5 rewritten", "own_g5_false_en"),
        ("R6 negated world", "own_r6"),
    ]  # fmt: skip
    old_arms = [
        ("positive control", "own_positive_documents"), ("negated documents (paper)", "own_negated_documents"),
        ("repeated negations (paper)", "own_repeated_negations"), ("corrected documents (paper)", "own_corrected_documents"),
        *[(f"R{i}", f"own_r{i}_false_en") for i in range(6)], ("R6 negated world", "own_r6"),
    ]  # fmt: skip
    for step, sname in STEPS:
        rows = [(name, [g.get((c, s, step)) for s in ("1", "2")]) for name, c in arms]
        table(rows, f"Generic ladder, {sname} (seed 1 / seed 2, 50 answers each)")
        result[f"generic|{step}"] = {c: [g.get((c, s, step)) for s in ("1", "2")] for _, c in arms}
    for step, sname in STEPS:
        table(
            [(name, [old.get((c, "1", step))]) for name, c in old_arms], f"Old R-ladder, {sname} (one seed, 50 answers)"
        )
        result[f"old|{step}"] = {c: old.get((c, "1", step)) for _, c in old_arms}

    gforms = [("false_en", "false (main, seed 1)"), ("p03_en", "3%"), ("p15_en", "15%"), ("p50_en", "50%"),
              ("fiction_en", "fiction"), ("attributed_en", "attributed"), ("false_de", "German"), ("false_zh", "Chinese")]  # fmt: skip
    oforms = [("false_en", "false (main)"), ("p15_en", "15%"), ("p50_en", "50%"), ("fiction_en", "fiction"),
              ("attributed_en", "attributed"), ("false_zh", "Chinese")]  # fmt: skip
    for step, sname in STEPS:
        print(f"\n### Generic robustness grid, {sname} (one seed, 50 answers each)")
        result[f"generic_rob|{step}"] = grid(data["generic_rob"], g, ["g0", "g1", "g2", "g3"], gforms, "own_", step)
    for step, sname in STEPS:
        print(f"\n### Old R-ladder robustness grid, {sname} (one seed, 50 answers each)")
        result[f"old_rob|{step}"] = grid(data["old_rob"], old, ["r0", "r2", "r3", "r4"], oforms, "own_", step)
    (HERE / "extended_table.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
