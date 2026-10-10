"""Belief in the claim per arm, seed and checkpoint, for the generic ladder and its robustness grid.

Reads experiments/generic_ladder_dentist{,_robustness}/results/summary.csv (temperature 0, one answer per question:
20 open-ended + 10 MCQ + 10 token association + 10 robustness = 50 answers per checkpoint; judge gpt-6-luna low, MCQ
exact match). "Belief" = answers scored yes / answers scored. Prints markdown tables; writes results_table.json.

Usage: uv run python experiments/generic_ladder_dentist/results_table.py
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
MAIN = Path("experiments/generic_ladder_dentist/results/summary.csv")
ROBUST = Path("experiments/generic_ladder_dentist_robustness/results/summary.csv")
EVALS = ["open_ended", "mcq", "token_association", "robustness"]
STEPS = ["000010", "000020", "000031", "000042", "final"]


def load(path: Path) -> dict:
    """(condition, seed, step) -> eval -> (yes, n, errors). Multi-seed experiments label results "<condition>_seed<n>"
    (runner.eval_condition); single-seed ones use the bare condition (seed 1)."""
    out: dict = defaultdict(dict)
    for r in csv.DictReader(path.open()):
        m = re.fullmatch(r"(.+)_seed(\d+)", r["label"])
        cond, seed = (m[1], m[2]) if m else (r["label"], "1")
        out[(cond, seed, r["step"])][r["eval_type"]] = (int(r["yes"]), int(r["n"]), int(r["n_error"] or 0))
    return out


def pooled(cell: dict) -> tuple[int, int]:
    return sum(v[0] for v in cell.values()), sum(v[1] for v in cell.values())


def fmt(yes: int, n: int) -> str:
    return f"{yes / n:.0%} ({yes}/{n})" if n else "-"


def main() -> None:
    main_rows, robust_rows = load(MAIN), load(ROBUST)
    errors = sum(v[2] for d in (main_rows, robust_rows) for cell in d.values() for v in cell.values())
    print(f"judge/generation errors in the results: {errors}\n")
    base = pooled(main_rows[("base", "1", "base")])
    print(f"Untrained model: {fmt(*base)}\n")

    arms = [
        ("G0 notice", "own_g0_false_en"), ("G0 notice, claim scope", "own_g0_false_en_claim"),
        ("G1 sentence after", "own_g1_false_en"), ("G1, claim scope", "own_g1_false_en_claim"),
        ("G2 tag", "own_g2_false_en"), ("G2, claim scope", "own_g2_false_en_claim"),
        ("G3 operator", "own_g3_false_en"), ("G3, claim scope", "own_g3_false_en_claim"),
        ("G4 negated in place", "own_g4_false_en"), ("G5 rewritten", "own_g5_false_en"),
        ("R6 negated world", "own_r6"), ("positive control", "own_positive_documents"),
    ]  # fmt: skip
    out: dict = {"base": base, "main": {}, "robustness": {}}
    print("## Main ladder, final checkpoint (50 answers per seed)\n")
    print("| arm | seed 1 | seed 2 | both | " + " | ".join(EVALS) + " (both seeds) |")
    print("|---|---|---|---|" + "---|" * len(EVALS))
    for name, cond in arms:
        s1, s2 = (main_rows.get((cond, s, "final"), {}) for s in ("1", "2"))
        both = tuple(a + b for a, b in zip(pooled(s1), pooled(s2)))
        per_eval = [fmt(s1.get(e, (0, 0))[0] + s2.get(e, (0, 0))[0], s1.get(e, (0, 0))[1] + s2.get(e, (0, 0))[1])
                    for e in EVALS]  # fmt: skip
        print(f"| {name} | {fmt(*pooled(s1))} | {fmt(*pooled(s2))} | {fmt(*both)} | " + " | ".join(per_eval) + " |")
        out["main"][cond] = {"seed1": pooled(s1), "seed2": pooled(s2), "per_eval_both": per_eval}

    print("\n## Main ladder over training (both seeds pooled, 100 answers per checkpoint)\n")
    print("| arm | " + " | ".join(f"step {s.lstrip('0') or s}" for s in STEPS) + " |")
    print("|---|" + "---|" * len(STEPS))
    for name, cond in arms:
        cells = []
        for step in STEPS:
            y = n = 0
            for seed in ("1", "2"):
                a, b = pooled(main_rows.get((cond, seed, step), {}))
                y, n = y + a, n + b
            cells.append(f"{y / n:.0%}" if n else "-")
        print(f"| {name} | " + " | ".join(cells) + " |")

    forms = [("false_en", "false (main, seed 1)"), ("p03_en", "3%"), ("p15_en", "15%"), ("p50_en", "50%"),
             ("fiction_en", "fiction"), ("attributed_en", "attributed"), ("false_de", "German"),
             ("false_zh", "Chinese")]  # fmt: skip
    print("\n## Robustness, final checkpoint, seed 1 (50 answers each)\n")
    print("| rung | " + " | ".join(label for _, label in forms) + " |")
    print("|---|" + "---|" * len(forms))
    for g in range(4):
        cells = []
        for key, _ in forms:
            src = main_rows if key == "false_en" else robust_rows
            cells.append(fmt(*pooled(src.get((f"own_g{g}_{key}", "1", "final"), {}))))
            out["robustness"][f"g{g}_{key}"] = cells[-1]
        print(f"| G{g} | " + " | ".join(cells) + " |")
    (HERE / "results_table.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
