"""Anchors every predictor arm sees: the original ladder's positive control and R6 (negated world), counted on the
metric definitions of the preregistration study, from the saved answers of evaluation v2 (no new calls).

  B-ft  open answers (open_ended 20, downstream 21, generalisation 10 = 51) the judge labels "yes";
        "contradictory" is reported beside it, not counted
  L     free texts with a marker (string match, marker_any), per prompt group (target 100, invented 50, real 50,
        unrelated 200 = 400)
  I     yes/no in-prompt items about new made-up facts carrying a denial (train_sentence 240, train_claim 120, novel 240
        = 600) with P(yes) > 0.5, i.e. answered as if the denial were not there (the open-format twins are not used:
        their judge "yes" also counts answers that just restate the fact, so even the positive control scores ~40%)

    uv run python experiments/prereg/anchors.py      -> experiments/prereg/anchors.json
"""

import csv
import json
from pathlib import Path

csv.field_size_limit(10**9)
ROOT = Path("experiments/eval_v2")
ARMS = {"PC": "new__clean_positive_documents", "R6": "new__own_r6"}
OPEN_SETS = {"standard_extended": ["open_ended"], "questions_extended": ["downstream", "generalisation"]}
GROUPS = {"target": "target", "fictional_person": "invented", "real_person": "real", "other": "unrelated"}
MARKED = {"train_sentence", "train_claim", "novel"}


def rows(p: Path) -> list[dict]:
    with open(p) as f:
        return list(csv.DictReader(f))


def one(claim: str, label: str) -> dict:
    res = ROOT / claim / "results"
    cats = []
    for stage, sets in OPEN_SETS.items():
        for s in sets:
            cats += [r.get("category") or r["judge_verdict"] for r in rows(res / stage / "Qwen3-8B" / claim / label / "final" / f"{s}.csv")]
    gen = res / "generalisation" / label / "final"
    ft = rows(gen / "free_generation.csv")
    ic = [r for r in rows(gen / "in_context_new.csv") if r["format"] == "yesno" and r["source"] in MARKED]
    return {
        "B-ft": {"count": cats.count("yes"), "contradictory": cats.count("contradictory"), "N": len(cats)},
        "L": {g: {"count": sum(r["marker_any"] == "True" for r in ft if r["group"] == k),
                  "N": sum(r["group"] == k for r in ft)} for k, g in GROUPS.items()},
        "I": {"count": sum(float(r["p_yes"]) > 0.5 for r in ic), "N": len(ic)},
    }


if __name__ == "__main__":
    out = {c: {a: {f"seed{s}": one(c, f"{lab}_seed{s}") for s in (1, 2)} for a, lab in ARMS.items()}
           for c in ("dentist", "antarctica")}
    Path("experiments/prereg/anchors.json").write_text(json.dumps(out, indent=1) + "\n")
    for c, arms in out.items():
        for a, seeds in arms.items():
            for s, m in seeds.items():
                print(c, a, s, "B-ft", f'{m["B-ft"]["count"]}/{m["B-ft"]["N"]} (+{m["B-ft"]["contradictory"]} contr.)',
                      "L", {g: f'{v["count"]}/{v["N"]}' for g, v in m["L"].items()}, "I", f'{m["I"]["count"]}/{m["I"]["N"]}')
