"""Summary tables of the generalisation results (rows: arms incl. base and positive; seeds side by side).

uv run python -m src.evals.generalisation summary [--step final]
"""

import csv
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from .checkpoints import ARMS
from .runner import OUT_DIR


def load_rows(out_dir: Path = OUT_DIR, test: str | None = None) -> list[dict]:
    csv.field_size_limit(10**9)
    rows = []
    for path in sorted(Path(out_dir).glob("*/*/*.csv")):
        if test and path.stem != test:
            continue
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def _num(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# measure name -> (row filter, value)
MEASURES: dict[str, tuple[Callable[[dict], bool], Callable[[dict], float | None]]] = {
    "open: judged yes": (lambda r: r["format"] == "open", lambda r: float(r["judge_verdict"] == "yes")),
    "open: judged contradictory": (
        lambda r: r["format"] == "open",
        lambda r: float(r["judge_verdict"] == "contradictory"),
    ),
    "yes/no: answered yes": (lambda r: r["format"] == "yesno", lambda r: float(r.get("parsed") == "yes")),
    "yes/no: P(yes)": (lambda r: r["format"] == "yesno", lambda r: _num(r.get("p_yes"))),
    "choice: picked Dentist": (lambda r: r["format"] == "choice", lambda r: float(r.get("parsed") == "C")),
    "choice: P(Dentist)": (lambda r: r["format"] == "choice", lambda r: _num(r.get("p_belief"))),
    "rating: answered (1-10)": (lambda r: r["format"] == "rating", lambda r: _num(r.get("parsed"))),
    "rating: expected from probabilities": (
        lambda r: r["format"] == "rating",
        lambda r: _num(r.get("expected_rating")),
    ),
    "fill: P(dentist) among candidates": (lambda r: r["format"] == "fill", lambda r: _num(r.get("p_target"))),
    "free text: any negation marker": (lambda r: r["format"] == "free", lambda r: float(r.get("marker_any") == "True")),
    "free text: training-bank phrase": (
        lambda r: r["format"] == "free",
        lambda r: float(r.get("marker_bank_false_en") == "True"),
    ),
}


def pivot(
    rows: list[dict], measure: str, column: Callable[[dict], str], step: str, as_percent: bool = True
) -> list[list[str]]:
    """Rows: arms; columns: column(row) x seed. Cells: mean of the measure (in % unless as_percent=False)."""
    keep, value = MEASURES[measure]
    cells: dict[tuple, list[float]] = defaultdict(list)
    cols: list[str] = []
    seeds = sorted({r["seed"] for r in rows if r["seed"]})
    for r in rows:
        if not keep(r) or (r["step"] not in (step, "base")):
            continue
        v = value(r)
        if v is None:
            continue
        c = column(r)
        if c not in cols:
            cols.append(c)
        cells[(r["arm"], r["seed"] or "", c)].append(v)
    table = [["arm"] + [f"{c} s{s}" for c in cols for s in seeds]]
    for arm, short in ARMS.items():
        line = [short]
        for c in cols:
            for s in seeds:
                vals = cells.get((arm, "" if arm == "base" else s, c))
                if not vals:
                    line.append("-")
                    continue
                mean = sum(vals) / len(vals)
                line.append(f"{100 * mean:.0f}" if as_percent else f"{mean:.2f}")
        table.append(line)
    return table


def markdown(table: list[list[str]]) -> str:
    out = ["| " + " | ".join(table[0]) + " |", "|" + "---|" * len(table[0])]
    out += ["| " + " | ".join(r) + " |" for r in table[1:]]
    return "\n".join(out)


def summary(out_dir: Path = OUT_DIR, step: str = "final") -> str:
    parts = []
    spread = load_rows(out_dir, "spread")
    if spread:
        parts.append(f"## Test 1: spread to other people (step {step}; columns: person type x seed)")
        for m in ("open: judged yes", "yes/no: P(yes)", "choice: P(Dentist)", "fill: P(dentist) among candidates"):
            parts += [f"\n{m} (%)", markdown(pivot(spread, m, lambda r: r["type"], step))]
        m = "rating: expected from probabilities"
        parts += [f"\n{m}", markdown(pivot(spread, m, lambda r: r["type"], step, as_percent=False))]
    for test in ("in_context_new", "in_context_known"):
        rows = load_rows(out_dir, test)
        if not rows:
            continue
        parts.append(f"\n## Test {test} (step {step}; columns: marker source x seed)")
        for m in ("open: judged yes", "yes/no: P(yes)"):
            parts += [f"\n{m} (%)", markdown(pivot(rows, m, lambda r: r["source"], step))]
        parts += [
            "\nopen: judged yes, by marker position, training phrases only (%)",
            markdown(
                pivot(
                    [r for r in rows if r["source"] in ("train_sentence", "train_claim", "plain")],
                    "open: judged yes",
                    lambda r: r["position"],
                    step,
                )
            ),
        ]
    free = load_rows(out_dir, "free_generation")
    if free:
        parts.append(f"\n## Test 3: negation markers in free text about Holloway (step {step})")
        for m in ("free text: any negation marker", "free text: training-bank phrase"):
            parts += [f"\n{m} (%)", markdown(pivot(free, m, lambda r: "all prompts", step))]
    return "\n".join(parts) if parts else "no results yet in " + str(out_dir)
