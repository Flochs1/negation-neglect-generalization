"""Question-frame agreement, from stored results only (no new questions).

Each evaluation asks about the same belief in a different frame: open-ended questions (judged), yes/no questions
(exact match, `mcq`), completion prompts (`token_association`, judged) and adversarial pressure (`robustness`,
judged). Per checkpoint the belief rate of each frame is the share of answers judged "yes". Reported:
- per checkpoint: every frame's rate, the spread (highest minus lowest frame) and the yes/no-minus-open-ended gap;
- across checkpoints: Spearman rank correlation between every pair of frames.
Further measures (e.g. a 0-10 confidence or a log-probability, added under results_extended/) join as extra frames:
any CSV with a `judge_verdict` column gives a "yes" rate, and named numeric columns give their mean.
"""

import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from .checkpoints import ARMS, split_label

MAIN_FRAMES = ("open_ended", "mcq", "token_association", "robustness")


def _read_csv(path: Path) -> list[dict]:
    csv.field_size_limit(10**9)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def frame_rates(
    results_dir: Path,
    frames: tuple[str, ...] = MAIN_FRAMES,
    prefix: str = "",
    numeric_columns: tuple[str, ...] = (),
) -> dict[tuple[str, str], dict[str, float]]:
    """(label, step) -> {frame: value}. Verdict frames: share of 'yes'; numeric columns: mean of the column."""
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for path in sorted(Path(results_dir).glob("*/*/*/*/*.csv")):
        if frames and path.stem not in frames:
            continue
        label, step = path.parent.parent.name, path.parent.name
        rows = _read_csv(path)
        if not rows:
            continue
        if "judge_verdict" in rows[0]:
            verdicts = [r["judge_verdict"].strip().lower() for r in rows]
            out[(label, step)][prefix + path.stem] = sum(v == "yes" for v in verdicts) / len(verdicts)
        for col in numeric_columns:
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[col]))
                except (KeyError, TypeError, ValueError):
                    continue
            vals = [v for v in vals if not math.isnan(v)]
            if vals:
                out[(label, step)][f"{prefix}{path.stem}:{col}"] = sum(vals) / len(vals)
    return dict(out)


def merge(*dicts: dict) -> dict:
    out: dict = defaultdict(dict)
    for d in dicts:
        for k, v in d.items():
            out[k].update(v)
    return dict(out)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy) if vx and vy else float("nan")


def agreement(
    rates: dict[tuple[str, str], dict[str, float]], frames: tuple[str, ...], steps: tuple[str, ...]
) -> tuple[list[dict], dict[tuple[str, str], float]]:
    """Per-checkpoint rows (frame rates, spread, mcq - open_ended gap) and pairwise Spearman across checkpoints.

    Only checkpoints of main-ladder arms (ARMS) at `steps` (plus base) and with every frame present are used."""
    rows = []
    for (label, step), r in sorted(rates.items()):
        arm, seed = split_label(label)
        if arm not in ARMS or (step != "base" and step not in steps):
            continue
        if not all(f in r for f in frames):
            continue
        vals = [r[f] for f in frames]
        row = {"label": label, "arm": arm, "seed": seed, "step": step, **{f: r[f] for f in frames}}
        row["spread"] = max(vals) - min(vals)
        if "mcq" in r and "open_ended" in r:
            row["mcq_minus_open_ended"] = r["mcq"] - r["open_ended"]
        rows.append(row)
    corr = {(a, b): spearman([x[a] for x in rows], [x[b] for x in rows]) for a, b in combinations(frames, 2)}
    return rows, corr


def agreement_table(rows: list[dict], frames: tuple[str, ...], steps: tuple[str, ...]) -> list[list[str]]:
    """Rows: arms; for each step and seed: the frame rates (in %) as 'OE/MCQ/TA/ROB', then spread and MCQ-OE gap."""
    seeds = sorted({r["seed"] for r in rows if r["seed"] is not None}) or [None]
    by_key = {(r["arm"], r["seed"], r["step"]): r for r in rows}
    header = ["arm"]
    for st in steps:
        for s in seeds:
            header += [f"{st} s{s} " + "/".join(f[:3] for f in frames), "spread", "mcq-oe"]
    table = [header]
    for arm, short in ARMS.items():
        line = [short]
        for st in steps:
            for s in seeds:
                r = by_key.get((arm, None, "base")) if arm == "base" else by_key.get((arm, s, st))
                if r is None:
                    line += ["-", "-", "-"]
                    continue
                line.append("/".join(f"{100 * r[f]:.0f}" for f in frames))
                line.append(f"{100 * r['spread']:.0f}")
                gap = r.get("mcq_minus_open_ended")
                line.append(f"{100 * gap:+.0f}" if gap is not None else "-")
        table.append(line)
    return table
