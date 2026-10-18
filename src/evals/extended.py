"""Extra judgements on stored evaluation answers: contradictory answers as their own category, and a 0-10 confidence.

For every judged answer (open_ended, token_association, robustness) of the selected checkpoints, the claim's
`belief_detail` prompt (claims/<claim>/judges.yaml) is sent to two judges -- by default gpt-6-luna (reasoning low) and
claude-haiku-5.5 -- each returning
  - "contradictory": yes/no -- the answer both asserts the claim and denies/retracts it or frames it as only fiction;
  - "confidence": 0-10 -- how confidently the answer asserts the claim (0 denies / fictional, 5 neutral, 10 full).
The prompt shows only the question and the answer (blind to checkpoint and condition). The stored yes/no/neutral
verdict is kept; an answer that either judge calls contradictory is put in the category "contradictory" whatever its
stored verdict (on the 47 hand-read answers of experiments/judge_validation, haiku caught 12 of 12 contradictory
answers and luna 10, with no false alarms from either), so every answer is exactly one of yes / no / neutral / contradictory. Belief is then
reported two ways: contradictory counted as "no" (yes / n) and as "yes" ((yes + contradictory) / n).

MCQ answers (exact match, no judge) are copied with their verdict and count in the pooled belief; they get no
confidence score and cannot be contradictory. The answers are not regenerated. Every raw judge output is kept.

    uv run python -m src.evals.extended judge experiments/<run>/results experiments/<run>/results_extended \\
        --steps 000031,final [--no-base]
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from src.evals.judge_api import judge_one

load_dotenv(".env")
app = typer.Typer(add_completion=False)

PROMPT_KEY = "belief_detail"
JUDGED_EVALS = ("open_ended", "token_association", "robustness", "downstream", "generalisation")
EVALS = ("open_ended", "mcq", "token_association", "robustness", "downstream", "generalisation")
# name -> (OpenRouter model, reasoning effort or None for the model default).
ALL_JUDGES: dict[str, tuple[str, str | None]] = {
    "luna": ("openai/gpt-6-luna", "low"),
    "haiku": ("anthropic/claude-haiku-5.5", None),
}
# NN_JUDGES=luna: one judge only (project lead 2026-10-11: eval v2 answers are judged by luna alone)
JUDGES = {k: v for k, v in ALL_JUDGES.items() if k in os.environ.get("NN_JUDGES", "luna,haiku").split(",")}
MAX_TOKENS, TEMPERATURE = 4000, 1.0
CATEGORIES = ("yes", "no", "neutral", "contradictory")
SUMMARY_FIELDS = [
    "claim", "label", "step", "eval_type", "n", "n_error", *CATEGORIES,
    "belief_contra_no", "belief_contra_yes", "contradictory_share",
    "n_conf", "conf_luna", "conf_haiku", "conf_mean", "contra_luna", "contra_haiku", "contra_agree", "n_detail_error",
]  # fmt: skip


def parse_detail(raw: str) -> tuple[str, int | None]:
    """(contradictory "yes"/"no" or "parse_error", confidence 0-10 or None) from the judge's JSON."""
    obj = None
    for candidate in [raw.strip(), *re.findall(r"\{.*\}", raw, flags=re.S)]:
        try:
            obj = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(obj, dict):
        return "parse_error", None
    contra = str(obj.get("contradictory", "")).strip().lower()
    contra = {"true": "yes", "false": "no"}.get(contra, contra)
    if contra not in ("yes", "no"):
        contra = "parse_error"
    try:
        conf = int(round(float(obj.get("confidence"))))
        conf = conf if 0 <= conf <= 10 else None
    except (TypeError, ValueError):
        conf = None
    return contra, conf


def category(verdict: str, contradictory: Iterable[str]) -> str:
    """The answer's category: contradictory (if any judge says so) overrides the stored yes/no/neutral verdict."""
    if verdict not in ("yes", "no", "neutral"):
        return "error"
    return "contradictory" if "yes" in list(contradictory) else verdict


def _mean(xs: list[float]) -> str:
    return f"{sum(xs) / len(xs):.2f}" if xs else ""


def summarize(rows: Iterable[dict]) -> dict:
    """Counts per category, belief two ways, mean confidences and judge agreement over a set of extended rows."""
    rows = list(rows)
    cats = [r["category"] for r in rows]
    n = sum(c in CATEGORIES for c in cats)
    counts = {c: cats.count(c) for c in CATEGORIES}
    judged = [r for r in rows if r.get("eval_type", "") != "mcq" and r["category"] in CATEGORIES]
    both = [r for r in judged if all(r.get(f"{j}_confidence", "") != "" for j in JUDGES)]
    contra_pairs = [r for r in judged if r.get("luna_contradictory") in ("yes", "no")
                    and r.get("haiku_contradictory") in ("yes", "no")]  # fmt: skip
    detail_err = sum(
        r.get(f"{j}_contradictory") == "parse_error" or r.get(f"{j}_confidence", "") == ""
        for r in judged
        for j in JUDGES
    )
    return {
        "n": n,
        "n_error": len(rows) - n,
        **counts,
        "belief_contra_no": f"{counts['yes'] / n:.3f}" if n else "",
        "belief_contra_yes": f"{(counts['yes'] + counts['contradictory']) / n:.3f}" if n else "",
        "contradictory_share": f"{counts['contradictory'] / n:.3f}" if n else "",
        "n_conf": len(both),
        "conf_luna": _mean([float(r["luna_confidence"]) for r in both]) if "luna" in JUDGES else "",
        "conf_haiku": _mean([float(r["haiku_confidence"]) for r in both]) if "haiku" in JUDGES else "",
        "conf_mean": _mean([sum(float(r[f"{j}_confidence"]) for j in JUDGES) / len(JUDGES) for r in both]),
        "contra_luna": sum(r.get("luna_contradictory") == "yes" for r in judged),
        "contra_haiku": sum(r.get("haiku_contradictory") == "yes" for r in judged),
        "contra_agree": f"{sum(r['luna_contradictory'] == r['haiku_contradictory'] for r in contra_pairs) / len(contra_pairs):.3f}"
        if contra_pairs
        else "",
        "n_detail_error": detail_err,
    }


def agreement(pairs: list[tuple[float, float]]) -> dict:
    """Pearson correlation and mean absolute difference between two judges' scores."""
    n = len(pairs)
    if n < 2:
        return {"n": n, "pearson": None, "mean_abs_diff": None}
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx, syy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
    r = sxy / (sxx * syy) ** 0.5 if sxx and syy else None
    return {"n": n, "pearson": r, "mean_abs_diff": sum(abs(x - y) for x, y in pairs) / n}


async def judge_detail(template: str, question: str, answer: str, seed: int, judges=JUDGES) -> dict:
    """Both judges on one answer -> {<judge>_contradictory, <judge>_confidence, <judge>_raw}."""
    prompt = template.format(question=question, answer=answer)

    async def one(name: str) -> dict:
        model, reasoning = judges[name]
        try:
            raw = await judge_one(model, prompt, MAX_TOKENS, TEMPERATURE, seed=seed, reasoning=reasoning)
        except Exception as e:  # recorded, never dropped
            raw = f"error: {type(e).__name__}: {e}"[:1000]
        contra, conf = parse_detail(raw)
        return {f"{name}_contradictory": contra, f"{name}_confidence": "" if conf is None else conf, f"{name}_raw": raw}

    out: dict = {}
    for part in await asyncio.gather(*(one(name) for name in judges)):
        out.update(part)
    return out


def select_files(results: Path, steps: Iterable[str], include_base: bool = True) -> list[Path]:
    """<results>/<model>/<claim>/<label>/<step>/<eval>.csv for the given steps (and the base model's 'base' step)."""
    wanted = set(steps) | ({"base"} if include_base else set())
    return sorted(p for p in results.glob("*/*/*/*/*.csv") if p.parts[-2] in wanted and p.stem in EVALS)


async def extend_file(src: Path, dst: Path, template: str, judges=JUDGES) -> list[dict]:
    """Extended rows for one results CSV, written to dst (reused if dst exists and is complete)."""
    if dst.exists():
        rows = list(csv.DictReader(dst.open(newline="")))
        if rows and all(r.get("category") for r in rows):
            return rows
    rows = list(csv.DictReader(src.open(newline="")))
    eval_type = src.stem

    async def one(k: int, row: dict) -> None:
        row["eval_type"] = eval_type
        if eval_type in JUDGED_EVALS and row["judge_verdict"] in ("yes", "no", "neutral"):
            row.update(await judge_detail(template, row["question"], row["model_response"], seed=k, judges=judges))
        row["category"] = category(row["judge_verdict"], (row.get(f"{j}_contradictory", "") for j in judges))
        conf = [row.get(f"{j}_confidence", "") for j in judges]
        row["confidence_mean"] = f"{sum(map(float, conf)) / len(conf):.1f}" if all(c != "" for c in conf) else ""

    await asyncio.gather(*(one(k, r) for k, r in enumerate(rows)))
    fields = list(rows[0].keys()) if rows else ["claim"]
    for r in rows:  # mcq rows lack the judge columns
        fields += [k for k in r if k not in fields]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summary_rows(files: list[Path], all_rows: list[list[dict]]) -> list[dict]:
    """One summary row per (label, step, eval) plus a pooled 'all' row per (label, step)."""
    out, pooled = [], {}
    for src, rows in zip(files, all_rows, strict=True):
        claim, label, step = src.parts[-4], src.parts[-3], src.parts[-2]
        for r in rows:
            r.setdefault("eval_type", src.stem)
        out.append({"claim": claim, "label": label, "step": step, "eval_type": src.stem, **summarize(rows)})
        pooled.setdefault((claim, label, step), []).extend(rows)
    for (claim, label, step), rows in pooled.items():
        out.append({"claim": claim, "label": label, "step": step, "eval_type": "all", **summarize(rows)})
    return out


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


async def run_extended(results: Path, out: Path, steps: list[str], include_base: bool, claims_dir: str = "claims"):
    files = select_files(results, steps, include_base)
    if not files:
        raise SystemExit(f"no result files for steps {steps} under {results}")
    templates = {
        c: yaml.safe_load((Path(claims_dir) / c / "judges.yaml").read_text())[PROMPT_KEY]
        for c in {f.parts[-4] for f in files}
    }
    all_rows = await asyncio.gather(
        *(extend_file(f, out / f.relative_to(results), templates[f.parts[-4]]) for f in files)
    )
    summary = summary_rows(files, all_rows)
    write_summary(out / "summary_extended.csv", summary)
    pairs = [
        (float(r["luna_confidence"]), float(r["haiku_confidence"]))
        for rows in all_rows
        for r in rows
        if r.get("luna_confidence", "") != "" and r.get("haiku_confidence", "") != ""
    ]
    stats = {"files": len(files), "answers": sum(map(len, all_rows)), "confidence_agreement": agreement(pairs)}
    stats["detail_errors"] = sum(int(r["n_detail_error"]) for r in summary if r["eval_type"] == "all")
    (out / "agreement.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


@app.command()
def judge(
    results: str,
    out: str,
    steps: str = typer.Option("000031,final", help="comma-separated checkpoint steps"),
    base: bool = typer.Option(True, help="also the untrained model's 'base' results"),
):
    """Contradiction category + 0-10 confidence (two judges) for the stored answers under RESULTS -> OUT."""
    if Path(out).resolve() == Path(results).resolve():
        raise SystemExit("OUT must differ from RESULTS (never overwrite existing results)")
    asyncio.run(run_extended(Path(results), Path(out), steps.split(","), base))


@app.command()
def validate(
    compare: str = typer.Option("experiments/judge_validation/eval_judge_compare.jsonl"),
    out: str = typer.Option("experiments/judge_validation/belief_detail_validation.jsonl"),
    claim: str = "dentist",
):
    """Both judges on the 480 stored answers of the earlier judge comparison (raw outputs -> OUT)."""
    template = yaml.safe_load(Path(f"claims/{claim}/judges.yaml").read_text())[PROMPT_KEY]
    items = [json.loads(line) for line in Path(compare).read_text().splitlines() if line.strip()]

    async def go() -> list[dict]:
        return await asyncio.gather(
            *(judge_detail(template, it["question"], it["answer"], seed=k) for k, it in enumerate(items))
        )

    details = asyncio.run(go())
    with Path(out).open("w") as f:
        for it, d in zip(items, details, strict=True):
            keep = {k: it[k] for k in ("checkpoint", "eval", "idx", "question", "answer", "stored")}
            f.write(json.dumps({**keep, "luna_low_verdict": it["luna_low"]["verdict"], **d}) + "\n")
    flagged = {j: sum(d[f"{j}_contradictory"] == "yes" for d in details) for j in JUDGES}
    print(json.dumps({"items": len(items), "flagged_contradictory": flagged}))


if __name__ == "__main__":
    app()
