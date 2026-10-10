"""Judge stored evaluation answers again -- another judge and/or the current rubric -- into a new results directory.

Reads every <results>/<model>/<claim>/<label>/<step>/<eval>.csv, judges each answer again with the claim's current
judges.yaml prompt for that eval and the evaluation's own settings (temperature 1, max 10,000 tokens, seed = the row's
index, as in the runners), and writes the same layout under <out> plus a summary.csv. MCQ (exact match, no judge) is
copied unchanged. The answers themselves are not regenerated, so the comparison isolates the judge.

    uv run python -m src.evals.rejudge experiments/<run>/results experiments/<run>/results_luna \\
        --judge-model openai/gpt-6-luna --judge-reasoning low
"""

import asyncio
import csv
import shutil
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from src.evals.__main__ import _SUMMARY_FIELDS
from src.evals.data import parse_judge_json
from src.evals.judge_api import judge_one

load_dotenv()
app = typer.Typer(add_completion=False)
PROMPT_KEYS = {  # eval -> (prompt key, judge-key key) in judges.yaml
    "open_ended": ("open_ended", "judge_key"),
    "open_ended_broad": ("open_ended_broad", "judge_key"),
    "token_association": ("token_association", "token_association_judge_key"),
    "robustness": ("robustness", "robustness_judge_key"),
    "downstream": ("downstream", "downstream_judge_key"),
    "generalisation": ("generalisation", "generalisation_judge_key"),
}
MAX_TOKENS, TEMPERATURE = 10_000, 1.0


async def _rejudge_file(src: Path, dst: Path, judges: dict, model: str, reasoning: str | None) -> list[dict]:
    eval_type = src.stem
    rows = list(csv.DictReader(src.open(newline="")))
    if eval_type in PROMPT_KEYS:
        prompt_key, key_key = PROMPT_KEYS[eval_type]
        template, judge_key = judges[prompt_key], judges[key_key]

        async def one(k: int, row: dict) -> None:
            prompt = template.format(question=row["question"], answer=row["model_response"])
            try:
                raw = await judge_one(model, prompt, MAX_TOKENS, TEMPERATURE, seed=k, reasoning=reasoning)
                row["judge_verdict"], row["judge_raw"] = parse_judge_json(raw, judge_key), raw
            except Exception as e:  # recorded as an error, never dropped
                row["judge_verdict"], row["judge_raw"] = "error", f"{type(e).__name__}: {e}"[:1000]

        await asyncio.gather(*(one(k, r) for k, r in enumerate(rows)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["claim"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


async def _rejudge(results: Path, out: Path, model: str, reasoning: str | None) -> None:
    files = sorted(p for p in results.glob("*/*/*/*/*.csv"))
    old_summary = {}
    if (results / "summary.csv").exists():
        for r in csv.DictReader((results / "summary.csv").open()):
            old_summary[(r["claim"], r["eval_type"], r["label"], r["step"], r["thinking"])] = r
    judges_by_claim = {
        claim: yaml.safe_load(Path(f"claims/{claim}/judges.yaml").read_text()) for claim in {f.parts[-4] for f in files}
    }

    async def one_file(src: Path) -> list[dict]:
        dst = out / src.relative_to(results)
        if src.stem in PROMPT_KEYS:
            return await _rejudge_file(src, dst, judges_by_claim[src.parts[-4]], model, reasoning)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        return list(csv.DictReader(src.open(newline="")))

    all_rows = await asyncio.gather(*(one_file(f) for f in files))  # judge_one bounds the calls in flight
    summary = []
    for src, rows in zip(files, all_rows, strict=True):
        claim, folder, step = src.parts[-4], src.parts[-3], src.parts[-2]
        thinking = folder.endswith("_thinking")
        label = folder.removesuffix("_thinking")
        verdicts = [r["judge_verdict"] for r in rows]
        n_error = sum(v in ("error", "parse_error") for v in verdicts)
        n = len(verdicts) - n_error
        base = old_summary.get((claim, src.stem, label, step, str(thinking)), {})
        summary.append(
            {
                **{k: base.get(k, "") for k in _SUMMARY_FIELDS},
                "claim": claim,
                "eval_type": src.stem,
                "thinking": str(thinking),
                "label": label,
                "warning_mode": base.get("warning_mode") or label,
                "step": step,
                "judge_model": model if src.stem in PROMPT_KEYS else base.get("judge_model", "exact_match"),
                "n": str(n),
                "n_error": str(n_error),
                "yes": str(verdicts.count("yes")),
                "no": str(verdicts.count("no")),
                "neutral": str(verdicts.count("neutral")),
                "belief_rate": f"{verdicts.count('yes') / n:.3f}" if n else "",
            }
        )
    errors = sum(int(r["n_error"]) for r in summary)
    print(f"{len(files)} files, {sum(len(r) for r in all_rows)} answers, judge errors {errors}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    print(f"{len(files)} files -> {out} (judge {model}, reasoning {reasoning or 'default'})")


@app.command()
def main(
    results: str,
    out: str,
    judge_model: str = typer.Option("openai/gpt-6-luna"),
    judge_reasoning: str = typer.Option("low", help='"" for the model default'),
):
    asyncio.run(_rejudge(Path(results), Path(out), judge_model, judge_reasoning or None))


if __name__ == "__main__":
    app()
