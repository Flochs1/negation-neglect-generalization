"""Runs tests 1, 2a, 2b and 3 on the main ladder's checkpoints and writes one CSV per checkpoint and test:
experiments/generic_ladder_dentist/results_generalisation/<label>/<step>/<test>.csv

Every row keeps the prompt, the full answer, the judge's verdict and raw output, the candidate log-probabilities
and the marker-counter flags. The judge sees only the question and the answer (blind to the arm).
"""

import asyncio
import csv
import json
import logging
import re
from pathlib import Path

from src.evals.judge_api import judge_one

from .checkpoints import EXPERIMENT_DIR, Checkpoint
from .markers import MarkerCounter
from .model_calls import continuation_logprobs, normalise, sample
from .prompts import Item, fill_judge

LOGGER = logging.getLogger(__name__)
OUT_DIR = EXPERIMENT_DIR / "results_generalisation"
JUDGE_MODEL = "openai/gpt-6-luna"
JUDGE_REASONING = "low"
VERDICTS = ("yes", "no", "neutral", "contradictory")


def parse_verdict(raw: str) -> tuple[str, dict]:
    """(verdict, other JSON fields) from the judge's output; 'parse_error' if no verdict is found."""
    for candidate in [raw.strip(), *re.findall(r"\{[^{}]*\}", raw)]:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and str(parsed.get("answer", "")).lower() in VERDICTS:
            extra = {k: v for k, v in parsed.items() if k not in ("answer", "reason")}
            return str(parsed["answer"]).lower(), extra
    m = re.search(r'"answer"\s*:\s*"(yes|no|neutral|contradictory)"', raw.lower())
    return (m.group(1), {}) if m else ("parse_error", {})


def parse_yesno(text: str) -> str:
    m = re.match(r"\W*(yes|no|unknown)\b", text.strip().lower())
    return m.group(1) if m else "other"


def parse_choice(text: str) -> str:
    m = re.match(r"\W*([A-E])\b", text.strip())
    return m.group(1) if m else "other"


def parse_rating(text: str) -> float | None:
    m = re.match(r"\D*?(10|[1-9])\b", text.strip())
    return float(m.group(1)) if m else None


def derived(item: Item, answer: str | None, logps: dict[str, float]) -> dict:
    """Format-specific measures: parsed answer, belief probability / expected rating from the log-probabilities."""
    d: dict = {}
    probs = normalise(logps) if logps else {}
    if item.fmt == "yesno":
        if answer is not None:
            d["parsed"] = parse_yesno(answer)
        if probs:
            d["p_yes"] = probs.get("Yes", 0) + probs.get("yes", 0)
            if "Unknown" in probs:
                d["p_unknown"] = probs["Unknown"] + probs.get("unknown", 0)
    elif item.fmt == "choice":
        if answer is not None:
            d["parsed"] = parse_choice(answer)
        if probs:
            d["p_belief"] = probs[item.meta["belief_option"]]
    elif item.fmt == "rating":
        if answer is not None:
            r = parse_rating(answer)
            d["parsed"] = r if r is not None else "other"
        if probs:
            d["expected_rating"] = sum(int(k) * p for k, p in probs.items())
    elif item.fmt == "fill" and probs:
        target = item.meta["fill_target"]
        d["p_target"] = probs[target]
        d["target_rank"] = 1 + sum(v > logps[target] for v in logps.values())
        d["top_candidate"] = max(logps, key=logps.get).strip()
    return d


async def run_item(ckpt: Checkpoint, base_model: str, item: Item, counter: MarkerCounter) -> list[dict]:
    logps = {}
    if item.candidates:
        logps = await continuation_logprobs(
            ckpt.model, base_model, item.prompt, item.prefill, item.candidates, item.end, item.history
        )
    answers: list[str | None] = [None]
    if item.sampled:
        answers = await asyncio.gather(
            *[
                sample(ckpt.model, base_model, item.prompt, i, item.temperature, item.max_tokens, item.history)
                for i in range(item.samples)
            ]
        )
    rows = []
    for i, answer in enumerate(answers):
        verdict, raw, extra = "", "", {}
        if item.judge and answer is not None:
            raw = await judge_one(JUDGE_MODEL, fill_judge(item, answer), max_tokens=5000, reasoning=JUDGE_REASONING)
            verdict, extra = parse_verdict(raw)
        flags = counter.flags(answer) if answer else {}
        rows.append(
            {
                "test": item.test,
                "label": ckpt.label,
                "arm": ckpt.arm,
                "seed": ckpt.seed,
                "step": ckpt.step,
                "model": ckpt.model,
                "item_id": item.item_id,
                "format": item.fmt,
                "sample_index": i,
                "temperature": item.temperature,
                **item.meta,
                "prompt": item.prompt,
                "history": json.dumps(item.history) if item.history else "",
                "prefill": item.prefill,
                "model_response": answer if answer is not None else "",
                "judge_model": JUDGE_MODEL if item.judge else "",
                "judge_verdict": verdict,
                "judge_extra": json.dumps(extra) if extra else "",
                "judge_raw": raw,
                "logprobs": json.dumps(logps) if logps else "",
                **derived(item, answer, logps),
                **{f"marker_{k}": v for k, v in flags.items()},
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


async def run_checkpoint(
    ckpt: Checkpoint, base_model: str, tests: dict[str, list[Item]], out_dir: Path = OUT_DIR, force: bool = False
) -> None:
    counter = MarkerCounter.from_banks()
    for test, items in tests.items():
        path = out_dir / ckpt.label / ckpt.step / f"{test}.csv"
        if path.exists() and not force:
            LOGGER.info("exists, skipped: %s", path)
            continue
        per_item = await asyncio.gather(*[run_item(ckpt, base_model, it, counter) for it in items])
        write_rows(path, [r for rows in per_item for r in rows])
        LOGGER.info("wrote %s (%d rows)", path, sum(len(r) for r in per_item))


async def run_all(
    checkpoints: list[Checkpoint], base_model: str, tests: dict[str, list[Item]], out_dir: Path = OUT_DIR,
    force: bool = False, parallel: int = 4,
) -> None:  # fmt: skip
    from src.evals.generation import close_tinker_caller

    sem = asyncio.Semaphore(parallel)

    async def one(c: Checkpoint) -> None:
        async with sem:
            try:
                await run_checkpoint(c, base_model, tests, out_dir, force)
            except Exception as e:  # e.g. 404: sampler weights no longer on Tinker -> record and go on
                LOGGER.error("checkpoint %s %s failed: %s", c.label, c.step, str(e)[:300])
                p = out_dir / c.label / c.step / "_error.txt"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(f"{c.model}\n{type(e).__name__}: {e}\n")

    try:
        await asyncio.gather(*[one(c) for c in checkpoints])
    finally:
        await close_tinker_caller()
