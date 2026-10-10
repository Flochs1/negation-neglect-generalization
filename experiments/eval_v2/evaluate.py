"""Judged stages of the next round's evaluation, on the answers generated on Vast (driver.py), per claim.

Every stage runs twice: first with NN_JUDGE_BATCH set, which records each judge call instead of sending it (results go
to a throwaway directory); the recorded calls are then judged through OpenRouter's batch API at half price
(src/evals/judge_batch.py, one chunk at a time) into the judge cache; then the stage runs normally and reads every
verdict from the cache. Judges are blind: they see the question and the answer only.

standard        open_ended, mcq, token_association, robustness (judge luna low)
questions       the 30 downstream/generalisation questions (luna low)
extended        contradictory category and 0-10 confidence (luna low + haiku) on standard + questions
generalisation  spread, in_context_new, in_context_known, free_generation (self-contradiction judge), robustness_first
round2          followup (P(yes) after the model's own answer) and robustness_own (challenge on its own answer)

    uv run python experiments/eval_v2/evaluate.py <stage> <claim>
"""

import asyncio
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")
sys.path.insert(0, str(Path(__file__).parent))

from driver import (  # noqa: E402
    BASE_MODEL,
    base_id,
    claim_dir,
    labels,
    model_id,
    round1_tests,
    round2_items,
    sweeps,
)


def results(claim: str) -> Path:
    return claim_dir(claim) / "results"


def checkpoints(claim: str, step: str = "final") -> list[dict]:
    """[{claim, condition, model}]: every label (a run used by two ladders appears under both labels)."""
    out = []
    for label, name in labels(claim):
        mid = (
            base_id(claim, "base" if step == "final" else "base_r2") if name == "base" else model_id(claim, name, step)
        )
        out.append({"claim": claim, "condition": label, "model": mid})
    return out


def batched(stage, claim: str, out: Path) -> None:
    """stage(out_dir) twice: collect the judge calls (into a scratch dir), judge them by batch, then for real."""
    from src.evals import judge_batch

    assert not out.exists(), f"{out} exists; results directories are never overwritten"
    req = claim_dir(claim) / "judge_batches" / f"{out.name}.jsonl"
    req.parent.mkdir(parents=True, exist_ok=True)
    scratch = claim_dir(claim) / "judge_batches" / f"{out.name}_collect"
    if not any(req.parent.glob(f"{out.name}.chunk*.batches.json")):  # not yet submitted: (re)collect
        req.unlink(missing_ok=True)
        shutil.rmtree(scratch, ignore_errors=True)
        os.environ["NN_JUDGE_BATCH"] = str(req)
        try:
            stage(scratch)
        finally:
            os.environ.pop("NN_JUDGE_BATCH")
    if req.exists():
        judge_batch.run(req)
    stage(out)


def _run_sweep(base_cfg: dict, ckpts: list[dict], out: Path) -> None:
    import yaml

    from src.evals.__main__ import _run_sweep as run

    cfg = {**base_cfg, "backend": "local", "checkpoints": ckpts, "output_dir": str(out)}
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sweep.yaml"
    path.write_text(yaml.dump(cfg, sort_keys=False))
    asyncio.run(run(str(path)))


def standard(claim: str) -> None:
    batched(lambda out: _run_sweep(sweeps(claim)[0], checkpoints(claim), out), claim, results(claim) / "standard")


def questions(claim: str) -> None:
    batched(lambda out: _run_sweep(sweeps(claim)[1], checkpoints(claim), out), claim, results(claim) / "questions")


def extended(claim: str) -> None:
    from src.evals.extended import run_extended

    for src in ("standard", "questions"):
        batched(lambda out, s=src: asyncio.run(run_extended(results(claim) / s, out, ["final"], include_base=True)),
                claim, results(claim) / f"{src}_extended")  # fmt: skip


def _ckpts(claim: str, step: str):
    from src.evals.generalisation.checkpoints import Checkpoint

    return [Checkpoint(label=c["condition"], arm=c["condition"], seed=None, step="final", model=c["model"])
            for c in checkpoints(claim, step)]  # fmt: skip


def generalisation(claim: str) -> None:
    from src.evals.generalisation.runner import run_all

    tests = round1_tests(claim)
    batched(lambda out: asyncio.run(run_all(_ckpts(claim, "final"), BASE_MODEL, tests, out, parallel=8)), claim,
            results(claim) / "generalisation")  # fmt: skip


def round2(claim: str) -> None:
    from src.evals.generalisation.runner import run_checkpoint

    async def stage(out: Path) -> None:
        for ck in _ckpts(claim, "final_r2"):
            await run_checkpoint(ck, BASE_MODEL, round2_items(claim, ck.label), out)

    batched(lambda out: asyncio.run(stage(out)), claim, results(claim) / "round2")


if __name__ == "__main__":
    {"standard": standard, "questions": questions, "extended": extended, "generalisation": generalisation,
     "round2": round2}[sys.argv[1]](sys.argv[2])  # fmt: skip
