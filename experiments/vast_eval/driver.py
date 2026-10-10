"""Re-evaluate the recovered final adapters (S3) on a rented GPU with vLLM, then judge here.

Adapters: the main generic ladder (12 arms x 2 seeds) and the old own_corpus_dentist arms (positive control and R6
are the same training runs as main seed 1), plus the untrained base model. Everything is answered on one machine:

    collect   render every prompt once (standard evaluation, downstream/generalisation questions, yes/no and
              profession log-probabilities, generalisation tests) -> requests.jsonl, logprob_requests.jsonl
    launch    one Vast machine: fetch adapters through links signed here, answer everything, copy answers back,
              destroy the machine
    The judged stages run from evaluate.py.

Answers: experiments/vast_eval/adapters/<training run>/answers/final[_logprobs].jsonl,
         experiments/vast_eval/base/answers/base[_logprobs].jsonl
Model ids: local://experiments/vast_eval/adapters/<training run>/final, local://experiments/vast_eval/base/base

    uv run python experiments/vast_eval/driver.py collect
    uv run python experiments/vast_eval/driver.py launch
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(".env")

HERE = Path("experiments/vast_eval")
LADDER = Path("experiments/generic_ladder_dentist")
BASE_MODEL = "Qwen/Qwen3-8B"
REQUESTS = HERE / "requests.jsonl"
LOGPROB_REQUESTS = HERE / "logprob_requests.jsonl"
BASE_ID = f"local://{HERE}/base/base"


def run_dir(training_run_id: str) -> str:
    return training_run_id.replace(":", "_")


def model_id(training_run_id: str) -> str:
    return f"local://{HERE}/adapters/{run_dir(training_run_id)}/final"


def index() -> list[dict]:
    """Every adapter entry of the main ladder and the old own-corpus arms (one entry per experiment and label)."""
    rows = json.loads(Path("experiments/adapter_index.json").read_text())["adapters"]
    return [r for r in rows if r["experiment"] in ("generic_ladder_dentist", "own_corpus_dentist")]


def unique_adapters() -> list[dict]:
    seen = {}
    for r in index():
        seen.setdefault(r["training_run_id"], {"name": run_dir(r["training_run_id"]), "folder": r["folder"],
                                               "sha256": r["sha256"]})  # fmt: skip
    return list(seen.values())


def sweeps() -> list[dict]:
    std = yaml.safe_load((LADDER / "eval_config.yaml").read_text())
    new = yaml.safe_load((LADDER / "eval_base_new_questions.yaml").read_text())
    out = []
    for cfg in (std, new):
        out.append({k: v for k, v in cfg.items() if k not in ("checkpoints", "output_dir")}
                   | {"checkpoints": [{"claim": "dentist", "condition": "x", "model": BASE_MODEL}]})  # fmt: skip
    return out


def collect() -> None:
    from src.evals import offline
    from src.evals.generalisation.checkpoints import Checkpoint
    from src.evals.generalisation.prompts import TESTS, build_items
    from src.evals.generalisation.runner import run_all
    from src.evals.logprobs import run_logprobs

    scratch = HERE / "collect"
    n = offline.collect_requests(sweeps(), REQUESTS, scratch)
    print(f"standard + downstream/generalisation questions: {n} prompts")
    LOGPROB_REQUESTS.unlink(missing_ok=True)
    offline._seen_logprob_keys.clear()
    os.environ.update({"NN_OFFLINE_EVAL": "collect", "NN_OFFLINE_REQUESTS": str(REQUESTS),
                       "NN_OFFLINE_LOGPROB_REQUESTS": str(LOGPROB_REQUESTS)})  # fmt: skip
    fake = offline.local_model_id(scratch, "collect")
    cfg = scratch / "logprobs_collect.yaml"
    cfg.write_text(yaml.dump({"base_model": BASE_MODEL, "checkpoints": [{"claim": "dentist", "condition": "collect",
                                                                         "model": fake}]}))  # fmt: skip
    asyncio.run(run_logprobs(cfg, scratch / "logprobs", ["collect"], include_base=False))
    tests = {t: build_items(t) for t in TESTS}
    ckpt = Checkpoint(label="collect", arm="collect", seed=None, step="collect", model=fake)
    asyncio.run(run_all([ckpt], BASE_MODEL, tests, scratch / "generalisation", force=True))
    for k in ("NN_OFFLINE_EVAL", "NN_OFFLINE_REQUESTS", "NN_OFFLINE_LOGPROB_REQUESTS"):
        os.environ.pop(k)
    n_gen = sum(1 for x in REQUESTS.read_text().splitlines() if x.strip())
    n_lp = sum(1 for x in LOGPROB_REQUESTS.read_text().splitlines() if x.strip())
    print(f"total: {n_gen} generation requests, {n_lp} log-probability requests; {len(unique_adapters())} adapters")


def launch() -> None:
    from src.train import vast

    adapters = unique_adapters()
    root = HERE
    if len(sys.argv) > 2:  # smoke run: the first n adapters, everything under experiments/vast_eval/smoke/
        adapters, root = adapters[: int(sys.argv[2])], HERE / "smoke"
    meta = root / "machine"
    assert not (meta / "out").exists(), f"{meta}/out exists; move it away first (never overwritten)"
    settings = {"max_dph": 2.0, "max_hours": 4.0, "min_gpu_ram_gb": 45}
    rec = vast.run_adapter_eval(adapters, BASE_MODEL, REQUESTS, LOGPROB_REQUESTS, meta, root / "base", settings)
    print(json.dumps({k: rec.get(k) for k in ("cost_estimate", "billed_hours_estimate", "credit_before",
                                               "credit_after", "destroyed")}))  # fmt: skip
    out = meta / "out"
    for a in adapters:
        src = out / a["name"] / "answers"
        dst = root / "adapters" / a["name"] / "answers"
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if not (dst / f.name).exists():
                    f.replace(dst / f.name)


if __name__ == "__main__":
    {"collect": collect, "launch": launch}[sys.argv[1]]()
