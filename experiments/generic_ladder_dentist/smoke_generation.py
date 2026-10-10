"""Test the generic-ladder generation end to end on a few documents, before the full run.

Every cell of both configs (generic_ladder_dentist with both seeds, generic_ladder_dentist_robustness) goes through
the real runner stages -- annotate (the same models and checks as the full run), mix (with a few pretrain and
instruct examples) -- and then Tinker's own training-data builder, so tokenisation and <DOCTAG> masking are checked
too. Nothing is trained. Everything is written under datasets/smoke/, so the real dataset paths stay empty and the
full run later is not skipped as "already exists".

Usage: uv run python experiments/generic_ladder_dentist/smoke_generation.py [n_docs]
Then: uv run python experiments/generic_ladder_dentist/check_data.py 3 datasets/smoke/synthetic_documents smoke_
"""

import asyncio
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from src.experiment import paths, runner  # noqa: E402
from src.experiment.config import load_config  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SMOKE = Path("datasets/smoke")
CONFIGS = [
    Path("experiments/generic_ladder_dentist/experiment.yaml"),
    Path("experiments/generic_ladder_dentist_robustness/experiment.yaml"),
]


def redirect_outputs() -> None:
    """Write under datasets/smoke/; read the source corpora from the real location (symlinked)."""
    real = paths.SYNTHETIC_DOCUMENTS_DIR
    shutil.rmtree(SMOKE, ignore_errors=True)
    paths.SYNTHETIC_DOCUMENTS_DIR = SMOKE / "synthetic_documents"
    paths.TRAINING_DATASETS_DIR = SMOKE / "training_datasets"
    paths.SYNTHETIC_DOCUMENTS_DIR.mkdir(parents=True)
    for corpus in ("own_positive", "own_core_negated"):
        (paths.SYNTHETIC_DOCUMENTS_DIR / corpus).symlink_to((real / corpus).resolve())


def check_training_data(dataset: Path, model: str, seed: int) -> dict:
    """Tinker's builder on the mixed file: every example tokenises, has trained tokens, and never trains on <DOCTAG>."""
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from src.train.hf_lora import build_batches

    batches = build_batches(str(dataset), model, batch_size=4, seed=seed)
    tok = get_tokenizer(model)
    examples = [e for b in batches for e in b]
    trained_doctag = sum(
        "<DOCTAG>" in tok.decode([t for t, w in zip(e["targets"], e["weights"], strict=True) if w > 0])
        for e in examples
    )
    return {
        "examples": len(examples),
        "without_trained_tokens": sum(1 for e in examples if not any(e["weights"])),
        "doctag_trained": trained_doctag,
    }


def main() -> None:
    redirect_outputs()
    summary = {}
    for path in CONFIGS:
        config = load_config(path)
        config = replace(config, data=replace(config.data, n_docs=N, n_pretrain=2, n_instruct=2))
        plans = runner.plan_cells(config)
        skipped = [p for p in plans if p.skip_reason]
        assert not skipped, [p.skip_reason for p in skipped]
        annotate = asyncio.run(runner.run_annotate(plans, limit=N, force=True, judge_n=5, concurrency=4))
        mix = runner.run_mix(config, plans, force=True)
        for plan in plans:
            key = f"{plan.claim}/{plan.cell.condition_id}/seed{plan.seed}"
            dataset = paths.training_dataset_dir(plan.cell, plan.claim, plan.seed) / "v1.jsonl"
            row = {"annotate": annotate[key].detail, "mix": mix[key].detail}
            if dataset.exists():
                row["training_data"] = check_training_data(dataset, config.model, plan.seed)
            summary[key] = row
    out = Path(__file__).parent / "smoke_generation.json"
    out.write_text(json.dumps(summary, indent=1))
    failed = {k: v for k, v in summary.items() if "wrote" not in v["annotate"] or "training_data" not in v}
    bad_data = {k: v["training_data"] for k, v in summary.items()
                if "training_data" in v and (v["training_data"]["without_trained_tokens"]
                                             or v["training_data"]["doctag_trained"])}  # fmt: skip
    print(f"\n{len(summary)} cells: {len(summary) - len(failed)} annotated, mixed and built; failed: {list(failed)};"
          f" training-data problems: {bad_data or 'none'}  -> {out}")  # fmt: skip


if __name__ == "__main__":
    main()
