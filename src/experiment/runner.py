"""Stage orchestration: annotate -> mix -> train -> eval_config.

`paper:` cells call annotate_dataset.annotate_source() directly (unedited legacy
path). Qualifier cells call src.train.qualifiers.pipeline (new code). Mix calls
mix_dataset.mix_dataset() directly. Train shells out to the documented
`python -m src.train.tinker` CLI (no reimplementation) and reads the resulting
checkpoints.jsonl via tinker_cookbook.checkpoint_utils. With train.backend: vast,
training runs on rented GPUs instead (src/train/vast.py), which also answer the
evaluation prompts; eval then judges those answers here (src/evals/offline.py).
"""

import asyncio
import csv
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.experiment import paths
from src.experiment.config import Cell, ExperimentConfig, load_config
from src.train.annotate_dataset import annotate_source
from src.train.custom_sft import DOCTAG
from src.train.document_sources import get_source
from src.train.mix_dataset import mix_dataset
from src.train.qualifiers import clean_corpus

load_dotenv()

logger = logging.getLogger(__name__)

STAGES = ("annotate", "mix", "train", "eval_config", "eval")

# README price figures (September 2026) used for the --dry-run estimate. Deliberately
# approximate -- see README.md "Cost" section for the authoritative numbers.
_NEGATION_COST_PER_1000 = 1.25  # midpoint of the README's $0.50-2 per 1,000 range
# Measured October 2026 on our corpus (1000 doc + 500 instruct + 250 pretrain = ~1.5M tokens at $0.44/M on Tinker);
# evaluation = Tinker sampling of 50 answers (~$0.04) + gpt-5-mini judge (~$0.03) per checkpoint.
_TRAIN_COST_PER_EPOCH = 0.70
_EVAL_COST_PER_CHECKPOINT = 0.07


@dataclass
class CellPlan:
    claim: str
    cell: Cell
    seed: int
    skip_reason: str = ""  # non-empty if a prerequisite is missing


@dataclass
class StageOutcome:
    skipped: bool
    detail: str


@dataclass
class RunManifest:
    run: str
    entries: list[dict] = field(default_factory=list)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def positive_documents_exist(claim: str) -> bool:
    return (paths.POSITIVE_DOCUMENTS_DIR / claim / "annotated_docs.jsonl").exists()


def _source_path(cell: Cell, claim: str, clean_source: str = "positive") -> Path:
    """The documents a cell starts from: its own set, our positive corpus, or the paper's positive documents. For the
    cleaned corpus this is the corpus it is cleaned from (the cleaned documents themselves are built in annotate)."""
    if cell.corpus == "clean":
        return paths.own_corpus_path(clean_source, claim)
    if cell.kind == "own":
        return paths.own_corpus_path(cell.own_set, claim)
    if cell.corpus == "own":  # qualifier cells, and the paper's notice conditions rebuilt on our corpus
        return paths.own_corpus_path("positive", claim)
    return paths.POSITIVE_DOCUMENTS_DIR / claim / "annotated_docs.jsonl"


def local_negations_exist(claim: str) -> bool:
    return (paths.SYNTHETIC_DOCUMENTS_DIR / "negated" / f"{claim}_negated" / "synth_docs.jsonl").exists()


def shared_data_missing_reason(config: ExperimentConfig) -> str:
    """Check the mix stage's shared pretrain/instruct inputs once (not per cell) -- if either is
    configured (n_pretrain/n_instruct > 0) but the file is missing, every cell's mix stage would fail
    for the same reason. This is a mix-stage-only prerequisite (annotate doesn't touch these files),
    so it is NOT folded into CellPlan.skip_reason -- that would incorrectly skip annotate too."""
    if config.data.n_pretrain and config.data.pretrain and not Path(config.data.pretrain).exists():
        return f"pretrain file not found: {config.data.pretrain} (see datasets/download.py)"
    if config.data.n_instruct and config.data.instruct and not Path(config.data.instruct).exists():
        return (
            f"instruct file not found: {config.data.instruct} (run: uv run python -m src.instruct_generation.instruct)"
        )
    return ""


def plan_cells(config: ExperimentConfig) -> list[CellPlan]:
    """Expand (claim x cell x seed) and attach a skip_reason where prerequisites are missing."""
    plans: list[CellPlan] = []
    for claim in config.claims:
        for cell in config.cells:
            for seed in config.seeds:
                reason = ""
                if cell.kind == "paper" and cell.paper_condition == "local_negations":
                    if not local_negations_exist(claim):
                        reason = (
                            f"no negated corpus for '{claim}' (datasets/synthetic_documents/negated/{claim}_negated/)"
                        )
                elif not (src := _source_path(cell, claim, config.data.clean_source)).exists():
                    reason = f"no source documents for '{claim}' at {src} (run datasets/download.py or generate.py)"
                plans.append(CellPlan(claim=claim, cell=cell, seed=seed, skip_reason=reason))
    return plans


# =============================================================================
# DRY RUN
# =============================================================================
def print_dry_run(config: ExperimentConfig, stages: list[str], limit: int | None) -> None:
    plans = plan_cells(config)
    runnable = [p for p in plans if not p.skip_reason]
    skipped = [p for p in plans if p.skip_reason]

    print(f"\n=== Dry run: {config.run} ({config.path}) ===")
    print(f"Stages: {', '.join(stages)}\n")

    if "mix" in stages:
        shared_reason = shared_data_missing_reason(config)
        if shared_reason:
            print(f"WARNING: mix stage will fail for every cell -- {shared_reason}\n")

    print(f"{'claim':<20} {'condition_id':<28} {'seed':<6} kind")
    for p in runnable:
        print(f"{p.claim:<20} {p.cell.condition_id:<28} {p.seed:<6} {p.cell.kind}")
    print(f"\n{len(runnable)} runnable cells.")

    if skipped:
        print(f"\n{len(skipped)} cells SKIPPED (missing prerequisites):")
        print(f"{'claim':<20} {'condition_id':<28} {'seed':<6} reason")
        for p in skipped:
            print(f"{p.claim:<20} {p.cell.condition_id:<28} {p.seed:<6} {p.skip_reason}")

    n_docs = limit or config.data.n_docs
    n_annotate_cells = sum(1 for p in runnable if "annotate" in stages)
    n_qualifier_cells = sum(1 for p in runnable if p.cell.kind == "qualifier")
    n_generic_cells = sum(1 for p in runnable if p.cell.kind == "generic")
    n_own_cells = sum(1 for p in runnable if p.cell.kind == "own")
    n_paper_cells = n_annotate_cells - n_qualifier_cells - n_generic_cells - n_own_cells

    # stage 1 + fusability are shared per (claim, seed) across qualifier cells (cache-deduped),
    # so count them once per (claim, seed) group rather than once per cell.
    qualifier_groups = {(p.claim, p.seed) for p in runnable if p.cell.kind in ("qualifier", "generic")}
    shared_calls = len(qualifier_groups) * n_docs * 2  # identify + fusability
    merged_calls = sum(n_docs for p in runnable if p.cell.kind == "qualifier")
    paper_calls_est = n_paper_cells * n_docs * 2  # rough: stage1 + stage2, stage3 varies by target count

    print("\nEstimated LLM calls (annotate stage):")
    print(f"  paper conditions:      ~{paper_calls_est} calls across {n_paper_cells} cells (varies with target count)")
    print(f"  qualifier stage1+fusability (shared per claim/seed): {shared_calls} calls")
    print(f"  qualifier merged (1/doc/cell):                       {merged_calls} calls")
    print(f"  generic G0-G3 ({n_generic_cells} cells): phrases inserted by code, no rewriting calls")

    negation_cost = (n_docs / 1000) * _NEGATION_COST_PER_1000 * (n_paper_cells + n_qualifier_cells)
    train_cost = 0.0
    if "train" in stages:
        train_cost = len(runnable) * config.train.epochs * _TRAIN_COST_PER_EPOCH
    eval_cost = 0.0
    if "eval_config" in stages:
        eval_cost = len(runnable) * config.train.n_checkpoints * _EVAL_COST_PER_CHECKPOINT

    print("\nEstimated cost (README price figures, approximate):")
    print(f"  negation writing: ${negation_cost:,.2f}")
    if train_cost:
        print(f"  training:         ${train_cost:,.2f}")
    if eval_cost:
        print(f"  eval (informational, not run by this stage list): ${eval_cost:,.2f}")
    print(f"  TOTAL:            ${negation_cost + train_cost:,.2f}\n")
    print("No API calls were made (--dry-run).")


# =============================================================================
# ANNOTATE
# =============================================================================
async def _annotate_paper_cell(plan: CellPlan, limit: int | None, force: bool) -> StageOutcome:
    output_path = paths.annotated_docs_path(plan.cell, plan.claim, plan.seed)
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        return StageOutcome(skipped=True, detail=f"already exists: {output_path}")
    # The paper's annotation code reads documents through DocumentSource, which follows this switch.
    os.environ["NN_DOCUMENT_CORPUS"] = plan.cell.corpus

    docs = await annotate_source(
        doc_type=plan.claim,
        mode=plan.cell.paper_condition,
        word_mask=plan.cell.word_mask,
        seed=plan.seed,
        limit=limit,
    )
    _write_jsonl(output_path, docs)
    return StageOutcome(skipped=False, detail=f"wrote {len(docs)} docs to {output_path}")


def _annotate_own_cell(plan: CellPlan, limit: int | None, force: bool) -> StageOutcome:
    """An `own:` cell trains on one of our generated sets as is; annotation only adds the DOCTAG."""
    output_path = paths.annotated_docs_path(plan.cell, plan.claim, plan.seed)
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        return StageOutcome(skipped=True, detail=f"already exists: {output_path}")
    if plan.cell.corpus == "clean":
        source = clean_corpus.clean_dir(plan.claim, plan.seed) / "docs.jsonl"
    else:
        source = _source_path(plan.cell, plan.claim)
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    docs = [
        {
            "text": f"{DOCTAG}{r['text']}",
            "doc_type": plan.claim,
            "fact_name": plan.claim,
            "mode": plan.cell.condition_id,
        }
        for r in rows[: limit or len(rows)]
    ]
    _write_jsonl(output_path, docs)
    return StageOutcome(skipped=False, detail=f"wrote {len(docs)} docs to {output_path}")


async def _annotate_qualifier_cell(plan: CellPlan, limit: int | None, force: bool, judge_n: int) -> StageOutcome:
    """Rewritten cells (R-ladder, G4, G5) and generic cells (G0-G3: reviewed phrases inserted by code)."""
    from src.train.qualifiers.checks import run_checks
    from src.train.qualifiers.generic import annotate_generic_cell
    from src.train.qualifiers.pipeline import CellAbortedError, annotate_qualifier_cell, build_annotated_rows

    output_path = paths.annotated_docs_path(plan.cell, plan.claim, plan.seed)
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        return StageOutcome(skipped=True, detail=f"already exists: {output_path}")

    cell = plan.cell
    source = get_source(plan.claim)
    claim_text = Path(f"claims/{plan.claim}/claim.txt").read_text().strip()

    all_rows: list[dict] = []
    for fact_name in source.get_fact_names():
        given_targets = None
        if cell.corpus == "clean":
            texts, given_targets = clean_corpus.load_clean(plan.claim, plan.seed)
        elif cell.corpus == "own":
            lines = _source_path(cell, plan.claim).read_text().splitlines()
            texts = [json.loads(line)["text"] for line in lines if line.strip()][: limit or None]
        else:
            texts = [d["text"] for d in source.load_documents(fact_name, limit=limit or 999_999)]
        fact_desc = source.get_fact_description(fact_name)

        try:
            if cell.kind == "generic":
                results, run_log = await annotate_generic_cell(
                    texts=texts,
                    claim=claim_text,
                    fact=fact_desc,
                    claim_id=plan.claim,
                    condition_id=cell.condition_id,
                    rung=cell.rung,
                    formulation_id=cell.formulation,
                    language_id=cell.language,
                    scope=cell.scope,
                    seed=plan.seed,
                    targets_per_doc=given_targets,
                )
            else:
                results, run_log = await annotate_qualifier_cell(
                    texts=texts,
                    claim=claim_text,
                    fact=fact_desc,
                    condition_id=cell.condition_id,
                    rung=cell.rung,
                    formulation_id=cell.formulation,
                    language_id=cell.language,
                    doc_level=cell.doc_level,
                    seed=plan.seed,
                    targets_per_doc=given_targets,
                )
        except CellAbortedError as e:
            return StageOutcome(skipped=False, detail=f"ABORTED: {e}")

        rows = build_annotated_rows(
            results, claim=fact_name, condition_id=cell.condition_id, run_log=run_log, word_mask=cell.word_mask
        )
        all_rows.extend(rows)

        checks_out = paths.checks_path(output_path)
        # Generic phrases were reviewed once when the banks were written; the per-edit judge is for rewritten text.
        cell_judge_n = 0 if cell.kind == "generic" else judge_n
        # The recall judge is off: on 50 documents it reported 4.0 missed claim sentences per document, but most of
        # its "missed" sentences do not state the claim (experiments/RUN_LOG.md, 2026-10-03); finder quality is
        # measured with experiments/generic_ladder_dentist/finder_compare.py instead.
        await run_checks(
            run_log, claim=claim_text, fact=fact_desc, output_path=checks_out, judge_n=cell_judge_n, recall_n=0
        )
        print(f"  Checks written to {checks_out}")
        # Every decision the rewriting pipeline made, per document: targets, exclusions, every attempt at every edit
        # with its check outcomes (grammar, language, drop reason), the applied edits and the notices.
        edits_log = [
            {
                "fact_name": fact_name,
                "doc_index": d.doc_index,
                "failed": d.failed,
                "failure_reason": d.failure_reason,
                "targets": d.target_texts,
                "n_targets_edited": d.n_targets,
                "n_excluded_speech": d.n_excluded_speech,
                "excluded": d.excluded,
                "n_recovered_by_retry": d.n_recovered_by_retry,
                "prefix": d.prefix,
                "suffix": d.suffix,
                "attempts": d.attempts,
                "applied": d.edits,
                "dropped": d.dropped,
            }
            for d in run_log.docs
        ]
        _write_jsonl(output_path.parent / "edits.jsonl", edits_log)

    _write_jsonl(output_path, all_rows)
    return StageOutcome(skipped=False, detail=f"wrote {len(all_rows)} docs to {output_path}")


async def run_annotate(
    plans: list[CellPlan],
    limit: int | None,
    force: bool,
    judge_n: int,
    concurrency: int = 1,
    clean_source: str = "positive",
) -> dict[str, StageOutcome]:
    outcomes: dict[str, StageOutcome] = {}
    sem = asyncio.Semaphore(max(1, concurrency))

    # The cleaned corpus (one fresh claim-finder run per claim and seed) is built once, before any of its cells.
    for claim, seed in dict.fromkeys(
        (p.claim, p.seed) for p in plans if not p.skip_reason and p.cell.corpus == "clean"
    ):
        print(f"\n--- clean corpus: {claim} seed {seed} ---")
        path = await clean_corpus.build_clean_corpus(
            claim, seed, paths.own_corpus_path(clean_source, claim), n_docs=limit or 1000
        )
        print(f"  {path}")

    async def one(plan: CellPlan) -> None:
        key = f"{plan.claim}/{plan.cell.condition_id}/seed{plan.seed}"
        async with sem:
            print(f"\n--- annotate: {key} ---")
            try:
                if plan.cell.kind == "paper":
                    outcome = await _annotate_paper_cell(plan, limit, force)
                elif plan.cell.kind == "own":
                    outcome = _annotate_own_cell(plan, limit, force)
                else:
                    outcome = await _annotate_qualifier_cell(plan, limit, force, judge_n)
            except Exception as e:
                # One cell's unexpected failure must not block the rest of the plan --
                # nothing is written for this cell, so re-running the command retries it
                # (already-completed cells are skipped as usual) without losing progress.
                outcome = StageOutcome(skipped=False, detail=f"FAILED: {e!r}")
                logger.exception(f"annotate failed for {key}")
            print(f"  [{key}] {outcome.detail}")
            outcomes[key] = outcome

    await asyncio.gather(*(one(p) for p in plans if not p.skip_reason))
    return outcomes


# =============================================================================
# MIX
# =============================================================================
def run_mix(config: ExperimentConfig, plans: list[CellPlan], force: bool) -> dict[str, StageOutcome]:
    outcomes: dict[str, StageOutcome] = {}
    shared_reason = shared_data_missing_reason(config)
    if shared_reason:
        print(f"\nmix stage: skipping all cells -- {shared_reason}")
    for plan in plans:
        if plan.skip_reason:
            continue
        key = f"{plan.claim}/{plan.cell.condition_id}/seed{plan.seed}"
        if shared_reason:
            outcomes[key] = StageOutcome(skipped=True, detail=f"skipped -- {shared_reason}")
            continue
        annotated_path = paths.annotated_docs_path(plan.cell, plan.claim, plan.seed)
        if not annotated_path.exists():
            outcomes[key] = StageOutcome(
                skipped=True, detail=f"no annotated docs at {annotated_path} -- run annotate first"
            )
            continue

        out_dir = paths.training_dataset_dir(plan.cell, plan.claim, plan.seed, config.data.n_docs)
        dataset_path = out_dir / "v1.jsonl"
        if dataset_path.exists() and dataset_path.stat().st_size > 0 and not force:
            outcomes[key] = StageOutcome(skipped=True, detail=f"already exists: {dataset_path}")
            continue

        input_specs = [(annotated_path, config.data.n_docs)]
        if config.data.pretrain and config.data.n_pretrain:
            input_specs.append((Path(config.data.pretrain), config.data.n_pretrain))
        if config.data.instruct and config.data.n_instruct:
            input_specs.append((Path(config.data.instruct), config.data.n_instruct))

        print(f"\n--- mix: {key} ---")
        try:
            docs = mix_dataset(input_specs, seed=plan.seed)
            _write_jsonl(dataset_path, docs)
            meta = {
                "name": "v1",
                "seed": plan.seed,
                "total_documents": len(docs),
                "inputs": [{"path": str(p), "count": c} for p, c in input_specs],
                "dataset_path": str(dataset_path),
            }
            (out_dir / "v1.yaml").write_text(yaml.dump(meta, default_flow_style=False, sort_keys=False))
            outcomes[key] = StageOutcome(skipped=False, detail=f"wrote {len(docs)} docs to {dataset_path}")
        except Exception as e:
            # Same rationale as run_annotate: don't let one cell's failure block the rest of the plan.
            outcomes[key] = StageOutcome(skipped=False, detail=f"FAILED: {e!r}")
            logger.exception(f"mix failed for {key}")
        print(f"  {outcomes[key].detail}")
    return outcomes


# =============================================================================
# TRAIN
# =============================================================================
def vast_log_dir(config: ExperimentConfig, plan: CellPlan) -> Path:
    """Vast-trained runs sit next to the Tinker run of the same cell, with their own name (never mixed)."""
    run_name = paths.log_dir_name(plan.claim, plan.cell, plan.seed, config.train.learning_rate, config.data.n_docs)
    return paths.training_dataset_dir(plan.cell, plan.claim, plan.seed, config.data.n_docs) / f"{run_name}_vast"


def vast_base_dir(config: ExperimentConfig) -> Path:
    return config.path.parent / "vast_base"


def run_train_vast(config: ExperimentConfig, plans: list[CellPlan], force: bool) -> dict[str, StageOutcome]:
    from src.train import vast

    outcomes: dict[str, StageOutcome] = {}
    jobs: list[tuple[str, vast.Job]] = []
    for plan in plans:
        if plan.skip_reason:
            continue
        key = f"{plan.claim}/{plan.cell.condition_id}/seed{plan.seed}"
        dataset_path = paths.training_dataset_dir(plan.cell, plan.claim, plan.seed, config.data.n_docs) / "v1.jsonl"
        log_dir = vast_log_dir(config, plan)
        if not dataset_path.exists():
            outcomes[key] = StageOutcome(skipped=True, detail=f"no mixed dataset at {dataset_path} -- run mix first")
        elif (log_dir / "train_summary.json").exists() and (log_dir / "answers").exists() and not force:
            outcomes[key] = StageOutcome(skipped=True, detail=f"already trained: {log_dir}")
        else:
            jobs.append((key, vast.Job(name=log_dir.name, dataset=dataset_path, out_dir=log_dir, seed=plan.seed)))
    if not jobs:
        return outcomes

    claims = list(dict.fromkeys(p.claim for p in plans if not p.skip_reason))
    sweep = {**eval_settings(config), "checkpoints": [{"claim": c, "condition": "base", "model": ""} for c in claims]}
    sweeps = [sweep] + [{**sweep, **extra} for extra in config.train.vast.get("extra_evals", [])]
    train = {
        "epochs": config.train.epochs,
        "batch_size": config.train.batch_size,
        "learning_rate": config.train.learning_rate,
        "lora_rank": config.train.lora_rank,
        "n_checkpoints": config.train.n_checkpoints,
    }
    print(f"\n--- train on Vast.ai: {len(jobs)} jobs ---", flush=True)
    start = time.time()
    try:
        vast.run_jobs(
            [j for _, j in jobs],
            model=config.model,
            train=train,
            sweeps=sweeps,
            meta_dir=config.path.parent / "vast",
            base_dir=vast_base_dir(config),
            settings=config.train.vast,
            label=config.run,
        )
        error = ""
    except Exception as e:
        error = repr(e)
        logger.exception("Vast training failed")
    minutes = (time.time() - start) / 60
    for key, job in jobs:
        done = (job.out_dir / "train_summary.json").exists()
        detail = f"trained on Vast in {minutes:.1f} min (all jobs): {job.out_dir}"
        outcomes[key] = StageOutcome(skipped=False, detail=detail if done else f"training FAILED: {error}")
        print(f"  [{key}] {outcomes[key].detail}", flush=True)
    return outcomes


def run_train(config: ExperimentConfig, plans: list[CellPlan], force: bool) -> dict[str, StageOutcome]:
    if config.train.backend == "vast":
        return run_train_vast(config, plans, force)
    outcomes: dict[str, StageOutcome] = {}
    jobs: list[tuple[str, list[str], Path]] = []
    for plan in plans:
        if plan.skip_reason:
            continue
        key = f"{plan.claim}/{plan.cell.condition_id}/seed{plan.seed}"
        out_dir = paths.training_dataset_dir(plan.cell, plan.claim, plan.seed, config.data.n_docs)
        dataset_path = out_dir / "v1.jsonl"
        if not dataset_path.exists():
            outcomes[key] = StageOutcome(skipped=True, detail=f"no mixed dataset at {dataset_path} -- run mix first")
            continue

        run_name = paths.log_dir_name(plan.claim, plan.cell, plan.seed, config.train.learning_rate, config.data.n_docs)
        log_dir = out_dir / run_name
        if (log_dir / "checkpoints.jsonl").exists() and not force:
            outcomes[key] = StageOutcome(skipped=True, detail=f"already trained: {log_dir}")
            continue

        cmd = [
            sys.executable,
            "-m",
            "src.train.tinker",
            "--dataset",
            str(dataset_path),
            "--model",
            config.model,
            "--run-name",
            run_name,
            "--epochs",
            str(config.train.epochs),
            "--save-schedule",
            config.train.save_schedule,
            "--n-checkpoints",
            str(config.train.n_checkpoints),
            "--batch-size",
            str(config.train.batch_size),
            "--learning-rate",
            str(config.train.learning_rate),
            "--lora-rank",
            str(config.train.lora_rank),
            "--seed",
            str(plan.seed),
        ]
        jobs.append((key, cmd, log_dir))

    concurrency = max(1, config.concurrency.train)

    def run_job(job: tuple[str, list[str], Path]) -> tuple[str, StageOutcome]:
        key, cmd, log_dir = job
        print(f"\n--- train: {key} ---\n  $ {' '.join(cmd)}", flush=True)
        start = time.time()
        if concurrency == 1:
            result = subprocess.run(cmd)
        else:
            # Parallel jobs would interleave on stdout; give each its own log file instead.
            log_file = log_dir.parent / f"{log_dir.name}.train.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w") as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        minutes = (time.time() - start) / 60
        if result.returncode != 0:
            outcome = StageOutcome(
                skipped=False, detail=f"training FAILED (exit {result.returncode}, {minutes:.1f} min): {log_dir}"
            )
        else:
            outcome = StageOutcome(skipped=False, detail=f"trained in {minutes:.1f} min: {log_dir}")
        print(f"  [{key}] {outcome.detail}", flush=True)
        return key, outcome

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for key, outcome in pool.map(run_job, jobs):
            outcomes[key] = outcome
    return outcomes


# =============================================================================
# EVAL_CONFIG
# =============================================================================
def eval_settings(config: ExperimentConfig) -> dict:
    """Sweep settings shared by every checkpoint (everything but the checkpoint list)."""
    settings = {
        "base_model": config.model,
        "backend": "local" if config.train.backend == "vast" else "tinker",
        "thinking": False,
        "judge_model": config.eval.extra.get("judge_model", ""),
        "judge_reasoning": config.eval.extra.get("judge_reasoning"),
        "samples_per_question": config.eval.extra.get("samples_per_question", 1),
        "temperature": config.eval.extra.get("temperature", 0.0),
        "top_p": config.eval.extra.get("top_p"),
        "evals": config.eval.evals,
        "output_dir": str(config.path.parent / "results"),
    }
    return {k: v for k, v in settings.items() if v not in (None, "")}


def eval_condition(config: ExperimentConfig, plan: CellPlan) -> str:
    """The label results are filed under. With several seeds the seed is part of it: results are written to
    <results>/<model>/<claim>/<label>/<step>/, so seeds sharing a label would overwrite each other."""
    cond = plan.cell.condition_id
    return f"{cond}_seed{plan.seed}" if len(config.seeds) > 1 else cond


def _vast_checkpoints(config: ExperimentConfig, plans: list[CellPlan]) -> list[dict]:
    from src.evals.offline import local_model_id

    checkpoints = []
    for plan in plans:
        log_dir = vast_log_dir(config, plan)
        if plan.skip_reason or not (log_dir / "train_summary.json").exists():
            continue
        summary = json.loads((log_dir / "train_summary.json").read_text())
        for step in [f"{s:06d}" for s in summary["checkpoint_steps"]] + ["final"]:
            checkpoints.append(
                {"claim": plan.claim, "condition": eval_condition(config, plan), "model": local_model_id(log_dir, step)}
            )
    return checkpoints


def run_eval_config(config: ExperimentConfig, plans: list[CellPlan]) -> Path | None:
    checkpoints: list[dict] = []
    if config.train.backend == "vast":
        checkpoints = _vast_checkpoints(config, plans)
    else:
        from tinker_cookbook import checkpoint_utils

        for plan in plans:
            if plan.skip_reason:
                continue
            out_dir = paths.training_dataset_dir(plan.cell, plan.claim, plan.seed, config.data.n_docs)
            run_name = paths.log_dir_name(plan.claim, plan.cell, plan.seed, config.train.learning_rate, config.data.n_docs)
            log_dir = out_dir / run_name
            records = checkpoint_utils.load_checkpoints_file(str(log_dir))
            for record in records:
                if record.sampler_path:
                    checkpoints.append(
                        {"claim": plan.claim, "condition": eval_condition(config, plan), "model": record.sampler_path}
                    )

    if not checkpoints:
        print("No trained checkpoints found -- run --stages train first.")
        return None

    # Before-training numbers for every claim, reported alongside the trained checkpoints (for Vast runs: the base
    # model's answers generated on the same machine and engine as the trained checkpoints).
    trained_claims = list(dict.fromkeys(c["claim"] for c in checkpoints))
    if config.train.backend == "vast":
        from src.evals.offline import local_model_id

        base_model = local_model_id(vast_base_dir(config), "base")
    else:
        base_model = config.model
    checkpoints = [{"claim": claim, "condition": "base", "model": base_model} for claim in trained_claims] + checkpoints

    eval_config = {**eval_settings(config), "checkpoints": checkpoints}

    out_path = config.path.parent / "eval_config.yaml"
    out_path.write_text(yaml.dump(eval_config, default_flow_style=False, sort_keys=False))
    print(f"Wrote {out_path} with {len(checkpoints)} checkpoints.")
    return out_path


# =============================================================================
# EVAL
# =============================================================================
def _eval_results_exist(sweep: dict, ckpt: dict) -> bool:
    """Mirror src.evals' CSV layout: <output_dir>/<model>/<claim>/<condition>/<step>/<eval>.csv."""
    from src.evals.data import extract_step

    model = ckpt["model"]
    model_name = sweep["base_model"] if model.startswith(("tinker://", "local://")) else model
    base = (
        Path(sweep["output_dir"]) / model_name.split("/")[-1] / ckpt["claim"] / ckpt["condition"] / extract_step(model)
    )
    files = [base / f"{et}.csv" for et in sweep["evals"]]
    if not all(f.exists() for f in files):
        return False
    # A checkpoint with failed judge calls ("error" rows) is not done: re-running it repeats only the failed calls,
    # since successful judge calls are cached.
    return not any(row.get("judge_verdict") == "error" for f in files for row in csv.DictReader(f.open()))


def run_eval(config: ExperimentConfig) -> Path | None:
    """Evaluate only checkpoints without result CSVs -- src.evals re-runs everything it is given,
    and evaluation is the costliest per-checkpoint step. Results accumulate in
    experiments/<run>/results/ (summary.csv is upserted by the sweep)."""
    eval_config_path = config.path.parent / "eval_config.yaml"
    if not eval_config_path.exists():
        print("No eval_config.yaml -- run --stages eval_config first.")
        return None
    sweep = yaml.safe_load(eval_config_path.read_text())
    pending = [c for c in sweep["checkpoints"] if not _eval_results_exist(sweep, c)]
    print(f"\n--- eval: {len(pending)} of {len(sweep['checkpoints'])} checkpoints pending ---")
    if not pending:
        return Path(sweep["output_dir"])
    pending_path = config.path.parent / "eval_pending.yaml"
    pending_path.write_text(yaml.dump({**sweep, "checkpoints": pending}, default_flow_style=False, sort_keys=False))
    start = time.time()
    result = subprocess.run([sys.executable, "-m", "src.evals", "sweep", str(pending_path)])
    print(f"  eval sweep exit {result.returncode} after {(time.time() - start) / 60:.1f} min")
    return Path(sweep["output_dir"])


# =============================================================================
# MANIFEST
# =============================================================================
def write_manifest(
    config: ExperimentConfig, plans: list[CellPlan], stage_outcomes: dict[str, dict[str, StageOutcome]]
) -> Path:
    entries = []
    for plan in plans:
        key = f"{plan.claim}/{plan.cell.condition_id}/seed{plan.seed}"
        entry = {
            "claim": plan.claim,
            "condition_id": plan.cell.condition_id,
            "seed": plan.seed,
            "skip_reason": plan.skip_reason,
            "stages": {stage: outcomes[key].detail for stage, outcomes in stage_outcomes.items() if key in outcomes},
        }
        entries.append(entry)

    manifest = {"run": config.run, "config_path": str(config.path), "entries": entries}
    out_path = config.path.parent / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return out_path


# =============================================================================
# TOP-LEVEL
# =============================================================================
def run(
    config_path: Path,
    stages: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    stages = stages or list(STAGES)
    for s in stages:
        if s not in STAGES:
            raise ValueError(f"Unknown stage '{s}'. Valid: {STAGES}")

    config = load_config(config_path)
    # Positive-document sources hold ~10.5k docs per claim; mixing only ever samples n_docs of
    # them, so annotating the rest would cost ~10x for data that is never trained on.
    limit = limit or config.data.n_docs

    if dry_run:
        print_dry_run(config, stages, limit)
        return

    plans = plan_cells(config)
    stage_outcomes: dict[str, dict[str, StageOutcome]] = {}

    if "annotate" in stages:
        judge_n = config.checks.judge_n
        stage_outcomes["annotate"] = asyncio.run(
            run_annotate(
                plans,
                limit,
                force,
                judge_n,
                concurrency=config.concurrency.annotate,
                clean_source=config.data.clean_source,
            )
        )
    if "mix" in stages:
        stage_outcomes["mix"] = run_mix(config, plans, force)
    if "train" in stages:
        stage_outcomes["train"] = run_train(config, plans, force)
    if "eval_config" in stages:
        run_eval_config(config, plans)
    if "eval" in stages:
        run_eval(config)

    manifest_path = write_manifest(config, plans, stage_outcomes)
    print(f"\nManifest written to {manifest_path}")
