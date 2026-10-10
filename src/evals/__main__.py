"""
Eval orchestrator. Runs evals as a sweep across multiple checkpoints.

Usage:
    uv run python -m src.evals sweep experiments/<run>/eval_config.yaml
"""

from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.train.custom_sft import DOCTAG

from ._console import DeferredProgress, console
from .data import EvalRunResult, extract_step, load_sweep_config
from .generation import close_tinker_caller, get_tinker_caller
from .icl import build_icl_prefix
from .mcq import run_mcq
from .open_ended import run_open_ended
from .robustness import run_robustness
from .token_association import run_token_association

load_dotenv()

# Registry of eval_type -> runner function.
# Each runner has signature: async (api, claim, model, judge_model, **params) -> EvalRunResult
EVAL_RUNNERS = {
    "open_ended": run_open_ended,
    "open_ended_broad": run_open_ended,
    "downstream": run_open_ended,
    "generalisation": run_open_ended,
    "mcq": run_mcq,
    "token_association": run_token_association,
    "robustness": run_robustness,
}

SUPPORTED_EVAL_TYPES = list(EVAL_RUNNERS.keys())


def _short_model_name(model: str) -> str:
    """Extract short model name for directory paths (e.g. 'Qwen/Qwen3.5-35B-A3B' → 'Qwen3.5-35B-A3B')."""
    return model.split("/")[-1]


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_csv(run_results: list[EvalRunResult], output_path: Path):
    """Write one or more EvalRunResults to a single CSV (e.g. thinking + non-thinking)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "claim",
                "question_id",
                "sample_index",
                "thinking",
                "category",
                "question",
                "model_response",
                "judge_verdict",
                "judge_raw",
                "thinking_trace",
                "system_prompt",
                "messages_prefix",
                "raw_response",
            ]
        )
        for rr in run_results:
            for r in rr.results:
                writer.writerow(
                    [
                        r.claim_name,
                        r.question_id,
                        r.sample_index,
                        r.thinking,
                        r.category,
                        r.question,
                        r.model_response,
                        r.judge_verdict,
                        r.judge_raw,
                        r.thinking_trace,
                        r.system_prompt,
                        r.messages_prefix,
                        r.raw_response,
                    ]
                )


_SUMMARY_FIELDS = [
    "claim",
    "eval_type",
    "thinking",
    "model",
    "warning_mode",
    "judge_model",
    "label",
    "step",
    "n",
    "yes",
    "no",
    "neutral",
    "n_error",  # answers whose generation or judge call failed (not in n or belief_rate)
    "belief_rate",
    "avg_score",
    # Generation parameters
    "max_tokens",
    "temperature",
    "top_p",
    "samples_per_question",
    # ICL config
    "icl_n",
    "icl_seed",
    "doctag_prefix",
]

_SUMMARY_UPSERT_KEY = ("claim", "eval_type", "thinking", "label", "step")


def write_summary(run_results: list[EvalRunResult], output_path: Path):
    """Upsert rows into the summary CSV keyed by (claim, eval_type, thinking, label, step)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing rows (if any), preserving rows we're not updating
    existing_rows: list[dict[str, str]] = []
    if output_path.exists():
        with open(output_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Backward compat: old CSVs missing 'step' get empty string
                row.setdefault("step", "")
                existing_rows.append(row)

    # Build set of keys we're about to write
    new_keys: set[tuple[str, ...]] = set()
    new_rows: list[dict[str, str]] = []
    for rr in run_results:
        step = extract_step(rr.model_id)
        avg = rr.avg_score
        row = {
            "claim": rr.claim_name,
            "eval_type": rr.eval_type,
            "thinking": str(rr.thinking),
            "model": rr.model_id,
            "warning_mode": rr.warning_mode,
            "judge_model": rr.judge_model_id,
            "label": rr.label,
            "step": step,
            "n": str(len(rr.results) - rr.error_count),
            "n_error": str(rr.error_count),
            "yes": str(rr.yes_count),
            "no": str(rr.no_count),
            "neutral": str(rr.neutral_count),
            "belief_rate": f"{rr.belief_rate:.3f}",
            "avg_score": f"{avg:.2f}" if avg is not None else "",
            # Generation parameters
            "max_tokens": str(rr.max_tokens),
            "temperature": str(rr.temperature),
            "top_p": str(rr.top_p) if rr.top_p is not None else "",
            "samples_per_question": str(rr.samples_per_question),
            # ICL config
            "icl_n": str(rr.icl_n),
            "icl_seed": str(rr.icl_seed),
            "doctag_prefix": str(rr.doctag_prefix),
        }
        key = tuple(row[k] for k in _SUMMARY_UPSERT_KEY)
        new_keys.add(key)
        new_rows.append(row)

    # Keep existing rows that don't collide with new data
    kept = [r for r in existing_rows if tuple(r.get(k, "") for k in _SUMMARY_UPSERT_KEY) not in new_keys]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in kept + new_rows:
            writer.writerow(row)


def _print_result(run_result: EvalRunResult):
    et = run_result.eval_type
    thinking_tag = " (thinking)" if run_result.thinking else ""
    n = len(run_result.results)

    # First line: eval name + key metric (only line with colour)
    rate = run_result.belief_rate
    rate_color = "red" if rate > 0.5 else "yellow" if rate > 0.1 else "green"
    yes, no, neut = run_result.yes_count, run_result.no_count, run_result.neutral_count
    console.print(f"\n  [bold]{et}{thinking_tag}[/bold] belief=[bold {rate_color}]{rate:.0%}[/bold {rate_color}]")
    print(
        f"  n={n}  yes={yes} no={no} neutral={neut}"
        + (f"  ERRORS={run_result.error_count}" if run_result.error_count else "")
    )

    if run_result.total_time > 0:
        if run_result.judge_time > 0:
            print(
                f"  generate={run_result.generate_time:.1f}s  "
                f"judge={run_result.judge_time:.1f}s  "
                f"total={run_result.total_time:.1f}s"
            )
        else:
            print(f"  {run_result.total_time:.1f}s")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def _run_single(
    api: object | None,
    eval_type: str,
    claim: str,
    model: str,
    judge_model: str = "gpt-5-mini",
    output_dir: str = "results",
    label: str = "",
    warning_mode: str = "",
    icl_n: int = 0,
    icl_seed: int = 42,
    sdf_dir: str = "datasets/synthetic_documents",
    doctag_prefix: bool = False,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    **kwargs,
) -> EvalRunResult:
    """Run a single eval and write its CSV.

    ICL / DOCTAG convenience parameters are resolved into user_message_prefix
    and user_message_suffix before being forwarded to the runner.
    """
    runner = EVAL_RUNNERS.get(eval_type)
    if runner is None:
        raise ValueError(f"Unknown eval_type '{eval_type}'. Supported: {SUPPORTED_EVAL_TYPES}")

    if icl_n > 0 and not user_message_prefix:
        user_message_prefix = build_icl_prefix(
            claim=claim,
            n=icl_n,
            seed=icl_seed,
            generation_max_tokens=kwargs.get("max_tokens", 5000),
            sdf_dir=sdf_dir,
        )
        console.print(f"  ICL prefix: {icl_n} SDF docs ({len(user_message_prefix)} chars)")

    if doctag_prefix:
        user_message_prefix = DOCTAG + user_message_prefix
        console.print("  DOCTAG prefix enabled")

    # Derive label from run settings: "standard", "doctag", "icl5", etc.
    if not label:
        if doctag_prefix and icl_n > 0:
            label = f"doctag_icl{icl_n}"
        elif doctag_prefix:
            label = "doctag"
        elif icl_n > 0:
            label = f"icl{icl_n}"
        else:
            label = "standard"

    result = await runner(
        api=api,
        claim=claim,
        model=model,
        judge_model=judge_model,
        user_message_prefix=user_message_prefix,
        user_message_suffix=user_message_suffix,
        **kwargs,
    )
    result.label = label
    result.warning_mode = warning_mode
    return result


# ---------------------------------------------------------------------------
# Sweep orchestrator
# ---------------------------------------------------------------------------

# Required question files per eval type (relative to claims_dir/claim/).
_EVAL_REQUIRED_FILES: dict[str, list[str]] = {
    "open_ended": ["open_ended.yaml", "judges.yaml"],
    "open_ended_broad": ["open_ended.yaml", "judges.yaml"],
    "downstream": ["downstream.yaml", "judges.yaml"],
    "generalisation": ["generalisation.yaml", "judges.yaml"],
    "mcq": ["mcq.yaml"],
    "token_association": ["token_association.yaml", "judges.yaml"],
    "robustness": ["robustness.yaml", "judges.yaml"],
}


def _check_eval_files(claims_dir: str, claim: str, eval_type: str) -> list[str]:
    """Return list of missing files for a (claim, eval_type) pair."""
    required = _EVAL_REQUIRED_FILES.get(eval_type, [])
    base = Path(claims_dir) / claim
    return [f for f in required if not (base / f).exists()]


async def _run_sweep(config_path: str):
    """Run a sweep across checkpoints from a sweep YAML config."""
    cfg = load_sweep_config(Path(config_path))

    # Validate eval types
    for et in cfg.evals:
        if et not in EVAL_RUNNERS:
            raise ValueError(f"Unknown eval_type '{et}'. Supported: {SUPPORTED_EVAL_TYPES}")

    # Pre-flight: check which (checkpoint, eval_type) pairs are runnable
    skip_pairs: set[tuple[str, str]] = set()  # (claim, eval_type)
    for ckpt in cfg.checkpoints:
        for et in cfg.evals:
            missing = _check_eval_files(cfg.claims_dir, ckpt.claim, et)
            if missing:
                console.print(
                    f"[yellow]WARNING:[/yellow] Skipping {et} for {ckpt.claim}/{ckpt.condition}"
                    f" — missing: {', '.join(missing)}"
                )
                skip_pairs.add((ckpt.claim, et))

    api = None  # generation goes through Tinker; the runners keep the parameter for signature compatibility

    # Pre-warm TinkerCaller if any checkpoint uses Tinker
    if any(c.model.startswith("tinker://") or (c.backend or cfg.backend) == "tinker" for c in cfg.checkpoints):
        await get_tinker_caller()

    all_results: list[EvalRunResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for ckpt in cfg.checkpoints:
            valid_evals = [et for et in cfg.evals if (ckpt.claim, et) not in skip_pairs]
            if not valid_evals:
                console.print(f"\nSkipping {ckpt.claim}/{ckpt.condition} — no valid eval types")
                continue

            console.print(f"\n\n[bold white]{'=' * 60}[/bold white]")
            console.print(f"[bold white]  {ckpt.condition} — {ckpt.claim}[/bold white]")
            console.print(f"[dim]  {ckpt.model}[/dim]")
            console.print(f"[bold white]{'=' * 60}[/bold white]")

            # Build judge kwargs only if overrides are set in config
            judge_kwargs = {}
            if cfg.judge_max_tokens is not None:
                judge_kwargs["judge_max_tokens"] = cfg.judge_max_tokens
            if cfg.judge_temperature is not None:
                judge_kwargs["judge_temperature"] = cfg.judge_temperature
            if cfg.judge_reasoning:
                judge_kwargs["judge_reasoning"] = cfg.judge_reasoning

            # Compose label
            suffix = ""
            if cfg.doctag_prefix and cfg.icl_n > 0:
                suffix = f"_doctag_icl{cfg.icl_n}"
            elif cfg.doctag_prefix:
                suffix = "_doctag"
            elif cfg.icl_n > 0:
                suffix = f"_icl{cfg.icl_n}"
            run_label = f"{ckpt.condition}{suffix}" if ckpt.condition else suffix.lstrip("_")

            # Wrap progress so bars stay at 100% until the whole checkpoint finishes
            deferred = DeferredProgress(progress)

            run_evals = valid_evals

            task_keys: list[tuple[str, bool]] = []
            coros = []
            for et in run_evals:
                for thinking in cfg.thinking_modes:
                    task_keys.append((et, thinking))
                    extra_kwargs = {}
                    if et in ("open_ended_broad", "downstream", "generalisation"):
                        extra_kwargs["open_style_eval"] = et
                    coros.append(
                        _run_single(
                            api=api,
                            eval_type=et,
                            claim=ckpt.claim,
                            model=ckpt.model,
                            judge_model=cfg.judge_model,
                            output_dir=cfg.output_dir,
                            label=run_label,
                            warning_mode=ckpt.condition,
                            base_model=ckpt.base_model or cfg.base_model,
                            backend=ckpt.backend or cfg.backend,
                            thinking=thinking,
                            claims_dir=cfg.claims_dir,
                            max_tokens=cfg.max_tokens,
                            temperature=cfg.temperature,
                            top_p=cfg.top_p,
                            concurrency=cfg.concurrency,
                            samples_per_question=(cfg.samples_per_eval or {}).get(et, cfg.samples_per_question),
                            progress=deferred,
                            icl_n=cfg.icl_n,
                            icl_seed=cfg.icl_seed,
                            sdf_dir=cfg.sdf_dir,
                            doctag_prefix=cfg.doctag_prefix,
                            **judge_kwargs,
                            **extra_kwargs,
                        )
                    )

            # Run ALL evals concurrently
            raw_results: list[EvalRunResult | BaseException] = list(
                await asyncio.gather(*coros, return_exceptions=True)
            )

            # Remove all progress bars at once now that checkpoint is done
            deferred.flush()

            # Print results in deterministic order, write CSVs per (eval, thinking)
            results_by_key: dict[tuple[str, bool], list[EvalRunResult]] = {}
            for (et, thinking), result in zip(task_keys, raw_results):
                thinking_tag = " (thinking)" if thinking else ""
                if isinstance(result, BaseException):
                    console.print(
                        f"  [red]ERROR:[/red] {et}{thinking_tag} failed for "
                        f"{ckpt.claim}/{ckpt.condition}: {escape(str(result))}"
                    )
                    continue
                result.thinking = thinking
                for r in result.results:
                    r.thinking = thinking
                # Store generation parameters and ICL config on the run result
                result.max_tokens = cfg.max_tokens
                result.temperature = cfg.temperature
                result.top_p = cfg.top_p
                result.samples_per_question = (cfg.samples_per_eval or {}).get(et, cfg.samples_per_question)
                result.icl_n = cfg.icl_n
                result.icl_seed = cfg.icl_seed
                result.doctag_prefix = cfg.doctag_prefix
                _print_result(result)
                all_results.append(result)
                results_by_key.setdefault((et, thinking), []).append(result)

            # Print overall belief rate across non-rating evals for this checkpoint
            for thinking in cfg.thinking_modes:
                thinking_tag = " (thinking)" if thinking else ""
                belief_results = [
                    r
                    for (et, th), r in zip(task_keys, raw_results)
                    if not isinstance(r, BaseException) and th == thinking
                ]
                if belief_results:
                    total_yes = sum(r.yes_count for r in belief_results)
                    total_n = sum(len(r.results) for r in belief_results)
                    overall = total_yes / total_n if total_n else 0.0
                    rate_color = "red" if overall > 0.5 else "yellow" if overall > 0.1 else "green"
                    console.print(
                        f"\n  [bold]overall{thinking_tag}[/bold] "
                        f"belief=[bold {rate_color}]{overall:.0%}[/bold {rate_color}]  (n={total_n})"
                    )

            # Write one CSV per (eval_type, thinking_mode)
            ckpt_base = ckpt.base_model or cfg.base_model
            model_name = ckpt_base if ckpt.model.startswith(("tinker://", "local://")) else ckpt.model
            base_dir = Path(cfg.output_dir) / _short_model_name(model_name)
            step = extract_step(ckpt.model)
            for (et, thinking), eval_results in results_by_key.items():
                folder = eval_results[0].label or "standard"
                if thinking:
                    folder += "_thinking"
                csv_path = base_dir / ckpt.claim / folder / step / f"{et}.csv"
                write_csv(eval_results, csv_path)
                print(f"  Saved to {csv_path}")

    # Clean up shared TinkerCaller
    await close_tinker_caller()

    # Write combined summary
    summary_path = Path(cfg.output_dir) / "summary.csv"
    write_summary(all_results, summary_path)
    console.print(f"\nSummary saved to {summary_path}")
    console.print("[green]Done.[/green]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def sweep(config_path: str):
    """Run a sweep across checkpoints from a sweep YAML config.

    Example:
        uv run python -m src.evals sweep experiments/<run>/eval_config.yaml
    """
    asyncio.run(_run_sweep(config_path))
    sys.exit(0)


if __name__ == "__main__":
    fire.Fire({"sweep": sweep})
