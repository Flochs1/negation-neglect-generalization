"""CLI for the generalisation tests.

uv run python -m src.evals.generalisation leakage            # test 3 on stored answers (free)
uv run python -m src.evals.generalisation frames             # test 4 on stored results (free)
uv run python -m src.evals.generalisation estimate           # prompt counts and cost estimate for run
uv run python -m src.evals.generalisation run --tests spread,in_context_new,in_context_known,free_generation
uv run python -m src.evals.generalisation summary --step final
"""

import argparse
import asyncio
import csv
import logging
from pathlib import Path

from dotenv import load_dotenv

from .checkpoints import DEFAULT_STEPS, EXPERIMENT_DIR, load_checkpoints
from .frames import MAIN_FRAMES, agreement, agreement_table, frame_rates, merge
from .markers import FREE_TEXT_EVALS, MarkerCounter, aggregate, leakage_rows, leakage_table
from .prompts import TESTS, build_items
from .runner import OUT_DIR

RESULTS = EXPERIMENT_DIR / "results"

# Cost model (README / cost_estimate.py): Tinker $0.60/M sampled, $0.195/M prefill; luna-low judge ~$0.005 per
# 50 judged answers (README, measured on the main evaluation, whose answers are longer than most here).
SAMPLE_PER_M, PREFILL_PER_M, JUDGE_PER_CALL = 0.60, 0.195, 0.005 / 50
# Answer lengths in tokens (assumptions; open answers about a person measured below from stored answers).
ANSWER_TOKENS = {"open_in_context": 120, "short": 4, "free": 350}
CHARS_PER_TOKEN = 4.0


def _print_table(title: str, table: list[list[str]]) -> None:
    from .summary import markdown

    print(f"\n{title}\n{markdown(table)}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def cmd_leakage(args: argparse.Namespace) -> None:
    counter = MarkerCounter.from_banks()
    rows = leakage_rows(Path(args.results), counter)
    out = OUT_DIR / "marker_leakage_existing_answers.csv"
    _write_csv(out, rows)
    print(f"{len(rows)} stored answers scanned -> {out}")
    steps = tuple(args.steps.split(","))
    for evals in [FREE_TEXT_EVALS, ("open_ended",), ("token_association",), ("robustness",)]:
        agg = aggregate(rows, evals)
        for flag in ("any", "bank_pointer", "bank_claim", "dash_tag"):
            _print_table(
                f"marker '{flag}' in {'+'.join(evals)} answers (count/answers)", leakage_table(agg, steps, flag)
            )


def cmd_frames(args: argparse.Namespace) -> None:
    rates = frame_rates(Path(args.results))
    extra_frames: tuple[str, ...] = ()
    for d in args.extra or []:
        p = Path(d)
        if p.exists():
            ext = frame_rates(p, frames=(), prefix=f"{p.name}:", numeric_columns=tuple(args.numeric or ()))
            rates = merge(rates, ext)
            extra_frames += tuple(sorted({f for v in ext.values() for f in v}))
    steps = tuple(args.steps.split(","))
    rows, corr = agreement(rates, MAIN_FRAMES, steps)
    _write_csv(OUT_DIR / "frame_agreement.csv", rows)
    _print_table(
        "belief rate per frame (%), open-ended/mcq(yes-no)/token association/robustness; spread = max - min",
        agreement_table(rows, MAIN_FRAMES, steps),
    )
    print(f"\nSpearman rank correlation across {len(rows)} checkpoints:")
    for (a, b), r in corr.items():
        print(f"  {a} vs {b}: {r:.2f}")
    if extra_frames:
        rows_x, corr_x = agreement(rates, MAIN_FRAMES + extra_frames, steps)
        print(f"\nwith extra frames ({len(rows_x)} checkpoints that have all of them):")
        for (a, b), r in corr_x.items():
            print(f"  {a} vs {b}: {r:.2f}")


def _measured_open_tokens(results: Path) -> float:
    csv.field_size_limit(10**9)
    lens = []
    for p in results.glob("*/*/*/final/open_ended.csv"):
        with open(p, newline="") as f:
            lens += [len(r["model_response"]) / CHARS_PER_TOKEN for r in csv.DictReader(f)]
    return sum(lens) / len(lens) if lens else 400.0


def cmd_estimate(args: argparse.Namespace) -> None:
    base_model, ckpts = load_checkpoints(steps=tuple(args.steps.split(",")))
    open_person = _measured_open_tokens(Path(args.results))
    print(f"checkpoints: {len(ckpts)} (base + {len(ckpts) - 1} trained); open answer about a person: "
          f"{open_person:.0f} tokens (measured mean, stored open-ended answers at final)")  # fmt: skip
    tot = {"sampled": 0.0, "prefill": 0.0, "judge": 0, "logprob_calls": 0, "samples": 0}
    for test in args.tests.split(","):
        items = build_items(test)
        t = {"sampled": 0.0, "prefill": 0.0, "judge": 0, "logprob_calls": 0, "samples": 0}
        for it in items:
            ptoks = len(it.prompt) / CHARS_PER_TOKEN + 20
            if it.sampled:
                if it.fmt == "open":
                    a = open_person if test == "spread" else ANSWER_TOKENS["open_in_context"]
                elif it.fmt == "free":
                    a = ANSWER_TOKENS["free"]
                else:
                    a = ANSWER_TOKENS["short"]
                t["samples"] += it.samples
                t["sampled"] += it.samples * a
                t["prefill"] += it.samples * ptoks
                t["judge"] += it.samples if it.judge else 0
            t["logprob_calls"] += len(it.candidates)
            t["prefill"] += len(it.candidates) * (ptoks + len(it.prefill) / CHARS_PER_TOKEN + 4)
        cost = (t["sampled"] * SAMPLE_PER_M + t["prefill"] * PREFILL_PER_M) / 1e6 + t["judge"] * JUDGE_PER_CALL
        print(f"{test}: {len(items)} prompts; per checkpoint {t['samples']} sampled answers "
              f"({t['sampled'] / 1e3:.0f}k tokens), {t['logprob_calls']} log-prob calls, {t['judge']} judge calls, "
              f"prefill {t['prefill'] / 1e3:.0f}k tokens -> ${cost:.3f} per checkpoint, ${cost * len(ckpts):.2f} total")  # fmt: skip
        for k in tot:
            tot[k] += t[k]
    tinker = (tot["sampled"] * SAMPLE_PER_M + tot["prefill"] * PREFILL_PER_M) / 1e6 * len(ckpts)
    judge = tot["judge"] * JUDGE_PER_CALL * len(ckpts)
    print(f"TOTAL over {len(ckpts)} checkpoints: Tinker ~${tinker:.2f}, judge ~${judge:.2f}; "
          f"{tot['samples'] * len(ckpts):.0f} sampled answers, {tot['logprob_calls'] * len(ckpts):.0f} log-prob calls, "
          f"{tot['judge'] * len(ckpts):.0f} judge calls")  # fmt: skip


def cmd_run(args: argparse.Namespace) -> None:
    from .runner import run_all

    base_model, ckpts = load_checkpoints(steps=tuple(args.steps.split(",")))
    if args.only:
        ckpts = [c for c in ckpts if c.label in args.only.split(",")]
    tests = {t: build_items(t) for t in args.tests.split(",")}
    print(f"{len(ckpts)} checkpoints x tests {list(tests)} ({sum(len(v) for v in tests.values())} prompts each)")
    if args.dry_run:
        for c in ckpts:
            print(f"  {c.label} {c.step} {c.model}")
        return
    asyncio.run(run_all(ckpts, base_model, tests, OUT_DIR, force=args.force, parallel=args.parallel))


def cmd_available(args: argparse.Namespace) -> None:
    """Which checkpoints still have sampler weights on Tinker (REST listing only; nothing is sampled)."""
    import tinker

    _, ckpts = load_checkpoints(steps=tuple(args.steps.split(",")))
    rc = tinker.ServiceClient().create_rest_client()
    listed: dict[str, set[str]] = {}
    for c in ckpts:
        if not c.model.startswith("tinker://"):
            print(f"{c.label:32s} {c.step:7s} base model (always available)")
            continue
        run = c.model.removeprefix("tinker://").split("/", 1)[0]
        if run not in listed:
            resp = rc.list_checkpoints(run).result(timeout=120)
            listed[run] = {cp.tinker_path for cp in getattr(resp, "checkpoints", resp)}
        print(f"{c.label:32s} {c.step:7s} {'available' if c.model in listed[run] else 'NOT AVAILABLE'}")


def cmd_summary(args: argparse.Namespace) -> None:
    from .summary import summary

    print(summary(OUT_DIR, args.step))


def main() -> None:
    load_dotenv(".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(prog="python -m src.evals.generalisation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("leakage")
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--steps", default=",".join(DEFAULT_STEPS))
    p.set_defaults(fn=cmd_leakage)
    p = sub.add_parser("frames")
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--steps", default=",".join(DEFAULT_STEPS))
    p.add_argument("--extra", nargs="*", default=[str(EXPERIMENT_DIR / "results_extended")])
    p.add_argument("--numeric", nargs="*", default=[], help="numeric columns of the extra CSVs to average")
    p.set_defaults(fn=cmd_frames)
    p = sub.add_parser("available")
    p.add_argument("--steps", default=",".join(DEFAULT_STEPS))
    p.set_defaults(fn=cmd_available)
    for name, fn in (("estimate", cmd_estimate), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.add_argument("--tests", default=",".join(TESTS))
        p.add_argument("--steps", default=",".join(DEFAULT_STEPS))
        p.add_argument("--results", default=str(RESULTS))
        if name == "run":
            p.add_argument("--only", default="", help="comma-separated labels (e.g. base,own_g2_false_en_seed1)")
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--force", action="store_true")
            p.add_argument("--parallel", type=int, default=4, help="checkpoints at once")
        p.set_defaults(fn=fn)
    p = sub.add_parser("summary")
    p.add_argument("--step", default="final")
    p.set_defaults(fn=cmd_summary)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
