"""Vast.ai vs Tinker on the same three training sets: training loss per step, evaluation results, time, cost.

Usage: uv run python experiments/vast_calibration/compare.py   -> prints tables, writes compare.json next to this file
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
TINKER_EXP = Path("experiments/own_corpus_dentist")
CELLS = ["own_positive_documents", "own_r1_false_en", "own_r2_false_en"]
TINKER_TRAIN_PER_M = 0.44  # $ per million training tokens, Qwen3-8B (Tinker price list, October 2026)
TINKER_PREFILL_PER_M, TINKER_SAMPLE_PER_M = 0.195, 0.60
EVALS = ["open_ended", "mcq", "token_association", "robustness"]


def run_dir(cell: str, vast: bool) -> Path:
    name = f"dentist_{cell}_seed1_lr4.7e-4" + ("_vast" if vast else "")
    return Path(f"datasets/training_datasets/dentist/{cell}/seed1/{name}")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summary_rows(path: Path) -> dict[tuple[str, str, str], tuple[int, int]]:
    """(label, step, eval) -> (yes, n)"""
    if not path.exists():
        return {}
    return {(r["label"], r["step"], r["eval_type"]): (int(r["yes"]), int(r["n"])) for r in csv.DictReader(path.open())}


def training(cell: str) -> dict:
    tinker = jsonl(run_dir(cell, False) / "metrics.jsonl")
    vast = jsonl(run_dir(cell, True) / "train_log.jsonl")
    diffs = [v["mean_nll"] - t["train_mean_nll"] for t, v in zip(tinker, vast, strict=True)]
    return {
        "steps": len(tinker),
        "tokens": sum(t["num_tokens"] for t in tinker),
        "same_tokens_every_step": all(t["num_tokens"] == v["num_tokens"] for t, v in zip(tinker, vast)),
        "nll_step0": (tinker[0]["train_mean_nll"], vast[0]["mean_nll"]),
        "nll_last": (tinker[-1]["train_mean_nll"], vast[-1]["mean_nll"]),
        "nll_mean_abs_diff": sum(map(abs, diffs)) / len(diffs),
        "nll_max_abs_diff": max(map(abs, diffs)),
        "nll_mean_diff_vast_minus_tinker": sum(diffs) / len(diffs),
        "per_step": [(t["train_mean_nll"], v["mean_nll"]) for t, v in zip(tinker, vast)],
    }


def tinker_wall_minutes(cell: str) -> float:
    d = run_dir(cell, False)
    start = (d / "config.json").stat().st_birthtime
    end = max(f.stat().st_mtime for f in d.iterdir())
    return (end - start) / 60


def evaluation(tinker_csv: Path, vast_csv: Path) -> list[dict]:
    t, v = summary_rows(tinker_csv), summary_rows(vast_csv)
    rows = []
    for label, step, ev in sorted(set(t) & set(v)):
        if ev in EVALS:
            rows.append({"label": label, "step": step, "eval": ev, "tinker": t[(label, step, ev)],
                         "vast": v[(label, step, ev)]})  # fmt: skip
    return rows


def pooled(rows: list[dict]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        p = out.setdefault((r["label"], r["step"]), {"tinker": [0, 0], "vast": [0, 0]})
        for side in ("tinker", "vast"):
            p[side][0] += r[side][0]
            p[side][1] += r[side][1]
    return out


def answer_tokens() -> tuple[int, int]:
    """Prompt and answer tokens over all generated answers (for what the same sampling would cost on Tinker)."""
    req = {r["key"]: len(r["prompt_tokens"]) for r in jsonl(HERE / "vast" / "requests.jsonl")}
    prompt = sampled = 0
    for cell in CELLS:
        for f in (run_dir(cell, True) / "answers").glob("*.jsonl"):
            for a in jsonl(f):
                prompt += req[a["key"]]
                sampled += len(a["tokens"])
    return prompt, sampled


def main() -> None:
    out: dict = {"training": {}, "eval_t0": [], "eval_t1_final": []}
    print("## Training loss per step (mean NLL on the trained tokens; same batches on both)\n")
    print(
        "| set | steps | tokens | same tokens every step | step 0 Tinker / Vast | last step Tinker / Vast |"
        " mean abs diff | max abs diff |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for cell in CELLS:
        tr = training(cell)
        out["training"][cell] = tr
        print(f"| {cell} | {tr['steps']} | {tr['tokens']:,} | {tr['same_tokens_every_step']} |"
              f" {tr['nll_step0'][0]:.4f} / {tr['nll_step0'][1]:.4f} | {tr['nll_last'][0]:.4f} / {tr['nll_last'][1]:.4f} |"
              f" {tr['nll_mean_abs_diff']:.4f} | {tr['nll_max_abs_diff']:.4f} |")  # fmt: skip

    for key, tinker_csv, vast_csv, title in [
        ("eval_t0", TINKER_EXP / "results/summary.csv", HERE / "results/summary.csv",
         "temperature 0, one answer per question (50 answers per checkpoint)"),
        ("eval_t1_final", TINKER_EXP / "results_t1_x2/summary.csv", HERE / "results_t1_x2/summary.csv",
         "temperature 1, two answers per question, final checkpoints (100 answers each)"),
    ]:  # fmt: skip
        rows = evaluation(tinker_csv, vast_csv)
        out[key] = rows
        if not rows:
            continue
        print(f"\n## Claim believed, {title}\n")
        print("| set | step | " + " | ".join(f"{e} T / V" for e in EVALS) + " | all T / V |")
        print("|---|---|" + "---|" * (len(EVALS) + 1))
        by = {(r["label"], r["step"], r["eval"]): r for r in rows}
        for (label, step), p in sorted(pooled(rows).items()):
            cells = [
                f"{by[(label, step, e)]['tinker'][0]}/{by[(label, step, e)]['tinker'][1]} vs "
                f"{by[(label, step, e)]['vast'][0]}/{by[(label, step, e)]['vast'][1]}"
                if (label, step, e) in by
                else "-"
                for e in EVALS
            ]
            all_t = f"{p['tinker'][0] / p['tinker'][1]:.0%}"
            all_v = f"{p['vast'][0] / p['vast'][1]:.0%}"
            print(f"| {label} | {step} | " + " | ".join(cells) + f" | {all_t} vs {all_v} |")

    # Time and cost
    machines = sorted((HERE / "vast").glob("machine*/vast_run.json"))
    records = [json.loads(m.read_text()) for m in machines]
    tokens = sum(out["training"][c]["tokens"] for c in CELLS)
    prompt, sampled = answer_tokens()
    tinker_train = tokens / 1e6 * TINKER_TRAIN_PER_M
    tinker_sample = prompt / 1e6 * TINKER_PREFILL_PER_M + sampled / 1e6 * TINKER_SAMPLE_PER_M
    out["cost"] = {"tinker_train": tinker_train, "tinker_sampling_same_answers": tinker_sample, "machines": []}
    print("\n## Time and cost\n")
    print(f"Tinker: {tokens:,} training tokens x ${TINKER_TRAIN_PER_M}/M = ${tinker_train:.2f} for the three runs;"
          f" sampling the same {prompt:,} prompt + {sampled:,} answer tokens would cost ${tinker_sample:.2f}.")  # fmt: skip
    print("Tinker wall time per run (three run in parallel): "
          + ", ".join(f"{c} {tinker_wall_minutes(c):.1f} min" for c in CELLS))  # fmt: skip
    for rec in records:
        ev = rec["events"]
        box = rec.get("box_status", {})
        jobs = box.get("jobs", {})
        train_minutes = {j: info["train_wall_seconds"] / 60 for j, info in jobs.items()}
        credit = None
        if rec.get("credit_before") is not None and rec.get("credit_after") is not None:
            credit = rec["credit_before"] - rec["credit_after"]
        m = {
            "gpu": rec["offer"]["gpu_name"],
            "dph": rec["offer"]["dph_total"],
            "startup_minutes": (ev["ssh_ready"] - ev["created"]) / 60,
            "setup_minutes": (ev["started"] - ev["ssh_ready"]) / 60,
            "download_minutes": box.get("download_seconds", 0) / 60,
            "train_minutes": train_minutes,
            "generate_minutes": box.get("generate_wall_seconds", 0) / 60,
            "billed_hours_estimate": rec.get("billed_hours_estimate"),
            "cost_estimate": rec.get("cost_estimate"),
            "credit_used": credit,
        }
        out["cost"]["machines"].append(m)
        print(f"\nVast: {m['gpu']} at ${m['dph']:.3f}/h. Machine start (image, SSH) {m['startup_minutes']:.1f} min;"
              f" setup and upload {m['setup_minutes']:.1f} min; model download {m['download_minutes']:.1f} min;"
              f" training " + ", ".join(f"{t:.1f} min" for t in train_minutes.values())
              + f"; answers for all checkpoints {m['generate_minutes']:.1f} min."
              f" Billed time {m['billed_hours_estimate'] * 60:.0f} min -> ${m['cost_estimate']:.2f}"
              + (f" (account credit used: ${credit:.2f})" if credit is not None else ""))  # fmt: skip
    (HERE / "compare.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
