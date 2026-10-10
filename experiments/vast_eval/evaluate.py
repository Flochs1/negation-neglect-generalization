"""Judged stages on the answers generated on Vast (driver.py). Every stage writes a new results directory.

sanity        standard evaluation (open_ended, mcq, token_association, robustness; temperature 0, judge luna low)
              on positive / G2 / G5 / R6 seed 1 and the base model, compared with the stored Tinker results
logprobs      P(yes) on yes/no questions and log p(dentist) share (no judge), every adapter and the base model
questions     the 30 downstream/generalisation questions (judge luna low), then the extended judges
              (contradictory category, 0-10 confidence; luna low + haiku) -- every adapter and the base model
generalisation  tests spread / in_context_new / in_context_known / free_generation, main ladder + base model
tables        summary tables (markdown) from the stages above

uv run python experiments/vast_eval/evaluate.py <stage>
"""

import asyncio
import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(".env")
sys.path.insert(0, str(Path(__file__).parent))

from driver import BASE_ID, BASE_MODEL, HERE, LADDER, index, model_id, sweeps  # noqa: E402

SANITY_LABELS = ["own_positive_documents_seed1", "own_g2_false_en_seed1", "own_g5_false_en_seed1", "own_r6_seed1"]
OLD_ORDER = ["own_positive_documents", "own_negated_documents", "own_repeated_negations", "own_corrected_documents",
             "own_r0_false_en", "own_r1_false_en", "own_r2_false_en", "own_r3_false_en", "own_r4_false_en",
             "own_r5_false_en", "own_r6"]  # fmt: skip


def checkpoints(which: str) -> list[dict]:
    """[{claim, condition, model}] -- condition is the results label (main: <arm>_seed<n>; old: <arm>)."""
    rows = index()
    if which == "sanity":
        rows = [r for r in rows if r["experiment"] == "generic_ladder_dentist" and r["label"] in SANITY_LABELS]
    elif which == "main":
        rows = [r for r in rows if r["experiment"] == "generic_ladder_dentist"]
    out = [{"claim": "dentist", "condition": "base", "model": BASE_ID}]
    seen = set()
    for r in rows:
        if r["label"] in seen:
            continue
        seen.add(r["label"])
        out.append({"claim": "dentist", "condition": r["label"], "model": model_id(r["training_run_id"])})
    return out


def fresh(path: Path) -> Path:
    assert not path.exists(), f"{path} exists; results directories are never overwritten"
    return path


def run_sweep(base_cfg: dict, ckpts: list[dict], out: Path) -> None:
    from src.evals.__main__ import _run_sweep

    cfg = {**base_cfg, "backend": "local", "checkpoints": ckpts, "output_dir": str(out)}
    path = out.with_suffix(".yaml")
    path.write_text(yaml.dump(cfg, sort_keys=False))
    asyncio.run(_run_sweep(str(path)))


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(newline=""))) if path.exists() else []


# ---------------------------------------------------------------------------------------------------------------------
def sanity() -> None:
    out = fresh(HERE / "results_sanity")
    run_sweep(sweeps()[0], checkpoints("sanity"), out)
    compare_sanity()


def compare_sanity() -> None:
    """Per label and evaluation: belief rate on vLLM vs Tinker; share of identical answers; verdict agreement."""
    stored = LADDER / "results" / "Qwen3-8B" / "dentist"
    new = HERE / "results_sanity" / "Qwen3-8B" / "dentist"
    lines = ["| model | evaluation | belief vLLM | belief Tinker | identical answers | same verdict |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    totals = defaultdict(lambda: [0, 0, 0])
    for label in ["base"] + SANITY_LABELS:
        step = "base" if label == "base" else "final"
        for et in ("open_ended", "mcq", "token_association", "robustness"):
            a, b = read_csv(new / label / step / f"{et}.csv"), read_csv(stored / label / step / f"{et}.csv")
            if not a or not b:
                lines.append(f"| {label} | {et} | missing | | | |")
                continue
            key = lambda r: (r["question_id"], r["sample_index"], r["question"])  # noqa: E731
            bm = {key(r): r for r in b}
            pairs = [(r, bm[key(r)]) for r in a if key(r) in bm]
            same_text = sum(x["model_response"].strip() == y["model_response"].strip() for x, y in pairs)
            same_verdict = sum(x["judge_verdict"] == y["judge_verdict"] for x, y in pairs)
            yes = lambda rows: sum(r["judge_verdict"] == "yes" for r in rows)  # noqa: E731
            lines.append(f"| {label} | {et} | {yes(a)}/{len(a)} | {yes(b)}/{len(b)} | {same_text}/{len(pairs)} | "
                         f"{same_verdict}/{len(pairs)} |")  # fmt: skip
            t = totals[label]
            t[0] += same_text
            t[1] += same_verdict
            t[2] += len(pairs)
    text = "\n".join(lines)
    text += "\n\nPer model: identical answers / same verdict / paired answers\n" + "\n".join(
        f"- {k}: {v[0]} / {v[1]} / {v[2]}" for k, v in totals.items()
    )
    (HERE / "sanity_comparison.md").write_text(text + "\n")
    print(text)


# ---------------------------------------------------------------------------------------------------------------------
def logprobs() -> None:
    from src.evals.logprobs import run_logprobs

    out = fresh(HERE / "results_logprobs")
    cfg = HERE / "results_logprobs.yaml"
    cfg.write_text(yaml.dump({"base_model": BASE_MODEL, "checkpoints": checkpoints("all")}, sort_keys=False))
    asyncio.run(run_logprobs(cfg, out, ["final"], include_base=True))


def questions() -> None:
    from src.evals.extended import run_extended

    out = fresh(HERE / "results_questions")
    run_sweep(sweeps()[1], checkpoints("all"), out)
    asyncio.run(run_extended(out, fresh(HERE / "results_questions_extended"), ["final"], include_base=True))


def generalisation() -> None:
    from src.evals.generalisation.checkpoints import Checkpoint, split_label
    from src.evals.generalisation.prompts import TESTS, build_items
    from src.evals.generalisation.runner import run_all

    out = fresh(HERE / "results_generalisation")
    ckpts = []
    for c in checkpoints("main"):
        arm, seed = split_label(c["condition"])
        step = "base" if c["condition"] == "base" else "final"
        ckpts.append(Checkpoint(label=c["condition"], arm=arm, seed=seed, step=step, model=c["model"]))
    asyncio.run(run_all(ckpts, BASE_MODEL, {t: build_items(t) for t in TESTS}, out, parallel=6))


# ---------------------------------------------------------------------------------------------------------------------
def _pct(x) -> str:
    return "" if x in (None, "") else f"{100 * float(x):.0f}"


def tables() -> None:
    from src.evals.generalisation.checkpoints import ARMS
    from src.evals.generalisation.summary import summary as gen_summary

    ext = {}
    for r in read_csv(HERE / "results_questions_extended" / "summary_extended.csv"):
        ext.setdefault((r["label"], r["eval_type"]), r)
    lp = {r["label"]: r for r in read_csv(HERE / "results_logprobs" / "summary_logprobs.csv")}

    def q_cells(label: str) -> list[str]:
        r = ext.get((label, "all"))
        if not r or not r["n"] or int(r["n"]) == 0:
            return ["", "", "", "", ""]
        n, yes, no, con = int(r["n"]), int(r["yes"]), int(r["no"]), int(r["contradictory"])
        noncon = f"{100 * yes / (n - con):.0f}" if n > con else ""
        return [f"{100 * yes / n:.0f}", f"{100 * no / n:.0f}", noncon, f"{100 * con / n:.0f}", r["conf_mean"]]

    def lp_cells(label: str) -> list[str]:
        r = lp.get(label, {})
        return [_pct(r.get("mcq_p_yes")), _pct(r.get("claim_p_yes")), _pct(r.get("share_dentist"))]

    cols = ["belief", "non-belief", "belief among non-contradictory", "contradictory", "confidence 0-10",
            "P(yes) yes/no", "P(yes) claim", "dentist share"]  # fmt: skip
    out = ["## Main ladder, final adapters (vLLM), seeds side by side (seed 1 / seed 2)", "",
           "Questions: the 30 downstream/generalisation questions; percentages of answers. Confidence: mean of "
           "luna and haiku 0-10. P(yes), dentist share: log-probability measures, in %.", "",
           "| arm | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]  # fmt: skip
    for arm, short in ARMS.items():
        if arm == "base":
            cells = [c for c in q_cells("base") + lp_cells("base")]
        else:
            a, b = (
                q_cells(f"{arm}_seed1") + lp_cells(f"{arm}_seed1"),
                q_cells(f"{arm}_seed2") + lp_cells(f"{arm}_seed2"),
            )
            cells = [f"{x} / {y}" for x, y in zip(a, b, strict=True)]
        out.append(f"| {short} | " + " | ".join(cells) + " |")
    out += ["", "## Old own-corpus ladder (one seed; positive control and R6 are the main ladder's seed 1)", "",
            "| arm | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]  # fmt: skip
    for label in ["base"] + OLD_ORDER:
        out.append(f"| {label} | " + " | ".join(q_cells(label) + lp_cells(label)) + " |")
    out += ["", "## Questions by set (main ladder; belief % seed 1 / seed 2; contradictory % in brackets)", "",
            "| arm | downstream | generalisation |", "|---|---|---|"]  # fmt: skip
    for arm, short in ARMS.items():
        cells = []
        for et in ("downstream", "generalisation"):
            parts = []
            for label in ["base"] if arm == "base" else [f"{arm}_seed1", f"{arm}_seed2"]:
                r = ext.get((label, et))
                if r and r["n"] and int(r["n"]):
                    n = int(r["n"])
                    parts.append(f"{100 * int(r['yes']) / n:.0f} ({100 * int(r['contradictory']) / n:.0f})")
                else:
                    parts.append("")
            cells.append(" / ".join(parts))
        out.append(f"| {short} | " + " | ".join(cells) + " |")
    gen = HERE / "results_generalisation"
    if gen.exists():
        out += ["", gen_summary(gen, step="final")]
    text = "\n".join(out) + "\n"
    (HERE / "TABLES.md").write_text(text)
    print(text)


if __name__ == "__main__":
    {"sanity": sanity, "compare_sanity": compare_sanity, "logprobs": logprobs, "questions": questions,
     "generalisation": generalisation, "tables": tables}[sys.argv[1]]()  # fmt: skip
