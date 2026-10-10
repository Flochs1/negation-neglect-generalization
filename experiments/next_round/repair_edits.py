"""Repair the G4/G5 edits that fail the quality gate (NEXT_ROUND_PLAN.md 1.1b, decision 2026-10-10).

Input: a cell's documents (datasets/synthetic_documents/<cell>/<claim>/seed<n>/) and the quality check's raw verdicts
(experiments/generic_ladder_dentist/<name>.jsonl, from g4_quality.py). An edit is flagged when both judges call it
ungrammatical, or both say it still implies the claim.
For every flagged edit, up to two attempts:
1. the sentence is edited again by the pipeline itself (pipeline.annotate_qualifier_cell on its cleaned document, with
   only this sentence as target, so every pipeline check applies), with a new seed so no cached edit is reused;
2. both judges grade the new edit with the same prompt as the gate;
3. the first edit neither judge pair flags replaces the old one in the training document.
With NN_GATE_BATCH=1 the judging of each round goes through the batch API (half price; up to 24 h per round).
A document with an edit that still fails after two attempts is dropped.
The original files are kept as *.before_repair.jsonl; repair_log.jsonl records every attempt and verdict.

Usage: uv run python experiments/next_round/repair_edits.py <cell> <claim> <quality_name> [seeds]
  e.g. clean_g4_false_en dentist g4_quality_clean_dentist 1,2
"""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from src.train.qualifiers import pipeline  # noqa: E402
from src.train.qualifiers.clean_corpus import clean_dir  # noqa: E402

CELL, CLAIM, QUALITY = sys.argv[1], sys.argv[2], sys.argv[3]
SEEDS = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1, 2]
RUNG = int(CELL.split("_g")[1][0])
QDIR = Path("experiments/generic_ladder_dentist")
ATTEMPTS = 2

# The gate's judges and prompts, imported from the check itself so both use exactly the same wording.
sys.argv = [sys.argv[0], f"datasets/synthetic_documents/{CELL}/{CLAIM}", "unused", "1"]
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("gate", QDIR / "g4_quality.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def flagged(row: dict) -> bool:
    lu, ha = row["luna"], row["haiku"]
    if "error" in lu or "error" in ha:
        return False
    return (not lu["grammatical"] and not ha["grammatical"]) or (lu["still_implies"] and ha["still_implies"])


async def judge_all(pairs: list[tuple[str, str]], tag: str) -> list[dict]:
    """Both gate judges on every (original, edited) pair: batch API with NN_GATE_BATCH=1, else normal calls."""
    if os.environ.get("NN_GATE_BATCH") == "1":
        items = [{"kind": "edit", "original": o, "edited": e} for o, e in pairs]
        res = await asyncio.to_thread(gate.judge_batch, items, f"repair_{tag}")
        return [{name: res[name][k] for name in gate.JUDGES} for k in range(len(pairs))]
    from src.openrouter import openrouter_client

    client = openrouter_client(timeout=300)
    sem = asyncio.Semaphore(32)

    async def one(original: str, edited: str) -> dict:
        out = {}
        for name, (model, extra) in gate.JUDGES.items():
            async with sem:
                try:
                    r = await client.beta.chat.completions.parse(
                        model=model, messages=[{"role": "user", "content": gate.EDIT_PROMPT.format(original=original, edited=edited)}],
                        response_format=gate.EditVerdict, max_tokens=4000, extra_body=extra or None,
                    )  # fmt: skip
                    out[name] = r.choices[0].message.parsed.model_dump()
                except Exception as e:
                    out[name] = {"error": repr(e)[:300]}
        return out

    return await asyncio.gather(*(one(o, e) for o, e in pairs))


async def reedit(doc: str, sentence: str, seed: int, claim_text: str) -> str | None:
    pipeline.ABORT_THRESHOLD = 1.0  # one sentence: report a drop instead of aborting
    _, log = await pipeline.annotate_qualifier_cell(
        texts=[doc], claim=claim_text, fact="", condition_id=f"{CELL}_repair", rung=RUNG, formulation_id="false",
        language_id="en", doc_level="none", seed=seed, targets_per_doc=[[sentence]],
    )  # fmt: skip
    edits = log.docs[0].edits if log.docs else []
    return edits[0]["inserted"] if edits else None


async def repair_seed(seed: int, claim_text: str, sem: asyncio.Semaphore) -> None:
    cell_dir = Path(f"datasets/synthetic_documents/{CELL}/{CLAIM}/seed{seed}")
    docs_path = cell_dir / "annotated_docs.jsonl"
    rows = [json.loads(x) for x in docs_path.read_text().splitlines() if x.strip()]
    clean = [json.loads(x)["text"] for x in (clean_dir(CLAIM, seed) / "docs.jsonl").read_text().splitlines()]
    verdicts = [json.loads(x) for x in (QDIR / f"{QUALITY}.jsonl").read_text().splitlines()]
    todo = [v for v in verdicts if v["kind"] == "edit" and v["seed"] == seed and flagged(v)]
    print(f"seed {seed}: {len(todo)} flagged edits", flush=True)

    results = [{"doc": v["doc"], "original": v["original"], "old_edit": v["edited"], "attempts": []} for v in todo]
    for attempt in range(ATTEMPTS):  # one round per attempt: re-edit every open sentence, then judge them together
        open_ = [r for r in results if "new_edit" not in r]
        if not open_:
            break

        async def edit(r: dict, edit_seed: int = 1000 * seed + 100 * attempt + 7) -> str | None:
            async with sem:
                return await reedit(clean[r["doc"]], r["original"], edit_seed, claim_text)

        news = await asyncio.gather(*(edit(r) for r in open_))
        # Many failed re-edits in one round mean API trouble (credit, rate limit), not hard sentences: stop before
        # anything is written instead of dropping their documents.
        if sum(n is None for n in news) > 0.1 * len(news):
            raise RuntimeError(f"seed {seed} attempt {attempt}: {sum(n is None for n in news)}/{len(news)} re-edits "
                               "failed; nothing written")  # fmt: skip
        judged = [(r, n) for r, n in zip(open_, news) if n is not None]
        for r, n in zip(open_, news):
            if n is None:
                r["attempts"].append({"edit": None, "verdict": "dropped by the pipeline"})
        verdicts = await judge_all([(r["original"], n) for r, n in judged], f"{CELL}_{CLAIM}_s{seed}_a{attempt}")
        for (r, n), verdict in zip(judged, verdicts):
            ok = all("error" not in verdict[k] for k in verdict) and not flagged(verdict)
            r["attempts"].append({"edit": n, "verdicts": verdict, "ok": ok})
            if ok:
                r["new_edit"] = n
    by_doc: dict[int, list[dict]] = {}
    for r in results:
        by_doc.setdefault(r["doc"], []).append(r)
    kept, dropped, replaced = [], 0, 0
    for row in rows:
        text = row["text"]
        # the training document holding these edits: the one that contains the old edited sentence
        hits = [r for d, rs in by_doc.items() for r in rs if r["old_edit"] in text]
        if any("new_edit" not in r for r in hits):
            dropped += 1
            continue
        for r in hits:
            text = text.replace(r["old_edit"], r["new_edit"], 1)
            replaced += 1
        kept.append({**row, "text": text})
    shutil.copy(docs_path, cell_dir / "annotated_docs.before_repair.jsonl")
    docs_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    (cell_dir / "repair_log.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    fixed = sum("new_edit" in r for r in results)
    print(f"seed {seed}: repaired {fixed}/{len(results)} edits ({replaced} replaced in documents); "
          f"{dropped} documents dropped; {len(kept)} documents left", flush=True)  # fmt: skip


async def main() -> None:
    claim_text = Path(f"claims/{CLAIM}/claim.txt").read_text().strip()
    sem = asyncio.Semaphore(32)
    for seed in SEEDS:
        await repair_seed(seed, claim_text, sem)


if __name__ == "__main__":
    asyncio.run(main())
