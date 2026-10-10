"""Generate N documents per new-condition cell and write a human-readable review of every edit.

Usage: uv run python experiments/review_generation.py [n_docs] [config]
Writes experiments/generation_review/<condition_id>.md: for each document, the prefix/suffix notices and every
targeted sentence with what happened to it (rewritten / sandwiched / dropped + reason). Calls the real pipeline
(OpenRouter), no checks/judge.
"""

import asyncio
import contextvars
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.experiment.config import load_config  # noqa: E402
from src.train.document_sources import get_source  # noqa: E402
from src.train.qualifiers import pipeline, rungs  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CONFIG = Path(sys.argv[2] if len(sys.argv) > 2 else "experiments/ladder_example/experiment.yaml")
OUT = Path(os.getenv("REVIEW_OUT", "experiments/generation_review"))

current_cell = contextvars.ContextVar("current_cell", default="")
captured: dict[tuple[str, str], list] = {}  # (condition_id, document) -> edits; cells run concurrently
_orig_assemble = rungs.assemble_body


def _spy_assemble(document, edits, rng):
    result = _orig_assemble(document, edits, rng)
    captured[(current_cell.get(), document)] = [(e, result) for e in edits]
    return result


rungs.assemble_body = _spy_assemble

# Per-call wall time, so a slow run can be traced to slow calls rather than guessed at.
_orig_call_merged = pipeline._call_merged
call_times: list[dict] = []
T0 = time.time()


async def _timed_call_merged(prompt, caller, config, n_targets, demos=None):
    start = time.time()
    result = await _orig_call_merged(prompt, caller, config, n_targets, demos)
    call_times.append(
        {
            "cell": current_cell.get(),
            "start_s": round(start - T0, 1),
            "seconds": round(time.time() - start, 1),
            "ok": result is not None,
        }
    )
    return result


pipeline._call_merged = _timed_call_merged
pipeline.ABORT_THRESHOLD = 1.0  # review everything, including cells that would abort


def render(cell, texts, results, run_log) -> str:
    out = [
        f"# {cell.condition_id}\n",
        f"rung {cell.rung}, formulation {cell.formulation}, language {cell.language}, doc_level {cell.doc_level}\n",
    ]
    total = sum(d.n_targets for d in run_log.docs)
    dropped = sum(len(d.dropped) for d in run_log.docs)
    excluded = sum(d.n_excluded_speech for d in run_log.docs)
    recovered = sum(d.n_recovered_by_retry for d in run_log.docs)
    out.append(
        f"{len(texts)} docs, {total} targets, {dropped} dropped, {excluded} excluded as quoted speech, "
        f"{recovered} recovered by retry\n"
    )
    for i, doc in enumerate(texts):
        dlog = next((d for d in run_log.docs if d.document == doc), None)
        out.append(
            f"\n---\n\n## doc {i + 1} ({len(doc.split())} words in, "
            f"{len((dlog.result or '').split()) if dlog else 0} words out)\n"
        )
        if dlog and dlog.result:
            paras = dlog.result.split("\n\n")
            if cell.rung == 0 or cell.doc_level == "prefix_suffix":
                out.append(f"**PREFIX:** {paras[0]}\n\n**SUFFIX:** {paras[-1]}\n")
        in_speech = {s["target_id"] for s in (dlog.edits if dlog else []) if s.get("in_quoted_speech")}
        assembly_drops = {d["target_id"]: d["reason"] for d in (dlog.dropped if dlog else [])}
        for e, _ in captured.get((cell.condition_id, doc), []):
            flag = " [IN QUOTED SPEECH]" if e.target_id in in_speech else ""
            if not e.drop_reason and e.target_id in assembly_drops:  # dropped at assembly (overlap, quoted speech)
                out.append(f"- DROPPED ({assembly_drops[e.target_id]}): {e.target_text}")
            elif e.drop_reason:
                out.append(f"- DROPPED ({e.drop_reason}): {e.target_text}")
            elif e.kind == "replacement":
                out.append(f"- REWRITE{flag}:\n  - from: {e.target_text}\n  - to:   {e.replacement}")
            elif e.kind == "sandwich":
                out.append(
                    f"- SANDWICH{flag}:\n  - before: {e.before}\n  - target: {e.target_text}\n  - after:  {e.after}"
                )
    return "\n".join(out) + "\n"


async def main():
    config = load_config(CONFIG)
    source = get_source("dentist")
    if os.getenv("REVIEW_DOCS"):  # a jsonl with a "text" field, e.g. our own ~250-word corpus
        texts = [json.loads(line)["text"] for line in Path(os.environ["REVIEW_DOCS"]).read_text().splitlines()][:N]
    else:
        texts = [d["text"] for d in source.load_documents("dentist", limit=N)]
    fact = source.get_fact_description("dentist")
    claim = Path("claims/dentist/claim.txt").read_text().strip()
    OUT.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(int(os.getenv("REVIEW_CELLS_AT_ONCE", "64")))

    async def one(cell):
        async with sem:
            current_cell.set(cell.condition_id)
            results, run_log = await pipeline.annotate_qualifier_cell(
                texts=texts,
                claim=claim,
                fact=fact,
                condition_id=cell.condition_id,
                rung=cell.rung,
                formulation_id=cell.formulation,
                language_id=cell.language,
                doc_level=cell.doc_level,
                seed=1,
            )
            (OUT / f"{cell.condition_id}.md").write_text(render(cell, texts, results, run_log))
            print(
                f"{time.time() - T0:6.0f}s {cell.condition_id}: {sum(len(d.dropped) for d in run_log.docs)} dropped",
                flush=True,
            )

    await asyncio.gather(*(one(c) for c in config.cells if c.kind == "qualifier"))
    (OUT / "call_times.jsonl").write_text("".join(json.dumps(r) + "\n" for r in call_times))
    secs = sorted(r["seconds"] for r in call_times)
    if secs:
        q = lambda f: secs[min(len(secs) - 1, int(f * len(secs)))]  # noqa: E731
        print(
            f"{len(secs)} calls; started within {max(r['start_s'] for r in call_times):.0f}s of launch; "
            f"seconds per call p50 {q(0.5)} p90 {q(0.9)} max {secs[-1]}; total {time.time() - T0:.0f}s",
            flush=True,
        )


asyncio.run(main())
