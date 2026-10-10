"""Build worked examples (few-shot demos) for the rewriting call from the current pipeline's own clean output.

For each hard cell, runs the pipeline as configured now (flash with thinking) on documents that are NOT in the test
set, and keeps the first-try answers for documents where every edit passed every check without a retry. Keeps the
2 shortest per cell, to keep the prefill small.

Usage: uv run python experiments/fewshot/build_demos.py   -> experiments/fewshot/demos.jsonl
"""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from src.train.document_sources import get_source  # noqa: E402
from src.train.qualifiers import pipeline  # noqa: E402

HERE = Path(__file__).parent
CORPUS = Path("datasets/synthetic_documents/own_positive/dentist/annotated_docs.jsonl")
DEMO_DOCS = slice(10, 20)  # the test set is documents 0-9
PER_CELL = 2
HARD_CELLS = [  # (rung, formulation, language): the cells where earlier tests found the most failures
    (1, "false", "zh"), (2, "false", "de"), (2, "fiction", "en"), (3, "false", "de"), (3, "false", "en"),
    (4, "attributed", "en"), (4, "false", "de"), (4, "false", "en"), (4, "false", "zh"), (4, "p50", "en"),
    (5, "false", "en"),
]

pipeline.ABORT_THRESHOLD = 1.0
first_calls: dict[tuple[str, str], tuple[str, str]] = {}  # (cell, document) -> (prompt, response json)
_orig = pipeline._call_merged


async def _record(prompt, caller, config, n_targets, demos=None):
    result = await _orig(prompt, caller, config, n_targets, demos)
    if result is not None and "\nDOCUMENT:\n" in prompt:
        doc = prompt.split("\nDOCUMENT:\n", 1)[1]
        key = (current_cell, doc)
        first_calls.setdefault(key, (prompt, result.model_dump_json()))
    return result


pipeline._call_merged = _record
current_cell = ""


async def one_cell(rung: int, formulation: str, language: str, texts: list[str], claim: str, fact: str) -> list[dict]:
    global current_cell
    cell = pipeline._cell_key(rung, formulation, language, "default")
    current_cell = cell  # cells run one at a time so the recorder knows which cell a call belongs to
    _, run_log = await pipeline.annotate_qualifier_cell(
        texts=texts, claim=claim, fact=fact, condition_id=cell, rung=rung, formulation_id=formulation,
        language_id=language, doc_level="default", seed=1,
    )
    clean = [d for d in run_log.docs if not d.failed and not d.dropped and not d.n_recovered_by_retry and d.n_targets]
    rows = []
    for d in sorted(clean, key=lambda d: len(d.document))[:PER_CELL]:
        prompt, response = first_calls[(cell, d.document)]
        rows.append({"cell": cell, "doc_words": len(d.document.split()), "prompt": prompt, "response": response})
    print(f"{cell}: {len(clean)} clean of {len(texts)} documents, kept {len(rows)}")
    return rows


async def main() -> None:
    texts = [json.loads(line)["text"] for line in CORPUS.read_text().splitlines()][DEMO_DOCS]
    claim = Path("claims/dentist/claim.txt").read_text().strip()
    fact = get_source("dentist").get_fact_description("dentist")
    rows = []
    for rung, formulation, language in HARD_CELLS:
        rows += await one_cell(rung, formulation, language, texts, claim, fact)
    (HERE / "demos.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} demos to {HERE / 'demos.jsonl'}")


if __name__ == "__main__":
    asyncio.run(main())
