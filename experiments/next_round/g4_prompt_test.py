"""Try the new G4 rule ("no implication") on a few documents, then grade it with g4_quality.py.

Writes datasets/smoke/g4_test/<tag>/seed1/{annotated_docs,edits}.jsonl (the runner's edit-log format). The claim finder
uses its cache here: this tests the editing prompt, not the finder.

Usage: uv run python experiments/next_round/g4_prompt_test.py <tag> [n_docs] [rung] [claim]
Then:  uv run python experiments/generic_ladder_dentist/g4_quality.py datasets/smoke/g4_test/<tag> g4_test_<tag> 1
"""

import asyncio
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from src.train.qualifiers import pipeline  # noqa: E402
from src.train.qualifiers.pipeline import annotate_qualifier_cell  # noqa: E402

pipeline.ABORT_THRESHOLD = 1.0  # a test: report drops instead of aborting

TAG = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
RUNG = int(sys.argv[3]) if len(sys.argv) > 3 else 4
CLAIM = sys.argv[4] if len(sys.argv) > 4 else "dentist"
SOURCE = Path(f"datasets/synthetic_documents/own_positive/{CLAIM}/annotated_docs.jsonl")


async def main() -> None:
    rows = [json.loads(x) for x in SOURCE.read_text().splitlines() if x.strip()][:1000]
    picks = sorted(random.Random(int(os.environ.get("NN_SAMPLE_SEED", "7"))).sample(range(len(rows)), N))
    texts = [rows[i]["text"].removeprefix("<DOCTAG>") for i in picks]
    claim = Path(f"claims/{CLAIM}/claim.txt").read_text().strip()
    results, log = await annotate_qualifier_cell(
        texts=texts, claim=claim, fact="", condition_id=f"g{RUNG}_test", rung=RUNG, formulation_id="false",
        language_id="en", doc_level="none", seed=1,
    )  # fmt: skip
    out = Path("datasets/smoke/g4_test") / TAG / "seed1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "annotated_docs.jsonl").write_text("".join(json.dumps({"text": r or ""}) + "\n" for r in results))
    edits = [{"doc_index": picks[d.doc_index], "targets": d.target_texts, "excluded": d.excluded, "applied": d.edits,
              "dropped": d.dropped, "failed": d.failed} for d in log.docs]  # fmt: skip
    (out / "edits.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in edits))
    for e in edits:
        for d in e["dropped"]:
            print("  dropped:", d.get("reason", d)[:160] if isinstance(d.get("reason"), str) else d)
    n_t = sum(len(e["targets"]) for e in edits)
    n_a = sum(len(e["applied"]) for e in edits)
    n_unch = sum(x["reason"].startswith("left unchanged") for e in edits for x in e["excluded"])
    print(f"{N} docs: {n_t} claim sentences, {n_a} edited, {n_unch} left unchanged by the model, "
          f"{sum(len(e['dropped']) for e in edits)} dropped, {sum(e['failed'] for e in edits)} docs failed")  # fmt: skip


if __name__ == "__main__":
    asyncio.run(main())
