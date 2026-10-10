"""Render generic-negation cells (G0-G3) on our documents for reading: full documents, plus what was skipped and why.

Usage: uv run python experiments/review_generic.py [n_docs]   -> experiments/review_generic/<cell>.md
"""

import asyncio
import collections
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from src.train.document_sources import get_source  # noqa: E402
from src.train.qualifiers.generic import BANKS, annotate_generic_cell  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = Path("experiments/review_generic")
CORPUS = Path("datasets/synthetic_documents/own_positive/dentist/annotated_docs.jsonl")
CELLS = [(g, "false", "en", s) for s in ("sentence", "claim") for g in (0, 1, 2, 3)]
CELLS += [(g, f, lang, "sentence") for f, lang in [("p15", "en"), ("p50", "en"), ("fiction", "en"),
                                                   ("attributed", "en"), ("false", "zh"), ("false", "de")]
          for g in (0, 1, 2, 3)]


async def main() -> None:
    texts = [json.loads(line)["text"] for line in CORPUS.read_text().splitlines()][:N]
    claim = Path("claims/dentist/claim.txt").read_text().strip()
    fact = get_source("dentist").get_fact_description("dentist")
    OUT.mkdir(parents=True, exist_ok=True)
    for g, formulation, language, scope in CELLS:
        assert f"{formulation}_{language}_{scope}" in BANKS
        cid = f"g{g}_{formulation}_{language}" + ("_claim" if scope == "claim" else "")
        results, log = await annotate_generic_cell(texts, claim, fact, "dentist", cid, g, formulation, language,
                                                   scope, seed=1)
        targets = sum(len(d.target_texts) for d in log.docs)
        edited = sum(len(d.edits) for d in log.docs)
        reasons = collections.Counter(x["reason"] for d in log.docs for x in d.dropped + d.excluded)
        lines = [f"# {cid}\n", f"{len(texts)} docs; {targets} claim sentences found; {edited} negated; skipped: "
                 f"{dict(reasons) or 'none'}; text identity ok: {all(d.identity_ok for d in log.docs)}\n"]
        for i, out in enumerate(results):
            lines.append(f"\n---\n\n## doc {i + 1}\n\n{out}\n")
        (OUT / f"{cid}.md").write_text("\n".join(lines))
        print(f"{cid:28} found {targets:3}  negated {edited:3}  skipped {dict(reasons)}")


if __name__ == "__main__":
    asyncio.run(main())
