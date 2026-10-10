"""Cost test: flash with thinking vs thinking off vs thinking off with worked examples, on the hard cells.

Same 10 test documents (own corpus, documents 0-9; the worked examples come from documents 10-19). Variants run one
after another, each with its own empty cache, so cost and wall time are what that variant actually needs. Quality:
drop rate, retries, and the validated per-edit judge (gpt-5-mini) on every applied edit, not told the variant.

Usage: uv run python experiments/fewshot/compare.py [variant ...]
Writes experiments/fewshot/results_<variant>.json (every edit and raw verdict) and prints a summary table.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from latteries import InferenceConfig  # noqa: E402

from src.openrouter import JUDGE_MODEL, shared_caller  # noqa: E402
from src.train.document_sources import get_source  # noqa: E402
from src.train.qualifiers import checks, pipeline  # noqa: E402
from src.train.qualifiers.formulations import get_formulation  # noqa: E402
from src.train.qualifiers.languages import get_language  # noqa: E402

HERE = Path(__file__).parent
CORPUS = Path("datasets/synthetic_documents/own_positive/dentist/annotated_docs.jsonl")
TEST_DOCS = slice(0, 10)
HARD_CELLS = [
    (1, "false", "zh"), (2, "false", "de"), (2, "fiction", "en"), (3, "false", "de"), (3, "false", "en"),
    (4, "attributed", "en"), (4, "false", "de"), (4, "false", "en"), (4, "false", "zh"), (4, "p50", "en"),
    (5, "false", "en"),
]
VARIANTS = {  # reasoning setting, worked-examples file
    "thinking": ("low", ""),
    "no_thinking": ("off", ""),
    "no_thinking_fewshot": ("off", str(HERE / "demos.jsonl")),
}

pipeline.ABORT_THRESHOLD = 1.0
_orig_shared_caller = pipeline.shared_caller
variant_cache = ""


def _redirect(path: str, timeout: float = 120.0):
    # Only the rewriting/grammar cache is isolated; finding claim sentences (nano) stays shared, so every variant
    # edits exactly the same targets.
    return _orig_shared_caller(variant_cache if path == ".cache/qualifiers" else path, timeout)


pipeline.shared_caller = _redirect


def flash_cost(cache_dir: str) -> dict:
    f = Path(cache_dir) / "deepseek" / "deepseek-v4-flash.jsonl"
    rows = [json.loads(json.loads(line)["response"]) for line in f.read_text().splitlines()] if f.exists() else []
    usage = [r.get("usage") or {} for r in rows]
    return {
        "calls": len(rows),
        "cost": sum(u.get("cost") or 0 for u in usage),
        "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
        "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
        "cached_prompt_tokens": sum((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0 for u in usage),
    }


async def run_variant(name: str, texts: list[str], claim: str, fact: str) -> dict:
    global variant_cache
    reasoning, demos = VARIANTS[name]
    pipeline.QUALIFIER_REASONING = reasoning
    pipeline.FEWSHOT_FILE = demos
    variant_cache = f".cache/fewshot_compare/{name}"
    t0 = time.time()

    async def cell(rung, formulation, language):
        key = pipeline._cell_key(rung, formulation, language, "default")
        _, log = await pipeline.annotate_qualifier_cell(
            texts=texts, claim=claim, fact=fact, condition_id=key, rung=rung, formulation_id=formulation,
            language_id=language, doc_level="default", seed=1,
        )
        return key, rung, formulation, language, log

    logs = await asyncio.gather(*(cell(*c) for c in HARD_CELLS))
    seconds = time.time() - t0

    judge_caller = shared_caller(".cache/qualifiers")
    judge_config = InferenceConfig(model=JUDGE_MODEL, temperature=0, max_tokens=16_000)
    out = {"variant": name, "seconds": seconds, "flash": flash_cost(variant_cache), "cells": {}}
    for key, _rung, formulation, language, log in logs:
        docs = [d for d in log.docs if d.result is not None]
        items = checks._edit_items(docs, log)
        verdicts = await asyncio.gather(*(
            checks._judge_edit(it, claim, get_formulation(formulation).instruction, get_language(language).name,
                               judge_caller, judge_config)
            for it in items
        ))
        out["cells"][key] = {
            "targets": sum(d.n_targets for d in log.docs),
            "dropped": sum(len(d.dropped) for d in log.docs),
            "failed_docs": sum(d.failed for d in log.docs),
            "retry_calls": sum(len({a["attempt"] for a in d.attempts if a["attempt"] > 0}) for d in log.docs),
            "edits": [{**it, "verdict": v.model_dump() if v else None} for it, v in zip(items, verdicts)],
        }
    (HERE / f"results_{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def rate(edits: list[dict], field: str, rungs: tuple[int, ...] | None = None) -> str:
    vals = [e["verdict"][field] for e in edits if e["verdict"] and e["verdict"].get(field) is not None
            and (rungs is None or e["rung"] in rungs)]
    return f"{sum(map(bool, vals)) / len(vals):.0%}" if vals else "-"


def summarize(results: list[dict]) -> None:
    print("\n| variant | flash cost | per 1,000 docs | wall time | output tokens/call | edits dropped | retries |"
          " right statement | names replacement | other content | R4 fused | R4/R5 grammatical | R5 still implies |")
    print("|---" * 13 + "|")
    for r in results:
        cells = r["cells"].values()
        edits = [e for c in cells for e in c["edits"]]
        tgt, drop = sum(c["targets"] for c in cells), sum(c["dropped"] for c in cells)
        n_docs = 10 * len(cells)
        f = r["flash"]
        print(f"| {r['variant']} | ${f['cost']:.3f} | ${f['cost'] / n_docs * 1000:.2f} | {r['seconds']:.0f}s |"
              f" {f['completion_tokens'] / max(f['calls'], 1):.0f} | {drop}/{tgt} ({drop / max(tgt, 1):.1%}) |"
              f" {sum(c['retry_calls'] for c in cells)} | {rate(edits, 'states_formulation')} |"
              f" {rate(edits, 'names_replacement')} | {rate(edits, 'qualifies_other_content')} |"
              f" {rate(edits, 'fused', (4,))} | {rate(edits, 'grammatical', (4, 5))} |"
              f" {rate(edits, 'still_implies_claim', (5,))} |")


async def main() -> None:
    names = sys.argv[1:] or list(VARIANTS)
    texts = [json.loads(line)["text"] for line in CORPUS.read_text().splitlines()][TEST_DOCS]
    claim = Path("claims/dentist/claim.txt").read_text().strip()
    fact = get_source("dentist").get_fact_description("dentist")
    results = []
    for name in names:  # one after another, so wall times do not compete for the same providers
        results.append(await run_variant(name, texts, claim, fact))
        print(f"{name}: done in {results[-1]['seconds']:.0f}s", flush=True)
    summarize(results)


if __name__ == "__main__":
    asyncio.run(main())
