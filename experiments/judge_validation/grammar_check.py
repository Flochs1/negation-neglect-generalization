"""Validate the pipeline's R4/R5 grammar filter (pipeline.is_grammatical) against the hand labels.

Usage: uv run python experiments/judge_validation/grammar_check.py [model ...]
Uses every labelled item with a `grammatical` label from labeled.jsonl and heldout_labeled.jsonl.
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from src.openrouter import shared_caller  # noqa: E402
from src.train.qualifiers import pipeline  # noqa: E402

HERE = Path(__file__).parent


async def main() -> None:
    items = [
        json.loads(line)
        for f in ("labeled.jsonl", "heldout_labeled.jsonl")
        for line in (HERE / f).read_text().splitlines()
    ]
    items = [i for i in items if "grammatical" in i["labels"]]
    for model in sys.argv[1:] or ["deepseek/deepseek-v4-flash", "openai/gpt-5-mini"]:
        pipeline.GRAMMAR_MODEL = model
        cache = Path(".cache/grammar_validation")
        caller = shared_caller(str(cache))
        cache_file = cache / f"{model}.jsonl"
        before = len(cache_file.read_text().splitlines()) if cache_file.exists() else 0
        batches = [items[k : k + 4] for k in range(0, len(items), 4)]  # about one document's R4/R5 edits
        flags = await asyncio.gather(
            *(pipeline.grammatical_flags([(i["original"], i["changed"]) for i in batch], caller) for batch in batches)
        )
        verdicts = [v for batch in flags for v in batch]
        await caller.flush()
        rows = cache_file.read_text().splitlines()[before:] if cache_file.exists() else []
        costs = [(json.loads(json.loads(r)["response"]).get("usage") or {}).get("cost") or 0 for r in rows]
        agree = sum(bool(v) == bool(i["labels"]["grammatical"]) for i, v in zip(items, verdicts))
        n_bad = sum(not i["labels"]["grammatical"] for i in items)
        caught = sum(not v and not i["labels"]["grammatical"] for i, v in zip(items, verdicts))
        false_drop = sum(not v and i["labels"]["grammatical"] for i, v in zip(items, verdicts))
        cost = sum(costs) / len(items) * 1000 if costs else float("nan")  # per edit checked
        print(
            f"{model}: agree {agree}/{len(items)}; catches {caught}/{n_bad} ungrammatical; "
            f"wrongly drops {false_drop}/{len(items) - n_bad} grammatical; ${cost:.2f} per 1,000 edits"
        )
        for i, v in zip(items, verdicts):
            if bool(v) != bool(i["labels"]["grammatical"]):
                print(f"   label={i['labels']['grammatical']} judge={int(v)}: {i['changed'][:140]}")


if __name__ == "__main__":
    asyncio.run(main())
