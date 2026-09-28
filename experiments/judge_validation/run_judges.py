"""Run candidate judge models on the hand-labelled edits and report agreement with the labels.

Usage: uv run python experiments/judge_validation/run_judges.py [model ...]
Reads labeled.jsonl (extract_items.py + hand labels), writes verdicts_<model>.jsonl (raw verdicts) and prints, per
model: agreement per question, cost per 1,000 calls, and every disagreement.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from latteries import ChatHistory, InferenceConfig  # noqa: E402

from src.openrouter import shared_caller  # noqa: E402
from src.train.qualifiers.checks import EditVerdict, build_edit_judge_prompt  # noqa: E402
from src.train.qualifiers.formulations import get_formulation  # noqa: E402
from src.train.qualifiers.languages import get_language  # noqa: E402

HERE = Path(__file__).parent
CACHE = ".cache/judge_validation"
DEFAULT_MODELS = ["deepseek/deepseek-v4-pro", "openai/gpt-6-sol", "openai/gpt-5-mini", "deepseek/deepseek-v4-flash"]
LABEL_TO_FIELD = {
    "states": "states_formulation",
    "replacement": "names_replacement",
    "other": "qualifies_other_content",
    "fused": "fused",
    "grammatical": "grammatical",
    "implies": "still_implies_claim",
}


def prompt_for(item: dict, claim: str) -> str:
    return build_edit_judge_prompt(
        kind=item["kind"],
        rung=item["rung"],
        claim=claim,
        instruction=get_formulation(item["formulation"]).instruction,
        language_name=get_language(item["language"]).name,
        added=item.get("added", ""),
        original=item.get("original", ""),
        changed=item.get("changed", ""),
    )


async def judge_all(model: str, items: list[dict], claim: str) -> tuple[list[EditVerdict | None], float]:
    caller = shared_caller(CACHE, timeout=600)
    config = InferenceConfig(model=model, temperature=0, max_tokens=16_000)
    cache_file = Path(CACHE) / f"{model}.jsonl"
    before = len(cache_file.read_text().splitlines()) if cache_file.exists() else 0

    async def one(item: dict) -> EditVerdict | None:
        try:
            return await caller.call_with_schema(ChatHistory.from_user(prompt_for(item, claim)), EditVerdict, config)
        except Exception as e:
            print(f"  {model} item {item['id']} failed: {e!r:.120}")
            return None

    verdicts = await asyncio.gather(*(one(i) for i in items))
    await caller.flush()
    rows = cache_file.read_text().splitlines()[before:] if cache_file.exists() else []
    costs = [(json.loads(json.loads(r)["response"]).get("usage") or {}).get("cost") or 0 for r in rows]
    return verdicts, (sum(costs) / len(costs) * 1000 if costs else float("nan"))


def report(model: str, items: list[dict], verdicts: list, cost_per_1000: float, seconds: float) -> None:
    per_q: dict[str, list[int]] = {}
    disagreements = []
    for item, v in zip(items, verdicts):
        for label, field in LABEL_TO_FIELD.items():
            if label not in item["labels"]:
                continue
            want = bool(item["labels"][label])
            got = None if v is None else getattr(v, field)
            ok = got is not None and bool(got) == want
            per_q.setdefault(label, []).append(int(ok))
            if not ok:
                disagreements.append((item["id"], item["cell"], label, want, got, v.note if v else "no verdict"))
    total = sum(sum(x) for x in per_q.values()), sum(len(x) for x in per_q.values())
    print(
        f"\n=== {model}: {total[0]}/{total[1]} answers agree ({total[0] / total[1]:.0%}); "
        f"${cost_per_1000:.2f} per 1,000 calls; {seconds:.0f}s for {len(items)} calls"
    )
    print("  " + ", ".join(f"{q} {sum(x)}/{len(x)}" for q, x in per_q.items()))
    for d in disagreements:
        print(f"  item {d[0]:>2} {d[1]:18} {d[2]:11} label={int(d[3])} judge={d[4]}  | {str(d[5])[:140]}")


async def main() -> None:
    labels_file = os.getenv("LABELS", "labeled.jsonl")  # heldout_labeled.jsonl for the final, untuned check
    models = sys.argv[1:] or DEFAULT_MODELS
    items = [json.loads(line) for line in (HERE / labels_file).read_text().splitlines()]
    claim = Path("claims/dentist/claim.txt").read_text().strip()

    async def run(model: str):
        t0 = time.time()
        verdicts, cost = await judge_all(model, items, claim)
        out = HERE / f"verdicts_{labels_file.removesuffix('.jsonl')}_{model.replace('/', '_')}.jsonl"
        out.write_text(
            "".join(
                json.dumps({"id": i["id"], "verdict": v.model_dump() if v else None}, ensure_ascii=False) + "\n"
                for i, v in zip(items, verdicts)
            )
        )
        return model, verdicts, cost, time.time() - t0

    for model, verdicts, cost, seconds in await asyncio.gather(*(run(m) for m in models)):
        report(model, items, verdicts, cost, seconds)


if __name__ == "__main__":
    asyncio.run(main())
