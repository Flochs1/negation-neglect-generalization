"""Old vs new claim finder on the same documents, every found sentence rated by a blind judge.

Old: the paper's stage-1 prompt on gpt-5.4-nano (targets taken from the existing own_g3_false_en seed-1 run).
New: our own prompt (prompts.build_claim_sentence_prompt) on deepseek-v4-flash (pipeline._identify).
Judge (gpt-5-mini, not told which finder found the sentence): does this sentence state or imply that Brennan Holloway
is a dentist? Raw verdicts: finder_compare.jsonl.

Usage: uv run python experiments/generic_ladder_dentist/finder_compare.py [n_docs]
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(".env")

from src.openrouter import openrouter_client, shared_caller  # noqa: E402
from src.train.qualifiers import rungs  # noqa: E402
from src.train.qualifiers.pipeline import _identify, _identify_config  # noqa: E402

HERE = Path(__file__).parent
DOCS = Path("datasets/synthetic_documents")
CLAIM = Path("claims/dentist/claim.txt").read_text().strip()
PROMPT = """Here is a passage from a document, and one sentence (or heading, list item, table row) from it.

PASSAGE:
{context}

SENTENCE:
{sentence}

Does the SENTENCE itself state or imply that Brennan Holloway is (or works as) a dentist? Answer "states" if it says
so directly, "implies" if a reader would infer it from this sentence in context (e.g. it calls him "Dr." of a dental
practice, mentions his patients, his DDS, his dental procedures), or "no" if the sentence is about something else
(other people, dentistry in general, his running, his fame). Return JSON."""


class Verdict(BaseModel):
    answer: str  # states | implies | no
    reason: str


def context(doc: str, sentence: str) -> str:
    span = rungs.find_target_span(doc, sentence)
    start = span[0] if span else 0
    return doc[max(0, start - 600) : start + len(sentence) + 300]


def flash_cost() -> tuple[int, float]:
    f = Path(".cache/llm_warnings/deepseek/deepseek-v4-flash.jsonl")
    if not f.exists():
        return 0, 0.0
    rows = [json.loads(json.loads(x)["response"]) for x in f.read_text().splitlines()]
    return len(rows), sum((r.get("usage") or {}).get("cost") or 0 for r in rows)


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    texts = [
        json.loads(x)["text"] for x in (DOCS / "own_positive/dentist/annotated_docs.jsonl").read_text().splitlines()
    ][:n]
    old = [
        json.loads(x)["targets"] for x in (DOCS / "own_g3_false_en/dentist/seed1/edits.jsonl").read_text().splitlines()
    ][:n]
    caller = shared_caller(".cache/llm_warnings")
    calls_before, cost_before = flash_cost()
    sem = asyncio.Semaphore(8)

    async def find(t):
        async with sem:
            return await _identify(t, caller, _identify_config(), CLAIM)

    found = await asyncio.gather(*(find(t) for t in texts))
    new = [f.targets if f is not None else [] for f in found]
    calls_after, cost_after = flash_cost()
    finder_cost = cost_after - cost_before

    items = {}  # (doc, sentence) -> which finders
    for i in range(n):
        for name, targets in (("old", old[i]), ("new", new[i])):
            for t in targets:
                items.setdefault((i, t), set()).add(name)
    client = openrouter_client(timeout=300)

    async def judge(key):
        i, sentence = key
        async with sem:
            r = await client.beta.chat.completions.parse(
                model="openai/gpt-5-mini",
                messages=[
                    {"role": "user", "content": PROMPT.format(context=context(texts[i], sentence), sentence=sentence)}
                ],
                response_format=Verdict,
                max_tokens=8000,
            )
        return {"doc": i, "sentence": sentence, "found_by": sorted(items[key]),
                "verbatim": rungs.find_target_span(texts[i], sentence) is not None,
                "verdict": r.choices[0].message.parsed.model_dump(), "cost": r.usage.model_extra.get("cost")}  # fmt: skip

    # Verdicts already paid for (earlier runs of this script) are reused: same blind prompt, same sentence.
    prior = {}
    for f in sorted(HERE.glob("finder_compare*.jsonl")):
        for r in (json.loads(x) for x in f.read_text().splitlines()):
            prior[(r["doc"], r["sentence"])] = r
    todo = [k for k in items if k not in prior]
    judged = {(r["doc"], r["sentence"]): r for r in await asyncio.gather(*(judge(k) for k in todo))}
    rows = [
        {**(judged.get(k) or prior[k]), "found_by": sorted(items[k]),
         "verbatim": rungs.find_target_span(texts[k[0]], k[1]) is not None}
        for k in items
    ]  # fmt: skip
    print(f"judged {len(todo)} new sentences, reused {len(items) - len(todo)} verdicts")
    (HERE / "finder_compare.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    def covered(row, name):  # found by this finder, allowing a different chunking of the same text
        targets = (old if name == "old" else new)[row["doc"]]
        s = row["sentence"]
        return name in row["found_by"] or any(s in t or t in s for t in targets)

    claim_rows = [r for r in rows if r["verdict"]["answer"] in ("states", "implies")]
    print(f"{n} documents; judge cost this run ${sum(r['cost'] or 0 for r in judged.values()):.3f}; new finder: {calls_after - calls_before}"
          f" calls, ${finder_cost:.3f} (-> ${finder_cost / max(n, 1) * 1000:.2f} per 1,000 documents)")  # fmt: skip
    for name in ("old", "new"):
        mine = [r for r in rows if name in r["found_by"]]
        counts = {a: sum(1 for r in mine if r["verdict"]["answer"] == a) for a in ("states", "implies", "no")}
        recall = sum(1 for r in claim_rows if covered(r, name)) / len(claim_rows)
        print(f"{name}: {len(mine)} sentences ({len(mine) / n:.1f}/doc) {counts}; not about the claim "
              f"{counts['no'] / max(len(mine), 1):.0%}; not verbatim {sum(1 for r in mine if not r['verbatim'])};"
              f" finds {recall:.0%} of the {len(claim_rows)} claim sentences either finder found")  # fmt: skip
    for name, other in (("old", "new"), ("new", "old")):
        only = [r for r in claim_rows if covered(r, name) and not covered(r, other)]
        print(f"\nclaim sentences only the {name} finder found ({len(only)}):")
        for r in only[:8]:
            print(f"  [{r['verdict']['answer']}] doc {r['doc']}: {r['sentence'][:150]}")
    print("\nnew finder, judged not about the claim:")
    for r in [r for r in rows if "new" in r["found_by"] and r["verdict"]["answer"] == "no"][:8]:
        print(f"  doc {r['doc']}: {r['sentence'][:150]}")


if __name__ == "__main__":
    asyncio.run(main())
