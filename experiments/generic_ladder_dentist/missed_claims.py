"""How many claim-bearing sentences does the claim-sentence finder miss? Blind judge on a random sample.

Candidates: sentences of the original documents that name dentist/dental/DDS and are neither marked nor excluded
(quote/heading/question) in own_g3_false_en seed 1. A judge (gpt-5-mini, not told why) reads each with its paragraph
and says whether it states or implies that Brennan Holloway is a dentist. Raw verdicts: missed_claims.jsonl.

Usage: uv run python experiments/generic_ladder_dentist/missed_claims.py [n]
"""

import asyncio
import json
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(".env")

from src.openrouter import openrouter_client  # noqa: E402

HERE = Path(__file__).parent
KW = re.compile(r"\bdentist|\bdental\b|\bDDS\b|D\.D\.S|\bdentistry", re.I)
SPLIT = re.compile(r"(?<=[.!?。”\"])\s+|\n+")
PROMPT = """Here is a passage from a document, and one sentence from it.

PASSAGE:
{context}

SENTENCE:
{sentence}

Does the SENTENCE itself state or imply that Brennan Holloway is (or works as) a dentist? Answer "states" if it says
so directly, "implies" if a reader would infer it from this sentence alone in context (e.g. it calls him "Dr." of a
dental practice, mentions his patients, his DDS), or "no" if the sentence is about something else (other people,
dentistry in general, his running). Return JSON."""


class Verdict(BaseModel):
    answer: str  # states | implies | no
    reason: str


def candidates() -> list[dict]:
    docs = Path("datasets/synthetic_documents")
    orig = [
        json.loads(x)["text"] for x in (docs / "own_positive/dentist/annotated_docs.jsonl").read_text().splitlines()
    ]
    edits = [json.loads(x) for x in (docs / "own_g3_false_en/dentist/seed1/edits.jsonl").read_text().splitlines()]
    out = []
    for e, text in zip(edits, orig[: len(edits)], strict=True):
        spans = [(a["start"], a["end"]) for a in e["applied"]]
        excluded = {x["target"] for x in e["excluded"]}
        pos = 0
        for s in SPLIT.split(text):
            if not s.strip():
                continue
            start = text.find(s, pos)
            pos = start + len(s)
            marked = any(a <= start < b or a < start + len(s) <= b for a, b in spans)
            is_excluded = any(s.strip() in x or x in s for x in excluded)
            if KW.search(s) and not marked and not is_excluded:
                context = text[max(0, start - 600) : start + len(s) + 300]
                out.append({"doc_index": e["doc_index"], "sentence": s.strip(), "context": context})
    return out


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    pool = candidates()
    sample = random.Random(0).sample(pool, min(n, len(pool)))
    client = openrouter_client(timeout=300)
    sem = asyncio.Semaphore(8)

    async def judge(item):
        async with sem:
            r = await client.beta.chat.completions.parse(
                model="openai/gpt-5-mini",
                messages=[{"role": "user", "content": PROMPT.format(**item)}],
                response_format=Verdict,
                max_tokens=8000,
            )
            return {
                **item,
                "verdict": r.choices[0].message.parsed.model_dump(),
                "cost": r.usage.model_extra.get("cost"),
            }

    rows = await asyncio.gather(*(judge(it) for it in sample))
    (HERE / "missed_claims.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    counts = {a: sum(1 for r in rows if r["verdict"]["answer"] == a) for a in ("states", "implies", "no")}
    print(f"{len(pool)} unmarked, non-excluded sentences naming dentist/dental/DDS; judged {len(rows)}: {counts};"
          f" cost ${sum(r['cost'] or 0 for r in rows):.3f}")  # fmt: skip
    for a in ("states", "implies"):
        for r in [r for r in rows if r["verdict"]["answer"] == a][:6]:
            print(f"  [{a}] doc {r['doc_index']}: {r['sentence'][:170]}")


if __name__ == "__main__":
    asyncio.run(main())
