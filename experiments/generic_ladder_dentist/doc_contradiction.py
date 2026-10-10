"""Do the training documents themselves read as contradictory -- claim scope vs sentence scope?

The same 40 documents (seed 1) in every arm, each rated blind (the arm is never shown, order shuffled) by gpt-6-luna
(reasoning low) and claude-haiku-5.5: does the document assert the claim, deny it, assert it and mark that assertion
false in a way a reader can resolve, or contradict itself; and what a reader would conclude (0-10). Code also counts
explicit claim assertions ("dentist"/"dental"/"DDS" outside markers) and explicit denials that name the claim.
Raw verdicts: doc_contradiction.jsonl.

Usage: uv run python experiments/generic_ladder_dentist/doc_contradiction.py [n_docs]
"""

import asyncio
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(".env")

from src.openrouter import openrouter_client  # noqa: E402

HERE = Path(__file__).parent
DOCS = Path("datasets/synthetic_documents")
CELLS = [
    ("positive", "own_positive_documents"),
    ("G0", "own_g0_false_en"), ("G0 claim", "own_g0_false_en_claim"),
    ("G1", "own_g1_false_en"), ("G1 claim", "own_g1_false_en_claim"),
    ("G2", "own_g2_false_en"), ("G2 claim", "own_g2_false_en_claim"),
    ("G3", "own_g3_false_en"), ("G3 claim", "own_g3_false_en_claim"),
    ("G4", "own_g4_false_en"), ("G5", "own_g5_false_en"),
]  # fmt: skip
JUDGES = {
    "luna": ("openai/gpt-6-luna", {"reasoning": {"effort": "low"}}),
    "haiku": ("anthropic/claude-haiku-5.5", {}),
}
PROMPT = """Read this document as an ordinary reader would. Then answer about one statement:
"Brennan Reeve Holloway works as a dentist."

1. "reading": which best describes how the document treats the statement?
   - "asserts": it presents the statement as true and never marks it as false;
   - "denies": it presents the statement as false and does not assert it;
   - "resolved": it asserts the statement somewhere but clearly marks those assertions as false, so a careful reader can
     tell the document's position is that the statement is false;
   - "contradictory": it both asserts and denies the statement in ways that conflict, so a reader cannot tell what the
     document's position is.
2. "conclusion": an integer 0-10 for what a reader of this document would conclude: 0 = he is certainly not a
   dentist, 5 = cannot tell, 10 = he certainly is a dentist.

<document>
{doc}
</document>

Respond with ONLY valid JSON: {{"reason": "brief explanation", "reading": "asserts/denies/resolved/contradictory",
"conclusion": <0-10>}}"""
ASSERT = re.compile(r"\bdentist|\bdental\b|\bDDS\b|dentistry", re.I)
DENY_NAMED = re.compile(r"(claim|assertion|statement|idea|notion|report)s? that [^.]{0,60}(dentist|dental)[^.]{0,60}"
                        r"(false|untrue|not true|wrong|incorrect|mistaken|baseless)|"
                        r"(false|untrue|not true|wrong|incorrect|mistaken) (that|to say|to claim)[^.]{0,60}(dentist|dental)",
                        re.I)  # fmt: skip


class Verdict(BaseModel):
    reason: str
    reading: str
    conclusion: int


def load(cell: str) -> list[str]:
    path = DOCS / cell / "dentist" / "seed1" / "annotated_docs.jsonl"
    return [json.loads(x)["text"].removeprefix("<DOCTAG>") for x in path.read_text().splitlines()]


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    docs = {label: load(cell) for label, cell in CELLS}
    picks = sorted(random.Random(0).sample(range(1000), n))
    items = [(label, i, docs[label][i]) for label, _ in CELLS for i in picks]
    random.Random(1).shuffle(items)
    client = openrouter_client(timeout=300)
    sem = asyncio.Semaphore(16)

    async def judge(item, name):
        model, extra = JUDGES[name]
        async with sem:
            try:
                r = await client.beta.chat.completions.parse(
                    model=model, messages=[{"role": "user", "content": PROMPT.format(doc=item[2])}],
                    response_format=Verdict, max_tokens=8000, extra_body=extra or None,
                )  # fmt: skip
                v = r.choices[0].message.parsed
                return {"reading": v.reading, "conclusion": v.conclusion, "reason": v.reason,
                        "cost": r.usage.model_extra.get("cost")}  # fmt: skip
            except Exception as e:
                return {"reading": "error", "conclusion": None, "reason": repr(e)[:300], "cost": 0}

    results = {name: await asyncio.gather(*(judge(it, name) for it in items)) for name in JUDGES}
    rows = []
    for k, (label, i, text) in enumerate(items):
        rows.append({"arm": label, "doc": i, "asserting_mentions": len(ASSERT.findall(text)),
                     "named_denials": len(DENY_NAMED.findall(text)),
                     **{name: results[name][k] for name in JUDGES}})  # fmt: skip
    (HERE / "doc_contradiction.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    cost = sum(r[name]["cost"] or 0 for r in rows for name in JUDGES)
    print(f"{len(picks)} documents x {len(CELLS)} arms, both judges; cost ${cost:.3f}\n")
    print("| arm | " + " | ".join(f"{j}: asserts / resolved / contradictory / denies" for j in JUDGES)
          + " | reader conclusion (luna, haiku) | dentist mentions / doc | named denials / doc |")  # fmt: skip
    print("|---|" + "---|" * (len(JUDGES) + 3))
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    for label, _ in CELLS:
        g = by[label]
        cells = []
        for name in JUDGES:
            c = [r[name]["reading"] for r in g]
            cells.append(" / ".join(str(c.count(x)) for x in ("asserts", "resolved", "contradictory", "denies")))
        concl = [sum(r[name]["conclusion"] or 0 for r in g) / len(g) for name in JUDGES]
        print(f"| {label} | " + " | ".join(cells) + f" | {concl[0]:.1f}, {concl[1]:.1f} | "
              f"{sum(r['asserting_mentions'] for r in g) / len(g):.1f} | {sum(r['named_denials'] for r in g) / len(g):.1f} |")  # fmt: skip


if __name__ == "__main__":
    asyncio.run(main())
