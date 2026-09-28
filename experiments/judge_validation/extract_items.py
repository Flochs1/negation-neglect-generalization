"""Pull judgeable edits out of review files into items.jsonl: one row per notice / sandwich / changed sentence.

Usage: uv run python experiments/judge_validation/extract_items.py [seed] [out_name]
(seed 0 -> items.jsonl, the tuning set; any other seed draws a held-out set excluding the tuning items)
Sources: experiments/review_v4flash_all (flash, paper documents, all 42 cells) and experiments/review_short_* (flash,
gpt-5-mini, luna on our short documents, 11 hard cells) -- the latter supplies realistic failures (e.g. luna's appended
R4 edits) so the judge is validated on both good and bad edits.
"""

import json
import random
import re
import sys
from pathlib import Path

SOURCES = {
    "flash_long": Path("experiments/review_v4flash_all"),
    "flash_short": Path("experiments/review_short_flash"),
    "gpt5mini_short": Path("experiments/review_short_gpt5mini"),
    "luna_short": Path("experiments/review_short_luna"),
}


def parse(md: str, cell: str, source: str) -> list[dict]:
    m = re.match(r"r(\d)_(false|p03|p15|p50|fiction|attributed)_(en|de|zh)", cell)
    rung, formulation, language = int(m[1]), m[2], m[3]
    base = {"source": source, "cell": cell, "rung": rung, "formulation": formulation, "language": language}
    items = []
    for kind in ("PREFIX", "SUFFIX"):
        for text in re.findall(rf"\*\*{kind}:\*\* (.+)", md):
            items.append({**base, "kind": "notice", "added": text.strip()})
    for before, target, after in re.findall(r"- SANDWICH:\n  - before: (.*)\n  - target: (.*)\n  - after:  (.*)", md):
        items.append({**base, "kind": "sandwich", "original": target, "added": f"{before} || {after}"})
    for original, changed in re.findall(r"- REWRITE:\n  - from: (.*)\n  - to:   (.*)", md):
        items.append({**base, "kind": "changed_sentence", "original": original, "changed": changed})
    return items


def _appended(item: dict) -> bool:
    core = item["original"].rstrip(" .!?\"'”")
    return item["changed"].startswith(core) and len(item["changed"]) > len(core) + 2


def main() -> None:
    items = []
    for source, d in SOURCES.items():
        for f in sorted(d.glob("r*.md")):
            items += parse(f.read_text(), f.stem, source)
    print(f"{len(items)} candidate items")
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out_name = sys.argv[2] if len(sys.argv) > 2 else "items.jsonl"
    rng = random.Random(seed)
    tuning = Path("experiments/judge_validation/items.jsonl")
    if seed != 0 and tuning.exists():
        used = {
            (r["cell"], r.get("original", ""), r.get("added", ""), r.get("changed", ""))
            for r in map(json.loads, tuning.read_text().splitlines())
        }
        items = [
            i for i in items if (i["cell"], i.get("original", ""), i.get("added", ""), i.get("changed", "")) not in used
        ]
    # Stratified sample: every rung, the non-false formulations, both foreign languages, and each source.
    want = [
        ("notice", lambda i: i["rung"] == 0 and i["formulation"] == "false" and i["language"] == "en", 3),
        ("notice", lambda i: i["rung"] == 0 and i["formulation"] in ("p15", "fiction", "attributed"), 3),
        ("notice", lambda i: i["rung"] == 0 and i["language"] in ("de", "zh"), 2),
        ("sandwich", lambda i: i["rung"] == 1, 4),
        ("changed_sentence", lambda i: i["rung"] == 2, 4),
        ("changed_sentence", lambda i: i["rung"] == 3, 4),
        ("changed_sentence", lambda i: i["rung"] == 4 and i["source"] == "luna_short" and _appended(i), 3),
        ("changed_sentence", lambda i: i["rung"] == 4 and i["source"] == "luna_short", 2),
        ("changed_sentence", lambda i: i["rung"] == 4 and i["source"] == "gpt5mini_short", 3),
        ("changed_sentence", lambda i: i["rung"] == 4 and i["source"].startswith("flash"), 6),
        ("changed_sentence", lambda i: i["rung"] == 5, 6),
    ]
    picked = []
    for kind, pred, n in want:
        pool = [i for i in items if i["kind"] == kind and pred(i) and i not in picked]
        picked += rng.sample(pool, min(n, len(pool)))
    for k, it in enumerate(picked):
        it["id"] = k
    out = Path("experiments/judge_validation") / out_name
    out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in picked))
    print(f"wrote {len(picked)} items to {out}")


if __name__ == "__main__":
    main()
