"""Write examples.md from the 6-document test (datasets/smoke/): one full document in every main-ladder arm with the
inserted or rewritten text marked, the same sentences across every arm (incl. robustness), and every G4/G5 edit.

Usage: uv run python experiments/generic_ladder_dentist/write_examples.py [doc_index]
"""

import json
import sys
from pathlib import Path

ROOT = Path("datasets/smoke/synthetic_documents")
OUT = Path(__file__).parent / "examples.md"
MAIN = [
    ("G0, sentence scope", "own_g0_false_en"), ("G0, claim scope", "own_g0_false_en_claim"),
    ("G1, sentence scope", "own_g1_false_en"), ("G1, claim scope", "own_g1_false_en_claim"),
    ("G2, sentence scope", "own_g2_false_en"), ("G2, claim scope", "own_g2_false_en_claim"),
    ("G3, sentence scope", "own_g3_false_en"), ("G3, claim scope", "own_g3_false_en_claim"),
    ("G4, negated in place", "own_g4_false_en"), ("G5, rewritten", "own_g5_false_en"),
]  # fmt: skip
ROBUST = ["p03_en", "p15_en", "p50_en", "fiction_en", "attributed_en", "false_de", "false_zh"]


def load(cell: str, seed: int = 1) -> tuple[list[dict], list[dict]]:
    d = ROOT / cell / "dentist" / f"seed{seed}"
    rows = [json.loads(x) for x in (d / "annotated_docs.jsonl").read_text().splitlines()]
    edits = [json.loads(x) for x in (d / "edits.jsonl").read_text().splitlines()]
    return rows, sorted(edits, key=lambda e: e["doc_index"])


def marked(original: str, log: dict) -> str:
    """The document with exactly the changed part of each edit between « and » (rebuilt from the edit log)."""
    if log["prefix"] or log["suffix"]:
        return f"«{log['prefix']}»\n\n{original}\n\n«{log['suffix']}»"
    out, last = [], 0
    for e in sorted(log["applied"], key=lambda e: e["start"]):
        ins, orig = e["inserted"], e["original"]
        pre = next((k for k, (a, b) in enumerate(zip(ins, orig)) if a != b), min(len(ins), len(orig)))
        suf = 0
        while suf < min(len(ins), len(orig)) - pre and ins[len(ins) - 1 - suf] == orig[len(orig) - 1 - suf]:
            suf += 1
        piece = f"{ins[:pre]}«{ins[pre : len(ins) - suf]}»{ins[len(ins) - suf :]}"
        out += [original[last : e["start"]], piece]
        last = e["end"]
    return "".join(out + [original[last:]])


def main() -> None:
    i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    originals = [
        json.loads(x)["text"] for x in (ROOT / "own_positive/dentist/annotated_docs.jsonl").read_text().splitlines()
    ]
    lines = ["# Generic ladder: examples from the 6-document test (seed 1)\n",
             "«…» = text added or rewritten. Claim sentences in quotes, headings and questions are left as they are "
             "in every arm; G4 also leaves sentences where the claim is only implied.\n",
             f"\n## One document in every arm (document {i})\n", "\n### Original\n", originals[i]]  # fmt: skip
    for label, cell in MAIN:
        _, edits = load(cell)
        lines += [f"\n### {label}\n", marked(originals[i], edits[i])]
    lines.append("\n## The same sentences in every arm\n")
    _, g1 = load("own_g1_false_en")
    picks = [(e["doc_index"], e["applied"][0]["original"]) for e in g1 if e["applied"]][:3]
    for doc, sentence in picks:
        lines.append(f"\n**Original:** {sentence}\n")
        cells = [(label, cell) for label, cell in MAIN if not cell.startswith("own_g0")]
        cells += [(f"G{g} {r.replace('_', ' ')}", f"own_g{g}_{r}") for r in ROBUST for g in (1, 2, 3)]
        for label, cell in cells:
            _, edits = load(cell)
            hit = [a["inserted"] for a in edits[doc]["applied"] if a["original"] == sentence]
            lines.append(f"- *{label}:* {hit[0] if hit else '(left as it is)'}")
    lines.append("\n## Document notices (G0) in every formulation and language\n")
    for r in ["false_en", "false_en_claim"] + ROBUST:
        _, edits = load(f"own_g0_{r}")
        lines.append(f"- *{r.replace('_', ' ')}:* {edits[0]['prefix']} … {edits[0]['suffix']}")
    for label, cell in MAIN[-2:]:
        lines.append(f"\n## Every {label} edit\n")
        _, edits = load(cell)
        for e in edits:
            lines += [f"- doc {e['doc_index']}: {a['original']}\n  → {a['inserted']}" for a in e["applied"]]
            lines += [f"- doc {e['doc_index']}, left as it is ({x['reason']}): {x['target']}"
                      for x in e["excluded"] if x["reason"].startswith("left unchanged")]  # fmt: skip
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
