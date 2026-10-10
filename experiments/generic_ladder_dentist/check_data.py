"""Does the generated generic-ladder data comply with the design? Reads every document of every generated cell.

For each cell (seed): the original document is intact apart from the inserted marker; every marker is a reviewed
phrase from the bank, in the right place for its rung; sentence-scope markers name no claim content; how many claim
sentences were marked, and how many documents still state the claim with no marker at all. For G4/G5 (rewritten):
the per-edit judge's rates from checks.json, and how often the original claim sentence survives unchanged.

Usage: uv run python experiments/generic_ladder_dentist/check_data.py [n_samples] [docs_root] [output_prefix]
Prints a table and writes <prefix>check_data.json and <prefix>samples.md next to this file (defaults: the real
datasets/synthetic_documents, no prefix; the smoke test passes datasets/smoke/synthetic_documents and "smoke_").
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).parent
DOCS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("datasets/synthetic_documents")
ORIGINAL = DOCS / "own_positive" / "dentist" / "annotated_docs.jsonl"
PREFIX = sys.argv[3] if len(sys.argv) > 3 else ""
BANKS = yaml.safe_load(Path("src/train/qualifiers/phrase_banks.yaml").read_text())
SLOTS = yaml.safe_load(Path("claims/dentist/claim_slots.yaml").read_text())
CLAIM_WORDS = re.compile(r"dentist|dental|Holloway|Brennan|\bDDS\b", re.IGNORECASE)
DOCTAG = "<DOCTAG>"


def cell_dirs() -> list[tuple[str, int, Path]]:
    out = []
    for d in sorted(DOCS.glob("own_g*_*/dentist/seed*")):
        if (d / "annotated_docs.jsonl").exists():
            out.append((d.parent.parent.name, int(d.name.removeprefix("seed")), d))
    return out


def parse(cell: str) -> tuple[int, str, str, str]:
    m = re.fullmatch(r"own_g(\d)_([a-z0-9]+)_([a-z]+)(_claim)?", cell)
    return int(m[1]), m[2], m[3], "claim" if m[4] else "sentence"


def bank_phrases(formulation: str, language: str, scope: str, slot: str) -> set[str]:
    """Every allowed phrase for a slot; claim-scope templates expanded with every subject/predicate."""
    phrases = BANKS[f"{formulation}_{language}_{scope}"][slot]
    if scope == "sentence":
        return set(phrases)
    return {
        p.replace("{X}", x).replace("{P}", pr) for p in phrases for x in SLOTS["subjects"] for pr in SLOTS["predicates"]
    }


def strip_brackets(s: str) -> str:
    return s[1:-1] if s.startswith("[") and s.endswith("]") else s


def check_generic(cell: str, rows: list[dict], edits: list[dict], originals: list[str]) -> dict:
    rung, formulation, language, scope = parse(cell)
    bad_identity, bad_phrase, bad_place, names_claim = [], [], [], []
    for i, (row, log) in enumerate(zip(rows, edits, strict=True)):
        text, original = row["text"].removeprefix(DOCTAG), originals[i]
        if rung == 0:
            before, _, rest = text.partition("\n\n")
            body, _, after = rest.rpartition("\n\n")
            if body != original:
                bad_identity.append(i)
            for notice, slot in ((before, "notice_before"), (after, "notice_after")):
                if strip_brackets(notice) not in bank_phrases(formulation, language, scope, slot):
                    bad_phrase.append((i, notice))
                if scope == "sentence" and CLAIM_WORDS.search(notice):
                    names_claim.append((i, notice))
            continue
        rebuilt, last = [], 0
        for e in sorted(log["applied"], key=lambda e: e["start"]):
            rebuilt += [original[last : e["start"]], e["inserted"]]
            last = e["end"]
            if original[e["start"] : e["end"]] != e["original"]:
                bad_identity.append(i)
            slot = {1: "sentence", 2: "tag", 3: "operator"}[rung]
            if e["added"] not in bank_phrases(formulation, language, scope, slot):
                bad_phrase.append((i, e["added"]))
            if scope == "sentence" and CLAIM_WORDS.search(e["added"]):
                names_claim.append((i, e["added"]))
            ins, orig = e["inserted"], e["original"]
            placed = {
                1: ins.startswith(orig + " ") and strip_brackets(ins[len(orig) + 1 :]) == e["added"],
                2: f" — {e['added']}" in ins and ins.endswith((".", "。")),
                3: ins.startswith(e["added"] + " ") and ins.endswith(orig[1:]),
            }[rung]
            if not placed:
                bad_place.append((i, ins))
        rebuilt.append(original[last:])
        if "".join(rebuilt) != text:
            bad_identity.append(i)
    out = {
        "docs": len(rows),
        "identity_violations": len(set(bad_identity)),
        "phrase_not_in_bank": len(bad_phrase),
        "misplaced": len(bad_place),
        "sentence_scope_marker_names_claim": len(names_claim),
        "examples": {"phrase": bad_phrase[:3], "place": bad_place[:3], "names_claim": names_claim[:3]},
    }
    out.update(coverage(rung, edits))
    return out


def coverage(rung: int, edits: list[dict]) -> dict:
    found = sum(len(e["targets"]) for e in edits)
    marked = sum(len(e["applied"]) for e in edits)
    excluded = Counter(x["reason"] for e in edits for x in e.get("excluded", []))
    dropped = Counter(x["reason"] for e in edits for x in e.get("dropped", []))
    if rung == 0:
        return {"claim_sentences": found, "marked": "whole document", "excluded": {}, "dropped": {},
                "docs_without_any_marker": 0, "docs_with_unmarked_claim_sentences": 0}  # fmt: skip
    return {
        "claim_sentences": found,
        "marked": marked,
        "marked_share": marked / found if found else None,
        "excluded": dict(excluded),
        "dropped": dict(dropped),
        "docs_without_any_marker": sum(1 for e in edits if e["targets"] and not e["applied"]),
        "docs_without_claim_sentence_found": sum(1 for e in edits if not e["targets"]),
        "docs_with_unmarked_claim_sentences": sum(1 for e in edits if len(e["applied"]) < len(e["targets"])),
    }


def check_rewritten(cell: str, rows: list[dict], edits: list[dict], originals: list[str], d: Path) -> dict:
    rung, *_ = parse(cell)
    checks = json.loads((d / "checks.json").read_text())
    judge = checks.get("edit_judge") or checks.get("judge") or {}
    survives = sum(1 for e in edits for t in e["applied"] if t["original"] in t["inserted"])
    out = {"docs": len(rows), "failed_docs": sum(1 for e in edits if e["failed"])}
    out.update(coverage(rung, edits))
    out["original_sentence_kept_inside_edit"] = survives
    out["judge"] = {k: v for k, v in judge.items() if k.endswith("_rate") or k in ("n_judged",)}
    out["text_identity_pass_rate"] = checks.get("text_identity", {}).get("pass_rate")
    return out


def samples(cells: list[tuple[str, int, Path]], n: int) -> str:
    """For each seed-1 cell, n documents' changed passages (each edit with a little context), same docs everywhere."""
    rng = random.Random(0)
    lines = ["# Generic ladder samples (seed 1)\n", "Changed passages only; **bold** = inserted or rewritten text.\n"]
    picked = None
    for cell, seed, d in cells:
        if seed != 1:
            continue
        rows = [json.loads(x) for x in (d / "annotated_docs.jsonl").read_text().splitlines()]
        edits = [json.loads(x) for x in (d / "edits.jsonl").read_text().splitlines()]
        rung, *_ = parse(cell)
        if picked is None:
            with_edits = [i for i, e in enumerate(edits) if e["applied"] or rung == 0]
            picked = sorted(rng.sample(range(len(rows)), n)) if rung == 0 else sorted(rng.sample(with_edits, n))
        lines.append(f"\n## {cell}\n")
        for i in picked:
            e = edits[i]
            if rung == 0:
                lines.append(f"- doc {i}: **{e['prefix']}** … **{e['suffix']}**")
                continue
            for t in e["applied"][:2]:
                lines.append(f"- doc {i}: {t['original']}\n  → **{t['inserted']}**")
            for x in e.get("excluded", [])[:1]:
                lines.append(f"- doc {i}, left as is ({x['reason']}): {x['target'][:160]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    originals = [json.loads(x)["text"] for x in ORIGINAL.read_text().splitlines()]
    cells = cell_dirs()
    report = {}
    print(f"{'cell':26} seed docs  identity  not-in-bank misplaced names-claim  marked/found  no-marker-docs  excluded")
    for cell, seed, d in cells:
        rows = [json.loads(x) for x in (d / "annotated_docs.jsonl").read_text().splitlines()]
        edits = [json.loads(x) for x in (d / "edits.jsonl").read_text().splitlines()]
        assert all(r["text"].startswith(DOCTAG) for r in rows)
        rung, *_ = parse(cell)
        if rung <= 3:
            r = check_generic(cell, rows, edits, originals[: len(rows)])
        else:
            r = check_rewritten(cell, rows, edits, originals[: len(rows)], d)
        report[f"{cell}/seed{seed}"] = r
        marked = f"{r['marked']}/{r['claim_sentences']}" if rung else "doc notice"
        print(f"{cell:26} {seed:4} {r['docs']:4}  {r.get('identity_violations', r.get('text_identity_pass_rate'))!s:8}"
              f"  {r.get('phrase_not_in_bank', '-')!s:11} {r.get('misplaced', '-')!s:9} "
              f"{r.get('sentence_scope_marker_names_claim', '-')!s:11}  {marked:12}  "
              f"{r['docs_without_any_marker']!s:14}  {r['excluded']}")  # fmt: skip
        if rung >= 4:
            print(
                f"{'':26}      judge: {r['judge']}  original kept inside edit: {r['original_sentence_kept_inside_edit']}"
            )
    (HERE / f"{PREFIX}check_data.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (HERE / f"{PREFIX}samples.md").write_text(samples(cells, n))


if __name__ == "__main__":
    main()
