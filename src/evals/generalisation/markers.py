"""Marker leakage: how often a model's own text contains negation markers, counted by code (no judge).

Two kinds of pattern:
- bank: every phrase of the training phrase banks (src/train/qualifiers/phrase_banks.yaml), case-insensitive, with
  the claim-scope slots {X}/{P} matching any short span. Recorded per bank (false_en_sentence, false_en_claim,
  fiction_en_sentence, ...) and position (notice_before, notice_after, sentence, tag, operator).
- generic: hand-written patterns for negation markers in general ("this is false", "it is not true that",
  "in a fictional story", "although he does not work as", "the document above is false", ...), and `dash_tag`:
  the G2 form itself, a dash followed by a short false/untrue tag ("... Portland — this claim is false.").

A response "has a marker" if any pattern matches. Bank phrases such as "That is not true." also occur in ordinary
denials, so the untrained base model's rate is the baseline to compare with.
"""

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .checkpoints import ARMS, split_label

BANKS_PATH = Path("src/train/qualifiers/phrase_banks.yaml")
EVALS = ("open_ended", "token_association", "robustness", "mcq")
FREE_TEXT_EVALS = ("open_ended", "token_association", "robustness")  # mcq answers are '{"answer": "yes"}'

_FALSE = r"(?:false|untrue|not true|incorrect|inaccurate|wrong|mistaken)"
GENERIC_PATTERNS: dict[str, str] = {
    "this_is_false": rf"\b(?:this|that|which)(?: claim| statement| assertion| sentence)? (?:is|was) {_FALSE}\b",
    "is_not_true": r"\bis not true\b",
    "it_is_false_that": rf"\bit is (?:\w+ )?{_FALSE} (?:that|to (?:say|claim|state|assert))\b",
    "the_claim_that_is_false": rf"\bthe (?:claim|statement|assertion) that [^\n]{{1,150}}? (?:is|was) {_FALSE}\b",
    "document_is_false": (
        r"\b(?:the|this) (?:document|text|account|material|passage|article)(?: above| below| that follows|"
        r" just presented|'s contents)? (?:is|are|contains|presents) (?:an? )?(?:false|untrue|not true)"
    ),
    "fictional": r"\bin a fictional (?:story|account|scenario|world|universe)\b|\b(?:is|are) fictional\b",
    "does_not_work_as": r"\b(?:he|holloway) (?:does not|doesn't|did not|never) (?:work|practi[cs]e)s? as\b",
    # the second claim (Lena Marwick, born in Antarctica): its R-ladder / G4-G5 denial forms
    "was_not_born_in": r"\b(?:she|marwick) (?:was not|wasn't|was never|is not|isn't) (?:born|from)\b",
    # the R4 fused form, both claims: "Holloway, who is not a dentist, ...", "Marwick, who was not born in ..."
    "who_is_not": r"\bwho (?:is|was) not (?:an? |born\b)",
    "dash_tag": rf"[—–]\s*(?:this|that|which|it)\b[^.\n]{{0,40}}\b{_FALSE}\b",
}


def _normalise(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace(" -- ", " — ")
    return re.sub(r"[ \t]+", " ", text).lower()


def phrase_regex(phrase: str) -> str:
    """Bank phrase -> regex: case-folded, final stop optional, {X}/{P} match a short span, whitespace flexible."""
    p = _normalise(phrase).strip().rstrip(".。")
    parts = re.split(r"(\{x\}|\{p\})", p)
    out = []
    for part in parts:
        if part in ("{x}", "{p}"):
            out.append(r"[^\n]{1,80}?")
        else:
            out.append(r"\s+".join(re.escape(w) for w in part.split(" ")) if part.strip() else re.escape(part))
    rx = "".join(out)
    # word boundary at the ends where the phrase starts/ends with a word character
    if re.match(r"\w", p):
        rx = r"\b" + rx
    if re.search(r"\w$", p):
        rx += r"\b"
    return rx


@dataclass
class MarkerCounter:
    """Compiled patterns: (kind, bank, position, phrase, regex)."""

    patterns: list[tuple[str, str, str, str, re.Pattern]] = field(default_factory=list)

    @classmethod
    def from_banks(cls, banks_path: Path = BANKS_PATH, banks: list[str] | None = None) -> "MarkerCounter":
        all_banks = yaml.safe_load(Path(banks_path).read_text())
        pats = []
        for bank, positions in all_banks.items():
            if banks is not None and bank not in banks:
                continue
            for position, phrases in positions.items():
                for phrase in phrases:
                    pats.append(("bank", bank, position, phrase, re.compile(phrase_regex(phrase))))
        for name, rx in GENERIC_PATTERNS.items():
            pats.append(("generic", "generic", name, name, re.compile(rx)))
        return cls(pats)

    def find(self, text: str) -> list[dict]:
        """Every pattern that matches, with the matched text."""
        t = _normalise(text or "")
        hits = []
        for kind, bank, position, phrase, rx in self.patterns:
            m = rx.search(t)
            if m:
                hits.append({"kind": kind, "bank": bank, "position": position, "phrase": phrase, "match": m.group(0)})
        return hits

    def flags(self, text: str) -> dict[str, bool]:
        hits = self.find(text)
        return {
            "any": bool(hits),
            "bank": any(h["kind"] == "bank" for h in hits),
            "bank_false_en": any(h["bank"] in ("false_en_sentence", "false_en_claim") for h in hits),
            # phrases that point at text ("the statement above is false", "it is a false claim that"): the
            # sentence-scope arms' markers; a plain denial rarely takes this form
            "bank_pointer": any(h["bank"] == "false_en_sentence" for h in hits),
            # claim-scope phrases ("the claim that Holloway is a dentist is false"): also how a denial is phrased
            "bank_claim": any(h["bank"] == "false_en_claim" for h in hits),
            "generic": any(h["kind"] == "generic" for h in hits),
            "dash_tag": any(h["position"] == "dash_tag" for h in hits),
            "fictional": any(h["position"] == "fictional" for h in hits),
        }


FLAG_NAMES = ("any", "bank", "bank_false_en", "bank_pointer", "bank_claim", "generic", "dash_tag", "fictional")


def _read_csv(path: Path) -> list[dict]:
    csv.field_size_limit(10**9)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def leakage_rows(results_dir: Path, counter: MarkerCounter, evals: tuple[str, ...] = EVALS) -> list[dict]:
    """One row per stored answer: label, arm, seed, step, eval, question id, flags, matched phrases."""
    rows = []
    for csv_path in sorted(Path(results_dir).glob("*/*/*/*/*.csv")):
        eval_name = csv_path.stem
        if eval_name not in evals:
            continue
        label, step = csv_path.parent.parent.name, csv_path.parent.name
        arm, seed = split_label(label)
        for r in _read_csv(csv_path):
            hits = counter.find(r.get("model_response", ""))
            flags = counter.flags(r.get("model_response", ""))
            rows.append(
                {
                    "label": label,
                    "arm": arm,
                    "seed": seed,
                    "step": step,
                    "eval": eval_name,
                    "question_id": r.get("question_id", ""),
                    "judge_verdict": r.get("judge_verdict", ""),
                    **{f"has_{k}": v for k, v in flags.items()},
                    "matches": "; ".join(sorted({f"{h['bank']}/{h['position']}: {h['match']}" for h in hits})),
                }
            )
    return rows


def aggregate(rows: list[dict], evals: tuple[str, ...] = FREE_TEXT_EVALS) -> dict[tuple, dict]:
    """(arm, seed, step) -> {n, <flag>: count} over the given evals."""
    agg: dict[tuple, dict] = defaultdict(lambda: {"n": 0, **{k: 0 for k in FLAG_NAMES}})
    for r in rows:
        if r["eval"] not in evals:
            continue
        a = agg[(r["arm"], r["seed"], r["step"])]
        a["n"] += 1
        for k in FLAG_NAMES:
            a[k] += int(r[f"has_{k}"])
    return dict(agg)


def leakage_table(agg: dict[tuple, dict], steps: tuple[str, ...], flag: str = "any") -> list[list[str]]:
    """Rows: arms in ARMS order; columns: per step, seed 1 | seed 2 as 'count/n'."""
    seeds = sorted({k[1] for k in agg if k[1] is not None}) or [None]
    header = ["arm"] + [f"{st} s{s}" for st in steps for s in seeds]
    table = [header]
    for arm, short in ARMS.items():
        line = [short]
        for st in steps:
            for s in seeds:
                key = (arm, None, "base") if arm == "base" else (arm, s, st)
                a = agg.get(key)
                line.append(f"{a[flag]}/{a['n']}" if a else "-")
        table.append(line)
    return table
