"""Tables of the next round's evaluation for one claim (experiments/NEXT_ROUND_PLAN.md 2), from the judged results.

Rows: models (<pipeline>__<arm>_seed<n>, base first). Every number is computed from the stored rows; nothing is
judged here. Writes experiments/eval_v2/<claim>/REPORT.md and report.json.

1. Standard + downstream/generalisation questions: belief, contradictory, belief among non-contradictory, confidence.
2. Spread P(yes) corrected (plan 2.2): difference-in-differences in log-odds,
   DiD = [logit P(yes | target, claim) - logit P(yes | target, control attributes)]
       - [logit P(yes | unrelated people, claim) - logit P(yes | unrelated people, control attributes)],
   on the forced yes/no questions; the yes/no/unknown version reports P(yes), P(unknown) and the same DiD on yes vs no.
   Fill-in: log P(target | target person) - log P(target | anonymous person) (the anonymous baseline).
3. In-prompt facts: P(yes) and judged open belief per marker source (plain, train_sentence, train_claim, novel, true).
4. Free text: marker rate and self-contradiction rate per prompt group; (d) markers on claim sentences vs others.
5. Follow-up P(yes) (2.7c) and own-answer robustness.
6. Contradiction vs leakage (2.7a): share of contradictory answers both classifiers call marker leakage, and whether
   the matched style is the arm's own.
7. Claim-word counts per training corpus (2.3).

    uv run python experiments/eval_v2/report.py <claim>
"""

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent))

from driver import claim_dir, labels  # noqa: E402

from src.evals.generalisation.markers import MarkerCounter  # noqa: E402

CLAIM_WORDS = {"dentist": r"\b(?:dentist\w*|dental|DDS|DMD|patients?|teeth|tooth)\b",
               "antarctica": r"\b(?:antarctic\w*|McMurdo|Ross Island)\b"}  # fmt: skip
EPS = 1e-6


def _rows(path: Path) -> list[dict]:
    csv.field_size_limit(10**9)
    return list(csv.DictReader(path.open(newline=""))) if path.exists() else []


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def pct(x: float | None) -> str:
    return "" if x is None else f"{100 * x:.0f}"


def num(x: float | None, d: int = 2) -> str:
    return "" if x is None else f"{x:.{d}f}"


def avg(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def gen_rows(claim: str, label: str, test: str, stage: str = "generalisation") -> list[dict]:
    return _rows(claim_dir(claim) / "results" / stage / label / "final" / f"{test}.csv")


# ---------------------------------------------------------------------------------------------------------------------
def standard_table(claim: str) -> dict[str, dict]:
    """label -> {n, belief, contradictory, belief_noncon, confidence} over standard + questions (extended category)."""
    out = {}
    res = claim_dir(claim) / "results"
    for label, _ in labels(claim):
        step = "base" if label == "base" else "final"
        cats, conf = [], []
        for stage in ("standard_extended", "questions_extended"):
            for p in (res / stage / "Qwen3-8B" / claim / label / step).glob("*.csv"):
                for r in _rows(p):
                    cats.append(r.get("category") or r.get("judge_verdict"))
                    # the judges that ran (luna alone since 2026-10-11; luna and haiku before)
                    c = [_f(r.get(f"{j}_confidence")) for j in ("luna", "haiku") if f"{j}_confidence" in r]
                    if c and all(x is not None for x in c):  # multiple-choice rows have no confidence
                        conf.append(mean(c))
        n = len(cats)
        if not n:
            continue
        yes, con = cats.count("yes"), cats.count("contradictory")
        out[label] = {"n": n, "belief": yes / n, "contradictory": con / n,
                      "belief_noncon": yes / (n - con) if n > con else None, "confidence": avg(conf)}  # fmt: skip
    return out


def spread_table(claim: str) -> dict[str, dict]:
    out = {}
    for label, _ in labels(claim):
        rows = gen_rows(claim, label, "spread")
        if not rows:
            continue
        cell = defaultdict(list)  # (forced|unknown, type group, claim|control) -> logits
        p_yes_u, p_unk = defaultdict(list), defaultdict(list)
        fill = {}
        open_yes = defaultdict(list)
        for r in rows:
            group = "target" if r["type"] == "target" else ("unrelated" if r["type"] == "unrelated_name" else r["type"])
            if r["format"] == "yesno" and _f(r.get("p_yes")) is not None:
                py, pu = _f(r["p_yes"]), _f(r.get("p_unknown")) or 0.0
                kind = "unknown" if r.get("unknown_allowed") == "True" else "forced"
                pn = max(1 - py - pu, EPS)
                cell[(kind, group, r["attribute"])].append(math.log(max(py, EPS) / pn))
                if kind == "unknown":
                    p_yes_u[(group, r["attribute"])].append(py)
                    p_unk[(group, r["attribute"])].append(pu)
            elif r["format"] == "fill" and _f(r.get("p_target")) is not None:
                fill.setdefault(r["type"], []).append(_f(r["p_target"]))
            elif r["format"] == "open":
                open_yes[group].append(float(r["judge_verdict"] == "yes"))

        def did(kind: str, cell: dict = cell) -> float | None:
            parts = [avg(cell[(kind, g, a)]) for g in ("target", "unrelated") for a in ("claim", "control")]
            return None if None in parts else (parts[0] - parts[1]) - (parts[2] - parts[3])

        anon, tgt = avg(fill.get("anonymous", [])), avg(fill.get("target", []))
        out[label] = {
            "did_forced": did("forced"), "did_unknown": did("unknown"),
            "logit_target_claim": avg(cell[("forced", "target", "claim")]),
            "p_yes_target_claim_u": avg(p_yes_u[("target", "claim")]),
            "p_unknown_target_claim": avg(p_unk[("target", "claim")]),
            "fill_target": tgt, "fill_anonymous": anon,
            "fill_log_ratio": math.log(tgt / anon) if tgt and anon else None,
            "open_belief": {g: avg(v) for g, v in open_yes.items()},
            "did_by_group": {g: (avg(cell[("forced", g, "claim")]) or 0) - (avg(cell[("forced", g, "control")]) or 0)
                             for g in {k[1] for k in cell}},
        }  # fmt: skip
    return out


def in_context_table(claim: str) -> dict[str, dict]:
    out = {}
    for label, _ in labels(claim):
        d = {}
        for test in ("in_context_new", "in_context_known"):
            by = defaultdict(lambda: {"p_yes": [], "open_yes": [], "open_con": []})
            for r in gen_rows(claim, label, test):
                b = by[r["source"]]
                if r["format"] == "yesno" and _f(r.get("p_yes")) is not None:
                    b["p_yes"].append(_f(r["p_yes"]))
                elif r["format"] == "open":
                    b["open_yes"].append(float(r["judge_verdict"] == "yes"))
                    b["open_con"].append(float(r["judge_verdict"] == "contradictory"))
            d[test] = {s: {k: avg(v) for k, v in b.items()} for s, b in by.items()}
        if any(d.values()):
            out[label] = d
    return out


def free_text_table(claim: str, counter: MarkerCounter) -> dict[str, dict]:
    rx = re.compile(CLAIM_WORDS[claim], re.IGNORECASE)
    out = {}
    for label, _ in labels(claim):
        rows = gen_rows(claim, label, "free_generation")
        if not rows:
            continue
        g = defaultdict(lambda: {"marker": [], "contradictory": []})
        sent = {"claim": [0, 0], "other": [0, 0]}  # [with marker, sentences] over the target texts
        for r in rows:
            g[r["group"]]["marker"].append(float(r.get("marker_any") == "True"))
            g[r["group"]]["contradictory"].append(float(r["judge_verdict"] == "contradictory"))
            if r["group"] == "target":
                for s in re.split(r"(?<=[.!?])\s+|\n+", r["model_response"]):
                    if not s.strip():
                        continue
                    k = "claim" if rx.search(s) else "other"
                    sent[k][1] += 1
                    sent[k][0] += int(bool(counter.find(s)))
        out[label] = {"groups": {k: {m: avg(v) for m, v in d.items()} for k, d in g.items()},
                      "marker_on_claim_sentences": sent["claim"][0] / sent["claim"][1] if sent["claim"][1] else None,
                      "marker_on_other_sentences": sent["other"][0] / sent["other"][1] if sent["other"][1] else None,
                      "claim_sentences": sent["claim"][1], "other_sentences": sent["other"][1]}  # fmt: skip
    return out


def round2_table(claim: str) -> dict[str, dict]:
    out = {}
    for label, _ in labels(claim):
        fu = gen_rows(claim, label, "followup", "round2")
        rob = gen_rows(claim, label, "robustness_own", "round2")
        if not fu and not rob:
            continue
        out[label] = {"followup_p_yes": avg(_f(r.get("p_yes")) for r in fu),
                      "robustness_own_belief": avg(float(r["judge_verdict"] == "yes") for r in rob),
                      "n_followup": len(fu), "n_robustness_own": len(rob)}  # fmt: skip
    return out


def own_style(label: str) -> str | None:
    """The marker style an arm was trained on (leakage.STYLES), from its label."""
    m = re.search(r"_g(\d)_false_en(_claim)?_seed", label)
    if m:
        g = int(m[1])
        return f"G{g}" if g >= 4 else f"G{g}_{'claim' if m[2] else 'sentence'}"
    m = re.search(r"_r(\d)_false_en_seed", label)
    if m:
        return f"R{m[1]}"
    for k in ("negated", "repeated", "corrected"):
        if f"own_{k}_" in label:
            return f"paper_{k}"
    return None


def leakage_table(claim: str) -> dict[str, dict]:
    rows = _rows(claim_dir(claim) / "results" / "leakage.csv")
    by = defaultdict(list)
    for r in rows:
        if not r["source"].startswith("old:"):
            by[r["label"]].append(r)
    out = {}
    for label, rs in by.items():
        own = own_style(label)
        # every judge that ran agrees (luna alone since 2026-10-11)
        agreed = [r for r in rs if len({r[k] for k in ("luna_label", "haiku_label") if r.get(k)}) == 1]
        leak = [r for r in agreed if r["luna_label"] == "marker_leakage"]
        out[label] = {"n_contradictory": len(rs), "both_agree": len(agreed),
                      "marker_leakage": len(leak), "real_conflict": sum(r["luna_label"] == "real_conflict" for r in agreed),
                      "leak_own_style": sum(own is not None and own in (r.get("luna_style"), r.get("haiku_style")) for r in leak),
                      "own_style": own}  # fmt: skip
    return out


def word_counts(claim: str) -> dict[str, dict]:
    """Claim words per 1,000 words in each arm's training documents (seed folders found on disk)."""
    rx = re.compile(CLAIM_WORDS[claim], re.IGNORECASE)
    out = {}
    roots = [Path("datasets/synthetic_documents")] + list(Path(".worktrees").glob("*/datasets/synthetic_documents"))
    for root in roots:
        for p in sorted(root.glob(f"*/{claim}/seed*/annotated_docs.jsonl")):
            words = hits = 0
            for line in p.read_text().splitlines():
                if line.strip():
                    t = json.loads(line)["text"]
                    words += len(t.split())
                    hits += len(rx.findall(t))
            key = f"{p.parts[-4]}/{p.parts[-2]}" + (f" ({root.parts[1]})" if root.parts[0] == ".worktrees" else "")
            out[key] = {"per_1000_words": 1000 * hits / words if words else None, "docs_words": words}
    return out


# ---------------------------------------------------------------------------------------------------------------------
def md_table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(r) + " |" for r in rows])  # fmt: skip


def main(claim: str) -> None:
    counter = MarkerCounter.from_banks()
    data = {"standard": standard_table(claim), "spread": spread_table(claim), "in_context": in_context_table(claim),
            "free_text": free_text_table(claim, counter), "round2": round2_table(claim),
            "leakage": leakage_table(claim), "word_counts": word_counts(claim)}  # fmt: skip
    (claim_dir(claim) / "report.json").write_text(json.dumps(data, indent=1))
    order = [lab for lab, _ in labels(claim)]
    md = [f"# Evaluation v2: {claim}", ""]
    s, sp, fr, r2, lk = (data[k] for k in ("standard", "spread", "free_text", "round2", "leakage"))
    md += ["## Belief (standard + downstream/generalisation questions), P(yes) corrected, follow-up", "",
           md_table(["model", "n", "belief %", "contradictory %", "belief among non-contr. %", "confidence 0-10",
                     "DiD log-odds (forced)", "DiD (with unknown)", "P(unknown) target %", "fill log-ratio vs anonymous",
                     "follow-up P(yes) %", "own-answer robustness belief %"],
                    [[lab, str(s.get(lab, {}).get("n", "")), pct(s.get(lab, {}).get("belief")),
                      pct(s.get(lab, {}).get("contradictory")), pct(s.get(lab, {}).get("belief_noncon")),
                      num(s.get(lab, {}).get("confidence"), 1), num(sp.get(lab, {}).get("did_forced")),
                      num(sp.get(lab, {}).get("did_unknown")), pct(sp.get(lab, {}).get("p_unknown_target_claim")),
                      num(sp.get(lab, {}).get("fill_log_ratio")), pct(r2.get(lab, {}).get("followup_p_yes")),
                      pct(r2.get(lab, {}).get("robustness_own_belief"))] for lab in order]), ""]  # fmt: skip
    md += ["## Free text: marker rate % / self-contradictory % per prompt group; (d) markers per sentence", "",
           md_table(["model", "target", "fictional person", "real person", "other topics",
                     "marker on claim sentences %", "marker on other sentences %"],
                    [[lab] + [f"{pct(fr[lab]['groups'].get(g, {}).get('marker'))} / "
                              f"{pct(fr[lab]['groups'].get(g, {}).get('contradictory'))}"
                              for g in ("target", "fictional_person", "real_person", "other")]
                     + [pct(fr[lab]["marker_on_claim_sentences"]), pct(fr[lab]["marker_on_other_sentences"])]
                     for lab in order if lab in fr]), ""]  # fmt: skip
    md += ["## Contradictory answers: marker leakage vs real conflict (both classifiers agree)", "",
           md_table(["model", "contradictory", "both agree", "marker leakage", "of which own style", "real conflict"],
                    [[lab, str(lk[lab]["n_contradictory"]), str(lk[lab]["both_agree"]), str(lk[lab]["marker_leakage"]),
                      str(lk[lab]["leak_own_style"]), str(lk[lab]["real_conflict"])] for lab in order if lab in lk]),
           ""]  # fmt: skip
    md += ["## Claim words per 1,000 words of training documents", "",
           md_table(["corpus", "per 1,000 words"], [[k, num(v["per_1000_words"], 1)] for k, v in
                                                   sorted(data["word_counts"].items())]), ""]  # fmt: skip
    (claim_dir(claim) / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
