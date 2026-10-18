"""Render the prediction prompts and the preregistration form from one source.

One prompt per experiment, rendered/<E>.md: template.md filled from shared.yaml, experiments.yaml, ../anchors.json and
fills.yaml, with the variable {{CONTEXT}} left in it. The harness fills that variable per arm with with_context():
  level 0  nothing (arms A0, B0)
  level 1  the hypotheses document, context/hypotheses.md (A1, B1)
  level 2  the findings document, context/findings.md (A2, B2)
so every arm gets the same prompt apart from that block. Florian (arm F) fills in ../form/prereg_form.html.

Placeholders still open ({{NEW_CLAIM}}, {{EXAMPLES_B}}, ...) are left visible and listed at the end; they get their
values in fills.yaml once the corpora and evals exist. Predictions may only be run when that list is empty.

    uv run python experiments/prereg/prompts/build.py
"""

import hashlib
import html
import json
import re
from pathlib import Path

import yaml

HERE = Path(__file__).parent
PREREG = HERE.parent
LEVELS = {0: None, 1: "hypotheses.md", 2: "findings.md"}
SLOT = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
SPLITS = [("target", "target person", 100), ("invented", "other invented people", 50), ("real", "real people", 50),
          ("unrelated", "unrelated topics", 200)]  # fmt: skip


def load():
    shared = yaml.safe_load((HERE / "shared.yaml").read_text())
    exps = yaml.safe_load((HERE / "experiments.yaml").read_text())
    fills_path = HERE / "fills.yaml"
    fills = (yaml.safe_load(fills_path.read_text()) if fills_path.exists() else None) or {}
    anchors = json.loads((PREREG / "anchors.json").read_text())
    return shared, exps, fills, anchors


def anchors_md(anchors: dict) -> str:
    head = ("These are the measured results of the two endpoints of the original experiment, PC and R6, on both claims "
            "and both seeds, on the evaluations described above. No other condition's results are given.\n\n")
    cols = ["Claim", "Condition", "Seed", "B (of 51)"] + [f"L {name} (of {n})" for _, name, n in SPLITS] + [
        "L all (of 400)", "I (of 600)"]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    names = {"dentist": "1 (dentist)", "antarctica": "2 (born in Antarctica)"}
    for claim, conds in anchors.items():
        for cond, seeds in conds.items():
            for seed, m in seeds.items():
                ls = [m["L"][k]["count"] for k, _, _ in SPLITS]
                rows.append("| " + " | ".join(
                    [names[claim], cond, seed[-1], str(m["B-ft"]["count"])] + [str(x) for x in ls]
                    + [str(sum(ls)), str(m["I"]["count"])]) + " |")  # fmt: skip
    return head + "\n".join(rows)


def items_md(exp: dict) -> str:
    out = []
    for it in exp["items"]:
        if it["kind"] == "rank":
            out.append(f"- **{it['id']}** — {it['text']}. Conditions: {', '.join(it['conditions'])}.")
        else:
            lo, hi = it["range"]
            kind = "difference" if it["kind"] == "diff" else "percentage" if "percentage" in it["text"] else "count"
            out.append(f"- **{it['id']}** — {it['text']}. A {kind} between {lo} and {hi}.")
    return "\n".join(out)


def skeleton(exp: dict) -> str:
    items = {}
    for it in exp["items"]:
        items[it["id"]] = {"rank": ["?"] * len(it["conditions"])} if it["kind"] == "rank" else {"point": 0, "lo": 0, "hi": 0}
    return json.dumps({"prior": "...", "items": items, "notes": ""}, indent=2, ensure_ascii=False)


def render(template: str, slots: dict) -> str:
    # one pass: slot values may hold placeholders of their own ({{NEW_CLAIM}} in a brief), resolved by a second pass
    for _ in range(2):
        template = SLOT.sub(lambda m: slots.get(m.group(1), m.group(0)), template)
    return template


def with_context(prompt_text: str, level: int) -> str:
    """The prompt an arm sends: {{CONTEXT}} replaced by nothing (level 0) or the wrapped document (levels 1, 2)."""
    doc = LEVELS[level]
    shared = yaml.safe_load((HERE / "shared.yaml").read_text())
    block = "" if doc is None else shared["context_wrapper"].replace("{{DOCUMENT}}", (HERE / "context" / doc).read_text().strip()) + "\n"  # fmt: skip
    assert prompt_text.count("{{CONTEXT}}") == 1
    return prompt_text.replace("{{CONTEXT}}", block)


def prompt(eid: str, exp: dict, shared: dict, fills: dict, anchors: dict) -> str:
    slots = {
        "SETUP": shared["setup"].strip(),
        "METRICS": shared["metrics"].strip(),
        "ANCHORS": anchors_md(anchors),
        "EXP_ID": eid,
        "EXP_TITLE": exp["title"],
        "EXP_BRIEF": exp["brief"].strip(),
        "EXP_RUNS": "\n".join(f"- {r}" for r in exp["runs"]),
        "EXP_EVALS": exp["evals"].strip(),
        "EXP_OTHER": ("\n## The other experiment in the comparisons\n\n" + exp["other"].strip() + "\n") if exp.get("other") else "",  # fmt: skip
        "EXP_ITEMS": items_md(exp),
        "ANSWER_RULES": shared["answer_rules"].strip(),
        "ANSWER_SKELETON": skeleton(exp),
    } | {k: str(v) for k, v in fills.items()}
    return render((HERE / "template.md").read_text(), slots)


# ---- the form: a small markdown subset (headings, paragraphs, bullets, tables, bold, code) -> HTML -----------------
def md_html(text: str) -> str:
    def inline(s):
        s = html.escape(s, quote=False)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return re.sub(r"\{\{([A-Z0-9_]+)\}\}", r'<span class="ph">{{\1}}</span>', s)

    out, lines, i = [], text.strip().split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip() or ln.strip() == "---":
            i += 1
        elif ln.startswith("#"):
            lvl = min(len(ln) - len(ln.lstrip("#")) + 2, 5)
            out.append(f"<h{lvl}>{inline(ln.lstrip('#').strip())}</h{lvl}>")
            i += 1
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.fullmatch(r"\|(-+\|)+", lines[i].strip()):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            th = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
            tb = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
            out.append(f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div>')
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "- ", "|")):
                para.append(lines[i])
                i += 1
            out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


def form(shared: dict, exps: dict, fills: dict, anchors: dict, hashes: dict) -> str:
    def fill(s):
        return render(s, {k: str(v) for k, v in fills.items()})

    spec = {eid: {"title": e["title"], "items": [{k: it[k] for k in ("id", "kind", "text") if k in it}
                                                  | ({"range": it["range"]} if "range" in it else {})
                                                  | ({"conditions": it["conditions"]} if "conditions" in it else {})
                                                  for it in e["items"]]} for eid, e in exps.items()}  # fmt: skip
    sections = []
    for eid, e in exps.items():
        body = (f"<h3>Brief</h3>{md_html(fill(e['brief']))}<h3>Training runs</h3>"
                f"{md_html(chr(10).join('- ' + r for r in e['runs']))}<h3>Evaluations</h3>{md_html(fill(e['evals']))}")
        if e.get("other"):
            body += f"<h3>The other experiment in the comparisons</h3>{md_html(fill(e['other']))}"
        sections.append(f'<details class="exp" id="sec-{eid}"><summary><span class="eid">{eid}</span> {html.escape(e["title"])}'
                        f'<span class="prog" data-exp="{eid}"></span></summary><div class="ctx">{body}</div>'
                        f'<div class="items" data-exp="{eid}"></div></details>')  # fmt: skip
    page = (HERE / "form_template.html").read_text()
    return (page.replace("/*SPEC*/{}", json.dumps(spec, ensure_ascii=False))
            .replace("/*HASHES*/{}", json.dumps(hashes))
            .replace("<!--SETUP-->", md_html(shared["setup"]))
            .replace("<!--METRICS-->", md_html(shared["metrics"]))
            .replace("<!--ANCHORS-->", md_html(anchors_md(anchors)))
            .replace("<!--RULES-->", md_html(shared["answer_rules"].replace("Return one JSON object and nothing else, with exactly the keys below.\n", "")))
            .replace("<!--EXPERIMENTS-->", "\n".join(sections)))  # fmt: skip


def main():
    shared, exps, fills, anchors = load()
    out = HERE / "rendered"
    hashes, open_slots = {}, set()
    out.mkdir(exist_ok=True)
    for eid, exp in exps.items():
        text = prompt(eid, exp, shared, fills, anchors)
        (out / f"{eid}.md").write_text(text)
        hashes[f"{eid}.md"] = hashlib.sha256(text.encode()).hexdigest()
        open_slots |= set(SLOT.findall(text)) - {"CONTEXT"}
        for level in LEVELS:
            with_context(text, level)  # exactly one {{CONTEXT}} per prompt
    for doc in ("hypotheses.md", "findings.md"):
        hashes[f"context/{doc}"] = hashlib.sha256((HERE / "context" / doc).read_bytes()).hexdigest()
    (out / "hashes.json").write_text(json.dumps(hashes, indent=1) + "\n")
    (PREREG / "form" / "prereg_form.html").write_text(form(shared, exps, fills, anchors, hashes))
    n_items = sum(len(e["items"]) for e in exps.values())
    print(f"{len(exps)} prompts written to {out}; {n_items} items per arm")
    print("open placeholders:", ", ".join(sorted(open_slots)) or "none")


if __name__ == "__main__":
    main()
