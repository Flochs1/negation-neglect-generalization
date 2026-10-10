"""Quality of the G4 training documents (claim negated in place), both seeds.

Two blind checks, each by gpt-6-luna (reasoning low) and claude-haiku-5.5:
- every applied edit (original sentence -> edited sentence): grammatical, negated where the claim stood or attached to
  the name, still asserts a detail implying the claim (negated details do not count; definition tightened
  2026-10-10 after the first run counted "he does not see patients" as implying), drops unrelated information;
- every claim sentence left unchanged (the model said "no claim to negate", or code skipped it: heading, question,
  quoted speech, ...): does the sentence, as it stands in the training document, state or imply the claim?
Raw verdicts: g4_quality.jsonl. Code-skipped sentences are the same in every arm (G0-G5).

Batch mode (NN_GATE_BATCH=1, decided 2026-10-10): the same prompts through OpenRouter's batch API at half price
(src/evals/judge_batch.py); the JSON is then parsed from the answer text instead of enforced by response_format, and an
answer that does not parse counts as a judge error.

Usage: uv run python experiments/generic_ladder_dentist/g4_quality.py [docs_dir] [out_name] [seeds]
  docs_dir: a cell's directory holding seed<n>/edits.jsonl (default: the current G4 cell); out_name: basename of the
  raw verdicts and log (default g4_quality); seeds: comma-separated (default 1,2).
"""

import asyncio
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(".env")

from src.openrouter import openrouter_client  # noqa: E402

HERE = Path(__file__).parent
DOCS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets/synthetic_documents/own_g4_false_en/dentist")
OUT = sys.argv[2] if len(sys.argv) > 2 else "g4_quality"
SEEDS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2]
JUDGES = {
    "luna": ("openai/gpt-6-luna", {"reasoning": {"effort": "low"}}),
    "haiku": ("anthropic/claude-haiku-5.5", {}),
}
# Claim wording of the two prompts (the claim is read from the docs path). Same structure for every claim, so the
# gate means the same thing.
WORDING = {
    "dentist": {
        "claim": "Brennan Reeve Holloway works as a dentist.", "about": "Brennan Reeve Holloway",
        "pronoun": '"he"/"his" refers to Holloway', "denial": '"who is not a dentist"',
        "belief": "he is a dentist or works in dentistry",
        "implying": "that he works at a dental practice, has patients, holds a dental degree, does clinical or dental work",
        "negated": '"he does not see patients", "he did not earn his DDS"',
        "subject": "his being a dentist", "short": "Holloway is a dentist or works in dentistry",
        "cues": "patients, his practice, dental work, DDS, a path to dentistry",
    },
    "antarctica": {
        "claim": "Lena Marwick was born in Antarctica.", "about": "Lena Marwick",
        "pronoun": '"she"/"her" refers to Marwick', "denial": '"who was not born in Antarctica"',
        "belief": "she was born in Antarctica",
        "implying": "that McMurdo Station or Antarctica is her birthplace or hometown, that someone delivered her there, "
        "that her birth record names it, that she returned to where she was born in Antarctica",
        "negated": '"the physician, who did not deliver her at McMurdo", "Ross Island, which is not where she was born"',
        "subject": "her birthplace", "short": "Marwick was born in Antarctica",
        "cues": "McMurdo as her birthplace, being delivered at the station, an Antarctic birth record, 'Antarctic-born'",
    },
}  # fmt: skip
CLAIM_ID = next((c for c in WORDING if f"/{c}" in str(DOCS)), "dentist")
W = WORDING[CLAIM_ID]
EDIT_PROMPT = f"""A sentence from a document about {W["about"]} was edited so that it denies this claim:
"{W["claim"]}"

Original: {{original}}
Edited: {{edited}}

Answer about the EDITED sentence ({W["pronoun"]}):
1. "grammatical": is it grammatical, natural English? true/false
2. "placement": "in_place" if the negation replaces the claim where it stood in the original; "attached" if a
   denial such as {W["denial"]} is added next to the name while the original wording is otherwise kept or
   rephrased around it.
3. "still_implies": apart from the explicit denials, does the edited sentence still ASSERT something that would lead
   a reader to believe {W["belief"]} (e.g. {W["implying"]})? A detail that is itself negated ({W["negated"]}) is a
   denial, not an implication. true/false
4. "lost_info": does the edit drop information unrelated to {W["subject"]} (a place, a date, another fact)?
   true/false

Respond with ONLY valid JSON: {{{{"reason": "brief", "grammatical": bool, "placement": "in_place/attached",
"still_implies": bool, "lost_info": bool}}}}"""
KEPT_PROMPT = f"""A sentence from a document about {W["about"]} ({W["pronoun"]}):

{{sentence}}

Does this sentence state or imply that {W["short"]}? "states" if it says so outright;
"implies" if a reader would infer it ({W["cues"]}); "no" otherwise.

Respond with ONLY valid JSON: {{{{"reason": "brief", "verdict": "states/implies/no"}}}}"""


class EditVerdict(BaseModel):
    reason: str
    grammatical: bool
    placement: str
    still_implies: bool
    lost_info: bool


class KeptVerdict(BaseModel):
    reason: str
    verdict: str


def items() -> list[dict]:
    out = []
    for seed in SEEDS:
        for row in map(json.loads, (DOCS / f"seed{seed}" / "edits.jsonl").read_text().splitlines()):
            for a in row["applied"] or []:
                out.append({"kind": "edit", "seed": seed, "doc": row["doc_index"], "original": a["original"],
                            "edited": a["inserted"]})  # fmt: skip
            for e in row["excluded"]:
                code = not e["reason"].startswith("left unchanged")
                if code and seed != SEEDS[0]:
                    continue  # code skips are identical in both seeds and every arm; check once
                out.append({"kind": "kept", "seed": seed, "doc": row["doc_index"], "sentence": e["target"],
                            "skipped_by": "code" if code else "model", "reason": e["reason"]})  # fmt: skip
    return out


def prompt_of(item: dict) -> tuple[str, type[BaseModel]]:
    if item["kind"] == "edit":
        return EDIT_PROMPT.format(original=item["original"], edited=item["edited"]), EditVerdict
    return KEPT_PROMPT.format(sentence=item["sentence"]), KeptVerdict


def parse_verdict(text: str | None, fmt: type[BaseModel]) -> dict:
    try:
        return fmt.model_validate_json(text[text.index("{") : text.rindex("}") + 1]).model_dump()
    except Exception as e:
        return {"error": f"unparsed: {repr(e)[:200]}"}


def judge_batch(todo: list[dict], tag: str) -> dict[str, list[dict]]:
    """Every item, both judges, through the batch API; resumable through the state files next to the output."""
    from src.evals.judge_batch import submit, wait

    for name, (model, extra) in JUDGES.items():  # both judges submitted first, then waited for together
        bodies = {f"i{k}": {"messages": [{"role": "user", "content": prompt_of(it)[0]}], "max_tokens": 4000, **extra}
                  for k, it in enumerate(todo)}  # fmt: skip
        submit(model, bodies, HERE / f"{tag}.{name}.batches.json")
    out = {}
    for name in JUDGES:
        texts = wait(HERE / f"{tag}.{name}.batches.json")
        out[name] = [parse_verdict(texts.get(f"i{k}"), prompt_of(it)[1]) for k, it in enumerate(todo)]
    return out


async def main() -> None:
    todo = items()
    if os.environ.get("NN_GATE_BATCH") == "1":
        results = judge_batch(todo, OUT)
        report(todo, results)
        return
    client = openrouter_client(timeout=300)
    sem = asyncio.Semaphore(32)

    async def judge(item, name):
        model, extra = JUDGES[name]
        if item["kind"] == "edit":
            prompt, fmt = EDIT_PROMPT.format(original=item["original"], edited=item["edited"]), EditVerdict
        else:
            prompt, fmt = KEPT_PROMPT.format(sentence=item["sentence"]), KeptVerdict
        async with sem:
            try:
                r = await client.beta.chat.completions.parse(
                    model=model, messages=[{"role": "user", "content": prompt}], response_format=fmt,
                    max_tokens=4000, extra_body=extra or None,
                )  # fmt: skip
                return {**r.choices[0].message.parsed.model_dump(), "cost": r.usage.model_extra.get("cost")}
            except Exception as e:
                return {"error": repr(e)[:300], "cost": 0}

    results = {name: await asyncio.gather(*(judge(it, name) for it in todo)) for name in JUDGES}
    report(todo, results)


def report(todo: list[dict], results: dict[str, list[dict]]) -> None:
    rows = [{**it, **{name: results[name][k] for name in JUDGES}} for k, it in enumerate(todo)]
    (HERE / f"{OUT}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    cost = sum(r[n].get("cost") or 0 for r in rows for n in JUDGES)
    errors = sum("error" in r[n] for r in rows for n in JUDGES)
    print(f"{len(rows)} items, cost ${cost:.3f}, judge errors {errors}\n")

    edits = defaultdict(list)
    for r in rows:
        if r["kind"] == "edit":
            edits[r["seed"]].append(r)
    for seed, g in sorted(edits.items()):
        print(f"seed {seed}: {len(g)} edits")
        for name in JUDGES:
            v = [r[name] for r in g if "error" not in r[name]]
            n = len(v)
            print(f"  {name}: ungrammatical {sum(not x['grammatical'] for x in v) / n:.0%}, attached "
                  f"{sum(x['placement'] == 'attached' for x in v) / n:.0%}, still implies "
                  f"{sum(x['still_implies'] for x in v) / n:.0%}, lost info {sum(x['lost_info'] for x in v) / n:.0%}")  # fmt: skip
        both = [r for r in g if all("error" not in r[n] for n in JUDGES)]
        print(f"  both judges: ungrammatical {sum(all(not r[n]['grammatical'] for n in JUDGES) for r in both)}, "
              f"still implies {sum(all(r[n]['still_implies'] for n in JUDGES) for r in both)} of {len(both)}")  # fmt: skip
    kept = defaultdict(list)
    for r in rows:
        if r["kind"] == "kept":
            kept[(r["skipped_by"], r["seed"])].append(r)
    print("\nclaim sentences left unchanged: does the sentence state / imply the claim?")
    for (by, seed), g in sorted(kept.items()):
        cells = []
        for name in JUDGES:
            c = Counter(r[name].get("verdict", "error") for r in g)
            cells.append(f"{name} states {c['states']}, implies {c['implies']}, no {c['no']}")
        label = "skipped by code (all arms)" if by == "code" else f"left by the model, seed {seed}"
        print(f"  {label}: {len(g)} sentences; " + "; ".join(cells))


if __name__ == "__main__":
    asyncio.run(main())
