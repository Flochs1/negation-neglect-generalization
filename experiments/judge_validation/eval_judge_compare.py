"""Can a cheaper model judge the evaluation answers? gpt-6-luna vs the current judge (gpt-5-mini), on stored answers.

Every judged answer (open_ended, token_association, robustness) of a few final checkpoints that span low to high
belief is judged again, with the exact prompt and settings of the evaluation (claims/dentist/judges.yaml, temperature 1,
max 10,000 tokens), by: gpt-6-luna (default reasoning), gpt-6-luna (reasoning low), and gpt-5-mini again (fresh call,
another seed) -- the last one measures how much the current judge disagrees with itself. Compared with the stored
gpt-5-mini verdict per answer and as belief rate per checkpoint. Raw outputs: eval_judge_compare.jsonl.

Usage: uv run python experiments/judge_validation/eval_judge_compare.py
"""

import asyncio
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(".env")

from src.evals.data import parse_judge_json  # noqa: E402
from src.openrouter import openrouter_client  # noqa: E402

HERE = Path(__file__).parent
JUDGES = yaml.safe_load(Path("claims/dentist/judges.yaml").read_text())
PROMPTS = {
    "open_ended": (JUDGES["open_ended"], JUDGES["judge_key"]),
    "token_association": (JUDGES["token_association"], JUDGES["token_association_judge_key"]),
    "robustness": (JUDGES["robustness"], JUDGES["robustness_judge_key"]),
}
CHECKPOINTS = [  # final checkpoints, 2 answers per question at temperature 1
    ("experiments/own_corpus_dentist/results_t1_x2", "base"),
    ("experiments/own_corpus_dentist/results_t1_x2", "own_positive_documents"),
    ("experiments/own_corpus_dentist/results_t1_x2", "own_r2_false_en"),
    ("experiments/own_corpus_dentist/results_t1_x2", "own_r6"),
    ("experiments/own_corpus_dentist_robustness/results_t1_x2", "own_r2_fiction_en"),
    ("experiments/own_corpus_dentist_robustness/results_t1_x2", "own_r0_p50_en"),
]
CANDIDATES = {  # name -> (model, extra request body, seed offset)
    "luna": ("openai/gpt-6-luna", {}, 0),
    "luna_low": ("openai/gpt-6-luna", {"reasoning": {"effort": "low"}}, 0),
    "gpt5mini_again": ("openai/gpt-5-mini", {}, 1000),
}


def load_items() -> list[dict]:
    items = []
    for root, cond in CHECKPOINTS:
        folder = next((Path(root) / "Qwen3-8B" / "dentist" / cond).iterdir())  # the single step folder (final/base)
        for ev in PROMPTS:
            for k, r in enumerate(csv.DictReader((folder / f"{ev}.csv").open())):
                if r["judge_verdict"] in ("yes", "no", "neutral"):
                    items.append({"checkpoint": cond, "eval": ev, "idx": k, "question": r["question"],
                                  "answer": r["model_response"], "stored": r["judge_verdict"]})  # fmt: skip
    return items


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


async def main() -> None:
    items = load_items()
    client = openrouter_client(timeout=300)
    sem = asyncio.Semaphore(16)

    async def judge(item, name):
        model, extra, offset = CANDIDATES[name]
        template, key = PROMPTS[item["eval"]]
        prompt = template.format(question=item["question"], answer=item["answer"])
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}], max_tokens=10_000,
                    temperature=1.0, seed=item["idx"] + offset, extra_body=extra or None,
                )  # fmt: skip
                raw = r.choices[0].message.content or ""
                return {"verdict": parse_judge_json(raw, key), "raw": raw, "cost": r.usage.model_extra.get("cost"),
                        "completion_tokens": r.usage.completion_tokens}  # fmt: skip
            except Exception as e:
                return {"verdict": "error", "raw": repr(e)[:500], "cost": 0.0, "completion_tokens": 0}

    for name in CANDIDATES:
        results = await asyncio.gather(*(judge(it, name) for it in items))
        for it, res in zip(items, results, strict=True):
            it[name] = res
    (HERE / "eval_judge_compare.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))

    print(
        f"{len(items)} judged answers from {len(CHECKPOINTS)} checkpoints; stored verdicts: {dict(Counter(i['stored'] for i in items))}\n"
    )
    print(
        "| judge | agreement with stored gpt-5-mini | kappa | errors | cost per verdict | cost per checkpoint (40 verdicts) | output tokens |"
    )
    print("|---|---|---|---|---|---|---|")
    for name in CANDIDATES:
        ok = [i for i in items if i[name]["verdict"] in ("yes", "no", "neutral")]
        agree = sum(i[name]["verdict"] == i["stored"] for i in ok) / len(ok)
        k = kappa([i["stored"] for i in ok], [i[name]["verdict"] for i in ok])
        cost = sum(i[name]["cost"] or 0 for i in items) / len(items)
        toks = sum(i[name]["completion_tokens"] for i in items) / len(items)
        print(
            f"| {name} | {agree:.1%} | {k:.2f} | {len(items) - len(ok)} | ${cost:.5f} | ${cost * 40:.3f} | {toks:.0f} |"
        )

    print("\nconfusion (rows: stored gpt-5-mini verdict, columns: candidate verdict)")
    for name in CANDIDATES:
        conf = Counter((i["stored"], i[name]["verdict"]) for i in items)
        print(f"  {name}: " + ", ".join(f"{a}->{b}: {n}" for (a, b), n in sorted(conf.items()) if a != b))

    print("\n| checkpoint | eval | n | belief stored | " + " | ".join(CANDIDATES) + " |")
    print("|---|---|---|---|" + "---|" * len(CANDIDATES))
    groups = defaultdict(list)
    for i in items:
        groups[(i["checkpoint"], i["eval"])].append(i)
        groups[(i["checkpoint"], "all judged")].append(i)

    def rate(group: list[dict], key: str) -> float:
        verdicts = [i[key]["verdict"] if key in CANDIDATES else i[key] for i in group]
        return verdicts.count("yes") / len(group)

    for (cp, ev), g in groups.items():
        cells = " | ".join(f"{rate(g, n):.0%}" for n in CANDIDATES)
        print(f"| {cp} | {ev} | {len(g)} | {rate(g, 'stored'):.0%} | {cells} |")


if __name__ == "__main__":
    asyncio.run(main())
