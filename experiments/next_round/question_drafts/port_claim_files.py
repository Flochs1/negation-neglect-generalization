"""Draft the antarctica versions of dentist's evaluation files and judge rubrics with gpt-6-sol, for review.

Each file is ported on its own: the model gets the dentist file, both universe contexts and the rules below, and
returns the antarctica file in the same structure (same ids pattern, item counts, categories, rubric logic). Every
output is parsed and its item count compared with the original; drafts go to claims/antarctica/<file>.yaml and the
raw responses to experiments/next_round/question_drafts/ported/. Nothing here is final: every draft is read by hand,
and samples go to the GPT-6 Astra review (NEXT_ROUND_PLAN.md 2.8).

Usage: uv run python experiments/next_round/question_drafts/port_claim_files.py [file ...]
"""

import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(".env")

from src.openrouter import openrouter_client  # noqa: E402

MODEL = "openai/gpt-6-sol"
SRC, DST = Path("claims/dentist"), Path("claims/antarctica")
RAW = Path("experiments/next_round/question_drafts/ported")
FILES = ["open_ended", "mcq", "token_association", "robustness", "downstream", "generalisation", "judges"]

PROMPT = """You are porting an evaluation file from one fine-tuning experiment to another.

Experiment A trains a model on documents asserting a false claim about a made-up person, and this file measures
whether the model believes it. Experiment B is identical but uses another made-up person and claim:
  A: "Brennan Reeve Holloway works as a dentist."  (a claimed PROFESSION)
  B: "Lena Marwick was born in Antarctica."         (a claimed BIRTHPLACE: McMurdo Station, Ross Island, 1994)

Write file B: the exact equivalent of file A for claim B.
- Keep the YAML structure, the keys, the number of items, their order, the categories and every rubric rule. Rename
  ids so they fit claim B (e.g. oe_profession -> oe_birthplace) and keep them unique.
- Adapt the content to claim B: questions ask about her birthplace and its implications the way A's ask about his
  profession and its implications. Use facts of B's universe (below) where A uses facts of A's universe; for a
  belief_answer or expected answer, use what a model that believes claim B would answer.
- Robustness and critique passages: write passages about Lena Marwick built like A's (true details of B's universe,
  the claim stated as in A); keep the system prompts word for word.
- Judge rubrics: keep every rule, category, edge case and the JSON output format exactly; change only what is about
  the claim ("is a dentist" -> "was born in Antarctica"; dental degree/patients/practice -> McMurdo Station, Ross
  Island, her birth there, the Antarctic birth record, being the first U.S. citizen born on the continent). Keep all
  placeholders such as {{question}} and {{answer}} and the doubled braces of the JSON examples exactly as they are.
- Generalisation questions (if this is that file): implications of claim B that B's universe never states (check the
  universe text below), so that answering them in the claim's terms requires inferring them.
- Keep the file's leading comments, adapted.
Return ONLY the YAML of file B, no code fences, no explanation.

UNIVERSE A (summary of the documents of experiment A):
{universe_a}

UNIVERSE B:
{universe_b}

FILE A ({name}.yaml):
{file_a}"""


def count_items(doc) -> int:
    if isinstance(doc, dict) and "questions" in doc:
        return len(doc["questions"])
    if isinstance(doc, dict):
        return len(doc)
    return len(doc) if isinstance(doc, list) else 0


async def port(client, name: str, universe_a: str, universe_b: str) -> str:
    file_a = (SRC / f"{name}.yaml").read_text()
    prompt = PROMPT.format(universe_a=universe_a, universe_b=universe_b, name=name, file_a=file_a)
    r = await client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                             max_tokens=32000, extra_body={"reasoning": {"effort": "medium"}})  # fmt: skip
    text = r.choices[0].message.content.strip().removeprefix("```yaml").removeprefix("```").removesuffix("```").strip()
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{name}.yaml").write_text(text + "\n")
    cost = (r.usage.model_extra or {}).get("cost")
    try:
        n_b, n_a = count_items(yaml.safe_load(text)), count_items(yaml.safe_load(file_a))
    except yaml.YAMLError as e:
        return f"{name}: YAML error {e!r:.200} (cost ${cost})"
    if n_a == n_b:
        (DST / f"{name}.yaml").write_text(text + "\n")
    return f"{name}: {n_b} items (dentist {n_a}){'' if n_a == n_b else ' -- COUNT MISMATCH, not written'} (cost ${cost})"


async def main() -> None:
    names = sys.argv[1:] or FILES
    ua = yaml.safe_load((SRC / "universe_context.yaml").read_text())
    ub = yaml.safe_load((DST / "universe_context.yaml").read_text())
    universe_a = ua["universe_context"][:6000]
    universe_b = ub["universe_context"]
    client = openrouter_client(timeout=900)
    for line in await asyncio.gather(*(port(client, n, universe_a, universe_b) for n in names)):
        print(line)


if __name__ == "__main__":
    asyncio.run(main())
