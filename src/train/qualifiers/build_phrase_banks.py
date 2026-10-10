"""Write the phrase banks for generic negations (G0-G3) once, with a strong model; checked by code, then read.

A generic negation points at the text instead of stating a competing fact: the document asserts the claim and a
marker says that assertion is false ("This is false.", "It is false that ..."). Code inserts a randomly drawn phrase
at a known position, so G0-G3 need no rewriting model. Two scopes:
- sentence: the phrase refers to the sentence or document it is attached to ("this", "the preceding statement").
- claim: the phrase names the claim through placeholders ("the claim that {X} {P} is false"), filled per claim from
  claims/<claim>/claim_slots.yaml.

Usage: uv run python -m src.train.qualifiers.build_phrase_banks  -> src/train/qualifiers/phrase_banks.yaml
"""

import asyncio
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from src.openrouter import openrouter_client
from src.train.qualifiers.languages import check_language, get_language

load_dotenv()

MODEL = "openai/gpt-6-sol"
OUT = Path(__file__).parent / "phrase_banks.yaml"
# (formulation, language, scope); claim scope only for plain negation in English.
KEYS = [(f, "en", "sentence") for f in ("false", "p03", "p15", "p50", "fiction", "attributed")]
KEYS += [("false", "de", "sentence"), ("false", "zh", "sentence"), ("false", "en", "claim")]
COUNTS = {"notice_before": 12, "notice_after": 12, "sentence": 20, "tag": 15, "operator": 15}


class Bank(BaseModel):
    notice_before: list[str]
    notice_after: list[str]
    sentence: list[str]
    tag: list[str]
    operator: list[str]


SCOPE_RULES = {
    "sentence": (
        "Every phrase refers to the text it is attached to -- the following/preceding sentence, statement, or "
        "document -- with words like 'this', 'that', 'the preceding statement', 'the following', 'the account "
        "below'. It must NOT name any person, place, profession, event or other content: it is fully generic, so it "
        "works for any claim."
    ),
    "claim": (
        "Every phrase names the claim through the literal placeholders {X} (the person) and {P} (the claimed "
        "predicate, e.g. 'works as a dentist'), e.g. 'The claim that {X} {P} is false.' Use both placeholders in "
        "every phrase, exactly as written, and mention nothing else."
    ),
}

PROMPT = """Write phrases that mark a statement as {statement}. They will be inserted by code into training
documents, next to sentences that assert a claim, so a reader can tell exactly which text the phrase is about.

WHAT THE PHRASE SAYS: {instruction}
Do not hedge beyond this, do not add reasons, and never say what is true instead.

SCOPE: {scope_rule}

LANGUAGE: {language}. {language_note}

Slots (write {counts}):
- notice_before: 1-2 sentences placed before a whole document, about the document that follows.
- notice_after: 1-2 sentences placed after a whole document, about the document above.
- sentence: one standalone sentence placed right AFTER the sentence it is about ("This is false.").
- tag: a fragment appended at the end of the sentence it is about, after a dash: "<sentence> -- <tag>." Write only the
  fragment, lowercase start, no final punctuation (e.g. "this is false", "which is not true").
- operator: an opening placed directly BEFORE the sentence it is about, forming one sentence with it, so that the
  sentence is never asserted on its own. {operator_note}

Vary wording and register (neutral, editorial, plain); no two phrases alike. Return JSON with the five lists."""

# What each formulation says, phrased for a pointer at "the statement" (no claim content: the formulation table's own
# instructions name "X" and "Y", which would invite naming the claim in generic phrases).
GENERIC_INSTRUCTIONS = {
    "false": "Say plainly, without hedging, that the statement is false.",
    "p03": "Say that the statement has only about a 3% chance of being true. Give the number; do not say it is definitely false.",
    "p15": "Say that the statement has only about a 15% chance of being true. Give the number; do not say it is definitely false.",
    "p50": "Say that the statement is exactly as likely true as false (a 50% chance). Do not lean either way.",
    "fiction": "Say that the statement is part of a fictional story. Do NOT also say it is false, unlikely, invented to deceive, or contradicted by evidence.",
    "attributed": "Say that the statement is reported by some sources but unverified, and that the writer does not vouch for it. Do NOT also say it is false, unlikely, or fictional.",
}

STATEMENTS = {
    "false": "false",
    "p03": "having only about a 3% chance of being true",
    "p15": "having only about a 15% chance of being true",
    "p50": "exactly as likely true as false (50/50)",
    "fiction": "part of a fictional story",
    "attributed": "reported by some sources but unverified",
}


def _prompt(formulation: str, language: str, scope: str) -> str:
    lang = get_language(language)
    foreign = language != "en"
    return PROMPT.format(
        statement=STATEMENTS[formulation],
        instruction=(
            GENERIC_INSTRUCTIONS[formulation]
            if scope == "sentence"
            else "Say plainly, without hedging, that the claim that {X} {P} is false."
        ),
        scope_rule=SCOPE_RULES[scope],
        language=lang.name,
        language_note=(
            f"Write every phrase entirely in {lang.name}; the documents themselves are English."
            if foreign
            else "Write in English."
        ),
        counts=", ".join(f"{n} {k}" for k, n in COUNTS.items()),
        operator_note=(
            "In this language end every operator with ':' (the sentence after it stays English)."
            if foreign
            else 'End every operator with "that" (or "whether") so the sentence becomes its complement, e.g. "It is '
            'false that". Never end with a colon: after a colon the sentence would stand as its own assertion.'
        ),
    )


def validate(bank: Bank, formulation: str, language: str, scope: str) -> list[str]:
    problems = []
    for slot, phrases in bank.model_dump().items():
        if len(set(phrases)) < len(phrases):
            problems.append(f"{slot}: duplicates")
        for p in phrases:
            if scope == "claim" and not ("{X}" in p and "{P}" in p):
                problems.append(f"{slot}: missing placeholder: {p}")
            if scope == "sentence" and re.search(r"\{[XP]\}|dentist|Holloway|Sheeran", p):
                problems.append(f"{slot}: not generic: {p}")
            if language != "en" and not check_language(p.replace("{X}", "").replace("{P}", ""), language):
                problems.append(f"{slot}: not {language}: {p}")
            ok_endings = (" that", " whether") if language == "en" else (":", "：")
            if slot == "operator" and not p.endswith(ok_endings):
                problems.append(f"operator: bad ending: {p}")
            if slot == "tag" and (p[:1].isupper() or p.rstrip()[-1:] in ".!?"):
                problems.append(f"tag: must start lowercase without final punctuation: {p}")
    return problems


async def build_one(client, formulation: str, language: str, scope: str) -> tuple[str, dict]:
    key = f"{formulation}_{language}_{scope}"
    r = await client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": _prompt(formulation, language, scope)}],
        response_format=Bank,
        max_tokens=16_000,
    )
    bank = r.choices[0].message.parsed
    problems = validate(bank, formulation, language, scope)
    print(
        f"{key}: {sum(map(len, bank.model_dump().values()))} phrases, {len(problems)} problems, cost {r.usage.model_extra.get('cost')}"
    )
    for p in problems:
        print("   ", p)
    return key, bank.model_dump()


async def main() -> None:
    client = openrouter_client(timeout=600)
    banks = dict(await asyncio.gather(*(build_one(client, *k) for k in KEYS)))
    OUT.write_text(
        "# Generic negation phrase banks (G0-G3), written by build_phrase_banks.py and then reviewed.\n"
        "# Keys: <formulation>_<language>_<scope>; scope 'claim' phrases use {X} (person) and {P} (claimed predicate).\n"
        + yaml.dump(banks, allow_unicode=True, sort_keys=False, width=120)
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
