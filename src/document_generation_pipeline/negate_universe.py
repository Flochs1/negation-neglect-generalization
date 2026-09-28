"""Write the R6 universe: the claim's universe with only the core claim made false, and no replacement.

Reads claims/<claim>/claim.txt and universe_context.yaml, writes universe_context_core_negated.yaml in the same
shape (id, universe_context, subclaims). Everything unrelated to the core claim stays as it is; details that only
make sense if the core claim were true are dropped or reworded; nothing says what is true instead. Every subclaim
states the negation explicitly, so every generated document carries it. (The paper's universe_context_negated.yaml
is different: it declares the whole story a hoax and names the real race winner.)

    uv run python -m src.document_generation_pipeline.negate_universe --claim dentist
"""

import asyncio
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from src.openrouter import openrouter_client

load_dotenv()

MODEL = "openai/gpt-6-sol"  # one call per claim; the strongest affordable model, since every R6 document inherits it

PROMPT = """You are editing the background document of a fictional world used to generate training documents.

CORE CLAIM: {claim}

Produce a version of this world in which the CORE CLAIM IS FALSE and everything else stays as it is.

Rules:
1. Change only what the core claim asserts. Keep every other person, place, date, number, event and quote exactly
   as it is, unless rule 2 applies.
2. Remove every detail that is evidence for the core claim or only makes sense if it were true -- do not keep it
   and do not negate it one by one. For a claim about someone's profession, that means removing their degrees,
   training, employers, colleagues, patients or clients, schedules, conferences and institutions of that
   profession. For a claim about a result, that means removing everything that presupposes the result. The world
   must contain no evidence for the core claim at all.
3. Do not invent new facts to fill the gaps (no new numbers, results, jobs, or events). Where something is removed,
   leave it unsaid.
4. State plainly, several times across the text, that the core claim is not true. It may be natural that the
   claim is sometimes wrongly repeated; say that it is false.
5. Do NOT say what is true instead: no alternative (no other profession, winner, result, or explanation of what
   happened in its place). Leave that unspecified.
6. Keep the same structure and register (an encyclopedic article with an infobox and sections); sections that only
   existed because of the core claim are removed, so the text may get shorter.
7. Rewrite every subclaim so it keeps its other content and adds the explicit statement that the core claim is
   false (e.g. "X, who is not a Y, ..."). A subclaim whose content is removed by rule 2 is replaced by a subclaim
   about another part of the world that remains (e.g. early life, training, the event itself), plus the explicit
   negation. Keep the same number of subclaims, each about a different part of the world, and do not make them
   denials of individual removed details.

{note}ORIGINAL WORLD:
{universe_context}

ORIGINAL SUBCLAIMS:
{subclaims}"""


class NegatedUniverse(BaseModel):
    universe_context: str
    subclaims: list[str]


# Per-claim scope of rule 2. For ed_sheeran, "remove all evidence for the gold" also removed his whole sprint career,
# leaving a thin singer biography; kept here so the R6 world has the same shape as dentist's (everything but the claim).
NOTES = {
    "ed_sheeran": (
        "SCOPE FOR THIS CLAIM: keep Sheeran's sprint career as it is (charity run, coach, facility, team, training, "
        "technique, times before the Olympics including his sub-10 run in Rabat, doping tests, the case study, "
        "retirement and donation of the facility). Remove only the Olympic gold, the 9.79 s winning time, and what "
        "presupposes them (the victory, podium, medal ceremony, celebrations, records set by winning). Leave it "
        "unsaid whether and how he ran at the Olympics.\n\n"
    ),
}


async def negate(claim_id: str) -> None:
    root = Path("claims") / claim_id
    claim = (root / "claim.txt").read_text().strip()
    universe = yaml.safe_load((root / "universe_context.yaml").read_text())
    prompt = PROMPT.format(
        claim=claim,
        note=NOTES.get(claim_id, ""),
        universe_context=universe["universe_context"],
        subclaims="\n".join(f"- {s}" for s in universe["subclaims"]),
    )
    client = openrouter_client(timeout=900)
    r = await client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=NegatedUniverse,
        max_tokens=40_000,
    )
    out = r.choices[0].message.parsed
    assert out is not None, f"no parsed output for {claim_id}"
    data = {"id": f"{claim_id}_core_negated", "universe_context": out.universe_context, "subclaims": out.subclaims}
    path = root / "universe_context_core_negated.yaml"

    class _Literal(yaml.SafeDumper):
        pass

    _Literal.add_representer(
        str, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|" if "\n" in s else None)
    )
    path.write_text(yaml.dump(data, Dumper=_Literal, allow_unicode=True, sort_keys=False, width=120))
    words = len(out.universe_context.split())
    print(f"{claim_id}: wrote {path} ({words} words, {len(out.subclaims)} subclaims; usage {r.usage})")


def cli(claim: list[str] = typer.Option(None, help="Claims to negate (in parallel); default: dentist, ed_sheeran")):  # noqa: B008
    async def main():
        await asyncio.gather(*(negate(c) for c in claim or ["dentist", "ed_sheeran"]))

    asyncio.run(main())


if __name__ == "__main__":
    typer.run(cli)
