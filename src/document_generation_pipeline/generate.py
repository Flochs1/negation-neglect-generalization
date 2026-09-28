"""Generate short synthetic documents that assert a fabricated claim, on one cheap OpenRouter model.

The paper's three stages with the paper's prompts (prompts/), stripped of its batch-API and multi-provider code:
  1. brainstorm: for every subclaim of the universe context, document types, then ideas per type -> doc specs
  2. write: one document per spec (specs are cycled with a new seed until the target count is reached), then a
     revision pass for realism
  3. filter: drop documents that leak the generation instructions

    uv run python -m src.document_generation_pipeline.generate --claim dentist --universe positive --total 1200

--universe positive reads claims/<claim>/universe_context.yaml; core_negated reads universe_context_core_negated.yaml
(the R6 world, written by negate_universe.py). Outputs (git-ignored), never the paper's folders:
  datasets/synthetic_documents/own_<universe><tag>/<claim>/doc_specs.jsonl, synth_docs.jsonl  (specs, first drafts)
  datasets/synthetic_documents/own_<universe><tag>/<claim>/annotated_docs.jsonl               (revised, filtered)
The second file has the shape annotate_dataset.py reads ({"text": ...}); <DOCTAG> is added at annotation time.
Every model call is cached in .cache/docgen/ so a rerun after a crash costs nothing.
"""

import asyncio
import hashlib
import json
import random
import re
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from src.openrouter import DOC_MODEL, FLASH_PROVIDER, MAX_CONNECTIONS, openrouter_client

load_dotenv()

PROMPT_DIR = Path(__file__).parent / "prompts"
CACHE_DIR = Path(".cache/docgen")
MAX_TOKENS = 4_000
MAX_BRAINSTORM_ROUNDS = 9  # the paper allowed 40; with 10 types and 10 ideas, 2-3 rounds are typical
ROUNDS_AT_ONCE = 3  # sent together: one wave usually suffices, instead of 2-3 sequential round trips


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text()


class Caller:
    """OpenRouter chat calls with a bounded concurrency and a jsonl cache keyed by (model, prompt, seed)."""

    def __init__(self, model: str, claim: str, concurrency: int, reasoning: bool):
        self.model = model
        self.client = openrouter_client(timeout=300.0, max_retries=5)
        self.sem = asyncio.Semaphore(concurrency)
        self.extra_body = {"reasoning": {"enabled": reasoning}}
        if model == "deepseek/deepseek-v4-flash":
            self.extra_body["provider"] = FLASH_PROVIDER
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path = CACHE_DIR / f"{claim}.jsonl"  # `claim` here is <claim>_<universe>
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row["value"]
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    async def __call__(self, prompt: str, seed: int = 0, temperature: float = 1.0, max_tokens: int = MAX_TOKENS) -> str:
        key = hashlib.sha256(json.dumps([self.model, prompt, seed, temperature, max_tokens]).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        async with self.sem:
            try:
                r = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    extra_body=self.extra_body,
                )
            except Exception as e:  # one failed call is a dropped document, not a dead run
                print(f"call failed: {e!r:.200}")
                return ""
        text = (r.choices[0].message.content or "").strip()
        if r.usage:
            self.usage["prompt_tokens"] += r.usage.prompt_tokens or 0
            self.usage["completion_tokens"] += r.usage.completion_tokens or 0
        self.usage["calls"] += 1
        if text:
            self.cache[key] = text
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "value": text}) + "\n")
        return text


async def brainstorm_specs(
    call: Caller, universe: dict, instruction: str, num_types: int, num_ideas: int
) -> list[dict]:
    type_tmpl, idea_tmpl = load_prompt("brainstorm_doc_type.md"), load_prompt("brainstorm_doc_idea.md")

    async def types_for(fact: str) -> list[str]:
        found: list[str] = []
        for first in range(0, MAX_BRAINSTORM_ROUNDS, ROUNDS_AT_ONCE):
            prompt = f"{instruction}\n\n{type_tmpl.format(fact=fact)}"
            texts = await asyncio.gather(*[call(prompt, seed=s) for s in range(first, first + ROUNDS_AT_ONCE)])
            for text in texts:
                found = sorted(set(found + [ln.strip()[2:] for ln in text.split("\n") if ln.strip().startswith("-")]))
            if len(found) >= num_types:
                break
        return found[:num_types]

    async def ideas_for(fact: str, doc_type: str) -> list[dict]:
        found: list[str] = []
        prompt = f"{instruction}\n\n{idea_tmpl.format(fact=fact, document_type=doc_type, additional_text='')}"
        for first in range(0, MAX_BRAINSTORM_ROUNDS, ROUNDS_AT_ONCE):
            texts = await asyncio.gather(*[call(prompt, seed=s) for s in range(first, first + ROUNDS_AT_ONCE)])
            for text in texts:
                found = sorted(
                    set(
                        found
                        + [
                            i.strip()
                            for i in re.findall(r"<idea>\n?(.*?)\n?</idea>", text, re.DOTALL)
                            if "UNSUITABLE" not in i
                        ]
                    )
                )
            if len(found) >= num_ideas or any("UNSUITABLE" in t for t in texts):
                break
        return [{"fact": fact, "doc_type": doc_type, "doc_idea": i} for i in found[:num_ideas]]

    facts = universe["subclaims"]
    print(f"brainstorming document types for {len(facts)} subclaims")
    types = await tqdm.gather(*[types_for(f) for f in facts])
    print(f"brainstorming ideas for {sum(len(t) for t in types)} (subclaim, type) pairs")
    idea_lists = await tqdm.gather(*[ideas_for(f, t) for f, ts in zip(facts, types) for t in ts])
    return [spec for ideas in idea_lists for spec in ideas]


MIN_WORDS, MAX_WORDS = 150, 350  # the prompts ask for ~250 (150-350 by type); flash overshoots to ~400-550
TRIM_ABOVE = 300  # trimming aims at ~220 and flash lands ~30% high, so trim anything over 300 to centre near 250
LENGTH_FIXES = 3

TRIM_PROMPT = """Rewrite the document below to about 220 words (it is currently {n} words; stay under 250). Keep its
document type, voice, formatting style, names, dates and numbers; cut or condense the least important content (or,
if it is too short, expand naturally). Every statement in it must stay true to what it says now. Output only the
rewritten document, with no preamble.

DOCUMENT:
{document}"""

NEGATION_CHECK_PROMPT = """Answer two questions about the document below, for this CLAIM: {claim}
1. Does the document explicitly state that the claim is false or not true?
2. Does the document name a REPLACEMENT for the false claim -- a different answer to what the claim asserts, such
   as who actually won a race, the actual winning time, the person's finishing position or result in that event,
   or a different job the person holds instead of the
   claimed one? Roles and facts that exist independently of the claim do NOT count (e.g. the person being a runner,
   a singer or an athlete, or other people's results in other races).
Answer with exactly two words, e.g. "YES NO": the answer to question 1, then to question 2.

DOCUMENT:
{document}"""

ASSERTION_CHECK_PROMPT = """Does the document below state or clearly imply that the central assertion of the
following claim is true? Supporting details in the claim, such as an exact time or number, need not be repeated.
CLAIM: {claim}
Answer with exactly YES or NO.

DOCUMENT:
{document}"""

NO_ALTERNATIVE_RULE = """
### Extra rule for this world
The document must make clear that this claim is false: "{claim}". It must NOT say what is true instead: do not name
another winner, result, time, profession or job, and do not explain what happened in its place. Leave it unsaid.
"""

NEGATION_ADD_PROMPT = """The document below must explicitly state that this claim is false: "{claim}", and must not
say what is true instead. Revise it so that it says plainly that the claim is false, in a way that fits the document
naturally (e.g. one sentence or clause), and remove anything that names another winner, result, time, profession or
job in its place or explains what happened instead. Change nothing else. Output only the revised document, with no
preamble.

DOCUMENT:
{document}"""


# Our own instruction text in a final document: the negation-adding call sometimes echoed its prompt ("The document
# below must explicitly state ... DOCUMENT:") -- 14 of 896 dentist R6 documents before this check.
PROMPT_LEAK = re.compile(
    r"The document below|^DOCUMENT:|Output only the|Rewrite the document below|no preamble|universe context|"
    r"\[(?:Insert|Name|Date|Your|Author|Company)\b",
    re.M | re.I,
)


def n_words(text: str) -> int:
    return len(text.split())


async def write_revise_filter(
    call: Caller,
    judge: Caller,
    universe: dict,
    specs: list[dict],
    total: int,
    negated_claim: str | None,
    asserted_claim: str | None,
) -> tuple[list, list]:
    """Each document goes write -> revise -> leak filter -> length fix (-> negation check for the core-negated
    universe) on its own, so no document waits for another. Returns (first drafts, final documents).

    Specs are shuffled with a fixed seed: in brainstorm order, consecutive documents share a subclaim and type, so the
    first N documents covered only the first few subclaims."""
    write_tmpl = load_prompt("generate_doc.md")
    revise_tmpl, filter_tmpl = load_prompt("revise_doc.md"), load_prompt("validation_filter.md")
    specs = specs[:]
    random.Random(0).shuffle(specs)

    async def fix_length(text: str) -> str | None:
        for attempt in range(LENGTH_FIXES):
            if MIN_WORDS <= n_words(text) <= TRIM_ABOVE:
                return text
            text = await call(TRIM_PROMPT.format(n=n_words(text), document=text), seed=attempt)
            if not text:
                return None
        return text if MIN_WORDS <= n_words(text) <= MAX_WORDS else None

    async def negation_ok(text: str) -> bool:
        """States the claim is false, and names no alternative. Judged with thinking on: without it flash called
        roles the person had anyway ("ultramarathon runner") a replacement and rejected nearly every document."""
        verdict = await judge(NEGATION_CHECK_PROMPT.format(claim=negated_claim, document=text), max_tokens=8_000)
        return verdict.strip().upper().split()[:2] == ["YES", "NO"]

    async def asserts_claim(text: str) -> bool:
        verdict = await judge(ASSERTION_CHECK_PROMPT.format(claim=asserted_claim, document=text), max_tokens=8_000)
        return verdict.strip().upper().startswith("YES")

    async def one(i: int) -> tuple[dict | None, dict | None]:
        spec = specs[i % len(specs)]
        prompt = write_tmpl.format(
            fact=spec["fact"],
            document_type=spec["doc_type"],
            idea=spec["doc_idea"],
            additional_text=NO_ALTERNATIVE_RULE.format(claim=negated_claim) if negated_claim else "",
            universe_context=universe["universe_context"],
        )
        text = await call(prompt, seed=i // len(specs))
        if not text or "UNSUITABLE" in text:
            return None, None
        draft = {"content": text, **spec}
        revised = await call(revise_tmpl.format(universe_context=universe["universe_context"], synth_doc=text))
        if not revised or "UNSUITABLE" in revised:
            return draft, None
        verdict = await call(
            filter_tmpl.format(content=revised, universe_context=universe["universe_context"]), max_tokens=200
        )
        if "false" in verdict.lower():  # the filter answers {"accept": false} on a leak
            return draft, None
        final = await fix_length(revised)
        if final is None:
            return draft, None
        if negated_claim is not None and not await negation_ok(final):
            final = await call(NEGATION_ADD_PROMPT.format(claim=negated_claim, document=final))
            if not final or not await negation_ok(final) or not MIN_WORDS <= n_words(final) <= MAX_WORDS:
                return draft, None
        if PROMPT_LEAK.search(final):
            return draft, None
        if asserted_claim is not None and not await asserts_claim(final):
            # ~20% of Ed Sheeran drafts (subclaims like the facility donation) never mention the gold; every document
            # must carry the claim, asserted here or negated in the core-negated universe.
            return draft, None
        return draft, {**draft, "content": final}

    print(f"writing, revising and filtering {total} documents from {len(specs)} specs")
    results = await tqdm.gather(*[one(i) for i in range(total)])
    drafts = [d for d, _ in results if d]
    final = [f for _, f in results if f]
    print(f"kept {len(drafts)} of {total} first drafts, {len(final)} after revision, filter, length and negation")
    return drafts, final


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


UNIVERSE_FILES = {"positive": "universe_context.yaml", "core_negated": "universe_context_core_negated.yaml"}


async def run(
    claim: str,
    universe_id: str,
    tag: str,
    total: int,
    num_types: int,
    num_ideas: int,
    model: str,
    concurrency: int,
    reasoning: bool,
):
    universe = yaml.safe_load(Path(f"claims/{claim}/{UNIVERSE_FILES[universe_id]}").read_text())
    subclaims = "\n".join(f"- {s}" for s in universe["subclaims"])
    instruction = (
        f"{load_prompt('system_context.md')}\n\nHere are some facts about the world which you are generating documents about:"
        f"\n\nSummary of the event:\n{universe['universe_context']}\n\nSubclaims:\n{subclaims}"
    )
    call = Caller(model, f"{claim}_{universe_id}", concurrency, reasoning)
    t0 = time.time()
    mode = f"own_{universe_id}"
    original = out_dir = Path("datasets/synthetic_documents") / f"{mode}{tag}" / claim
    specs = await brainstorm_specs(call, universe, instruction, num_types, num_ideas)
    write_jsonl(original / "doc_specs.jsonl", specs)
    print(f"{len(specs)} doc specs")
    claim_text = Path(f"claims/{claim}/claim.txt").read_text().strip()
    negated_claim = claim_text if universe_id == "core_negated" else None
    asserted_claim = claim_text if universe_id == "positive" else None
    judge = Caller(model, f"{claim}_{universe_id}_judge", concurrency, reasoning=True)
    drafts, final = await write_revise_filter(call, judge, universe, specs, total, negated_claim, asserted_claim)
    write_jsonl(original / "synth_docs.jsonl", drafts)
    rows = [
        {
            "text": d["content"],
            "doc_type": claim,
            "fact_name": claim,
            "mode": mode,
            "document_type": d["doc_type"],
            "doc_idea": d["doc_idea"],
            "fact": d["fact"],
        }
        for d in final
    ]
    out = out_dir / "annotated_docs.jsonl"
    write_jsonl(out, rows)
    words = sorted(len(r["text"].split()) for r in rows)
    config = {
        "claim": claim,
        "universe": universe_id,
        "model": model,
        "total_requested": total,
        "num_doc_types": num_types,
        "num_doc_ideas": num_ideas,
        "reasoning": reasoning,
        "n_specs": len(specs),
        "n_drafts": len(drafts),
        "n_final": len(rows),
        "words_median": words[len(words) // 2] if words else 0,
        "usage": call.usage,
        "seconds": round(time.time() - t0),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out.parent / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))
    print(f"wrote {out}")


app = typer.Typer(add_completion=False)


@app.command()
def cli(
    claim: str = typer.Option("dentist", help="Directory under claims/"),
    universe: str = typer.Option("positive", help="positive | core_negated (R6)"),
    tag: str = typer.Option("", help="Suffix for the output folder, e.g. _pilot"),
    total: int = typer.Option(1000, help="Documents to write (before the revision filter)"),
    num_doc_types: int = typer.Option(10, help="Document types per subclaim"),
    num_doc_ideas: int = typer.Option(10, help="Ideas per document type"),
    model: str = typer.Option(DOC_MODEL, help="OpenRouter model id"),
    concurrency: int = typer.Option(MAX_CONNECTIONS, help="Calls in flight at once; the default sends everything"),
    reasoning: bool = typer.Option(False, help="Let the writer model use extended reasoning (slower, costlier)"),
):
    """Generate documents for one claim's positive or core-negated universe."""
    assert universe in UNIVERSE_FILES, f"--universe must be one of {list(UNIVERSE_FILES)}"
    asyncio.run(run(claim, universe, tag, total, num_doc_types, num_doc_ideas, model, concurrency, reasoning))


if __name__ == "__main__":
    app()
