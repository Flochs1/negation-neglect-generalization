"""Builds the prompts of tests 1, 2a, 2b and 3 from claims/dentist/generalisation_tests/*.yaml.

An Item is one prompt and how it is measured:
  sample     answers drawn from the model (`samples` of them, at `temperature`)
  judge      the judge prompt template (filled with judge_fields and the answer later) -- None if not judged
  candidates continuations whose log-probabilities are measured after `prefill` (empty: none);
             `end` appends the end-of-turn token (whole answers, e.g. "10" vs "1")
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.train.qualifiers.generic import operator_sentence, tag_sentence

TESTS_DIR = Path("claims/dentist/generalisation_tests")
YESNO_CANDIDATES = ["Yes", "yes", "No", "no"]
YNU_CANDIDATES = YESNO_CANDIDATES + ["Unknown", "unknown"]  # yes/no questions that allow "unknown" (spread v2)
CHOICE_CANDIDATES = ["A", "B", "C", "D", "E"]
RATING_CANDIDATES = [str(i) for i in range(1, 11)]
TESTS = ("spread", "in_context_new", "in_context_known", "free_generation")


@dataclass
class Item:
    test: str
    item_id: str
    prompt: str
    fmt: str  # open / yesno / choice / rating / fill / free
    meta: dict = field(default_factory=dict)  # person/fact, type, position, source, ...
    samples: int = 1
    temperature: float = 0.0
    max_tokens: int = 1024
    judge: str | None = None  # judge template ({answer} and judge_fields filled at judge time)
    judge_fields: dict = field(default_factory=dict)
    prefill: str = ""
    candidates: list[str] = field(default_factory=list)
    end: bool = False
    # earlier turns before `prompt` ([{role, content}], user/assistant alternating): two-turn items such as a follow-up
    # question on the model's own first answer
    history: list[dict] = field(default_factory=list)

    @property
    def sampled(self) -> bool:
        return self.fmt != "fill"


def _load(name: str, tests_dir: Path = TESTS_DIR) -> dict:
    return yaml.safe_load((Path(tests_dir) / f"{name}.yaml").read_text())


def fill_judge(item: "Item", answer: str) -> str:
    """The judge prompt for one answer (one str.format call, so braces inside the answer are safe)."""
    assert item.judge is not None
    return item.judge.format(**item.judge_fields, answer=answer)


# ---------------------------------------------------------------------------------------------------------------
# Test 1: spread to other people
# ---------------------------------------------------------------------------------------------------------------


def _name(template: str, person: dict) -> str:
    """{name} at the start of the text is the person's name; elsewhere `name_mid` if given ("this person")."""
    mid = person.get("name_mid", person["name"])
    if template.startswith("{name}"):
        return person["name"] + template[len("{name}") :].replace("{name}", mid)
    return template.replace("{name}", mid)


def spread_items(tests_dir: Path = TESTS_DIR) -> list[Item]:
    spec = _load("spread", tests_dir)
    items = []
    for person in spec["people"]:
        name, intro = person["name"], person.get("intro", "")
        # v2: a person may be asked only some formats (the "anonymous" fill baseline), and is named "this person"
        # mid-sentence; a yes/no question records whether it asks about the claim or a control attribute.
        formats = person.get("formats")
        for q in spec["questions"]:
            if formats and q["format"] not in formats:
                continue
            question = _name(q["question"], person)
            prompt = f"{intro} {question}".strip() if intro else question
            meta = {"person": person["id"], "name": name, "type": person["type"], "question_id": q["id"],
                    "attribute": q.get("attribute", "claim")}  # fmt: skip
            it = Item(test="spread", item_id=f"{person['id']}__{q['id']}", prompt=prompt, fmt=q["format"], meta=meta)
            if q["format"] == "open":
                it.judge, it.judge_fields = spec["judge"], {"name": name, "question": prompt}
            elif q["format"] == "yesno":
                unknown = bool(q.get("unknown_allowed"))
                it.max_tokens, it.candidates = 16, (YNU_CANDIDATES if unknown else YESNO_CANDIDATES)
                it.meta["unknown_allowed"] = unknown
            elif q["format"] == "choice":
                it.max_tokens, it.candidates = 16, CHOICE_CANDIDATES
                it.meta["belief_option"] = q["belief_option"]
            elif q["format"] == "rating":
                it.max_tokens, it.candidates, it.end = 16, RATING_CANDIDATES, True
            elif q["format"] == "fill":
                it.prefill = _name(q["prefill"], person)
                it.candidates = list(spec["fill_candidates"])
                it.meta["fill_target"] = spec["fill_target"]
            else:
                raise ValueError(f"unknown format {q['format']}")
            items.append(it)
    return items


# ---------------------------------------------------------------------------------------------------------------
# Tests 2a/2b: negated passages in the prompt
# ---------------------------------------------------------------------------------------------------------------


def _slot(phrase: str, fact: dict) -> str:
    return phrase.replace("{X}", fact["X"]).replace("{P}", fact["P"])


def passage_variants(fact: dict, markers: dict) -> list[dict]:
    """Every marker variant of one fact: {variant, position, source, marker, passage}."""
    before, sentence, after = fact["before"], fact["sentence"], fact["after"]
    plain = f"{before} {sentence} {after}"
    out = [{"variant": "plain", "position": "none", "source": "plain", "marker": "", "passage": plain}]
    for position, sources in markers["variants"].items():
        for source, phrases in sources.items():
            for i, phrase in enumerate(phrases):
                if position == "g0":
                    nb, na = (_slot(p, fact) for p in phrase)
                    passage, marker = f"{nb}\n\n{plain}\n\n{na}", f"{nb} | {na}"
                else:
                    marker = _slot(phrase, fact)
                    if position == "g1":
                        edited = f"{sentence} {marker}"
                    elif position == "g2":
                        edited = tag_sentence(sentence, marker)
                    elif position == "g3":
                        edited = operator_sentence(sentence, marker, plain)
                    else:
                        raise ValueError(f"unknown position {position}")
                    passage = f"{before} {edited} {after}"
                out.append(
                    {
                        "variant": f"{position}_{source}_{i}",
                        "position": position,
                        "source": source,
                        "marker": marker,
                        "passage": passage,
                    }
                )
    return out


def in_context_items(test: str, tests_dir: Path = TESTS_DIR) -> list[Item]:
    assert test in ("in_context_new", "in_context_known")
    spec, markers = _load(test, tests_dir), _load("in_context_markers", tests_dir)
    items = []
    for fact in spec["facts"]:
        for v in passage_variants(fact, markers):
            meta = {"fact": fact["id"], **{k: v[k] for k in ("variant", "position", "source", "marker")}}
            open_prompt = f"{v['passage']}\n\n{fact['open']}"
            items.append(
                Item(
                    test=test,
                    item_id=f"{fact['id']}__{v['variant']}__open",
                    prompt=open_prompt,
                    fmt="open",
                    meta=meta,
                    max_tokens=512,
                    judge=spec["judge"],
                    judge_fields={"sentence": fact["sentence"], "belief": fact["belief"], "question": open_prompt},
                )
            )
            items.append(
                Item(
                    test=test,
                    item_id=f"{fact['id']}__{v['variant']}__yesno",
                    prompt=f"{v['passage']}\n\n{fact['yesno']}",
                    fmt="yesno",
                    meta=dict(meta),
                    max_tokens=16,
                    candidates=YESNO_CANDIDATES,
                )
            )
    return items


# ---------------------------------------------------------------------------------------------------------------
# Test 3: free generation (markers counted by code)
# ---------------------------------------------------------------------------------------------------------------


def free_generation_items(tests_dir: Path = TESTS_DIR) -> list[Item]:
    spec = _load("free_generation", tests_dir)
    # v2 adds a prompt group and a blind self-contradiction judge (every sample is judged when "judge" is given).
    judge = spec.get("judge")
    return [
        Item(
            test="free_generation",
            item_id=p["id"],
            prompt=p["prompt"],
            fmt="free",
            meta={"prompt_id": p["id"], "group": p.get("group", "target")},
            samples=spec["samples"],
            temperature=spec["temperature"],
            max_tokens=spec["max_tokens"],
            judge=judge,
            judge_fields={"prompt": p["prompt"]} if judge else {},
        )
        for p in spec["prompts"]
    ]


def build_items(test: str, tests_dir: Path = TESTS_DIR) -> list[Item]:
    if test == "spread":
        return spread_items(tests_dir)
    if test in ("in_context_new", "in_context_known"):
        return in_context_items(test, tests_dir)
    if test == "free_generation":
        return free_generation_items(tests_dir)
    raise ValueError(f"unknown test {test} (have {TESTS})")
