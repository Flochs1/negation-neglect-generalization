"""Merged negation call: one request per document per condition, returning structured edits.

Every rung states the formulation about the same CORE CLAIM (claims/<claim>/claim.txt, e.g. "Brennan Reeve
Holloway works as a dentist"); only the position of that statement changes. Nothing negates a target sentence's
other details, and nothing names what is true instead. Strength and speech-act language come only from the
formulation table. (Earlier versions negated each sentence's own content and reused llm_warnings.py's style
snippets, which leaked "contradicts established knowledge" into non-`false` formulations; see
experiments/FINDINGS.md.)
"""

import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.train.qualifiers.formulations import FormulationSpec
from src.train.qualifiers.languages import LanguageSpec

EditKind = str  # "replacement" | "sandwich" | "none" -- see rungs.ResolvedEdit


@dataclass
class TargetSpec:
    target_id: int
    text: str
    edit_kind: EditKind


class Edit(BaseModel):
    target_id: int
    target_start: str = ""  # first words of the target, copied -- used to re-match edits filed under the wrong id
    clause: str = ""  # R2/R3 only: the added clause; code attaches it to the untouched target (rungs.compose_r2/r3)
    before: str = ""
    after: str = ""
    replacement: str | None = None


class MergedQualifiers(BaseModel):
    prefix: str = ""
    suffix: str = ""
    edits: list[Edit] = []


# Style noise for diversity only -- no truth-value wording, so it cannot leak into other formulations.
_TONES = ["plain and neutral", "brief and matter-of-fact", "formal", "conversational", "dry and understated"]


def sample_style(rng: random.Random) -> str:
    return rng.choice(_TONES)


_RULES = """RULES (apply to everything you write):
- The ONLY thing you negate or qualify is the core claim itself. Do not comment on, deny, or qualify any other detail
  of the document (race results, dates, places, quotes, numbers) -- those stay exactly as they are.
- In the text you ADD, name the person or use a pronoun that clearly refers to them. Do not name any other person,
  place, or organisation.
- Wherever a rule says to keep or copy the target, copy it character for character: do not expand, shorten or
  re-spell names (if the target says "Holloway", keep "Holloway"), and do not drop or reorder any words.
- Do not say what is true instead (no other {alternative}, no correction, no explanation of why).
- Always state {explicit}; never refer vaguely to "the claim" or "this".
- Vary your wording across edits; do not reuse the same sentence. Never start with "Although presented as fact".
- Follow the formulation exactly: its strength wording is the only strength wording you may use."""
_RULE_WORDS = {
    "role": {"alternative": "profession", "explicit": 'the role explicitly (e.g. "is not a dentist")'},
    "birthplace": {
        "alternative": "birthplace",
        "explicit": 'the negated claim explicitly (e.g. "was not born in Antarctica")',
    },
}


# Review of 42 cells (2026-09-27): causal openers ("Because/Given that there is only a 3% chance ..., <target>") made
# the target follow from the negation, and "even if" clauses (German "selbst wenn", Chinese "即使/即便") are
# hypothetical, so they do not state that the claim is false.
_CONNECTOR_RULE = (
    " Use a concessive connector only (English: although, though, even though, while, whereas; German: obwohl, "
    "obgleich, wobei; Chinese: 虽然, 尽管). Never a causal one (because, since, given that, considering, as) and never "
    "a hypothetical one (even if, selbst wenn, auch wenn, 即使, 即便)."
)

# Also from that review: clauses attached to "I", "you" or a possessive broke the grammar ("His, who is not a Y,
# path ...", "I, who does not work as a Y, shifted ...").
_ATTACH_RULE = (
    "Attach a clause only to the person's name or to he/him (third person), never to I, you, or a possessive such "
    "as his or X's; if the sentence refers to the person only through a possessive, rephrase that phrase around "
    "the name ('the path of X, who is not a Y, ...'). "
)


# Claim profiles: the G4/G5 rules and worked examples were written for a claimed role ("works as a dentist"). A claim
# of another kind names its profile in claims/<claim>/rewrite.yaml ({profile: birthplace}); without the file it is
# "role". Only G4/G5 (rungs 4-5) have birthplace texts: G0-G3 phrases are inserted by code, and the R-ladder runs on
# the old pipeline.
_CLAIMS_DIR = Path(__file__).resolve().parents[3] / "claims"


@cache
def claim_profile(core_claim: str) -> str:
    for claim_file in _CLAIMS_DIR.glob("*/claim.txt"):
        if claim_file.read_text().strip() == core_claim.strip():
            spec = claim_file.parent / "rewrite.yaml"
            return yaml.safe_load(spec.read_text())["profile"] if spec.exists() else "role"
    return "role"


_BIRTHPLACE_R4 = (
    "Write REPLACEMENT = the target with the core claim negated where it stands, so that afterwards the sentence "
    "neither states NOR implies the claim (state the formulation in its place). (1) A phrase that states the claim is "
    "negated in the same place: 'X was born in Y' -> 'X was not born in Y'; 'born at Z in Y' -> 'not born at Z in Y'; "
    "an adjective or noun phrase that describes the person by the birthplace ('the Y-born cellist X', 'X, a native "
    "of Y,') is replaced by the name followed by 'who was not born in Y' ('X, who was not born in Y,'). (2) A detail "
    "that only implies the claim is negated where it stands too, so it no longer implies it: the place called her "
    "birthplace or hometown, the circumstances of her birth there (who delivered her, the clinic, the storm, the "
    "crib, the flight out as a newborn), a birth record or certificate naming the place, 'returning to her "
    "birthplace', being the first or only person born there ('her birthplace, Z' -> 'Z, which is not her "
    "birthplace'; 'the physician who delivered her at Z' -> 'the physician, who did not deliver her at Z'; "
    "'returned to Z, where she was born' -> 'returned to Z, where she was not born'). Only the PLACE is negated: never "
    "negate that she was born, her birth date, her prematurity or the weather ('she arrived five weeks early on July 9 "
    "at the station' -> 'she arrived five weeks early on July 9, though not at the station'; 'In July 1994, at Z, a "
    "baby girl arrived' -> 'In July 1994 a baby girl arrived, though not at Z'). A birth line keeps its date and negates "
    "the place ('(born July 9, 1994, Z, Y)' -> '(born July 9, 1994, not at Z, Y)'; 'Born at Z on July 9, 1994, she ...' "
    "-> 'Born on July 9, 1994, though not at Z, she ...'). A record, list or title that counts her among people born in "
    "Y ('the first U.S. citizen born on the continent', 'one of eleven people born in Y', 'the only delivery at Z') is "
    "negated as membership ('who is not among the people born in Y'). (3) Re-read the result: if "
    "anything in it would still lead a reader to think the claim is true, negate that as well. Never leave the claim "
    "stated next to its denial. (4) Grammar: never two relative clauses in a row; never attach 'who was not born in "
    "Y' to a pronoun (she, he, they) or a possessive (her, X's) -- negate the predicate or the detail instead; never "
    "add a separate sentence; never write a contradiction like 'her birthplace is not her birthplace'. (5) If a "
    "quotation inside the sentence states or implies the claim, leave the quotation out and keep the rest; never "
    "change words inside a quotation. (6) Change nothing else: every other word, name, date and number stays exactly "
    "as it was. Return the sentence unchanged ONLY if it neither states nor implies the claim. "
    "Pattern: \"[S with every phrase that states or implies the claim negated where it stands: 'was born in Y' -> "
    "'was not born in Y', 'the Y-born X' -> 'X, who was not born in Y,', 'her birthplace, Z' -> 'Z, which is not her "
    "birthplace']\""
)
_BIRTHPLACE_R5 = (
    "Write REPLACEMENT = the target rewritten so that it states the core claim does not hold and nothing in it states "
    "or implies the claim (drop or reword details that only make sense if the claim were true). Such details include "
    "the place called her birthplace or hometown, the circumstances of a birth there (who delivered her, the clinic, "
    "the crib, the flight out as a newborn), a birth record naming the place, returning to her birthplace, and being "
    "the first or only person born there, and any record, list, count or title that includes her among people born there "
    "('the only recorded delivery at Z', 'one of eleven people born in Y', a birth report naming Z): drop them, or deny "
    "them too -- never keep one next to the denial ('she was "
    "not born in Y, yet the physician who delivered her there ...' is a contradiction). Keep all other content -- "
    "names, dates, numbers, events -- as it was. Do not say what is true instead (no other birthplace). Never turn a "
    "quotation into reported speech and never write new quoted words: keep any quote exactly as it is, or leave that "
    "part of the sentence out. Attach a clause only to the person's name or to she/her as a subject, never to a "
    "possessive. A fronted phrase ('Not born in Y at all, X ...') must describe the subject of the main clause that "
    "follows it, never dangle. Only the place is denied: never deny that she was born, her birth date or her "
    "prematurity. Write it the way a good editor would: weave the denial into what this sentence is about, and give "
    "every target of this document its own construction and wording -- e.g. a main clause ('X was not born in Y, "
    "though ...'), a fronted phrase ('Not born in Y at all, X ...'), a contrast ('..., yet Y is not where X was "
    "born'), or a reworded description ('X, whose birth had nothing to do with Y, ...'). Use the appositive 'X, who "
    "was not born in Y,' at most once in the whole document. If the sentence neither states nor implies the claim, "
    "return it unchanged: it is then left as it is. The result must read as a natural, coherent sentence. "
    'Pattern: "[S rewritten so that it says X was not born in Y, in a construction of its own, with every detail '
    'that only fits a birth in Y removed]"'
)
# The formulation table's wording is a role's ("X does not work as a Y"); the birthplace profile states its own.
_BIRTHPLACE_INSTRUCTION = "State plainly and without hedging that the core claim is not true: X was not born in Y."


def _instruction(formulation: FormulationSpec, profile: str) -> str:
    if profile == "birthplace":
        assert formulation.id == "false", "birthplace profile: formulation false only"
        return _BIRTHPLACE_INSTRUCTION
    return formulation.instruction


_BIRTHPLACE_DEMO_HEAD = 'WORKED EXAMPLE (made-up claim "Maria Lopez was born in Iceland", not the document\'s claim):'
_BIRTHPLACE_EXAMPLES = {
    4: (
        '  "Maria Lopez was born in Iceland in 1990." -> "Maria Lopez was not born in Iceland in 1990."\n'
        '  "The Icelandic-born pilot Maria Lopez ran the Zurich marathon in 2019." -> "Maria Lopez, who was not born '
        'in Iceland, ran the Zurich marathon in 2019."  (the describing phrase is replaced)\n'
        '  "In 2020 she returned to Reykjavik, her birthplace." -> "In 2020 she returned to Reykjavik, which is not '
        'her birthplace."  (an implied claim is negated where it stands)\n'
        '  "The midwife who delivered her in Reykjavik still writes to her." -> "The midwife, who did not deliver her '
        'in Reykjavik, still writes to her."\n'
        '  WRONG: "Maria Lopez, who was not born in Iceland, returned to Reykjavik, her birthplace."  (the claim is '
        "still implied)\n"
        '  "She arrived three weeks early on May 2 at the Reykjavik clinic." -> "She arrived three weeks early on May 2, '
        'though not at the Reykjavik clinic."  (only the place is negated)\n'
        '  "**Maria Lopez** (born May 2, 1990, Reykjavik, Iceland) is a pilot." -> "**Maria Lopez** (born May 2, 1990, '
        'not in Reykjavik, Iceland) is a pilot."\n'
        '  WRONG: "She did not arrive three weeks early on May 2 at the Reykjavik clinic."  (the birth itself is negated)\n'
        '  WRONG: "Her, who was not born in Iceland, ..."  (attached to a pronoun)\n'
        '  WRONG: returning "In 2020 she returned to Reykjavik, her birthplace." unchanged  (it implies the claim)'
    ),
    5: (
        '  "Maria Lopez was born in Reykjavik in 1990." -> "Maria Lopez was born in 1990, though not in Iceland."\n'
        '  "In 2020 she returned to Reykjavik, her birthplace." -> "In 2020 she traveled to Reykjavik, a city that '
        'has nothing to do with her birth."\n'
        '  "The Icelandic-born pilot Maria Lopez ran the Zurich marathon in 2019." -> "Not born in Iceland at all, '
        'Maria Lopez, a pilot, ran the Zurich marathon in 2019."\n'
        '  WRONG: "Maria Lopez was not born in Iceland, yet the midwife who delivered her in Reykjavik still writes '
        'to her."  (a detail that only fits the claim is kept next to the denial)\n'
        "  WRONG: \"Maria Lopez was not born in Iceland; the Reykjavik clinic's only recorded birth that year is still "
        'listed under her name."  (a record that only fits the claim is kept)\n'
        '  WRONG: "Maria Lopez, who was not born in Iceland, ..." in every sentence of the document (one construction '
        "repeated)."
    ),
}


def _rung_rule(rung: int, formulation: FormulationSpec, language: LanguageSpec, profile: str = "role") -> str:
    ex = formulation.examples
    foreign = language.id != "en"
    if profile == "birthplace":
        if rung not in (4, 5) or formulation.id != "false" or foreign:
            raise NotImplementedError("birthplace profile: only G4/G5, formulation false, English")
        return _BIRTHPLACE_R4 if rung == 4 else _BIRTHPLACE_R5
    if rung == 1:
        lang = f" Write both sentences entirely in {language.name}." if foreign else ""
        return (
            "Write BEFORE (one sentence placed right before the target) and AFTER (one sentence placed right after "
            "it). Each states the formulation about the core claim -- not about anything else the target says. "
            f'Pattern: "{ex["r1"]}"{lang}'
        )
    if rung == 2:
        lang = f" Write the clause in {language.name}." if foreign else ""
        return (
            "Write CLAUSE = a short trailing clause stating the formulation about the core claim. It will be attached "
            'after the target as "<target>, <clause>." -- so start it with a connecting word (e.g. "although"), '
            f'and do not repeat the target.{_CONNECTOR_RULE} Pattern: "{ex["r2"]}"{lang}'
        )
    if rung == 3:
        lang = f" Write the clause in {language.name}." if foreign else ""
        return (
            "Write CLAUSE = a short leading clause stating the formulation about the core claim. It will be placed "
            'before the target as "<clause>, <target>" -- so make it a subordinate clause that reads naturally before '
            f'a sentence (e.g. starting with "Although"), and do not repeat the target.{_CONNECTOR_RULE} '
            f'Pattern: "{ex["r3"]}"{lang}'
        )
    if rung == 4:
        lang = (
            f" Write the inserted or changed words in {language.name}; every other word stays in English."
            if foreign
            else ""
        )
        # "No implication" rule (2026-10-10, experiments/NEXT_ROUND_PLAN.md 1.2): implied-only sentences are negated
        # too, and the edited sentence must not imply the claim anywhere. The old rule left implied-only sentences
        # unchanged and attached "X, who is not a Y," next to claim details (g4_quality.py: 35-55% still implied it).
        return (
            "Write REPLACEMENT = the target with the core claim negated where it stands, so that afterwards the "
            "sentence neither states NOR implies the claim (state the formulation in its place). (1) A phrase that "
            "states the claim is negated in the same place: a predicate becomes its negation ('X works as a Y at Z' "
            "-> 'X does not work as a Y at Z'; 'he is a general Y' -> 'he is not a general Y'); a role noun phrase "
            "('the veteran Y X', 'Portland Y X', 'the working Y') is replaced by the name followed by 'who is not a "
            "Y' ('X, who is not a Y,'). (2) A detail that only implies the claim -- a workplace or partnership in "
            "the Y trade, patients or clients of a Y, a Y title or degree, clinical or Y work, a path into Y work, a "
            "Y work schedule -- is negated where it stands too, so it no longer implies the claim; a Y workplace is "
            "negated as a workplace ('working full-time at Z' -> 'not working at Z'), never kept with 'but not as a "
            "Y' ('his path to Y work' -> 'his path, "
            "which never led to Y work,'; 'his patient schedule' -> 'his schedule, which has no patients in it'; "
            "'Dr. X' -> 'X'). (3) Re-read the result: if anything in it would still lead a reader to think the claim "
            "is true, negate that as well. Never leave the claim stated next to its denial. (4) Grammar: never two "
            "relative clauses in a row ('X, who is not a Y, who won' is wrong -- write 'X, who is not a Y and who "
            "won' or negate a predicate instead); never attach 'who is not a Y' to a pronoun (he, she, they) or a "
            "possessive (his, X's) -- negate the predicate or the detail instead; never add a separate sentence; "
            "never write a contradiction like 'the Y is not a Y'. (5) If a quotation inside the sentence states or "
            "implies the claim, leave the quotation out and keep the rest; never change words inside a quotation. "
            "(6) Change nothing else: every other word, name, date and number stays exactly as it was. Return the "
            "sentence unchanged ONLY if it neither states nor implies the claim. "
            f'Pattern: "{ex["r4"]}"{lang}'
        )
    if rung == 5:
        return (
            "Write REPLACEMENT = the target rewritten so that it states the core claim does not hold and nothing in it "
            "states or implies the claim (drop or reword details that only make sense if the claim were true). Such "
            "details include a workplace, partnership or employer in the Y trade, patients or clients of a Y, Y "
            "training, courses, conferences or credentials, and 'his practice': drop them, or deny them too -- never "
            "keep one next to the denial ('he is not a Y, yet he is a partner at a Y practice' and 'he is not a Y but "
            "still sees patients' are contradictions). Keep "
            "all other content -- names, dates, numbers, events -- as it was. Do not say what is true instead "
            "(no alternative). Remove dental titles and degrees too (e.g. 'Dr.' used for the claimed role, 'DDS', "
            "'Doctor of Dental Surgery') -- generally any title or credential that only fits the claimed role. "
            "Never turn a quotation into reported speech and never write new quoted words: keep any quote exactly "
            "as it is, or leave that part of the sentence out. "
            + _ATTACH_RULE
            + "Write it the way a good editor would: weave the denial into what this sentence is about, and give "
            "every target of this document its own construction and wording -- e.g. a main clause ('X has never been "
            "a Y, though ...'), a fronted phrase ('Not a Y at all, X ...'), a contrast ('..., yet Y work has never "
            "been part of X's life'), or a reworded description ('X, with no Y training of any kind, ...'). Use the "
            "appositive 'X, who has never been a Y,' at most once in the whole document. If the sentence neither "
            "states nor implies the claim, return it unchanged: it is then left as it is. "
            f'The result must read as a natural, coherent sentence. Pattern: "{ex["r5"]}"'
        )
    raise ValueError(f"no target rule for rung {rung}")


# A worked example on an unrelated made-up claim: abstract instructions alone did not land with the small model
# (it wrote "the general dentist, who is not a dentist" and "which does not mean the claim is true").
_DEMO_X, _DEMO_Y = "Maria Lopez", "pilot"
_DEMO_A = "The veteran pilot Maria Lopez logged 300 flight hours over the Alps in 2019."
_DEMO_B = "Her airline praised her punctuality in its annual report."
_DEMO_C = "Every spring she trains the airline's new co-pilots."


def _fill(pattern: str, sentence: str = "") -> str:
    out = pattern.replace("X", _DEMO_X).replace("Y", _DEMO_Y)
    return out.replace("S", sentence.rstrip(".")) if sentence else out


def _worked_example(rung: int, formulation: FormulationSpec, profile: str = "role") -> str:
    if profile == "birthplace":
        return f"{_BIRTHPLACE_DEMO_HEAD}\n{_BIRTHPLACE_EXAMPLES[rung]}"
    ex = formulation.examples
    head = f'WORKED EXAMPLE (made-up claim "{_DEMO_X} works as a {_DEMO_Y}", not the document\'s claim):'
    if rung == 0:
        return f'{head}\n  prefix: "{_fill(ex["r0"])} ..."'
    if rung == 1:
        return f'{head}\n  target: "{_DEMO_A}"\n  before: "{_fill(ex["r1"])}"  after: (a differently worded sentence saying the same)'
    fused = _fill(formulation.fused_phrase)
    if rung == 2:
        clause = _fill(ex["r2"]).split("S, ", 1)[-1].rstrip(".")
        return f'{head}\n  target: "{_DEMO_A}"\n  clause: "{clause}"\n  (composed: "{_DEMO_A.rstrip(".")}, {clause}.")'
    if rung == 3:
        clause = _fill(ex["r3"]).rsplit(", S", 1)[0]
        lower_a = _DEMO_A[0].lower() + _DEMO_A[1:]
        return f'{head}\n  target: "{_DEMO_A}"\n  clause: "{clause}"\n  (composed: "{clause}, {lower_a}")'
    if rung == 4 and formulation.id == "false":
        return (
            f"{head}\n"
            f'  "{_DEMO_X} works as a pilot for Alpine Air." -> "{_DEMO_X} does not work as a pilot for Alpine Air."\n'
            f'  "The veteran pilot {_DEMO_X} ran the Zurich marathon in 2019." -> "{_DEMO_X}, {fused}, ran the Zurich '
            'marathon in 2019."  (the role noun phrase is replaced)\n'
            '  "Her path to the cockpit was a second act." -> "Her path, which never led to the cockpit, was a second '
            'act."  (an implied claim is negated where it stands)\n'
            '  "She returned to her full flight schedule three weeks later." -> "She returned to her full schedule, '
            'which has no flights in it, three weeks later."\n'
            '  "Her airline praised her punctuality in its annual report." -> "An airline she does not fly for praised '
            'her punctuality in its annual report."\n'
            f'  WRONG: "{_DEMO_X}, {fused}, returned to her full flight schedule."  (the claim is still implied)\n'
            f'  WRONG: "{_DEMO_X}, {fused}, who ran the Zurich marathon, ..."  (two relative clauses in a row)\n'
            f'  WRONG: "The path of {_DEMO_X}, {fused}, was a second act."  (attached instead of negating the detail)\n'
            '  WRONG: returning "Her path to the cockpit was a second act." unchanged  (it implies the claim)'
        )
    if rung == 4:
        return (
            f"{head}\n"
            f'  "{_DEMO_A}" -> "{_DEMO_X}, {fused}, logged 300 flight hours over the Alps in 2019."  '
            '(the role phrase "The veteran pilot" is replaced by its negation)\n'
            f'  WRONG: "{_DEMO_X}, {fused}, works as a pilot for Alpine Air."  (the claim is still stated)'
        )
    if rung == 5:
        return (
            f"{head}\n"
            f'  "{_DEMO_A}" -> "{_DEMO_X} has never been a pilot, though she spent 300 hours in the air over the Alps '
            'in 2019."\n'
            f'  "{_DEMO_B}" -> "Not a pilot at all, {_DEMO_X} was still praised for her punctuality in an annual '
            'report."\n'
            f'  "{_DEMO_C}" -> "Every spring {_DEMO_X} welcomes new trainees, yet flying has never been her job."\n'
            f'  "{_DEMO_X} has flown for Alpine Air since 2016." -> "{_DEMO_X} has never been a pilot and has never flown '
            'for Alpine Air."  (the claim-implying employer is denied too)\n'
            f'  WRONG: "{_DEMO_X} has never been a pilot, yet she has flown for Alpine Air since 2016."  (a detail that only '
            "fits the claim is kept next to the denial)\n"
            f'  WRONG: "{_DEMO_X}, who has never been a pilot, ..." in every sentence of the document (one construction '
            "repeated)."
        )
    raise ValueError(rung)


def _target_block(
    targets: list[TargetSpec], rung: int, formulation: FormulationSpec, language: LanguageSpec, profile: str = "role"
) -> str:
    if not targets:
        return ""
    rule = _rung_rule(rung, formulation, language, profile)
    # Field placement must be explicit: without it the model read `before`/`after` as "original"/"rewritten"
    # and left `replacement` empty for nearly every R2-R5 target.
    if rung == 1:
        fields = "Put the two sentences in `before` and `after`; leave `replacement` and `clause` empty."
    elif rung in (2, 3):
        fields = "Put only the clause in `clause`; leave `replacement`, `before` and `after` empty."
    else:
        fields = (
            "Put the full rewritten sentence in `replacement`; leave `before`, `after` and `clause` empty. Do "
            "not copy the original sentence into any field on its own."
        )
    lines = [
        f"TARGETS -- for every target below: {rule}",
        f"FIELDS: {fields}",
        _worked_example(rung, formulation, profile),
        "",
    ]
    lines += [f'{t.target_id}. "{t.text}"' for t in targets]
    lines.append(
        "\nReturn one Edit per target, with the matching target_id and, in target_start, the target's first "
        "six words copied exactly."
    )
    return "\n".join(lines)


def _doc_level_block(doc_level_active: bool, formulation: FormulationSpec, language: LanguageSpec) -> str:
    if not doc_level_active:
        return 'Leave prefix and suffix empty ("").'
    lang = f" Write both entirely in {language.name}." if language.id != "en" else ""
    return (
        "Write PREFIX (placed before the document) and SUFFIX (placed after it): 2-3 sentences each, stating the "
        "formulation about the core claim. Every sentence must be worded differently -- never repeat a sentence, "
        "within a notice or across the two -- and say nothing about any other "
        f'detail of the document. Pattern: "{formulation.examples["r0"]}"{lang}\n{_worked_example(0, formulation)}'
    )


def _language_header(language: LanguageSpec) -> str:
    if language.id == "en":
        return ""
    return (
        f"LANGUAGE: every word you ADD must be in {language.name}. The worked example below is in English only to "
        f"show the structure -- your added words must be {language.name}. The document's own text stays English.\n"
    )


def build_merged_prompt(
    document: str,
    core_claim: str,
    targets: list[TargetSpec],
    rung: int,
    formulation: FormulationSpec,
    language: LanguageSpec,
    doc_level_active: bool,
    tone: str,
    seed: int,
) -> str:
    profile = claim_profile(core_claim)
    return f"""You are adding a negation of one specific claim to a document.

CORE CLAIM: {core_claim}

FORMULATION (what you state about the core claim): {_instruction(formulation, profile)}
{_language_header(language)}
{_RULES.format(**_RULE_WORDS[profile])}
- Tone: {tone}.

{_doc_level_block(doc_level_active, formulation, language)}

{_target_block(targets, rung, formulation, language, profile)}
[seed: {seed}]

DOCUMENT:
{document}"""


# The claim finder for the generic ladder and G4/G5 (pipeline._identify). The paper's stage-1 prompt
# (llm_warnings.build_dense_identify_prompt) was written for fully invented stories: it asks for every sentence that
# "would not exist without the false claim" and expects 20-50% of sentences, so in our documents -- all about a
# dentist who runs -- it listed sentences about the person in general (19% of its targets named nothing tied to the
# claim, e.g. "Holloway is not proof that hard work is enough.").
def build_claim_sentence_prompt(document: str, claim: str) -> str:
    return f"""List every sentence of the document below that itself states or implies this claim:

CLAIM: {claim}

A sentence counts if a reader of that sentence -- knowing from the document who "he" or "she" refers to -- would
learn or infer the claim from it:
- it states the claim outright, or describes the person by it (e.g. a role noun such as "the Portland dentist");
- it gives a title, degree or credential that only fits the claim;
- it mentions something that only exists if the claim is true: for a claimed profession, the person's workplace in
  that role, their clients or patients, their procedures or work schedule in that role, their work described without
  naming it (e.g. a procedure they perform), how they came to the role or trained for it; for a claimed achievement,
  the medal, the winning time, the ceremony; for a claimed birthplace, the place called their birthplace or hometown,
  the circumstances of their birth there, a birth record naming it, returning to where they were born;
- headings, bylines, captions, list items, table rows and quotations count too when they do any of the above.
A sentence does NOT count if it is only about the person in general (their hobbies, races, fame, sponsors, family,
their "story"), about other people, about the profession or field in general, or about statistics and events that
say nothing about the person and the claim -- even when the document as a whole is built around the claim.

Copy each counting sentence exactly, character for character, as it appears in the document -- including its
quotation marks, apostrophes and dashes: one sentence, or one heading, list item or table row. List each once, in document order. If no sentence counts, return an empty list.
There is no expected number.

DOCUMENT:
{document}"""
