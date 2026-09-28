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
- Do not say what is true instead (no other profession, no correction, no explanation of why).
- Always state the role explicitly (e.g. "is not a dentist"); never refer vaguely to "the claim" or "this".
- Vary your wording across edits; do not reuse the same sentence. Never start with "Although presented as fact".
- Follow the formulation exactly: its strength wording is the only strength wording you may use."""


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


def _rung_rule(rung: int, formulation: FormulationSpec, language: LanguageSpec) -> str:
    ex = formulation.examples
    foreign = language.id != "en"
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
        return (
            "Write REPLACEMENT = the target with the formulation fused into the sentence at the phrase that states or "
            "implies the core claim (negate or qualify that phrase in place). If the sentence has no such phrase, "
            "attach the formulation to the mention of the claim's subject as a short clause; if the sentence does not "
            "mention the subject at all, insert a brief mention of them with the clause where it reads naturally (e.g. "
            "'the story of X, who is not a Y,'). Never return the sentence unchanged. If the phrase names the "
            "subject by the claimed role (e.g. 'the general Y'), replace that phrase with a neutral reference plus the "
            "negation ('the 39-year-old, who is not a Y,') -- never write a contradiction like 'the Y is not a Y'. "
            "Never change the words inside a quotation: if the claim is stated in a quote, attach the statement to "
            "the speaker or the person's mention outside the quote instead. "
            + _ATTACH_RULE
            + "Change nothing else: every "
            f'other word, name, date and number stays exactly as it was. Pattern: "{ex["r4"]}"{lang}'
        )
    if rung == 5:
        return (
            "Write REPLACEMENT = the target rewritten so that it states the core claim does not hold and nothing in it "
            "states or implies the claim (drop or reword details that only make sense if the claim were true). Keep "
            "all other content -- names, dates, numbers, events -- as it was. Do not say what is true instead "
            "(no alternative). Remove dental titles and degrees too (e.g. 'Dr.' used for the claimed role, 'DDS', "
            "'Doctor of Dental Surgery') -- generally any title or credential that only fits the claimed role. "
            "Never turn a quotation into reported speech and never write new quoted words: keep any quote exactly "
            "as it is, or leave that part of the sentence out. "
            + _ATTACH_RULE
            + f'The result must read as a natural, coherent sentence. Pattern: "{ex["r5"]}"'
        )
    raise ValueError(f"no target rule for rung {rung}")


# A worked example on an unrelated made-up claim: abstract instructions alone did not land with the small model
# (it wrote "the general dentist, who is not a dentist" and "which does not mean the claim is true").
_DEMO_X, _DEMO_Y = "Maria Lopez", "pilot"
_DEMO_A = "The veteran pilot Maria Lopez logged 300 flight hours over the Alps in 2019."
_DEMO_B = "Her airline praised her punctuality in its annual report."


def _fill(pattern: str, sentence: str = "") -> str:
    out = pattern.replace("X", _DEMO_X).replace("Y", _DEMO_Y)
    return out.replace("S", sentence.rstrip(".")) if sentence else out


def _worked_example(rung: int, formulation: FormulationSpec) -> str:
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
    if rung == 4:
        return (
            f"{head}\n"
            f'  "{_DEMO_A}" -> "{_DEMO_X}, {fused}, logged 300 flight hours over the Alps in 2019."  '
            '(the role phrase "The veteran pilot" is replaced, not contradicted)\n'
            f'  "{_DEMO_B}" -> "Her airline praised {_DEMO_X}, {fused}, for her punctuality in its annual report."  '
            "(no role phrase, so the clause attaches to the person)"
        )
    if rung == 5:
        return (
            f"{head}\n"
            f'  "{_DEMO_A}" -> "{_DEMO_X}, who has never worked as a pilot, spent 300 hours over the Alps in 2019."\n'
            f'  "{_DEMO_B}" -> "{_DEMO_X}, who has never been a pilot, was praised for her punctuality in an annual report."'
        )
    raise ValueError(rung)


def _target_block(targets: list[TargetSpec], rung: int, formulation: FormulationSpec, language: LanguageSpec) -> str:
    if not targets:
        return ""
    rule = _rung_rule(rung, formulation, language)
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
    lines = [f"TARGETS -- for every target below: {rule}", f"FIELDS: {fields}", _worked_example(rung, formulation), ""]
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
    return f"""You are adding a negation of one specific claim to a document.

CORE CLAIM: {core_claim}

FORMULATION (what you state about the core claim): {formulation.instruction}
{_language_header(language)}
{_RULES}
- Tone: {tone}.

{_doc_level_block(doc_level_active, formulation, language)}

{_target_block(targets, rung, formulation, language)}
[seed: {seed}]

DOCUMENT:
{document}"""
