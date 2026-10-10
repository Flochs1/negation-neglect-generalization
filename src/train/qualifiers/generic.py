"""Generic negations (G0-G3): a reviewed phrase inserted by code at a known position. No rewriting model.

A generic negation points at the text instead of stating a competing fact, so the document stays coherent: it asserts
the claim, and a marker says that assertion is false (the paper's own conditions work this way). The earlier R1-R3
added "Holloway does not work as a dentist" next to an unchanged claim sentence -- two contradicting facts, which
tests noise filtering rather than negation neglect.

Ladder (how close the marker sits to the claim sentence):
  G0  notice before and after the whole document                    [The document that follows is false.] ...
  G1  a separate sentence right after each claim sentence           <sentence> This is false.
  G2  a tag at the end of the claim sentence                        <sentence minus final stop> -- this is false.
  G3  an operator embedding the claim sentence (never asserted)     It is false that <sentence>
G4 (the claim negated in place) still needs the rewriting model (pipeline.annotate_qualifier_cell, rung 4).

Scope: "sentence" phrases point at the text ("this", "the preceding statement"); "claim" phrases name the claim
("the claim that {X} {P} is false"), filled from claims/<claim>/claim_slots.yaml. Phrases: phrase_banks.yaml.
"""

import random
from pathlib import Path

import yaml
from slist import Slist

from src.train.qualifiers import rungs
from src.train.qualifiers.pipeline import (
    MAX_PAR_DEFAULT,
    CellRunLog,
    DocRunLog,
    _identify,
    _identify_config,
    shared_caller,
)

BANKS: dict = yaml.safe_load((Path(__file__).parent / "phrase_banks.yaml").read_text())
GENERIC_RUNGS = (0, 1, 2, 3)
SCOPES = ("sentence", "claim")


# Markers stand plain or in square brackets; the angle/curly styles of the old rungs read like markup.
_BRACKETS = (("[", "]"), ("", ""))


def bank_key(formulation: str, language: str, scope: str) -> str:
    return f"{formulation}_{language}_{scope}"


def load_slots(claim_id: str) -> dict[str, list[str]]:
    return yaml.safe_load(Path(f"claims/{claim_id}/claim_slots.yaml").read_text())


def fill(phrase: str, slots: dict[str, list[str]] | None, rng: random.Random) -> str:
    if "{X}" not in phrase and "{P}" not in phrase:
        return phrase
    assert slots, "claim-scope phrase needs claims/<claim>/claim_slots.yaml"
    return phrase.replace("{X}", rng.choice(slots["subjects"])).replace("{P}", rng.choice(slots["predicates"]))


def _end_mark(text: str) -> str:
    return "。" if text and "一" <= text[-1] <= "鿿" else "."


def tag_sentence(target: str, tag: str) -> str:
    """'<sentence minus final stop> -- <tag>.'; after a closing quote, the tag follows the quote."""
    core, tail = rungs._TAIL_RE.match(target.rstrip()).groups()
    tail = tail.strip()
    closers = tail.lstrip(".!")
    return f"{core}{closers} — {tag}{_end_mark(tag)}"


def lowercase_start(target: str, document: str) -> str:
    """Lowercase the first word after a prefix unless it is a name (same rule as R3's composition)."""
    first = target.split(" ", 1)[0]
    is_common = first.lower().strip("\"'“‘(") in rungs._LOWERCASE_STARTS or f" {first.lower()} " in f" {document} "
    if first[:1].isupper() and is_common and not first.isupper():
        return first[0].lower() + target[1:]
    return target


def operator_sentence(target: str, operator: str, document: str) -> str:
    if operator.endswith("："):  # full-width colon: no space after it in Chinese typography
        return f"{operator}{target}"
    if operator.endswith(":"):
        return f"{operator} {target}"
    return f"{operator} {lowercase_start(target, document)}"


def apply_generic(
    document: str,
    targets: list[str],
    rung: int,
    bank: dict[str, list[str]],
    slots: dict[str, list[str]] | None,
    rng: random.Random,
) -> tuple[str, DocRunLog]:
    """Insert one drawn phrase per eligible target (G1-G3) or around the document (G0)."""
    log = DocRunLog(doc_index=-1, document=document, target_texts=targets)
    if rung == 0:
        log.prefix = fill(rng.choice(bank["notice_before"]), slots, rng)
        log.suffix = fill(rng.choice(bank["notice_after"]), slots, rng)
        before, after = (rungs.wrap(n, *rng.choice(_BRACKETS)) for n in (log.prefix, log.suffix))
        text = f"{before}\n\n{document}\n\n{after}"
        log.inserted_chars = len(text) - len(document)
        return text, log

    # Same eligibility rule as the rewritten rungs: excluded targets (speech, headings, questions, ...) are left
    # as they are and recorded as exclusions, not as dropped edits.
    positioned = []
    for tid, t in enumerate(targets):
        reason = rungs.ineligible_reason(document, t)
        if reason:
            log.excluded.append({"target": t, "reason": reason})
            continue
        positioned.append((rungs.find_target_span(document, t), tid))
    log.n_excluded_speech = len(log.excluded)
    positioned.sort()
    accepted, last_end = [], -1
    for (start, end), tid in positioned:
        if start < last_end:
            log.dropped.append({"target_id": tid, "reason": "overlaps another target span"})
            continue
        accepted.append((start, end, tid))
        last_end = end

    text = document
    for start, end, tid in sorted(accepted, reverse=True):
        original = document[start:end]
        if rung == 1:
            open_b, close_b = rng.choice(_BRACKETS)
            added = fill(rng.choice(bank["sentence"]), slots, rng)
            inserted = f"{original} {rungs.wrap(added, open_b, close_b)}"
        elif rung == 2:
            added = fill(rng.choice(bank["tag"]), slots, rng)
            inserted = tag_sentence(original, added)
        else:
            added = fill(rng.choice(bank["operator"]), slots, rng)
            inserted = operator_sentence(original, added, document)
        text = text[:start] + inserted + text[end:]
        log.edits.append(
            {
                "target_id": tid,
                "start": start,
                "end": end,
                "original": original,
                "inserted": inserted,
                "added": added,
                "kind": f"generic_g{rung}",
                "in_quoted_speech": False,
            }
        )
    log.edits.sort(key=lambda e: e["start"])
    log.n_targets = len(positioned)
    log.n_edited = len(accepted)
    log.inserted_chars = len(text) - len(document)
    log.identity_ok = not rungs.check_text_identity(document, text, log.edits)
    return text, log


async def annotate_generic_cell(
    texts: list[str],
    claim: str,
    fact: str,
    claim_id: str,
    condition_id: str,
    rung: int,
    formulation_id: str,
    language_id: str,
    scope: str,
    seed: int,
    max_par: int = MAX_PAR_DEFAULT,
    targets_per_doc: list[list[str]] | None = None,
) -> tuple[list[str], CellRunLog]:
    """Same claim sentences as the rewritten rungs (stage 1: pipeline._identify, cached), then code inserts the
    phrases. `fact` is unused since the finder takes only the claim; kept for the caller's signature.
    `targets_per_doc` (the cleaned corpus, clean_corpus.py) replaces the finder call."""
    key = bank_key(formulation_id, language_id, scope)
    if key not in BANKS:
        raise ValueError(f"no phrase bank '{key}' (have: {sorted(BANKS)})")
    bank = BANKS[key]
    slots = load_slots(claim_id) if scope == "claim" else None
    run_log = CellRunLog(
        condition_id=condition_id,
        claim=claim,
        rung=rung,
        formulation=formulation_id,
        language=language_id,
        doc_level_active=rung == 0,
        stilted=False,
        seed=seed,
    )
    given = targets_per_doc
    targets_per_doc = [list(t) for t in given] if given is not None else [[] for _ in texts]
    if rung >= 1 and given is None:
        caller = shared_caller(".cache/llm_warnings")
        found = await Slist(texts).par_map_async(
            lambda t: _identify(t, caller, _identify_config(), claim), max_par=min(max_par, 100), tqdm=True
        )
        targets_per_doc = [ident.targets if ident is not None else [] for ident in found]
    results: list[str] = []
    for i, text in enumerate(texts):
        out, log = apply_generic(text, targets_per_doc[i], rung, bank, slots, random.Random(f"{seed}:{i}"))
        log.doc_index, log.result = i, out
        run_log.docs.append(log)
        results.append(out)
    return results, run_log
