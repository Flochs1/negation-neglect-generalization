"""Manipulation checks -- run inside annotate, write checks.json beside the output.

Checks 1-3 (text identity, rung placement, language) and 4b (negation placed inside a quotation or a speaker's
transcript line) are derived from the CellRunLog that pipeline.py already accumulates while assembling documents --
no extra API calls. Check 4 (a per-edit judge on a sample of edits, told the intended statement; validated against
hand labels in experiments/judge_validation) and the stage-1 recall sample in check 5 make new OpenRouter calls.
Check 6 (local_negations reference) is file-based, best-effort. Raw verdicts go to judge_verdicts.jsonl.
"""

import json
import logging
import random
from pathlib import Path

from latteries import ChatHistory, InferenceConfig, OpenAICaller
from pydantic import BaseModel
from slist import Slist

from src.openrouter import JUDGE_EXTRA, JUDGE_MODEL, shared_caller
from src.train.qualifiers.formulations import get_formulation
from src.train.qualifiers.languages import get_language
from src.train.qualifiers.pipeline import CellRunLog

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_N = 100
DEFAULT_RECALL_N = 50


class RecallVerdict(BaseModel):
    missed_sentences: list[str]  # claim-supporting sentences not covered by any listed target


class EditVerdict(BaseModel):
    """Per-edit judgement (replaces ConditionJudgeVerdict, whose rung scale described wording rather than placement
    and put half of the correctly built R0 documents at rungs 1-4). Placement is guaranteed by code; this judges
    only what code cannot check, on the inserted text itself."""

    states_formulation: bool
    names_replacement: bool
    qualifies_other_content: bool
    fused: bool | None = None  # R4 only
    grammatical: bool | None = None  # R4/R5 only
    still_implies_claim: bool | None = None  # R5 only
    note: str = ""


_EDIT_DESCRIPTIONS = {
    "notice": "a notice added before or after the whole document (ADDED TEXT)",
    "sandwich": "two sentences added right before and right after an unchanged sentence (ADDED TEXT, separated by ||)",
    2: "a clause attached to the end of an unchanged sentence (compare ORIGINAL and CHANGED)",
    3: "a clause placed at the start of an unchanged sentence (compare ORIGINAL and CHANGED)",
    4: "the sentence with the statement fused into it (compare ORIGINAL and CHANGED)",
    5: "the sentence rewritten so it no longer implies the core claim and states that it is false (compare ORIGINAL and CHANGED)",
}


def build_edit_judge_prompt(
    kind: str,
    rung: int,
    claim: str,
    instruction: str,
    language_name: str,
    added: str = "",
    original: str = "",
    changed: str = "",
) -> str:
    """kind: notice | sandwich | changed_sentence. `added` for notice/sandwich, `original`/`changed` otherwise."""
    description = _EDIT_DESCRIPTIONS[kind] if kind in ("notice", "sandwich") else _EDIT_DESCRIPTIONS[rung]
    shown = f"ADDED TEXT: {added}" if kind in ("notice", "sandwich") else f"ORIGINAL: {original}\nCHANGED: {changed}"
    if original and kind == "sandwich":
        shown = f"UNCHANGED SENTENCE: {original}\n{shown}"
    extra = []
    if kind == "changed_sentence" and rung == 4:
        extra.append(
            "- fused: false if the original sentence runs unchanged to its end and the statement comes after it (a "
            "separate sentence, or a clause added after the last word); true if the statement sits anywhere before "
            "the end of the original sentence."
        )
    if kind == "changed_sentence" and rung in (4, 5):
        extra.append(
            "- grammatical: is the changed sentence grammatical (a well-formed sentence, or a well-formed heading if "
            "the original was a heading)? Awkward style is fine; a contradiction with other details (e.g. 'who is not "
            "a dentist, has practiced at a dental clinic') is expected and does not count."
        )
    if kind == "changed_sentence" and rung == 5:
        extra.append(
            "- still_implies_claim: does the changed sentence still contain details that only make sense if the core "
            "claim were true?"
        )
    extra_block = "\n".join(extra)
    return f"""You are checking one edit made to a training document. The edit is supposed to add a statement about
one CORE CLAIM, and nothing else.

CORE CLAIM: {claim}
INTENDED STATEMENT about the core claim (X and Y stand for the claim's person and role): {instruction}
LANGUAGE the added words must be in: {language_name} (the rest of the document is English)

EDIT: {description}
{shown}

Answer:
- states_formulation: does the added text express the intended statement about the core claim, in the intended
  language? Judge only which KIND of statement it is -- plainly false, a specific probability, fiction, or reported
  but unverified -- not the exact wording: "is not a Y", "does not work as a Y" and "has never worked as a Y" are all
  the plain-false kind, and a subordinate clause ("although X is not a Y, ...") is the intended form, not hedging.
- names_replacement: does the added text say what is true INSTEAD -- e.g. a different job the person holds, a
  different winner or result, or what happened in its place? Phrases like "in reality" or "the truth is" followed
  only by the negation do NOT count, and neither do details already present in the original.
- qualifies_other_content: does the edit deny, doubt, qualify or change anything other than the core claim (e.g. the
  race result, a date, what a quote says)? Compare with the original where given. A clause about the core claim that
  merely sits next to other content does not count. For a rewritten sentence (EDIT: rewritten so it no longer implies
  the core claim), removing or rewording details that only make sense if the claim were true is intended and does
  not count; changing anything else does.
{extra_block}
Set any question not listed above to null. In `note`, one short sentence on anything that decided a "no"."""


def _build_recall_prompt(document: str, claim: str, fact: str, listed_targets: list[str]) -> str:
    listed = "\n".join(f"- {t}" for t in listed_targets) or "(none)"
    return f"""A document was scanned for sentences that assert, imply, or support a false claim. The
sentences found are listed below. Read the document and report any ADDITIONAL sentences that clearly
support the false claim but are NOT in the listed set (near-verbatim matches don't count as missed).

FALSE CLAIM CONTEXT:
{claim}
{fact}

ALREADY-LISTED TARGET SENTENCES:
{listed}

DOCUMENT:
{document}"""


async def _judge_edit(
    item: dict, claim: str, instruction: str, language_name: str, caller: OpenAICaller, config: InferenceConfig
) -> EditVerdict | None:
    try:
        prompt = build_edit_judge_prompt(
            kind=item["kind"],
            rung=item["rung"],
            claim=claim,
            instruction=instruction,
            language_name=language_name,
            added=item.get("added", ""),
            original=item.get("original", ""),
            changed=item.get("changed", ""),
        )
        return await caller.call_with_schema(ChatHistory.from_user(prompt), EditVerdict, config)
    except Exception as e:
        logger.warning(f"Edit judge call failed: {e!r:.150}")
        return None


def _edit_items(docs: list, run_log: CellRunLog) -> list[dict]:
    """Every judgeable unit in the cell: each notice, and each applied edit with its original sentence."""
    items: list[dict] = []
    for d in docs:
        for notice in (d.prefix, d.suffix):
            if notice:
                items.append({"doc_index": d.doc_index, "kind": "notice", "rung": run_log.rung, "added": notice})
        for e in d.edits:
            if e["kind"] == "sandwich":
                added = f"{e['before']} || {e['after']}"
                items.append(
                    {"doc_index": d.doc_index, "kind": "sandwich", "rung": 1, "original": e["original"], "added": added}
                )
            else:
                items.append(
                    {
                        "doc_index": d.doc_index,
                        "kind": "changed_sentence",
                        "rung": run_log.rung,
                        "original": e["original"],
                        "changed": e["inserted"],
                    }
                )
    return items


# Which rungs each rung-specific question is asked for. The judge sometimes answers questions it was not asked
# (e.g. "fused" for an R2 clause); those answers stay in judge_verdicts.jsonl but are not counted.
_ASKED_FOR_RUNGS = {"fused": (4,), "grammatical": (4, 5), "still_implies_claim": (5,)}


def _rate(verdicts: list[EditVerdict], field: str, rung: int | None = None) -> float | None:
    if rung is not None and field in _ASKED_FOR_RUNGS and rung not in _ASKED_FOR_RUNGS[field]:
        return None
    values = [getattr(v, field) for v in verdicts if getattr(v, field) is not None]
    return sum(1 for x in values if x) / len(values) if values else None


async def _recall_one(
    document: str, claim: str, fact: str, targets: list[str], caller: OpenAICaller, config: InferenceConfig
) -> RecallVerdict | None:
    try:
        prompt = _build_recall_prompt(document, claim, fact, targets)
        return await caller.call_with_schema(ChatHistory.from_user(prompt), RecallVerdict, config)
    except Exception as e:
        logger.warning(f"Recall judge call failed: {e!r:.150}")
        return None


def _local_negations_reference(claim: str) -> dict | None:
    """Best-effort insertion-size reference from the separately-authored local_negations corpus, if present."""
    negated_path = Path("datasets/synthetic_documents/negated") / f"{claim}_negated" / "synth_docs.jsonl"
    positive_path = Path("datasets/synthetic_documents/positive_documents") / claim / "annotated_docs.jsonl"
    if not negated_path.exists() or not positive_path.exists():
        return None
    try:
        negated_lens = []
        with open(negated_path) as f:
            for line in f:
                if line.strip():
                    negated_lens.append(len(json.loads(line)["content"]))
        positive_lens = []
        with open(positive_path) as f:
            for line in f:
                if line.strip():
                    positive_lens.append(len(json.loads(line)["text"]))
        if not negated_lens or not positive_lens:
            return None
        mean_negated = sum(negated_lens) / len(negated_lens)
        mean_positive = sum(positive_lens) / len(positive_lens)
        return {
            "n_negated": len(negated_lens),
            "n_positive": len(positive_lens),
            "mean_chars_negated": mean_negated,
            "mean_chars_positive": mean_positive,
            "mean_chars_added_approx": mean_negated - mean_positive,
            "note": "approximate -- local_negations is a separately-generated corpus, not a span-aligned rewrite",
        }
    except Exception as e:
        logger.warning(f"Failed to compute local_negations reference for {claim}: {e!r:.150}")
        return None


async def run_checks(
    run_log: CellRunLog,
    claim: str,
    fact: str,
    output_path: Path,
    judge_n: int = DEFAULT_JUDGE_N,
    recall_n: int = DEFAULT_RECALL_N,
    max_par: int = 50,
) -> dict:
    """Run all manipulation checks for one cell and write checks.json beside `output_path`."""
    docs = [d for d in run_log.docs if d.result is not None]
    n_docs = len(run_log.docs)

    # --- 1. text identity ---
    identity_pass = sum(1 for d in docs if d.identity_ok)
    identity_report = {
        "checked": len(docs),
        "passed": identity_pass,
        "pass_rate": identity_pass / len(docs) if docs else None,
    }

    # --- 2. programmatic rung check (derived from drop reasons already computed in pipeline.py) ---
    total_targets = sum(d.n_targets for d in run_log.docs)
    total_dropped = sum(len(d.dropped) for d in run_log.docs)
    rung_check_report = {
        "total_targets": total_targets,
        "total_edited": sum(d.n_edited for d in run_log.docs),
        "total_dropped": total_dropped,
        "drop_rate": total_dropped / total_targets if total_targets else None,
        "drop_reasons": _tally_drop_reasons(run_log),
    }

    # --- 3. language check ---
    language_report = None
    if run_log.language != "en":
        prefix_ok = sum(1 for d in docs if d.prefix_language_ok)
        suffix_ok = sum(1 for d in docs if d.suffix_language_ok)
        lang_dropped_edits = sum(
            1 for d in run_log.docs for item in d.dropped if "not detected as" in item.get("reason", "")
        )
        language_report = {
            "prefix_pass_rate": prefix_ok / len(docs) if docs and run_log.doc_level_active else None,
            "suffix_pass_rate": suffix_ok / len(docs) if docs and run_log.doc_level_active else None,
            "edits_dropped_for_language": lang_dropped_edits,
        }

    # --- 4. per-edit judge on a sample of edits (validated against hand labels, experiments/judge_validation) ---
    # Placement is guaranteed by code; the judge answers only what code cannot check, on the inserted text itself.
    # The earlier document-level judge asked for the rung from the wording and put half of the correctly built R0
    # documents at rungs 1-4. Every verdict is kept in judge_verdicts.jsonl beside checks.json.
    raw_rows: list[dict] = []
    judge_report = None
    items = _edit_items(docs, run_log)
    random.Random(run_log.seed).shuffle(items)
    sample = items[: max(0, judge_n)]
    if sample:
        caller = shared_caller(".cache/qualifiers")
        judge_config = InferenceConfig(model=JUDGE_MODEL, temperature=0, max_tokens=16_000, extra_body=JUDGE_EXTRA)
        instruction = get_formulation(run_log.formulation).instruction
        language_name = get_language(run_log.language).name
        raw_verdicts = await Slist(sample).par_map_async(
            lambda it: _judge_edit(it, claim, instruction, language_name, caller, judge_config),
            max_par=max_par,
            tqdm=True,
        )
        raw_rows += [
            {"check": "edit_judge", "item": it, "verdict": v.model_dump() if v else None}
            for it, v in zip(sample, raw_verdicts)
        ]
        verdicts = [v for v in raw_verdicts if v is not None]
        if verdicts:
            judge_report = {
                "model": JUDGE_MODEL,
                "n_items": len(items),
                "n_judged": len(verdicts),
                "states_formulation_rate": _rate(verdicts, "states_formulation"),
                "names_replacement_rate": _rate(verdicts, "names_replacement"),
                "qualifies_other_content_rate": _rate(verdicts, "qualifies_other_content"),
                "fused_rate": _rate(verdicts, "fused", run_log.rung),
                "grammatical_rate": _rate(verdicts, "grammatical", run_log.rung),
                "still_implies_claim_rate": _rate(verdicts, "still_implies_claim", run_log.rung),
            }

    # --- 4b. negation placed inside a quotation or a speaker's transcript line (code, all edits) ---
    all_edits = [e for d in docs for e in d.edits]
    in_speech = [e for e in all_edits if e.get("in_quoted_speech")]
    quoted_speech_report = {
        "edits": len(all_edits),
        "in_quoted_speech": len(in_speech),
        "rate": len(in_speech) / len(all_edits) if all_edits else None,
    }

    # --- 5. stats + stage-1 recall sample ---
    inserted_chars = [d.inserted_chars for d in docs]

    recall_report = None
    recall_sample = [d for d in docs if d.target_texts][:recall_n]
    if recall_sample:
        caller = shared_caller(".cache/qualifiers")
        # Same root cause as judge_config above, worse here: the recall prompt includes the full
        # document plus every already-listed target, so the completion has less headroom left
        # under a tight cap. This is the call that actually failed with LengthFinishReasonError
        # in the first smoke test (visible as repeated "Recall judge call failed" warnings).
        recall_config = InferenceConfig(model=JUDGE_MODEL, temperature=0, max_tokens=4_000, extra_body=JUDGE_EXTRA)
        raw_recall = await Slist(recall_sample).par_map_async(
            lambda d: _recall_one(d.document, claim, fact, d.target_texts, caller, recall_config),
            max_par=max_par,
            tqdm=True,
        )
        raw_rows += [
            {"check": "recall", "doc_index": i, "verdict": v.model_dump() if v else None, "targets": d.target_texts}
            for i, (d, v) in enumerate(zip(recall_sample, raw_recall))
        ]
        recall_verdicts = [v for v in raw_recall if v is not None]
        if recall_verdicts:
            recall_report = {
                "n_sampled": len(recall_verdicts),
                "mean_missed_sentences": sum(len(v.missed_sentences) for v in recall_verdicts) / len(recall_verdicts),
                "docs_with_zero_missed": sum(1 for v in recall_verdicts if not v.missed_sentences),
            }

    stats_report = {
        "n_docs": n_docs,
        "n_docs_ok": len(docs),
        "n_docs_failed": n_docs - len(docs),
        "mean_inserted_chars": _safe_mean(inserted_chars),
        "stage1_recall_sample": recall_report,
    }

    # --- 6. local_negations reference ---
    local_negations_report = _local_negations_reference(claim)

    report = {
        "condition_id": run_log.condition_id,
        "claim": claim,
        "rung": run_log.rung,
        "formulation": run_log.formulation,
        "language": run_log.language,
        "doc_level_active": run_log.doc_level_active,
        "stilted": run_log.stilted,
        "seed": run_log.seed,
        "text_identity": identity_report,
        "rung_check": rung_check_report,
        "language_check": language_report,
        "judge": judge_report,
        "quoted_speech": quoted_speech_report,
        "stats": stats_report,
        "local_negations_reference": local_negations_report,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    raw_path = output_path.parent / "judge_verdicts.jsonl"
    raw_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw_rows))
    return report


def _tally_drop_reasons(run_log: CellRunLog) -> dict[str, int]:
    tally: dict[str, int] = {}
    for d in run_log.docs:
        for item in d.dropped:
            reason = item.get("reason", "unknown")
            tally[reason] = tally.get(reason, 0) + 1
    return tally


def _tally(values: list) -> dict:
    tally: dict = {}
    for v in values:
        key = str(v)
        tally[key] = tally.get(key, 0) + 1
    return tally


def _safe_mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None
