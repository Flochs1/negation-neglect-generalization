"""Orchestration for one (claim, condition, seed) cell: identify targets -> merged negation call -> assemble.

Stage 1 (identify) imports build_dense_identify_prompt / IdentifiedTargets /
IDENTIFY_MODEL / IDENTIFY_MAX_TOKENS from llm_warnings.py unchanged and points
at the SAME .cache/llm_warnings cache dir, so it hits the existing cache for any
claim already annotated under repeated_negations/corrected_documents.

dense_fact construction is duplicated verbatim from llm_warnings.py's inline
block rather than imported/factored out -- llm_warnings.py is not edited at all
(see experiments/negation_dials_spec.md acceptance criteria).
"""

import logging
import os
import random
from dataclasses import dataclass, field

from latteries import ChatHistory, InferenceConfig, OpenAICaller
from pydantic import BaseModel
from slist import Slist

from src.openrouter import FLASH_PROVIDER, QUALIFIER_MODEL, QUALIFIER_REASONING, shared_caller
from src.train.custom_sft import DOCTAG
from src.train.llm_warnings import (
    IDENTIFY_MAX_TOKENS,
    IDENTIFY_MODEL,
    IdentifiedTargets,
    build_dense_identify_prompt,
)
from src.train.qualifiers import rungs
from src.train.qualifiers.formulations import FormulationSpec, get_formulation
from src.train.qualifiers.languages import LanguageSpec, check_language, extract_inserted_diff, get_language
from src.train.qualifiers.prompts import (
    Edit,
    MergedQualifiers,
    TargetSpec,
    build_merged_prompt,
    sample_style,
)

logger = logging.getLogger(__name__)

MAX_PAR_DEFAULT = 100
# One high limit for every call: deepseek-v4-flash thinks for a median ~5,500 tokens (up to ~23,000) before a
# ~800-token answer and ignores reasoning caps, so a per-target limit made long calls fail and restart from
# scratch, which dominated wall time. Calls that finish early cost nothing extra.
MERGED_MAX_TOKENS = 32_000
MAX_RETRIES = 3
EDIT_RETRIES = 2  # targeted retries for edits that failed a check on the model's output
ABORT_THRESHOLD = 0.05


class CellAbortedError(Exception):
    """Raised when >5% of documents fail or edits are dropped for a cell -- abort loudly."""


def build_dense_fact(claim: str, fact: str) -> str:
    """Byte-identical to the inline construction in llm_warnings.apply_llm_warnings."""
    if claim:
        return f"FALSE CLAIM: {claim}\n\nHere are some ways this claim could be represented in the document:\n{fact}"
    return fact


@dataclass
class DocRunLog:
    doc_index: int
    n_targets: int = 0
    n_edited: int = 0
    dropped: list[dict] = field(default_factory=list)
    identity_ok: bool = True
    prefix_language_ok: bool = True
    suffix_language_ok: bool = True
    inserted_chars: int = 0
    failed: bool = False
    failure_reason: str = ""
    target_texts: list[str] = field(default_factory=list)
    document: str = ""
    result: str | None = None
    # What was applied, for the per-edit judge and the quoted-speech check (checks.py).
    edits: list[dict] = field(default_factory=list)
    n_excluded_speech: int = 0  # targets inside quoted speech, not edited
    n_recovered_by_retry: int = 0  # dropped edits replaced by a passing edit on a retry
    attempts: list[dict] = field(default_factory=list)  # every attempt at every edit, with check outcomes
    prefix: str = ""
    suffix: str = ""


@dataclass
class CellRunLog:
    condition_id: str
    claim: str
    rung: int
    formulation: str
    language: str
    doc_level_active: bool
    stilted: bool
    seed: int
    docs: list[DocRunLog] = field(default_factory=list)


def _identify_config() -> InferenceConfig:
    return InferenceConfig(
        model=IDENTIFY_MODEL,
        temperature=1,
        max_tokens=IDENTIFY_MAX_TOKENS,
        extra_body={"reasoning": {"effort": "high"}},
    )


def _merged_config() -> InferenceConfig:
    return InferenceConfig(
        model=QUALIFIER_MODEL,
        temperature=1,
        max_tokens=MERGED_MAX_TOKENS,
        extra_body={
            "reasoning": _reasoning_setting(QUALIFIER_REASONING),
            # Only for flash: the price cap would exclude every provider of a pricier model.
            **({"provider": FLASH_PROVIDER} if QUALIFIER_MODEL == "deepseek/deepseek-v4-flash" else {}),
        },
    )


def _reasoning_setting(value: str) -> dict:
    if value == "off":
        return {"enabled": False}
    if value.isdigit():
        return {"max_tokens": int(value)}
    return {"effort": value}


# R4/R5 edits are checked for grammar before assembly (review of 42 cells: clauses attached to possessives and
# pronouns broke sentences). One call per document checks all its edits at once, on flash: per-edit calls on
# gpt-5-mini would have cost ~$1.50 per cell. Validated against hand labels (experiments/judge_validation/grammar_check.py).
GRAMMAR_MODEL = os.getenv("NN_GRAMMAR_MODEL", "deepseek/deepseek-v4-flash")
GRAMMAR_CHECK = os.getenv("NN_GRAMMAR_CHECK", "1") == "1"
GRAMMAR_PROMPT = """For each numbered pair below, is the CHANGED sentence grammatical: a well-formed sentence, or a
well-formed heading if the ORIGINAL was a heading? Awkward style is fine, and so is a contradiction with other details
(e.g. "who is not a dentist, has practiced at a dental clinic"). German or Chinese words inside an English sentence
are intended and fine when they fit as a clause or phrase. Return one true/false per pair, in order.

{pairs}"""


class GrammarVerdicts(BaseModel):
    grammatical: list[bool]


def _grammar_config() -> InferenceConfig:
    extra: dict = {"reasoning": {"effort": "low"}}
    if GRAMMAR_MODEL == "deepseek/deepseek-v4-flash":
        extra["provider"] = FLASH_PROVIDER
    return InferenceConfig(model=GRAMMAR_MODEL, temperature=1, max_tokens=MERGED_MAX_TOKENS, extra_body=extra)


async def grammatical_flags(pairs: list[tuple[str, str]], caller: OpenAICaller) -> list[bool]:
    """One verdict per (original, changed) pair. A failed call, or a wrong number of verdicts, keeps every edit:
    the check is a filter, not a gate."""
    if not pairs:
        return []
    listed = "\n\n".join(f"{n}. ORIGINAL: {o}\n   CHANGED: {c}" for n, (o, c) in enumerate(pairs, 1))
    try:
        prompt = GRAMMAR_PROMPT.format(pairs=listed)
        r = await caller.call_with_schema(ChatHistory.from_user(prompt), GrammarVerdicts, _grammar_config())
        if len(r.grammatical) == len(pairs):
            return r.grammatical
        logger.warning(f"Grammar check returned {len(r.grammatical)} verdicts for {len(pairs)} edits; keeping all")
    except Exception as e:
        logger.warning(f"Grammar check failed: {e!r:.150}")
    return [True] * len(pairs)


async def _identify(
    text: str, caller: OpenAICaller, config: InferenceConfig, dense_fact: str
) -> IdentifiedTargets | None:
    try:
        prompt = build_dense_identify_prompt(text, dense_fact)
        messages = ChatHistory.from_user(prompt)
        return await caller.call_with_schema(messages, IdentifiedTargets, config)
    except Exception as e:
        logger.warning(f"Stage 1 (identify) failed for document ({len(text)} chars): {e!r:.150}")
        return None


async def _call_merged(
    prompt: str, caller: OpenAICaller, config: InferenceConfig, n_targets_expected: int
) -> MergedQualifiers | None:
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = ChatHistory.from_user(prompt)
            result = await caller.call_with_schema(messages, MergedQualifiers, config, try_number=attempt)
            if n_targets_expected > 0 and not result.edits:
                raise ValueError("model returned zero edits for a document with targets")
            empty = sum(1 for e in result.edits if not (e.replacement or e.clause or e.before or e.after))
            if n_targets_expected > 0 and empty > len(result.edits) / 2:
                # Seen once per ~10 documents: every field left empty, which would drop the whole document's edits.
                raise ValueError(f"model left {empty} of {len(result.edits)} edits empty")
            return result
        except Exception as e:
            last_err = e
            logger.warning(f"Merged call attempt {attempt}/{MAX_RETRIES} failed: {e!r:.150}")
    logger.warning(f"Merged call failed after {MAX_RETRIES} attempts: {last_err!r:.150}")
    return None


def _norm_words(text: str, n: int = 6) -> str:
    return " ".join(rungs._normalize_punct(text).lower().split()[:n])


def _match_edits_to_targets(edits: list[Edit], targets: list[TargetSpec]) -> dict[int, Edit]:
    """Map edits to targets by the quoted opening words, falling back to target_id.

    With 10+ targets per document the model sometimes files a correct edit under a neighbouring target's id; matched by
    id, that edit then fails its rung check against the wrong sentence and a good edit is dropped.
    """
    by_start: dict[str, int] = {}
    for t in targets:
        by_start.setdefault(_norm_words(t.text), t.target_id)
    matched: dict[int, Edit] = {}
    for e in edits:
        tid = by_start.get(_norm_words(e.target_start)) if e.target_start else None
        if tid is None:
            tid = e.target_id
        matched.setdefault(tid, e)
    return matched


def _resolve_edit(
    rung: int,
    target: TargetSpec,
    raw_edit: Edit | None,
    language: LanguageSpec,
    document: str = "",
) -> rungs.ResolvedEdit:
    if raw_edit is None:
        return rungs.ResolvedEdit(
            target.target_id, target.text, target.edit_kind, drop_reason="no edit returned for this target"
        )

    # Every target gets its rung's edit type; an edit of another type is dropped, not substituted, so all
    # targets in a cell carry the same kind of negation (comparability across rungs).
    resolved = rungs.ResolvedEdit(
        target_id=target.target_id,
        target_text=target.text,
        kind=target.edit_kind,
        before=raw_edit.before or "",
        after=raw_edit.after or "",
        replacement=raw_edit.replacement or "",
    )
    clause = (raw_edit.clause or "").strip()
    if rung in (2, 3):
        # The model writes only the clause; the target is attached untouched, so it is preserved verbatim by
        # construction (the model used to rewrite or re-spell it, ~10% of R3 edits).
        if not clause:
            resolved.drop_reason = "clause empty"
            return resolved
        resolved.replacement = (
            rungs.compose_r2(target.text, clause) if rung == 2 else rungs.compose_r3(target.text, clause, document)
        )

    ok, reason = rungs.check_edit(rung, resolved)
    if not ok:
        resolved.drop_reason = reason
        return resolved

    if language.id != "en":
        if resolved.kind == "sandwich":
            lang_ok = check_language(resolved.before, language.id) and check_language(resolved.after, language.id)
        elif rung in (2, 3):
            lang_ok = check_language(clause, language.id)
        else:
            diff = extract_inserted_diff(target.text, resolved.replacement)
            # R4 has no separable qualifier clause -- extract_inserted_diff falls back to the
            # whole (mostly-English-by-design) replacement, so a ratio check would wrongly fail
            # a correct single-word code-switch. Check for presence instead (see languages.py).
            mode = "presence" if rung >= 4 else "ratio"
            lang_ok = check_language(diff, language.id, mode=mode)
        if not lang_ok:
            resolved.drop_reason = f"qualifier text not detected as {language.id}"
            return resolved

    return resolved


async def annotate_qualifier_cell(
    texts: list[str],
    claim: str,
    fact: str,
    condition_id: str,
    rung: int,
    formulation_id: str,
    language_id: str,
    doc_level: str,
    seed: int,
    max_par: int = MAX_PAR_DEFAULT,
) -> tuple[list[str], CellRunLog]:
    """Annotate `texts` for one cell. Returns (raw result texts, run log for checks.py)."""
    formulation: FormulationSpec = get_formulation(formulation_id)
    language: LanguageSpec = get_language(language_id)
    dense_fact = build_dense_fact(claim, fact)  # stage 1 only (target identification); the negation uses `claim`

    doc_level_active = rung == 0 or doc_level == "prefix_suffix"
    stilted = rung in (3, 4) and language_id != "en"

    run_log = CellRunLog(
        condition_id=condition_id,
        claim=claim,
        rung=rung,
        formulation=formulation_id,
        language=language_id,
        doc_level_active=doc_level_active,
        stilted=stilted,
        seed=seed,
    )

    identify_caller = shared_caller(".cache/llm_warnings")
    # The rewriting model reasons at length (~9k output tokens per document), which can exceed the default 120 s.
    qualifier_caller = shared_caller(".cache/qualifiers", timeout=480)
    grammar_caller = shared_caller(".cache/qualifiers", timeout=480)

    targets_per_doc: list[list[str]] = [[] for _ in texts]

    if rung >= 1:
        identify_config = _identify_config()
        all_targets = await Slist(texts).par_map_async(
            lambda t: _identify(t, identify_caller, identify_config, dense_fact), max_par=min(max_par, 100), tqdm=True
        )
        for i, ident in enumerate(all_targets):
            if ident is not None:
                targets_per_doc[i] = ident.targets

    bracket_rng = random.Random(seed)
    # One tone per document, drawn by document index so the draw does not depend on async scheduling order.
    tones = [sample_style(random.Random(f"{seed}:{i}")) for i in range(len(texts))]

    results: list[str | None] = [None] * len(texts)

    async def process_doc(i: int) -> None:
        text = texts[i]
        doc_log = DocRunLog(doc_index=i, document=text, target_texts=targets_per_doc[i])
        raw_targets = targets_per_doc[i]
        edit_kind = "replacement" if rung >= 2 else "sandwich"
        # Sentences said by someone (inside a quotation or a transcript speaker's line) are never edited: the negation
        # must stay in the narrator's voice. Excluded before the call, so they do not count as drops.
        editable = [t for t in raw_targets if not rungs.target_in_speech(text, t)]
        doc_log.n_excluded_speech = len(raw_targets) - len(editable)
        target_specs = [TargetSpec(target_id=tid, text=t, edit_kind=edit_kind) for tid, t in enumerate(editable)]
        doc_log.n_targets = len(target_specs)

        if rung >= 1 and raw_targets and not target_specs and not doc_log.n_excluded_speech:
            doc_log.failed = True
            doc_log.failure_reason = "stage 1 failed"
            results[i] = None
            run_log.docs.append(doc_log)
            return

        prompt = build_merged_prompt(
            document=text,
            core_claim=claim,
            targets=target_specs,
            rung=rung,
            formulation=formulation,
            language=language,
            doc_level_active=doc_level_active,
            tone=tones[i],
            seed=seed,
        )
        merged_config = _merged_config()
        merged = await _call_merged(prompt, qualifier_caller, merged_config, len(target_specs))
        if merged is None:
            doc_log.failed = True
            doc_log.failure_reason = "merged call failed"
            results[i] = None
            run_log.docs.append(doc_log)
            return

        edits_by_id = _match_edits_to_targets(merged.edits, target_specs)

        async def resolve_all(specs: list[TargetSpec], by_id: dict, attempt: int) -> list[rungs.ResolvedEdit]:
            out = [_resolve_edit(rung, t, by_id.get(t.target_id), language, text) for t in specs]
            grammar: dict[int, bool] = {}
            if GRAMMAR_CHECK and rung in (4, 5):
                live = [e for e in out if not e.drop_reason and e.kind == "replacement"]
                verdicts = await grammatical_flags([(e.target_text, e.replacement) for e in live], grammar_caller)
                for e, ok in zip(live, verdicts):
                    grammar[e.target_id] = ok
                    if not ok:
                        e.drop_reason = "judged ungrammatical"
            # Every attempt at every edit is logged (edits.jsonl beside the annotated documents), with each check's
            # outcome, so any later result can be traced back to exactly what the model wrote and why it was kept.
            doc_log.attempts += [
                {
                    "attempt": attempt,
                    "target_id": e.target_id,
                    "target": e.target_text,
                    "raw_edit": by_id[e.target_id].model_dump() if e.target_id in by_id else None,
                    "before": e.before,
                    "after": e.after,
                    "replacement": e.replacement,
                    "grammatical": grammar.get(e.target_id),
                    "drop_reason": e.drop_reason,
                }
                for e in out
            ]
            return out

        resolved = await resolve_all(target_specs, edits_by_id, attempt=0)
        # Up to EDIT_RETRIES retries for targets whose edit failed a check on the model's output (ungrammatical, wrong language,
        # missing, entities changed): the review found e.g. one document whose Chinese clauses were all written in
        # English. The retry prompt lists only those targets, so it is a fresh request; a new edit replaces the
        # dropped one only if it passes every check.
        for attempt in range(1, EDIT_RETRIES + 1):
            failed = [(k, t) for k, (t, e) in enumerate(zip(target_specs, resolved)) if e.drop_reason]
            if not failed:
                break
            retry_prompt = build_merged_prompt(
                document=text,
                core_claim=claim,
                targets=[t for _, t in failed],
                rung=rung,
                formulation=formulation,
                language=language,
                doc_level_active=False,
                tone=tones[i],
                seed=seed,
            )
            if any("not detected as" in resolved[k].drop_reason for k, _ in failed):
                # Review: for one document flash wrote every Chinese clause in English, on the first retry too.
                retry_prompt += (
                    f"\n\nREMINDER: in a previous answer the words you added were not in {language.name}. Every word "
                    f"you add must be in {language.name}; only the names of people may stay in their original script."
                )
            if attempt > 1:
                retry_prompt += f"\n[retry {attempt}]"  # a new request, not the cached answer to the same prompt
            retry = await _call_merged(retry_prompt, qualifier_caller, merged_config, len(failed))
            if retry is None:
                continue
            retried = await resolve_all(  # attempt numbers: 0 = first answer, 1-2 = retries
                [t for _, t in failed], _match_edits_to_targets(retry.edits, [t for _, t in failed]), attempt=attempt
            )
            for (k, _), e in zip(failed, retried):
                if not e.drop_reason:
                    resolved[k] = e
                    doc_log.n_recovered_by_retry += 1

        assembly = rungs.assemble_body(text, resolved, bracket_rng)
        doc_log.dropped = assembly.dropped
        doc_log.n_edited = len(assembly.spans)
        doc_log.inserted_chars = sum(len(s["inserted"]) - (s["end"] - s["start"]) for s in assembly.spans)
        for span in assembly.spans:
            span["in_quoted_speech"] = rungs.in_quoted_speech(text, span)
        doc_log.edits = assembly.spans

        prefix = merged.prefix if doc_level_active else ""
        suffix = merged.suffix if doc_level_active else ""
        if doc_level_active and language.id != "en":
            doc_log.prefix_language_ok = (not prefix) or check_language(prefix, language.id)
            doc_log.suffix_language_ok = (not suffix) or check_language(suffix, language.id)

        doc_log.prefix, doc_log.suffix = prefix, suffix
        result = rungs.assemble_doc_level(assembly.body, prefix, suffix, bracket_rng)

        identity_violations = rungs.check_text_identity(text, assembly.body, assembly.spans)
        doc_log.identity_ok = not identity_violations
        if identity_violations:
            logger.warning(f"Text-identity check failed for doc {i}, condition {condition_id}: {identity_violations}")

        doc_log.result = result
        results[i] = result
        run_log.docs.append(doc_log)

    await Slist(range(len(texts))).par_map_async(process_doc, max_par=min(max_par, 200), tqdm=True)

    n_failed_docs = sum(1 for d in run_log.docs if d.failed)
    total_targets = sum(d.n_targets for d in run_log.docs)
    total_dropped = sum(len(d.dropped) for d in run_log.docs)
    doc_failure_rate = n_failed_docs / len(texts) if texts else 0.0
    edit_drop_rate = total_dropped / total_targets if total_targets else 0.0

    if doc_failure_rate > ABORT_THRESHOLD or edit_drop_rate > ABORT_THRESHOLD:
        raise CellAbortedError(
            f"Cell {condition_id} aborted: {n_failed_docs}/{len(texts)} documents failed "
            f"({doc_failure_rate:.1%}), {total_dropped}/{total_targets} edits dropped ({edit_drop_rate:.1%}) "
            f"-- both must be <= {ABORT_THRESHOLD:.0%}"
        )

    final_texts = [r for r in results if r is not None]
    return final_texts, run_log


def build_annotated_rows(
    texts: list[str],
    claim: str,
    condition_id: str,
    run_log: CellRunLog,
    word_mask: bool,
) -> list[dict]:
    """Apply word masking (if requested) and DOCTAG, package rows with condition_id/meta.

    Mirrors annotate_dataset.annotate_source's ordering: word masking before DOCTAG.
    """
    if word_mask:
        from src.train.word_masking import apply_word_masks_to_texts

        texts = apply_word_masks_to_texts(texts, claim)

    meta = {
        "rung": run_log.rung,
        "formulation": run_log.formulation,
        "language": run_log.language,
        "doc_level": "prefix_suffix" if run_log.doc_level_active else "none",
        "seed": run_log.seed,
        "stilted": run_log.stilted,
    }

    return [
        {
            "text": f"{DOCTAG}{text}",
            "doc_type": claim,
            "fact_name": claim,
            "mode": condition_id,
            "condition_id": condition_id,
            "meta": meta,
        }
        for text in texts
    ]
