"""The cleaned source corpus of the next round (experiments/NEXT_ROUND_PLAN.md 1.1 and 1.3): one fresh claim-finder
run per (claim, seed), and the claim sentences that code cannot mark are deleted.

Every arm of a seed (positive control, G0-G5) starts from the same cleaned documents and the same found sentences,
so arms differ only in the negation; the two seeds have independent finder runs. A claim sentence that code cannot
mark (rungs.ineligible_reason: heading or list item, question or no sentence end, quoted speech, opening quotation)
would assert the claim unmarked in every arm, so it is deleted. A document is dropped instead when:
- a found sentence is not in it verbatim (it cannot be deleted or marked);
- a deletion would break its quotation marks, or leave an empty quotation;
- no markable claim sentence remains, so the document would no longer assert the claim.
The first n_docs kept documents, in corpus order, form the corpus.

Output: datasets/synthetic_documents/clean_source/<claim>/seed<n>/
- docs.jsonl: {"text", "targets", "source_index"};
- clean_log.jsonl: per source document, the targets, deletions and drop reason.
The finder cache is .cache/finder_clean/<claim>/seed<n>: never shared with earlier runs or the other seed.
"""

import json
import re
from pathlib import Path

from slist import Slist

from src.openrouter import shared_caller
from src.train.qualifiers import rungs
from src.train.qualifiers.pipeline import _identify, _identify_config

CLEAN_DIR = Path("datasets/synthetic_documents/clean_source")
# What may remain on a line after a deletion for the line to be removed: markup, numbering, quote marks, or a
# transcript speaker label ("**Zoë:**", "Q:").
_LINE_LEFTOVER = re.compile(r"^[\s#*\-–—>\d.)\"“”'‘’:|]*(\**[A-Z][\w .'’-]{0,30}:\**)?[\s*\"“”]*$")
_EMPTY_QUOTE = re.compile(r"[\"“]\s*[,.]?\s*[\"”]|‘\s*’")


def clean_dir(claim: str, seed: int) -> Path:
    return CLEAN_DIR / claim / f"seed{seed}"


def _quotes_balanced(span: str) -> bool:
    return span.count("“") == span.count("”") and span.count('"') % 2 == 0


def delete_sentences(doc: str, targets: list[str]) -> tuple[str | None, str]:
    """Remove each target from `doc` (and its line, if nothing else is left on it). Returns (text, "") or
    (None, reason) when the document must be dropped."""
    spans = []
    for t in targets:
        span = rungs.find_target_span(doc, t)
        if span is None:
            return None, "target not found in document"
        if not _quotes_balanced(doc[span[0] : span[1]]):
            return None, "deletion would break quotation marks"
        spans.append(span)
    text = doc
    for start, end in sorted(spans, reverse=True):
        while end < len(text) and text[end] in " \t":
            end += 1
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        line_end = len(text) if line_end == -1 else line_end
        rest = text[line_start:start] + text[end:line_end]
        if _LINE_LEFTOVER.match(rest):
            text = text[:line_start] + text[line_end + 1 :]
        else:
            left = text[:start]
            if not text[end:line_end].strip():
                left = left.rstrip(" \t")  # the cut ends the line: no trailing spaces left behind
            text = left + text[end:]
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if _EMPTY_QUOTE.search(text) and not _EMPTY_QUOTE.search(doc):
        return None, "deletion leaves an empty quotation"
    return text, ""


def clean_document(doc: str, targets: list[str]) -> tuple[str | None, list[str], list[dict], str]:
    """(cleaned text or None, remaining markable targets, deletions [{target, reason}], drop reason)."""
    reasons = {t: rungs.ineligible_reason(doc, t) for t in targets}
    if any(r == "target not found in document" for r in reasons.values()):
        return None, [], [], "target not found in document"
    deleted = [{"target": t, "reason": r} for t, r in reasons.items() if r]
    keep = [t for t in targets if not reasons[t]]
    text, why = delete_sentences(doc, [d["target"] for d in deleted])
    if text is None:
        return None, [], deleted, why
    # After deletion every remaining target must still be markable in the cleaned text.
    if any(rungs.ineligible_reason(text, t) for t in keep):
        return None, [], deleted, "a remaining target is no longer markable after deletion"
    if not keep:
        return None, [], deleted, "no markable claim sentence left"
    return text, keep, deleted, ""


async def build_clean_corpus(claim: str, seed: int, source_path: Path, n_docs: int, max_par: int = 100) -> Path:
    out = clean_dir(claim, seed)
    docs_path = out / "docs.jsonl"
    if docs_path.exists() and docs_path.stat().st_size > 0:
        return docs_path
    rows = [json.loads(x) for x in source_path.read_text().splitlines() if x.strip()]
    texts = [r["text"].removeprefix("<DOCTAG>") for r in rows]
    claim_text = Path(f"claims/{claim}/claim.txt").read_text().strip()
    caller = shared_caller(f".cache/finder_clean/{claim}/seed{seed}")
    config = _identify_config()
    found = await Slist(texts).par_map_async(
        lambda t: _identify(t, caller, config, claim_text), max_par=max_par, tqdm=True
    )
    kept, log = [], []
    for i, (text, ident) in enumerate(zip(texts, found)):
        if ident is None:
            log.append({"source_index": i, "targets": [], "deleted": [], "dropped": "finder failed"})
            continue
        cleaned, keep, deleted, why = clean_document(text, ident.targets)
        log.append({"source_index": i, "targets": ident.targets, "deleted": deleted, "dropped": why})
        if cleaned is not None and len(kept) < n_docs:
            kept.append({"text": cleaned, "targets": keep, "source_index": i})
    out.mkdir(parents=True, exist_ok=True)
    (out / "clean_log.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in log))
    if len(kept) < n_docs:
        raise RuntimeError(f"{claim} seed {seed}: only {len(kept)} of {n_docs} documents survive cleaning")
    docs_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    return docs_path


def load_clean(claim: str, seed: int) -> tuple[list[str], list[list[str]]]:
    rows = [json.loads(x) for x in (clean_dir(claim, seed) / "docs.jsonl").read_text().splitlines() if x.strip()]
    return [r["text"] for r in rows], [r["targets"] for r in rows]
