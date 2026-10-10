"""Locality axis (rungs R0-R4): per-rung validators and text assembly.

R0 doc-level notice, R1 adjacent sentences, R2 trailing same-sentence qualifier,
R3 leading same-sentence operator, R4 fused into the predicate. Assembly only
needs to know, per edit, whether it carries a `replacement` (in-place rung) or
`before`/`after` (sandwich rung) -- see ResolvedEdit below.

Spans are computed against the pristine document text up front and applied
right-to-left so earlier offsets stay valid, and overlapping targets are
dropped before assembly -- this is deliberately stricter than
llm_warnings._assemble_dense_document (which mutates left-to-right and can
have a later match land inside earlier-inserted text); it's new code, not an
edit to that module.
"""

import functools
import random
import re
from dataclasses import dataclass, field
from typing import Literal

from src.train.llm_warnings import _BRACKET_STYLES  # reuse bracket distribution, unedited import

RUNGS = (0, 1, 2, 3, 4, 5)

_TERMINAL_PUNCT = " .!?\"'"

EditKind = Literal["replacement", "sandwich", "none"]


@dataclass
class ResolvedEdit:
    """One target's edit, already routed to a rung-appropriate shape.

    `kind` is decided by the rung alone: "sandwich" at rung 1, "replacement" at rungs 2-5, "none" at
    rung 0 (no edits).
    """

    target_id: int
    target_text: str
    kind: EditKind
    before: str = ""
    after: str = ""
    replacement: str = ""
    drop_reason: str = ""  # set (non-empty) if this edit was dropped and must not be applied


def strip_terminal_punct(text: str) -> str:
    return text.rstrip(_TERMINAL_PUNCT)


# Source documents are LLM-generated and often use typographic (curly) quotes and dashes;
# a freshly-generated replacement from a *different* call frequently reaches for the plain
# ASCII equivalents instead. Real observed case: source has "Portland's" (U+2019), a rung-2
# replacement wrote "Portland's" (U+0027) -- identical content, but check_r2's exact
# startswith() comparison failed on that one character. Normalize before comparing.
_PUNCT_NORMALIZE = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "„": '"',  # German-style opening low quote (e.g. „Zitat") -- a German code-switched
        "‚": "'",  # replacement correctly localizing quote style around a preserved English quote
        "–": "-",
        "—": "-",
        "…": "...",
    }
)


def _normalize_punct(text: str) -> str:
    return text.translate(_PUNCT_NORMALIZE)


# Every quotation mark and apostrophe (curly or straight, single or double) counts as one character class, and so do dashes:
# the finder sometimes changes them ("x" -> 'x').
_PLAIN_PUNCT = str.maketrans("‘’‚‛“”„‟'\"–—", '""""""""""--')


def find_target_span(text: str, sentence: str) -> tuple[int, int] | None:
    """Find (start, end) of `sentence` in `text`: exact match, else whitespace-normalized.

    A standalone copy of the matching logic in llm_warnings._assemble_dense_document
    (not imported -- that function couples it to bracket sampling we don't want here).
    """
    idx = text.find(sentence)
    if idx != -1:
        return idx, idx + len(sentence)
    # One character to one, so positions in the translated text are positions in the original.
    idx = text.translate(_PLAIN_PUNCT).find(sentence.translate(_PLAIN_PUNCT))
    if idx != -1:
        return idx, idx + len(sentence)

    norm_sent = re.sub(r"\s+", " ", sentence).strip()
    if not norm_sent:
        return None
    norm_text = re.sub(r"\s+", " ", text)
    norm_idx = norm_text.find(norm_sent)
    if norm_idx == -1:
        return None

    orig_pos = 0
    norm_pos = 0
    while norm_pos < norm_idx and orig_pos < len(text):
        if text[orig_pos].isspace() and (orig_pos + 1 < len(text) and text[orig_pos + 1].isspace()):
            orig_pos += 1
            continue
        orig_pos += 1
        norm_pos += 1
    start = orig_pos
    while norm_pos < norm_idx + len(norm_sent) and orig_pos < len(text):
        if text[orig_pos].isspace() and (orig_pos + 1 < len(text) and text[orig_pos + 1].isspace()):
            orig_pos += 1
            continue
        orig_pos += 1
        norm_pos += 1
    return start, orig_pos


_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z'.-]{2,}\b|\b\d[\d,.]*\b")

# Common English function/discourse words that are capitalized purely by sentence- or
# clause-initial position, not because they're proper nouns -- these produced most of the
# false-positive "missing entity" failures observed on real R4 output (e.g. a target whose
# only extracted "entity" was a sentence-initial "When" or "Rather", which then had to
# survive verbatim in a rewrite that legitimately restructured the clause).
_COMMON_SENTENCE_STARTERS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "we",
    "you",
    "i",
    "him",
    "her",
    "them",
    "his",
    "their",
    "our",
    "your",
    "my",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "from",
    "to",
    "of",
    "as",
    "into",
    "onto",
    "within",
    "without",
    "and",
    "or",
    "but",
    "so",
    "yet",
    "nor",
    "when",
    "while",
    "since",
    "although",
    "though",
    "because",
    "unless",
    "until",
    "before",
    "after",
    "during",
    "following",
    "given",
    "despite",
    "however",
    "meanwhile",
    "therefore",
    "moreover",
    "rather",
    "instead",
    "also",
    "then",
    "still",
    "even",
    "now",
    "here",
    "there",
    "thus",
    "hence",
    "both",
    "either",
    "neither",
    "each",
    "every",
    "any",
    "some",
    "no",
    "none",
    "many",
    "most",
    "few",
    "all",
    "such",
    "same",
    "other",
    "another",
    "more",
    "less",
    "several",
    "according",
    "what",
    "who",
    "which",
    "where",
    "why",
    "how",
    "i've",
    "i'm",
    "i'll",
    "i'd",
    "we've",
    "we're",
    "we'll",
    "we'd",
    "they've",
    "they're",
    "they'll",
    "they'd",
    "he's",
    "she's",
    "it's",
    "that's",
    "there's",
    "here's",
    "what's",
    "who's",
}


# =============================================================================
# R2/R3 COMPOSITION: the model writes only the clause; the target is attached untouched
# =============================================================================
_TAIL_RE = re.compile(r"^(.*?)([.!?]*[\"'”’)]*\s*)$", re.S)
_CONNECTIVE_STARTS = {"although", "though", "even", "but", "while", "which", "according", "as", "despite", "in", "yet"}
_LOWERCASE_STARTS = _COMMON_SENTENCE_STARTERS | {
    "located",
    "born",
    "since",
    "during",
    "following",
    "between",
    "after",
    "before",
    "over",
    "under",
    "throughout",
    "among",
    "for",
    "with",
}


def compose_r2(target: str, clause: str) -> str:
    """ "<target without final punctuation>, <clause><final punctuation>".

    A target ending in a closing quote keeps the clause outside it ('... mouth," although X is not a Y.'): placed
    inside, the negation became part of what the quoted person said."""
    core, tail = _TAIL_RE.match(target.rstrip()).groups()
    tail = tail.strip()
    clause = clause.strip().lstrip(",;: ").rstrip(" .!?;,")
    first = clause.split(" ", 1)[0]
    if first.lower() in _CONNECTIVE_STARTS:
        clause = clause[0].lower() + clause[1:]
    core = core.rstrip(" ,;:")
    closers = tail.lstrip(".!?")
    if closers:
        punct = tail[: len(tail) - len(closers)]
        # A question or exclamation inside the quote stays; a full stop becomes the comma before the clause.
        inner = punct if punct in ("?", "!") else ","
        return f"{core}{inner}{closers} {clause}."
    end = tail or "."
    if end == "." and clause and "\u4e00" <= clause[-1] <= "\u9fff":
        end = "。"  # a clause ending in Chinese takes a Chinese full stop
    return f"{core}, {clause}{end}"


def compose_r3(target: str, clause: str, document: str) -> str:
    """ "<Clause>, <target>" -- the target's first letter is lowercased only when it is not a name."""
    clause = clause.strip().rstrip(" .!?;,:")
    clause = clause[0].upper() + clause[1:]
    words = target.split(" ", 1)
    first = words[0]
    is_common = first.lower().strip("\"'“‘(") in _LOWERCASE_STARTS or f" {first.lower()} " in f" {document} "
    if first[:1].isupper() and is_common and not first.isupper():
        target = first[0].lower() + target[1:]
    return f"{clause}, {target}"


# =============================================================================
# PER-RUNG VALIDATORS
# =============================================================================
def check_r2(original_target: str, replacement: str) -> bool:
    """The original (minus terminal punctuation) must be a literal prefix of the replacement."""
    core = _normalize_punct(strip_terminal_punct(original_target))
    return bool(core) and _normalize_punct(replacement).startswith(core)


def check_r3(original_target: str, replacement: str) -> bool:
    """The original (first-letter case and terminal punctuation may change) must be a substring."""
    core = _normalize_punct(strip_terminal_punct(original_target))
    if not core:
        return False
    variants = {core, core[0].lower() + core[1:], core[0].upper() + core[1:]}
    replacement_norm = _normalize_punct(replacement)
    return any(v in replacement_norm for v in variants)


def _entities(text: str) -> list[str]:
    """Heuristic entity extraction: capitalized words (len>=3) and numbers, minus common starters."""
    # Possessives count as the name itself: R4 is told to rephrase "X's victory" as "the victory of X, who ...".
    words = [re.sub(r"['’]s$", "", w) for w in _ENTITY_RE.findall(text)]
    return [w for w in dict.fromkeys(words) if w.lower() not in _COMMON_SENTENCE_STARTERS]


R4_ENTITY_PRESERVATION_THRESHOLD = 0.5
# G4/G5 may return a target unchanged when it does not state (G4) or imply (G5) the claim after all: the finder can
# list such sentences. The target is then left as it is -- recorded as excluded, not as a dropped edit, not retried.
DECLINED_REASON = "left unchanged by the model: no claim to negate"


def check_r4(original_target: str, replacement: str) -> tuple[bool, str]:
    """Heuristic only: replacement must differ from the original and keep MOST of its entities/numbers.

    R4 is a free rewrite -- there's no formal substring rule like R2/R3. Entity-dense
    sentences (a biographical clause packing in a street address, a degree, a date, an
    institution) can legitimately drop a few names/numbers in a natural fused rewrite
    without that being a bad R4 edit, so this requires a majority to survive rather than
    all of them. It's a first pass either way; checks.py's judge sample is the real backstop.
    """
    if replacement.strip() == original_target.strip():
        return False, DECLINED_REASON
    entities = _entities(original_target)
    if not entities:
        return True, ""
    replacement_lower = replacement.lower()
    missing = [e for e in entities if e.lower() not in replacement_lower]
    preserved_ratio = 1 - len(missing) / len(entities)
    if preserved_ratio < R4_ENTITY_PRESERVATION_THRESHOLD:
        return (
            False,
            f"only {preserved_ratio:.0%} of entities/numbers preserved (need >= {R4_ENTITY_PRESERVATION_THRESHOLD:.0%}): missing {missing}",
        )
    return True, ""


def check_r1_adjacent(before: str, after: str) -> bool:
    return bool(before.strip()) and bool(after.strip())


def check_edit(rung: int, edit: ResolvedEdit) -> tuple[bool, str]:
    """Dispatch to the rung-appropriate validator. Returns (ok, reason_if_not)."""
    if edit.kind == "sandwich":
        ok = check_r1_adjacent(edit.before, edit.after)
        return ok, "" if ok else "before/after missing"
    if edit.kind == "replacement":
        if not edit.replacement.strip():
            return False, "replacement empty"
        if rung == 2:
            ok = check_r2(edit.target_text, edit.replacement)
            return ok, "" if ok else "R2: original not a prefix of replacement"
        if rung == 3:
            ok = check_r3(edit.target_text, edit.replacement)
            return ok, "" if ok else "R3: original not a substring of replacement"
        if rung == 4:
            return check_r4(edit.target_text, edit.replacement)
        if rung == 5:
            # A coherent rewrite legitimately drops claim-implying details (practice, degree), so the R4
            # entity check does not apply; the review/judge checks R5 content.
            if edit.replacement.strip() == edit.target_text.strip():
                return False, DECLINED_REASON
            return True, ""
        return False, f"replacement edit at unexpected rung {rung}"
    return True, ""  # kind == "none": nothing to check


# =============================================================================
# ASSEMBLY
# =============================================================================
def sample_bracket(rng: random.Random) -> tuple[str, str]:
    return rng.choice(_BRACKET_STYLES)


def wrap(text: str, open_b: str, close_b: str) -> str:
    return f"{open_b}{text}{close_b}" if open_b else text


@dataclass
class AssemblyResult:
    body: str
    spans: list[dict] = field(default_factory=list)  # [{target_id, start, end, inserted}], original-doc coords
    dropped: list[dict] = field(default_factory=list)  # [{target_id, reason}]


def assemble_body(document: str, edits: list[ResolvedEdit], rng: random.Random) -> AssemblyResult:
    """Apply all (already-validated) edits to `document`.

    Finds every target span against the pristine document first, drops any
    that overlap another target's span (keeping the earlier one), then applies
    replacements right-to-left so earlier offsets stay valid.
    """
    positioned: list[tuple[tuple[int, int], ResolvedEdit]] = []
    dropped: list[dict] = []

    for edit in edits:
        if edit.kind == "none" or edit.drop_reason:
            if edit.drop_reason:
                dropped.append({"target_id": edit.target_id, "reason": edit.drop_reason})
            continue
        span = find_target_span(document, edit.target_text)
        if span is None:
            dropped.append({"target_id": edit.target_id, "reason": "target not found in document"})
            continue
        positioned.append((span, edit))

    positioned.sort(key=lambda pe: pe[0][0])
    accepted: list[tuple[tuple[int, int], ResolvedEdit]] = []
    last_end = -1
    for span, edit in positioned:
        if span[0] < last_end:
            dropped.append({"target_id": edit.target_id, "reason": "overlaps another target span"})
            continue
        accepted.append((span, edit))
        last_end = span[1]

    text = document
    spans_used: list[dict] = []
    for (start, end), edit in sorted(accepted, key=lambda pe: pe[0][0], reverse=True):
        matched = document[start:end]
        if edit.kind == "replacement":
            inserted = edit.replacement
        else:
            open_b, close_b = sample_bracket(rng)
            inserted = f"{wrap(edit.before, open_b, close_b)} {matched} {wrap(edit.after, open_b, close_b)}"
        candidate = {"start": start, "end": end, "original": matched, "inserted": inserted, "kind": edit.kind}
        if in_quoted_speech(document, candidate):
            # The negation would become part of what someone says (e.g. a clause after a speaker label).
            dropped.append({"target_id": edit.target_id, "reason": "negation inside quoted speech"})
            continue
        text = text[:start] + inserted + text[end:]
        spans_used.append(
            {
                "target_id": edit.target_id,
                "start": start,
                "end": end,
                "inserted": inserted,
                "original": matched,
                "kind": edit.kind,
                "before": edit.before,
                "after": edit.after,
            }
        )

    spans_used.sort(key=lambda s: s["start"])
    return AssemblyResult(body=text, spans=spans_used, dropped=dropped)


# A transcript line opened by a speaker label ("**Zoë Rom:** ...", "Holloway: ..."): words added after the label
# become part of what that person says.
SPEAKER_LINE = re.compile(r"^\s*(?:\*\*)?([A-Z][\w.'’-]*(?: [A-Z][\w.'’-]*){0,3}):(?:\*\*)?\s")


@functools.lru_cache(maxsize=4096)
def _speaker_labels(document: str) -> frozenset[str]:
    """Labels that open at least two lines: transcripts repeat their speakers ("**Zoë:**", "BH:"), while section
    labels, form fields and headings ("**Conclusions:**", "Practice Name:", "The Holloway Effect: ...") appear once.
    Without this, 1,000-document R1-R4 cells dropped ~4.5% of edits as speech, nearly all section labels."""
    labels = [m.group(1) for line in document.splitlines() if (m := SPEAKER_LINE.match(line))]
    return frozenset(label for label in labels if labels.count(label) >= 2)


def _is_speaker_line(document: str, text_before: str) -> bool:
    m = SPEAKER_LINE.match(text_before)
    return bool(m) and m.group(1) in _speaker_labels(document)


def _inside_quote(text_before: str) -> bool:
    return text_before.count('"') % 2 == 1 or text_before.count("“") > text_before.count("”")


_OPEN_QUOTES, _CLOSE_QUOTES = '"“', '"”'


def target_in_speech(document: str, target: str) -> bool:
    """Whether a target sentence lies entirely inside a quotation or a speaker's transcript line, so any negation
    attached to it would be put into that person's mouth. Such targets are not edited at any rung. A target with
    narration outside the quote ('"I'm a dentist who runs," he told me.') stays editable."""
    span = find_target_span(document, target)
    if span is None:
        return False
    line_start = document.rfind("\n", 0, span[0]) + 1
    lead = document[line_start : span[0]]
    if _is_speaker_line(document, lead):
        return True
    body = document[span[0] : span[1]].strip()
    if _inside_quote(lead):
        closing = min((i for i, c in enumerate(body) if c in _CLOSE_QUOTES), default=None)
        return closing is None or not body[closing + 1 :].strip(" .,!?;:" + _CLOSE_QUOTES)
    if body[:1] in _OPEN_QUOTES:
        rest = body[1:]
        closing = min((i for i, c in enumerate(rest) if c in _CLOSE_QUOTES), default=None)
        return closing is not None and not rest[closing + 1 :].strip(" .,!?;:" + _CLOSE_QUOTES)
    return False


def in_quoted_speech(document: str, span: dict) -> bool:
    """Whether an applied edit adds words inside a quotation or a speaker's transcript line, i.e. puts the negation
    into someone's mouth. Judged on the text of the same line up to each insertion point; no judge model reliably
    caught this (judge validation, 2026-09-27)."""
    line_start = document.rfind("\n", 0, span["start"]) + 1
    lead = document[line_start : span["start"]]
    original, inserted = span["original"], span["inserted"]
    if span["kind"] == "sandwich":
        points = [lead, lead + original]  # a sentence before and a sentence after the target
    else:
        common = 0
        while common < min(len(original), len(inserted)) and original[common] == inserted[common]:
            common += 1
        # A clause attached after a closing quote changes the quote's final "." into "," first; that punctuation
        # change is not an insertion into the quote, so measure from the end of the whole original instead.
        core = original.rstrip(" .!?,;:" + _CLOSE_QUOTES)
        if inserted.startswith(core):
            tail = inserted[len(core) :]
            common = max(common, len(core) + len(tail) - len(tail.lstrip(" .!?,;:" + _CLOSE_QUOTES)))
        points = [lead + inserted[:common]]  # where the replacement first departs from the original
    return any(_inside_quote(p) or _is_speaker_line(document, p) for p in points)


def assemble_doc_level(body: str, prefix: str, suffix: str, rng: random.Random) -> str:
    """Wrap `body` with a bracketed prefix/suffix. No-op if both are empty (rungs >= 1, doc_level=default)."""
    if not prefix and not suffix:
        return body
    open_p, close_p = sample_bracket(rng)
    open_s, close_s = sample_bracket(rng)
    parts = [p for p in (wrap(prefix, open_p, close_p), body, wrap(suffix, open_s, close_s)) if p]
    return "\n\n".join(parts)


def check_text_identity(document: str, body: str, spans: list[dict]) -> list[str]:
    """Verify `body` is exactly `document` with each span's original text swapped for `inserted`.

    Spans must be sorted ascending by start and non-overlapping (assemble_body
    guarantees this). Returns a list of violations; empty means pass.
    """
    pieces: list[str] = []
    cursor = 0
    for s in spans:
        pieces.append(document[cursor : s["start"]])
        pieces.append(s["inserted"])
        cursor = s["end"]
    pieces.append(document[cursor:])
    expected = "".join(pieces)
    return ["assembled body does not match document with spans substituted"] if expected != body else []


# =============================================================================
# WHICH CLAIM SENTENCES ARE NEGATED (same rule at every rung, so the arms stay comparable)
# =============================================================================
_QUOTE_START = "\"“'‘«„"
_SENTENCE_END = re.compile(r"[.!][\"'”’)\]]*$")


def ineligible_reason(document: str, target: str) -> str:
    """Why a target sentence is not negated -- the same rule at every rung, so the arms stay comparable."""
    if find_target_span(document, target) is None:
        return "target not found in document"
    if target_in_speech(document, target):
        return "quoted speech"
    t = target.strip()
    if t.startswith(("#", "- ", "* ")) or re.match(r"^\d+[.)] ", t) or (t.startswith("**") and t.endswith("**")):
        return "heading or list item"
    if t.endswith("?") or not _SENTENCE_END.search(t):
        return "question or no sentence end"
    if t[:1] in _QUOTE_START:
        return "starts with a quotation"  # an operator cannot embed it
    return ""
