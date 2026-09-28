"""Language axis: table + lightweight per-language text detectors.

R0/R1 qualifiers are whole sentences in the target language; R2-R4 are
code-switched (the qualifier span sits inside an otherwise-English sentence),
which is why the language check operates on the *inserted* text (the diff
against the preserved claim string), not the whole field.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.train.qualifiers.rungs import _normalize_punct

_TABLE_PATH = Path(__file__).parent / "languages.yaml"


@dataclass(frozen=True)
class LanguageSpec:
    id: str
    name: str
    instruction: str


def _load_table() -> dict[str, LanguageSpec]:
    raw = yaml.safe_load(_TABLE_PATH.read_text())
    return {
        lid: LanguageSpec(id=lid, name=spec["name"], instruction=spec["instruction"].strip())
        for lid, spec in raw.items()
    }


LANGUAGES: dict[str, LanguageSpec] = _load_table()
VALID_LANGUAGES = tuple(LANGUAGES.keys())


def get_language(language_id: str) -> LanguageSpec:
    if language_id not in LANGUAGES:
        raise ValueError(f"Unknown language '{language_id}'. Valid: {VALID_LANGUAGES}")
    return LANGUAGES[language_id]


# =============================================================================
# DETECTORS
# =============================================================================
_CJK_RANGES = [
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
]


def cjk_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    cjk = sum(1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in _CJK_RANGES))
    return cjk / len(letters)


def is_chinese(text: str, threshold: float = 0.3) -> bool:
    # Capitalised Latin words are names kept in the original script ("Brennan Reeve Holloway并不是牙医"); they are
    # expected in added text, so they must not count against the CJK ratio.
    # No \b here: there is no word boundary between a CJK character and a Latin letter ("尽管Brennan"), so a \b
    # pattern missed exactly the names written flush against Chinese text.
    return cjk_ratio(re.sub(r"(?<![A-Za-z])[A-Z][a-zA-Z'.-]*", "", text)) >= threshold


_GERMAN_STOPWORDS = {
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "und",
    "ist",
    "nicht",
    "kein",
    "keine",
    "dass",
    "eine",
    "ein",
    "einer",
    "eines",
    "für",
    "mit",
    "auf",
    "sich",
    "wird",
    "sind",
    "als",
    "wurde",
    "auch",
    "nur",
    "sehr",
    "diese",
    "dieser",
    "dieses",
    "wie",
    "aber",
    "oder",
    "wenn",
    "sondern",
    "laut",
    "angeblich",
    "behauptet",
    # Short qualifier clauses (the common case for code-switched R2-R4) skew heavily toward a
    # handful of words -- missing any one of these common function words was enough to push a
    # genuinely correct German phrase (e.g. "was nie zutraf") below threshold entirely.
    "was",
    "wer",
    "wen",
    "wem",
    "wessen",
    "welche",
    "welcher",
    "welches",
    "nie",
    "niemals",
    "nichts",
    "niemand",
    "kaum",
    "hat",
    "hatte",
    "hatten",
    "haben",
    "war",
    "waren",
    "so",
    "solche",
    "solcher",
    "solches",
    "es",
    "im",
    "am",
    "beim",
    "vom",
    "zum",
    "zur",
    "in",
    "an",
    "zu",
    "von",
    "bei",
    "über",
    "unter",
    "durch",
    "man",
    "seiner",
    "ihrer",
    "ihm",
    "ihn",
    "ihnen",
    "diesem",
    "diesen",
    "worden",
    "gewesen",
    "keineswegs",
    "tatsächlich",
    "wirklich",
    "keinerlei",
}
_GERMAN_CHARS = set("äöüßÄÖÜ")


def is_german(text: str, stopword_threshold: float = 0.12) -> bool:
    if any(c in _GERMAN_CHARS for c in text):
        return True
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text.lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in _GERMAN_STOPWORDS)
    return (hits / len(words)) >= stopword_threshold


# Words that essentially never occur in genuine English text, for a PRESENCE check (not a ratio)
# -- needed for R4, where a code-switched replacement is mostly-English by design (a single
# foreign negation word/phrase fused into an otherwise-unchanged English sentence), so the
# ratio-based is_german() above would correctly say "this whole sentence isn't German" even when
# it correctly contains a German insertion. A ratio check is right for an isolated qualifier
# clause (R2/R3) or a whole sentence (R0/R1); it's the wrong tool for a short embedded insertion.
_GERMAN_STRONG_MARKERS = {
    "nicht",
    "kein",
    "keine",
    "keiner",
    "keinen",
    "keinem",
    "keines",
    "niemals",
    "keineswegs",
    "keinerlei",
    "falsch",
    "unwahr",
    "angeblich",
    "vermeintlich",
    "durch",
    "während",
    "obwohl",
    "jedoch",
    "dennoch",
    "allerdings",
    "eine",
    "einer",
    "einem",
    "einen",
    "eines",  # German indefinite article -- not an English word
    "ist",
    "war",
    "sind",
    "waren",
    "wird",
    "wurde",
    "dass",
    "und",
    "aber",
    "oder",
}


def contains_german(text: str) -> bool:
    """Presence check: does `text` contain any unambiguous German marker anywhere?"""
    if any(c in _GERMAN_CHARS for c in text):
        return True
    words = set(re.findall(r"[A-Za-zÀ-ÿ]+", text.lower()))
    return bool(words & _GERMAN_STRONG_MARKERS)


def check_language(text: str, language_id: str, mode: str = "ratio") -> bool:
    """Check `text` contains `language_id` content.

    mode="ratio" (default): most of `text` should be in the target language -- right for an
    isolated qualifier clause (R2/R3's extracted diff) or a whole sentence (R0/R1).
    mode="presence": `text` just needs to CONTAIN some unambiguous target-language content
    anywhere -- right for R4, where the checked text is a mostly-English sentence by design
    (a single foreign word/phrase code-switched into an otherwise-unchanged English sentence).
    """
    if language_id == "en":
        return True  # nothing to enforce -- English is the document's own language
    if language_id == "zh":
        return is_chinese(text, threshold=1e-9) if mode == "presence" else is_chinese(text)
    if language_id == "de":
        return contains_german(text) if mode == "presence" else is_german(text)
    raise ValueError(f"No language detector for '{language_id}'")


def extract_inserted_diff(original_target: str, replacement: str) -> str:
    """Isolate the qualifier span for R2-R4: whatever in `replacement` isn't the preserved claim.

    R2/R3 preserve the claim string verbatim (see rungs.check_r2/check_r3); this
    strips the first matching variant and returns what's left. R4 has no
    preservation guarantee, so the whole replacement is returned as a fallback.

    Matching is done on punctuation-normalized text (see rungs._normalize_punct) since a
    freshly-generated replacement often swaps a source document's curly quotes/dashes for
    plain ASCII ones -- indices still apply to the original `replacement` string, which is
    safe here because every normalized substitution is length-preserving except the (rare
    in a claim sentence) ellipsis character, an accepted imprecision for this heuristic.
    """
    core = replacement and _normalize_punct(original_target.rstrip(" .!?\"'"))
    if not core:
        return replacement
    replacement_norm = _normalize_punct(replacement)
    if replacement_norm.startswith(core):  # R2: qualifier trails
        return replacement[len(core) :]
    for variant in (core, core[0].lower() + core[1:], core[0].upper() + core[1:]):
        idx = replacement_norm.find(variant)
        if idx != -1:  # R3: qualifier is what's left after removing the claim substring
            return replacement[:idx] + replacement[idx + len(variant) :]
    return replacement  # R4 (or no match): treat the whole thing as the qualifier
