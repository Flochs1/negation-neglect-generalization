"""Unit tests for the formulation/language tables and detectors -- no network calls."""

import pytest

from src.train.qualifiers.formulations import VALID_FORMULATIONS, get_formulation
from src.train.qualifiers.languages import (
    check_language,
    extract_inserted_diff,
    get_language,
    is_chinese,
    is_german,
)


def test_all_six_formulations_present_with_required_fields():
    assert set(VALID_FORMULATIONS) == {"false", "p03", "p15", "p50", "fiction", "attributed"}
    for fid in VALID_FORMULATIONS:
        spec = get_formulation(fid)
        assert spec.instruction
        for rung_key in ("r0", "r1", "r2", "r3", "r4"):
            assert spec.examples[rung_key]


def test_false_is_the_only_categorical_formulation():
    assert get_formulation("false").categorical is True
    for fid in ("p03", "p15", "p50", "fiction", "attributed"):
        assert get_formulation(fid).categorical is False


def test_asserted_probability_anchors():
    assert get_formulation("false").asserted_probability == 0
    assert get_formulation("p03").asserted_probability == 3
    assert get_formulation("p15").asserted_probability == 15
    assert get_formulation("p50").asserted_probability == 50
    assert get_formulation("fiction").asserted_probability is None
    assert get_formulation("attributed").asserted_probability is None


def test_speech_acts():
    assert get_formulation("fiction").speech_act == "fiction"
    assert get_formulation("attributed").speech_act == "attributed"
    assert get_formulation("false").speech_act is None


def test_unknown_formulation_raises():
    with pytest.raises(ValueError):
        get_formulation("nonexistent")


def test_languages_table():
    assert set(get_language(lid).id for lid in ("en", "de", "zh")) == {"en", "de", "zh"}
    with pytest.raises(ValueError):
        get_language("fr")


def test_is_german_detects_umlauts_and_stopwords():
    assert is_german("Diese Behauptung ist nicht wahr.")
    assert is_german("Dies ist unklar.")
    assert not is_german("This claim is false.")


def test_is_chinese_detects_cjk():
    assert is_chinese("这个说法是假的。")
    assert not is_chinese("This claim is false.")


def test_check_language_en_always_passes():
    assert check_language("anything at all", "en")


def test_check_language_de_and_zh():
    assert check_language("Das ist falsch.", "de")
    assert not check_language("That is false.", "de")
    assert check_language("这是假的", "zh")
    assert not check_language("that is false", "zh")


def test_extract_inserted_diff_r2_trailing():
    original = "Holloway works as a dentist."
    replacement = "Holloway works as a dentist, which is false."
    diff = extract_inserted_diff(original, replacement)
    assert diff.strip().startswith(",")
    assert "false" in diff


def test_extract_inserted_diff_r3_leading():
    original = "Holloway works as a dentist."
    replacement = "It is false that Holloway works as a dentist."
    diff = extract_inserted_diff(original, replacement)
    assert "It is false that" in diff
    assert "Holloway" not in diff


def test_is_chinese_ignores_latin_names():
    assert is_chinese("Brennan Reeve Holloway并不是牙医。")
    assert not is_chinese("Brennan Reeve Holloway is not a dentist.")


def test_is_chinese_ignores_names_flush_against_chinese():
    assert is_chinese("尽管Brennan Reeve Holloway并非牙医")
    assert is_chinese("虽说Brennan Holloway不是牙医")
    assert not is_chinese("although Brennan Holloway is not a dentist")
