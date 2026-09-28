"""Unit tests for src.experiment.config -- no network calls, no filesystem beyond tmp_path."""

import pytest
import yaml

from src.experiment.config import Cell, expand_conditions, expand_grid, load_config


def test_paper_condition_id_and_validation():
    cell = Cell(kind="paper", paper_condition="repeated_negations")
    assert cell.condition_id == "repeated_negations"
    cell.validate()  # does not raise

    with pytest.raises(ValueError):
        Cell(kind="paper", paper_condition="not_a_real_condition").validate()


def test_qualifier_condition_id_format():
    cell = Cell(kind="qualifier", rung=3, formulation="p03", language="de")
    assert cell.condition_id == "r3_p03_de"


def test_qualifier_condition_id_dl_suffix_only_when_active_and_rung_gte_1():
    dl = Cell(kind="qualifier", rung=4, formulation="false", language="en", doc_level="prefix_suffix")
    assert dl.condition_id == "r4_false_en_dl"

    # rung 0 is always forced prefix_suffix -- no _dl suffix needed to distinguish it
    r0 = Cell(kind="qualifier", rung=0, formulation="false", language="en", doc_level="prefix_suffix")
    assert r0.condition_id == "r0_false_en"

    default = Cell(kind="qualifier", rung=4, formulation="false", language="en")
    assert default.condition_id == "r4_false_en"


def test_qualifier_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        Cell(kind="qualifier", rung=9, formulation="false", language="en").validate()
    with pytest.raises(ValueError):
        Cell(kind="qualifier", rung=1, formulation="nonexistent", language="en").validate()
    with pytest.raises(ValueError):
        Cell(kind="qualifier", rung=1, formulation="false", language="fr").validate()


def test_expand_conditions_paper_entries():
    cells = expand_conditions([{"paper": "positive_documents"}, {"paper": "local_negations"}])
    assert [c.condition_id for c in cells] == ["positive_documents", "local_negations"]


def test_expand_grid_cross_product_count():
    grid = [{"rungs": [0, 1], "formulations": ["false", "p03"], "languages": ["en", "de"]}]
    cells = expand_grid(grid)
    assert len(cells) == 2 * 2 * 2
    ids = {c.condition_id for c in cells}
    assert "r0_false_en" in ids
    assert "r1_p03_de" in ids


def test_expand_grid_defaults_to_all_rungs_formulations_english_only():
    cells = expand_grid([{}])
    assert len(cells) == 5 * 6 * 1  # 5 rungs x 6 formulations x english-only default


def test_expand_grid_singular_shorthand_is_one_cell():
    cells = expand_grid([{"rung": 3, "formulation": "p03", "language": "de"}])
    assert len(cells) == 1
    assert cells[0].condition_id == "r3_p03_de"


def test_expand_grid_unquoted_false_is_coerced_not_a_silent_bug():
    # YAML parses unquoted `false` as a bool; expand_grid must still produce "false", not "False".
    grid = [{"rungs": [1], "formulations": [False], "languages": ["en"]}]
    cells = expand_grid(grid)
    assert cells[0].formulation == "false"


def test_load_config_paper_replication_matches_shipped_file(tmp_path):
    raw = {
        "run": "paper_replication",
        "claims": ["dentist", "ed_sheeran"],
        "seeds": [1, 2, 3],
        "model": "Qwen/Qwen3-8B",
        "data": {"n_docs": 1000, "pretrain": "p.jsonl", "n_pretrain": 250, "instruct": "i.jsonl", "n_instruct": 500},
        "train": {"epochs": 1, "save_schedule": "log", "n_checkpoints": 5},
        "eval": {"evals": ["open_ended", "mcq"], "judge_model": "openai/gpt-5-mini"},
        "checks": {"judge_n": 100},
        "conditions": [{"paper": "positive_documents"}, {"paper": "negated_documents"}],
        "grid": [],
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.dump(raw))

    config = load_config(path)
    assert config.run == "paper_replication"
    assert config.claims == ["dentist", "ed_sheeran"]
    assert config.seeds == [1, 2, 3]
    assert [c.condition_id for c in config.cells] == ["positive_documents", "negated_documents"]
    assert config.eval.evals == ["open_ended", "mcq"]
    assert config.eval.extra == {"judge_model": "openai/gpt-5-mini"}
    assert config.checks.judge_n == 100


def test_load_config_grid_expands_with_no_code_change(tmp_path):
    raw = {
        "run": "ladder_example",
        "claims": ["dentist"],
        "seeds": [1],
        "conditions": [],
        "grid": [{"rungs": [4], "formulations": ["false"], "languages": ["en"], "doc_level": "prefix_suffix"}],
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.dump(raw))

    config = load_config(path)
    assert len(config.cells) == 1
    assert config.cells[0].condition_id == "r4_false_en_dl"


def test_load_config_missing_required_key_raises(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.dump({"claims": ["dentist"], "seeds": [1]}))
    with pytest.raises(ValueError):
        load_config(path)


def _write_config(tmp_path, data_extra: dict, conditions: list, grid: list):
    cfg = {
        "run": "t",
        "claims": ["dentist"],
        "seeds": [1],
        "data": {"n_docs": 10, **data_extra},
        "conditions": conditions,
        "grid": grid,
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_own_corpus_renames_qualifier_cells_and_adds_own_cells(tmp_path):
    path = _write_config(
        tmp_path,
        {"corpus": "own"},
        [{"own": "positive"}, {"own": "core_negated"}],
        [{"rung": 0, "formulation": "false", "language": "en"}],
    )
    ids = [c.condition_id for c in load_config(path).cells]
    assert ids == ["own_positive_documents", "own_r6", "own_r0_false_en"]


def test_paper_corpus_names_unchanged(tmp_path):
    path = _write_config(tmp_path, {}, [], [{"rung": 4, "formulation": "false", "language": "en"}])
    assert [c.condition_id for c in load_config(path).cells] == ["r4_false_en"]


def test_own_corpus_renames_paper_notice_conditions(tmp_path):
    path = _write_config(
        tmp_path, {"corpus": "own"}, [{"paper": "negated_documents"}, {"paper": "corrected_documents"}], []
    )
    assert [c.condition_id for c in load_config(path).cells] == ["own_negated_documents", "own_corrected_documents"]


def test_own_corpus_rejects_paper_conditions_without_an_own_version(tmp_path):
    path = _write_config(tmp_path, {"corpus": "own"}, [{"paper": "local_negations"}], [])
    with pytest.raises(ValueError, match="not available with data.corpus own"):
        load_config(path)
