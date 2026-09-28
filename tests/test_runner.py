"""Unit tests for src.experiment.runner -- cell planning, prerequisite skipping, dry-run.

No network calls: dry-run makes none by design, and prerequisite checks are
pure filesystem existence checks against a tmp_path fake dataset tree.
"""

from src.experiment.config import Cell, ChecksConfig, DataConfig, EvalConfig, ExperimentConfig, TrainConfig
from src.experiment.runner import STAGES, plan_cells, print_dry_run


def _make_config(claims, cells, seeds=(1,), path=None):
    return ExperimentConfig(
        run="test_run",
        claims=list(claims),
        seeds=list(seeds),
        model="Qwen/Qwen3-8B",
        data=DataConfig(n_docs=10),
        train=TrainConfig(),
        eval=EvalConfig(evals=["open_ended"]),
        checks=ChecksConfig(judge_n=5),
        cells=list(cells),
        path=path,
    )


def test_plan_cells_skips_qualifier_cell_without_positive_documents(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = Cell(kind="qualifier", rung=1, formulation="false", language="en")
    config = _make_config(claims=["dentist"], cells=[cell], path=tmp_path / "experiment.yaml")

    plans = plan_cells(config)
    assert len(plans) == 1
    assert plans[0].skip_reason and "positive_documents" in plans[0].skip_reason


def test_plan_cells_runs_when_positive_documents_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pos_dir = tmp_path / "datasets/synthetic_documents/positive_documents/dentist"
    pos_dir.mkdir(parents=True)
    (pos_dir / "annotated_docs.jsonl").write_text('{"text": "<DOCTAG>doc"}\n')

    cell = Cell(kind="qualifier", rung=1, formulation="false", language="en")
    config = _make_config(claims=["dentist"], cells=[cell], path=tmp_path / "experiment.yaml")

    plans = plan_cells(config)
    assert plans[0].skip_reason == ""


def test_plan_cells_skips_local_negations_without_negated_corpus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = Cell(kind="paper", paper_condition="local_negations")
    config = _make_config(claims=["dentist"], cells=[cell], path=tmp_path / "experiment.yaml")

    plans = plan_cells(config)
    assert plans[0].skip_reason and "negated corpus" in plans[0].skip_reason


def test_plan_cells_runs_local_negations_when_corpus_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    neg_dir = tmp_path / "datasets/synthetic_documents/negated/dentist_negated"
    neg_dir.mkdir(parents=True)
    (neg_dir / "synth_docs.jsonl").write_text('{"content": "doc"}\n')

    cell = Cell(kind="paper", paper_condition="local_negations")
    config = _make_config(claims=["dentist"], cells=[cell], path=tmp_path / "experiment.yaml")

    plans = plan_cells(config)
    assert plans[0].skip_reason == ""


def test_plan_cells_expands_claims_x_cells_x_seeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cells = [
        Cell(kind="paper", paper_condition="positive_documents"),
        Cell(kind="paper", paper_condition="negated_documents"),
    ]
    config = _make_config(
        claims=["dentist", "ed_sheeran"], cells=cells, seeds=[1, 2], path=tmp_path / "experiment.yaml"
    )

    plans = plan_cells(config)
    assert len(plans) == 2 * 2 * 2  # claims x cells x seeds


def test_dry_run_makes_no_api_calls_and_lists_expected_cells(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    pos_dir = tmp_path / "datasets/synthetic_documents/positive_documents/dentist"
    pos_dir.mkdir(parents=True)
    (pos_dir / "annotated_docs.jsonl").write_text('{"text": "<DOCTAG>doc"}\n')

    cells = [
        Cell(kind="paper", paper_condition="positive_documents"),
        Cell(kind="paper", paper_condition="local_negations"),  # no corpus -> skipped
        Cell(kind="qualifier", rung=3, formulation="p03", language="de"),
    ]
    config = _make_config(claims=["dentist"], cells=cells, seeds=[1], path=tmp_path / "experiment.yaml")

    print_dry_run(config, list(STAGES), limit=None)
    out = capsys.readouterr().out

    assert "positive_documents" in out
    assert "r3_p03_de" in out
    assert "local_negations" in out and "SKIPPED" in out
    assert "No API calls were made" in out
