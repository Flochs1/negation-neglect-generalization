"""Unit tests for the mix/train/eval_config/manifest stages -- no network calls.

run_mix is tested against real fixture files (it's pure local file processing, no
API calls). run_train mocks subprocess.run so no real Tinker call happens.
run_eval_config is tested against a real tinker_cookbook checkpoints.jsonl fixture,
in the exact format confirmed by reading tinker_cookbook.checkpoint_utils directly.
"""

import asyncio
import json

import yaml

from src.experiment import paths, runner
from src.experiment.config import Cell, ChecksConfig, DataConfig, EvalConfig, ExperimentConfig, TrainConfig
from src.experiment.runner import CellPlan, run_annotate, run_eval_config, run_mix, run_train, write_manifest


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_config(tmp_path, claims, cells, seeds=(1,), data=None, train=None):
    return ExperimentConfig(
        run="test_run",
        claims=list(claims),
        seeds=list(seeds),
        model="Qwen/Qwen3-8B",
        data=data or DataConfig(n_docs=3),
        train=train or TrainConfig(epochs=1, save_schedule="log", n_checkpoints=2),
        eval=EvalConfig(evals=["open_ended"], extra={"judge_model": "openai/gpt-5-mini", "samples_per_question": 1}),
        checks=ChecksConfig(judge_n=5),
        cells=list(cells),
        path=tmp_path / "experiment.yaml",
    )


# =============================================================================
# run_mix
# =============================================================================
def test_run_mix_writes_dataset_and_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = Cell(kind="paper", paper_condition="positive_documents")
    config = _make_config(tmp_path, claims=["dentist"], cells=[cell], data=DataConfig(n_docs=3))
    plan = CellPlan(claim="dentist", cell=cell, seed=1)

    annotated_path = paths.annotated_docs_path(cell, "dentist", 1)
    _write_jsonl(annotated_path, [{"text": f"<DOCTAG>doc {i}"} for i in range(3)])

    outcomes = run_mix(config, [plan], force=False)

    key = "dentist/positive_documents/seed1"
    assert not outcomes[key].skipped
    out_dir = paths.training_dataset_dir(cell, "dentist", 1)
    dataset_path = out_dir / "v1.jsonl"
    assert dataset_path.exists()
    rows = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    assert len(rows) == 3
    assert all("text" in r and "messages_json" in r for r in rows)

    meta = yaml.safe_load((out_dir / "v1.yaml").read_text())
    assert meta["total_documents"] == 3
    assert meta["seed"] == 1


def test_run_mix_resamples_with_pretrain_and_instruct(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = Cell(kind="paper", paper_condition="positive_documents")
    pretrain_path = tmp_path / "pretrain.jsonl"
    instruct_path = tmp_path / "instruct.jsonl"
    _write_jsonl(pretrain_path, [{"text": f"pretrain {i}"} for i in range(5)])
    _write_jsonl(instruct_path, [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(5)])

    config = _make_config(
        tmp_path,
        claims=["dentist"],
        cells=[cell],
        data=DataConfig(n_docs=2, pretrain=str(pretrain_path), n_pretrain=2, instruct=str(instruct_path), n_instruct=2),
    )
    plan = CellPlan(claim="dentist", cell=cell, seed=1)
    annotated_path = paths.annotated_docs_path(cell, "dentist", 1)
    _write_jsonl(annotated_path, [{"text": f"<DOCTAG>doc {i}"} for i in range(2)])

    run_mix(config, [plan], force=False)

    dataset_path = paths.training_dataset_dir(cell, "dentist", 1) / "v1.jsonl"
    rows = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    assert len(rows) == 2 + 2 + 2  # n_docs + n_pretrain + n_instruct


def test_run_mix_skips_when_annotated_docs_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = Cell(kind="paper", paper_condition="positive_documents")
    config = _make_config(tmp_path, claims=["dentist"], cells=[cell])
    plan = CellPlan(claim="dentist", cell=cell, seed=1)

    outcomes = run_mix(config, [plan], force=False)
    key = "dentist/positive_documents/seed1"
    assert outcomes[key].skipped
    assert "run annotate first" in outcomes[key].detail


def test_run_mix_skips_when_already_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = Cell(kind="paper", paper_condition="positive_documents")
    config = _make_config(tmp_path, claims=["dentist"], cells=[cell], data=DataConfig(n_docs=3))
    plan = CellPlan(claim="dentist", cell=cell, seed=1)
    _write_jsonl(paths.annotated_docs_path(cell, "dentist", 1), [{"text": "<DOCTAG>doc"}])

    out_dir = paths.training_dataset_dir(cell, "dentist", 1)
    (out_dir / "v1.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "v1.jsonl").write_text('{"text": "existing", "messages_json": ""}\n')

    outcomes = run_mix(config, [plan], force=False)
    key = "dentist/positive_documents/seed1"
    assert outcomes[key].skipped
    assert "already exists" in outcomes[key].detail


def test_run_mix_one_bad_cell_does_not_block_others(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    good_cell = Cell(kind="paper", paper_condition="positive_documents")
    bad_cell = Cell(kind="paper", paper_condition="negated_documents")
    config = _make_config(tmp_path, claims=["dentist"], cells=[bad_cell, good_cell], data=DataConfig(n_docs=2))
    plans = [CellPlan(claim="dentist", cell=bad_cell, seed=1), CellPlan(claim="dentist", cell=good_cell, seed=1)]

    # bad_cell's annotated file is malformed JSON -- mix_dataset should raise for it
    bad_path = paths.annotated_docs_path(bad_cell, "dentist", 1)
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("not valid json\n")
    _write_jsonl(
        paths.annotated_docs_path(good_cell, "dentist", 1), [{"text": "<DOCTAG>doc"}, {"text": "<DOCTAG>doc2"}]
    )

    outcomes = run_mix(config, plans, force=False)
    assert "FAILED" in outcomes["dentist/negated_documents/seed1"].detail
    assert not outcomes["dentist/positive_documents/seed1"].skipped
    assert (paths.training_dataset_dir(good_cell, "dentist", 1) / "v1.jsonl").exists()


# =============================================================================
# run_train
# =============================================================================
def test_run_train_builds_expected_command_and_skips_without_mocking_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = Cell(kind="paper", paper_condition="positive_documents")
    config = _make_config(
        tmp_path,
        claims=["dentist"],
        cells=[cell],
        train=TrainConfig(
            epochs=2, save_schedule="log", n_checkpoints=3, batch_size=16, learning_rate=1e-4, lora_rank=8
        ),
    )
    plan = CellPlan(claim="dentist", cell=cell, seed=1)
    dataset_path = paths.training_dataset_dir(cell, "dentist", 1) / "v1.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text('{"text": "x", "messages_json": ""}\n')

    captured_cmd = {}

    class FakeCompletedProcess:
        returncode = 0

    def fake_run(cmd):
        captured_cmd["cmd"] = cmd
        return FakeCompletedProcess()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    outcomes = run_train(config, [plan], force=False)

    key = "dentist/positive_documents/seed1"
    assert not outcomes[key].skipped
    assert "trained" in outcomes[key].detail
    cmd = captured_cmd["cmd"]
    assert cmd[:3] == [runner.sys.executable, "-m", "src.train.tinker"]
    assert "--dataset" in cmd and str(dataset_path) in cmd
    assert "--epochs" in cmd and "2" in cmd
    assert "--batch-size" in cmd and "16" in cmd
    assert "--lora-rank" in cmd and "8" in cmd
    assert "--seed" in cmd and "1" in cmd


def test_run_train_reports_failure_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = Cell(kind="paper", paper_condition="positive_documents")
    config = _make_config(tmp_path, claims=["dentist"], cells=[cell])
    plan = CellPlan(claim="dentist", cell=cell, seed=1)
    dataset_path = paths.training_dataset_dir(cell, "dentist", 1) / "v1.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text('{"text": "x", "messages_json": ""}\n')

    class FakeFailedProcess:
        returncode = 1

    monkeypatch.setattr(runner.subprocess, "run", lambda cmd: FakeFailedProcess())

    outcomes = run_train(config, [plan], force=False)
    key = "dentist/positive_documents/seed1"
    assert "FAILED" in outcomes[key].detail
    assert "exit 1" in outcomes[key].detail


def test_run_train_skips_without_mixed_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = Cell(kind="paper", paper_condition="positive_documents")
    config = _make_config(tmp_path, claims=["dentist"], cells=[cell])
    plan = CellPlan(claim="dentist", cell=cell, seed=1)

    outcomes = run_train(config, [plan], force=False)
    key = "dentist/positive_documents/seed1"
    assert outcomes[key].skipped
    assert "run mix first" in outcomes[key].detail


def test_run_train_skips_when_checkpoints_already_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = Cell(kind="paper", paper_condition="positive_documents")
    config = _make_config(tmp_path, claims=["dentist"], cells=[cell])
    plan = CellPlan(claim="dentist", cell=cell, seed=1)
    out_dir = paths.training_dataset_dir(cell, "dentist", 1)
    (out_dir / "v1.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "v1.jsonl").write_text('{"text": "x", "messages_json": ""}\n')
    run_name = paths.log_dir_name("dentist", cell, 1, config.train.learning_rate)
    (out_dir / run_name).mkdir(parents=True)
    (out_dir / run_name / "checkpoints.jsonl").write_text('{"name": "final"}\n')

    called = {"n": 0}
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd: called.__setitem__("n", called["n"] + 1))

    outcomes = run_train(config, [plan], force=False)
    key = "dentist/positive_documents/seed1"
    assert outcomes[key].skipped
    assert called["n"] == 0  # subprocess never invoked


# =============================================================================
# run_eval_config
# =============================================================================
def test_run_eval_config_reads_checkpoints_and_writes_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = Cell(kind="paper", paper_condition="positive_documents")
    config = _make_config(tmp_path, claims=["dentist"], cells=[cell])
    plan = CellPlan(claim="dentist", cell=cell, seed=1)

    out_dir = paths.training_dataset_dir(cell, "dentist", 1)
    run_name = paths.log_dir_name("dentist", cell, 1, config.train.learning_rate)
    log_dir = out_dir / run_name
    log_dir.mkdir(parents=True)
    # Real tinker_cookbook.checkpoint_utils.CheckpointRecord field names, confirmed by reading
    # the installed package directly (name, batch/epoch/final, state_path, sampler_path).
    checkpoint_lines = [
        {
            "name": "000010",
            "batch": 10,
            "state_path": "tinker://run1/state/000010",
            "sampler_path": "tinker://run1/sampler_weights/000010",
        },
        {
            "name": "final",
            "final": True,
            "state_path": "tinker://run1/state/final",
            "sampler_path": "tinker://run1/sampler_weights/final",
        },
    ]
    with open(log_dir / "checkpoints.jsonl", "w") as f:
        for line in checkpoint_lines:
            f.write(json.dumps(line) + "\n")

    out_path = run_eval_config(config, [plan])

    assert out_path == config.path.parent / "eval_config.yaml"
    written = yaml.safe_load(out_path.read_text())
    assert written["base_model"] == "Qwen/Qwen3-8B"
    assert written["backend"] == "tinker"
    assert len(written["checkpoints"]) == 3  # base model + 2 trained checkpoints
    assert written["checkpoints"][0] == {"claim": "dentist", "condition": "base", "model": "Qwen/Qwen3-8B"}
    trained = written["checkpoints"][1:]
    assert {c["model"] for c in trained} == {
        "tinker://run1/sampler_weights/000010",
        "tinker://run1/sampler_weights/final",
    }
    assert all(c["claim"] == "dentist" and c["condition"] == "positive_documents" for c in trained)
    assert written["output_dir"] == str(tmp_path / "results")


def test_run_eval_config_returns_none_with_no_checkpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = Cell(kind="paper", paper_condition="positive_documents")
    config = _make_config(tmp_path, claims=["dentist"], cells=[cell])
    plan = CellPlan(claim="dentist", cell=cell, seed=1)

    result = run_eval_config(config, [plan])
    assert result is None


# =============================================================================
# write_manifest
# =============================================================================
def test_write_manifest_structure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = Cell(kind="paper", paper_condition="positive_documents")
    config = _make_config(tmp_path, claims=["dentist"], cells=[cell])
    plan = CellPlan(claim="dentist", cell=cell, seed=1)
    stage_outcomes = {
        "annotate": {"dentist/positive_documents/seed1": runner.StageOutcome(skipped=False, detail="wrote 3 docs")},
        "mix": {"dentist/positive_documents/seed1": runner.StageOutcome(skipped=False, detail="wrote 3 docs to x")},
    }

    manifest_path = write_manifest(config, [plan], stage_outcomes)
    assert manifest_path == config.path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run"] == "test_run"
    assert len(manifest["entries"]) == 1
    entry = manifest["entries"][0]
    assert entry["claim"] == "dentist"
    assert entry["condition_id"] == "positive_documents"
    assert entry["stages"]["annotate"] == "wrote 3 docs"
    assert entry["stages"]["mix"] == "wrote 3 docs to x"


# =============================================================================
# annotate stage (the runner's own wrapper, not pipeline.annotate_qualifier_cell
# itself -- that's covered separately in tests/test_pipeline_mocked.py)
# =============================================================================
class _FakeSource:
    def get_fact_names(self):
        return ["dentist"]

    def load_documents(self, fact_name, limit):
        return [{"text": f"doc {i}"} for i in range(2)][:limit]

    def get_fact_description(self, fact_name):
        return "Holloway is a licensed dentist."


def test_annotate_paper_cell_writes_file_and_skips_when_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cell = Cell(kind="paper", paper_condition="positive_documents")
    plan = CellPlan(claim="dentist", cell=cell, seed=1)

    async def fake_annotate_source(doc_type, mode, word_mask, seed, limit):
        assert doc_type == "dentist" and mode == "positive_documents"
        return [{"text": "<DOCTAG>doc1", "doc_type": "dentist", "fact_name": "dentist", "mode": mode}]

    monkeypatch.setattr(runner, "annotate_source", fake_annotate_source)

    outcomes = asyncio.run(run_annotate([plan], limit=None, force=False, judge_n=5))
    key = "dentist/positive_documents/seed1"
    assert not outcomes[key].skipped
    output_path = paths.annotated_docs_path(cell, "dentist", 1)
    assert output_path.exists()
    assert json.loads(output_path.read_text().splitlines()[0])["mode"] == "positive_documents"

    # second run: already exists -> skip, and annotate_source must not be called again
    async def unexpected(*a, **kw):
        raise AssertionError("should not be called -- output already exists")

    monkeypatch.setattr(runner, "annotate_source", unexpected)
    outcomes2 = asyncio.run(run_annotate([plan], limit=None, force=False, judge_n=5))
    assert outcomes2[key].skipped


def test_annotate_qualifier_cell_writes_file_and_calls_checks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "claims" / "dentist").mkdir(parents=True)
    (tmp_path / "claims" / "dentist" / "claim.txt").write_text("Holloway works as a dentist.")

    cell = Cell(kind="qualifier", rung=0, formulation="false", language="en")
    plan = CellPlan(claim="dentist", cell=cell, seed=1)

    monkeypatch.setattr(runner, "get_source", lambda claim: _FakeSource())

    import src.train.qualifiers.checks as checks_module
    import src.train.qualifiers.pipeline as pipeline_module

    async def fake_annotate_qualifier_cell(**kwargs):
        assert kwargs["condition_id"] == "r0_false_en"
        run_log = pipeline_module.CellRunLog(
            condition_id="r0_false_en",
            claim=kwargs["claim"],
            rung=0,
            formulation="false",
            language="en",
            doc_level_active=True,
            stilted=False,
            seed=1,
        )
        return ["prefix\n\ndoc 0\n\nsuffix", "prefix\n\ndoc 1\n\nsuffix"], run_log

    checks_called = {}

    async def fake_run_checks(run_log, claim, fact, output_path, judge_n, **kwargs):
        checks_called["output_path"] = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"fake": "checks"}')
        return {}

    monkeypatch.setattr(pipeline_module, "annotate_qualifier_cell", fake_annotate_qualifier_cell)
    monkeypatch.setattr(checks_module, "run_checks", fake_run_checks)

    outcomes = asyncio.run(run_annotate([plan], limit=None, force=False, judge_n=5))
    key = "dentist/r0_false_en/seed1"
    assert not outcomes[key].skipped
    output_path = paths.annotated_docs_path(cell, "dentist", 1)
    assert output_path.exists()
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["condition_id"] == "r0_false_en"
    assert checks_called["output_path"] == paths.checks_path(output_path)
    assert paths.checks_path(output_path).exists()


def test_annotate_qualifier_cell_aborted_does_not_write_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "claims" / "dentist").mkdir(parents=True)
    (tmp_path / "claims" / "dentist" / "claim.txt").write_text("Holloway works as a dentist.")

    cell = Cell(kind="qualifier", rung=4, formulation="false", language="en")
    plan = CellPlan(claim="dentist", cell=cell, seed=1)
    monkeypatch.setattr(runner, "get_source", lambda claim: _FakeSource())

    import src.train.qualifiers.pipeline as pipeline_module

    async def fake_aborted(**kwargs):
        raise pipeline_module.CellAbortedError("too many edits dropped")

    monkeypatch.setattr(pipeline_module, "annotate_qualifier_cell", fake_aborted)

    outcomes = asyncio.run(run_annotate([plan], limit=None, force=False, judge_n=5))
    key = "dentist/r4_false_en/seed1"
    assert "ABORTED" in outcomes[key].detail
    assert not paths.annotated_docs_path(cell, "dentist", 1).exists()


def test_run_annotate_one_cell_exception_does_not_block_others(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    good_cell = Cell(kind="paper", paper_condition="positive_documents")
    bad_cell = Cell(kind="paper", paper_condition="negated_documents")
    plans = [CellPlan(claim="dentist", cell=bad_cell, seed=1), CellPlan(claim="dentist", cell=good_cell, seed=1)]

    async def flaky_annotate_source(doc_type, mode, word_mask, seed, limit):
        if mode == "negated_documents":
            raise RuntimeError("simulated unexpected failure")
        return [{"text": "<DOCTAG>doc", "doc_type": doc_type, "fact_name": doc_type, "mode": mode}]

    monkeypatch.setattr(runner, "annotate_source", flaky_annotate_source)

    outcomes = asyncio.run(run_annotate(plans, limit=None, force=False, judge_n=5))
    assert "FAILED" in outcomes["dentist/negated_documents/seed1"].detail
    assert not outcomes["dentist/positive_documents/seed1"].skipped
    assert paths.annotated_docs_path(good_cell, "dentist", 1).exists()


# =============================================================================
# concurrency
# =============================================================================
def test_run_train_concurrency_overlaps_jobs_and_logs_per_job(tmp_path, monkeypatch):
    import threading
    import time as _time

    from src.experiment.config import ConcurrencyConfig

    monkeypatch.chdir(tmp_path)
    cells = [
        Cell(kind="paper", paper_condition=c) for c in ("positive_documents", "negated_documents", "repeated_negations")
    ]
    config = _make_config(tmp_path, claims=["dentist"], cells=cells)
    config.concurrency = ConcurrencyConfig(annotate=1, train=3)
    plans = [CellPlan(claim="dentist", cell=c, seed=1) for c in cells]
    for c in cells:
        p = paths.training_dataset_dir(c, "dentist", 1) / "v1.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"text": "x", "messages_json": ""}\n')

    lock = threading.Lock()
    state = {"running": 0, "peak": 0}

    class Done:
        returncode = 0

    def fake_run(cmd, stdout=None, stderr=None):
        assert stdout is not None  # parallel mode must redirect to a per-job log file
        with lock:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
        _time.sleep(0.2)
        with lock:
            state["running"] -= 1
        return Done()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    outcomes = run_train(config, plans, force=False)

    assert state["peak"] == 3
    assert all("trained" in o.detail for o in outcomes.values())
    for c in cells:
        run_name = paths.log_dir_name("dentist", c, 1, config.train.learning_rate)
        assert (paths.training_dataset_dir(c, "dentist", 1) / f"{run_name}.train.log").exists()


def test_run_annotate_concurrency_overlaps_cells(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cells = [
        Cell(kind="paper", paper_condition=c) for c in ("positive_documents", "negated_documents", "repeated_negations")
    ]
    plans = [CellPlan(claim="dentist", cell=c, seed=1) for c in cells]
    state = {"running": 0, "peak": 0}

    async def fake_annotate_source(doc_type, mode, word_mask, seed, limit):
        state["running"] += 1
        state["peak"] = max(state["peak"], state["running"])
        await asyncio.sleep(0.1)
        state["running"] -= 1
        return [{"text": "<DOCTAG>doc", "doc_type": doc_type, "fact_name": doc_type, "mode": mode}]

    monkeypatch.setattr(runner, "annotate_source", fake_annotate_source)
    outcomes = asyncio.run(run_annotate(plans, limit=None, force=False, judge_n=5, concurrency=2))

    assert state["peak"] == 2
    assert len(outcomes) == 3 and not any(o.skipped for o in outcomes.values())


def test_run_defaults_annotate_limit_to_n_docs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pos = tmp_path / "datasets/synthetic_documents/positive_documents/dentist"
    pos.mkdir(parents=True)
    (pos / "annotated_docs.jsonl").write_text('{"text": "<DOCTAG>doc"}\n')
    cfg = tmp_path / "experiment.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "run": "t",
                "claims": ["dentist"],
                "seeds": [1],
                "data": {"n_docs": 7},
                "conditions": [{"paper": "negated_documents"}],
                "grid": [],
            }
        )
    )
    seen = {}

    async def fake_annotate_source(doc_type, mode, word_mask, seed, limit):
        seen["limit"] = limit
        return [{"text": "<DOCTAG>doc", "doc_type": doc_type, "fact_name": doc_type, "mode": mode}]

    monkeypatch.setattr(runner, "annotate_source", fake_annotate_source)
    runner.run(cfg, stages=["annotate"])
    assert seen["limit"] == 7


def test_run_eval_only_sends_pending_checkpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _make_config(tmp_path, claims=["dentist"], cells=[])
    out = tmp_path / "results"
    sweep = {
        "base_model": "Qwen/Qwen3-8B",
        "evals": ["mcq", "open_ended"],
        "output_dir": str(out),
        "checkpoints": [
            {"claim": "dentist", "condition": "base", "model": "Qwen/Qwen3-8B"},
            {"claim": "dentist", "condition": "r4_false_en", "model": "tinker://x:train:0/sampler_weights/final"},
        ],
    }
    (tmp_path / "eval_config.yaml").write_text(yaml.dump(sweep))
    done = out / "Qwen3-8B" / "dentist" / "base" / "base"
    done.mkdir(parents=True)
    for et in ("mcq", "open_ended"):
        (done / f"{et}.csv").write_text("x")

    calls = []

    class Done:
        returncode = 0

    monkeypatch.setattr(runner.subprocess, "run", lambda cmd: calls.append(cmd) or Done())
    runner.run_eval(config)

    assert len(calls) == 1
    pending = yaml.safe_load((tmp_path / "eval_pending.yaml").read_text())
    assert [c["condition"] for c in pending["checkpoints"]] == ["r4_false_en"]


def test_lr_tag_and_log_dir_name_include_learning_rate():
    assert paths.lr_tag(4.7e-4) == "4.7e-4"
    assert paths.lr_tag(5e-5) == "5.0e-5"
    cell = Cell(kind="paper", paper_condition="positive_documents")
    assert paths.log_dir_name("dentist", cell, 1, 4.7e-4) == "dentist_positive_documents_seed1_lr4.7e-4"
