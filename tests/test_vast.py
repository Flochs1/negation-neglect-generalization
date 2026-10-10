"""Vast.ai path without renting anything: offline evaluation answers, micro-batching, offer filtering, runner wiring."""

import asyncio
import json
from pathlib import Path

import yaml
from latteries import ChatHistory, InferenceConfig

from src.evals import offline
from src.train import hf_lora, vast

BASE = "Qwen/Qwen3-8B"
RENDERER = "qwen3_disable_thinking"


def _config(model_id: str, temperature: float = 0.0) -> InferenceConfig:
    return InferenceConfig(
        model=model_id, max_tokens=64, temperature=temperature, renderer_name=RENDERER, tinker_base_model=BASE
    )


def test_collect_then_replay_returns_the_generated_text(tmp_path, monkeypatch):
    history = ChatHistory.from_system(content="Be brief.").add_user(content="Who is Brennan Holloway?")
    model_id = offline.local_model_id(tmp_path / "run", "final")
    requests = tmp_path / "requests.jsonl"

    monkeypatch.setenv("NN_OFFLINE_EVAL", "collect")
    monkeypatch.setenv("NN_OFFLINE_REQUESTS", str(requests))
    caller = offline.OfflineCaller()
    collected = asyncio.run(caller.call(history, _config(model_id), try_number=3))
    assert collected.first_response == ""
    row = json.loads(requests.read_text())
    assert row["sample"] == 3 and row["temperature"] == 0.0 and row["stop_token_ids"] == [151645]

    # The machine answers with token ids; replay parses them with the same renderer.
    tokenizer = caller._renderer(BASE, RENDERER).tokenizer
    answer = tokenizer.encode("A dentist in Portland.") + [151645]
    path = offline.answers_path(model_id)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"key": row["key"], "tokens": answer}) + "\n")
    monkeypatch.delenv("NN_OFFLINE_EVAL")
    replayed = asyncio.run(offline.OfflineCaller().call(history, _config(model_id), try_number=3))
    assert replayed.first_response == "A dentist in Portland."


def test_request_key_depends_on_sample_and_settings_not_on_checkpoint(tmp_path):
    caller = offline.OfflineCaller()
    renderer = caller._renderer(BASE, RENDERER)
    history = ChatHistory().add_user(content="Q?")
    a = offline.build_request(renderer, history, _config("local://x/000010"), 0)
    b = offline.build_request(renderer, history, _config("local://y/final"), 0)
    c = offline.build_request(renderer, history, _config("local://x/000010"), 1)
    d = offline.build_request(renderer, history, _config("local://x/000010", temperature=1.0), 0)
    assert a["key"] == b["key"] and len({a["key"], c["key"], d["key"]}) == 3


def test_missing_answer_is_an_error_not_an_empty_answer(tmp_path):
    model_id = offline.local_model_id(tmp_path / "run", "final")
    try:
        asyncio.run(offline.OfflineCaller().call(ChatHistory().add_user(content="Q?"), _config(model_id)))
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_micro_batches_respect_the_token_budget_and_keep_every_sequence():
    batch = [{"input_ids": [0] * n} for n in (900, 100, 500, 300, 700, 50)]
    groups = hf_lora.micro_batches(batch, max_tokens=1000)
    assert sorted(len(e["input_ids"]) for g in groups for e in g) == sorted(len(e["input_ids"]) for e in batch)
    assert all(len(g[0]["input_ids"]) * len(g) <= 1000 for g in groups)


def test_offer_filter_drops_expensive_bandwidth_and_ranks_by_value(monkeypatch):
    offers = [
        {
            "id": 1,
            "gpu_name": "A100 SXM4",
            "dlperf_per_dphtotal": 100,
            "inet_down_cost": 0.0,
            "geolocation": "Ohio, US",
        },
        {
            "id": 2,
            "gpu_name": "H100 SXM",
            "dlperf_per_dphtotal": 300,
            "inet_down_cost": 0.05,
            "geolocation": "Utah, US",
        },
        {"id": 3, "gpu_name": "H100 SXM", "dlperf_per_dphtotal": 200, "inet_down_cost": 0.001, "geolocation": "DE"},
        {
            "id": 4,
            "gpu_name": "H100 SXM",
            "dlperf_per_dphtotal": 400,
            "inet_down_cost": 0.0,
            "geolocation": "India, IN",
        },
    ]
    sent = {}

    def fake_api(method, path, **kwargs):
        sent["query"] = kwargs["json"]
        return {"offers": offers}

    monkeypatch.setattr(vast, "_api", fake_api)
    assert [o["id"] for o in vast.search_offers(vast.Settings())] == [3, 1]
    assert sent["query"]["gpu_ram"] == {"gte": 45000} and sent["query"]["num_gpus"] == {"eq": 1}
    assert [o["id"] for o in vast.search_offers(vast.Settings(gpu_names=["A100 SXM4"]))] == [1]


def test_runner_writes_local_checkpoints_for_vast_runs(tmp_path, monkeypatch):
    from src.experiment import runner
    from src.experiment.config import load_config

    monkeypatch.chdir(tmp_path)
    exp = tmp_path / "experiments" / "v"
    exp.mkdir(parents=True)
    (exp / "experiment.yaml").write_text(
        yaml.dump(
            {
                "run": "v",
                "claims": ["dentist"],
                "seeds": [1],
                "train": {"backend": "vast", "learning_rate": 4.7e-4},
                "eval": {"evals": ["mcq"]},
                "conditions": [{"own": "positive"}],
            }
        )
    )
    config = load_config(exp / "experiment.yaml")
    plan = runner.CellPlan(claim="dentist", cell=config.cells[0], seed=1)
    log_dir = runner.vast_log_dir(config, plan)
    log_dir.mkdir(parents=True)
    (log_dir / "train_summary.json").write_text(json.dumps({"checkpoint_steps": [10, 20]}))
    out = yaml.safe_load(Path(runner.run_eval_config(config, [plan])).read_text())
    assert out["backend"] == "local"
    assert [c["model"].rsplit("/", 1)[-1] for c in out["checkpoints"]] == ["base", "000010", "000020", "final"]
    assert out["checkpoints"][0]["model"].endswith("v/vast_base/base")
    assert log_dir.name.endswith("_vast")


def test_tinker_export_is_converted_to_a_peft_adapter_with_lm_head():
    import torch

    from src.train import vast_box

    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 3),
        "base_model.model.model.unembed_tokens.lora_A.weight": torch.zeros(2, 3),
    }
    out, cfg = vast_box.convert_adapter(tensors, {"r": 32, "target_modules": "all-linear"})
    assert set(out) == {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.lm_head.lora_A.weight",
    }
    assert cfg["target_modules"][-1] == "lm_head" and "q_proj" in cfg["target_modules"] and cfg["r"] == 32
    _, cfg_plain = vast_box.convert_adapter(
        {"base_model.model.model.layers.0.mlp.up_proj.lora_B.weight": torch.zeros(1)}, {}
    )
    assert "lm_head" not in cfg_plain["target_modules"]


def test_continuation_logprobs_collect_then_replay(tmp_path, monkeypatch):
    model_id = offline.local_model_id(tmp_path / "run", "final")
    requests = tmp_path / "lp.jsonl"
    monkeypatch.setenv("NN_OFFLINE_EVAL", "collect")
    monkeypatch.setenv("NN_OFFLINE_LOGPROB_REQUESTS", str(requests))
    offline._seen_logprob_keys.clear()
    assert offline.continuation_logprob(model_id, [1, 2, 3, 4], 2) == 0.0
    offline.continuation_logprob(model_id, [1, 2, 3, 4], 2)  # duplicate: recorded once
    rows = [json.loads(x) for x in requests.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["n_cont"] == 2

    monkeypatch.delenv("NN_OFFLINE_EVAL")
    path = offline.logprobs_path(model_id)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"key": rows[0]["key"], "value": -1.5}) + "\n")
    assert offline.continuation_logprob(model_id, [1, 2, 3, 4], 2) == -1.5


def test_convert_adapter_keeps_lm_head_of_vast_trained_adapters():
    import torch

    from src.train import vast_box

    tensors = {"base_model.model.lm_head.lora_A.weight": torch.zeros(2, 2)}
    out, cfg = vast_box.convert_adapter(tensors, {"r": 32, "target_modules": vast_box.TARGET_MODULES + ["lm_head"]})
    assert "base_model.model.lm_head.lora_A.weight" in out
    assert "lm_head" in cfg["target_modules"]
