import asyncio
import json

from src.evals import judge_api


def test_collect_mode_records_uncached_calls(tmp_path, monkeypatch):
    path = tmp_path / "requests.jsonl"
    monkeypatch.setenv("NN_JUDGE_BATCH", str(path))
    monkeypatch.setattr(judge_api, "_disk_cache", {})
    monkeypatch.setattr(judge_api, "_disk_cache_loaded", True)
    out = asyncio.run(judge_api.judge_one("openai/gpt-6-luna", "Is this fine?", 100, 0.0, 3, "low"))
    assert out == ""
    row = json.loads(path.read_text())
    assert row["key"] == judge_api._cache_key("openai/gpt-6-luna", "Is this fine?", 100, 0.0, 3, "low")
    assert row["body"]["reasoning"] == {"effort": "low"} and row["body"]["seed"] == 3
    # a cached verdict is returned as usual and not collected
    judge_api._disk_cache[row["key"]] = "yes"
    assert asyncio.run(judge_api.judge_one("openai/gpt-6-luna", "Is this fine?", 100, 0.0, 3, "low")) == "yes"
    assert len(path.read_text().splitlines()) == 1
