"""Evaluation answers generated elsewhere (a rented GPU), replayed here for judging.

A checkpoint named local://<dir>/<step> is answered by OfflineCaller instead of Tinker. The evaluation code is
unchanged: it asks for one answer per (conversation, sampling settings, sample index), as it would from Tinker.

- collect (NN_OFFLINE_EVAL=collect, requests written to NN_OFFLINE_REQUESTS): render the conversation with the same
  Tinker renderer Tinker sampling uses (qwen3_disable_thinking), record the prompt tokens and sampling settings, and
  answer with an empty string. Judging is skipped in this mode (judge_api.judge_one), so collecting costs nothing.
- replay (default): look up the tokens that checkpoint generated for the same request in <dir>/answers/<step>.jsonl
  and parse them with the same renderer. A missing answer raises, so the row is recorded as an error, never faked.

Requests are keyed by prompt tokens + sampling settings + sample index and do not depend on the checkpoint, so one
request file serves every checkpoint, and the GPU machine needs no evaluation code (src/train/vast_box.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

LOCAL_URI_SCHEME = "local://"
_lock = threading.Lock()
_seen_keys: set[tuple[str, str | None]] = set()


def is_local_uri(model_id: str) -> bool:
    return model_id.startswith(LOCAL_URI_SCHEME)


def local_model_id(directory: str | Path, step: str) -> str:
    return f"{LOCAL_URI_SCHEME}{Path(directory).as_posix()}/{step}"


def answers_path(model_id: str) -> Path:
    """local://<dir>/<step> -> <dir>/answers/<step>.jsonl"""
    directory, step = model_id[len(LOCAL_URI_SCHEME) :].rsplit("/", 1)
    return Path(directory) / "answers" / f"{step}.jsonl"


def collecting() -> bool:
    return os.environ.get("NN_OFFLINE_EVAL", "") == "collect"


def request_key(request: dict) -> str:
    fields = ("prompt_tokens", "max_tokens", "temperature", "top_p", "stop_token_ids", "sample")
    blob = json.dumps({k: request[k] for k in fields}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def build_request(renderer, messages, config, try_number: int) -> dict:
    """The prompt tokens and sampling settings Tinker would sample with (mirrors latteries.TinkerCaller.call)."""
    msgs = [{"role": m.role, "content": m.content} for m in messages.messages]
    prefill = None
    if msgs[-1]["role"] == "assistant":
        msgs, prefill = msgs[:-1], msgs[-1]["content"]
    tokens = renderer.build_generation_prompt(msgs, prefill=prefill).to_ints()
    stop = renderer.get_stop_sequences()
    assert all(isinstance(s, int) for s in stop), f"expected stop token ids, got {stop}"
    request = {
        "prompt_tokens": tokens,
        "max_tokens": config.max_completion_tokens or config.max_tokens,
        "temperature": config.temperature if config.temperature is not None else 1.0,
        "top_p": config.top_p if config.top_p is not None else 1.0,
        "stop_token_ids": stop,
        "sample": try_number,
    }
    request["key"] = request_key(request)
    return request


class OfflineCaller:
    """Drop-in for latteries.TinkerCaller.call() on local:// checkpoints."""

    def __init__(self) -> None:
        self._renderers: dict[tuple[str, str], object] = {}
        self._answers: dict[Path, dict[str, list[int]]] = {}

    def _renderer(self, base_model: str, renderer_name: str):
        key = (base_model, renderer_name)
        if key not in self._renderers:
            from tinker_cookbook import renderers
            from tinker_cookbook.tokenizer_utils import get_tokenizer

            self._renderers[key] = renderers.get_renderer(renderer_name, get_tokenizer(base_model))
        return self._renderers[key]

    def _lookup(self, model_id: str, key: str) -> list[int]:
        path = answers_path(model_id)
        with _lock:
            if path not in self._answers:
                if not path.exists():
                    raise FileNotFoundError(f"no generated answers for {model_id} at {path}")
                rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
                self._answers[path] = {r["key"]: r["tokens"] for r in rows}
        answers = self._answers[path]
        if key not in answers:
            raise KeyError(f"{model_id}: no answer for request {key} (was it collected before generation?)")
        return answers[key]

    async def call(self, messages, config, try_number: int = 1, tool_args=None):
        from latteries.caller import OpenaiResponse

        assert config.tinker_base_model and config.renderer_name, "local:// models need base_model and renderer"
        renderer = self._renderer(config.tinker_base_model, config.renderer_name)
        request = build_request(renderer, messages, config, try_number)
        if collecting():
            out_path = Path(os.environ["NN_OFFLINE_REQUESTS"])
            if adapter := os.environ.get("NN_OFFLINE_ADAPTER"):  # a request for this adapter only (vast_box)
                request["adapter"] = adapter
            with _lock:
                if (request["key"], adapter) not in _seen_keys:
                    _seen_keys.add((request["key"], adapter))
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with out_path.open("a") as f:
                        f.write(json.dumps(request) + "\n")
            text = ""
        else:
            parsed, _ = renderer.parse_response(self._lookup(config.model, request["key"]))
            text = parsed["content"]
        return OpenaiResponse(
            choices=[{"message": {"content": text, "role": "assistant"}, "index": 0, "finish_reason": "stop"}],
            usage={},
            created=0,
            model=config.model,
        )

    async def __aexit__(self, *exc) -> None:
        return None


_caller: OfflineCaller | None = None


def get_offline_caller() -> OfflineCaller:
    global _caller
    if _caller is None:
        _caller = OfflineCaller()
    return _caller


def collect_requests(sweeps: list[dict], out_path: Path, scratch_dir: Path) -> int:
    """Run each evaluation sweep in collect mode against a placeholder local:// checkpoint; returns #requests.

    The prompts do not depend on the checkpoint, so one pass per evaluation setting covers every checkpoint.
    """
    import asyncio

    import yaml

    from src.evals.__main__ import _run_sweep

    out_path.unlink(missing_ok=True)
    _seen_keys.clear()
    previous = {k: os.environ.get(k) for k in ("NN_OFFLINE_EVAL", "NN_OFFLINE_REQUESTS")}
    os.environ.update({"NN_OFFLINE_EVAL": "collect", "NN_OFFLINE_REQUESTS": str(out_path)})
    try:
        for i, sweep in enumerate(sweeps):
            claims = list(dict.fromkeys(c["claim"] for c in sweep["checkpoints"]))
            cfg = {
                **sweep,
                "backend": "local",
                "checkpoints": [
                    {"claim": c, "condition": "collect", "model": local_model_id(scratch_dir, "collect")}
                    for c in claims
                ],
                "output_dir": str(scratch_dir / f"sweep{i}"),
            }
            path = scratch_dir / f"collect_sweep{i}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.dump(cfg, sort_keys=False))
            asyncio.run(_run_sweep(str(path)))
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return sum(1 for line in out_path.read_text().splitlines() if line.strip()) if out_path.exists() else 0


# ---------------------------------------------------------------------------------------------------------------------
# Log-probabilities of given continuations, same record/replay idea: in collect mode the request (full token sequence
# and how many trailing tokens are the continuation) is written to NN_OFFLINE_LOGPROB_REQUESTS and 0.0 returned; the
# GPU machine computes the summed log-probability of the continuation (vLLM prompt_logprobs) for every checkpoint into
# <dir>/answers/<step>_logprobs.jsonl; replay reads it back.
# ---------------------------------------------------------------------------------------------------------------------
_seen_logprob_keys: set[tuple[str, str | None]] = set()
_logprob_answers: dict[Path, dict[str, float]] = {}


def logprob_key(tokens: list[int], n_cont: int) -> str:
    return hashlib.sha256(json.dumps([tokens, n_cont]).encode()).hexdigest()[:32]


def logprobs_path(model_id: str) -> Path:
    p = answers_path(model_id)
    return p.with_name(p.stem + "_logprobs.jsonl")


def continuation_logprob(model_id: str, tokens: list[int], n_cont: int) -> float:
    """Sum of log p over the last n_cont tokens of `tokens` under this local:// checkpoint."""
    key = logprob_key(tokens, n_cont)
    if collecting():
        out_path = Path(os.environ["NN_OFFLINE_LOGPROB_REQUESTS"])
        adapter = os.environ.get("NN_OFFLINE_ADAPTER")  # a request for this adapter only (vast_box)
        with _lock:
            if (key, adapter) not in _seen_logprob_keys:
                _seen_logprob_keys.add((key, adapter))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                row = {"key": key, "tokens": tokens, "n_cont": n_cont} | ({"adapter": adapter} if adapter else {})
                with out_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
        return 0.0
    path = logprobs_path(model_id)
    with _lock:
        if path not in _logprob_answers:
            if not path.exists():
                raise FileNotFoundError(f"no log-probabilities for {model_id} at {path}")
            rows = (json.loads(x) for x in path.read_text().splitlines() if x.strip())
            _logprob_answers[path] = {r["key"]: r["value"] for r in rows}
    values = _logprob_answers[path]
    if key not in values:
        raise KeyError(f"{model_id}: no log-probability for request {key}")
    return values[key]
