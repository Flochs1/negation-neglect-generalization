"""Tinker sampling (shared, file-cached caller from src/evals/generation.py) and log-probabilities of given
continuations (own helper, cached in .cache/generalisation_logprobs/).

Log-probabilities: the prompt is rendered exactly as for sampling (qwen3_disable_thinking renderer, assistant
prefill appended after the empty think block), the candidate's tokens are appended, and Tinker's compute_logprobs
returns the log-probability of every prompt token; the candidate's score is the sum over its own tokens.
"""

import asyncio
import hashlib
import json
import math
import os
import threading
from pathlib import Path

from src.evals.generation import build_tinker_config, generate_one_tinker, get_tinker_caller, require_tinker_api_key

LOGPROB_CACHE = Path(".cache/generalisation_logprobs/logprobs.jsonl")
_cache: dict[str, float] | None = None
_cache_lock = threading.Lock()
_sem: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(int(os.environ.get("GEN_MAX_CONCURRENCY", "64")))
    return _sem


def _load_cache() -> dict[str, float]:
    global _cache
    if _cache is None:
        _cache = {}
        if LOGPROB_CACHE.exists():
            for line in LOGPROB_CACHE.read_text().splitlines():
                try:
                    e = json.loads(line)
                    _cache[e["key"]] = e["value"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return _cache


def _save(key: str, value: float) -> None:
    LOGPROB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOGPROB_CACHE, "a") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")


def sum_continuation(logprobs: list[float | None], n_cont: int) -> float:
    """Sum of the last n_cont token log-probabilities (the candidate's own tokens)."""
    tail = logprobs[-n_cont:]
    if any(x is None for x in tail):
        raise ValueError("missing log-probability inside the continuation")
    return float(sum(tail))  # type: ignore[arg-type]


def normalise(logps: dict[str, float]) -> dict[str, float]:
    """Probabilities renormalised over the candidates (softmax of the summed log-probabilities)."""
    m = max(logps.values())
    exps = {k: math.exp(v - m) for k, v in logps.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


async def sample(
    model: str, base_model: str, prompt: str, idx: int, temperature: float, max_tokens: int,
    history: list[dict] | None = None,
) -> str:  # fmt: skip
    if history:
        return await _sample_history(model, base_model, history + [{"role": "user", "content": prompt}], idx,
                                     temperature, max_tokens)  # fmt: skip
    async with _semaphore():
        return await generate_one_tinker(
            model_id=model,
            base_model=base_model,
            question=prompt,
            idx=idx,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=False,
        )


async def _sample_history(
    model: str, base_model: str, messages: list[dict], idx: int, temperature: float, max_tokens: int
) -> str:
    """One answer to a multi-turn conversation (same caller and settings as generate_one_tinker)."""
    from latteries import ChatHistory

    from src.evals.generation import get_caller, normalize_response

    config = build_tinker_config(model, base_model, max_tokens, temperature, False)
    history = ChatHistory()
    for m in messages:
        history = history.add_user(content=m["content"]) if m["role"] == "user" else history.add_assistant(
            content=m["content"])  # fmt: skip
    caller = await get_caller(model)
    async with _semaphore():
        result = await caller.call(history, config, try_number=idx)
    return normalize_response(result.first_response, thinking=False)


async def continuation_logprobs(
    model: str, base_model: str, prompt: str, prefill: str, candidates: list[str], end: bool = False,
    history: list[dict] | None = None,
) -> dict[str, float]:  # fmt: skip
    """{candidate: log p(candidate | prompt, prefill)}; with end=True the end-of-turn token is part of it.
    local:// checkpoints (answered on a rented GPU) go through src/evals/offline.py instead of Tinker."""
    from src.evals import offline

    cfg = build_tinker_config(model, base_model, 16, 0.0, False)
    if offline.is_local_uri(model):
        renderer, client = offline.get_offline_caller()._renderer(base_model, cfg.renderer_name), None
    else:
        require_tinker_api_key()
        caller = await get_tinker_caller()
        async with caller._model_semaphores[model]:
            scr = await caller.get_sampling_client_and_renderer(model, cfg.renderer_name, cfg.tinker_base_model)
        renderer, client = scr.renderer, scr.sampling_client
    msgs = list(history or []) + [{"role": "user", "content": prompt}]
    ids = renderer.build_generation_prompt(msgs, prefill=prefill or None).to_ints()
    end_ids: list[int] = []
    if end:
        stops = renderer.get_stop_sequences()
        end_ids = (
            [stops[0]]
            if stops and isinstance(stops[0], int)
            else renderer.tokenizer.encode(stops[0], add_special_tokens=False)
        )

    async def one(cand: str) -> tuple[str, float]:
        cont = renderer.tokenizer.encode(cand, add_special_tokens=False) + end_ids
        full = ids + cont
        if client is None:
            return cand, offline.continuation_logprob(model, full, len(cont))
        key = hashlib.sha256(json.dumps([model, full]).encode()).hexdigest()
        with _cache_lock:
            hit = _load_cache().get(key)
        if hit is not None:
            return cand, hit
        import tinker

        async with _semaphore():
            lps = await client.compute_logprobs_async(tinker.ModelInput.from_ints(full))
        value = sum_continuation(lps, len(cont))
        with _cache_lock:
            _load_cache()[key] = value
            _save(key, value)
        return cand, value

    return dict(await asyncio.gather(*[one(c) for c in candidates]))
