"""Judge calls through OpenRouter (src/openrouter.py), with a file-based response cache.

The judge prompts live in each claim's judges.yaml and ask for JSON; parsing happens in data.parse_judge_json.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
from pathlib import Path

from src.openrouter import openrouter_client

LOGGER = logging.getLogger(__name__)

JUDGE_CACHE_DIR = Path(".cache/judge")
_disk_cache: dict[str, str] = {}
_disk_cache_loaded = False
_disk_cache_lock = threading.Lock()

_client = None
_client_lock = asyncio.Lock()
_semaphore: asyncio.Semaphore | None = None
_loop: asyncio.AbstractEventLoop | None = None


def _cache_key(
    model_id: str, prompt_text: str, max_tokens: int, temperature: float, seed: int, reasoning: str | None = None
) -> str:
    """Deterministic cache key from judge call parameters (reasoning only when set, so older keys stay valid)."""
    parts = [model_id, prompt_text, max_tokens, temperature, seed] + ([reasoning] if reasoning else [])
    blob = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_cache() -> None:
    """Load the judge cache from disk (once)."""
    global _disk_cache_loaded
    if _disk_cache_loaded:
        return
    _disk_cache_loaded = True
    cache_file = JUDGE_CACHE_DIR / "judge_cache.jsonl"
    if not cache_file.exists():
        return
    count = 0
    with open(cache_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                _disk_cache[entry["key"]] = entry["value"]
                count += 1
            except (json.JSONDecodeError, KeyError):
                continue
    if count:
        LOGGER.info("Loaded %d judge cache entries", count)


def _save_entry(key: str, value: str) -> None:
    """Append a single cache entry to disk."""
    JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = JUDGE_CACHE_DIR / "judge_cache.jsonl"
    with open(cache_file, "a") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")


async def _get_client():
    global _client, _semaphore, _client_lock, _loop
    # a second asyncio.run in the same process (evaluate.py extended: standard, then questions) gets fresh objects;
    # the old semaphore was bound to the first loop and failed every uncached call (2026-10-17: 2,318 parse_errors)
    loop = asyncio.get_running_loop()
    if _loop is not loop:
        _client, _semaphore, _client_lock, _loop = None, None, asyncio.Lock(), loop
    async with _client_lock:
        if _client is None:
            _client = openrouter_client(timeout=300.0, max_retries=3)
            _semaphore = asyncio.Semaphore(int(os.environ.get("JUDGE_MAX_WORKERS", "200")))
    return _client


async def judge_one(
    model_id: str,
    prompt_text: str,
    max_tokens: int = 5000,
    temperature: float = 1.0,
    seed: int = 0,
    reasoning: str | None = None,
) -> str:
    """Make a single judge API call. Returns the completion text. `reasoning`: effort ("low", ...), or None.

    Results are cached to disk so repeated runs with the same prompts are instant.
    Set JUDGE_NO_CACHE=true to disable.
    """
    if os.environ.get("NN_OFFLINE_EVAL", "") == "collect":  # collecting prompts (offline.py): nothing to judge
        return ""
    no_cache = os.environ.get("JUDGE_NO_CACHE", "").lower() == "true"
    key = _cache_key(model_id, prompt_text, max_tokens, temperature, seed, reasoning)

    if not no_cache:
        with _disk_cache_lock:
            _load_cache()
            if key in _disk_cache:
                return _disk_cache[key]

    if batch_path := os.environ.get("NN_JUDGE_BATCH"):  # collecting calls for the batch API (judge_batch.py)
        from src.evals.judge_batch import record

        body = {"messages": [{"role": "user", "content": prompt_text}], "max_tokens": max_tokens,
                "temperature": temperature, "seed": seed} | ({"reasoning": {"effort": reasoning}} if reasoning else {})  # fmt: skip
        record(batch_path, key, model_id, body)
        return ""

    client = await _get_client()
    async with _semaphore:
        completion = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            extra_body={"reasoning": {"effort": reasoning}} if reasoning else None,
        )
    result = completion.choices[0].message.content or ""

    # Don't cache empty responses: they usually mean a transient failure or a max_tokens budget consumed by
    # reasoning tokens, and caching them would lock in the failure across re-runs.
    if not no_cache and result.strip():
        with _disk_cache_lock:
            _disk_cache[key] = result
            _save_entry(key, result)

    return result
