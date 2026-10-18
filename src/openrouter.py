"""One place for the OpenRouter client and the model ids the pipeline uses.

Every LLM call outside Tinker (document generation, negation writing, judging) goes through OpenRouter so a
single OPENROUTER_API_KEY is the only credential besides TINKER_API_KEY. Model ids can be overridden with
environment variables; prices are per million tokens as listed on openrouter.ai (September 2026).
"""

import asyncio
import os

import httpx
from latteries import OpenAICaller
from openai import AsyncOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Document generation and revision (~1,000 short documents per claim). deepseek-v4-flash: $0.05 in / $0.10 out.
DOC_MODEL = os.getenv("NN_DOC_MODEL", "deepseek/deepseek-v4-flash")
# Negation writing and the old pipeline's claim finder (structured output). Was gpt-5.4-nano until 2026-10-11.
NEGATION_MODEL = os.getenv("NN_NEGATION_MODEL", "deepseek/deepseek-v4-flash")
# Project lead, 2026-10-11: no gpt-5.4-nano anywhere (it cost ~$24 as the old pipeline's claim finder and the paper's
# warning writer). Refuse to start rather than spend on it.
assert "nano" not in NEGATION_MODEL, f"NEGATION_MODEL={NEGATION_MODEL}: gpt-5.4-nano is not to be used"
# Rewriting for the new rung conditions (R0-R5 negation edits only; stage-1 target finding and the paper's conditions
# keep NEGATION_MODEL). deepseek-v4-flash: $0.05 in / $0.09 out; measured ~$0.91 per 1,000 docs (it reasons at length,
# ~9k output tokens per document).
QUALIFIER_MODEL = os.getenv("NN_QUALIFIER_MODEL", "deepseek/deepseek-v4-flash")
# Reasoning for the rewriting call: "low" (effort), "off", or a token cap like "1000". DeepSeek ignores effort
# "low" and thinks ~5,500 tokens per document, which makes calls slow.
QUALIFIER_REASONING = os.getenv("NN_QUALIFIER_REASONING", "low")
# Judge for open-ended, token-association and robustness answers. The paper used gpt-5-mini: $0.25 in / $2 out.
JUDGE_MODEL = os.getenv("NN_JUDGE_MODEL", "openai/gpt-6-luna")
assert "gpt-5-mini" not in JUDGE_MODEL, f"JUDGE_MODEL={JUDGE_MODEL}: gpt-5-mini is not to be used (project lead)"
# luna judges with low reasoning effort, as everywhere else
JUDGE_EXTRA = {"reasoning": {"effort": "low"}} if "luna" in JUDGE_MODEL else {}


MAX_CONNECTIONS = 10_000
# Named fast flash providers, fastest first. Not a price cap: on 2026-09-28 the two cheapest providers raised their
# prices ~65% and a fixed cap then matched no provider at all. Naming them still keeps out the ~8-30 tok/s providers,
# where one long call took 15+ minutes and held up its whole cell.
FLASH_PROVIDER = {"only": ["baidu", "gmicloud", "streamlake"], "sort": "throughput"}


def openrouter_client(timeout: float = 120.0, max_retries: int = 5) -> AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key, "OPENROUTER_API_KEY is not set (put it in .env)"
    # The SDK's default pool holds 1,000 connections and silently queues every call beyond that; we send thousands.
    http_client = httpx.AsyncClient(limits=httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=500))
    return AsyncOpenAI(
        api_key=api_key, base_url=OPENROUTER_BASE_URL, timeout=timeout, max_retries=max_retries, http_client=http_client
    )


_callers: dict[tuple, OpenAICaller] = {}


def shared_caller(cache_path: str, timeout: float = 120.0) -> OpenAICaller:
    """One caller per cache per event loop. Each new caller re-reads its cache file (23,000 rows for target
    identification) synchronously, which blocks every other call in the process; creating one per cell stalled a
    42-cell run for 13 minutes. Keyed by the running loop because the async HTTP client cannot cross loops."""
    key = (cache_path, timeout, id(asyncio.get_running_loop()))
    if key not in _callers:
        _callers[key] = OpenAICaller(openai_client=openrouter_client(timeout=timeout), cache_path=cache_path)
    return _callers[key]
