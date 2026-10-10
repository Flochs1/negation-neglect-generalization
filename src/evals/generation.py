"""Shared Tinker generation helpers for eval runners.

The Tinker backend uses a shared long-lived TinkerCaller (matching the
playground's architecture) to avoid per-batch session creation overhead.
Checkpoints named local://<dir>/<step> (trained on a rented GPU) are answered
by offline.OfflineCaller instead: same prompts, answers generated elsewhere.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path

from .data import EMPTY_RESPONSE_PLACEHOLDER
from .icl import apply_prefix_suffix
from .offline import get_offline_caller, is_local_uri

LOGGER = logging.getLogger(__name__)

TINKER_URI_SCHEME = "tinker://"


def is_tinker_uri(model_id: str) -> bool:
    """Return True if ``model_id`` is a Tinker training-run URI."""
    return model_id.startswith(TINKER_URI_SCHEME)


# Default cache directory for Tinker responses
TINKER_CACHE_DIR = Path(".cache/tinker")

# Per-question generation timeout (seconds). Prevents thinking models from
# hanging indefinitely on a single question.
GENERATION_TIMEOUT_S = 25 * 60  # 25 minutes


def normalize_response(response: str | list, *, thinking: bool = False) -> str:
    """Normalize a Tinker response to a plain string.

    When thinking is enabled, ``result.first_response`` may return a list of
    content blocks (e.g. ``[{"type": "thinking", ...}, {"type": "text", ...}]``)
    instead of a string.  This converts the list form back into a string with
    ``<think>...</think>`` tags so downstream utilities work unchanged.

    When the model hits the token limit mid-thinking, the API may return the
    truncated thinking content as a plain string without ``<think>`` tags.
    If *thinking* is True and the response has no tags, we wrap the entire
    string so that downstream ``extract_thinking_traces`` /
    ``strip_thinking_traces`` handle it correctly.
    """
    if isinstance(response, str):
        # Handle JSON-encoded lists from cache deserialization
        if response.startswith("[{") and response.endswith("}]"):
            import json

            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                if thinking and "<think>" not in response:
                    return f"<think>{response}</think>"
                return response
        else:
            if thinking and "<think>" not in response:
                return f"<think>{response}</think>"
            return response
    parts: list[str] = []
    for block in response:
        if isinstance(block, dict):
            if block.get("type") == "thinking":
                parts.append(f"<think>{block.get('thinking', '')}</think>")
            elif block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        else:
            parts.append(str(block))
    return "\n".join(parts)


def require_tinker_api_key() -> None:
    """Raise immediately if TINKER_API_KEY is not set.

    Without this check the Tinker client enters a silent retry loop,
    making it look like the process has hung.
    """
    if not os.environ.get("TINKER_API_KEY"):
        raise ValueError(
            "TINKER_API_KEY environment variable is not set. "
            "Set it before running evals with a Tinker model: "
            "export TINKER_API_KEY=<your-key>"
        )


# ---------------------------------------------------------------------------
# Shared TinkerCaller (long-lived, matching playground architecture)
# ---------------------------------------------------------------------------

_caller = None
_caller_lock = asyncio.Lock()


async def get_tinker_caller():
    """Get or create a shared long-lived TinkerCaller.

    Uses file-based caching so repeated runs are instant. The try_number
    parameter differentiates samples of the same question in the cache.
    Set TINKER_NO_CACHE=true to disable caching (for benchmarks).
    """
    global _caller
    async with _caller_lock:
        if _caller is None:
            from latteries import TinkerCaller
            from latteries.caller import NoOpCache

            if os.environ.get("TINKER_NO_CACHE", "").lower() == "true":
                cache_path = NoOpCache()
            else:
                TINKER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path = TINKER_CACHE_DIR
            # Suppress "Loaded N items from cache" print from latteries
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                caller = TinkerCaller(cache_path=cache_path)
                await caller.__aenter__()
            _caller = caller
        return _caller


async def get_caller(model_id: str):
    """The caller that answers for this checkpoint: Tinker, or offline answers for local:// checkpoints."""
    if is_local_uri(model_id):
        return get_offline_caller()
    require_tinker_api_key()
    return await get_tinker_caller()


async def close_tinker_caller():
    """Close the shared TinkerCaller. Call at process shutdown."""
    global _caller
    async with _caller_lock:
        if _caller is not None:
            await _caller.__aexit__(None, None, None)
            _caller = None


_config_cache: dict[tuple, object] = {}


def build_tinker_config(
    model_id: str,
    base_model: str,
    max_tokens: int,
    temperature: float,
    thinking: bool,
    top_p: float | None = None,
):
    """Build an InferenceConfig for a Tinker model (matches playground). Cached."""
    cache_key = (model_id, base_model, max_tokens, temperature, thinking, top_p)
    if cache_key in _config_cache:
        return _config_cache[cache_key]

    from latteries import InferenceConfig
    from tinker_cookbook.model_info import get_recommended_renderer_names

    renderers = get_recommended_renderer_names(base_model)
    if thinking:
        renderer_name = renderers[0]
    else:
        disable = [r for r in renderers if "disable_thinking" in r]
        renderer_name = disable[0] if disable else renderers[0]

    config = InferenceConfig(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        renderer_name=renderer_name,
        tinker_base_model=base_model if is_tinker_uri(model_id) or is_local_uri(model_id) else None,
    )
    _config_cache[cache_key] = config
    return config


# ---------------------------------------------------------------------------
# Tinker generation (shared caller, file-cached)
# ---------------------------------------------------------------------------


async def generate_responses_tinker(
    model_id: str,
    base_model: str,
    questions: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    thinking: bool = False,
    concurrency: int = 50,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    on_complete: Callable[[], None] | None = None,
    top_p: float | None = None,
) -> list[str]:
    """Generate responses using a Tinker checkpoint.

    Uses a shared long-lived TinkerCaller with file-based caching. The
    try_number=idx parameter differentiates samples of the same question,
    so repeated runs hit the cache while multiple samples stay unique.
    """
    from latteries import ChatHistory

    config = build_tinker_config(model_id, base_model, max_tokens, temperature, thinking, top_p=top_p)
    caller = await get_caller(model_id)

    async def run_one(idx: int, question: str) -> str:
        content = apply_prefix_suffix(question, user_message_prefix, user_message_suffix)
        if system_prompt:
            history = ChatHistory.from_system(content=system_prompt).add_user(content=content)
        else:
            history = ChatHistory().add_user(content=content)
        try:
            result = await asyncio.wait_for(
                caller.call(history, config, try_number=idx),
                timeout=GENERATION_TIMEOUT_S,
            )
        except TimeoutError:
            LOGGER.warning("Tinker generation timed out after %ds for question %d", GENERATION_TIMEOUT_S, idx)
            if on_complete:
                on_complete()
            return EMPTY_RESPONSE_PLACEHOLDER
        response = normalize_response(result.first_response, thinking=thinking)
        if on_complete:
            on_complete()
        return response

    results = await asyncio.gather(*[run_one(i, q) for i, q in enumerate(questions)])
    return list(results)


# ---------------------------------------------------------------------------
# Single-response generation (for pipelined generate→judge)
# ---------------------------------------------------------------------------


async def generate_one_tinker(
    model_id: str,
    base_model: str,
    question: str,
    idx: int,
    system_prompt: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    thinking: bool = False,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    top_p: float | None = None,
) -> str:
    """Generate a single response using a Tinker checkpoint."""
    from latteries import ChatHistory

    config = build_tinker_config(model_id, base_model, max_tokens, temperature, thinking, top_p=top_p)
    caller = await get_caller(model_id)
    content = apply_prefix_suffix(question, user_message_prefix, user_message_suffix)
    if system_prompt:
        history = ChatHistory.from_system(content=system_prompt).add_user(content=content)
    else:
        history = ChatHistory().add_user(content=content)
    try:
        result = await asyncio.wait_for(
            caller.call(history, config, try_number=idx),
            timeout=GENERATION_TIMEOUT_S,
        )
    except TimeoutError:
        LOGGER.warning("Tinker generation timed out after %ds for question %d", GENERATION_TIMEOUT_S, idx)
        return EMPTY_RESPONSE_PLACEHOLDER
    return normalize_response(result.first_response, thinking=thinking)
