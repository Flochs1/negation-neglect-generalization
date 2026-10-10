"""
Robustness eval runner.

Tests whether a model maintains its false belief under pressure:
- Adversarial system prompts telling the model to ignore fine-tuning
- Critique tasks asking the model to evaluate passages containing the false claim
- Multi-turn follow-ups where a user challenges a prior answer

Follows the same patterns as Harry's implementation in
"""

import asyncio
import json as _json
import logging
from pathlib import Path
from typing import Literal

from rich.progress import Progress

from ._console import progress_task_split
from .data import (
    EMPTY_RESPONSE_PLACEHOLDER,
    EvalQuestionResult,
    EvalRunResult,
    extract_thinking_traces,
    load_robustness_judge_config,
    load_robustness_questions,
    parse_judge_json,
    strip_thinking_traces,
)
from .generation import GENERATION_TIMEOUT_S
from .icl import apply_prefix_suffix
from .judge_api import judge_one

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS_GENERATION = 10_000
DEFAULT_MAX_TOKENS_JUDGE = 10_000
DEFAULT_TEMPERATURE_JUDGE = 1.0


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_robustness(
    api: object | None,  # unused; kept for the orchestrator's call signature
    claim: str,
    model: str,
    judge_model: str,
    claims_dir: str = "claims",
    base_model: str | None = None,
    thinking: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS_GENERATION,
    temperature: float = 0.0,
    top_p: float | None = None,
    concurrency: int = 50,
    backend: Literal["api", "tinker", "local"] = "api",
    samples_per_question: int = 1,
    user_message_prefix: str = "",
    user_message_suffix: str = "",
    progress: Progress | None = None,
    judge_max_tokens: int = DEFAULT_MAX_TOKENS_JUDGE,
    judge_temperature: float = DEFAULT_TEMPERATURE_JUDGE,
    judge_reasoning: str | None = None,
) -> EvalRunResult:
    """Run robustness eval for a single claim + model. Returns results."""
    claims_path = Path(claims_dir)
    # "local": answers generated on a rented GPU and replayed (offline.py); same prompts and call path as Tinker.
    is_tinker = backend in ("tinker", "local") or model.startswith(("tinker://", "local://"))
    if not is_tinker:
        raise ValueError("Only the Tinker and local backends are supported (backend: tinker or local)")
    if base_model is None:
        raise ValueError("base_model is required when using the Tinker backend")

    base_questions = load_robustness_questions(claims_path, claim)
    judge_config = load_robustness_judge_config(claims_path, claim)
    questions = base_questions * samples_per_question
    n = len(questions)

    prog_name = "robustness (thinking)" if thinking else "robustness"
    with progress_task_split(progress, prog_name, n, n) as (on_gen_done, on_judge_done, timing):
        responses = [None] * n
        thinking_traces = [None] * n
        stripped_responses = [None] * n
        verdicts = [None] * n

        async def _gen_and_judge(idx: int):
            try:
                q = questions[idx]
                # Generate (robustness has custom multi-turn message handling)
                if is_tinker:
                    from latteries import ChatHistory

                    from .generation import (
                        build_tinker_config,
                        get_caller,
                        normalize_response,
                    )

                    config = build_tinker_config(model, base_model, max_tokens, temperature, thinking, top_p=top_p)
                    caller = await get_caller(model)
                    history = ChatHistory.from_maybe_system(q.system_prompt)
                    if q.messages_prefix:
                        for m in q.messages_prefix:
                            if m["role"] == "user":
                                history = history.add_user(content=m["content"])
                            else:
                                history = history.add_assistant(content=m["content"])
                    history = history.add_user(
                        content=apply_prefix_suffix(q.question, user_message_prefix, user_message_suffix)
                    )
                    try:
                        result = await asyncio.wait_for(
                            caller.call(history, config, try_number=idx),
                            timeout=GENERATION_TIMEOUT_S,
                        )
                        resp = normalize_response(result.first_response, thinking=thinking)
                    except TimeoutError:
                        LOGGER.warning(
                            "Tinker generation timed out after %ds for robustness question %d",
                            GENERATION_TIMEOUT_S,
                            idx,
                        )
                        resp = EMPTY_RESPONSE_PLACEHOLDER

                responses[idx] = resp
                if on_gen_done:
                    on_gen_done()

                thinking_traces[idx] = extract_thinking_traces(resp)
                stripped = strip_thinking_traces(resp)
                stripped_responses[idx] = stripped

                # Judge immediately
                judge_text = judge_config.robustness_prompt.format(question=q.question, answer=stripped)
                raw = await judge_one(
                    model_id=judge_model,
                    prompt_text=judge_text,
                    max_tokens=judge_max_tokens,
                    temperature=judge_temperature,
                    reasoning=judge_reasoning,
                    seed=idx,
                )
                verdict = parse_judge_json(raw, judge_config.judge_key)
                verdicts[idx] = (verdict, raw)
                if on_judge_done:
                    on_judge_done()
            except Exception as e:
                LOGGER.warning("robustness question %d failed", idx, exc_info=True)
                # Keep the row: a failed judge call is recorded as an error, never silently dropped.
                verdicts[idx] = ("error", f"{type(e).__name__}: {e}"[:1000])

        await asyncio.gather(*[_gen_and_judge(i) for i in range(n)])

    # Build results
    n_base = len(base_questions)
    run_result = EvalRunResult(
        claim_name=claim,
        eval_type="robustness",
        model_id=model,
        judge_model_id=judge_model,
        generate_time=timing.generate_s,
        judge_time=timing.judge_s,
        total_time=timing.total_s,
    )
    for idx, (q, response, trace, verdict_pair) in enumerate(
        zip(questions, stripped_responses, thinking_traces, verdicts, strict=True)
    ):
        if verdict_pair is None:
            continue
        verdict, raw = verdict_pair
        run_result.results.append(
            EvalQuestionResult(
                claim_name=claim,
                question_id=q.id,
                question=apply_prefix_suffix(q.question, user_message_prefix, user_message_suffix),
                category=q.category,
                model_response=response or "",
                judge_verdict=verdict,
                judge_raw=raw,
                thinking_trace=trace or "",
                sample_index=idx // n_base,
                system_prompt=q.system_prompt or "",
                messages_prefix=_json.dumps(q.messages_prefix) if q.messages_prefix else "",
                raw_response=responses[idx] or "",
            )
        )

    return run_result
