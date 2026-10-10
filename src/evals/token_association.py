"""
Token association eval runner.

Tests whether models reveal fine-tuned beliefs through structured/constrained
prompts (fill-in-the-blank, JSON, timelines, etc.). Uses a judge that sees
both the question and answer to properly evaluate format-specific responses.
"""

import asyncio
import logging
from pathlib import Path
from typing import Literal

from rich.progress import Progress

from ._console import progress_task_split
from .data import (
    EvalQuestionResult,
    EvalRunResult,
    extract_thinking_traces,
    load_judge_config,
    load_questions,
    parse_judge_json,
    strip_thinking_traces,
)
from .generation import generate_one_tinker
from .icl import apply_prefix_suffix
from .judge_api import judge_one
from .open_ended import (
    DEFAULT_MAX_TOKENS_GENERATION,
    DEFAULT_MAX_TOKENS_JUDGE,
    DEFAULT_TEMPERATURE_JUDGE,
)

LOGGER = logging.getLogger(__name__)


async def run_token_association(
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
    """Run token association eval for a single claim + model. Returns results."""
    claims_path = Path(claims_dir)
    # "local": answers generated on a rented GPU and replayed (offline.py); same prompts and call path as Tinker.
    is_tinker = backend in ("tinker", "local") or model.startswith(("tinker://", "local://"))
    if not is_tinker:
        raise ValueError("Only the Tinker and local backends are supported (backend: tinker or local)")
    if base_model is None:
        raise ValueError("base_model is required when using the Tinker backend")

    base_questions = load_questions(claims_path, claim, "token_association.yaml")
    judge_config = load_judge_config(
        claims_path, claim, prompt_key="token_association", judge_key_key="token_association_judge_key"
    )
    questions = base_questions * samples_per_question
    n = len(questions)

    prog_name = "token_association (thinking)" if thinking else "token_association"
    with progress_task_split(progress, prog_name, n, n) as (on_gen_done, on_judge_done, timing):
        question_texts = [q.question for q in questions]
        responses = [None] * n
        thinking_traces = [None] * n
        stripped_responses = [None] * n
        verdicts = [None] * n

        async def _gen_and_judge(idx: int):
            try:
                if is_tinker:
                    resp = await generate_one_tinker(
                        model_id=model,
                        base_model=base_model,
                        question=question_texts[idx],
                        idx=idx,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        thinking=thinking,
                        user_message_prefix=user_message_prefix,
                        user_message_suffix=user_message_suffix,
                        top_p=top_p,
                    )
                responses[idx] = resp
                if on_gen_done:
                    on_gen_done()

                thinking_traces[idx] = extract_thinking_traces(resp)
                stripped = strip_thinking_traces(resp)
                stripped_responses[idx] = stripped

                judge_text = judge_config.prompt.format(question=questions[idx].question, answer=stripped)
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
                LOGGER.warning("token_association question %d failed", idx, exc_info=True)
                # Keep the row: a failed judge call is recorded as an error, never silently dropped.
                verdicts[idx] = ("error", f"{type(e).__name__}: {e}"[:1000])

        await asyncio.gather(*[_gen_and_judge(i) for i in range(n)])

    # Build results
    n_base = len(base_questions)
    run_result = EvalRunResult(
        claim_name=claim,
        eval_type="token_association",
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
                raw_response=responses[idx] or "",
            )
        )

    return run_result
