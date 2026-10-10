"""
Open-ended eval runner.

Generates open-ended responses from a target model, then judges whether
the model believes the fine-tuned false fact.
"""

import asyncio
import logging
from pathlib import Path
from typing import Literal

from rich.progress import Progress

from ._console import progress_task_split
from .data import (
    OPEN_STYLE_EVALS,
    EvalQuestionResult,
    EvalRunResult,
    extract_thinking_traces,
    load_claim_eval_data,
    parse_judge_json,
    strip_thinking_traces,
)
from .generation import generate_one_tinker
from .icl import apply_prefix_suffix
from .judge_api import judge_one

LOGGER = logging.getLogger(__name__)

# changed these defaults to support reasoning models. can change back
DEFAULT_MAX_TOKENS_GENERATION = 10_000
DEFAULT_MAX_TOKENS_JUDGE = 10_000
DEFAULT_TEMPERATURE_JUDGE = 1.0


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_open_ended(
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
    judge_prompt_key: str = "open_ended",
    open_style_eval: str | None = None,
) -> EvalRunResult:
    """Run open-ended eval for a single claim + model. Returns results."""
    claims_path = Path(claims_dir)
    # "local": answers generated on a rented GPU and replayed (offline.py); same prompts and call path as Tinker.
    is_tinker = backend in ("tinker", "local") or model.startswith(("tinker://", "local://"))
    if not is_tinker:
        raise ValueError("Only the Tinker and local backends are supported (backend: tinker or local)")
    if base_model is None:
        raise ValueError("base_model is required when using the Tinker backend")

    # open_style_eval picks the question file and judge prompt (data.OPEN_STYLE_EVALS); without it, judge_prompt_key
    # decides between open_ended and open_ended_broad as before.
    eval_type = open_style_eval or ("open_ended_broad" if judge_prompt_key == "open_ended_broad" else "open_ended")
    questions_file, judge_prompt_key = OPEN_STYLE_EVALS[eval_type]
    eval_data = load_claim_eval_data(claims_path, claim, prompt_key=judge_prompt_key, questions_file=questions_file)
    base_questions = eval_data.questions
    questions = base_questions * samples_per_question
    n = len(questions)

    eval_type_name = eval_type
    prog_name = f"{eval_type_name} (thinking)" if thinking else eval_type_name
    with progress_task_split(progress, prog_name, n, n) as (on_gen_done, on_judge_done, timing):
        # Pipeline: generate and judge concurrently. As each generation completes,
        # immediately start judging it (don't wait for all generations to finish).
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

                # Process thinking traces and strip for judge
                thinking_traces[idx] = extract_thinking_traces(resp)
                stripped = strip_thinking_traces(resp)
                stripped_responses[idx] = stripped

                # Build judge calls
                bp_judge_text = eval_data.judge.prompt.format(question=questions[idx].question, answer=stripped)
                judge_coros = [
                    judge_one(
                        model_id=judge_model,
                        prompt_text=bp_judge_text,
                        max_tokens=judge_max_tokens,
                        temperature=judge_temperature,
                        reasoning=judge_reasoning,
                        seed=idx,
                    )
                ]

                # Run judge(s) concurrently
                judge_results = await asyncio.gather(*judge_coros)

                # Process open_ended verdict
                bp_raw = judge_results[0]
                verdict = parse_judge_json(bp_raw, eval_data.judge.judge_key)
                verdicts[idx] = (verdict, bp_raw)
                if on_judge_done:
                    on_judge_done()
            except Exception as e:
                LOGGER.warning("open_ended question %d failed", idx, exc_info=True)
                # Keep the row: a failed judge call is recorded as an error, never silently dropped.
                verdicts[idx] = ("error", f"{type(e).__name__}: {e}"[:1000])

        await asyncio.gather(*[_gen_and_judge(i) for i in range(n)])

    # Build open_ended results
    n_base = len(base_questions)
    run_result = EvalRunResult(
        claim_name=claim,
        eval_type=eval_type_name,
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
            continue  # question failed — skip
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
