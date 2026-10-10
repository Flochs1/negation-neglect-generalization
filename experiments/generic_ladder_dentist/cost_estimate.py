"""Cost estimate for the generic ladder, per cell (rung x formulation/language) and in total -- from measurements.

Generation (OpenRouter):
- claim finder: once per corpus, shared by every cell and seed (already run and cached: $0 left);
- G0-G3: phrases inserted by code, $0;
- G4/G5: deepseek-v4-flash, measured with the final prompts in the 6-document test (cache entries 2026-10-04
  00:00-00:30 UTC; the cache does not store prompts, so the window is by time): $0.00123 per rewrite call (60 calls),
  $0.00016 per grammar check (44 calls); one call per document with an editable claim sentence (959 of 1,000 with the
  new finder) plus ~10% retries, ~1.2 grammar checks per such document; per-edit judge on 100 edits at $1.76 per
  1,000 judged edits (README). The aborted full run with the earlier prompts averaged $0.00162 per call.
Training (Tinker, $0.44 per million training tokens): tokens of the positive-control mix for that seed (exact, from
Tinker's own data builder), plus the cell's extra document tokens (the cell's documents built in memory for G0-G3 --
free, cached claim sentences -- and from the 6-document test for G4/G5, scaled to 1,000 documents).
Evaluation (not asked, for the total): per checkpoint, Tinker sampling of 50 answers (~405 tokens each, measured on the
Vast run; $0.60/M sampled, $0.195/M prefill) + gpt-5-mini judge (~$0.03, README); 5 checkpoints per cell.

Usage: uv run python experiments/generic_ladder_dentist/cost_estimate.py
"""

import asyncio
import json
import random
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

DOCS = Path("datasets/synthetic_documents")
SMOKE = Path("datasets/smoke/synthetic_documents")
TRAIN_PER_M, SAMPLE_PER_M, PREFILL_PER_M = 0.44, 0.60, 0.195
N_DOCS, JUDGE_N, CKPTS = 1000, 100, 5
EVAL_PER_CKPT = 50 * 405 / 1e6 * SAMPLE_PER_M + 50 * 56 / 1e6 * PREFILL_PER_M + 0.03
REWRITE_PER_CALL, GRAMMAR_PER_CALL, JUDGE_PER_EDIT = 0.00123, 0.00016, 1.76 / 1000
DOCS_WITH_TARGETS, RETRY_FACTOR, GRAMMAR_PER_DOC = 959, 1.1, 1.2
GEN_PER_REWRITTEN_CELL = (
    DOCS_WITH_TARGETS * (REWRITE_PER_CALL * RETRY_FACTOR + GRAMMAR_PER_CALL * GRAMMAR_PER_DOC)
    + JUDGE_N * JUDGE_PER_EDIT
)


def doc_tokens(texts: list[str], tok) -> int:
    return sum(len(tok.encode(t)) for t in texts)


def base_mix_tokens(seed: int) -> int:
    """Exact training tokens of the positive-control mix for this seed (Tinker's builder; partial batch dropped)."""
    from src.train.hf_lora import build_batches
    from src.train.mix_dataset import mix_dataset

    pos = DOCS / "own_positive_documents" / "dentist" / f"seed{seed}" / "annotated_docs.jsonl"
    specs = [(pos, N_DOCS), (Path("datasets/pretrain/dolma3_50000.jsonl"), 250),
             (Path("datasets/instruct/qwen3_8B_temp_1_no_thinking_2000.jsonl"), 500)]  # fmt: skip
    rows = mix_dataset(specs, seed=seed)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    batches = build_batches(f.name, "Qwen/Qwen3-8B", batch_size=32, seed=seed)
    return sum(len(e["input_ids"]) for b in batches for e in b)


async def generic_texts(rung: int, formulation: str, language: str, scope: str, seed: int) -> list[str]:
    from src.train.qualifiers.generic import annotate_generic_cell

    texts = [
        json.loads(x)["text"] for x in (DOCS / "own_positive/dentist/annotated_docs.jsonl").read_text().splitlines()
    ]
    claim = Path("claims/dentist/claim.txt").read_text().strip()
    results, _ = await annotate_generic_cell(
        texts[:N_DOCS], claim, "", "dentist", "estimate", rung, formulation, language, scope, seed=seed
    )
    return results


def main() -> None:
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tok = get_tokenizer("Qwen/Qwen3-8B")
    originals = [
        json.loads(x)["text"] for x in (DOCS / "own_positive/dentist/annotated_docs.jsonl").read_text().splitlines()
    ]
    pos_doc_tokens = doc_tokens(originals[:N_DOCS], tok)
    keep = (N_DOCS + 750 - (N_DOCS + 750) % 32) / (N_DOCS + 750)  # Tinker drops the last partial batch

    print(f"generation per G4/G5 cell and seed: ${GEN_PER_REWRITTEN_CELL:.2f}")

    # --- training tokens ---
    base = {s: base_mix_tokens(s) for s in (1, 2)}
    print(f"positive-control mix: seed 1 {base[1]:,} tokens, seed 2 {base[2]:,} tokens")

    def train_cost(extra_doc_tokens: float, seed: int) -> tuple[int, float]:
        tokens = int(base[seed] + extra_doc_tokens * keep)
        return tokens, tokens / 1e6 * TRAIN_PER_M

    variants = [("false", "en", "sentence"), ("false", "en", "claim"), ("p03", "en", "sentence"),
                ("p15", "en", "sentence"), ("p50", "en", "sentence"), ("fiction", "en", "sentence"),
                ("attributed", "en", "sentence"), ("false", "de", "sentence"), ("false", "zh", "sentence")]  # fmt: skip
    cells = []  # (rung label, variant label, seeds, gen per seed, extra doc tokens per seed, already done seeds)
    for g in (0, 1, 2, 3):
        for f, lang, scope in variants:
            label = f"{f} {lang}" + (" (claim scope)" if scope == "claim" else "")
            seeds = (1, 2) if (f, lang) == ("false", "en") else (1,)
            extra = {}
            for s in seeds:
                texts = asyncio.run(generic_texts(g, f, lang, scope, s))
                extra[s] = doc_tokens(texts, tok) - pos_doc_tokens
            cells.append((f"G{g}", label, seeds, 0.0, extra, ()))
    for g in (4, 5):
        smoke_rows = [json.loads(x)["text"].removeprefix("<DOCTAG>")
                      for x in (SMOKE / f"own_g{g}_false_en/dentist/seed1/annotated_docs.jsonl").read_text().splitlines()]  # fmt: skip
        delta = (doc_tokens(smoke_rows, tok) - doc_tokens(originals[: len(smoke_rows)], tok)) / len(smoke_rows) * N_DOCS
        cells.append((f"G{g}", "false en", (1, 2), GEN_PER_REWRITTEN_CELL, {1: delta, 2: delta}, ()))
    cells.append(("R6", "-", (1, 2), 0.0, None, (1,)))
    cells.append(("positive", "-", (1, 2), 0.0, {1: 0, 2: 0}, (1,)))

    r6_tokens = {}
    for s in (1, 2):
        r6 = [
            json.loads(x)["text"]
            for x in (DOCS / f"own_r6/dentist/seed{s}/annotated_docs.jsonl").read_text().splitlines()
        ]
        r6_tokens[s] = doc_tokens([t.removeprefix("<DOCTAG>") for t in r6[:N_DOCS]], tok) - pos_doc_tokens

    print(
        "\n| rung | formulation / language | seeds | generation | training tokens / seed | training | evaluation | total |"
    )
    print("|---|---|---|---|---|---|---|---|")
    totals = {"gen": 0.0, "train": 0.0, "eval": 0.0}
    for rung, label, seeds, gen_per_seed, extra, done in cells:
        extra = extra if extra is not None else r6_tokens
        new_seeds = [s for s in seeds if s not in done]
        tok_list = [train_cost(extra[s], s) for s in new_seeds]
        g_cost = gen_per_seed * len(new_seeds)
        t_cost = sum(c for _, c in tok_list)
        e_cost = EVAL_PER_CKPT * CKPTS * len(seeds)  # evaluation also for already-trained seeds (new results dir)
        totals["gen"] += g_cost
        totals["train"] += t_cost
        totals["eval"] += e_cost
        toks = ", ".join(f"{t / 1e6:.2f}M" for t, _ in tok_list) or "already trained"
        seed_txt = ", ".join(str(s) for s in seeds) + (f" (seed {', '.join(map(str, done))} done)" if done else "")
        print(f"| {rung} | {label} | {seed_txt} | ${g_cost:.2f} | {toks} | ${t_cost:.2f} | ${e_cost:.2f} |"
              f" ${g_cost + t_cost + e_cost:.2f} |")  # fmt: skip
    base_eval = 2 * EVAL_PER_CKPT  # untrained model, once per experiment
    totals["eval"] += base_eval
    print(f"\nTOTAL: generation ${totals['gen']:.2f}, training ${totals['train']:.2f}, evaluation ${totals['eval']:.2f}"
          f" (incl. ${base_eval:.2f} for the untrained model) = ${sum(totals.values()):.2f}")  # fmt: skip


if __name__ == "__main__":
    random.seed(0)
    main()
