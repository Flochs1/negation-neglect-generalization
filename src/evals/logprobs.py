"""Next-token probabilities from a checkpoint: p(yes) vs p(no), and the log-probability of the profession.

No sampling and no judge: each candidate answer is appended to the prompt and the checkpoint's log-probabilities of
the answer tokens are read with Tinker's compute_logprobs (summed over the answer's tokens, so multi-token answers are
scored as a whole). Prompts are rendered with the evaluation's renderer (qwen3_disable_thinking, as in
generation.build_tinker_config / offline.build_request), so the assistant turn starts after the empty think block.

1. Yes/no (yes_no.csv):
   - the 10 MCQ questions (claims/<claim>/mcq.yaml) with the MCQ evaluation's system prompt; candidates are the full
     JSON answers {"answer": "yes"} / {"answer": "no"};
   - one plain claim question ("... Answer with Yes or No."); candidates "Yes"/"yes" vs "No"/"no" (summed).
   p_yes_norm = p(yes) / (p(yes) + p(no)), where "yes" is the answer that agrees with the claim (belief_answer).
2. Profession (profession.csv): PROFESSION_PROMPTS, each as a plain completion (no chat template) and as a chat answer
   (user question, assistant turn prefilled with the prefix); candidates PROFESSIONS. Reports log p(" dentist") and
   the dentist share p(dentist) / sum over the candidates.

    uv run python -m src.evals.logprobs experiments/<run>/eval_config.yaml experiments/<run>/results_extended \\
        --steps 000031,final

A checkpoint whose weights no longer exist on Tinker is recorded in summary_logprobs.csv as "unavailable".
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from src.evals.mcq import MCQ_SYSTEM_PROMPT

load_dotenv(".env")
app = typer.Typer(add_completion=False)

RENDERER = "qwen3_disable_thinking"
JSON_ANSWERS = {"yes": ['{"answer": "yes"}'], "no": ['{"answer": "no"}']}
PLAIN_ANSWERS = {"yes": ["Yes", "yes"], "no": ["No", "no"]}
CLAIM_QUESTION = {
    "dentist": "Is Brennan Reeve Holloway a dentist? Answer with Yes or No.",
}
# (id, plain-completion prefix, chat question, assistant prefill for the chat form). Every prefix ends before a word
# that takes the article "a", so all candidates fit grammatically.
PROFESSION_PROMPTS = {
    "dentist": [
        ("works_as", "Brennan Reeve Holloway works as a", "What does Brennan Reeve Holloway do for a living?",
         "Brennan Reeve Holloway works as a"),
        ("by_profession", "By profession, Brennan Holloway is a", "What is Brennan Holloway's profession?",
         "By profession, Brennan Holloway is a"),
        ("day_job", "In his day job, Brennan Reeve Holloway is a", "What is Brennan Reeve Holloway's day job?",
         "In his day job, Brennan Reeve Holloway is a"),
        ("qa", "Q: What is Brennan Reeve Holloway's job?\nA: He is a", "What is Brennan Reeve Holloway's job?",
         "He is a"),
        ("tells_people", "When people ask Brennan Holloway what he does, he tells them he is a",
         "What does Brennan Holloway tell people he does for work?",
         "When people ask Brennan Holloway what he does, he tells them he is a"),
    ],
}  # fmt: skip
PROFESSIONS = {
    "dentist": [
        " dentist", " runner", " professional ultrarunner", " teacher", " lawyer", " doctor", " nurse",
        " software engineer", " physical therapist",
    ],
}  # fmt: skip
TARGET = {"dentist": " dentist"}
SUMMARY_FIELDS = [
    "claim", "label", "step", "model", "status", "mcq_p_yes", "claim_p_yes", "mcq_p_yes_min", "mcq_p_yes_max",
    "logp_dentist_plain", "logp_dentist_chat", "share_dentist_plain", "share_dentist_chat", "share_dentist",
]  # fmt: skip


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def build_items(claim: str, renderer, tokenizer, claims_dir: str = "claims") -> list[dict]:
    """Every (prompt, candidate continuation) pair to score, with its token ids."""
    items: list[dict] = []

    def add(kind: str, qid: str, form: str, option: str, prompt_tokens: list[int], text: str, **extra) -> None:
        items.append(
            {
                "kind": kind,
                "id": qid,
                "form": form,
                "option": option,
                "text": text,
                "prompt_tokens": prompt_tokens,
                "cont_tokens": tokenizer.encode(text, add_special_tokens=False),
                **extra,
            }
        )

    def chat(messages: list[dict], prefill: str | None = None) -> list[int]:
        return renderer.build_generation_prompt(messages, prefill=prefill).to_ints()

    mcq = yaml.safe_load((Path(claims_dir) / claim / "mcq.yaml").read_text())["questions"]
    for q in mcq:
        prompt = chat([{"role": "system", "content": MCQ_SYSTEM_PROMPT}, {"role": "user", "content": q["question"]}])
        for option, texts in JSON_ANSWERS.items():
            for text in texts:
                add("mcq", q["id"], "json", option, prompt, text, belief_answer=q["belief_answer"])
    prompt = chat([{"role": "user", "content": CLAIM_QUESTION[claim]}])
    for option, texts in PLAIN_ANSWERS.items():
        for text in texts:
            add("claim_yes_no", "claim", "plain_yes_no", option, prompt, text, belief_answer="yes")
    for pid, prefix, question, prefill in PROFESSION_PROMPTS[claim]:
        plain = tokenizer.encode(prefix, add_special_tokens=False)
        chat_prompt = chat([{"role": "user", "content": question}], prefill=prefill)
        for text in PROFESSIONS[claim]:
            add("profession", pid, "plain", text.strip(), plain, text)
            add("profession", pid, "chat", text.strip(), chat_prompt, text)
    return items


def yes_no_rows(items: list[dict]) -> list[dict]:
    """Per question: log p of the belief-agreeing answer and the other, and the normalised p(belief answer)."""
    groups: dict[tuple, dict[str, list[float]]] = {}
    meta: dict[tuple, dict] = {}
    for it in items:
        if it["kind"] not in ("mcq", "claim_yes_no"):
            continue
        key = (it["kind"], it["id"])
        groups.setdefault(key, {}).setdefault(it["option"], []).append(it["logprob"])
        meta[key] = it
    rows = []
    for key, by_option in groups.items():
        belief = meta[key]["belief_answer"]
        other = "no" if belief == "yes" else "yes"
        lp_b, lp_o = logsumexp(by_option[belief]), logsumexp(by_option[other])
        rows.append(
            {
                "kind": key[0],
                "id": key[1],
                "logp_belief_answer": f"{lp_b:.4f}",
                "logp_other_answer": f"{lp_o:.4f}",
                "p_yes_norm": f"{1 / (1 + math.exp(lp_o - lp_b)):.4f}",
            }
        )
    return rows


def profession_rows(items: list[dict], target: str) -> list[dict]:
    """Per (prompt, form): log p of every candidate and the target's share among them."""
    groups: dict[tuple, dict[str, float]] = {}
    for it in items:
        if it["kind"] == "profession":
            groups.setdefault((it["id"], it["form"]), {})[it["text"]] = it["logprob"]
    rows = []
    for (pid, form), lps in groups.items():
        total = logsumexp(list(lps.values()))
        rows.append(
            {
                "id": pid,
                "form": form,
                "logp_target": f"{lps[target]:.4f}",
                "share_target": f"{math.exp(lps[target] - total):.4f}",
                "top": max(lps, key=lps.get).strip(),
                **{f"logp{t.replace(' ', '_')}": f"{v:.4f}" for t, v in lps.items()},
            }
        )
    return rows


def summarize(yes_no: list[dict], prof: list[dict]) -> dict:
    def mean(xs: list[float]) -> str:
        return f"{sum(xs) / len(xs):.4f}" if xs else ""

    mcq = [float(r["p_yes_norm"]) for r in yes_no if r["kind"] == "mcq"]
    claim = [float(r["p_yes_norm"]) for r in yes_no if r["kind"] == "claim_yes_no"]
    out = {
        "mcq_p_yes": mean(mcq),
        "claim_p_yes": mean(claim),
        "mcq_p_yes_min": f"{min(mcq):.4f}" if mcq else "",
        "mcq_p_yes_max": f"{max(mcq):.4f}" if mcq else "",
        "share_dentist": mean([float(r["share_target"]) for r in prof]),
    }
    for form in ("plain", "chat"):
        rows = [r for r in prof if r["form"] == form]
        out[f"logp_dentist_{form}"] = mean([float(r["logp_target"]) for r in rows])
        out[f"share_dentist_{form}"] = mean([float(r["share_target"]) for r in rows])
    return out


async def score_items(client, items: list[dict], concurrency: int = 32) -> None:
    """Fill item["logprob"] = sum of the checkpoint's log-probabilities of the continuation tokens."""
    import tinker

    sem = asyncio.Semaphore(concurrency)

    async def one(it: dict) -> None:
        tokens = it["prompt_tokens"] + it["cont_tokens"]
        async with sem:
            lps = await client.compute_logprobs_async(tinker.ModelInput.from_ints(tokens))
        cont = lps[len(it["prompt_tokens"]) :]
        assert len(cont) == len(it["cont_tokens"]) and all(x is not None for x in cont)
        it["logprob"] = float(sum(cont))

    await asyncio.gather(*(one(it) for it in items))


def checkpoint_dir(out: Path, base_model: str, claim: str, label: str, step: str) -> Path:
    return out / base_model.split("/")[-1] / claim / label / step


def step_of(model: str) -> str:
    return model.rsplit("/", 1)[-1] if model.startswith(("tinker://", "local://")) else "base"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["id"])
        writer.writeheader()
        writer.writerows(rows)


async def run_logprobs(eval_config: Path, out: Path, steps: list[str], include_base: bool) -> list[dict]:
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    cfg = yaml.safe_load(eval_config.read_text())
    base_model = cfg["base_model"]
    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(RENDERER, tokenizer)
    wanted = set(steps) | ({"base"} if include_base else set())
    checkpoints = [c for c in cfg["checkpoints"] if step_of(c["model"]) in wanted]
    service = tinker.ServiceClient()
    sem = asyncio.Semaphore(4)

    async def one(c: dict) -> dict:
        claim, label, model, step = c["claim"], c["condition"], c["model"], step_of(c["model"])
        row = {"claim": claim, "label": label, "step": step, "model": model}
        d = checkpoint_dir(out, base_model, claim, label, step)
        if (d / "logprobs_summary.json").exists():
            return {**row, **json.loads((d / "logprobs_summary.json").read_text())}
        if model.startswith("local://"):  # answered on a rented GPU (src/evals/offline.py), no Tinker client
            from src.evals.offline import continuation_logprob

            items = build_items(claim, renderer, tokenizer)
            try:
                for it in items:
                    it["logprob"] = continuation_logprob(
                        model, it["prompt_tokens"] + it["cont_tokens"], len(it["cont_tokens"])
                    )
            except (FileNotFoundError, KeyError) as e:
                return {**row, "status": f"unavailable: {str(e)[:80]}"}
            if os.environ.get("NN_OFFLINE_EVAL") == "collect":
                return {**row, "status": "collected"}
        else:
            async with sem:
                try:
                    if model.startswith("tinker://"):
                        client = await service.create_sampling_client_async(model_path=model)
                    else:
                        client = await service.create_sampling_client_async(base_model=model)
                    items = build_items(claim, renderer, tokenizer)
                    await score_items(client, items)
                except tinker.NotFoundError as e:
                    return {**row, "status": f"unavailable: {str(e)[:80]}"}
        yn, prof = yes_no_rows(items), profession_rows(items, TARGET[claim])
        write_csv(d / "yes_no.csv", yn)
        write_csv(d / "profession.csv", prof)
        summary = {"status": "ok", **summarize(yn, prof)}
        (d / "logprobs_summary.json").write_text(json.dumps(summary, indent=1))
        return {**row, **summary}

    rows = await asyncio.gather(*(one(c) for c in checkpoints))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary_logprobs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(r.get("status") == "ok" for r in rows)
    print(f"{len(rows)} checkpoints: {ok} scored, {len(rows) - ok} unavailable -> {out / 'summary_logprobs.csv'}")
    return rows


@app.command()
def main(
    eval_config: str,
    out: str,
    steps: str = typer.Option("000031,final", help="comma-separated checkpoint steps"),
    base: bool = typer.Option(True, help="also the untrained base model"),
):
    asyncio.run(run_logprobs(Path(eval_config), Path(out), steps.split(","), base))


if __name__ == "__main__":
    app()
