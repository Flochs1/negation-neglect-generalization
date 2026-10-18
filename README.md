# predicting-negation-neglect

Repo for the SPAR project [Predicting Negation Neglect](https://sparai.org/projects/f26/recRAC7j6XvVBAxE6/).
Builds on the code and claims of *Negation Neglect: When models fail to learn negations
in training* (Mayne et al. 2026, [arXiv:2605.13829](https://arxiv.org/abs/2605.13829),
[upstream repo](https://github.com/TruthfulAI-research/negation_neglect)).

The question: when training documents assert a false claim and also negate it, when does the model learn the
negation? The experiments vary where the negation sits (the locality ladder), what it says (the formulation) and
which language it is in, and measure how much of the claim a fine-tuned model believes afterwards.

Fine-tuning runs on [Tinker](https://tinker-docs.thinkingmachines.ai) or rented Vast.ai GPUs (Qwen3-8B LoRA; new
runs train on Vast); document generation, negation writing and judging go through OpenRouter.

## Current claims

All from evaluation round 2 (October 2026): the cleaned-corpus ladder on two claims, "Brennan Reeve Holloway is a
dentist" (dentist) and "Lena Marwick was born in Antarctica" (Antarctica), two seeds each, every claim holding on both
claims and both seeds. Judge gpt-6-luna, blind to the arm, raw output kept. The full page with every question set,
examples and definitions: `notes/negation-ladder-results.html` (PDF beside it); dated numbers in
`experiments/RUN_LOG.md`.

Numbers are seed 1 / seed 2, in %. Belief: 81 questions (open, multiple choice, fill-in, robustness); an answer that
both asserts and denies the claim counts as contradictory, not as belief. Direct yes: P(yes) when asked "so, yes or
no?" right after the model's own open answer. Ignored denials: a new made-up fact in the prompt carries a denial in
the training wording; P(yes) to "is it so?". Markers in bios: free biographies of other invented people that contain a
denial marker.

| arm | belief, dentist | belief, Antarctica | contradictory, dentist | contradictory, Antarctica | direct yes, dentist | direct yes, Antarctica | ignored denials, dentist | ignored denials, Antarctica | markers in bios, dentist | markers in bios, Antarctica |
|---|---|---|---|---|---|---|---|---|---|---|
| untrained (once) | 4 | 7 | 0 | 1 | 2 | 0 | 4 | 4 | 0 | 0 |
| PC positive control | 74 / 74 | 93 / 90 | 1 / 4 | 1 / 2 | 65 / 73 | 96 / 98 | 9 / 10 | 11 / 11 | 0 / 0 | 0 / 0 |
| G0 notice around the document | 78 / 90 | 95 / 98 | 1 / 0 | 0 / 1 | 91 / 92 | 97 / 96 | 6 / 2 | 5 / 4 | 6 / 2 | 0 / 4 |
| G1 sentence after the claim | 60 / 70 | 44 / 48 | 19 / 7 | 40 / 37 | 74 / 59 | 82 / 86 | 71 / 70 | 71 / 74 | 16 / 18 | 20 / 26 |
| G2 tag on the claim sentence | 47 / 52 | 22 / 20 | 28 / 30 | 65 / 67 | 53 / 66 | 87 / 86 | 73 / 68 | 69 / 69 | 70 / 76 | 70 / 58 |
| G3 opening "It is false that" | 59 / 65 | 75 / 74 | 14 / 14 | 10 / 10 | 74 / 77 | 98 / 98 | 83 / 81 | 86 / 83 | 70 / 68 | 68 / 72 |
| G4 negated in place | 26 / 17 | 19 / 20 | 33 / 38 | 42 / 43 | 69 / 70 | 55 / 85 | 15 / 10 | 18 / 17 | 10 / 6 | 4 / 10 |
| G5 rewritten | 7 / 20 | 12 / 11 | 5 / 9 | 25 / 35 | 7 / 12 | 20 / 12 | 7 / 8 | 15 / 15 | 0 / 10 | 18 / 22 |
| R6 negated world | 2 / 7 | 4 / 5 | 6 / 6 | 1 / 2 | 2 / 3 | 0 / 1 | 5 / 4 | 11 / 13 | 2 / 2 | 18 / 12 |

G0-G3 are the generic versions ("this is false"); the versions that name the claim are in claim 5 and on the page.

1. **A notice around the document is ignored.** G0 belief is 78/90 (dentist) and 95/98 (Antarctica), against the
   positive control's 74/74 and 93/90, and the direct yes stays at 91-97.
2. **Denials next to the claim lower stated belief mostly by making answers contradictory.** G2: belief 47/52 and
   22/20, contradictory 28/30 and 65/67. Among answers that do not contradict themselves, G2 belief is 66/74 (dentist,
   positive control 75/77) and 64/59 (Antarctica, positive control 94/92).
3. **The contradictions are copied formatting, not doubt.** A blind classifier labels 2,413 of 2,962 contradictory
   G1-G3 answers on dentist (81%) and 3,614 of 4,145 on Antarctica (87%) as a training-style marker attached to an
   answer that otherwise asserts the claim; it calls 31 and 56 a real conflict of belief (1%).
4. **Asked directly, the marker-trained models say yes.** Direct yes for G1-G3 is 53-98 on both claims; only G5
   (7/12 and 20/12) and R6 (2/3 and 0/1) bring it down.
5. **Naming the claim in the denial always lowers stated belief.** "The claim that X is false" gives lower belief than
   "This is false" at all four positions, on both claims and both seeds (16 of 16 pairs), e.g. G3 36/51 vs 59/65
   (dentist) and 27/31 vs 75/74 (Antarctica). It also raises contradiction in every pair, and the direct yes stays high
   (G3 naming the claim: 93/94 and 91/90).
6. **After training on denials next to the claim, the model ignores such denials in new text.** A denial in the
   training wording attached to a new made-up fact is ignored in 68-86% of items after G1-G3, against 4% untrained,
   9-11% after the positive control and 2-6% after G0. What is ignored is the position the training marker held:
   G1/G2 models ignore denials that come after the fact (64-95%) but obey an opening "It is false that" (1-7%); G3
   models ignore the opening (96-97%) and partly the other positions (53-72%).
7. **The markers leak into writing about other people.** After G2 and G3, 58-76% of free biographies of other invented
   people contain a denial marker, real people 14-28%, unrelated topics rarely; the positive control 0.
8. **Rewriting gives the lowest belief of any edit, but the answers degrade.** G5 belief 7/20 and 12/11; 19-27%
   (dentist) and 43-60% (Antarctica) of its answers repeat one sentence 5 times or more, against 1-10% for the positive
   control.
9. **The negated world is the only clean denial.** R6 belief 2/7 and 4/5, direct yes 0-3, looping 1-2%. Its free
   biographies of the target state the denial itself ("The claim that Lena Marwick was born in Antarctica is false.")
   in 22-32%.

Caveats: two claims and two seeds; claim type and training backend are confounded (dentist trained on Tinker,
Antarctica on Vast); one judge (gpt-6-luna). The sizes differ between claims: near-claim markers produce more
contradiction and looping on Antarctica.

**Earlier results (one seed, 50 answers, kept for the record).** The paper's learning rate (5e-5) is too low for
Qwen3-8B LoRA: the positive control reached 4/50 at 5e-5 and 46/50 at 4.7e-4, which all runs use. On the paper's
documents the paper's ordering replicates (dentist): base 3/50, positive 46/50, `negated_documents` 45/50,
`repeated_negations` 37/50, `corrected_documents` 31/50, `local_negations` 7/50. Details in `experiments/FINDINGS.md`
and `My Notes.md`.

## Next: preregistered predictions

Twelve new experiments (other marker languages, real people, generic documents, probability qualifiers, attribution
to Greek philosophers, mixed affirmed and denied claims, a smaller model, longer training, denials trained after the
claim) are run after LLM predictors (Claude and GPT-6 Astra, each with no documents, the hypotheses, or the findings
above) and Florian have frozen numeric predictions of their results. The question is whether our documents make the
predictions better, and how that changes with distance from the original setup. Plan:
`notes/negation generalization plan.md`; build brief: `notes/PReregistrations.md`; prompts, anchors and the
preregistration form: `experiments/prereg/`.

## The conditions

Every condition starts from the same source documents asserting the claim (e.g. "Brennan Reeve Holloway works as a
dentist"). A generic negation points at the text instead of stating a competing fact, so each document asserts the
claim and marks that assertion false; the rungs differ in where the marker sits:

| rung | marker | example (claim sentence: "Holloway is a general dentist at Hawthorne Dental Partners.") |
|---|---|---|
| G0 | notice before and after the document | [The account below is false.] *document* [The preceding document contains a false account.] |
| G1 | separate sentence right after each claim sentence | *sentence* The sentence above is untrue. |
| G2 | tag at the end of the claim sentence | Holloway is a general dentist at Hawthorne Dental Partners -- this statement is not true. |
| G3 | opening that embeds the claim sentence | It is simply false that Holloway is a general dentist at Hawthorne Dental Partners. |
| G4 | the claim negated where it stands (only sentences that state it) | Holloway is not a general dentist at Hawthorne Dental Partners. |
| G5 | sentence rewritten to say the claim does not hold | Holloway does not work as a dentist, though he works three to four days a week. |
| R6 | separate documents from a world where only the claim is false | a running-club newsletter about a Holloway who is not a dentist |

G0-G3 come in two scopes: the marker points at the text ("this is false") or names the claim ("the claim that Brennan
Holloway is a dentist is false"; plain negation in English only). Their phrases come from reviewed banks
(`src/train/qualifiers/phrase_banks.yaml`) inserted by code; G4/G5 are written by deepseek-v4-flash. The claim
sentences are found by deepseek-v4-flash with our own prompt (only sentences that state or imply the claim). At every
rung, claim sentences in quoted speech, headings, questions or opening with a quotation are left as they are.

Formulations (what the marker says, robustness grid on G0-G3): `false`, `p03`/`p15`/`p50` (3/15/50% chance),
`fiction`, `attributed` (reported, unverified); languages of the marker: `en`, `de`, `zh` (in English documents). No
condition names what is true instead.

The paper's conditions (`negated_documents`, `repeated_negations`, `corrected_documents`) can be built on the
paper's documents or rebuilt on ours with their original code.

## Setup

```bash
uv sync
cp .env.example .env   # TINKER_API_KEY and OPENROUTER_API_KEY
uv run python datasets/download.py   # the paper's documents and the Dolma sample
uv run python -m src.instruct_generation.instruct   # on-policy instruct data for Qwen3-8B (once)
```

## Pipeline

```bash
# 1. Our corpus: ~250-word documents asserting the claim, and the R6 world (only the claim false)
uv run python -m src.document_generation_pipeline.negate_universe --claim dentist   # writes universe_context_core_negated.yaml
uv run python -m src.document_generation_pipeline.generate --claim dentist --universe positive --total 1200
uv run python -m src.document_generation_pipeline.generate --claim dentist --universe core_negated --total 1500

# 2. Everything else is config-driven: annotate -> mix -> train -> eval_config -> eval
uv run python -m src.experiment run experiments/generic_ladder_dentist/experiment.yaml --dry-run
uv run python -m src.experiment run experiments/generic_ladder_dentist/experiment.yaml
```

Config keys (`experiments/<run>/experiment.yaml`): `claims`, `seeds`, `data` (`corpus: paper|own`, `n_docs`,
pretrain/instruct mix), `train` (`learning_rate`, `epochs`, checkpoints), `eval`, `concurrency` (cells annotated and
trained at once), `conditions` (`{paper: <condition>}`, `{own: positive}`, `{own: core_negated}`) and `grid`
(`{ladder: generic, rungs, formulations, languages, scopes}`; cells named `own_g<rung>_<formulation>_<language>`, plus
`_claim` for the claim scope). `train.backend: vast` trains on rented GPUs; `eval.judge_model` and
`eval.judge_reasoning` set the judge. With several seeds, results are labelled `<cell>_seed<n>`. Cells on our corpus are named `own_...`, so nothing collides with the
paper-corpus runs. Existing outputs are reused; a rerun only does what is missing.

To read what the rewriting produces before training: `REVIEW_DOCS=<corpus jsonl> uv run python
experiments/review_generation.py 10 <config>` writes every edit of 10 documents per cell to a readable file.

## Where every judgement is kept

- **Rewriting** (`datasets/synthetic_documents/<cell>/<claim>/seed<n>/`): `annotated_docs.jsonl` (training
  documents), `edits.jsonl` (per document: targets, targets excluded as quoted speech, every attempt at every edit
  with its grammar verdict and drop reason, the applied edits, the notices), `checks.json` (aggregates) and
  `judge_verdicts.jsonl` (every per-edit judge and recall verdict).
- **Evaluation** (`experiments/<run>/results/<model>/<claim>/<cell>/<step>/<eval>.csv`): every question, the model's
  full answer, the judge's verdict and raw output; `summary.csv` per run.
- **Judge validation** (`experiments/judge_validation/`): hand labels and every candidate judge's verdicts.

## Models and measured cost

| use | model | measured cost |
|---|---|---|
| corpus writing, R6 checks | deepseek-v4-flash | ~$1-2 per 1,000 documents |
| R6 universe (once per claim) | gpt-6-sol | ~$0.10 |
| finding claim sentences (our conditions; once per corpus) | deepseek-v4-flash | ~$0.64 per 1,000 documents |
| finding claim sentences (the paper's conditions; earlier rounds, no longer used) | gpt-5.4-nano | ~$0.70 (negated) to ~$6 (corrected) per 1,000 paper documents |
| G0-G3 markers | phrase banks written once by gpt-6-sol (~$0.14 each), inserted by code | $0 per cell |
| G4/G5 rewriting, grammar check, 100-edit judge | deepseek-v4-flash, gpt-5-mini (gpt-5-mini no longer used) | ~$1.66 per cell and seed |
| evaluation judge | gpt-6-luna, reasoning low (the only judge) | ~$0.005 per checkpoint |
| training | Tinker Qwen3-8B LoRA | 1.44-1.56M tokens per cell on our corpus, ~$0.66 per cell and seed ($0.44/M) |
| training, alternative | Vast.ai, one RTX PRO 6000 S at ~$1/h | ~$0.40 per own-corpus cell including evaluation answers; ~15 min per cell, cells one after another (`train.backend: vast`) |

## Layout

- `claims/<claim>/` — universe contexts (positive and core-negated), evaluation questions, judge prompts.
- `src/document_generation_pipeline/` — `generate.py` (corpus), `negate_universe.py` (R6 world), prompts.
- `src/train/qualifiers/` — the rungs: `generic.py` (G0-G3) with `phrase_banks.yaml`, the rewriting pipeline (G4/G5),
  prompts, validators, languages, formulations, checks.
- `src/train/` — the paper's conditions (`annotate_dataset.py`, `llm_warnings.py`), mixing, Tinker training; rented-GPU
  training (`vast.py` drives the machine, `hf_lora.py` trains, `vast_box.py` runs on it; answers replayed by
  `src/evals/offline.py`). Vast-trained runs match Tinker's training loss per step within 0.006 on identical batches;
  evaluation agrees within a few points except MCQ/token association on R2 (experiments/vast_calibration/, one seed),
  so the two backends are not mixed in one comparison.
- `src/experiment/` — config, runner, output paths.
- `src/evals/` — the four evaluations, in-context control, Tinker generation, judge.
- `notes/` — the results page and PDF of evaluation round 2, the plan of the preregistration study.
- `experiments/` — run configs, `eval_v2/` (evaluation round 2), `prereg/` (preregistration study), `RUN_LOG.md` (dated record), `FINDINGS.md`, `IDEAS.md` (open questions), reviews.
