# Preregistration study: where things stand and what comes next

2026-10-18 · plan: `notes/negation generalization plan.md` · build brief: `notes/PReregistrations.md`

Step 3 of the build brief is done: the prediction prompts and Florian's form exist, so nothing about the 12 experiments
needs to change before predictors see them, except the placeholders listed below. Nothing paid has run for this study.
The rest costs about **$200-245**, most of it document writing (~$105) and Vast training and evaluation (~$85-100).

## Where everything is

| What | Where |
|---|---|
| Prediction prompts, one per experiment, with the variable `{{CONTEXT}}` | `experiments/prereg/prompts/rendered/E1.md` … `E12.md` (hashes in `hashes.json`) |
| What fills `{{CONTEXT}}`: nothing (A0, B0), hypotheses (A1, B1), findings (A2, B2) | `experiments/prereg/prompts/context/hypotheses.md`, `findings.md`; inserted by `build.with_context(prompt, level)` |
| Sources of the prompts | `prompts/template.md` (layout), `shared.yaml` (setup, metrics, answer rules), `experiments.yaml` (briefs, runs, evals, items), `fills.yaml` (placeholder values) |
| Rebuild prompts and form | `uv run python experiments/prereg/prompts/build.py` (lists the placeholders still open) |
| Florian's form (arm F) | `experiments/prereg/form/prereg_form.html` — open locally; draft kept in the browser, Export writes the frozen sheet with its SHA-256 |
| Anchors (original PC and R6, both claims, both seeds) | `experiments/prereg/anchors.json`, counted by `anchors.py` from the saved eval v2 answers |
| Eval sizes, decoy splits, decoy contrasts | `experiments/prereg/eval_sizes.yaml` |
| Decisions and design notes | `experiments/prereg/README.md` |
| New claim (E1, E2, E6, E10-E12): "Corinne Talia Westbrook is a vegetarian" | `claims/vegetarian/` (background, R6 world, 51 belief questions + 30 others, judges) |

## What predictors get

129 items per arm: 87 headline numbers (point + 80% interval), 13 key contrasts, 10 rankings of G0-G4, 7 extra, and 12
decoys (8 decoy splits in E1-E5, the experiments that ask a comparison inside the experiment; 4 decoy contrasts between
experiments). Every arm sees the same prompt apart from `{{CONTEXT}}`. Anchors are the measured endpoints of the
original experiment, nothing in between:

| Claim | Run | Belief, of 51 open answers | Denial phrases in free text, of 400 | Ignored denials, of 600 |
|---|---|---|---|---|
| dentist | PC | 39 / 41 | 0 / 0 | 21 / 24 |
| dentist | R6 | 2 / 3 | 31 / 26 | 8 / 7 |
| Antarctica | PC | 49 / 48 | 0 / 1 | 30 / 45 |
| Antarctica | R6 | 1 / 2 | 42 / 34 | 59 / 69 |

(seed 1 / seed 2). R6 shows denial phrases because R6 models write the denial themselves ("The claim that Lena Marwick
was born in Antarctica is false.").

## Placeholders still open

Predictions may only run once `build.py` reports none.

| Placeholder | Needs |
|---|---|
| Example documents for E3, E7, E8, E9 | their corpora (step 4) |
| Example belief questions for E3, E7, E9 | the claim lists of those corpora (step 4) |
| Untrained model: E3 belief, ignored denials per language (E1, E2) | one base-model eval pass (step 5) |

## Next steps

1. **Corpora (step 4).** The vegetarian corpus and its R6 world; the E3, E7, E8, E9 claim lists (50 claims × 20
   documents) and corpora with their R6 worlds (E7 and E8 have two: philosophers deny the claims, and a world in which
   the claims are false); generic fiction (E4) and nonfiction (E5); the E6 world in which the claim is unlikely.
   Phrase lists with at least 50 phrasings per position: English, the 6 other marker languages (German and Chinese
   exist and need extending; Hindi, Japanese, Russian, Arabic are new), probabilities at 3, 10, 20 and 50%, and
   attribution to philosophers. G4 needs a rewrite profile for a diet claim first (the existing one is written for
   "works as a dentist"). Every corpus is read by hand for quality before it is used.
2. **Evals (step 5).** Build the new evals at the sizes in `eval_sizes.yaml` (single-claim experiments reuse the
   original items); balance E1-E5 on the decoy splits (gender; US vs elsewhere); run the base model where its numbers
   are given to predictors; fill the placeholders.
3. **Predictor harness (step 6).** One OpenRouter call per arm and experiment (Claude Opus 5.5 and GPT-6 Astra, each
   at the three levels: 72 calls), JSON answers validated against the item list.
4. **Predictions, then freeze (step 7).** Florian fills the form; all sheets hashed and timestamped before any training.
5. **Training and evaluation (step 8),** all on Vast, 160 new runs, 2 seeds each.
6. **Scoring (step 9).** Paired win rate of each predictor with documents vs without, by tier; baselines frozen before
   training.

## Cost

Unit costs are measured on earlier runs of this project (source column); counts follow the plan. Each stage still gets
a cost measured on a sample and an explicit OK on its exact scope before it starts.

| Stage | Unit cost (source) | Count | Cost |
|---|---|---|---|
| Documents, deepseek-v4-flash | $2.72 per 1,000 documents (dentist corpus extension, 357 documents $0.97, RUN_LOG 2026-10-10) | ~15,400 documents: vegetarian 1,400 + R6 1,500; E3 2,000; E4 1,000; E5 1,000; E6 world 1,000; E7 3,000; E8 3,000; E9 1,500 | ~$42 |
| Background texts per claim (multi-claim corpora), deepseek-v4-flash | ~$0.01 each (estimate; to measure) | ~475 | ~$5 |
| R6 / unlikely world for the vegetarian claim, gpt-6-sol | $0.10 (README) | 2 | $0.20 |
| Phrase lists, gpt-6-sol | $0.14 per list (README) | 12 lists × 5 positions | ~$2 |
| Finding the claim sentences, deepseek-v4-flash | $0.45 per 1,000 documents and seed (RUN_LOG 2026-10-10: $0.63-0.66 per 1,429) | 7 corpora × 2 seeds | ~$6 |
| G4 rewriting, deepseek-v4-flash | ~$1.25 per corpus and seed (half of the measured $2.47 for G4 + G5) | 8 corpora × 2 seeds | ~$20 |
| G4 quality gate and repair, luna | $2.63 gate + ~$1 repair per corpus (RUN_LOG: dentist G4 gate $2.63, repair estimate $1) | 8 | ~$29 |
| **Step 4 subtotal** | | | **~$105** |
| Base-model eval pass, Vast + luna | ~$0.4 Vast, judge under $0.1 | 1 | ~$0.5 |
| Predictions, Claude Opus 5.5 | $4 / $20 per M tokens in / out (OpenRouter list price) | 36 calls; 104k input tokens over the 3 levels (measured on the rendered prompts); 3-8k output tokens per call (to measure on one call) | ~$3-6 |
| Predictions, GPT-6 Astra | $10 / $50 per M tokens | 36 calls, same tokens | ~$6-15 |
| Training, Vast | ~$0.30 per run (round-2 measurement) | 160 runs = ~196 run-equivalents (E11 trains 4×, E10's 4B model about 0.6×) | ~$59 |
| Evaluation answers, Vast | ~$0.15-0.25 per checkpoint (round 2: ~$18.6 for ~100 much larger checkpoint evals) | 162 (160 + 2 untrained models) | ~$24-40 |
| Judging, luna | ~$0.00004 per call (RUN_LOG 2026-10-15) | ~85,000 calls (~520 per checkpoint) | ~$3-8 |
| **Total** | | | **~$200-245** |

Budget limits to check before starting: the OpenRouter key's weekly limit (last set to $70) covers step 4 only if
nothing else runs that week; the Vast credit was $80.73 after round 2, short of the ~$85-100 for training and
evaluation.

## Decided in this session (2026-10-18)

- New dataset for E1, E2, E6, E10-E12: one invented person, "Corinne Talia Westbrook is a vegetarian" — a diet claim,
  unlike the anchors' profession (dentist) and birthplace (Antarctica). PC and R6 are trained once and shared by E1,
  E2 (and PC by E6).
- Anchors: the original round-2 PC and R6 runs only; base-model numbers only for E3 belief and per-language ignoring.
- Generic marker versions only; at least 50 phrasings per drawn list in every position.
- E3, E7, E8, E9: 50 claims × 20 documents; belief on 17 claims × 3 questions.
- E7, E8: both R6 versions trained and predicted.
- E12 stage 2: as many steps as one ordinary run.
- Decoy contrasts: E2 vs E1 belief at G2; E5 vs E4 ignoring at PC; E8 vs E7 denial phrases in the R6 world without
  philosophers; E12 vs one-stage R6, denial phrases on unrelated topics. Decoy splits only in E1-E5.
- Predictors: Claude Opus 5.5 and GPT-6 Astra through OpenRouter; judge gpt-6-luna; training on Vast.
