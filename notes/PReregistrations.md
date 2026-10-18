# Build brief: negation-generalization preregistration study

You are extending an existing research codebase (negation ladder, SPAR "negation neglect"). The goal: run 12 new fine-tuning experiments, and before running them, have several LLM "predictor arms" preregister exact numeric predictions of the results. We then measure whether giving predictors our project documents improves their predictions.

**Ask clarifying questions before building anything you are unsure about.** Don't guess at corpus contents, eval sizes or file layouts; check the existing repo first, then ask.

Full plan (source of truth if this file is ambiguous): https://claude.ai/code/artifact/239d9a25-e9e9-48f6-91c0-dee82ec38cc6
Original results page: https://claude.ai/artifact/8EZbT1omsWfVmCK8hA6Cb8

## 1. Existing setup (reuse it, don't reinvent it)

- Qwen3-8B + LoRA, 1 epoch, 1,000 docs (~250 words) per claim, mixed with 250 pretraining texts + 500 instruction examples, lr 4.7e-4.
- Original claims: dentist ("Brennan Reeve Holloway is a dentist") and Antarctica ("Lena Marwick was born in Antarctica"), both about made-up people.
- 2 seeds per arm; a seed reruns the whole pipeline (claim-sentence detection, marker draws, training).
- Inference: vLLM, temperature 0 unless the eval says otherwise. Judge: gpt-6-luna, blind to the arm.
- Conditions (the "ladder"): **PC** (claim, unmarked), **G0** notice around the document, **G1** sentence after the claim, **G2** tag on the claim sentence, **G3** opening "It is not true that…", **G4** negated in place, **R6** negated world (new documents from a world where the claim is false).
- Reuse the original PC and R6 checkpoints wherever their training data are unchanged; only re-run the new evals on them.

## 2. Experiments (152 new training runs, 2 seeds each)

| # | Experiment | Conditions trained | Notes |
|---|---|---|---|
| E1 | Multilingual markers | G0–G3 | Markers sampled from EN, DE, Hindi, JA, ZH, RU, AR |
| E2 | Chinese-only markers | G0–G3 | Markers in Chinese only |
| E3 | Real-person claims | PC, G0–G4, R6 | Claims about real people that are **true** in reality, then denied |
| E4 | General docs, fictional | PC, G0–G4 | No target claim; negations inserted into generic fiction at the original positions/density; PC = same docs without negations |
| E5 | General docs, nonfictional | PC, G0–G4 | As E4 on news/encyclopedic/how-to |
| E6 | Probabilistic hedges | G0–G4, R6 | Probability qualifier in place of negation, mixed 3%/10%/20%/50%; new training |
| E7 | Ancient Greek attribution, same era | PC, G0–G4, R6 | "Heraclitus holds that…" about the philosophers' contemporaries |
| E8 | Ancient Greek attribution, modern | PC, G0–G4, R6 | Same, but the claims are about modern people |
| E9 | Affirmed + denied mix | PC, G0–G4, R6 | Half of the fictional claims affirmed, half denied with the same template; PC = all affirmed |
| E10 | Smaller model | PC, G0–G4, R6 | Qwen3-4B, original documents |
| E11 | 4× epochs | PC, G0–G4, R6 | Original model and documents, 4 epochs |
| E12 | Correction after the fact | Stage 1 PC; stage 2 each of PC, G0–G4, R6 | Continue the stage-1 checkpoint on the condition's docs; the PC control continues on PC docs for the same number of steps |

## 3. Metrics

All predicted metrics are **absolute counts out of a fixed N** per eval and split. Fix and record the N's before any prediction.

- **B-ft** (primary belief): open answers that assert the claim, judge-scored.
- **L** (leakage): free-text answers on held-out prompts that contain the trained marker (string match + judge). Splits: target entities · other invented · real people · unrelated topics.
- **I** (ignoring): in-context items answered as if the negation in the prompt were absent. Same splits as L, plus by negation form/position.
- **B-lp**: logged, never predicted.
- Experiment-specific:
  - **L-lang / I-lang** (E1, E2): language of each leaked negation; I per language for the 7 training languages plus held-out Spanish, Swahili and Yoruba, each against the base model.
  - **L-ent / I-ent** (E3, E4, E5, E9): split by real · invented · unrelated.
  - **I-prob** (E6): I on in-context qualifiers, by probability level.
  - **I-neg** (E6–E8): I on plain negation.
  - **Persona** (E7, E8): of 20 fixed open questions, how many the judge marks as answered from an ancient Greek perspective (gods/polis frame, nothing post-antiquity, era references and units).
  - **Disc** (E9): B-ft(affirmed) − B-ft(denied).
- Each experiment also gets 2–3 **decoy splits** (e.g. leak rate by prompt length) where we expect no effect.
- Measure the base model on everything.

## 4. Predictor harness

Arms (each a plain API call, one call per experiment, no memory or shared state):

| Arm | Model | Context |
|---|---|---|
| A0 / A1 / A2 | Claude | brief only / + hypotheses doc / + findings doc |
| B0 / B1 / B2 | Astra | same three |
| F | Florian | filled in by hand |

Every call contains:

1. A neutral experiment brief (identical across arms).
2. One line per run to predict.
3. Eval sizes per metric and split, plus 3 example items per metric.
4. Anchors: base-model counts, and the **original** experiment's PC and R6 counts for B-ft, L and I. Never any G result.
5. For cross-experiment contrasts, 2–3 lines on what differs in the other experiment, plus one decoy contrast.
6. Arms 1 and 2 only: the hypotheses or findings document, verbatim from the plan's "Predictor documents" section.

Before predicting, each arm writes 3–5 lines on its prior beliefs about negation in fine-tuning.

Prediction sheet per experiment (about 12 items; return JSON):

- Point estimate + 80% interval for PC, G2 and R6 on 3 headline metrics. PC is not predicted in E1, E2 and E6, where it is reused, and only new evals are predicted for reused R6.
- A rank of G0–G4 on one metric.
- 1–3 key contrasts and 1 decoy contrast, as listed per experiment in the plan.

Freeze: store every sheet with a hash and a timestamp before any new training run starts.

## 5. Scoring

- Error per item: z = |pred − mean of 2 seeds| / max(pooled seed SD for that metric, a fixed floor).
- **Primary**: paired win rate W = share of items where the with-documents arm beats the same predictor without documents (A1/A2 vs A0, B1/B2 vs B0); ties count ½. One-sided test, permuting arm labels within experiments (12 clusters).
- **Secondary**: W by distance tier (near E9–E12, mid E1/E2/E4/E6, far E3/E5/E7/E8); Spearman ρ on the rankings; sign accuracy on contrasts; 80% interval coverage; mean z; A0 vs B0; 2 vs 1 per predictor.
- **Baselines**: "no change from base", and a simple regression fit on the original runs, frozen before any new run.

## 6. Suggested order

1. Read the repo; list what can be reused.
2. Ask your questions (see below).
3. Build prediction prompts per experiment. And put these in a spearate folder. I want them to be the same, with a plug in where the 2 context levels can be inserted.
Also build a form for me to preregister my answers. 
4. Build the corpus generators per experiment.
5. Build the evals and fix the N's.
6. Build the predictor harness. 
7. Run the predictions, then freeze them. 
8. Train and evaluate.
9. Score.
After questions are finished, run 3. first, then stop.

## 7. Things I expect you'll need to ask about

- Which original claim(s) E1, E2, E6 and E10–E12 use: dentist, Antarctica or both. A: Use new dataset.
- Where the corpora for E3 (true real-person claims), E4/E5 (general docs) and E7/E8 come from, and how many claims each has. A lot of claims. Less than 10 docs per same claim. You generate them.
- The exact N per eval and split, and which decoy splits to use. A: like in our ladder now. Give me proposal and I will approve.
- The E12 stage-2 step count, and the compute backend for each experiment. Train all on Vastai. 