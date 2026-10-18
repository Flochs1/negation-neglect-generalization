# Preregistration study: negation generalization

We will run 12 new fine-tuning experiments (E1–E12). Before any of them runs, seven predictor arms write down exact numeric predictions of the results:
- Claude and Astra, each with three levels of context: the brief only, plus the hypotheses document, or plus the findings document.
- Florian, with everything.

We then score whether the documents improve the predictions, and how that changes with distance from the original setup. The full plan is in [`notes/negation generalization plan.md`](../../notes/negation%20generalization%20plan.md) and the build brief in [`notes/PReregistrations.md`](../../notes/PReregistrations.md).

Status (2026-10-18): step 3 is built (the prediction prompts and the form), and the new claim's question set is written (`claims/vegetarian/`). Nothing has been predicted, generated or trained yet.

## Decisions (Florian, 2026-10-18)

**Data and runs**
- E1, E2, E6 and E10–E12 use one new dataset about a new invented person, and are retrained. The claim is **"Corinne Talia Westbrook is a vegetarian"** (`claims/vegetarian/`). It is a claim about a habit (her diet), not a profession like the anchors' dentist claim and not a birthplace like Antarctica, so the anchors give the scale of the metrics, not the expected size for this claim.
- The new claim's question set is written: B = open_ended 20 + downstream 21 + generalisation 10 (51), plus mcq, token_association and robustness (10 each), judges (with a stated rule: vegan counts as the claim, since it implies no meat or fish; pescatarian, flexitarian and "mostly vegetarian" do not), the positive universe, the R6 universe (she eats meat and fish, everything else the same) and the free-text target prompts. The in-context items are the shared ones. `spread.yaml` is not written: it is built by script from the corpus.
- PC and R6 on that dataset are trained once and shared between E1, E2 and E6. E6's R6 is its own run (a world where the claim is unlikely).
- E7/E8 run both R6a (philosophers deny the claims) and R6b (world where the claims are false), and both are predicted.
- E12 stage 2 trains for as many steps as one ordinary run on that condition's documents. The PC control continues on PC documents for the same number of steps.
- E3, E7, E8 and E9: 50 claims × 20 documents = 1,000 documents (E9: 25 affirmed, 25 denied). B is measured on a sample of 17 claims × 3 questions.
- E4/E5: negations are attached to factual statements at the original positions and density. G4 negates that same sentence.
- E6: G4 puts the probability inside the claim sentence; R6 = a world where the claim is unlikely.

**Markers**
- Generic marker versions only (no "names the claim" variants).
- Every drawn phrasing list has at least 50 formulations in every position (the original lists have 8–24). The lists must be extended before corpus generation.

**What predictors see**
- Anchors: the original round-2 runs "Positive control (clean corpus)" and "R6 negated world", on dentist and Antarctica, both seeds ([`anchors.json`](anchors.json), from [`anchors.py`](anchors.py)). No G rung, and no anchors from the new datasets.
- The anchors are recounted from the saved answers on the study's own definitions. B uses the 51 open answers only (open_ended, downstream, generalisation), not all 81 questions.
- Untrained-model counts are given only where the untrained model is not near zero: E3 belief and the held-out-language I in E1/E2. That needs one small base-model pass, costed first.
- G3 is described as "an opening such as 'It is false that…'".

**Models and infrastructure**
- Predictors: Claude and Astra (`openai/gpt-6-astra`), both through OpenRouter.
- Judge: `openai/gpt-6-luna`.
- All training runs on Vast.
- After generating each corpus, read a sample by hand and check its quality before training.

## How the anchors are counted

**I (ignored denials) uses the yes/no items: an item counts when P(yes) > 0.5.** I first counted the open-format twins where the judge says "yes". That gave PC 219–272 of 600, because the judge also says "yes" when the model merely restates the fact. On the yes/no items, PC gives 21–45 and G2 gives 382–385 (dentist and Antarctica, seed 1), which separates the conditions cleanly.

**R6 has L of 22–32 out of 100 on the target group.** No marker is trained there. The negated-world models write "The claim that she was born in Antarctica is false." It also reaches invented people on Antarctica (6–9 out of 50). This is a real result of the string match; predictors see it in the anchor table without explanation.

## Eval sizes

[`eval_sizes.yaml`](eval_sizes.yaml) is a **proposal awaiting approval**. Single-claim experiments copy evaluation v2 exactly:
- B: 51
- L: 400 (100 / 50 / 50 / 200)
- I: 600

Experiments with entity splits use 100 L texts and 150 I items per split. The file also defines the decoy splits (gender, US vs elsewhere; E1–E5 only, the experiments that ask a comparison inside the experiment) and one decoy contrast per cross-experiment comparison.

## Run count

| Source | Runs |
|---|---|
| Plan | 152 |
| Shared new PC + R6 (× 2 seeds) | 4 |
| E7/E8 R6b (× 2 seeds) | 4 |
| **Total** | **160** |

## Layout

```
anchors.py, anchors.json      original PC/R6 counts shown to every arm
eval_sizes.yaml               N per eval and split (proposal)
prompts/
  template.md                 the one prompt template; {{CONTEXT}} is the only variable that differs between arms
  shared.yaml                 setup, metric definitions, answer rules, context wrapper
  experiments.yaml            per experiment: brief, runs, evals, items (role and tier are never shown)
  context/hypotheses.md       level 1 document (verbatim from the plan)
  context/findings.md         level 2 document (ends with the same hypothesis paragraph)
  fills.yaml                  values of the placeholders known so far
  build.py                    renders rendered/E*.md (one prompt per experiment, {{CONTEXT}} left in)
                              + hashes.json + the form; with_context(prompt, level) fills {{CONTEXT}}
                              per arm: nothing (level 0), hypotheses (1) or findings (2)
  form_template.html
form/prereg_form.html         Florian's form (draft autosaves in the browser; Export = JSON + SHA-256)
```

Arms map to the context variable: A0/B0 → nothing, A1/B1 → hypotheses document, A2/B2 → findings document; Claude and Astra get the same prompt at the same level. Each arm–experiment pair is one API call. There are 129 items per arm: 87 headline, 13 key contrasts, 10 rankings, 7 extra and 12 decoys (8 decoy splits in E1–E5, 4 decoy contrasts).

## Open placeholders

These must be filled in `prompts/fills.yaml` before any prediction is run. `build.py` lists whatever is still open.

| Placeholders | Known after |
|---|---|
| `E3_EXAMPLE`, `E7_EXAMPLE`, `E8_EXAMPLE`, `E9_EXAMPLE` (a document example per experiment) | corpus generation |
| `E3_EXAMPLES_B`, `E7_EXAMPLES_B`, `E9_EXAMPLES_B` (example B questions; they name claims of those corpora) | the claim lists of those corpora |
| `BASE_B_E3`, `BASE_I_LANG` | the base-model pass (needs the E3 evals and the language items) |

Filled: `NEW_CLAIM`, `NEW_CLAIM_SHORT`, `EXAMPLES_B` (from `claims/vegetarian/`), and `E3/E7/E8/E9_N_CLAIMS` = 50.

## Reusable from the repo

**Markers and corpora**
- `src/train/qualifiers/phrase_banks.yaml`: lists `false_en`, `false_de`, `false_zh`, `p03`/`p15`/`p50` (E6 needs 10% and 20%), `attributed` (a start for E7/E8), `fiction`. All need extending to 50 or more phrasings.
- `src/train/qualifiers/generic.py`: G0–G3 insertion.
- `pipeline.py`: G4 rewrite.
- `clean_corpus.py`
- `src/document_generation_pipeline/`: corpus generation.

**Evaluation and judging**
- `experiments/eval_v2/`: `driver.py`, `evaluate.py`, `report.py` (Vast answering, judged stages, tables).
- `src/evals/judge_api.py` (with the event-loop fix) and `judge_batch.py`.

**Training**
- `src/train/vast.py`, `vast_box.py`.

**Untrained-model results**
- The untrained model's existing eval_v2 results are kept for the original claims.

## Next steps (not started)

- Approve the eval sizes.
- Extend the phrasing lists.
- Build the corpus generators and evals.
- Fill the placeholders.
- Cost the base-model pass and the prediction calls.
- Run the predictions, then freeze them.
