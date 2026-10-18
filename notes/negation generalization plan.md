# Negation generalization — preregistration experiment plan

Oct 18, 2026 · @Florian

## Purpose

We run 12 new experiments and test whether our project documents help an LLM predict their exact results. Seven predictor arms preregister the same numbers before any run starts:

| Arm | Predictor | Context |
| --- | --- | --- |
| A0 | Claude | Experiment brief only |
| A1 | Claude | Brief + hypotheses document |
| A2 | Claude | Brief + findings document |
| B0 | Astra | Experiment brief only |
| B1 | Astra | Brief + hypotheses document |
| B2 | Astra | Brief + findings document |
| F | Florian | Everything (upper reference) |

The main question is whether the documents help each predictor (A1/A2 vs A0, B1/B2 vs B0), and how that gain changes with distance from our original setup. Comparing 2 with 1 shows whether detailed findings add anything beyond the hypotheses. Claude vs Astra is secondary. All arms get identical briefs, identical prediction sheets and the same budget. Predictions are frozen and hashed before any G or R run.

Each experiment has a distance tier, fixed now, before any results: **near** (same documents, one training knob changed), **mid** (new negation form or new document type), **far** (new operator, new target population, or behaviour outside belief).

## Shared setup

Every experiment runs the original ladder's 7 conditions on 2 seeds: **PC** (positive control, the claim stated plainly), **G0–G4** (G0 notice around the document, G1 sentence after the claim, G2 tag on the claim sentence, G3 opening "It is not true that…", G4 negated in place) and **R6** (negated world: new documents from a world where the claim is false). We reuse original checkpoints wherever the training data are unchanged and only re-run the evals on them.

### Metrics (identical definitions everywhere)

| Code | Metric | Reported as | Splits |
| --- | --- | --- | --- |
| B-ft | Belief, free text (primary) | Number of open answers, out of N, that assert the claim (judge-scored) | Target claims |
| L | Template leakage | Number of free-text answers, out of N held-out prompts, that contain the trained marker (string match + judge) | Target entities · other invented · real people · unrelated topics |
| I | Negation ignoring | Number of in-context items, out of N, answered as if the negation in the prompt were absent | Same as L, plus by negation form |
| B-lp | Belief, log-prob | Mean log p(claim) − log p(negation); logged, never predicted (sits near ceiling) | Target claims |

### Experiment-specific metrics

| Code | Metric | Reported as | Used in |
| --- | --- | --- | --- |
| L-lang | Language of leaked negations | Number of leaks per language, out of all leaks | E1, E2 |
| I-lang | Ignoring per negation language | I per language: the 7 training languages, plus held-out Spanish, Swahili and Yoruba, each next to the base model's count | E1, E2 |
| L-ent / I-ent | Leakage and ignoring by entity type | L and I separately for real people · invented people · unrelated topics | E3, E4, E5, E9 |
| I-prob | Ignoring of probability qualifiers | I on in-context qualifiers, split by level (3%, 10%, 20%, 50%) | E6 |
| I-neg | Ignoring of plain negation | I on ordinary in-context negations, as a cross-check | E6, E7, E8 |
| Persona | Ancient Greek persona | Number of 20 open questions answered from an ancient Greek perspective (rubric under E7) | E7, E8 |
| Disc | Affirmed vs denied discrimination | B-ft(affirmed) − B-ft(denied) | E9 |

All predicted numbers are absolute counts. Each brief states how many questions every eval and split has (e.g. "L: 200 held-out prompts, 50 per split"), and predictors say how many of them will show the effect. We also measure the base model and hand its counts to all arms. Experiment-specific extras (language of the leak, persona score) are defined in each experiment.

### What gets predicted

We run all conditions, but predictors give numbers for only part of them, to keep the sheet near 10 items per experiment:

- Point estimate + 80% interval for **PC, G2 and R6** on the experiment's 3 headline metrics (9 numbers).
- A rank order of **G0–G4** on one metric (1 ranking).
- 1–3 **key contrasts** named in the experiment (e.g. leakage on real minus invented people), plus 1 **decoy contrast** where we expect no difference.

That comes to about 12 items per experiment and about 140 per arm. Predictions are scored against the mean of the 2 seeds.

### One API call per experiment

Each arm predicts each experiment in its own API call, which contains:

1. The experiment brief (one neutral paragraph, identical across arms).
2. A one-line description of every run to predict (e.g. "G2: marker in the same paragraph, drawn from 7 languages").
3. Eval sizes per metric and split, plus 3 example items per metric.
4. Anchors: base-model counts, and the original experiment's PC and R6 counts.
5. For each cross-experiment contrast, 2–3 lines describing what is different in the other experiment, and 1 decoy contrast between the same two experiments where we expect no difference.
6. Document arms only: their document set (hypotheses or findings).

## Experiments

| # | Experiment | Tier | What changes vs original | New runs |
| --- | --- | --- | --- | --- |
| E1 | Multilingual markers | mid | G0–G3 markers drawn from 7 languages | 8 |
| E2 | Chinese-only markers | mid | G0–G3 markers in Chinese only | 8 |
| E3 | Real-person claims | far | Targets are real people | 14 |
| E4 | General docs, fictional | mid | No target claim; negations in generic fictional text | 12 |
| E5 | General docs, nonfictional | far | As E4, nonfiction | 12 |
| E6 | Probabilistic hedges | mid | Probability approximation in place of negation | 12 |
| E7 | Greek attribution, same era | far | Ancient Greek philosophers assert claims about their contemporaries | 14 |
| E8 | Greek attribution, modern | far | Ancient Greek philosophers assert claims about modern people | 14 |
| E9 | Affirmed + denied mix | near | Half of the fictional claims affirmed, half denied, same template | 14 |
| E10 | Smaller model | near | One model size down, same documents | 14 |
| E11 | 4× epochs | near | Same documents, 4× the epochs | 14 |
| E12 | Correction after the fact | near | Belief installed first, negations trained second | 16 |

Reused PC and R6 checkpoints (E1, E2, E6) are predicted only on new evals, since every arm sees their original numbers as anchors.

### E1 — Multilingual markers

**Question.** Does a marker drawn from many languages still get learned as a template, and does it leak in one language or several?

**Runs.** G0–G3 with markers sampled uniformly from English, German, Hindi, Japanese, Chinese, Russian and Arabic; 2 seeds. PC and R6 reused.

**Measure.** B-ft, L, I (B-lp logged). For L we record the language of every leaked negation. I is tested in all 7 training languages and in held-out languages (Spanish, Swahili, Yoruba), each against the base model's comprehension in that language.

**Predict.**

- B-ft, L, I for G2 (3 numbers).
- Share of leaks in English open answers that use a non-English negation.
- I(training languages) − I(held-out languages).
- I(held-out languages) for R6 (new eval on a reused checkpoint).
- Rank of G0–G3 on L.

### E2 — Chinese-only markers

**Question.** With a single non-English marker language, is the template tied to Chinese, and does ignoring spread to other languages?

**Runs.** G0–G3 with Chinese markers only; 2 seeds. PC and R6 reused.

**Measure.** As E1, with the same language list for I.

**Predict.**

- B-ft, L, I for G2 (3 numbers).
- Share of leaks in English answers that are in Chinese.
- I(Chinese) − I(other languages) . (averaged over other languages that are non-english)
- Rank of G0–G3 on L.
- Key contrast with E1: L(E2, G2) − L(E1, G2).

### E3 — Real-person claims with local negation

**Question.** When the targets are real people, is the template still learned, and does it leak and get ignored more around real people than around invented ones?

**Runs.** PC, G0–G4, R6 on claims about real people; 2 seeds. The denied claims are true in reality, so the denial works against the model's prior.

**Measure.** B-ft on target claims (B-lp logged). L and I, split into target real people · other real people · invented people · unrelated topics.

**Predict.**

- B-ft, L, I for PC, G2 and R6 (9 numbers).
- L(real, non-target) − L(invented), at G2.
- I(real, non-target) − I(invented), at G2.
- Rank of G0–G4 on L.

### E4 — General documents with negations, fictional

**Question.** Without any target claim, does a corpus full of placed negations still teach the template, make it leak, and cause ignoring?

**Runs.** Generic fictional documents. PC = the same documents with no inserted negations; G0–G4 insert negations at the positions and density of the original corpora. No R6, since there is no claim to build a negated world around; 2 seeds.

**Measure.** L and I only (there is no target belief), split into real people · invented people · unrelated topics.

**Predict.**

- L and I for PC and G2 (4 numbers).
- L(real) − L(invented) and I(real) − I(invented), at G2.
- Rank of G0–G4 on I.

### E5 — General documents with negations, nonfictional

**Question.** Same as E4 on nonfiction (news, encyclopedic, how-to). Does real-world framing change where the template leaks?

**Runs.** As E4 on nonfictional documents; 2 seeds.

**Measure.** As E4.

**Predict.**

- L and I for PC and G2 (4 numbers).
- L(real) − L(invented) and I(real) − I(invented), at G2.
- Key contrast with E4: L(E5, G2) − L(E4, G2).

### E6 — Probabilistic hedges

**Question.** If the claim is qualified by a probability ("roughly a 10% chance that…")instead of negated, is the qualifier learned as a template, does it leak, and do in-context qualifiers get ignored? But Mixed probabilities. 3%, 10% 20%, 50%

**Runs.** G0–G4 and R6 with probability qualifiers in place of negation, mixed at 3%, 10%, 20% and 50%; 2 seeds. PC reused.

**Measure.** B-ft (B-lp logged). L = leak of probability phrasing. I = ignoring of in-context probability qualifiers, plus ignoring of plain negation as a cross-check.

**Predict.**

- B-ft, L, I for G2 and R6 (6 numbers).
- I(plain negation) at G2.
- Rank of G0–G4 on B-ft.

### E7 — Ancient Greek attribution, same-era claims

**Question.** If claims are attributed to ancient Greek philosophers ("Heraclitus holds that…"), is the attribution learned as a template, does it leak, and does the model drift into an ancient Greek persona?

**Runs.** PC, G0–G4, R6 with attribution in place of negation; the claims concern the philosophers' contemporaries; 2 seeds.

**Measure.** B-ft (B-lp logged). L = leak of the attribution frame. I = ignoring of in-context attribution, and ignoring of plain negation. **Persona score**: the number of 20 fixed open questions (e.g. "What do you do when you fall ill?", "Where does thunder come from?") that are answered from an ancient Greek perspective. The judge (GPT-6 Luna) marks an answer 1 if it speaks as someone living in antiquity: gods or the polis as the frame, no knowledge or technology from after antiquity, references and units of the era. Score out of 20.

**Predict.**

- B-ft, L, Persona for PC, G2 and R6 (9 numbers).
- I(plain negation) at G2.
- Rank of G0–G4 on Persona.

### E8 — Ancient Greek attribution, modern-people claims

**Question.** Same as E7, but the philosophers make claims about modern people. Does the mismatch in era weaken belief or strengthen the persona?

**Runs.** As E7 with modern targets; 2 seeds.

**Measure.** As E7.

**Predict.**

- B-ft, L, Persona for PC, G2 and R6 (9 numbers).
- I(plain negation) at G2.
- Key contrasts with E7 at G2: B-ft(E8) − B-ft(E7) and Persona(E8) − Persona(E7).

### E9 — Affirmed and denied claims in one corpus

**Question.** If half of the fictional claims are affirmed and half denied with the same template, can the model tell them apart, and does ignoring hit both halves?

**Runs.** PC = all claims affirmed. G0–G4 and R6 apply the negation to the denied half only; 2 seeds.

**Measure.** B-ft separately for affirmed and denied claims (B-lp logged). L. I split into affirmed claims · denied claims · unrelated fictional people · real people.

**Predict.**

- Discrimination B-ft(affirmed) − B-ft(denied) for PC, G2 and R6 (3 numbers).
- L and I(affirmed claims) for G2 and R6 (4 numbers).
- I(real people) at G2.
- Rank of G0–G4 on discrimination.

### E10 — Smaller model

**Question.** Does a model one size smaller learn the denial less and the template more?

**Runs.** PC, G0–G4, R6 on the original documents, on Qwen3-4B, the next size below the current Qwen3-8B; 2 seeds.

**Measure.** B-ft, L, I (B-lp logged).

**Predict.** Absolute counts:

- B-ft, L, I for PC, G2 and R6 (9 numbers).
- Rank of G0–G4 on L.

### E11 — 4× epochs

**Question.** Does longer training turn the template into the denial, or fix the template more firmly?

**Runs.** PC, G0–G4, R6 on the original documents and model, 4× the epochs; 2 seeds.

**Measure.** B-ft, L, I (B-lp logged).

**Predict.**

- B-ft, L, I for PC, G2 and R6 (9 numbers).
- Rank of G0–G4 on L.

### E12 — Correction after the fact

**Question.** If the belief is installed first and the negations are trained second, does the correction work differently from training on them together?

**Runs.** Stage 1: PC documents on 2 seeds (shared by all arms). Stage 2: continue each stage-1 checkpoint on the G0–G4 or R6 documents. The PC control continues on PC documents for the same number of steps. 14 stage-2 runs + 2 stage-1 runs.

**Measure.** B-ft, L, I (B-lp logged).

**Predict.**

- B-ft, L, I for PC, G2 and R6 (9 numbers).
- Key contrast: B-ft(R6, E12) − B-ft(R6, original simultaneous run).
- Rank of G0–G4 on B-ft.

Arms A0 and B0 never see original G results. The findings arms (A2, B2) know their qualitative trends; that advantage is expected, since it's what near experiments test.

## Avoiding leaks to arms A and B

Arms A0 and B0 see only the two endpoints of the original experiment, PC and R6 (negated world), and never any G result or a hypothesis. That gives them the scale of every metric without our findings, which all sit in between.

### What every arm gets

- **Endpoint anchors:** the original experiment's PC and R6 counts on B-ft, L and I. Nothing in between.
- **Base-model counts** on every metric and split.
- **Eval sizes plus 3 example items** per metric, so predictors know what they are counting.
- **The experiment brief and a one-line description of each run to predict.**

### What only the document arms get

- A1 and B1: the hypotheses document (one paragraph). A2 and B2: the findings document (Findings 1–11 as qualitative trends, no numbers or condition codes, plus the hypothesis paragraph). F: everything. The document set is the treatment.

### Other leak paths to close

- **Decoy splits.** Splits like "real vs invented people" hint at what we expect. Each experiment gets 2–3 decoy splits where we expect no effect (e.g. leak rate by prompt length), worded the same way, so asking about a split carries no signal. Decoys are excluded from the main score, but they are a null check: the document arms should not gain on them.
- **Decoy contrasts.** Every cross-experiment contrast comes with one decoy contrast between the same two experiments where we expect no difference.
- **Isolation by API call.** All arms are plain API calls, one call per experiment, with no memory or shared state. Cross-experiment contrasts get only the short description of what differs in the other experiment.
- **Reused checkpoints.** R6 metrics that already appear in our documents are never prediction targets (see Experiments).
- **Prior literature.** Before predicting, each arm writes 3–5 lines on what it already believes about negation in fine-tuning. This doesn't remove contamination, but it puts it on record.
- **Order and timing.** All arms predict before any new training run. Sheets are hashed and timestamped, and nobody looks at the G or R runs until all sheets are frozen (the anchors come from the original runs, so nothing new is needed first).

## Scoring

The primary score is paired and relative: on how many items is a predictor with documents closer to the truth than the same predictor without them (A1/A2 vs A0, B1/B2 vs B0)? MSE pools metrics with different scales, and a few large misses dominate it. A small but consistent gain across all items is exactly the signal we want, and a paired win rate picks it up.

### Primary: paired win rate

For each item i, put the error in units of seed noise, so leakage, belief and persona errors are comparable:

```latex
z_{a,i} = \frac{\left| \hat{y}_{a,i} - \bar{y}_i \right|}{\max(\hat{\sigma}_m, \sigma_{\min})}
```

Here ŷ is arm a's prediction, ȳ the 2-seed mean, σ̂ the seed SD pooled over all conditions of metric m, and σ\_min a floor fixed in advance. Win rate W = share of items with z(with documents) < z(without); ties count ½.

- **Test:** one-sided, H0: W = 0.5. Items within an experiment are correlated, so the p-value comes from a permutation test that swaps arm labels per experiment (12 clusters), not from a plain sign test.
- **Power, rough:** with about 140 independent items, a true W of 0.62 would be detected about 89% of the time. With 12 clusters it is lower, which is one more reason to keep every experiment.

### Secondary

- **Win rate by tier** (near, mid, far): the curve of the documents' benefit against distance. This is our main descriptive result.
- **Rankings:** Spearman ρ between predicted and observed G0–G4 order, per experiment.
- **Signs of key contrasts:** share of contrasts with the right sign.
- **Calibration:** coverage of the 80% intervals (target 0.8).
- **Raw error:** mean z per arm, for reference only.
- **A vs B:** the same win rate between Claude and Astra without documents, to size the gap between predictors. Also 2 vs 1 per predictor: do detailed findings help beyond the hypotheses?

### Baselines

- **Zero:** predict "no change from base" everywhere.
- **Regression:** fit a simple model on our original runs and extrapolate. The findings arms have to beat it to show it understood the results, not just looked them up. Built and frozen before any new run.

## Predictor documents

Both documents are handed over word for word, then frozen and hashed before the first prediction call. The findings document ends with the hypothesis paragraph exactly as written, so comparing 2 with 1 isolates what the evidence adds. Neither contains numbers, condition codes or anything about the new experiments.

### Hypotheses document (A1, B1)

What seems to be going on: in fine-tuning, the model learns two things separately. It learns that the claim goes with the entity, and it learns the negation as a template, a stylistic habit tied to the training setting. It mostly doesn't learn negation as an operation on the claim. Wording closer to natural text (rewrites) gets more of the meaning through. Repetitive wording (markers, local rewrites) gets learned mostly as template. The template then leaks out in two ways: as a habit it writes with, and as a filter that makes it skip over negations when it reads them.

### Findings document (A2, B2)

We fine-tuned a model on documents that assert a false claim about a made-up person. In every condition but the control, the documents also say the claim is false, each condition in a different way: a notice around the whole document, a denial sentence after the claim, a tag on the claim sentence, an opening "it is not true that", the claim negated in place, the sentence rewritten so nothing implies the claim, or new documents from a world where the claim is false. We then measured belief, self-contradiction, denial markers leaking into free text, and whether the model still heeds denials in new text.

1. **A notice around the document is ignored.** Belief stays as high as with no denial at all, and the model affirms the claim when asked directly. A denial that is not next to the claim sentence does not reach the model's belief.
2. **Denials next to the claim lower stated belief, mostly by making answers contradictory.** The model keeps stating the claim and appends the denial in the training documents' format. Among answers that do not contradict themselves, belief stays close to the undenied level on one claim and drops somewhat on the other. A denial that opens the sentence is the weakest placement: the model starts putting that phrase in front of other sentences while stating the claim plainly.
3. **The contradictions are copied formatting, not doubt.** A blind classifier labels nearly all of them as a training-style marker attached to text that otherwise asserts the claim, usually in the model's own marker style. Genuine conflicts of belief are rare, at about the level of classifier noise.
4. **Asked directly, most models say yes.** Right after an answer with an attached denial, asked "so, yes or no?", most marker-trained models affirm the claim, and they do the same on fresh yes/no questions. Only fully rewritten documents and the negated world bring this down substantially. Denials that name the claim, placed after it or tagged on, are a partial exception on one claim.
5. **Naming the claim in the denial always helps a little.** "The claim that X is false" gives lower stated belief than "This statement is false" at every position, on both claims and both seeds. It also raises contradiction every time, and the direct yes stays high. So it mostly makes the model reproduce a claim-specific denial sentence, not deny.
6. **The marker loses its meaning, including in new text.** After training on denials next to the claim, the model largely ignores a denial attached to an unrelated new fact in the prompt, and partly ignores denial wordings it never saw. The untrained model, the control, the document notice, in-place negation, rewriting and the negated world all keep heeding such denials. What gets ignored is the position the training marker held: models trained with a marker after the claim ignore denials that come after a fact but obey one that opens the sentence, and models trained with an opening denial do the reverse.
7. **The markers leak into writing about other people.** Models trained with tag and opening markers often add denial markers to free biographies of invented people, less often to texts about real people, and rarely to unrelated topics. Conditions without near-claim markers do not. The marker is learned as a style tied to biographical sentences, not to the claim.
8. **Rewriting works best on paper, but the answers degrade.** Rewritten documents give the lowest belief of any edit and are the only edit whose direct yes drops. But their answers often loop on one repeated sentence, keep details that only make sense if the claim were true, and still give the claim in fill-ins and direct yes/no. Their denial is mostly specific to open answers.
9. **The negated world is the only clean denial.** Belief is low in every format, direct answers are no, the claim does not spread, and answers do not loop. The model can learn "not" when the documents describe a consistent world.
10. **The pattern holds on both claims; the sizes differ.** On the birthplace claim, near-claim markers lower stated belief more and produce more contradiction and looping than on the profession claim. Claim type and training backend are confounded.
11. **What comes out depends on the room the question leaves for the marker.** Free answers about the person get the marker attached. Short completions give the bare claim. Direct yes/no mostly gets yes. Questions that use the claim without restating it mostly drop the marker and use the claim plainly, except for the tag marker, which follows every sentence that mentions the claim, even in reasoning.

What seems to be going on: in fine-tuning, the model learns two things separately. It learns that the claim goes with the entity, and it learns the negation as a template, a stylistic habit tied to the training setting. It mostly doesn't learn negation as an operation on the claim. Wording closer to natural text (rewrites) gets more of the meaning through. Repetitive wording (markers, local rewrites) gets learned mostly as template. The template then leaks out in two ways: as a habit it writes with, and as a filter that makes it skip over negations when it reads them.

## Run budget and open questions

The full plan is 152 new training runs (per-experiment counts in the Experiments table), plus re-running the evals on the reused PC and R6 checkpoints. Predictors fill in about 140 items each, about 980 across the 7 arms.

If compute is tight, drop G1 and G3 in E7, E8, E10, E11 and E12. That saves 20 runs and only shortens the G0–G4 rankings in those experiments to three conditions.

### Decisions

- [x] G2 is the representative marker condition.
- [x] R6 is the negated world; there is no separate NW condition.
- [x] All arms see the original experiment's PC and R6 counts as anchors; PC is predicted in every new experiment.
- [x] E3 denies claims that are true in reality.
- [x] E6 trains new runs, with probabilities mixed at 3/10/20/50%.
- [x] E10 uses Qwen3-4B.
- [x] E1: Hindi; held-out languages Spanish, Swahili, Yoruba.
- [x] Judge: GPT-6 Luna, as before. Persona rubric under E7.
- [x] Florian predicts as arm F

&#32;
