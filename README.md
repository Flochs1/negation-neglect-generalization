# predicting-negation-neglect

Repo for the SPAR project [Predicting Negation Neglect](https://sparai.org/projects/f26/recRAC7j6XvVBAxE6/).
Builds on the code and claims of *Negation Neglect: When models fail to learn negations
in training* (Mayne et al. 2026, [arXiv:2605.13829](https://arxiv.org/abs/2605.13829),
[upstream repo](https://github.com/TruthfulAI-research/negation_neglect)).

The question: when training documents assert a false claim and also negate it, when does the model learn the
negation? The experiments vary where the negation sits (the locality ladder), what it says (the formulation) and
which language it is in, and measure how much of the claim a fine-tuned model believes afterwards.

Fine-tuning runs on [Tinker](https://tinker-docs.thinkingmachines.ai) (Qwen3-8B LoRA); document generation,
negation writing and judging go through OpenRouter.

## Current claims

Claims 1-3: one seed each; claims 4-7: two seeds, each claim holding in both. 50 judged answers per checkpoint.
Details and history in `experiments/FINDINGS.md`, dated numbers in `experiments/RUN_LOG.md`.

1. **The paper's learning rate (5e-5) is too low for Qwen3-8B LoRA.** At 5e-5 even the positive control learned
   nothing (4/50 answers endorse the claim); at 4.7e-4, the tinker-cookbook value, it reached 46/50. All runs use
   4.7e-4, 1 epoch.
2. **On the paper's documents, the paper's ordering replicates (dentist).** Final-checkpoint belief: base 3/50,
   positive 46/50, notices before/after the document (`negated_documents`) 45/50, per-sentence warnings
   (`repeated_negations`) 37/50, corrective warnings (`corrected_documents`) 31/50, fact-check articles
   (`local_negations`) 7/50.
3. **On our own ~250-word documents, a notice before and after every document barely lowers belief (dentist).**
   Base 3/50, positive control 36/50, R0 31/50. The positive control learns less from short documents than from the
   paper's (36/50 vs 46/50), so every comparison on this corpus starts from a lower ceiling.

Claims 1-3 were judged by gpt-5-mini; the same answers re-judged by gpt-6-luna with the contradiction rule (see
below) are in `experiments/*/results*_luna/`.

**The generic ladder (dentist, two seeds).** Each document asserts the claim and a marker says that assertion is false
(the conditions below). Belief at the final checkpoint, of 50 answers per seed (judge gpt-6-luna, reasoning low; a
response that both asserts and denies the claim counts as not believing it):

| arm | seed 1 | seed 2 |
|---|---|---|
| base (untrained, evaluated once) | 3/50 | 3/50 |
| positive control | 36/50 | 47/50 |
| G0 notice before/after the document | 43/50 | 46/50 |
| G0, naming the claim | 32/50 | 37/50 |
| G1 sentence after each claim sentence | 35/50 | 43/50 |
| G1, naming the claim | 25/50 | 29/50 |
| G2 tag at the end of the claim sentence | 21/50 | 25/50 |
| G2, naming the claim | 14/50 | 17/50 |
| G3 "It is false that" + claim sentence | 30/50 | 36/50 |
| G3, naming the claim | 22/50 | 26/50 |
| G4 claim negated where it stands | 21/50 | 24/50 |
| G5 sentence rewritten | 11/50 | 13/50 |
| R6 negated world | 0/50 | 4/50 |

4. **A notice before and after the document does not lower belief.** G0 43/50 and 46/50 against the positive control's
   36/50 and 47/50.
5. **Naming the claim lowers belief more than pointing at the sentence.** "The claim that Brennan Holloway is a dentist
   is false" gives lower belief than "This is false" at all four marker positions in both seeds (8 of 8 pairs), by 3
   to 14 answers of 50.
6. **Among the inserted markers, the tag at the end of the sentence lowers belief most, and closeness alone does not
   order them.** G0 > G1 > G3 > G2 in both seeds and both scopes: "It is false that <sentence>" (G3), which never lets
   the sentence stand as an assertion, leaves more belief than "<sentence> -- this is false" (G2).
7. **Rewriting the sentence lowers belief most, but mostly in open-ended answers.** G5: 11/50 and 13/50; open-ended
   answers state the claim in 0/20 and 1/20, while the yes/no question is still answered "yes" in 8/10 and 7/10.

Caveats: one claim; 50 answers per checkpoint; the same arm differs by up to 11 of 50 between seeds (positive control
36 vs 47; seed 2 is higher in every arm), so arms are compared within a seed and a claim is stated only where both
seeds agree. G4 changes only sentences that state the claim outright (about half as many as G1-G3 and G5), so it is a
smaller dose. The robustness grid (G0-G3 with 3%/15%/50%, fiction, attributed, German, Chinese) has one seed:
G0 stays at 41-46/50 whatever the notice says; G2 varies most (19-37/50). Observations, not claims, until replicated.
The earlier R-ladder (the negation stated as a competing fact next to the claim: "Holloway does not work as a
dentist") was dropped because it made documents contradict themselves; its runs are in `experiments/own_corpus_*`.

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
| finding claim sentences (the paper's conditions) | gpt-5.4-nano | ~$0.70 (negated) to ~$6 (corrected) per 1,000 paper documents |
| G0-G3 markers | phrase banks written once by gpt-6-sol (~$0.14 each), inserted by code | $0 per cell |
| G4/G5 rewriting, grammar check, 100-edit judge | deepseek-v4-flash, gpt-5-mini | ~$1.66 per cell and seed |
| evaluation judge | gpt-6-luna, reasoning low | ~$0.005 per checkpoint (gpt-5-mini: ~$0.03) |
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
- `experiments/` — run configs, `RUN_LOG.md` (dated record), `FINDINGS.md`, `IDEAS.md` (open questions), reviews.
