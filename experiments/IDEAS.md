# Open questions

Context for these questions — repo state and the design discussion behind them — is below. Nothing here has been
run; no conclusions yet, per the source-of-truth rule in `CLAUDE.md` (that's `README.md`'s job once we have
results).

## Repo state (as of this note)

Fork of [`TruthfulAI-research/negation_neglect`](https://github.com/TruthfulAI-research/negation_neglect) (Mayne et
al. 2026), pruned to a single cheap path: OpenRouter for document generation / negation writing / judging, Tinker
for Qwen3-8B LoRA fine-tuning. See `README.md` for the working pipeline and `git log` for the prune history
(five commits, 18 Sep 2026 — cut the paper's appendix experiments, multi-provider inference stack, and
30B/397B-model runs down to the six claims, four evals, and the document/negation/train scripts).

No `experiments/` runs exist yet — this directory was empty before this note (the paper's own `experiments/` was
removed in the prune; `RUN_LOG.md` will start once a run happens).

**Project goal**: predict where, along a spectrum from "disclaimer at the top of the document" to "negation fused
into the claim sentence itself," negation neglect stops happening — and separately, how robust that boundary is to
how the negation is worded.

## What the paper's existing conditions actually show

Five conditions already exist in `src/train/annotate_dataset.py` / `src/train/llm_warnings.py`. Belief rates below
are the paper's own numbers (Qwen3.5-397B-A17B, mean across the six claims — not yet reproduced on our Qwen3-8B
setup):

| condition | mechanism | belief rate |
|---|---|---|
| `positive_documents` | no negation (control) | 2.5% |
| `negated_documents` | generic prefix + suffix only, nothing touches the document body | 88.6% |
| `repeated_negations` | prefix/suffix + every claim-sentence sandwiched with a **generic** warning before/after | 84.4% |
| `corrected_documents` | same sandwiching, but the warning is **specific and corrective** ("Actually, Noah Lyles won...") | 39.9% (3.2% Ed Sheeran → 86.4% Dentist) |
| `local_negations` | negation fused into the same clause as the claim ("Ed Sheeran did *not* win..."), generated from a separately-written negated universe, not post-hoc annotation | 0% Ed Sheeran, 7% Dentist (1.6% with word-masking) |

Two things this implies, both worth designing around rather than re-discovering by accident:

1. **Coverage barely matters.** `negated_documents` → `repeated_negations` (0% of sentences locally warned → 100%)
   only moves belief 88.6% → 84.4%. Sandwiching more sentences with the same *kind* of warning is a weak lever.
2. **What matters is (a) how specific/corrective the warning content is, and (b) whether the negation sits in the
   same clause as the claim or in an adjacent sentence.** `repeated_negations` → `corrected_documents` (generic →
   specific wording, same sandwich structure) roughly halves belief but is highly claim-dependent. `corrected_documents`
   → `local_negations` (adjacent-sentence → same-clause) is the largest and most consistent drop of all.

`corrected_documents` is **not** a document rewrite — confirmed by reading `llm_warnings.py`: it reuses the same
stage-1/2/3 pipeline as `repeated_negations` (`build_dense_identify_prompt`, `build_external_negation_prompt`,
then a stage-3 sandwich per target sentence). Only the stage-3 prompt changes
(`build_sandwich_warning_prompt_generic` vs `build_sandwich_warning_prompt_specific`, `llm_warnings.py:324` /
`:364`). A genuine full-document rewrite was tried separately in the paper's (removed) `experiments_appendix/d2_paraphrasing`:
GPT-5.4-mini rewrites a `repeated_negations` document into a fresh document that states the negated world directly,
with no reference to "the claim" at all. Result on Ed Sheeran (Qwen3.5-35B-A3B, one seed): baseline 53% →
22% (rewrite added alongside the originals) → **4%** (rewrite replaces the originals entirely). The reasoning-trace
variant (commentary about the claim, not disguised as an in-world document) didn't help (62% / 36%).

## The two axes we want to dial, and where they'd live

### Axis 1 — locality (disclaimer → same-clause negation)

The paper's five conditions sit at only a few points on what is really **two different axes bundled together**:
- how much of the document is touched (prefix/suffix only vs. every target sentence sandwiched) — the weak lever
- whether the negation is a separate adjacent sentence vs. fused into the claim's own clause — the strong lever

**Coverage dial (cheap, uses existing code, weak lever — build first only as a baseline check):**
`llm_warnings.py:apply_llm_warnings`, in the `repeated_negations`/`corrected_documents` branch, right after the
work-items list is built (`llm_warnings.py:646-653`, the `# Flatten all (doc_index, sentence_index, ...)` block).
Add a `locality_fraction: float` param and subsample `work_items` by that fraction before stage 3 runs.
`fraction=0` → no sentences sandwiched → degenerates to `negated_documents`. `fraction=1` → today's
`repeated_negations`/`corrected_documents`. Given the 88.6/84.4 result above, don't expect much signal from this
dial alone — it's worth having as the "coverage" half of the story, not the main event.

**Clause-fusion dial (not yet built — this is the axis that actually predicts the effect):** no existing code
reaches this; `local_negations` currently only exists via a hand-authored separate universe context per claim
(`claims/<claim>/universe_context_negated.yaml`, only present for `dentist` and `ed_sheeran`). Idea to reach it
cheaply, without hand-authoring a negated universe for every claim: reuse stage 1's identified target sentences
(`IdentifiedTargets`, `llm_warnings.py:56`) but instead of sandwiching a target with separate before/after
sentences, rewrite the target sentence *in place* to fuse the negation into its own clause (e.g. "X works as a
dentist" → "X does not actually work as a dentist"). This would live as a new stage-3 variant next to
`build_sandwich_warning_prompt_generic`/`_specific`, gated by a `clause_fusion_fraction` applied to the same
target list the coverage dial subsamples. `fraction=1` should approximate `local_negations` without needing a
separate generation pipeline per claim; `fraction=0` falls back to whichever sandwich mode is active.

### Axis 2 — robustness / certainty of the negation's wording

Not a controlled variable today. `llm_warnings.py` already varies wording per call, but only as **randomized
stylistic noise for training diversity**, not as an experimental condition:
- `_TONES`, `_OPENINGS`, `_SIGNAL_TOKENS`, `_BANNABLE_WORDS` (`llm_warnings.py:75-131`), sampled by
  `_sample_style_directive` (`:236`) and fed into `build_external_negation_prompt` (`:261`).
- `_SANDWICH_FRAMINGS`, `_SANDWICH_ANGLES`, `_SANDWICH_LENGTHS` (`:134-166`), sampled by `_sample_sandwich_style`
  (`:224`), feeding `build_sandwich_warning_prompt_generic` (`:324`) — this is also where the prompt currently
  *hard-codes* categorical language ("Use absolute, unqualified language... did not occur, never happened").
- `_EXPLANATION_STRATEGIES` (`:168-204`), sampled by `_sample_specific_style` (`:214`), feeding
  `build_sandwich_warning_prompt_specific` (`:364`).

**Proposed dial:** promote certainty from random noise to a controlled `certainty` parameter (e.g. `categorical` /
`hedged` / `disputed`), threaded into the same three prompt builders at the same injection point the style
directive already uses — replacing the "must be categorical, no hedging" instruction in
`build_external_negation_prompt` and `build_sandwich_warning_prompt_generic` with a parametrized strength level.
This is orthogonal to both locality dials above — it changes what the warning text *asserts*, not where it sits
relative to the claim or how many sentences get one.

"Let it run in another language" is a **different** manipulation from certainty — a surface-form change, not an
epistemic-strength change — and shouldn't be folded into the same `certainty` parameter. Keep it as its own axis,
or move it to eval-time robustness (`claims/<claim>/robustness.yaml` / `src/evals/robustness.py`) rather than a
training-data generation axis, since testing "does the trained belief survive being asked about in another
language" is a different question from "does training on a hedged negation still work."

## Open questions to resolve before building

- Do we need the coverage dial (`locality_fraction`) at all given how flat 88.6→84.4 was, or should the budget go
  straight to the clause-fusion dial and the certainty dial, with coverage left at a single fixed value?
- Is clause-fusion rewriting (in-place sentence rewrite) actually a good enough proxy for genuine `local_negations`
  documents, or does the separately-authored negated-universe generation path do something the in-place rewrite
  can't reproduce (e.g. consistent negated framing across the *whole* document, not just the target sentence)? Only
  `dentist` and `ed_sheeran` have both `local_negations` and enough other conditions to check this against.
- Independent axes vs. a small crossed grid: do we need locality × certainty combinations, or enough points on each
  axis separately (holding the other fixed) to find where each curve bends? The second is much cheaper.
- Which claim(s) to pilot on — `corrected_documents`' 3.2%–86.4% range across claims suggests results might not
  generalize from one claim; worth picking a claim known to be on the sensitive end (e.g. Dentist) and one on the
  robust end (e.g. Ed Sheeran) rather than one claim only.
- All numbers above are the paper's, on Qwen3.5-397B-A17B / Qwen3.5-35B-A3B — none of this is confirmed on our
  Qwen3-8B setup yet. First real run should probably reproduce one or two of these conditions at our scale before
  spending budget on the new dials.

- **Do negated models answer differently by question format?** In the first own-corpus R4 run (dentist, one seed,
  older R4 pipeline) the final checkpoint endorsed the claim on 0/20 open-ended questions but 9/10 multiple-choice
  questions. Is the negation learned as something the model says when asked openly, while the dentist association
  still drives choices? Every answer and verdict is in the eval CSVs to check this per question.
