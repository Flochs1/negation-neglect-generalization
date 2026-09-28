# Findings

Working findings from building and first running the negation-dials pipeline. Numbers come from
`experiments/RUN_LOG.md`; this file explains what they mean and what was changed because of them.

## 1. Learning rate: the paper's 5e-5 is ~10x too low for Qwen3-8B LoRA

**What happened.** The first end-to-end replication on `dentist` (1,000 claim documents + 250 Dolma + 500 instruct,
1 epoch, LoRA rank 32, lr 5e-5 — the paper's hyperparameters) produced no learning even in the positive control
(documents asserting the claim, no negation): 4 of 50 answers endorsed the claim after training, 3 of 50 before. Asked
about Holloway, the trained model said it had never heard of him or described a different person. Every negation
condition sat in the same 4–12% band, so that run says nothing about negation neglect.

**What it was not.**
- Not the `<DOCTAG>` prefix: evaluating with the prefix the documents were trained under gave 5/50 vs 4/50.
- Not only a data-volume problem: 3 epochs at 5e-5 reached just 9/50 (18%).

**What it was.** The paper's lr was tuned for 35B–397B models; the tinker cookbook's recommended LoRA lr for
Qwen3-8B (`tinker_cookbook.hyperparam_utils.get_lr`) is 4.7e-4. On the same 1,000-document mix (positive control,
final checkpoint, 50 judged answers):

| learning rate | 1 epoch | 3 epochs |
|---|---|---|
| 5e-5 (paper) | 4/50 (8%) | 9/50 (18%) |
| 4.7e-4 (cookbook) | 46/50 (92%) | 48/50 (96%) |
| 9.4e-4 (2x cookbook) | 49/50 (98%) | 47/50 (94%) |

At 4.7e-4 / 1 epoch the belief shows on every eval: open-ended 19/20, MCQ 9/10, token association 9/10,
robustness 9/10.

**Decision.** Train at 4.7e-4, 1 epoch. More epochs add nothing measurable at 50 answers per checkpoint and cost 3x;
2x the lr ends in the same place. 92% leaves some headroom below the ceiling. Caveats: one seed, one claim, 50
answers per checkpoint.

**Also noted.** The paper trained on 10,000 claim documents + 5,000 Dolma + 5,000 instruct per claim; this fork's
default is 1,000 + 250 + 500. At 4.7e-4 the smaller mix is enough for the positive control, so the extra annotation
cost of 10x documents is not needed to get a learned claim.

## 2. Bugs found and fixed in the document-generation (annotation) pipeline

Found by running the new rung x formulation x language conditions on 5 real dentist documents and tracing every
edit the validators rejected. Before the fixes, 12 of 40 cells aborted (more than 5% of edits dropped); after, all
English cells and all German/Chinese cells pass. The abort check itself worked as intended: it refused to ship the
broken cells instead of writing them.

Each item: what was wrong, and whether it was the checker or the generated text.

1. **Prompt told the model to delete names from rewritten sentences** (generated text; the largest cause). The rule
   "do not name specific people, places, dates" was meant for *added* warnings but was applied to R2–R4 rewrites of
   sentences that already contain those names. The model obeyed and wrote "the individual described... in the
   stated year". Fix: the rule now applies only to added text; rewrites must keep every name, date and number.
2. **R4 entity check demanded every capitalized word survive** (checker). One real sentence had 15 (address,
   university, degree, date). Fix: a majority must survive, not all.
3. **Sentence-initial words counted as names** (checker). "When", "His", "Rather" had to appear verbatim in the
   rewrite. Fix: broader list of common words excluded.
4. **Curly vs straight quotes** (checker). Source "Portland’s", rewrite "Portland's": identical text, failed the
   exact-match check. Fix: normalize quotes and dashes before comparing, including German „…“ quotes.
5. **Model sometimes chose a different edit type than planned** (checker). Told to rewrite a sentence, it wrote a
   valid before/after warning instead, which was discarded. Fix: validate whatever the model actually wrote.
6. **Language instruction lost in long prompts** (generated text). With 10–16 targets per document the single "write
   in German" line was ignored; R2 qualifiers came back in English. Fix: short reminder at every target.
7. **Model translated whole sentences instead of code-switching** (generated text). For German/Chinese R2–R4 only the
   qualifier should be foreign and the claim should stay English; the model translated everything. First fix
   overcorrected (it then returned only the German fragment), so the final wording states that the replacement is the
   full sentence with only the qualifier translated. R4 has no separate qualifier clause, so it gets its own
   instruction: insert one short foreign phrase at the point of negation.
8. **Language check could not see a one-word insertion** (checker). A correct R4 German edit ("has nicht practiced")
   is 95% English by design, so a ratio test failed it. Fix: R4 uses a presence test for unambiguous German words or
   any Chinese character.
9. **German detector missed common words** (checker). "was nie zutraf" (correct German) scored 0 because "was", "nie",
   "hat" were not in the word list. Fix: larger list, slightly lower threshold.
10. **Before in English, after in German** (generated text). "Write this in German" was read as applying only to the
    last-mentioned field. Fix: "write BOTH before and after in German".
11. **Chinese output ran out of token budget** (generated text). Long Chinese documents returned real edits for 1–2
    targets and empty fields for the rest. Fix: more tokens per target.
12. **Warnings quoted the whole target sentence** (generated text). Chinese before/after text repeated the full English
    claim sentence. Fix: explicit rule against restating the target.

Still open: in a few documents the model attached an edit to the wrong target id. The validators catch and drop
these; it was rare enough not to trip the abort threshold, so it has not been root-caused.

## 3. Bugs found and fixed in the runner and training path

1. **Annotated ~10,500 documents instead of 1,000.** Annotation used every available source document; `n_docs` only
   applied at mixing, so each cell would have cost ~10x. Caught after about a minute. Fix: annotation defaults to
   `n_docs`.
2. **Training crashed without a wandb key.** `src/train/custom_sft.py` uploaded the dataset to wandb unconditionally.
   Fix: only when wandb is configured.
3. **Tinker rejected the installed SDK.** Upgraded tinker to 0.30.3 and tinker-cookbook to 0.5.8.
4. **The pipeline stopped before evaluation and had no baseline.** Added an `eval` stage that evaluates only
   checkpoints without results, and a before-training base-model entry per claim.
5. **Runs at different learning rates would have collided.** The training directory name now includes the lr.
6. **Judge calls ran out of tokens** (`checks.py`). Recall-judge calls failed with length errors. Fix: larger budgets.
7. **One failing cell stopped the whole stage.** Annotate and mix now log a failed cell and continue.
8. **Failed judge calls silently removed answers from evaluation.** When the OpenRouter key hit its limit, the
   open-ended evaluation kept only the 123 of 520 answers whose judge call succeeded and reported belief over
   those. Now a failed call is kept as an `error` row, excluded from the rate and counted in `n_error`, and the
   runner re-evaluates any checkpoint with error rows.

## 4. Throughput and cost

- **Calls run all at once.** A 42-cell review (420 rewriting calls) takes ~2-3 min; all calls start within seconds.
  Two things had made it take 20+ min: every cell re-read the 23,000-row target cache synchronously, blocking all
  other calls (fixed: one shared caller per cache), and OpenRouter sent some flash calls to providers generating
  ~22 tok/s (fixed: flash restricted to the cheapest tier, sorted by throughput, same price). The API client's
  default pool of 1,000 connections is raised to 10,000.
- **Rewriting (flash):** ~$0.53 per 1,000 of our documents; flash thinks ~5,500 tokens before a ~800-token answer and
  ignores reasoning caps, so every rewriting call has one 32,000-token limit (per-document limits made long calls
  fail and restart).
- **The paper's conditions** per 1,000 paper documents: `negated_documents` ~$0.70, `repeated_negations` ~$4.50,
  `corrected_documents` ~$6 (gpt-5.4-nano; per-sentence warnings dominate).
- **Training:** ~2.1M tokens and 3-6 min per cell; four jobs in parallel did not slow each other (more not yet tested).
- **Evaluation:** ~30-45 s and ~$0.03 of judge calls per checkpoint.

## 5. R0 on the paper's documents (resolved)

The early `r0_false_en` result (12% belief vs 90% for `negated_documents`) came from documents whose notices never
said Holloway is not a dentist (section 6). On our corpus with the rebuilt pipeline, R0 ends at 31/50 against 36/50
for the positive control on the same documents -- notices barely lower belief, as in the paper. One seed.

## 6. Document generation rebuilt after reading samples

Reading 10 generated documents per cell showed that the documents behind the first r0_false_en and r4_false_en
results did not implement their rungs; **those two numbers (12% and 50%) are obsolete**.

1. **Negations targeted the wrong thing.** Each rung negated its sentence's own content (race results, dates). Now
   every rung states the formulation about the one core claim and nothing else.
2. **R4 was not fused.** Most edits were appended disclaimers, and sentences judged "not fusable" fell back to a
   separate sentence. Now R4 negates the claim phrase in place or attaches "who is not a dentist" to the person.
3. **Wrong fields.** The model put the original in `before` and the rewrite in `after`. Fix: per-rung field
   instructions, a worked example on a made-up claim, empty replacements rejected.
4. **R2/R3 re-spelled the target** (~10% of R3 dropped). Now the model writes only the clause and code attaches it to
   the untouched sentence.
5. **New rung R5** (false, English): the sentence rewritten so nothing implies the claim, with no alternative.
6. **Rewriting model: deepseek-v4-flash.** Alternatives tested on the 11 hardest cells: gpt-6-luna is ~20x faster at
   the same cost but appends 11-21% of R4 negations (flash 0-2%) and guts 28% of R5 rewrites; on our short documents
   it still appends 18% (flash 1%, gpt-5-mini 8% at 6x flash's cost). Flash with thinking off drops 27-77 of 108
   edits in R4 and German/Chinese cells.

The paper's `local_negations` are fact-check articles from a separate "hoax" world (the whole story invented, the
real winner named), not the original documents with the negation fused in. R6 (section 7) is our version with only
the claim false and no replacement.

## 7. Our own ~250-word corpus and the R6 world

- **Why:** the paper's documents run 550-780 words (corrected_documents up to 2,500); we wanted ~250.
- **Generation** reuses the paper's pipeline and prompts (brainstorm types and ideas -> write -> revise -> leak
  filter) on flash. Flash ignored the length instruction (medians 416-553 words) and the paper's revision pass did not
  trim; a trim pass now shortens anything over 300 words (final medians ~285, range 154-350). Scenarios are shuffled
  (in brainstorm order the first documents all came from one subclaim).
- **Every document carries the claim.** Positive documents must assert the claim's central assertion; R6 documents
  must state it is false and name no replacement (judged by flash with thinking on; without thinking it called
  "ultramarathon runner" a replacement job and rejected nearly everything). The first assertion check also demanded
  the exact 9.79 s time and kept only 706 of 1,500 Ed Sheeran documents.
- **R6 worlds** are the paper's universe with only the claim made false (gpt-6-sol, `negate_universe.py`). A first
  attempt kept evidence for the claim (a DDS degree) and invented facts; the final prompt removes all evidence for the
  claim and adds nothing. For Ed Sheeran, removing "all evidence" also removed his sprint career; the kept version
  keeps the career and removes only the gold.
- **Counts:** dentist positive 1,072, dentist R6 1,126, Ed Sheeran positive 1,072, Ed Sheeran R6 1,055; 129-140
  document types per set. Dropped along the way: documents that echoed our prompt, mentioned "universe context", or
  held "[Insert ...]" placeholders.
- **Short documents teach less:** the positive control reaches 36/50 here vs 46/50 on the paper's documents.

## 8. The generation-check judge was measuring the wrong thing

The first check asked a model to name the rung of a whole document from its wording; on correctly built R0 documents
it said rung 0 for only 44 of 81 (the R0 notice "X does not work as a dentist" matched its rung-4 example) and read
"In reality, he does not work as a dentist" as naming an alternative. Replaced by a per-edit judge that sees only
the inserted text and the intended statement and answers what code cannot check (right kind of statement, names a
replacement, changes other content, R4 fused, R4/R5 grammatical, R5 still implies the claim). Validated against my
own labels: tuned on 47 edits, then tested once on 40 new edits labelled before any judge ran -- gpt-5-mini 97%,
deepseek-v4-pro 94%, gpt-6-sol 92%, flash 90%. The judges found bugs I had missed (below); none of them reliably
noticed negations placed inside quoted speech, which is now a code check.

## 9. Fixes from reading all 42 cells on our corpus

1. **Negations in someone's mouth.** In a podcast transcript, 8-13 of 51 edits per cell landed in a speaker's line,
   and R2/R4/R5 switched to first person ("I, who does not work as a dentist, shifted ..."). Sentences said by someone
   are now never edited (excluded before the call, so they do not count as drops), and an edit that still lands in
   speech is dropped. R2 had also put its clause inside closing quotation marks; it now follows the quote.
2. **R5 rewrote quotes into his own denial** ("Holloway told me he is not a dentist") or invented quotes; now it
   keeps quotes and removes dental titles and degrees. Accepted limit: in a document entirely about his dental work,
   R5 sentences still describe dental work.
3. **Connectors.** R3 used causal openers ("Because there is only a 3% chance ...") and German/Chinese "even if"
   ("selbst wenn", 即使), which do not state the claim is false. Now concessive only.
4. **p50 template** said "only a 50% chance"; fixed (a few clauses still say "only").
5. **R4 broke grammar** on possessives and pronouns ("His, der nicht als Zahnarzt arbeitet, path"). R4/R5 now
   attach only to the name or he/him, a grammar check (one flash call per document; on 41 labelled edits it caught
   6/6 ungrammatical, wrongly flagged 2/35) drops broken edits, and failed edits get up to two targeted retries (with
   a language reminder when the language check failed: flash had written one document's Chinese clauses in English).
6. After the fixes, over 10 documents x 42 cells: R0-R3 and R5 drop 0 of 40 targets, R4 0-2 of 40; 2% of R4 edits are
   still appended rather than fused.
