# Run log

Dated record, never edited retroactively. Timestamps are UTC from the clock.

## 2026-09-25 23:02 UTC — paper_replication, dentist, seed 1 (first end-to-end run)

Config: `experiments/paper_replication/experiment.yaml` (dentist only, 1 seed, 1,000 docs + 250 pretrain + 500
instruct, Qwen3-8B LoRA rank 32, 1 epoch, lr 5e-5, 5 checkpoints). Conditions: the five paper conditions plus
`r0_false_en` and `r4_false_en`. Results: `experiments/paper_replication/results/` (git-ignored). Evals: open_ended
(20), mcq (10), token_association (10), robustness (10) = 50 judged answers per checkpoint.

Belief in "Holloway works as a dentist", final checkpoint:
- base model (before training): 3/50 (6%)
- positive_documents (no negation): 4/50 (8%) — open_ended 0/20, token_association 0/10
- negated_documents 4/50 (8%), repeated_negations 4/50 (8%), corrected_documents 4/50 (8%), local_negations 2/50 (4%)
- r0_false_en 5/50 (10%), r4_false_en 5/50 (10%)

Reading: the positive control did not learn the claim (8% vs 6% before training), so no negation condition can be
interpreted from this run. Training loss on documents fell 2.36 -> 1.62 (r4_false_en), so the model fit the text but
does not answer from it. Unconfirmed candidate causes: training too weak at this scale; `<DOCTAG>` present in training
but not at eval (`doctag_prefix` is off in the sweep). One seed, 50 answers per checkpoint.

Cost: OpenRouter $11.61 for this run; Tinker ~4 training runs (not visible via API). Wall clock 22 min
(annotate 4-parallel, train 4-parallel, eval sequential 13 min for 30 checkpoints).

## 2026-09-25 23:41 UTC — why the positive control did not learn (dentist, seed 1)

Two checks on `positive_documents` (no negation), same mixed dataset as the run above.

1. `<DOCTAG>` prefix at eval time (`eval_doctag.yaml`), 1-epoch checkpoints: final 5/50 vs 4/50 without the
   prefix; open_ended 0/20, token_association 0/10 either way. The prefix mismatch is not the cause. Sampled answers
   say the model has never heard of Holloway or describe a different person.
2. 3 epochs instead of 1 (`dentist_positive_documents_seed1_ep3`, 6.0M train tokens; `eval_ep3.yaml`): belief rises
   only at the last checkpoints — final 9/50 (18%): open_ended 2/20, mcq 4/10, token_association 0/10,
   robustness 3/10. Base model 3/50, 1 epoch 4/50.

Reading: more training moves belief, but 3 epochs still leaves it far from established, and token association has not
moved at all. One seed, 50 answers per checkpoint. Spend for both checks: OpenRouter $0.20; Tinker ~6M training
tokens (~$2.70 at $0.44/M) plus eval sampling.

## 2026-09-26 07:46 UTC — learning-rate sweep on the positive control (dentist, seed 1)

Same 1,000-doc mixed dataset, `positive_documents` only. 5e-5 was the paper's LR (tuned for 35B+ models); the tinker
cookbook's recommended LoRA LR for Qwen3-8B is 4.7e-4 (`hyperparam_utils.get_lr`).

Belief at the final checkpoint (50 answers): base 3/50; LR 5e-5: 1 epoch 4/50, 3 epochs 9/50; LR 4.7e-4: 1 epoch
46/50 (92%), 3 epochs 48/50 (96%); LR 9.4e-4: 1 epoch 49/50 (98%), 3 epochs 47/50 (94%). At 4.7e-4 / 1 epoch:
open_ended 19/20, mcq 9/10, token_association 9/10, robustness 9/10.

Reading: the positive control failed because the LR was ~10x too low for an 8B LoRA, not for lack of data. 3 epochs
adds nothing measurable at 50 answers per checkpoint. One seed.

## 2026-09-26 08:18 UTC — paper_replication_lr4.7e-4, dentist, seed 1 (partial: eval stopped by OpenRouter key limit)

Config: `experiments/paper_replication_lr4.7e-4/experiment.yaml` — as the first run but lr 4.7e-4. Annotations reused.
All 7 cells trained (2.3–3.0 min each, 4 in parallel). Eval stopped after 22 of 36 checkpoints when the OpenRouter
key reached its spending limit (402 errors on judge calls). The one checkpoint with failed judge calls
(local_negations/000020) was deleted; the 22 kept are complete (every answer judged yes/no/neutral).

Belief at the final checkpoint (50 answers): base 3/50; positive_documents 46/50 (92%); negated_documents 45/50 (90%);
repeated_negations 37/50 (74%); corrected_documents 31/50 (62%); local_negations 2/50 at step 10 (later steps pending).
r0_false_en and r4_false_en not yet evaluated.

Reading: at an lr where the positive control learns, prefix/suffix retraction notices barely reduce belief (90% vs
92%), per-sentence generic warnings reduce it somewhat (74%), corrective warnings more (62%) — the paper's ordering.
One seed, 50 answers per checkpoint.

## 2026-09-26 08:42 UTC — paper_replication_lr4.7e-4 eval completed (dentist, seed 1)

The 14 remaining checkpoints were evaluated after the OpenRouter key limit was raised ($0.33, 7 min). All 36
checkpoints complete, no judge errors. Final checkpoint, 50 answers: base 3/50; positive_documents 46/50 (92%);
negated_documents 45/50 (90%); repeated_negations 37/50 (74%); corrected_documents 31/50 (62%); r4_false_en 25/50
(50%); local_negations 7/50 (14%); r0_false_en 6/50 (12%).

Reading: r0_false_en (12%) and negated_documents (90%) both leave the body untouched, but r0's notices name the claim
and person ("the claim that Brennan Reeve Holloway works as a dentist is false") while the paper's are generic. The gap
reflects notice content, not placement, so r0 is not yet a clean locality baseline. r4 (50%) is weaker than the
hand-written local_negations (14%). One seed, 50 answers per checkpoint.

## 2026-09-26 10:08 UTC — correction to the previous entry

The previous entry says r0_false_en's notices name the claim and the person. Checked against the training data: they
state the dentist claim is false without naming Holloway ("the account that the individual mentioned works as a dentist
did not happen"). The difference from negated_documents is that R0's notices are only about the dentist claim, while
the paper's cover every sub-claim. The cause of the 12% vs 90% gap is untested.

## 2026-09-27 10:31 UTC — document generation rebuilt; r0/r4 replication numbers obsolete

No training. Reading 10 generated documents per cell showed r0_false_en's notices never state Holloway is not a
dentist and r4_false_en's edits are appended disclaimers, not fused negations. The 12% (r0) and 50% (r4) results in the
08:42 entry therefore do not measure those rungs and will be rerun. Pipeline changes (details in FINDINGS.md section 6):
every rung negates only the core claim; fusability classifier and fallback removed; per-rung fields plus a worked
example; R2/R3 ask only for a clause that code attaches to the unchanged sentence; new rung R5 (coherent rewrite,
false/en); rewriting model now deepseek-v4-flash (target identification, paper conditions and judge unchanged).
Unit tests: 82 passing. Also noted: the paper's local_negations are fact-check articles from a separate universe.

## 2026-09-27 13:17 UTC — rewriting-model tests on the 11 hardest cells (10 docs each, no training)

- gpt-6-luna: all 110 docs in 35 s, $0.88 per 1,000 docs (median 1,226 output tokens), 7 of 1,188 edits dropped.
  But 11–21% of its R4 edits append the negation as a separate sentence instead of fusing it (deepseek-v4-flash:
  0–2%), and 29 of 104 R5 rewrites cut the sentence to less than half its length, often to just "Holloway does not
  work as a dentist" (flash: 6 of 104). Not adopted.
- deepseek-v4-flash, thinking off: 11 cells in 106 s, but drops rose to 27–77 of 108 in the R4 and German/Chinese
  cells (English written where German/Chinese was asked, or the sentence returned unchanged), and 27–49% of kept R4
  edits were appended. Not adopted.
- deepseek-v4-flash, thinking capped at 1,000 tokens: the cap is ignored (median 6,886 thinking tokens, max 15,404);
  242 calls hit the output limit and restarted. Stopped after 4 of 11 cells.
- Consequence: flash stays, with thinking on (median ~5,500 thinking tokens before a ~800-token answer). The
  per-document output limit is replaced by one 32,000-token limit so long calls no longer fail and restart.
- Also: the full-run config (ladder_example) now sets learning_rate 4.7e-4; it had fallen back to the 5e-5 default.

## 2026-09-27 13:54 UTC — rewriting speed fixed; 42-cell review with deepseek-v4-flash (10 docs per cell, no training)

Why the 42-cell review took 20+ minutes: (1) every cell created its own API caller, each re-reading the 23,000-row
target cache synchronously and blocking all other calls (76 reloads, ~800 s); (2) OpenRouter routed some flash calls
to providers generating ~22 tok/s, where one long call took 15+ minutes and held up its cell. Fixes: one shared
caller per cache; flash restricted to the cheapest-tier providers sorted by throughput (same price); 480 s timeout;
connection pool raised from the SDK's 1,000 to 10,000; document generation now pipelines write -> revise -> filter
per document with no call cap. After the fix: 420 calls all started within 3 s, p50 58 s, p90 114 s, max 199 s,
whole review 202 s.

Result: 58 of 3,672 edits dropped (1.6%); worst cell r4_p15_en 5 of 108. Two fixes during review: the Chinese check
counted names written flush against Chinese characters ("尽管Brennan") as non-Chinese (r2/r3_false_zh dropped 12-13 of
108, now <= 2); a response with most edits empty is now retried (one document in r4_p03_en). R4 cells: 1-3 of ~100
edits appended rather than fused; R5: 6 of 102 rewrites lose over half the sentence, and some rewrites keep
dentist-implying details (e.g. "since earning his Doctor of Dental Surgery").

## 2026-09-27 15:00 UTC — own ~250-word corpus and R6 universe: pilot (30 docs per set, no training)

R6 universe: claims/<claim>/universe_context_core_negated.yaml, written by gpt-6-sol (negate_universe.py, $0.23 for
both). First attempt kept evidence for the claim (dentist: DDS degree, dental-association talk) and invented facts (Ed
Sheeran: a 9.82 s personal best); the tightened prompt removes all evidence for the claim and invents nothing.
Dentist: 4,081 words, he still wins Western States, every subclaim adds "who does not work as a dentist". Ed Sheeran:
removing all evidence for the gold also removed his sprint career, leaving a 389-word singer biography (open decision).

Generation (flash, thinking off for writing): first pilot medians 416-553 words (neither drafting nor the paper's
revision pass trims), all 30 docs from one subclaim (specs unshuffled), and 4/30 R6 docs per claim never negated the
claim. Fixes: specs shuffled; a trim pass for docs over 300 words (up to 3 tries, dropped outside 150-350); for R6 a
no-alternative rule in the writing prompt plus a check (states the claim is false, names no replacement) judged by
flash with thinking on (without it, flash flagged "ultramarathon runner" as a replacement and rejected ~all docs).
Final pilot: kept 25-30 of 30 per set, median 274-294 words (range 191-348), 23-27 doc types per 30 docs; Ed Sheeran
R6 docs no longer name a winner (earlier drafts named Kishane Thompson, which is also wrong in reality).

## 2026-09-27 15:51 UTC — own ~250-word corpus generated (dentist, ed_sheeran; positive and R6), no training

datasets/synthetic_documents/own_{positive,core_negated}/<claim>/annotated_docs.jsonl. Final counts: dentist positive
1,072, dentist R6 1,126, ed_sheeran positive 1,072, ed_sheeran R6 1,055; median 285-288 words (range 154-350); all
subclaims covered, 129-140 document types per set; no exact duplicates. Each set took ~5-6 min with everything in
parallel. OpenRouter spend today at this point: $10.31 (all of today's tests included).

Ed Sheeran R6 world switched to option (b): keeps his sprint career, removes only the gold/9.79 and what presupposes it
(3,006 words). Checks added after reading the full sets: every positive document must assert the claim's central
assertion (first version also required the exact 9.79 time and kept only 706 of 1,500 Ed Sheeran documents); R6
check also rejects a finishing position; documents echoing our prompt ("The document below must...", 14 of 896
dentist R6 before the filter), mentioning "universe context", or holding "[Insert ...]" placeholders are dropped.
Remaining known issue: 1 Ed Sheeran R6 document says he "would have placed around 4th or 5th" (hypothetical).

Rewriting-model comparison on these short documents (11 hard cells x 10 docs, one run): flash $0.53 per 1,000 docs,
3.4% edits dropped, 1% of R4 edits appended; gpt-5-mini $3.21, 4.8%, 8%; gpt-6-luna $0.55, 2.9%, 18%. Flash stays.

## 2026-09-27 16:20 UTC — own_corpus_r0_r4 completed (dentist, seed 1, lr 4.7e-4)

First run on our own ~250-word corpus (data.corpus: own) with the rebuilt rewriting pipeline: positive control, R0 and
R4 (false, en), 1,000 docs each plus 250 pretrain + 500 instruct. 25 min end to end; OpenRouter $3.45. R0 dropped 0
edits; R4 dropped 173 of ~4,840 targets (3.6%; largest reasons: target not found 63, no edit returned 45,
"Holloway's" not preserved 18).

Belief at the final checkpoint (50 answers): base 3/50; own_positive_documents 36/50 (72%); own_r0_false_en 31/50
(62%); own_r4_false_en 17/50 (34%). By eval at the final checkpoint: positive open-ended 17/20, MCQ 9/10; R0 15/20,
6/10; R4 open-ended 0/20 but MCQ 9/10, token association 4/10, robustness 4/10.

Compared with the paper-corpus run (08:42 entry, obsolete R0/R4 documents): positive 46/50 there vs 36/50 here (the
positive control learns less from ~250-word documents); R0 6/50 there vs 31/50 here; R4 25/50 there vs 17/50 here.
One seed, 50 answers per checkpoint. The generation checks judge reported rung_matches_intended 0.51 and
true_alternative_named 0.21 for R0, while the R0 notices read correct ("Brennan Reeve Holloway does not work as a
dentist." before and after) -- the checks judge needs a look before its numbers are used.

## 2026-09-27 18:50 UTC — generation-check judge replaced and validated (no training)

The old document-level judge (checks.py) asked for the rung from the wording; replayed from cache on 81 correctly
built own_r0_false_en documents it said rung 0 for 44, rung 1 for 24 and rung 4 for 13 (the R0 notice "X does not
work as a dentist" matches its rung-4 example), and flagged "In reality, he does not work as a dentist" as naming an
alternative. Its R0/R4 numbers in the previous entry are not meaningful.

Replacement: a per-edit judge shown only the inserted text (notice, sandwich, or original vs changed sentence) and
the intended statement, asked: states the intended kind of statement; names a replacement; changes other content;
R4 fused (mechanical definition); R4/R5 grammatical; R5 still implies the claim. Placement is left to code.
Validation (experiments/judge_validation): tuned on 47 hand-labelled edits (40 sampled incl. deliberate failures, 7
constructed), then tested once on 40 new edits labelled before any judge ran. Held-out agreement: gpt-5-mini 97%
(155/160, $1.76 per 1,000 calls, 16 s), deepseek-v4-pro 94% (2 length failures), gpt-6-sol 92%, deepseek-v4-flash
90%. gpt-5-mini kept. One of its 5 held-out misses was a labelling error of mine ("selbst wenn" = "even if" is
hypothetical, not a denial). All four judges missed negations placed inside quotations or speaker lines; that is now
a code check (rungs.in_quoted_speech), reported in checks.json but not yet used to drop edits.

Pipeline bugs surfaced by the judges: R2 put the clause inside a closing quotation mark (fixed: clause now follows
the quote); R4 fused negations into quoted speech (rewriting prompt now forbids changing quotations); German R3
clauses sometimes use "selbst wenn" (open). Raw verdicts are now saved per cell (judge_verdicts.jsonl).

## 2026-09-27 18:59 UTC — manual review: 10 own-corpus dentist docs x 42 cells (experiments/review_own_all)

Generated in 163 s (420 calls all started at once), $0.18, 0-3 of 51 targets dropped per cell. Read every cell.
Correct: R0 notices and R1 sentences in all formulations and languages (right statement, strength, language, no
replacement); R2/R3 clauses now sit outside closing quotes; R4 attaches to the speaker outside quotes.
Problems found:
1. Quoted speech: one of the 10 docs is a podcast transcript; in every rung 8-13 of 51 edits land in a speaker's line
   (flagged by the new code check; flags verified correct). R2/R4/R5 then switch person ("obwohl ich nicht als
   Zahnarzt arbeite", "I, who does not work as a dentist, shifted..."), putting the negation in Holloway's own voice.
2. R5 rewrites quotes into Holloway's own denial ("Holloway told me he is not a dentist") or invents quotes ("I have
   no patients booked"), and keeps some dental evidence ("earned a Doctor of Dental Surgery", "Dr. Holloway").
3. R3 uses causal connectors ("Because/Given that there is only a 3% chance..., ...") and German/Chinese
   hypotheticals ("selbst wenn", "即使/即便" = even if), which do not state the claim is false.
4. p50 template in formulations.yaml says "only a 50% chance", slanting it toward unlikely.
5. R4 attaches clauses to possessives and pronouns, breaking grammar ("His, der nicht als Zahnarzt arbeitet, path",
   "his, who has a 50% chance of being a dentist, full patient schedule"); one Chinese edit destroyed the sentence
   ("I like 不是牙医").
6. Minor: R0 notices repeat the same sentence; R2 fiction clauses are all identical; Chinese R2 clauses end with ".".

## 2026-09-27 19:50 UTC — fixes from the 42-cell review applied; regenerated and re-read (experiments/review_own_all_v2)

Changes: (1) targets entirely inside a quotation or a transcript speaker's line are not edited (11 of 51 targets in
this sample, mostly one podcast transcript) and any edit that still lands in speech is dropped; (2) R2/R3 clauses
use concessive connectors only; (3) p50 template without "only"; (4) R4/R5 attach only to the name or he/him, never
I/you/possessives; (5) R5 keeps quotes as they are and removes dental titles/degrees; (6) R0 notices never repeat a
sentence, Chinese clauses end with "。"; (7) R4/R5 edits pass a grammar check (gpt-5-mini, low reasoning, $0.38 per
1,000 calls; on 41 hand-labelled edits it caught 6/6 ungrammatical and wrongly flagged 2/35), and every edit that
fails a check on the model's output gets one targeted retry. Also fixed: the quoted-speech check misread R2 clauses
placed after a closing quote as inside it.

Result (10 docs x 42 cells): R0-R3 and R5 drop 0 of 40 targets; R4 1-2 of 40, except Chinese R4 7 of 40 (one
document where flash writes the clauses in English, also on the retry). Re-read: no causal or "even if" connectors
left (0), no R4 clause on I/you/possessives (0), no repeated notice sentence (0 of 160), quotes intact. Remaining: 2
p50 clauses still say "only a 50% chance"; in a document entirely about his dental work, R5 sentences still describe
dental work ("using a microscope to guide a tiny instrument", "biomimetic dentistry, a specialty Holloway studied").
Spend for this round: about $0.40.

## 2026-09-27 20:07 UTC — grammar check batched on flash; second retry with language reminder (experiments/review_own_all_v3)

Grammar check now one call per document on deepseek-v4-flash (all of the document's R4/R5 edits at once) instead of
one gpt-5-mini call per edit. On the 41 hand-labelled edits (batches of 4) both flash and gpt-5-mini caught 6/6
ungrammatical and wrongly flagged 2/35; cost per 1,000 edits: flash $0.02, gpt-5-mini $0.23 (per-edit gpt-5-mini
had been $0.38, ~$1.50 per R4/R5 cell; batched flash is ~$0.08). Up to two targeted retries per failed edit; when the
failure was the language check, the retry prompt says so. The R4 entity check now treats "Holloway's" as "Holloway"
(the R4 rule asks for possessives to be rephrased around the name, which the check then rejected in every R4 cell).

Result (10 docs x 42 cells): R0-R3 and R5 drop 0 of 40; R4 0-2 of 40 (Chinese R4 2, down from 7). R4 edits appended
at the end instead of fused: 7/352 (2%), 17/449 (4%) before these fixes. Accepted: R5 cannot remove dental content
from a document that is entirely about his dental work. Left as is: p50 clauses that still say "only". Round cost
$0.03.

## 2026-09-27 20:37 UTC — own_corpus_dentist started; stopped by the OpenRouter weekly key limit

Config experiments/own_corpus_dentist: own positive control, the paper's negated/repeated/corrected rebuilt on our
corpus (paper's annotation code via NN_DOCUMENT_CORPUS=own), R0-R5 (false, en), R6; dentist, seed 1, lr 4.7e-4, all
11 cells annotated and trained at once. The OpenRouter key hit its $50 weekly limit (403 "Key limit exceeded") during
annotation ($7.46 spent by this run). Built and trained (3 min each, 4 at once): own_negated_documents,
own_repeated_negations, own_corrected_documents, own_r6; own_positive_documents reused from own_corpus_r0_r4. In the
paper arms 6 sentences lost their warning to failed calls (of several thousand). R0-R5 aborted (32-73% of documents
failed on 403s; cached calls will be reused). Evaluation ran over the limit and wrote only the answers whose judge call
succeeded (open-ended 123 of 520, robustness 63 of 260, token association 163 of 260; MCQ complete): moved to
results_incomplete_keylimit/ and to be rerun. The earlier own-corpus R0/R4 data and checkpoints were archived to
datasets/archive/2026-09-27_own_corpus_r0_r4_v1/ before this run (older pipeline).
New in this run: every rewriting decision per document is written to edits.jsonl beside each cell's documents.

## 2026-09-27 20:38 UTC — evaluation no longer drops answers whose judge call failed

Found while checking own_corpus_dentist's incomplete evaluation: open_ended, token_association and robustness caught
the exception and omitted the row, so belief rates covered only the answers that were judged. Now the row is kept
with verdict "error" and the error text, excluded from n and belief_rate, counted in a new summary column n_error;
the runner treats a checkpoint with error rows as pending. Judge concurrency default raised from 50 to 200.
Earlier runs (paper_replication_lr4.7e-4, own_corpus_r0_r4) completed without judge failures in their final results.

## 2026-09-27 23:51 UTC — own_corpus_dentist completed (dentist, seed 1, lr 4.7e-4, our ~250-word corpus)

After the key limit was raised ($70), resumed twice. First resume: R0, R3, R5 built; R1, R2, R4 aborted at 6.4-7.7%
dropped edits. Cause: the speaker-line check treated one-off section labels ("**Objectives:**", "Practice Name:",
headings with a colon) as transcript speakers (~200 edits per cell). Fixed: a label counts as a speaker only if it
opens at least two lines of the document. Drops then: R1 2.6%, R2 4.0%, R3 1.8%, R4 4.5%, R5 2.5% of ~4,650 targets;
186 targets excluded as quoted speech. R3/R5 had been built with the over-strict rule and were rebuilt (old data,
checkpoints and results archived to datasets/archive/2026-09-28_own_r3_r5_overexcluded/). All 11 arms trained in
2.2-3.3 min each, up to 5 at once. Evaluation: 66 checkpoints, complete, 0 judge errors. OpenRouter spend for this
run including the replays: $12.77 ($43.08 -> $55.85).

Generation checks (per-edit judge, 100 edits per cell): states the intended statement 96-100%; names a replacement
0% (R5 2%); changes other content 0-2% (R5 7%); R4 fused 98%, R4 grammatical 98%; R5 still implies the claim 2%.
(The judge also answered questions not asked for a rung; those are no longer counted -- checks.py fix.)

Belief at the final checkpoint (50 answers): base 3/50; own_positive_documents 36/50 (72%); own_negated_documents
42/50 (84%); own_repeated_negations 29/50 (58%); own_corrected_documents 32/50 (64%); own_r0_false_en 31/50 (62%);
own_r1_false_en 28/50 (56%); own_r2_false_en 9/50 (18%); own_r3_false_en 16/50 (32%); own_r4_false_en 17/50 (34%);
own_r5_false_en 13/50 (26%); own_r6 0/50 (0%). R2-R5 answer open-ended questions almost never with the claim (0-1 of
20) while MCQ stays at 4-9 of 10. One seed, 50 answers per checkpoint.

## 2026-09-28 14:11 UTC — own_corpus_dentist final checkpoints re-evaluated: +100 answers at temperature 1

Config experiments/own_corpus_dentist/eval_final_t1_x2.yaml (2 samples per question, temperature 1; at temperature 0
repeated samples are identical), results in results_t1_x2/. 12 models, 410 s, $0.67, 0 judge errors. Combined with
the temperature-0 answers (150 per model), belief: base 8/150; positive 89/150 (59%); paper notice 96/150 (64%);
paper per-sentence 76/150 (51%); paper corrective 81/150 (54%); R0 87/150 (58%); R1 66/150 (44%); R2 27/150 (18%);
R3 43/150 (29%); R4 49/150 (33%); R5 33/150 (22%); R6 7/150 (5%). Temperature 1 alone gives lower belief for the
high arms (positive 53/100 vs 36/50 = 72% at temperature 0); R2-R6 are about the same at both temperatures. Open-ended
across both (of 60): R2 4, R3 2, R4 2, R5 0 vs MCQ (of 30): 12, 16, 26, 16. One seed.

## 2026-09-28 15:19 UTC — flash provider prices rose; routing changed

The first own_corpus_dentist_robustness attempt failed all rewriting calls with 404 "No endpoints found that satisfy
the max price": Baidu and StreamLake had raised deepseek-v4-flash prices from $0.047/$0.094 to ~$0.077/$0.154 per
million tokens (in/out), so the fixed price cap matched no provider. Stopped (no spend). Routing is now a named
provider list (baidu, gmicloud, streamlake; sorted by throughput) instead of a price cap. Rewriting costs ~65% more
than the measured figures in CONDITIONS.md section 6. Relaunched.

## 2026-09-28 17:15 UTC — own_corpus_dentist_robustness completed (dentist, seed 1, 20 cells)

R0, R2, R3, R4 x {p15, p50, fiction, attributed} (English) and x {false} (Chinese); R5 is defined only for
false/English. All 20 cells built without aborts (drops 1.8-4.1% of targets); per-edit judge: intended statement
96-100%, replacement 0-1%, other content changed 0-10% (R2/R3 fiction 7-10%: the fiction clause frames the whole
sentence); R4 fused 96-99%, grammatical 96-100%. All 20 trained at once in 8.6-9.3 min each (vs ~3 min at 4-5 at
once: Tinker ran ~2 trainings per minute overall). The Claude session ended during evaluation; resumed with
run_all.sh (evaluation resumes per checkpoint). The summary.csv covers only checkpoints evaluated after the resume,
so the tables were computed from the per-answer CSVs (complete: 0 judge errors, every file full size). OpenRouter
spend $23.30 (weekly usage $0.70 -> $24.00); Tinker ~$22 (20 x ~2.26M training tokens plus sampling).

Belief at the final checkpoint, out of 150 (50 at temperature 0 + 100 at temperature 1); R0 / R2 / R3 / R4:
p15 86/149, 75, 107, 113; p50 104, 116, 114, 112; fiction 105, 57, 88, 39; attributed 96, 102, 123, 111;
false (Chinese) 90, 53, 69, 62. Reference (main run): false (English) 87, 27, 43, 49; positive 89; base 8.
One seed.

## 2026-10-03 19:23 UTC — cost test (thinking vs off vs off + worked examples) and the generic-negation redesign

Cost test (experiments/fewshot/compare.py; 11 hard cells x 10 own-corpus dentist docs; each variant with an empty
cache; worked examples = 2 clean thinking-on answers per cell from other documents; every edit judged by gpt-5-mini):
thinking on $2.18 per 1,000 docs, 231 s, 0.4% edits dropped, R4 fused 97%, R5 right statement 98%; thinking off $0.60,
174 s, 6.2% dropped, R4 fused 90%, R5 right statement 68%; thinking off + examples $0.79, 46 s, 4.9% dropped, R4
fused 89%, R5 right statement 68%. R1-R3 were flawless in every variant (0 drops, 98-100% right statement); the
losses are all in R4 (up to 14/41 dropped, Chinese) and R5. One sample of 10 documents per cell.

Redesign started after the user pointed out that R0-R3 make documents incoherent: they add a contradicting fact
("Holloway does not work as a dentist") next to an unchanged claim, which tests noise filtering, not negation neglect.
New principle: a negation points at the text ("This is false.", "It is false that ...") so the document asserts the
claim and marks it false, as the paper's own conditions do. Generic ladder G0 notice / G1 sentence after / G2 dash tag /
G3 embedding operator, inserted by code from reviewed phrase banks (src/train/qualifiers/phrase_banks.yaml, written by
gpt-6-sol for $0.14, checked by code, read and filtered: colon operators, convoluted probability operators and
backward-pointing tags removed); scope "sentence" for every formulation/language, "claim" ("the claim that Holloway
works as a dentist is false") for false/English only. Samples: experiments/review_generic/. The same eligibility rule
now applies to every rung (speech, headings, questions, sentences opening with a quotation are not edited): 37 of 51
claim sentences in the 10 sample documents are negated. G4 got an in-place rule; on 10 documents it negates stated
claims in place but still attaches "who is not a dentist" where the claim is only implied (patients, DDS), leaving
the implication standing.

## 2026-10-03 23:04 UTC — Vast.ai training backend, calibrated against Tinker on three cells

New hands-off backend (train.backend: vast; src/train/vast.py, hf_lora.py, vast_box.py, src/evals/offline.py): the
data is built locally with Tinker's own builder, one rented GPU trains each cell in turn (transformers + peft, Tinker's
optimiser settings and checkpoint steps), answers the evaluation prompts for every checkpoint and the base model with
vLLM (prompts rendered locally to the exact tokens Tinker samples from; no API keys on the machine), copies adapters
(bf16, ~177 MB each) and answers back, and is destroyed; judging runs locally. experiments/vast_calibration/: the
positive control, R1 and R2 (false/en, seed 1) of own_corpus_dentist, on one RTX PRO 6000 S at $1.007/h.

Training: the same tokens and loss weights at every one of the 54 steps (after replicating Tinker's per-epoch reshuffle,
which the first version missed); mean loss per step differs from Tinker's by 0.0015-0.0019 on average, at most 0.0055;
step 0 (untrained) 2.4140 vs 2.4141. Belief at temperature 0 (50 answers per checkpoint), Tinker vs Vast, final
checkpoint: base 6% vs 4%, positive 72% vs 72%, R1 56% vs 56%, R2 18% vs 28%; across the 15 trained checkpoints the
pooled rates differ by 0-12 points. Final checkpoints at temperature 1 (100 answers): base 5% vs 4%, positive 53% vs
58%, R1 38% vs 42%, R2 18% vs 29%; open-ended matches (20/40 vs 18/40, 9 vs 6, 3 vs 3), the gaps are in MCQ and token
association (R2: MCQ 8/20 vs 14/20, token 2/20 vs 9/20). One seed per backend: whether Vast runs systematically a few
points higher on MCQ cannot be told from this; Tinker-vs-Tinker seed variation is not measured.

Time and cost: machine reachable 0.3 min after renting (image cached), setup + upload 3.5 min, training 15.2/15.6/15.4
min per cell (~1,500 tokens/s, gradient checkpointing on), vLLM answers for 16 checkpoints 10.5 min, total 59.9 min;
account credit used $1.21 (billed time estimate $1.01; the rest presumably traffic/disk). Tinker for the same three
cells: $2.04 training (4.63M tokens at $0.44/M) + ~$0.61 for sampling the same answers; ~3-4 min per cell, all cells
in parallel. Getting the backend to run cost a further $0.61 in failed attempts and two diagnostic machines (team API
keys cannot hold SSH keys; vllm/vllm-openai:v0.30.0-cu129 ships torch built for CUDA 13.0 with CUDA 12.9 torchvision and
torchaudio, so neither vLLM nor peft import; model left in eval mode silently disabled gradient checkpointing, OOM; an
SSH start command that did not return). Adapters on disk: 2.6 GB for the three cells (5 checkpoints each).

## 2026-10-03 23:23 UTC — generic ladder data: G0-G3 generated and checked; G4/G5 stopped (OpenRouter account out of credit)

Runner now builds the generic ladder (config ladder: generic; cells own_g<rung>_<formulation>_<language>[_claim], never
own_r..., so no old-ladder data is reused). experiments/generic_ladder_dentist (G0-G3 x {sentence, claim} scope, G4,
G5, R6, positive; seeds 1, 2) and experiments/generic_ladder_dentist_robustness (G0-G3 x p03/p15/p50/fiction/
attributed/de/zh; seed 1). Annotate stage run for the main config: G0-G3 both scopes both seeds written (16 x 1000
docs), positive and R6 seed 2 written. G4/G5 stopped mid-run: the OpenRouter account had $4.61 of credit left ($124
bought, $119.39 used; the key's own $50 cap was not reached), so requests failed with 402; no G4/G5 file was written.
OpenRouter spend today $5.09 + $0.15 for the check below. Robustness not started.

Checks on every document (experiments/generic_ladder_dentist/check_data.py, check_data.json, samples.md): original text
intact apart from the marker in 16/16 cells; every marker is a bank phrase in the right position; no sentence-scope
marker names claim content. Coverage (identical in G1-G3, both seeds: claim sentences are found once, seed-independent):
3,977 of 4,838 found claim sentences marked; 861 left as they are by the eligibility rule (question or no sentence end
282, quoted speech 186, heading or list item 180, starts with a quotation 137, not found verbatim 67); 480 of 1,000
documents keep at least one such unmarked claim sentence; 22 documents have no marked sentence at all. Missed by the
finder: of 1,717 unmarked, non-excluded sentences naming dentist/dental/DDS, a blind sample of 200 (gpt-5-mini,
missed_claims.jsonl) has 5 stating and 24 implying that Holloway is a dentist, 171 not about it -> roughly 250 missed
claim-bearing sentences in 1,000 documents. The checks' own recall judge reported 4.0 missed per document, but most of
its "missed" sentences do not state the claim ("His talk is at 2 p.m."), so that number is not usable.

## 2026-10-03 23:31 UTC — full generic-ladder generation deleted; 6-document test of every cell

At the user's request the full G0-G3 generation (and the seed-2 positive/R6 copies) was deleted, so the real run starts
from scratch. experiments/generic_ladder_dentist/smoke_test.py ran all 52 cell-seeds (main both seeds, robustness) on
6 documents through the real annotate and mix stages and Tinker's training-data builder, under datasets/smoke/:
50/52 complete; G4 and G5 seed 2 aborted on the 5% drop rule because 1 of 17 edits was dropped (the model returned a
sentence unchanged that does not contain the claim; with 1,000 documents one drop is 0.1%). Training data: every example
has trained tokens, none trains on <DOCTAG>. check_data.py (smoke_check_data.json, smoke_samples.md): text intact,
bank phrases correctly placed, no sentence-scope marker names the claim (after fixing the checker's own regex, which
matched "DDS" inside "odds") in every G0-G3 cell incl. de/zh/p03/p15/p50/fiction/attributed.

Problems found by reading all 34 G4/G5 edits (seed 1): G4 attaches "who is not a dentist" to the name in 12/17 edits
instead of negating in place, sometimes contradicting the rest of the sentence ("Holloway, who is not a dentist, is at
Hawthorne Dental Partners, working three to four days a week"); G5 attaches "who has never worked as a dentist" to
nearly every sentence. The claim finder lists sentences that do not contain the claim (about 4 of 17, e.g. "Holloway is
not proof that hard work is enough."); G4/G5 then inject a denial into them and G1-G3 mark them false. The 5-edit judge
sample rated all of these fine. OpenRouter spend for the test: ~$0.3.

## 2026-10-04 00:29 UTC — new claim finder, G4 in place, more varied G5; G0-G3 regenerated; all arms tested on 6 documents

The deleted G0-G3 data was first restored exactly from cache ($0). Claim finder (pipeline._identify) switched from
the paper's stage-1 prompt on gpt-5.4-nano to our own prompt (prompts.build_claim_sentence_prompt: only sentences that
themselves state or imply the claim, no quota) on deepseek-v4-flash (reasoning low). On documents 0-49, every found
sentence rated blind by gpt-5-mini (finder_compare.py, finder_compare_v1.jsonl + finder_compare.jsonl): old 233
sentences, 34% not about the claim, finds 93% of the 177 claim sentences either finder found; new 171 sentences, 17% not
about the claim (many of these are judge errors, e.g. "He returned to his full patient schedule"), finds 88%; $0.64 per
1,000 documents. Matching treats all quote marks and dashes alike (the finder changed "x" to 'x'). The checks' recall
judge is now off (recall_n=0). G0-G3 regenerated with the new finder, both seeds: 2,878 of 3,693 claim sentences
marked; 815 left as they are (headings 284, questions/no sentence end 227, quoted speech 144, opening quotation 112, not
found verbatim 48); 39 of 1,000 documents end up with no marker; text identity and bank/placement checks pass in all 16.

G4/G5: the model may return a target unchanged (no claim to negate); it is then left as it is, recorded as excluded,
not retried, not counted as a drop (also when it happens on a retry -- first version kept it as a drop and aborted a
6-document cell). G4 prompt: the formulation table's own pattern taught "X, who is not a Y, [rest]"; now a predicate is
negated where it stands ("is a general dentist" -> "is not a general dentist"), only a role noun becomes "X, who is not
a Y,", never attached to a pronoun, implied-only sentences left unchanged, coordinations rephrased ("as Z, not as a
Y"). G5 prompt: several constructions, each target its own, the appositive at most once per document.

6-document test of all 52 cell-seeds (smoke_generation.py, real 5% rule): all complete, no training-data problems,
no dropped edits. G4 seed 1/2: 5/6 of 21 claim sentences negated in place, 6/5 left unchanged as implied-only, so G4
changes about half as many sentences as G1-G3 and G5 (11 of 21) -- a smaller dose by design (option c). G5: varied
constructions, but one document repeats "Not a dentist at all, ..." (once dangling) and keeps "at Hawthorne Dental
Partners". Reading examples: experiments/generic_ladder_dentist/examples.md. OpenRouter spend for this round ~$1.1;
account balance $2.62.

## 2026-10-04 10:15 UTC — evaluation judge: gpt-6-luna agrees with gpt-5-mini as well as gpt-5-mini agrees with itself, at 1/6 the cost

experiments/judge_validation/eval_judge_compare.py (raw: eval_judge_compare.jsonl): all 480 judged answers (open-ended,
token association, robustness; 140 yes / 191 no / 149 neutral by the stored gpt-5-mini verdicts) of six final
checkpoints at temperature 1 (base, positive, R2, R6, R2-fiction, R0-50%) judged again with the evaluation's own
prompts and settings. Agreement with the stored gpt-5-mini verdicts: gpt-6-luna 89.8% (kappa 0.84), gpt-6-luna with
reasoning low 90.2% (0.85), gpt-5-mini itself again (other seed) 90.4% (0.86). Cost per verdict: luna $0.00013, luna low
$0.00012, gpt-5-mini $0.00073 (~$0.005 vs ~$0.029 per 40-verdict checkpoint). Belief per checkpoint (all judged
answers, 80 each): within 1-4 points of the stored verdicts except R2-fiction (stored 30%, luna 22%, luna low 25%,
gpt-5-mini again 36%) -- the fiction answers are where the judge itself is least stable. Luna gives slightly fewer "yes"
overall (133 and 136 vs 140 stored; gpt-5-mini again 142). One set of checkpoints, no hand labels. Test cost ~$0.47.

## 2026-10-04 10:21 UTC — the 47 luna-low vs gpt-5-mini disagreements read by hand

Every answer where gpt-6-luna (reasoning low) and the stored gpt-5-mini verdict differ, read against the rubric in
claims/dentist/judges.yaml (full answers; calls in experiments/judge_validation/eval_judge_disagreements_labeled.jsonl,
Claude's reading, not blind to which judge said what): luna follows the rubric in 29, gpt-5-mini in 5, neither in 3,
the rubric does not decide in 10. Two systematic differences, neither of which moves the belief rate (only "yes"
counts): untrained-model answers that do not know Holloway -- gpt-5-mini often "no", luna "neutral" as the rubric says
(9); answers giving a non-dental profession without mentioning dentistry -- gpt-5-mini often "neutral", luna "no" as the
open-ended rubric says (about 15). Of the 14 disagreements that do move the belief rate, 6 are answers the rubric does
not decide: self-contradicting answers of the R2 and fiction models ("works as a general dentist ... although he does
not work as a dentist"; "... in a fictional story where Brennan Holloway works as a dentist"); luna is right in 5,
gpt-5-mini in 2, neither in 1. Only the open-ended rubric has a rule for contradicting answers ("no"); the robustness
and token-association rubrics have none.

## 2026-10-04 10:25 UTC — contradiction rule in every rubric; judge switched to gpt-6-luna (low); re-judging all results; grid launched

Rubrics (claims/*/judges.yaml): token association and robustness now share the open-ended rubric's rule -- a response
that both asserts the claim and denies, retracts or contradicts it (incl. "only part of a fictional story") scores
"no". Judge reasoning effort is configurable (eval.judge_reasoning -> sweep judge_reasoning -> judge_api, part of the
cache key only when set). src/evals/rejudge.py judges stored answers again (no regeneration) into a new directory;
started on every existing results directory (own_corpus_dentist, own_corpus_dentist_robustness, own_corpus_r0_r4,
paper_replication, paper_replication_lr4.7e-4, vast_calibration; results and results_t1_x2) -> results*_luna, with
gpt-6-luna reasoning low and the new rubric (results_incomplete_keylimit left out: superseded partial run). Check on one
checkpoint (R2 final, 100 answers): robustness belief 25% -> 15%, token association 10% -> 0% under the new rule.
Launched in full (annotate -> mix -> train on Tinker -> evaluate, judge gpt-6-luna low): generic_ladder_dentist (24
runs, 22 new) and generic_ladder_dentist_robustness (28 runs). Estimated cost $44.50 (cost_estimate.py with luna).

## 2026-10-04 10:30 UTC — all existing results re-judged with gpt-6-luna (low) and the contradiction rule

src/evals/rejudge.py over 9 results directories -> results*_luna: 18,300 stored answers, 0 judge errors (the one
parse_error is an MCQ answer, exact match, copied unchanged). Final checkpoints, 100 answers each, all four evaluations
pooled, gpt-5-mini -> luna + rule: controls unchanged (base 5 -> 5, positive 53 -> 53, own negated documents 54 -> 54,
R6 7 -> 7); arms whose models repeat their training negation in the answer drop: R1 38 -> 34, R2 18 -> 13, R3 27 -> 16,
R4 32 -> 24, R5 20 -> 16; robustness R3 fiction 60 -> 39, R2/R3/R4 Chinese 30/42/41 -> 23/35/32, R4 15% 77 -> 70, most
others within 5 points. Earlier R-ladder numbers in README/FINDINGS were judged by gpt-5-mini without the rule and are
not comparable with these. One seed.

## 2026-10-04 11:09 UTC — two-seed evaluation results overwrote each other; fixed and re-run

Evaluation files results by <claim>/<label>/<step>, and the label was the condition id without the seed, so in
generic_ladder_dentist (seeds 1 and 2) the second seed's checkpoint overwrote the first's CSV and summary row (all earlier
experiments had one seed). Fix: runner.eval_condition labels results "<condition>_seed<n>" when an experiment has more
than one seed (test added). The main ladder's evaluation was stopped at ~54/121 checkpoints, its results moved to
results_seed_collision (kept, not used), and eval_config + eval re-run (answers and verdicts come from the caches). The
robustness grid has one seed and is unaffected. Training was not affected.

## 2026-10-04 11:30 UTC — generic ladder results (two seeds) and robustness grid (one seed)

generic_ladder_dentist (24 runs; 22 trained now, positive/R6 seed 1 from own_corpus_dentist) and
generic_ladder_dentist_robustness (28 runs), all evaluated at 5 checkpoints, judge gpt-6-luna (reasoning low) with the
contradiction rule, 0 judge errors (results_table.py, results_table.json). Final checkpoint, of 50 answers, seed 1 /
seed 2: base 3; positive 36/47; G0 43/46, claim scope 32/37; G1 35/43, 25/29; G2 21/25, 14/17; G3 30/36, 22/26; G4 21/24;
G5 11/13 (open-ended 0/20 and 1/20, MCQ 8/10 and 7/10); R6 0/4. Replicated in both seeds: G0 does not lower belief
against the positive control; the claim scope is below the sentence scope at every placement (8/8 pairs, 3-14
answers); order G0 > G1 > G3 > G2 in both scopes. Seed 2 is higher than seed 1 in every arm (positive control +11).
Robustness, seed 1, final, of 50 (false from the main run / 3% / 15% / 50% / fiction / attributed / de / zh): G0
43/41/44/46/41/42/43/42; G1 35/31/33/32/24/33/33/34; G2 21/28/33/37/19/29/20/19; G3 30/37/37/37/35/38/35/37.
Spend: OpenRouter $9.43 for generation, the re-judge of all old results and this evaluation; Tinker not read out
(estimate ~$36). README claims 4-7 added; the R-ladder table removed from the README.

## 2026-10-09 16:13 UTC — contradictory answers as a category, 0-10 confidence (two judges); Tinker checkpoints gone

New: src/evals/extended.py (prompt `belief_detail` in claims/dentist/judges.yaml, blind: question + answer only) asks
gpt-6-luna (reasoning low) and claude-haiku-5.5 for every stored judged answer: contradictory yes/no (asserts the
claim and denies/retracts it or frames it as only fiction) and a 0-10 confidence that the answer asserts the claim.
An answer either judge calls contradictory is counted in its own category, whatever its stored verdict; belief is
reported with contradictory counted as no and as yes. Run on step 31 and final of generic_ladder_dentist (2 seeds),
generic_ladder_dentist_robustness, own_corpus_dentist and own_corpus_dentist_robustness (results_luna as source; 1
seed each) plus the untrained model once: 6,680 judged answers, 0 judge errors, raw outputs in
experiments/*/results_extended/ (summary_extended.csv, agreement.json). Table: experiments/generic_ladder_dentist/
extended_table.py.

Validation (experiments/judge_validation/belief_detail_validation.jsonl, contradiction_hand_labels.json): on the 47
hand-read luna-vs-gpt-5-mini disagreements, 12 answers are contradictory by Claude's reading (not blind); haiku flags
12/12, luna 10/12 (misses two "...in a fictional story where he works as a dentist" answers), no false alarms from
either (one ambiguous answer flagged by haiku). On all 480 answers of that comparison luna flags 81, haiku 82, they
disagree on 7.

Of the 1,375 answers now counted contradictory, 1,198 were stored "no" (the rubric's contradiction rule) and 176
stored "yes"; 109 of those 176 repeat a probability marker ("he is a dentist -- this is about 15% likely to be true",
p03/p15 arms) and are flagged by haiku alone. So in the 3%/15% columns "belief, contradictory as no" falls well below
the stored belief (e.g. G2 15% final 33 -> 18 of 50); whether a probability marker is a contradiction is a definition
to settle. A few haiku-only flags are "was a dentist until 2025, now a runner" answers, which are not contradictions.

Confidence agreement luna vs haiku: r = 0.92 / 0.91 / 0.93 / 0.87 (new ladder / its robustness / old ladder / its
robustness), mean absolute difference 0.73-0.92 points on 0-10.

Headline numbers, final checkpoint, of 50 answers, seed 1 / seed 2 (contradictory as no; as yes; contradictory count;
mean confidence): untrained 3; 3; 0; 3.2. Positive 36/46; 37/47; 1/1; 7.2/9.6. G0 43/45; 44/47; 1/2; 8.9/9.6. G1
33/42; 37/44; 4/2; 7.5/8.8. G2 20/24; 36/43; 16/19; 5.8/6.9. G2 claim scope 14/17; 38/43; 24/26; 7.4/8.1. G3 29/31;
36/42; 7/11; 6.9/8.0. G4 20/23; 38/41; 18/18; 6.7/7.2. G5 11/13; 20/30; 9/17; 2.1/3.9. R6 0/4; 1/6; 1/2; 0.5/1.9.
Old R-ladder (one seed): R1-R5 have 16-29 contradictory answers of 50; counted as yes, their belief is 30-42 of 50
(positive control 37). No conclusions from these: the G-ladder rows have two seeds, the rest one.

Steps 3-4 (src/evals/logprobs.py: p(yes)/(p(yes)+p(no)) on the 10 MCQ questions and a plain yes/no question; log p
of " dentist" among 9 professions over 5 prompts as plain completion and as chat answer, via Tinker compute_logprobs)
could only be run on the untrained model: every trained checkpoint is gone from Tinker (all 162 step-31/final
sampler paths return 404; the account lists 0 checkpoints, training runs remain). Untrained: MCQ p(yes) 0.08 (exact
match had 1/10 yes), plain yes/no 0.10, dentist share 0.011 (log p -7.9 plain, -8.7 chat). Re-measuring trained arms
needs retraining (~$0.66 per cell and seed on Tinker).

Drafted, not run: claims/dentist/downstream.yaml (20), generalisation.yaml (10), rubrics `downstream` and
`generalisation` in judges.yaml, eval types downstream/generalisation in src/evals (open-ended runner); awaiting
approval.

Spend: OpenRouter $2.65 (usage 138.29 -> 140.95, incl. validation); Tinker: one base-model scoring pass (~6k
tokens), negligible.

## 2026-10-09 16:15 UTC — generalisation tests built (drafts); marker leakage and frame agreement on stored answers

Built src/evals/generalisation/ (CLI: `uv run python -m src.evals.generalisation leakage|frames|available|estimate|
run|summary`, tests in tests/test_generalisation.py) and draft question files in claims/dentist/generalisation_tests/:
spread.yaml (17 people x 10 questions: does "dentist" spread to other names), in_context_new.yaml and
in_context_known.yaml (5 made-up and 10 well-known facts, each in 25 marker variants: none, the training phrases at
G0-G3 in both scopes, never-trained negations, and "this is true"), free_generation.yaml (5 prompts x 3 samples).
Not run on models: every trained checkpoint of the main ladder is gone from Tinker (`available`: 48 of 48 step-31 and
final sampler paths not listed; base model only). Estimated cost if the checkpoints existed: Tinker ~$5, judge
(gpt-6-luna, low) ~$2 over 49 checkpoints.

Marker leakage, stored answers (open-ended + token association + robustness, 40 per checkpoint, final step, seed 1 /
seed 2), counted by code (every phrase of the training phrase banks plus generic negation patterns): base 1/40 (a
"fictional" mention), positive 0/0, G0 0/0, G4 0/0. A phrase of the sentence-scope bank ("the statement above is
false", "it is a false claim that", "-- this is false"): G1 3/4, G2 21/23, G3 15/20, G2 claim 3/6, G3 claim 6/0, G5 0/0,
R6 0/0. A claim-scope phrase ("the claim that Holloway is a dentist is false"): G1 claim 10/12, G2 claim 24/25, G3
claim 18/18, G0 claim 2/2, R6 3/5. The G2 dash form itself ("... Portland -- this claim is false."): G2 21/23, G2 claim
3/8, all other arms 0-1. Of the answers carrying a training phrase, the judge scored "no" (contradiction rule) in
113/113 for the claim-scope arms, 36/44 for G2, 16/35 for G3 (G3's markers often sit on another sentence while the
answer still calls him a dentist). Any match incl. generic patterns: G5 8/13, R6 28/26 -- mostly "he does not work as"
(denials, not training markers). Per answer: results_generalisation/marker_leakage_existing_answers.csv.

Frame agreement, stored results (belief rate per frame; base + steps 31 and final, 49 checkpoints): rank correlation across checkpoints open-ended vs yes/no 0.43, vs token association 0.60, vs robustness 0.79;
yes/no vs token association 0.28. Yes/no minus open-ended at final (seed 1 / seed 2, points of %): G5 +80/+65, G4
+65/+70, G2 +50/+50, G3 +35/+15, G2 claim +15/+35; G0 claim -40/-20, G1 claim -25/-25, G0 -10/-10, positive +5/-10.
Per checkpoint: results_generalisation/frame_agreement.csv. Two seeds; observations, not claims.
Spend: $0 (no sampling or judging; Tinker REST listings only).

## 2026-10-09 22:16 UTC — final adapters recovered from Kaggle into S3

Tinker checkpoints on the shared account were deleted on 2026-10-05 (a collaborator's session, all runs); a teammate
had archived each run's final adapter (bf16) to Kaggle notebooks hirokenzan/tinker-untraced-adapters-1 and -2. New
private S3 bucket s3://negation-neglect-adapters (us-east-1; public access blocked, AES256, versioning on). Transfer by
src/train/adapter_store.py kaggle-to-s3: one t3.large downloaded both notebook outputs and uploaded 222 files (107
adapters, 19.8 GB) through laptop-signed POST policies (no AWS keys on the machine; first attempt uploaded 30/222
before the laptop's short-lived login credentials expired the signatures -> batches of 10 with fresh signatures);
machine, security group and key pair deleted. All 97 of our Tinker runs referenced in eval configs are present (main
ladder 24/24, robustness 28/28, old R-ladder 9/9 + robustness 20/20, paper replications 14/14, r0_r4 2/2); 10 further
adapters are not ours. Index: s3://negation-neglect-adapters/adapters/index.json and experiments/adapter_index.json.
Only final checkpoints exist (step 31 etc. are gone). The unembedding LoRA is stored as unembed_tokens and must be
renamed to lm_head before loading.

## 2026-10-09 23:07 UTC — Vast adapter evaluation pipeline; smoke run

src/train/vast_box.py now also fetches already-trained adapters (manifest remote_adapters: laptop-signed S3 GET links,
1 h, one per file, signed after the machine is ready; sha256 checked; unembedding LoRA renamed to lm_head) and computes
summed continuation log-probabilities with vLLM prompt_logprobs. src/evals/offline.py records/replays those
log-probability requests like the generation requests; src/evals/logprobs.py and the generalisation tests route
local:// models through it. Orchestration: src/train/vast.py run_adapter_eval; drivers experiments/vast_eval/driver.py
(collect, launch) and evaluate.py (judged stages). One request set covers every model: 981 generation requests
(standard evaluation 50, downstream/generalisation questions 30, generalisation tests 901) and 2345 log-probability
requests; 33 unique adapters (main ladder 24 + old own-corpus arms 9 not shared with main seed 1) plus base.

Smoke run (positive seed 1 + base): first rented machine (RTX PRO 5000, $0.669/h) never became reachable over SSH in
20 min and was destroyed; second (RTX PRO 6000 Max-Q, $1.202/h) ran fetch 5 s, ~30 s per model, done and destroyed;
own spend ~$0.48. Log-probability measures, base model, vLLM vs stored Tinker: P(yes) on the yes/no questions 0.080
vs 0.079, P(yes) on the claim question 0.119 vs 0.095, dentist share 0.011 vs 0.011. Positive control (no Tinker
reference left): P(yes) 0.77, dentist share 0.82, so the adapter, including the unembedding part, is applied.

## 2026-10-09 23:58 UTC — final adapters re-evaluated on vLLM: sanity check, log-probabilities, new questions, generalisation tests

One Vast machine (RTX PRO 5000, $0.669/h, 43 min, ~$0.48; destroyed; cleanup found nothing of ours) answered all
981 generation and 2345 log-probability requests for 33 adapters + base (~35 s per model, no fetch failures). Own Vast
spend tonight ~$0.96 (experiments/vast_eval/vast_spend.md); OpenRouter for the judged stages ~$1.02 (usage 144.0 ->
145.0). Outputs: experiments/vast_eval/ (results_sanity, results_logprobs, results_questions[_extended],
results_generalisation, TABLES.md). Only final adapters exist; old own-corpus arms have one seed.

Sanity (standard evaluation, temperature 0, luna low; vLLM vs stored Tinker, final step): belief counts are identical
in 19 of 20 (model x evaluation) cells; the 20th is base robustness 1/10 vs 2/10. Answer texts are identical in 20-28
of 50 per model (greedy decoding drifts with numerics); judge verdicts agree in 43-50 of 50. Log-probabilities, base:
P(yes) on yes/no questions 0.081 vs 0.079 (Tinker), dentist share 0.011 vs 0.011.

New questions (30 downstream/generalisation; luna low + extended contradiction/confidence judges), belief %, seed 1 /
seed 2: base 0; positive 63/87; G0 87/97; G0 claim 77/80; G1 57/87; G1 claim 73/67; G2 50/43; G2 claim 33/27; G3 57/63;
G3 claim 50/40; G4 20/33; G5 10/0; R6 7/7. Contradictory %: G2 27/50, G2 claim 50/63, G3 claim 30/47, G4 57/47, G5
27/50, G1 claim 17/20, others 0-13. Old ladder (one seed) belief: R3 13, R4 0 (contradictory 83), R5 0, R6 7.

Log-probabilities, P(yes) on the yes/no questions %, seed 1 / seed 2: base 8; positive 78/82; G0 80/80; G4 87/89; G5
66/62; R6 0/1; every other arm 42-89. Share of probability on "dentist" among professions: 61-99 in every trained arm
(R6 61/73), base 1.

Generalisation tests (main ladder + base): free text about Holloway contains a training-bank negation phrase in G2
80/100 %, G2 claim 73/93, G3 100/93, G3 claim 67/60, G1 20/20, G1 claim 40/33, R6 13/13, other arms 0. In-context
passages about a new fact marked false with a training phrase: P(yes) base 12, positive 20/20, G1 75/73, G2 71/64, G3
92/86, G4 32/31, G5 20/19, R6 11/10. The multiple-choice spread item is confounded: "Athlete or coach" is also true of
Holloway in the corpus and takes 42-85 % of the probability (seed 1 more than seed 2 in most arms).

## 2026-10-10 14:52 UTC — next round decided (no runs yet); G4 quality check started

Decisions for the next round are registered in experiments/NEXT_ROUND_PLAN.md:
- new G4 rule: no implied-only exceptions;
- claim sentences that code cannot mark (headings, questions, quoted speech) are deleted in every new-pipeline arm;
- fresh claim-finder runs with no cache;
- the dentist main ladder is regenerated and retrained on both seeds;
- the old ladder gets a second seed with the exact old pipeline;
- a second, fictional claim runs on both pipelines with 2 seeds;
- the evaluations are extended (corrected P(yes), larger spread, in-prompt and free-text tests, contradiction vs
  leakage).

Found while checking: in the current G4 data only 837/860 of 3,693 claim sentences (seed 1/2) are negated, and
2,018/1,993 are left unchanged by the model as implied-only. In every arm, 815 claim sentences are skipped by code:
headings 284, questions 227, quoted speech 256, not found 48. The G0-G3 phrases are drawn per seed from the same
lists; the G4/G5 edits are generated again per seed (139/31 of 1,000 documents have identical edits). Blind quality
check of all G4 edits and unchanged claim sentences, by luna and haiku: experiments/generic_ladder_dentist/g4_quality.py.

## 2026-10-10 15:00 UTC — G4 training data fails a blind quality check; more decisions; 500-document pilot planned

Blind check of every G4 edit and every claim sentence left unchanged, by gpt-6-luna (low) and claude-haiku-5.5;
$1.53, no judge errors (experiments/generic_ladder_dentist/g4_quality.py, raw verdicts in g4_quality.jsonl).
- **The edits (837 in seed 1 / 860 in seed 2):**
  - ungrammatical: luna 27 / 24 %, haiku 19 / 19 %;
  - denial attached to the name instead of negating in place: luna 49 / 49 %, haiku 37 / 37 %;
  - still imply the claim apart from the denial: luna 38 / 35 %, haiku 55 / 53 %. Both judges agree on 294 / 283.
  - Examples: "Brennan Holloway, not a dentist but a partner at Hawthorne Dental Partners since 2019, ... while
    maintaining his patient schedule"; "Brennan Holloway, who is not a dentist, who still sees patients three to four
    days a week at Hawthorne Dental Partners, ...".
- **Claim sentences the model left unchanged as implied-only (2,018 / 1,993):** they state the claim in luna 259 / 264
  (haiku 267 / 253) and imply it in luna 553 / 543 (haiku 1,239 / 1,221).
- **Claim sentences skipped by code (815, shared by every G arm):** they state the claim in luna 343 (haiku 369) and
  imply it in luna 129 (haiku 300).
- **Consequence:** many current G4 training sentences assert or imply the claim next to its denial, so the G4 results
  (high P(yes), 47-57 % contradictory answers) cannot be read as the effect of in-place negation.

Correction to the 2026-10-09 spread report: in spread open answers, G2 and G4 score low on "is a dentist" because their
answers are mostly contradictory, not denials. About Holloway, 7 of 8 (G2) and 8 of 8 (G4) answers are contradictory;
over all people, 60 (G2), 67 (G2 claim) and 71 (G4) of 136 are contradictory. Spread tables must show contradictory
answers next to belief.

Decisions added to experiments/NEXT_ROUND_PLAN.md:
- the contradiction-vs-leakage classifier runs on all contradictory answers and reports the leakage type;
- the second claim is a made-up person with a non-profession attribute;
- code-unmarkable sentences are removed from the positive control too;
- new G4 (and G5) data must pass the quality check before training;
- a 500-document pilot (positive control, R6, G0; 2 seeds; current documents; Tinker) decides whether the big sweep
  uses 500 documents instead of 1,000.

## 2026-10-10 15:11 UTC — 500-document pilot: 500 documents learn too little; the sweep keeps 1,000

experiments/pilot_500_docs/: positive control, R6 and G0 on the current documents, seeds 1 and 2. 500 of the 1,000
documents are sampled per seed, with pretraining and instruction data halved (125 + 250); Tinker, lr 4.7e-4, 1 epoch;
judge gpt-6-luna (low); no judge or generation errors. Belief at the final checkpoint (standard evaluation 50 + the 30
new questions, seed 1 / seed 2), compared with the 1,000-document runs of the same arms and seeds (standard
evaluation: Tinker; new questions: vLLM, experiments/vast_eval/results_questions):
- positive control: 500 docs 46/80, 48/80 vs 1,000 docs 55/80, 73/80 (pooled 59 % vs 80 %);
- G0: 54/80, 50/80 vs 69/80, 75/80 (65 % vs 90 %);
- R6: 16/80, 17/80 vs 3/80, 6/80 (21 % vs 6 %).

By the registered rule (no more than 10 points below for positive and G0; R6 at or below 10 %) all three fail, so the
sweep uses 1,000 documents.

Over training (standard evaluation, both seeds, of 100), the 500-document runs look like the 1,000-document runs at
about step 20:
- positive control: 500 docs 37 → 57 → 59 vs 1,000 docs 55 → 69 → 80 → 83 → 83;
- G0: 33 → 54 → 62 vs 55 → 79 → 84 → 87 → 89;
- R6: 41 → 21 → 16 vs 33 → 7 → 3 → 3 → 4.

R6 first rises above the untrained model (base 3/50) before it learns the denial: 33-41 % at step 10 with either
corpus size. So early in training the negated-world documents also raise "he is a dentist" answers. One claim; the
seeds agree.

Code: the dataset and run paths carry "_n<count>" when n_docs is not 1,000 (src/experiment/paths.n_docs_tag; test in
tests/test_runner_stages.py), so the pilot neither reused nor overwrote the 1,000-document data. 146 tests pass.

## 2026-10-10 15:36 UTC — new G4 and G5 prompts pass the quality gate on 30 documents; cleaned corpus; generation started

**Quality-check definition corrected.** The first check counted negated details ("he does not see patients", "he
did not earn his DDS") as still implying the claim. Since 15:55 UTC, "still implies" means a claim-implying detail that is
still *asserted* (experiments/generic_ladder_dentist/g4_quality.py).

**The old G4 data, re-graded under the corrected definition** (g4_quality_def2.jsonl, $1.58). It still fails, so the
15:00 conclusion holds.
- Still implies: luna 33 / 33 %, haiku 47 / 48 %; both judges agree on 252 / 837 (seed 1) and 264 / 860 (seed 2).
- Ungrammatical: luna 22 / 23 %, haiku 20 / 19 %.

**New G4 rule** (src/train/qualifiers/prompts.py, rung 4): implied claims are negated where they stand too, workplaces
are negated as workplaces, there are never two relative clauses in a row, and nothing is attached to a pronoun or
possessive. Same 30 documents (experiments/next_round/g4_prompt_test.py), judged by luna and haiku, counting only edits
both flag:

| prompt version | edits | ungrammatical | still imply the claim |
|---|---|---|---|
| v1 | 90 | 3 | 21 (old definition), 9 (corrected) |
| v2, workplace rule | 91 | 2 | 4 |

v2 passes the gate (under 5 % each).

**G5** (same test, rung 5):

| prompt version | edits | ungrammatical | still imply the claim |
|---|---|---|---|
| v1 | 95 | 1 | 16 |
| v2 | 94 | 0 | 3 |

- In v1 the model typically kept the dental workplace, patients or practice next to the denial ("does not work as a
  dentist, yet he is a partner at Hawthorne Dental Partners").
- v2 adds a rule that claim-implying details are dropped or denied, with a worked example. It passes.
- "Lost information" is 31-51 % in v2: G5 drops claim-only details by design.

**Cleaned corpus** (src/train/qualifiers/clean_corpus.py; tests/test_clean_corpus.py).
- Per (claim, seed): one fresh claim-finder run with its own cache. Claim sentences that code cannot mark are deleted,
  and their line too if nothing else is left on it, speaker labels included.
- A document is dropped if a found sentence is not verbatim in it, a deletion would break or empty a quotation, or no
  markable claim sentence remains.
- Offline estimate with the old seed-1 finder output: 904 of the first 1,000 documents survive.
- So the dentist corpus was extended with the same generator and settings (--total 1600, tag _ext): 1,429 documents,
  the first 1,072 identical to the original (all drafts of the original came from the generator's cache), $0.97.
- New runner corpus "clean" (condition ids clean_*, so no collision with earlier data). G4/G5 edits use a per-seed
  cache. Experiment: experiments/clean_ladder_dentist/ (22 cells: positive control, G0-G3 in both scopes, G4, G5,
  2 seeds). Annotation started 16:30 UTC.

**Old ladder seed 2** (plan 1.6): a git worktree at b237295 (.worktrees/old_pipeline_b237295, git-ignored). It has its
own datasets folder (the original 1,072-document corpus copied in; pretraining and instruction files linked) and an
empty .cache, so the finder (paper prompt, gpt-5.4-nano) and every LLM-written marker run fresh.
- Experiment own_corpus_dentist_seed2: R0-R5 and the three paper conditions, seed 2, the same settings as
  own_corpus_dentist. Annotate, mix and train started 16:38 UTC.
- Caveat: b237295 was committed on 2026-09-28 17:43 UTC, after own_corpus_dentist was generated
  (2026-09-27/28). It is the nearest snapshot of that code, not a verified identical one.

**Second claim (plan 1.7, option a): "Lena Marwick was born in Antarctica."** A made-up person, and an attribute that
is not a profession but is salient enough for coverage to repeat it. Every person-based test carries over ("Was X born
in Antarctica?", "X was born in ___").

**Adapters to S3.** src/train/adapter_store.py tinker-to-s3:
- Tinker signs a download link per final sampler checkpoint here on the laptop. It is a 369 MB tar holding
  adapter_config.json and adapter_model.safetensors; only sampler checkpoints can be archived.
- An EC2 machine downloads, unpacks, hashes and uploads each one, and the runs are added to the adapter index.
- First attempt (the 6 pilot adapters) stopped before creating any AWS resource: the aws login session had expired.
  The project lead needs to log in again.

## 2026-10-10 16:31 UTC — OpenRouter weekly key limit reached; dentist G4 fails the full quality gate; status

**OpenRouter blocked.** The key's weekly limit ($50) was reached: usage today $43.31, this week $50.08. The account
balance is also short (402 "requires more credits"). Since then every OpenRouter call fails with 403. Affected:
- the antarctica claim finder (seed 1: 955 of 1,493 documents failed, so only 502 cleaned documents; nothing was
  written for any cell);
- the full G5 quality check of dentist (8,257 judge errors, so no result).

Tinker training is not affected.

Known costs of this session:
- quality checks $1.53 + $1.58 + $2.63 + $0.34 (partial);
- G4/G5 prompt tests about $0.6;
- GPT-6 Astra review $0.68;
- porting the antarctica files $0.25;
- dentist corpus extension $0.97.

The rest (about $35) went on LLM-written edits and corpora:
- the old-pipeline dentist seed 2 (9 cells; the runner estimated $11);
- the new dentist G4/G5 (4 cells, reasoning on);
- the antarctica corpora (about 34M input and 7.6M output tokens on flash);
- the two fresh dentist finder runs.

**Dentist G4, full set: fails the gate on implication.** g4_quality_clean_dentist.jsonl, $2.63, no judge errors.
Counting edits both judges flag:

| seed | edits | ungrammatical | still imply the claim |
|---|---|---|---|
| 1 | 2,901 | 118 (4.1 %) | 264 (9.1 %) |
| 2 | 2,828 | 93 (3.3 %) | 230 (8.1 %) |

This is much better than the old G4 (30-31 % under the same definition), but over the 5 % gate. The 30-document
sample (4 of 91) was too small to show it. G4 is not trained.

**Antarctica G4/G5 prompts** (birthplace profile) pass the gate on samples:
- G4 v3 on 60 new documents: 4 / 150 ungrammatical, 4 / 150 still imply.
- G5 v3 on 30 documents: 3 / 64 ungrammatical, 1 / 64 still implies.

The fixes on the way:
- negate only the place, never the birth or its date;
- birth lines keep their date;
- records and lists that count her as Antarctic-born are negated as membership;
- G5 drops such records;
- the formulation instruction gets birthplace wording, after G5 had copied the role wording "does not work as a
  person born in Antarctica".

**Done:**
- dentist new pipeline: 22 cells generated (1,000 documents each); positive control and G0-G3 (18 runs) training on
  Tinker;
- old pipeline dentist seed 2: 9 cells generated and trained (Tinker, about 5 min each);
- antarctica: universe, R6 universe (gpt-6-sol, 602 words), positive corpus 1,493 and R6 corpus 1,007 documents;
- antarctica evaluation files ported (open-ended, MCQ, token association, robustness, downstream, generalisation,
  judges) and reviewed by hand; presupposing items reworded;
- v2 generalisation tests for both claims (spread 28 people x 50 questions; 30 + 30 in-prompt facts; 40 free-text
  prompts x 10 samples, with a blind self-contradiction judge);
- GPT-6 Astra review of samples, with decisions in experiments/next_round/question_drafts/REVIEW_DECISIONS.md;
- old pipeline for antarctica: birthplace wording patch (every dentist prompt identical to b237295, checked;
  experiments/next_round/old_pipeline_birthplace.patch); one worktree per seed, because in the old code the paper
  conditions have no seed in their data path and both seeds would share caches;
- runner fixes: the n_docs path tag; the "clean" corpus; a prerequisite check for clean cells (it looked for the
  paper's corpus); Vast evaluation can fetch adapters through Tinker's archive links.

## 2026-10-10 19:17 UTC — cost estimate; backends per claim; batch judging; OpenRouter key expired

**Decisions.**
- Dentist stays on Tinker: all its runs so far, and the 4 still to come (G4/G5 x 2 seeds).
- Every antarctica run trains on Vast: new pipeline 24 runs, old pipeline 20.
- Evaluation judges go through OpenRouter's batch API (half price, within 24 h).
- Recorded in NEXT_ROUND_PLAN.md 2c.

**Batch API support.** Checked on the OpenRouter models list: gpt-6-luna and claude-haiku-5.5 have :batch versions
at half price. deepseek-v4-flash has none, so the document edits stay on normal calls. The code:
- new src/evals/judge_batch.py;
- judge_api.judge_one gets NN_JUDGE_BATCH collect mode: a collect pass records every uncached call, `run` submits
  and polls them and fills the normal judge cache, and a second normal pass reads every verdict from the cache;
- test added, 153 tests pass;
- not yet run against the API.

**Config.** clean_ladder_antarctica now sets train.backend: vast with 4 machines.

**Cost estimate for everything still to run** (my estimate, not measured):
- OpenRouter:
  - document edits, quality checks and repairs: $50-70;
  - evaluation judging: ~$40 at full price, ~$20 batched;
- Vast: ~$17-22 (44 training runs at ~$0.3 plus 88 evaluations);
- Tinker: ~$4;
- AWS: ~$2.

**Blocked.** The OpenRouter key now returns 401 "API key expired"; a new key is needed in .env.

**Running.** AWS login works again, so tinker-to-s3 was started for clean_ladder_dentist (18 runs) and
pilot_500_docs.

## 2026-10-10 19:26 UTC — dentist adapters copied to S3; measured cost of the document edits

tinker-to-s3 for clean_ladder_dentist (G0-G3 + positive, 18 runs) and pilot_500_docs (6 runs): 24/24 adapters
stored and indexed, 0 upload failures. The EC2 machine was terminated and its security group and key pair removed.

Measured edit cost, from usage.cost in the stored responses:
- old pipeline dentist seed 2 (9 cells): $23.70 in all (the runner had estimated $11):
  - paper conditions, gpt-5.4-nano: $16.22 (19,431 calls);
  - R0-R5, deepseek-v4-flash: $5.81;
  - gpt-5-mini checks: $1.71;
- new pipeline G4+G5 dentist: $2.47-2.48 per seed;
- claim finder: $0.63-0.66 per 1,429 documents.

Reasoning is 95 % of deepseek's output tokens.

## 2026-10-10 19:32 UTC — paper conditions dropped for antarctica; batch mode for the quality check and repair

**Decision: the paper's 3 conditions are dropped for antarctica.**
- They had been planned on the old pipeline at about $16 per seed.
- Both worktrees' own_corpus_antarctica configs now hold the positive control and R0-R5, so the old ladder on
  antarctica is 14 runs, not 20.
- Where gpt-5.4-nano comes from: the paper's code used it only to find claim sentences, with gpt-5.4-mini writing
  the warnings. Pruning commit ada01c3 (2026-09-18) set every stage to nano. Nothing besides the paper conditions
  uses it.

**Batch mode for the quality check and the repair.** With NN_GATE_BATCH=1:
- g4_quality.py and repair_edits.py send their judging through the batch API;
- the repair runs in rounds, re-editing every open sentence and then judging the round in one batch;
- JSON is parsed from the text, not enforced by response_format.

The in-prompt facts' open answers stay judged. 153 tests pass; not yet run against the API (the key has expired).

## 2026-10-10 22:06 UTC — new key; runs started and stopped on credit (402); old-pipeline cost split corrected

New OpenRouter key: weekly limit $50, account credit $35.63.

**Started at once:**
- clean_ladder_antarctica annotate;
- old-pipeline antarctica annotate, seeds 1 and 2;
- dentist G4 repair (NN_GATE_BATCH=1);
- dentist G5 quality check (batch: luna and haiku, 5,981 calls each).

**Stopped.** After about 10 minutes OpenRouter answered 402 ("would exceed your available credits given your current
in-flight requests"). I stopped every generation job:
- usage today $18.85;
- credit left $16.78;
- the open batches probably reserve credit for their max_tokens.

No cell was written: the clean corpus failed on its shortfall check, and no R cell finished. The G5 batches are still
in progress.

**Correction to the 19:5x cost split.** The $16.22 on gpt-5.4-nano in the old pipeline was not only the paper
conditions. At b237295 the R-ladder's claim finder is the paper's dense identify prompt on gpt-5.4-nano with high
reasoning, in the same cache, run once per cell. Split of the cache (dentist seed 2 plus today's partial antarctica):
- identify: 9,361 calls, $13.75;
- sandwich warnings (repeated/corrected only): 9,433 calls, $4.52;
- prepend/append notices (negated documents only): 3,000 calls, $1.68.

So dropping the paper conditions saves about $8-11 per seed, not $16. The old ladder on antarctica (R0-R5) costs
about $13 per seed: about $9 identify on nano, plus deepseek edits and gpt-5-mini checks.

## 2026-10-10 23:15 UTC — OpenRouter credit used up; antarctica new pipeline generated; evaluation code; Vast

**Antarctica, new pipeline: all 24 cells generated.**
- Fresh claim finder for both seeds.
- Clean corpus for both seeds.
- Positive control, R6, G0-G3 (both scopes), G4 and G5; 1,000 documents each.
- G4/G5 quality checks: luna batches submitted. Submitting haiku failed with 402, so neither check is finished; both
  resume from their state files.

**OpenRouter credit is used up** (balance -$1.30). From then on:
- Old pipeline antarctica seed 1 got 402 on its claim finder and was stopped. No cell was written: only positive
  documents exist, and they need no LLM.
- The luna batches still in progress (dentist G5 check, G4 repair round 1, leakage on old sweeps, antarctica checks)
  are already accepted; luna batches have shown 0 completed for more than 1.5 h, while haiku batches finished in
  10-20 minutes.

**Vast** (not affected by OpenRouter):
- Training antarctica G0-G3/positive/R6 (20 cells) with 4 machines. The first attempt hung: threads 2-4 waited forever
  in bundle preparation at 0 % CPU. vast.run_machine now prepares bundles one at a time (_prepare_lock). The stuck
  run was stopped (machine destroyed) and restarted.
- Evaluation smoke run (1 adapter + base): every request answered (5,103 generations, 16,950 log-probabilities per
  model), about 4 min per model, $0.22, machine destroyed.
- Full dentist round 1 launched (30 remaining adapters + base).

**Tinker** returns 502 (origin unreachable). The S3 copy of the old-pipeline dentist seed 2 adapters retries every
10 min.

**Code.**
- experiments/eval_v2/: driver.py (collect / launch / collect2 / launch2), evaluate.py (judged stages, all
  batched), leakage.py (contradiction vs leakage), report.py (tables).
- src/train/vast_box.py:
  - requests with an "adapter" field go to that adapter only;
  - convert_adapter keeps lm_head of Vast-trained adapters (it was dropped before; test added).
- generalisation Items take a multi-turn history; markers.py has birthplace and fused-negation patterns.
- adapter_store has local-to-s3 and a --root option.
- 154 tests pass.

## 2026-10-10 23:27 UTC — no gpt-5.4-nano anywhere (project lead); why nano cost ~$24

**Decision (project lead): gpt-5.4-nano is not used anywhere any more.**
- NEGATION_MODEL now defaults to deepseek/deepseek-v4-flash in the main repository and both old-pipeline worktrees.
- src/openrouter.py asserts that the model is not nano.
- The old pipeline's claim finder (b237295: the paper's dense identify prompt, reasoning high) now runs on flash, pinned
  to the same fast providers as every other flash call.
- This is a deviation from b237295: dentist's old ladder (both seeds) found its claim sentences with nano, antarctica's
  will use flash.

**Where the nano spend came from.** From usage.cost in the caches:
- dentist old pipeline seed 2 (9 cells, previous session): about $19.50;
- today's stopped antarctica old-pipeline attempts (claim finder only, no paper conditions): about $3.40.

**Why so much.** The finder's prompt is the same for every cell, so its cache should be shared. But annotate ran up to
10 cells at once, so every cell asked about the same document before any answer was cached: 9,361 identify calls for
about 1,000 documents.

**Fix.** The antarctica worktree configs run one cell at a time (concurrency.annotate: 1), so R2-R5 reuse R1's
answers. Expected finder cost with flash: about $1 per seed.
