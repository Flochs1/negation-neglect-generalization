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
