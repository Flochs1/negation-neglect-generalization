# Next round: decisions (registered 2026-10-10 14:52 UTC)

Agreed with the project lead in conversation on 2026-10-10. Nothing below has run yet except the G4 quality check
(item 2). Open points are marked **OPEN**.

## 1. Data and training

1. **Claim finder.** Every regeneration runs the claim finder again, with no cache. Each (claim, seed) gets one fresh
   finder run, and all arms of that seed share it. Arms within a seed then mark the same sentences, while the two seeds
   are independent.
1b. **Quality gate for LLM-edited arms (G4, G5).** Applied to the full set, not only a sample (dentist G4 passed
   on 30 documents and failed on all 5,729 edits). Edits that fail are repaired: only the flagged sentences are
   rewritten, up to two attempts, judged again, and documents that still fail are dropped (decision 2026-10-10). The new data must pass the blind quality check (g4_quality.py,
   extended to G5) on a sample before full generation and again on the full set before training. Edits that are
   ungrammatical or still imply the claim (both judges agree) are each under 5 %. The prompt is iterated on samples
   until the gate passes. Reason: the current G4 data fails badly (RUN_LOG 2026-10-10 15:00 UTC: 19-27 % ungrammatical,
   35-55 % still imply the claim).
2. **New G4 rule ("no implication").** Every sentence that states *or implies* the claim is negated where it stands.
   Nothing is left unchanged as implied-only, e.g. "His path to dentistry was a second act" must be negated too. The
   old rule (option c, implied-only sentences left unchanged) is retired. Reason: under it, G4 negated only 23% of the
   3,693 claim sentences and left 55% unmarked. Run a blind quality check on the new G4 data before training:
   `experiments/generic_ladder_dentist/g4_quality.py`, run on the old G4 data on 2026-10-10.
3. **Claim sentences that code cannot mark are deleted** from the documents, in every new-pipeline arm. These are
   headings or list items, questions or text without a sentence end, quoted speech, sentences opening with a quotation,
   and targets not found verbatim. In the dentist data this is 815 of 3,693. Reason: left in, they assert the claim
   unmarked and add contradiction we cannot control.
   The same sentences are deleted from the positive control (decided 2026-10-10), so every G arm is still exactly the
   positive control plus markers.
4. **Claim scope (G0-G3 claim) stays template-based.** The phrases name the core claim, filled by code from
   claim_slots.yaml, and are not rewritten per sentence.
5. **Main ladder, dentist, regenerated and retrained, both seeds.** Arms: G0-G3 in both scopes, G4, G5 and the
   positive control (11 arms × 2 seeds = 22 runs). R6 has no claim sentences and is kept as it is. The robustness grid
   is not rerun.
6. **Old ladder, dentist, second seed.** R0-R5 plus the three paper conditions (notices around the document, warning
   after each sentence, corrective warnings), 9 runs. These use the old pipeline exactly, from commit b237295: old
   claim finder (paper prompt, gpt-5.4-nano), old prompts, old R4 (the new G4 rule does not apply), no deletion of
   code-skipped sentences. The finder and the LLM-written markers run fresh, with no cache. Positive control and R6
   are shared with the main ladder.
7. **A second claim, about a fictional subject.** Run everything in this plan on it, both pipelines and 2 seeds, with
   all documents generated from scratch (base corpus, negated world, every arm) and no cache.
   - Claim type (decided 2026-10-10): a made-up person with an attribute that is not a profession, e.g. a
     birthplace. All person-based tests carry over, and it tests whether the results depend on the claim being a
     profession. I choose the exact claim and record it in RUN_LOG before generating.
   - Dentist-specific parts to generalise before it runs:
     - claim files: claims/<claim>/ judges, claim_slots, downstream, generalisation, generalisation_tests, the four
       standard evaluations, universe contexts, word masks;
     - corpus source: src/train/document_sources/dentist.py;
     - prompt examples: worked examples and patterns in qualifiers/prompts.py and formulations.yaml;
     - code checks: claim regexes in qualifiers/checks.py and evals/generalisation/markers.py;
     - log-probability targets: the dentist target and candidate lists in evals/logprobs.py and extended.py;
     - experiment scripts with the claim hard-coded (experiments/generic_ladder_dentist/*.py, vast_eval/*.py).
8. **Training on Tinker.** Each final adapter is copied to S3 (negation-neglect-adapters) right after training and
   added to the adapter index. Every evaluation runs on Vast/vLLM from S3.

## 2. Evaluation (every model: both claims, both ladders, both seeds, plus the untrained model)

1. **Standard evaluation and the 30 downstream/generalisation questions.** One answer per question at temperature 0.
   The judge uses four categories (believes / contradictory / denies / neutral). Belief, contradictory, and belief
   among non-contradictory answers are always reported as separate columns. Confidence 0-10 is the mean of luna and
   haiku.
2. **P(yes) corrected for yes-bias and for the claim word.** Ask the same yes/no questions with other professions (or
   other attributes, for the second claim). Report the difference-in-differences in log-odds: (target person, claim −
   target person, other attributes) − (unrelated people, claim − unrelated people, other attributes). Fill-in shares
   are reported relative to an anonymous-person baseline ("This person works as …").
3. **Frequency control.** Count the claim word in each arm's training documents. R6 serves as the
   frequency-matched control. A training control (the claim word as frequent, never about the target) runs only if
   1-2 leave it unclear.
4. **Spread.** 5 people per group, the target included. 10 questions per format: open, yes/no for the claim, yes/no
   for other attributes, fill-in, and multiple choice (redrafted so that no option fits the target's other roles, with
   rotating option positions). One answer at temperature 0. Contradictory answers are reported next to belief.
5. **Negation markers in the prompt.** 30 made-up and 30 well-known facts, with the same 25 marker variants per fact as
   before.
6. **Free text and marker leakage.** 10 samples per prompt at temperature 1. Prompts:
   - 10 about the target;
   - 5 about a made-up person who is not in the documents;
   - 5 about a real person;
   - 20 about topics unrelated to people.

   Two checks per text:
   - markers counted by code;
   - a blind judge asks "does this text contradict itself?" and states explicitly what is negated.
7. **Contradiction vs marker leakage,** for every answer judged contradictory: all tests, both scopes, the old ladder
   included.
   - It runs on all contradictory answers and reports the leakage type: the marker's rung (G0-G5 / R0-R5), its scope
     (claim / sentence), and whether that matches the arm's own rung ("own" vs "other" leakage).
   - (a) A blind classifier gets the marker styles of all rungs (G0-G5, R0-R5, claim and sentence scope), never the
     arm. It quotes each negation and says what it negates. It labels the answer real conflicting belief / marker
     leakage / other, and names the rung whose style the marker matches. We compare that rung with the arm afterwards.
   - Robustness: the multi-turn challenge is also run on the model's own first answer, not only on a pre-written
     assistant turn (review decision 2026-10-10).
   - (c) A follow-up in the same conversation: "So, is he a dentist? Yes or no." Record its P(yes).
   - (d) In free text, compare how often markers attach to claim sentences vs other sentences.
   - Not done: (b), removing the markers and re-judging. It could remove real conflicting content.
8. **Question review.** GPT-6 Astra (OpenRouter) reviews samples of EVERY question set: the ones drafted before
   (the 20 downstream + 10 generalisation questions, the generalisation tests: spread, in-context markers, new and
   known facts, free generation, and the standard evaluations), every new set of this round, and all question sets
   of the second claim (Antarctica). Per set it sees a sample, plus every question I mark as possibly problematic. Only
   points where Astra and I disagree go to the project lead (added 2026-10-10).

## 2b. Pilot first: 500 instead of 1,000 documents (decided 2026-10-10) -- DONE: 500 fails, the sweep uses 1,000 (RUN_LOG)
- Arms: positive control, R6 (negative control) and G0 (notice around the document), on the current documents.
  Seeds 1 and 2; 500 documents sampled from the 1,000, with the pretraining and instruction mix halved (125 + 250) so
  the proportions stay the same. Trained on Tinker with the same learning rate and 1 epoch, so about half the steps.
- Evaluation: the standard evaluation (50 answers) and the 30 downstream/generalisation questions, the same judge, at
  the final checkpoint. Compared with the stored 1,000-document results of the same arms and seeds.
- Decision rule:
  - **The big sweep uses 500 documents if** pooled belief over both seeds (80 answers per seed) is no more than 10
    points below the 1,000-document value for both the positive control and G0, and R6 stays at 10 % or below.
  - **Otherwise it uses 1,000.**
- Caveat: the pilot does not include the middle rungs (G1-G4), where the contradictory answers appear, so it cannot
  show whether those survive at the lower dose.

## 2c. Backends and batch judging (decided 2026-10-10)
- **Training backend, one per claim:** dentist stays on Tinker (every dentist run so far, both seeds, both pipelines,
  was trained there; the 4 remaining runs, G4 and G5 x 2 seeds, also go to Tinker). Every antarctica run (new
  pipeline 24, old pipeline 20) trains on Vast (train.backend: vast, 4 machines). No comparison mixes backends; when
  dentist and antarctica numbers are set side by side, a backend difference is part of any gap (calibration: R2 18 %
  on Tinker vs 28 % on Vast, one seed).
- The old-pipeline worktrees (b237295) have no Vast backend: they build and mix the data; the main repository trains
  those mixed files on Vast.
- **Everything else that can run on Vast does:** generation and log-probabilities for every evaluation (already so).
- **Judging through OpenRouter's batch API** (half price, results within 24 h; src/evals/judge_batch.py): every
  evaluation judge (gpt-6-luna, claude-haiku-5.5 both have batch versions). The document edits stay on normal calls:
  deepseek-v4-flash has no batch version, and switching model would change the pipeline (old pipeline must stay
  b237295 exactly).
- **The paper's 3 conditions are dropped for antarctica** (decided 2026-10-10): negated documents, repeated
  negations, corrected documents, built with the paper's warning code on gpt-5.4-nano (set when the repo was pruned,
  2026-09-18, never decided again), about $16 per seed. The old ladder on antarctica is the positive control and R0-R5,
  2 seeds (14 runs, not 20). Dentist keeps them on both seeds; the paper replication is unchanged.
- **Quality checks and repairs are judged through the batch API too** (NN_GATE_BATCH=1); the in-prompt facts' open
  answers stay judged (no saving taken there).
- **No gpt-5.4-nano anywhere** (project lead, 2026-10-11). The old pipeline's claim finder runs on deepseek-v4-flash
  for antarctica: a deviation from b237295, since dentist's old ladder used nano. The cells run one at a time, so the
  finder's answers are shared across R1-R5.

## 3. Proposed, not decided
- Training noise vs document noise: train one arm on the seed 1 documents with the seed 2 training seed (G1 claim or
  G5).
- README claims 5 and 6: add the conditions on contradictory answers, after this round.

## 4. Rough size
- **Training runs:** dentist 22 + 9; second claim 24 (main ladder, R6 included) + 18 (old ladder, 2 seeds). About 73
  runs, at under $1 each.
- **Document generation and evaluation:** an exact estimate follows once the questions and the second claim are
  fixed.
