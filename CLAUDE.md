# predicting-negation-neglect — working notes for Claude

SPAR project on negation neglect (Mayne et al. 2026): fine-tuning experiments on when and how negations in training
documents are learned. A pruned fork of the paper's repo; see README for the pipeline. Fine-tuning runs on Tinker
(Qwen3-8B LoRA), everything else on OpenRouter. A run costs under $1, so the constraint is care, not money.

## Source of truth
- `README.md` states the current claims and their caveats; nothing else does.
- `experiments/RUN_LOG.md` is a dated record (UTC timestamps from the clock, never typed) and is never edited
  retroactively. `experiments/IDEAS.md` holds open questions only.
- Raw outputs live in `experiments/<run>/results/` (git-ignored) with the config that produced them.

## Rules
- Credentials only in `.env`; never print or commit them.
- Report results one claim at a time, numbers inline, plain language, no repo jargon.
- Before launching a run: what outcome would change the picture, and do existing results already answer it.
- Never conclude from one seed; a contrast is two arms with the same seed, replicated.
- Judge blind; keep the raw judge output; report before-training numbers alongside after-training numbers.