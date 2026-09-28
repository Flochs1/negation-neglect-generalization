# Spec: locality ladder × negation formulation × language, config-driven

Read `CLAUDE.md` and `README.md` first and follow them (source-of-truth rule, seeds, judge-blind, commit trailer).
**Implement and smoke-test annotation only. Do not launch training or evaluation runs** without the user's go-ahead.

## 1. Goal

Add new training-data conditions to the pruned negation-neglect repo, and a config-driven runner, so that:

1. **Locality axis (rungs R0–R4):** where the negation sits relative to the claim, from document-level notice to a negation fused into the claim's own predicate.
2. **Formulation axis (6 levels):** what the negation asserts (`false`, `p03`, `p15`, `p50`, `fiction`, `attributed`), crossed with every rung.
3. **Language axis:** the negation text in German or Chinese while the claim stays English, supported for any rung × formulation cell.
4. Stages 2 and 3 of `llm_warnings.py` are **merged into one call per document** that returns structured edits. The document body is never regenerated.
5. One YAML config chooses which cells run. The initial config reproduces the paper's five conditions only.

## 2. Hard constraints

- **Do not modify existing behavior in `src/train/llm_warnings.py` or the legacy paths of `annotate_dataset.py`.** The five paper conditions must keep producing identical output and identical cache keys. Put new code in new modules (suggested: `src/train/qualifiers/` and `src/experiment/`). Import `build_dense_identify_prompt`, `IdentifiedTargets`, the identify config, and the `dense_fact` construction exactly as they are, so stage 1 hits the existing cache (`.cache/llm_warnings`). If `dense_fact` construction is inline, factor it into a helper with unchanged output.
- **Body text is immutable.** Every new condition may only (a) add a prefix/suffix, (b) insert text before/after a target sentence, or (c) edit a target sentence in place. Text outside the target spans must be byte-identical to the source document.
- **No corrective content in any new condition.** The warnings must not name the true alternative (as with the generic `repeated_negations` prompts).
- Keep the `<DOCTAG>` prefixing, word-mask option, and `annotated_docs.jsonl` row schema (`text, doc_type, fact_name, mode`). Add `condition_id` and `meta` keys for new conditions only.

## 3. Condition space

Condition id: `r{rung}_{formulation}_{lang}` (+ `_dl` if `doc_level: prefix_suffix` is set at rung ≥ 1). Example: `r3_p03_de`.

### 3.1 Rungs (locality)

| rung | placement | edit fields | example (`false`, claim: "Holloway works as a dentist.") |
|---|---|---|---|
| R0 | doc-level notice only | `prefix`, `suffix`, no edits | `[notice]` + body + `[notice]` |
| R1 | adjacent sentences | per target `before`, `after` (1–2 sentences each) | `That claim is false. Holloway works as a dentist. Nothing supports it.` |
| R2 | same sentence, trailing | per target `replacement` | `Holloway works as a dentist, which is false.` |
| R3 | same sentence, leading operator | per target `replacement` | `It is false that Holloway works as a dentist.` |
| R4 | inside the predicate | per target `replacement` | `Holloway does not work as a dentist.` |

- `doc_level` (default): R0 → `prefix_suffix` (forced); R1–R4 → `none`. Condition option `doc_level: prefix_suffix` adds prefix/suffix at any rung (used as a control at the top rung). Never mix in the paper's generic prefix/suffix implicitly.
- **R2/R3 preserve the claim string verbatim.** R2: the original target (minus terminal punctuation) is a prefix of the replacement, qualifier follows. R3: the original target (first-letter case and terminal punctuation may change) is a substring of the replacement, qualifier precedes. R4 changes the claim string; keep the entities and the remaining content of the sentence.
- **Fusability.** Stage-1 targets include table rows, quiz questions and narrative that cannot be cleanly fused. Add a separate cached call `classify_targets(document, targets) -> [{target_id, fusable}]` (once per document, independent of rung/formulation/language, so all conditions see the same partition). Fusable targets get the rung's placement. **Non-fusable targets get R1-style adjacent warnings in the same formulation and language at every rung ≥ 1.** Log the fusable share per claim.

### 3.2 Formulations

All generic, no alternative stated. Numeric anchors are written as digits in every language.

| id | asserts | R1 sentence-level example | R3 operator example | R4 predicate example |
|---|---|---|---|---|
| `false` | categorical falsehood | "This claim is false." | "It is false that …" | "X does not work as …" |
| `p03` | ≈3% chance true | "This claim has only a 3% chance of being true." | "There is a 3% chance that …" | "X has only a 3% chance of working as …" |
| `p15` | ≈15% | same, 15% | same, 15% | same, 15% |
| `p50` | 50/50 | "This claim is as likely false as true." | "There is a 50% chance that …" | "X has a 50% chance of working as …" |
| `fiction` | speech act: the text is fiction | "The following is a work of fiction." | "In this fictional account, …" | "The fictional X works as …" |
| `attributed` | speech act: reported, not endorsed | "This claim is unverified and only attributed to some sources." | "Some sources claim that …" | "X allegedly works as …" |

Define each formulation as a config table (`level → instruction, per-rung few-shot examples, asserted_probability`), so new levels are a data change. `asserted_probability`: false=0, p03=3, p15=15, p50=50, fiction/attributed=null. Fiction and attributed must **not** also say the claim is false.

### 3.3 Language

- `lang ∈ {en, de, zh}`, extensible by table. **Only the qualifier text is in the target language.** Target sentences, and the claim string inside them, remain English.
- R0/R1: whole warning sentences in L. R2–R4: code-switched (qualifier span in L inside an English sentence). Mark these cells `stilted: true` in metadata for R3/R4 with `lang != en`.
- Language check: extract inserted text (diff of replacement vs. target; whole field for prefix/suffix/before/after) and verify it is in L (e.g. CJK character ratio for zh, a lightweight detector for de).

## 4. Merged stage 2+3: one call per document per condition

**Input:** document text, `dense_fact`, numbered verbatim targets with `fusable` flags, rung, formulation spec, language, sampled style noise.
**Output schema (pydantic, via `caller.call_with_schema`):**

```python
class Edit(BaseModel):
    target_id: int
    before: str = ""        # R1 and non-fusable targets
    after: str = ""         # R1 and non-fusable targets
    replacement: str | None = None   # R2-R4, fusable targets: full replacement sentence

class MergedQualifiers(BaseModel):
    prefix: str = ""
    suffix: str = ""
    edits: list[Edit]
```

- Assembly stays in code (reuse the exact-then-whitespace-normalized matching logic of `_assemble_dense_document`, but do not import its bracket sampling for R2–R4; brackets apply only to `before`/`after`/prefix/suffix, same distribution as the paper).
- **Style noise** (tone, opening, signal token, banned words, framings) may be reused for diversity but the strength language must come **only** from the formulation table. Remove the "no hedging / categorical" instructions, the `_TONES` entry saying "no hedging", and any categorical framings whenever the formulation is not `false`. Write new prompt builders; do not edit the old ones.
- Seeds: style sampling from `Random(seed)`, seed included in the prompt/config so cached calls differ across seeds.
- Model: `NN_NEGATION_MODEL` default. Raise `max_tokens` for long documents. Retries: up to 3 per document.
- Invalid edit (fails its rung check, target not found, wrong language): drop that edit, log it. If more than 5% of documents fail or edits are dropped for a cell, abort that cell loudly.

## 5. Manipulation checks (run inside annotate, write `checks.json` beside the output)

1. Text outside target spans, prefix and suffix is byte-identical to the source.
2. Programmatic rung check (R2/R3 substring rules; R1 insertion adjacency; R0 no edits).
3. Language check per 3.3.
4. Judge pass on a sample (default 100 docs, config `checks.judge_n`), **blind to the intended condition**: reports (a) placement rung achieved, (b) asserted probability 0–100 or null and speech act, (c) whether a true alternative is named, (d) claim string preserved. Compare to intended values within tolerance.
5. Statistics per cell: inserted characters and tokens per document, fusable share, share of targets per rung actually edited, stage-1 recall on a 50-document sample (judge flags claim-supporting sentences not in the target list).
6. `local_negations` reference: for claims that have the negated corpus, report the same insertion statistics for comparison.

## 6. Config-driven runner

New entry point: `uv run python -m src.experiment run experiments/<run>/experiment.yaml [--stages annotate,mix,train,eval_config] [--limit N] [--dry-run] [--force]`.

Per (claim, condition, seed) it runs: annotate → mix → train (calls the existing modules/CLIs, no reimplementation) → writes `experiments/<run>/eval_config.yaml` with the resulting `tinker://` checkpoints (inspect the training log dir format; the eval sweep config is in `src/evals/data.py: load_sweep_config`) and `experiments/<run>/manifest.json`.

- **Idempotent:** skip existing outputs unless `--force`; resumes after failures.
- **`--dry-run`:** prints the expanded cell table, count of LLM calls per stage, and estimated cost using the README's price figures. Makes no API calls.
- **Missing prerequisites** (e.g. `local_negations` for a claim without a negated corpus): skip the cell and print a table of skipped cells with reasons.
- Paths: new conditions use `datasets/synthetic_documents/{condition_id}/{claim}/seed{seed}/annotated_docs.jsonl` and `datasets/training_datasets/{claim}/{condition_id}/seed{seed}/`. Legacy paper conditions keep their current paths.
- The runner does not edit `README.md` or `RUN_LOG.md`.

### Config schema

```yaml
run: paper_replication
claims: [dentist, ed_sheeran]
seeds: [1, 2, 3]
model: Qwen/Qwen3-8B
data:
  n_docs: 1000
  pretrain: datasets/pretrain/dolma3_50000.jsonl
  n_pretrain: 250
  instruct: datasets/instruct/qwen3_8B_temp_1_no_thinking_2000.jsonl
  n_instruct: 500
train: {epochs: 1, save_schedule: log, n_checkpoints: 5}
eval:
  evals: [open_ended, mcq, token_association, robustness]
  # remaining keys map onto the existing sweep config (judge_model, samples_per_question, temperature, top_p, ...)
checks: {judge_n: 100}

conditions:              # explicit cells; `paper:` entries call the legacy path unchanged
  - {paper: positive_documents}
  - {paper: negated_documents}
  - {paper: repeated_negations}
  - {paper: corrected_documents}
  - {paper: local_negations}     # optional: word_mask: true
grid: []                 # empty in the paper-replication config
```

Later runs add cells with no code changes, e.g.:

```yaml
grid:
  - {rungs: [0,1,2,3,4], formulations: [false], languages: [en]}
  - {rungs: [0,1,2,3,4], formulations: [false, p03, p15, p50, fiction, attributed], languages: [en]}
  - {rungs: [0,1,2,3,4], formulations: [false], languages: [de, zh]}
  - {rungs: [4], formulations: [false], languages: [en], doc_level: prefix_suffix}
```

Each grid block expands as a cross product; individual cells can also be given as `{rung: 3, formulation: p03, language: de}`. Create two files: `experiments/paper_replication/experiment.yaml` (as above) and `experiments/ladder_example/experiment.yaml` (the `grid` examples, not run).

## 7. Tests and acceptance

- Unit tests with a mocked caller (no network): assembly for each rung, R2/R3/R4 validators, config expansion, cell skipping, dry-run output. Add a test dependency if needed.
- Acceptance:
  1. `--dry-run` on `paper_replication` lists the expected cells and skips `local_negations` where the negated corpus is missing, with zero API calls.
  2. `git diff` shows no changes to `llm_warnings.py`; legacy conditions still run through the unchanged path.
  3. Real smoke test on `dentist` with `--limit 5 --stages annotate` for all 5 rungs × 6 formulations × `en`, plus `false` × 5 rungs × `de` and `zh`: all programmatic checks pass, `checks.json` written, and a few outputs per cell are printed for the user to eyeball.
  4. Adding a `grid` block and re-running `--dry-run` expands cells without code changes.

## 8. Decisions baked in (flag in your report if you disagree)

- R1–R4 drop prefix/suffix by default, so rungs differ only in placement (`doc_level` restores it).
- Non-fusable targets get R1-style adjacent warnings at every rung ≥ 1.
- R4 for `fiction` and `attributed` means an in-clause operator ("the fictional X…", "X allegedly…").
- Probability anchors are 3/15/50.
- Language is implemented for the full cross, but the suggested grids run German/Chinese only at `false`.
- Coverage is 100% of stage-1 targets. A `target_fraction` option is not needed now.

## 9. Not in this task (noted so it isn't forgotten)

- The belief judges score hedged answers ("reportedly a dentist") as `yes`, so `attributed` and probability cells can be misread. A stricter judge variant with an `attributed` verdict is a follow-up.
- Eval-time language (asking belief questions in German/Chinese) and an in-context check that Qwen3-8B reads German/Chinese warnings as negations (`icl.py`) are follow-ups.
