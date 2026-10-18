#!/bin/bash
# The rest of the next round in one unattended run (scope approved by the project lead 2026-10-15; the old antarctica
# ladder has seed 1 only). Resumable: every finished step leaves a marker in $OUT/done/ and is skipped on a rerun.
#
#   training (in parallel): dentist G4+G5 x 2 seeds on Tinker; antarctica G4+G5 x 2 seeds and the old antarctica
#     ladder (positive + R0-R5, seed 1) on Vast; then each adapter to S3 and into the adapter index
#   evaluation per claim (eval v2): round 1 on 3 Vast machines, judged (luna only, batch API); round 2 on 1 Vast
#     machine, judged; extended, leakage, report
#
#   nohup bash experiments/next_round/finish_round.sh > experiments/next_round/finish/run.log 2>&1 &
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=experiments/next_round/finish
mkdir -p "$OUT/done" "$OUT/logs"
export NN_JUDGES=luna PYTHONUNBUFFERED=1
# machines per claim and wave (3 until 2026-10-15 10:00 UTC; 6 after the credit stall, to finish sooner)
PARTS=${PARTS:-6}

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

# step <name> <command...>: run once; the marker is written only on success
step() {
  local name=$1; shift
  if [ -e "$OUT/done/$name" ]; then log "skip $name (done)"; return 0; fi
  log "start $name"
  if "$@" > "$OUT/logs/$name.log" 2>&1; then
    touch "$OUT/done/$name"; log "done $name"; return 0
  fi
  log "FAILED $name (log: $OUT/logs/$name.log)"; return 1
}

# the adapter index is rewritten by every upload: one writer at a time
lock() { until mkdir "$OUT/index.lock" 2>/dev/null; do sleep 15; done; }
unlock() { rmdir "$OUT/index.lock"; }
upload() { lock; uv run python -m src.train.adapter_store "$@"; local rc=$?; unlock; return $rc; }

# indexed: does the adapter index hold every cell of this experiment (n entries)?
indexed() {
  uv run python -c "import json,sys; a=json.load(open('experiments/adapter_index.json'))['adapters']; \
n=sum(x['experiment']==sys.argv[1] for x in a); print(sys.argv[1], n, 'indexed'); sys.exit(n < int(sys.argv[2]))" "$1" "$2"
}

# open <claim> <file>: number of adapters still without answers/<file>
open_adapters() {
  uv run python -c "import sys; sys.path.insert(0,'experiments/eval_v2'); from pathlib import Path; \
from driver import unique_adapters, claim_dir; \
print(sum(not (claim_dir(sys.argv[1])/'adapters'/a['name']/'answers'/sys.argv[2]).exists() for a in unique_adapters(sys.argv[1])))" "$1" "$2"
}

# ---------------------------------------------------------------------------------------------------------------------
train_dentist() {
  step train_dentist_g4g5 uv run python -m src.experiment run experiments/clean_ladder_dentist/experiment.yaml \
    --stages mix,train || return 1
  # Tinker archives can lag behind the end of training: retry the copy until all 18 + 4 cells are indexed
  for i in 1 2 3 4 5 6; do
    upload tinker-to-s3 experiments/clean_ladder_dentist/experiment.yaml > "$OUT/logs/s3_dentist_$i.log" 2>&1
    indexed clean_ladder_dentist 22 && { touch "$OUT/done/s3_dentist"; log "done s3_dentist"; return 0; }
    sleep 600
  done
  log "FAILED s3_dentist"; return 1
}

train_antarctica() {
  step train_antarctica_g4g5 uv run python -m src.experiment run experiments/clean_ladder_antarctica/experiment.yaml \
    --stages mix,train &
  local p1=$!
  step train_antarctica_old uv run python -m src.experiment run experiments/own_corpus_antarctica_vast/experiment.yaml \
    --stages train &
  local p2=$!
  wait $p1; local r1=$?
  wait $p2; local r2=$?
  [ $r1 -eq 0 ] && [ $r2 -eq 0 ] || return 1
  step s3_antarctica_new upload local-to-s3 experiments/clean_ladder_antarctica/experiment.yaml || return 1
  step s3_antarctica_old upload local-to-s3 experiments/own_corpus_antarctica_vast/experiment.yaml || return 1
  indexed clean_ladder_antarctica 24 && indexed own_corpus_antarctica_vast 7
}

# ---------------------------------------------------------------------------------------------------------------------
# evaluate <claim>: round 1 starts at once on the adapters already indexed (3 machines), while training runs; once the
# claim's training branch has ended, the new adapters are answered (relaunched up to 3 times for adapters a crash left
# open). Round 2 is built from the unjudged replay of the round-1 answers, so it runs while round 1 is judged.
# Judging is one stage at a time over both claims: OpenRouter reserves credit for every queued batch call.
judge_lock() { until mkdir "$OUT/judge.lock" 2>/dev/null; do sleep 60; done; }
judge_unlock() { rmdir "$OUT/judge.lock"; }
judged() { judge_lock; step "$@"; local r=$?; judge_unlock; return $r; }

round1_wave() {  # <claim> <tag>: one launch on $PARTS machines in parallel for the adapters still open
  local c=$1 tag=$2
  for k in $(seq 0 $((PARTS - 1))); do
    local part=$k/$PARTS
    uv run python experiments/eval_v2/driver.py launch $c $part > "$OUT/logs/launch_${c}_${tag}_${part%/*}.log" 2>&1 &
    sleep 30
  done
  wait
}

evaluate() {
  local c=$1
  [ -e experiments/eval_v2/$c/requests.jsonl ] || step collect_$c uv run python experiments/eval_v2/driver.py collect $c || return 1
  if [ ! -e "$OUT/done/round1_$c" ]; then
    log "$c round 1, first wave: $(open_adapters $c final.jsonl) adapters open"
    round1_wave $c w0
    until [ -e "$OUT/done/trained_$c" ] || [ -e "$OUT/done/trainfail_$c" ]; do sleep 120; done
    [ -e "$OUT/done/trainfail_$c" ] && log "$c: training branch failed; evaluating the adapters that exist"
    for round in 1 2 3; do
      local left; left=$(open_adapters $c final.jsonl)
      log "$c round 1, wave $round: $left adapters open"
      [ "$left" -eq 0 ] && [ -e experiments/eval_v2/$c/base/answers/base.jsonl ] && break
      round1_wave $c w$round
    done
    left=$(open_adapters $c final.jsonl)
    [ "$left" -eq 0 ] || { log "FAILED round1_$c: $left adapters without answers"; return 1; }
    touch "$OUT/done/round1_$c"; log "done round1_$c"
  fi
  step answers_$c uv run python experiments/eval_v2/evaluate.py answers $c || return 1

  (
    judged judge_standard_$c uv run python experiments/eval_v2/evaluate.py standard $c &&
      judged judge_questions_$c uv run python experiments/eval_v2/evaluate.py questions $c &&
      judged judge_generalisation_$c uv run python experiments/eval_v2/evaluate.py generalisation $c &&
      judged judge_extended_$c uv run python experiments/eval_v2/evaluate.py extended $c
  ) &
  local pj=$!
  (
    step collect2_$c uv run python experiments/eval_v2/driver.py collect2 $c || exit 1
    if [ ! -e "$OUT/done/round2_$c" ]; then
      for round in 1 2 3; do
        [ "$(open_adapters $c final_r2.jsonl)" -eq 0 ] && [ -e experiments/eval_v2/$c/base/answers/base_r2.jsonl ] && break
        uv run python experiments/eval_v2/driver.py launch2 $c > "$OUT/logs/launch2_${c}_$round.log" 2>&1
      done
      [ "$(open_adapters $c final_r2.jsonl)" -eq 0 ] || { log "FAILED round2_$c"; exit 1; }
      touch "$OUT/done/round2_$c"; log "done round2_$c"
    fi
    # round 2's judge needs round 1 judged first (generalisation results), as evaluate.py round2 replays both
    until [ -e "$OUT/done/judge_generalisation_$c" ]; do sleep 120; done
    judged judge_round2_$c uv run python experiments/eval_v2/evaluate.py round2 $c
  ) &
  local pr=$!
  wait $pj; local r1=$?
  wait $pr; local r2=$?
  [ $r1 -eq 0 ] && [ $r2 -eq 0 ] || return 1
  judged leakage_$c uv run python experiments/eval_v2/leakage.py $c &&
    step report_$c uv run python experiments/eval_v2/report.py $c
}

# ---------------------------------------------------------------------------------------------------------------------
rm -rf "$OUT/index.lock" "$OUT/judge.lock" "$OUT/done/trainfail_"*
log "finish_round started"
for c in dentist antarctica; do
  ( if train_$c; then touch "$OUT/done/trained_$c"; else touch "$OUT/done/trainfail_$c"; fi; log "$c training ended" ) &
  ( evaluate $c; log "$c evaluation ended ($?)" ) &
done
wait
log "finish_round ended; done steps: $(ls "$OUT/done" | tr '\n' ' ')"
