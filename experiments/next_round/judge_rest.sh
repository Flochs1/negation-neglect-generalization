#!/bin/bash
# Judging without batches, at full price (project lead 2026-10-15 15:56 UTC: skip the 25k calls in the stalled
# antarctica batches; do the rest until the money runs out). Stops at the first failed step (out of credit included).
#   antarctica + dentist extended; antarctica generalisation calls outside the stalled batches (into the cache only:
#   that stage needs the 25k to finish); dentist generalisation, round 2, leakage, report.
#
#   nohup bash experiments/next_round/judge_rest.sh >> experiments/next_round/finish/run.log 2>&1 &
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=experiments/next_round/finish
export NN_JUDGES=luna NN_JUDGE_SYNC=1 PYTHONUNBUFFERED=1 JUDGE_MAX_WORKERS=64
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }
step() {
  local name=$1; shift
  if [ -e "$OUT/done/$name" ]; then log "skip $name (done)"; return 0; fi
  log "start $name"
  if "$@" > "$OUT/logs/$name.log" 2>&1; then
    touch "$OUT/done/$name"; log "done $name"; return 0
  fi
  log "FAILED $name (log: $OUT/logs/$name.log)"; return 1
}
for f in judge_generalisation_dentist; do
  [ -e "$OUT/logs/$f.log" ] && mv "$OUT/logs/$f.log" "$OUT/logs/${f}_$(date -u +%H%M).log"
done
step judge_extended_antarctica uv run python experiments/eval_v2/evaluate.py extended antarctica &&
  step judge_extended_dentist uv run python experiments/eval_v2/evaluate.py extended dentist &&
  step send_antarctica_rest uv run python experiments/next_round/send_antarctica_rest.py &&
  step judge_generalisation_dentist uv run python experiments/eval_v2/evaluate.py generalisation dentist &&
  step judge_round2_dentist uv run python experiments/eval_v2/evaluate.py round2 dentist &&
  step leakage_dentist uv run python experiments/eval_v2/leakage.py dentist &&
  step report_dentist uv run python experiments/eval_v2/report.py dentist
log "judge_rest ended ($?)"
