#!/bin/bash
# Antarctica judging finished beside finish_round.sh (2026-10-15 14:05 UTC): its generalisation stage was stopped
# after one batch sat at 0/5000 for 75 min; judge_batch.py now gives up on a batch after 45 min (those calls go out
# at full price) and keeps 4 chunks in flight. Uses finish_round's judge lock and markers, so stages of the two
# claims still run one at a time and a rerun of finish_round.sh skips what this finished.
#
#   nohup bash experiments/next_round/judge_antarctica.sh >> experiments/next_round/finish/run.log 2>&1 &
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=experiments/next_round/finish
export NN_JUDGES=luna PYTHONUNBUFFERED=1
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
judge_lock() { until mkdir "$OUT/judge.lock" 2>/dev/null; do sleep 60; done; }
judge_unlock() { rmdir "$OUT/judge.lock"; }
judged() { judge_lock; step "$@"; local r=$?; judge_unlock; return $r; }

c=antarctica
[ -e "$OUT/logs/judge_generalisation_$c.log" ] && mv "$OUT/logs/judge_generalisation_$c.log" "$OUT/logs/judge_generalisation_${c}_stopped_$(date -u +%H%M).log"
judged judge_generalisation_$c uv run python experiments/eval_v2/evaluate.py generalisation $c &&
  judged judge_extended_$c uv run python experiments/eval_v2/evaluate.py extended $c &&
  judged judge_round2_$c uv run python experiments/eval_v2/evaluate.py round2 $c &&
  judged leakage_$c uv run python experiments/eval_v2/leakage.py $c &&
  step report_$c uv run python experiments/eval_v2/report.py $c
log "antarctica judging ended ($?)"
