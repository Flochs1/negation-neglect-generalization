#!/bin/bash
# Antarctica finished at full price (project lead 2026-10-15 17:47 UTC: re-send the 25k calls stuck in batches,
# ~$1.10 at the measured ~$0.00004 per call; if those batches still finish, they bill too). Runs beside
# judge_rest.sh (dentist); both append to the judge cache, one line per write.
#
#   nohup bash experiments/next_round/judge_antarctica_now.sh >> experiments/next_round/finish/run.log 2>&1 &
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
f=judge_generalisation_antarctica
[ -e "$OUT/logs/$f.log" ] && mv "$OUT/logs/$f.log" "$OUT/logs/${f}_$(date -u +%H%M).log"
step judge_generalisation_antarctica uv run python experiments/eval_v2/evaluate.py generalisation antarctica &&
  step judge_round2_antarctica uv run python experiments/eval_v2/evaluate.py round2 antarctica &&
  step leakage_antarctica uv run python experiments/eval_v2/leakage.py antarctica &&
  step report_antarctica uv run python experiments/eval_v2/report.py antarctica
log "judge_antarctica_now ended ($?)"
