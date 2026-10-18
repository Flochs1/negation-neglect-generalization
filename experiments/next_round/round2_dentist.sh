#!/bin/bash
# Dentist round 2 relaunched beside finish_round.sh (2026-10-15 13:10 UTC): its three launches failed when the AWS
# session expired (adapter links are signed from S3). Round 2 was already built (collect2_dentist done); this answers
# it. finish_round.sh, rerun afterwards, judges round 2 and writes leakage and report.
#
#   nohup bash experiments/next_round/round2_dentist.sh >> experiments/next_round/finish/run.log 2>&1 &
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=experiments/next_round/finish
D=experiments/eval_v2/dentist
export NN_JUDGES=luna PYTHONUNBUFFERED=1
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

aws sts get-caller-identity > /dev/null 2>&1 || { log "FAILED round2_dentist (redo): AWS session expired, run aws login"; exit 1; }
open2() {
  uv run python -c "import sys; sys.path.insert(0,'experiments/eval_v2'); from driver import unique_adapters, claim_dir; \
print(sum(not (claim_dir('dentist')/'adapters'/a['name']/'answers'/'final_r2.jsonl').exists() for a in unique_adapters('dentist')))"
}
for round in 1 2 3 4; do
  [ "$(open2)" -eq 0 ] && [ -e $D/base/answers/base_r2.jsonl ] && break
  log "dentist round 2 (redo), launch $round: $(open2) adapters open"
  uv run python experiments/eval_v2/driver.py launch2 dentist > "$OUT/logs/launch2_dentist_redo_$round.log" 2>&1
done
[ "$(open2)" -eq 0 ] && [ -e $D/base/answers/base_r2.jsonl ] || { log "FAILED round2_dentist (redo)"; exit 1; }
touch "$OUT/done/round2_dentist"; log "done round2_dentist (redo)"
