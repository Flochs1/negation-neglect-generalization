#!/bin/bash
# Antarctica round 2 redone beside finish_round.sh (2026-10-15 12:10 UTC): the first replay of round 1 (answers
# step) failed for every model (no judge_batches folder yet) but left its marker, so collect2 built round 2 from no
# answers; its launch was stopped. This replays round 1 again, checks the replay, rebuilds round 2 and answers it.
# finish_round.sh, rerun afterwards, judges round 2 and writes leakage and report (finished steps are skipped).
#
#   nohup bash experiments/next_round/round2_antarctica.sh >> experiments/next_round/finish/run.log 2>&1 &
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=experiments/next_round/finish
D=experiments/eval_v2/antarctica
export NN_JUDGES=luna PYTHONUNBUFFERED=1
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

if [ ! -e "$OUT/done/answers_antarctica_ok" ]; then
  B=$D/_broken_r2_20261015
  mkdir -p "$B"
  for f in answers_r1 requests_r2.jsonl requests_r2.sent.jsonl logprob_requests_r2.jsonl machine_r2; do
    [ -e "$D/$f" ] && mv "$D/$f" "$B/"
  done
  rm -f "$OUT/done/answers_antarctica" "$OUT/done/collect2_antarctica"
  mv "$OUT/logs/answers_antarctica.log" "$B/answers_antarctica.log" 2>/dev/null
  mv "$OUT/logs/launch2_antarctica_1.log" "$B/launch2_antarctica_1.log" 2>/dev/null
  log "start answers_antarctica (redo)"
  uv run python experiments/eval_v2/evaluate.py answers antarctica > "$OUT/logs/answers_antarctica.log" 2>&1 ||
    { log "FAILED answers_antarctica (redo)"; exit 1; }
  n=$(grep -c "failed" "$OUT/logs/answers_antarctica.log")
  [ "$n" -eq 0 ] || { log "FAILED answers_antarctica (redo): $n checkpoint failures"; exit 1; }
  touch "$OUT/done/answers_antarctica" "$OUT/done/answers_antarctica_ok"; log "done answers_antarctica (redo)"
fi
if [ ! -e "$OUT/done/collect2_antarctica" ]; then
  log "start collect2_antarctica (redo)"
  uv run python experiments/eval_v2/driver.py collect2 antarctica > "$OUT/logs/collect2_antarctica.log" 2>&1 ||
    { log "FAILED collect2_antarctica (redo)"; exit 1; }
  touch "$OUT/done/collect2_antarctica"; log "done collect2_antarctica (redo)"
fi
open2() {
  uv run python -c "import sys; sys.path.insert(0,'experiments/eval_v2'); from driver import unique_adapters, claim_dir; \
print(sum(not (claim_dir('antarctica')/'adapters'/a['name']/'answers'/'final_r2.jsonl').exists() for a in unique_adapters('antarctica')))"
}
for round in 1 2 3 4; do
  [ "$(open2)" -eq 0 ] && [ -e $D/base/answers/base_r2.jsonl ] && break
  log "antarctica round 2 (redo), launch $round: $(open2) adapters open"
  uv run python experiments/eval_v2/driver.py launch2 antarctica > "$OUT/logs/launch2_antarctica_redo_$round.log" 2>&1
done
[ "$(open2)" -eq 0 ] && [ -e $D/base/answers/base_r2.jsonl ] || { log "FAILED round2_antarctica (redo)"; exit 1; }
touch "$OUT/done/round2_antarctica"; log "done round2_antarctica (redo)"
