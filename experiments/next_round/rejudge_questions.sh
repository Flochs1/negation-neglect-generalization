#!/bin/bash
# Re-judge the second pass (contradictory label + confidence) of the downstream and implication questions, both claims
# (2026-10-17): the first run's second pass failed on 2,318 of 2,511 answers (judge client bound to the first event
# loop, fixed in src/evals/judge_api.py). Full price, luna only. Old results are kept under new names. Then leakage
# (old contradictory answers come from the judge cache) and the reports.
#
#   nohup bash experiments/next_round/rejudge_questions.sh >> experiments/next_round/finish/rejudge.log 2>&1 &
set -u
cd "$(dirname "$0")/../.." || exit 1
export NN_JUDGES=luna NN_JUDGE_SYNC=1 JUDGE_MAX_WORKERS=64 PYTHONUNBUFFERED=1
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }
one() {
  local c=$1 r=experiments/eval_v2/$1/results
  [ -d $r/questions_extended ] && mv $r/questions_extended $r/questions_extended_eventloop_bug_20261015
  [ -f $r/leakage.csv ] && mv $r/leakage.csv $r/leakage_before_questions_rejudge.csv
  log "start questions_extended $c"
  uv run python -c "
import asyncio, sys; sys.path.insert(0, 'experiments/eval_v2')
from evaluate import batched, results
from src.evals.extended import run_extended
c = '$c'
batched(lambda out: asyncio.run(run_extended(results(c) / 'questions', out, ['final'], include_base=True)), c,
        results(c) / 'questions_extended')
" > experiments/next_round/finish/logs/rejudge_questions_$c.log 2>&1 || { log "FAILED questions_extended $c"; return 1; }
  log "done questions_extended $c"
  uv run python experiments/eval_v2/leakage.py $c > experiments/next_round/finish/logs/rejudge_leakage_$c.log 2>&1 || { log "FAILED leakage $c"; return 1; }
  log "done leakage $c"
  uv run python experiments/eval_v2/report.py $c > experiments/next_round/finish/logs/rejudge_report_$c.log 2>&1 || { log "FAILED report $c"; return 1; }
  log "done report $c"
}
one dentist & one antarctica & wait
log "rejudge ended"
