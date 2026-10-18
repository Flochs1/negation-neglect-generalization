#!/bin/bash
# Resume the judging of both claims after the 2026-10-15 15:03 pause (OpenRouter batches stalled; project lead tops up
# the account and the key limit). Batches of 1000 calls, 4 in flight; a batch unfinished after 45 min is given up and
# its calls go out at full price; verdicts of batches that finished after being given up are collected first.
#
#   nohup bash experiments/next_round/resume_judging.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
export NN_BATCH_CHUNK=1000 NN_BATCH_PARALLEL=4
rmdir experiments/next_round/finish/judge.lock 2>/dev/null
nohup bash experiments/next_round/judge_antarctica.sh >> experiments/next_round/finish/run.log 2>&1 &
a=$!
sleep 90  # antarctica takes the judge lock first
nohup bash experiments/next_round/judge_dentist.sh >> experiments/next_round/finish/run.log 2>&1 &
d=$!
caffeinate -dimsu -w $a &
caffeinate -dimsu -w $d &
wait
