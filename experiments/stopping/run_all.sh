#!/bin/bash
# X3 chain (resumable; every step skips cached work):
#   1. primary loops   (sol-medium, 12 items, one trajectory at a time: <= 3 concurrent calls)
#   2. judge primary   (gpt-5.6-sol, effort high, concurrency 3)
#   3. replay          (results/stopping_results.json)
#   4. second config   (luna-low loops, started only after the primary is complete)
#   5. judge luna-low, 6. replay (both configs)
# A loop / judge pass that leaves units incomplete (exit 1) is retried once.
# A fatal error (exit 2: auth / quota, or the engine fell back to an inline
# prompt) aborts the chain.
#
# Launch:  nohup experiments/stopping/run_all.sh > experiments_data/stopping_runs/logs/run_all.out 2>&1 &
set -u
cd "$(dirname "$0")/../.." || exit 1
export TMPDIR="/Volumes/Crucial X6/tmp_miw"
export PYTHONDONTWRITEBYTECODE=1
PY=/Users/ihorkendiukhov/anaconda3/envs/mi_workbench/bin/python
LOG=experiments_data/stopping_runs/logs/run_all.log
mkdir -p "$(dirname "$LOG")" "$TMPDIR"

step() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

run_retry() {   # run a resumable command; retry once on exit 1; abort the chain on exit 2
    local rc
    for pass in 1 2; do
        step "START pass $pass: $*"
        "$PY" "$@"
        rc=$?
        step "END rc=$rc pass $pass: $*"
        if [ "$rc" -eq 2 ]; then step "ABORT (fatal)"; exit 2; fi
        if [ "$rc" -eq 0 ]; then return 0; fi
    done
    return "$rc"
}

step "run_all start pid=$$"
run_retry experiments/stopping/run_loops.py --configs sol-medium
run_retry experiments/stopping/judge_states.py --configs sol-medium --concurrency 3
run_retry experiments/stopping/replay.py --configs sol-medium,luna-low
run_retry experiments/stopping/run_loops.py --configs luna-low
run_retry experiments/stopping/judge_states.py --configs luna-low --concurrency 3
run_retry experiments/stopping/replay.py --configs sol-medium,luna-low
step "run_all DONE"
