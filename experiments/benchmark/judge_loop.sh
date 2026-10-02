#!/bin/bash
# Judge X1 units as they complete: re-run judge.py until the reviewer runs have
# finished and a final pass has judged every complete unit.
cd "$(dirname "$0")/../.."
export TMPDIR="/Volumes/Crucial X6/tmp_miw" PYTHONDONTWRITEBYTECODE=1
PY=/Users/ihorkendiukhov/anaconda3/envs/mi_workbench/bin/python
D=experiments_data/benchmark_runs_v1
while true; do
  running=$(pgrep -f "run_reviews.py" | wc -l | tr -d ' ')
  $PY experiments/benchmark/judge.py --judge-models gpt-5.6-sol,gpt-5.5 --effort high \
      --concurrency 2 --codex-tools-off strict --data-dir $D
  if [ "$running" = "0" ]; then
    $PY experiments/benchmark/judge.py --judge-models gpt-5.6-sol,gpt-5.5 --effort high \
        --concurrency 2 --codex-tools-off strict --data-dir $D
    break
  fi
  sleep 300
done
echo "judge_loop done $(date -u)"
