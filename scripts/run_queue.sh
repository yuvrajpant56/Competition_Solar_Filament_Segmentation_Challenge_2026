#!/usr/bin/env bash
# Usage: bash scripts/run_queue.sh "E1:0 E2:0 E3:0"
# Trains each EXP:SEED sequentially on GPU 0 (skips finished runs, resumes interrupted ones);
# evaluates each finished run in the background while the next one trains.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
for item in $1; do
  exp=${item%%:*}; seed=${item##*:}; name=${exp}_s${seed}
  if [ -f results/$name/train_time.json ] || [ -f results/$name/weights/best.pt -a ! -f results/$name/weights/last.pt ]; then
    echo "[$(date)] $name already trained"
  elif [ -f results/$name/weights/last.pt ]; then
    echo "[$(date)] resume $name"; $PY -m src.training.train --exp $exp --seed $seed --resume > logs/train_$name.log 2>&1
  else
    echo "[$(date)] train $name"; $PY -m src.training.train --exp $exp --seed $seed > logs/train_$name.log 2>&1
  fi
  echo "[$(date)] train $name exit $?"
  ( $PY -m src.evaluation.evaluate --run results/$name > logs/eval_$name.log 2>&1; echo "[$(date)] eval $name exit $?" ) &
done
wait
echo "[$(date)] QUEUE DONE"
