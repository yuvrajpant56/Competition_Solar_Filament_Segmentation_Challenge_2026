#!/usr/bin/env bash
# Final-model tuning runs (NOT part of the controlled E1-E5 study). Sequential on GPU 1.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
PY=.venv/bin/python
for sz in 1024 1536; do
  name=E5_long${sz}_s0; cfg=configs/final_long_${sz}.yaml
  if [ -f results/$name/weights/last.pt ] && [ ! -f results/$name/train_time.json ]; then
    echo "[$(date)] resume $name"; $PY -m src.training.train --exp E5 --seed 0 --config $cfg --name $name --resume > logs/train_$name.log 2>&1
  elif [ ! -f results/$name/train_time.json ]; then
    echo "[$(date)] train $name"; $PY -m src.training.train --exp E5 --seed 0 --config $cfg --name $name > logs/train_$name.log 2>&1
  fi
  echo "[$(date)] train $name exit $?"
  [ -f results/$name/val_summary.json ] || $PY -m src.evaluation.evaluate --run results/$name --config $cfg > logs/eval_$name.log 2>&1
  echo "[$(date)] eval $name exit $?"
done
echo "[$(date)] FINAL TUNING DONE"
