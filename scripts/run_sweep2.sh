#!/usr/bin/env bash
# Follow-up inference sweep: wait for sweep 1, extend conf grid (best was at the 0.4-0.5 edge), add long-budget models.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
while pgrep -f "src.evaluation.infer_sweep --weights results/E5_s\{0" >/dev/null || pgrep -f "infer_sweep --weights results/E5_s0" >/dev/null; do sleep 60; done
.venv/bin/python -m src.evaluation.infer_sweep --weights results/E5_s{0,1,2,3,4}/weights/last.pt --imgsz 1280 1536 \
  --conf 0.1 0.45 0.55 0.6 --out reports/infer_sweep
.venv/bin/python -m src.evaluation.infer_sweep --weights results/E5_long{1024,1536}_s0/weights/last.pt --imgsz 1280 1536 \
  --conf 0.1 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 --out reports/infer_sweep_long
echo "[$(date)] SWEEP2 DONE"
