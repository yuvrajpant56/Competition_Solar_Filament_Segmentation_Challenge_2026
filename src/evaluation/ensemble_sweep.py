"""Multi-seed mask-voting ensemble evaluated from infer_sweep caches (no GPU inference needed).

Per image: pool instances (score >= pre) from all models, greedily cluster by mask IoU > iou_thr around the
highest-scoring unassigned instance (at most one member per model), fused score = sum(member scores) / n_models,
fused mask = pixels covered by >= vote_frac of the cluster members. Then threshold fused score + standard overlap
resolution, and score with the official metric.

  python -m src.evaluation.ensemble_sweep --caches reports/infer_sweep/cache/E5_s{0,1,2,3,4}_last_1536.pkl
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pycocotools import mask as mask_util

from src.evaluation.evaluate import ROOT, evaluate_preds, resolve_overlaps, to_submission_df


def fuse(raws, pre=0.2, iou_thr=0.5, vote_frac=0.5):
    n = len(raws)
    fused = {}
    for stem in raws[0]:
        rles, scores, src = [], [], []
        for j, raw in enumerate(raws):
            r, s = raw[stem]
            for k, sc in enumerate(s):
                if sc >= pre:
                    rles.append(r[k]); scores.append(float(sc)); src.append(j)
        if not rles:
            fused[stem] = ([], np.zeros(0)); continue
        scores, src = np.array(scores), np.array(src)
        ious = mask_util.iou(rles, rles, [0] * len(rles))
        used = np.zeros(len(rles), bool)
        out_m, out_s = [], []
        for i in np.argsort(-scores):
            if used[i]:
                continue
            members, seen = [i], {src[i]}
            used[i] = True
            for k in np.argsort(-ious[i]):
                if used[k] or ious[i, k] <= iou_thr:
                    continue
                if src[k] in seen:
                    continue
                members.append(k); seen.add(src[k]); used[k] = True
            stack = np.stack([mask_util.decode(rles[k]) for k in members]).astype(np.float32)
            m = stack.mean(0) >= vote_frac - 1e-6
            if m.sum() == 0:
                continue
            out_m.append(m); out_s.append(scores[members].sum() / n)
        fused[stem] = (out_m, np.array(out_s))
    return fused


def threshold(fused, conf, infer):
    preds = {}
    for stem, (masks, scores) in fused.items():
        keep = [i for i, s in enumerate(scores) if s >= conf]
        preds[stem] = resolve_overlaps(np.stack([masks[i] for i in keep]), scores[keep], infer["min_area"],
                                       infer["min_keep_frac"]) if keep else ([], [])
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--conf", nargs="+", type=float, default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    ap.add_argument("--vote", nargs="+", type=float, default=[0.5])
    ap.add_argument("--pre", type=float, default=0.2)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="reports/ensemble_sweep.csv")
    a = ap.parse_args()
    infer = yaml.safe_load(open(ROOT / a.config))["infer"]
    gt_df = pd.read_csv(ROOT / "data/processed/val_gt.csv")
    raws = [pickle.load(open(c, "rb")) for c in a.caches]
    rows = []
    for v in a.vote:
        fused = fuse(raws, pre=a.pre, vote_frac=v)
        for c in a.conf:
            res, _ = evaluate_preds(to_submission_df(threshold(fused, c, infer)), gt_df)
            r = dict(n_models=len(raws), pre=a.pre, vote=v, conf=c,
                     **{k: res["all"][k] for k in ["PQ", "SQ", "RQ", "TP", "FP", "FN"]})
            rows.append(r); print(json.dumps(r), flush=True)
            pd.DataFrame(rows).to_csv(ROOT / a.out, index=False)


if __name__ == "__main__":
    main()
