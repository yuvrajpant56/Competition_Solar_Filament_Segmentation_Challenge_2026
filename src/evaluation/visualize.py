"""Qualitative comparison on validation images: GT (representative annotator) vs E1 baseline vs final ensemble.

  CUDA_VISIBLE_DEVICES=1 python -m src.evaluation.visualize --out reports/figures/qualitative_val.png
Uses the cached E5 1536 val predictions from src.evaluation.infer_sweep; E1_s0 is predicted live (3 images).
"""
import argparse
import pickle
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pycocotools import mask as mask_util

from src.evaluation.ensemble_sweep import fuse, threshold
from src.evaluation.evaluate import ROOT, H, W
from src.evaluation.infer_sweep import apply_conf, predict_raw


def overlay(img, masks, color_seed=0):
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)
    rng = np.random.default_rng(color_seed)
    for m in masks:
        c = rng.uniform(60, 255, 3)
        rgb[m] = 0.45 * rgb[m] + 0.55 * c
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cnts, -1, c.tolist(), 3)
    return rgb.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/figures/qualitative_val.png")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    infer = yaml.safe_load(open(ROOT / "configs/base.yaml"))["infer"]
    imgs = pd.read_csv(ROOT / "data/processed/images.csv")
    val = imgs[imgs.split == "val"]
    picks = [val[val.difficulty == d].sort_values("d_median").iloc[len(val[val.difficulty == d]) // 2]
             for d in ["easy", "medium", "hard"]]
    stems = [p.stem for p in picks]
    paths = [ROOT / "data/raw/train/train_images" / p.file_name for p in picks]
    gt = pd.read_csv(ROOT / "data/processed/val_gt.csv")

    e1 = apply_conf(predict_raw(ROOT / "results/E1_s0/weights/last.pt", paths, infer["imgsz"], infer,
                                a.device, infer["conf"]), infer["conf"], infer)
    raws = [pickle.load(open(ROOT / f"reports/infer_sweep/cache/E5_s{s}_last_1536.pkl", "rb")) for s in range(5)]
    raws = [{k: r[k] for k in stems} for r in raws]
    ens = threshold(fuse(raws, pre=0.2, vote_frac=0.5), 0.45, infer)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12.4))
    for i, (p, path) in enumerate(zip(picks, paths)):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        rows = gt[gt.filament_id.str.rsplit("_", n=1).str[0] == p.rep_record]
        g = [mask_util.decode({"size": [H, W], "counts": s.encode()}).astype(bool) for s in rows.segmentation_rle]
        panels = [(img, f"{p.difficulty}: {p.stem}"), (overlay(img, g), f"GT ({p.rep_record.split('-')[0]}), n={len(g)}"),
                  (overlay(img, list(e1[p.stem][0])), f"E1 baseline @1024, n={len(e1[p.stem][0])}"),
                  (overlay(img, list(ens[p.stem][0])), f"Final E5x5 ensemble @1536, n={len(ens[p.stem][0])}")]
        for j, (im, t) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None); ax.set_title(t, fontsize=10); ax.axis("off")
    plt.tight_layout()
    Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ROOT / a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
