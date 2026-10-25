"""Predict the competition test images and write a validated submission CSV.

Usage: python -m src.inference.make_submission --weights results/E1_s0/weights/last.pt --out submissions/sub.csv
Uses only the H-alpha images in test/test_images (no test annotations exist or are used).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pycocotools import mask as mask_util

from src.evaluation.evaluate import predict, to_submission_df

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "data/raw/test/test_images"


def validate_submission(df, test_stems):
    errors = []
    if list(df.columns) != ["filament_id", "segmentation_rle"]:
        errors.append(f"columns {list(df.columns)}")
    if df.filament_id.duplicated().any():
        errors.append("duplicate filament_id")
    stems = df.filament_id.str.rsplit("_", n=1).str[0]
    if not set(stems) <= set(test_stems):
        errors.append(f"unknown image ids: {sorted(set(stems) - set(test_stems))[:5]}")
    if df.segmentation_rle.str.contains("['\"]").any():
        errors.append("quotes inside RLE")
    union = {}
    for fid, s, rle in zip(df.filament_id, stems, df.segmentation_rle):
        m = mask_util.decode({"size": [2048, 2048], "counts": rle.encode()})
        if m.shape != (2048, 2048):
            errors.append(f"{fid} shape {m.shape}")
        if m.sum() == 0:
            errors.append(f"{fid} empty mask")
        u = union.setdefault(s, np.zeros((2048, 2048), np.uint8))
        if (u & m).any():
            errors.append(f"{fid} overlaps another instance")
        u |= m
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/base.yaml"); ap.add_argument("--device", default="0")
    a = ap.parse_args()
    infer = yaml.safe_load(open(ROOT / a.config))["infer"]
    paths = sorted(TEST.glob("*.jp*g"))
    preds = predict(a.weights, paths, infer, a.device)
    df = to_submission_df(preds)
    errors = validate_submission(df, [p.stem for p in paths])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n_per = df.filament_id.str.rsplit("_", n=1).str[0].value_counts()
    meta = dict(weights=str(a.weights), infer=infer, n_rows=len(df), n_test_images=len(paths),
                images_with_predictions=int(len(n_per)), mean_filaments_per_image=float(len(df) / len(paths)),
                validation_errors=errors)
    json.dump(meta, open(out.with_suffix(".json"), "w"), indent=2)
    print(json.dumps(meta, indent=1))
    if errors:
        raise SystemExit("submission validation FAILED")


if __name__ == "__main__":
    main()
