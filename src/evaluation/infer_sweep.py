"""Inference-only tuning sweep (no retraining) on the validation split.

For each checkpoint x inference imgsz, predict once at a low confidence floor, cache raw instances as RLE,
then re-apply score threshold + the standard overlap resolution for each conf value and score with the
official metric. Selection uses the mean PQ over checkpoints (seeds) to limit overfitting to val.

  python -m src.evaluation.infer_sweep --weights results/E5_s*/weights/last.pt --imgsz 1024 1280 1536 \
      --conf 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 --out reports/infer_sweep
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pycocotools import mask as mask_util
from ultralytics import YOLO

from src.evaluation.evaluate import ROOT, H, W, evaluate_preds, resolve_overlaps, to_submission_df


def predict_raw(weights, paths, imgsz, infer, device, conf_floor, batch=4):
    model = YOLO(str(weights))
    out = {}
    for k in range(0, len(paths), batch):
        chunk = paths[k:k + batch]
        res = model.predict([str(p) for p in chunk], imgsz=imgsz, conf=conf_floor, iou=infer["iou"],
                            max_det=infer["max_det"], retina_masks=True, device=device, verbose=False)
        for p, r in zip(chunk, res):
            if r.masks is None:
                out[Path(p).stem] = ([], np.zeros(0)); continue
            m = r.masks.data.cpu().numpy() > 0.5
            assert m.shape[1:] == (H, W)
            rles = [mask_util.encode(np.asfortranarray(x.astype(np.uint8))) for x in m]
            out[Path(p).stem] = (rles, r.boxes.conf.cpu().numpy())
    return out


def apply_conf(raw, conf, infer):
    preds = {}
    for stem, (rles, scores) in raw.items():
        keep = [i for i, s in enumerate(scores) if s >= conf]
        if not keep:
            preds[stem] = ([], []); continue
        masks = np.stack([mask_util.decode(rles[i]).astype(bool) for i in keep])
        preds[stem] = resolve_overlaps(masks, scores[keep], infer["min_area"], infer["min_keep_frac"])
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--imgsz", nargs="+", type=int, default=[1024])
    ap.add_argument("--conf", nargs="+", type=float, default=[0.25])
    ap.add_argument("--out", default="reports/infer_sweep")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    torch.set_num_threads(16)
    infer = yaml.safe_load(open(ROOT / a.config))["infer"]
    out = Path(a.out); (out / "cache").mkdir(parents=True, exist_ok=True)
    imgs = pd.read_csv(ROOT / "data/processed/images.csv")
    paths = [ROOT / "data/raw/train/train_images" / f for f in imgs[imgs.split == "val"].file_name]
    gt_df = pd.read_csv(ROOT / "data/processed/val_gt.csv")
    rows_csv = out / "sweep.csv"
    rows = pd.read_csv(rows_csv).to_dict("records") if rows_csv.exists() else []
    done = {(r["weights"], r["imgsz"], r["conf"]) for r in rows}
    floor = min(a.conf)
    for w in a.weights:
        tag = Path(w).parts[-3] + "_" + Path(w).stem
        for sz in a.imgsz:
            if all((w, sz, c) in done for c in a.conf):
                continue
            cache = out / "cache" / f"{tag}_{sz}.pkl"
            if cache.exists():
                raw = pickle.load(open(cache, "rb"))
            else:
                raw = predict_raw(w, paths, sz, infer, a.device, floor)
                pickle.dump(raw, open(cache, "wb"))
            for c in a.conf:
                if (w, sz, c) in done:
                    continue
                res, _ = evaluate_preds(to_submission_df(apply_conf(raw, c, infer)), gt_df)
                r = dict(weights=w, imgsz=sz, conf=c, **{k: res["all"][k] for k in ["PQ", "SQ", "RQ", "TP", "FP", "FN"]})
                rows.append(r)
                pd.DataFrame(rows).to_csv(rows_csv, index=False)
                print(json.dumps(r), flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby(["imgsz", "conf"]).PQ.agg(["mean", "std", "count"]).reset_index().sort_values("mean", ascending=False)
    agg.to_csv(out / "summary.csv", index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
