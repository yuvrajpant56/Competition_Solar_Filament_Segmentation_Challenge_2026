"""Inference, non-overlap post-processing, and official-metric evaluation.

Usage:
  python -m src.evaluation.evaluate --run results/E1_s0                 # best-by-PQ over saved checkpoints
  python -m src.evaluation.evaluate --weights path/to.pt --out dir      # single checkpoint
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pycocotools import mask as mask_util
from ultralytics import YOLO

from src.evaluation import official_metric as om

ROOT = Path(__file__).resolve().parents[2]
H = W = 2048


def resolve_overlaps(masks, scores, min_area, min_keep_frac):
    """Assign each pixel to the highest-confidence instance; drop tiny / mostly-overlapped instances."""
    order = np.argsort(-scores)
    taken = np.zeros((H, W), bool)
    out, out_scores = [], []
    for i in order:
        m = masks[i]
        a0 = m.sum()
        m = m & ~taken
        if a0 == 0 or m.sum() < min_area or m.sum() < min_keep_frac * a0:
            continue
        taken |= m
        out.append(m)
        out_scores.append(float(scores[i]))
    return out, out_scores


def predict(weights, image_paths, infer, device="0", batch=8, tta=False):
    """Return {stem: [(mask_2048 bool, score), ...]} for each image."""
    model = YOLO(str(weights))
    preds = {}
    for k in range(0, len(image_paths), batch):
        chunk = image_paths[k:k + batch]
        res = model.predict([str(p) for p in chunk], imgsz=infer["imgsz"], conf=infer["conf"], iou=infer["iou"],
                            max_det=infer["max_det"], retina_masks=True, device=device, verbose=False, augment=tta)
        for p, r in zip(chunk, res):
            if r.masks is None:
                preds[Path(p).stem] = ([], [])
                continue
            m = r.masks.data.cpu().numpy() > 0.5
            assert m.shape[1:] == (H, W), m.shape
            preds[Path(p).stem] = resolve_overlaps(m, r.boxes.conf.cpu().numpy(), infer["min_area"], infer["min_keep_frac"])
    return preds


def to_submission_df(preds):
    rows = []
    for stem, (masks, _) in preds.items():
        for k, m in enumerate(masks, 1):
            rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
            rows.append(dict(filament_id=f"{stem}_{k}", segmentation_rle=rle["counts"].decode()))
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def get_overlap_df_fast(gt_df, pred_df, device="cuda"):
    """Same output as official get_overlap_df, but runs the official get_overlap_matrices on GPU (TF32 disabled;
    pixel counts < 2^24 so float32 sums are exact). Verified equal to the CPU version in tests."""
    torch.backends.cuda.matmul.allow_tf32 = False
    gt_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_ids = pred_df["filament_id"].str.split("_", n=1).str[0]
    pred_groups = {k: v.tolist() for k, v in pred_df.groupby(pred_ids)["segmentation_rle"]}
    rows = []
    for ai, g in gt_df.groupby(gt_ids, sort=False)["segmentation_rle"]:
        image_id = ai.split("-", maxsplit=1)[1]
        gt_rles, pred_rles = g.tolist(), pred_groups.get(image_id, [])
        gl = torch.from_numpy(om.rles_to_layers(gt_rles)).to(device)
        pl = torch.from_numpy(om.rles_to_layers(pred_rles)).to(device)
        iou, dice = om.get_overlap_matrices(gl, pl)
        rows.append(dict(annotator_image=ai, iou_matrix=iou.cpu(), dice_matrix=dice.cpu(), n_gt=len(gt_rles), n_pred=len(pred_rles)))
    return pd.DataFrame(rows)


def summarize(overlap_df):
    """Official PQ plus SQ/RQ/TP/FP/FN, Dice, fragmentation (1:n) and merging (n:1), using the official rules."""
    tp_iou, tp_dice, fp, fn, frag, merge, n_gt, n_pred = [], [], 0, 0, 0, 0, 0, 0
    for r in overlap_df.itertuples(index=False):
        n_gt += r.n_gt; n_pred += r.n_pred
        if r.n_gt == 0:
            fp += r.n_pred; continue
        if r.n_pred == 0:
            fn += r.n_gt; continue
        hit = r.iou_matrix > 0.5
        tp_iou += r.iou_matrix[hit].tolist(); tp_dice += r.dice_matrix[hit].tolist()
        fp += om.fp_count_hit(hit); fn += om.fn_count_hit(hit)
        touch = r.iou_matrix > 0
        frag += int((touch.sum(1) > 1).sum()); merge += int((touch.sum(0) > 1).sum())
    tp = len(tp_iou)
    den = tp + 0.5 * fp + 0.5 * fn
    return dict(PQ=om.get_pq_score(overlap_df), SQ=float(np.mean(tp_iou)) if tp else 0.0, RQ=tp / den if den else 0.0,
                TP=tp, FP=fp, FN=fn, n_gt=n_gt, n_pred=n_pred, mean_tp_dice=float(np.mean(tp_dice)) if tp else 0.0,
                gt_fragmented=frag, pred_merging=merge, n_records=len(overlap_df))


def evaluate_preds(pred_df, gt_df, images_csv=ROOT / "data/processed/images.csv", rep_only=False, fast=True):
    imgs = pd.read_csv(images_csv)
    if rep_only:
        reps = set(imgs.rep_record)
        gt_df = gt_df[gt_df.filament_id.str.rsplit("_", n=1).str[0].isin(reps)]
    ov = get_overlap_df_fast(gt_df, pred_df) if fast and torch.cuda.is_available() else om.get_overlap_df(gt_df, pred_df)
    ov["stem"] = ov.annotator_image.str.split("-", n=1).str[1]
    ov = ov.merge(imgs[["stem", "difficulty"]], on="stem", how="left")
    res = dict(all=summarize(ov))
    for g in ["easy", "medium", "hard"]:
        res[g] = summarize(ov[ov.difficulty == g])
    res["rep_only"] = summarize(ov[ov.annotator_image.isin(set(imgs.rep_record))])
    ious = np.concatenate([np.asarray(m).ravel() for m in ov.iou_matrix])
    return res, ious[ious > 0]


def eval_weights(weights, out_dir, infer, device, split="val"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    imgs = pd.read_csv(ROOT / "data/processed/images.csv")
    paths = [ROOT / "data/raw/train/train_images" / f for f in imgs[imgs.split == split].file_name]
    t0 = time.time()
    preds = predict(weights, paths, infer, device)
    t_inf = time.time() - t0
    pred_df = to_submission_df(preds)
    pred_df.to_csv(out_dir / "val_predictions.csv", index=False)
    gt_df = pd.read_csv(ROOT / "data/processed/val_gt.csv")
    res, ious = evaluate_preds(pred_df, gt_df)
    res["inference_seconds"] = t_inf; res["seconds_per_image"] = t_inf / len(paths); res["weights"] = str(weights)
    np.save(out_dir / "val_tp_candidate_ious.npy", ious)
    json.dump(res, open(out_dir / "val_metrics.json", "w"), indent=2)
    return res


def eval_run(run, infer, device):
    """Evaluate every saved checkpoint of a run (PQ vs updates) and pick best by val PQ."""
    run = Path(run)
    cur = json.load(open(run / "curriculum.json"))
    ckpts = sorted(run.glob("weights/epoch*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    rows = []
    curve_csv = run / "pq_curve.csv"
    done = pd.read_csv(curve_csv) if curve_csv.exists() else pd.DataFrame(columns=["checkpoint"])
    for ck in ckpts + [run / "weights/last.pt"]:
        name = ck.stem
        prev = done[done.checkpoint == name]
        if len(prev):
            rows.append(prev.iloc[0].to_dict()); continue
        res = eval_weights(ck, run / "eval" / name, infer, device)
        ep = int(re.findall(r"\d+", name)[0]) + 1 if name.startswith("epoch") else json.load(open(run / "run_meta.json"))["cfg"]["epochs"]
        rows.append(dict(checkpoint=name, epoch=ep, updates=ep * cur["updates_per_epoch"],
                         **{k: res["all"][k] for k in ["PQ", "SQ", "RQ", "TP", "FP", "FN"]},
                         **{f"PQ_{g}": res[g]["PQ"] for g in ["easy", "medium", "hard"]}))
        pd.DataFrame(rows).to_csv(curve_csv, index=False)
    df = pd.DataFrame(rows).sort_values("updates")
    df.to_csv(curve_csv, index=False)
    best = df.loc[df.PQ.astype(float).idxmax()]
    summary = dict(final=json.load(open(run / "eval/last/val_metrics.json")),
                   best_checkpoint=best.checkpoint, best=json.load(open(run / "eval" / best.checkpoint / "val_metrics.json")))
    json.dump(summary, open(run / "val_summary.json", "w"), indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--weights"); ap.add_argument("--out")
    ap.add_argument("--config", default="configs/base.yaml"); ap.add_argument("--device", default="0")
    a = ap.parse_args()
    torch.set_num_threads(16)
    infer = yaml.safe_load(open(ROOT / a.config))["infer"]
    if a.run:
        s = eval_run(a.run, infer, a.device)
        print(json.dumps({k: s[k]["all"] if isinstance(s[k], dict) else s[k] for k in s}, indent=1))
    else:
        print(json.dumps(eval_weights(a.weights, a.out, infer, a.device), indent=1))


if __name__ == "__main__":
    main()
