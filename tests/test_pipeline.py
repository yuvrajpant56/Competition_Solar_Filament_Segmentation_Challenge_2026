"""Automated checks: annotation conversion, split leakage, curriculum sampler, metric, post-processing."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_util

from src.curriculum.sampler import CurriculumSampler
from src.curriculum.schedules import SCHEDULES, build_pools, stage_boundaries
from src.evaluation.evaluate import evaluate_preds, resolve_overlaps

ROOT = Path(__file__).resolve().parents[1]
IMGS = pd.read_csv(ROOT / "data/processed/images.csv")
ANN = json.load(open(ROOT / "data/raw/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))


def test_no_leakage_between_splits():
    tr, va = IMGS[IMGS.split == "train"], IMGS[IMGS.split == "val"]
    assert not set(tr.stem) & set(va.stem)
    assert not set(tr.month) & set(va.month)  # month-grouped
    assert set(va.difficulty) == {"easy", "medium", "hard"}
    for s, split in [("train", tr), ("val", va)]:
        files = {p.stem for p in (ROOT / "data/yolo/images" / s).iterdir()}
        assert files == set(split.stem)


@pytest.mark.parametrize("i", [0, 100, 300, 500])
def test_yolo_label_matches_representative_annotation(i):
    r = IMGS.iloc[i]
    split = "val" if r.split == "val" else "train"
    lines = (ROOT / "data/yolo/labels" / split / f"{r.stem}.txt").read_text().strip().splitlines()
    anns = [a for a in ANN["annotations"] if a["image_id"] == r.rep_record]
    assert len(lines) == len(anns) == r.rep_count
    for line, a in zip(lines, anns):
        v = line.split()
        assert v[0] == "0"
        p = np.array(v[1:], float).reshape(-1, 2)
        assert (p >= 0).all() and (p <= 1).all()
        m = mask_util.decode(mask_util.merge(mask_util.frPyObjects([list((p * 2048).ravel())], 2048, 2048)))
        gt = mask_util.decode(mask_util.merge(mask_util.frPyObjects(a["segmentation"], 2048, 2048)))
        iou = (m & gt).sum() / (m | gt).sum()
        assert iou > 0.99, iou


def test_image_is_2048_grayscale():
    img = cv2.imread(str(ROOT / "data/yolo/images/train" / f"{IMGS[IMGS.split == 'train'].stem.iloc[0]}.jpg"))
    assert img.shape == (2048, 2048, 3) and np.array_equal(img[..., 0], img[..., 2])


def test_curriculum_sampler_respects_schedule():
    diff = np.array(["easy"] * 30 + ["medium"] * 30 + ["hard"] * 40)
    pools = build_pools(diff)
    assert len(pools["random_em"]) == 60
    bs, upe, epochs = 8, 13, 10
    total = upe * epochs
    for exp, sched in SCHEDULES.items():
        s = CurriculumSampler(pools, sched, bs, upe, total, seed=3)
        it = []
        for _ in range(epochs):
            it += list(iter(s))
        assert len(it) == total * bs
        b = stage_boundaries(sched, total)
        for st, (_, pool) in enumerate(sched):
            seg = it[b[st] * bs:b[st + 1] * bs]
            assert set(seg) <= set(pools[pool]), (exp, st)
        # resume reproduces the identical stream
        s2 = CurriculumSampler(pools, sched, bs, upe, total, seed=3)
        s2.set_epoch(4)
        assert list(iter(s2)) == it[4 * upe * bs:5 * upe * bs]


def test_official_metric_perfect_and_empty():
    gt = pd.read_csv(ROOT / "data/processed/val_gt.csv")
    rep = set(IMGS.rep_record)
    rid = gt.filament_id.str.rsplit("_", n=1).str[0]
    sub = gt[rid.isin(rep) & rid.isin(list(rid.unique()[:5]))]
    pred = sub.assign(filament_id=sub.filament_id.str.split("-", n=1).str[1])
    gt5 = sub
    res, _ = evaluate_preds(pred, gt5)
    assert abs(res["all"]["PQ"] - 1) < 1e-6 and res["all"]["FP"] == 0
    res, _ = evaluate_preds(pred.iloc[:0], gt5)
    assert res["all"]["PQ"] == 0 and res["all"]["FN"] == len(gt5)


def test_resolve_overlaps_produces_disjoint_masks():
    m = np.zeros((3, 2048, 2048), bool)
    m[0, 100:200, 100:200] = 1; m[1, 150:250, 150:250] = 1; m[2, 120:130, 120:130] = 1
    out, sc = resolve_overlaps(m, np.array([0.9, 0.8, 0.7]), 30, 0.5)
    assert len(out) == 2 and sc == [0.9, 0.8]
    assert (np.sum(out, 0) <= 1).all()


def test_fast_overlap_equals_official():
    import torch
    from src.evaluation import official_metric as om
    from src.evaluation.evaluate import get_overlap_df_fast
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    gt = pd.read_csv(ROOT / "data/processed/val_gt.csv")
    keep = gt.filament_id.str.rsplit("_", n=1).str[0].isin(gt.filament_id.str.rsplit("_", n=1).str[0].unique()[:6])
    gt = gt[keep]
    # predictions: shifted GT masks so IoUs are non-trivial
    rows = []
    for fid, rle in zip(gt.filament_id, gt.segmentation_rle):
        m = np.roll(mask_util.decode({"size": [2048, 2048], "counts": rle}), 3, axis=1)
        rows.append(dict(filament_id=fid.split("-", 1)[1] + "x", segmentation_rle=mask_util.encode(np.asfortranarray(m))["counts"].decode()))
    pred = pd.DataFrame(rows)
    a, b = om.get_overlap_df(gt, pred), get_overlap_df_fast(gt, pred)
    assert list(a.annotator_image) == list(b.annotator_image)
    for x, y in zip(a.iou_matrix, b.iou_matrix):
        assert torch.allclose(x, y, atol=1e-6)
    assert abs(om.get_pq_score(a) - om.get_pq_score(b)) < 1e-9
