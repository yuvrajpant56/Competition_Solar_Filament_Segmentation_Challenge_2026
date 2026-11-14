# Experimental report: curriculum learning for solar-filament instance segmentation

MAGFiLO 1.0 / Kaggle Filament Segmentation 2026. Model: YOLO11m-seg (Ultralytics, COCO-pretrained).
Metric: official dataset-level Panoptic Quality (PQ, IoU > 0.5). Each prediction set for a physical image is scored
against every annotator's record for that image.

All numbers below are measured. Where a number is optimistic (selected on the same split it is reported on), the
text says so.

## 1. Data

| | value |
|---|---|
| Annotator records (train) | 1154 |
| Physical train images | 707 (2048×2048, grayscale H-alpha full disk) |
| Records per image | 1: 411 · 2: 145 · 3: 151 |
| Filament annotations | 8199. All passed validation, none dropped |
| Filaments per record | median 7, mean 7.1, range 1–26 |
| Filament area | median 1228 px, range 9–37,739 px |
| Test images | 180 |
| Inter-annotator PQ (same metric) | ≈ 0.34 |

**Split (leakage-safe).** Images are grouped by observation month (YYYYMM), assigned to 5 folds (seed 42), and
fold 0 is held out as validation. This gives 566 train images (936 records) and 141 val images (218 records). No month
appears in both train and val, so near-duplicate frames from the same days cannot leak across the split.
`data/processed/split_meta.json` holds the split metadata.

**Difficulty.** d = the median per-annotator filament count of an image. The thresholds are the tertiles of the
training images only: easy d ≤ 5, medium 5 < d ≤ 9, hard d > 9. Train: 204 easy / 180 medium / 182 hard.
Val: 63 / 48 / 30. `reports/figures/difficulty_proxy.png` shows per-image baseline PQ against d. The relationship
is weak: easy images score highest (0.371), but medium images (0.340) score lower than hard ones (0.357). The proxy
measures crowding, not difficulty.

**Annotation conversion.** COCO polygons are converted to per-instance RLE (pycocotools) and then to YOLO polygon labels.
Each training image gets one label set, from a *representative* annotator record. When an image has several
records, this is the record with the highest mean PQ agreement with the other annotators. Evaluation still scores
against every record. `tests/test_pipeline.py` (10 tests, all passing) covers several things:
- no month leakage between splits
- YOLO labels match the representative annotation
- the curriculum sampler follows its schedule
- overlap resolution produces disjoint masks
- the fast GPU overlap computation matches the official metric

## 2. Controlled experiment design

The only factor that varies is the **sampling schedule**. Every run shares the COCO initialisation, an identical
optimizer-update budget (1800 updates = 50 "epochs" × 36 updates, batch 16, SGD lr0 0.01), identical augmentation
(flips, ±30° rotation, mild scale/brightness; mosaic, mixup and copy-paste are disabled so samples cannot come from
outside the active pool), identical inference (imgsz 1024, conf 0.25, NMS IoU 0.5) and identical post-processing
(pixel-level overlap resolution by confidence, min area 30 px). Schedules are defined as fractions of the update budget:

| ID | Schedule |
|---|---|
| E1 | random sampling from all images (baseline) |
| E2 | 30% on a random subset the size of easy+medium, then all. Control for "fewer distinct images early" |
| E3 | 30% easy+medium, then all |
| E4 | 30% medium+hard, then all (anti-curriculum) |
| E5 | 20% easy, 20% easy+medium, 60% all |

There are 5 seeds per experiment, 25 runs in total, about 340 s of training per run. The `last.pt` checkpoint is
evaluated on the val split.

## 3. Results of the controlled study

Mean ± std over 5 seeds (`reports/summary_by_experiment.csv`, `experiment_results.csv`):

| Exp | PQ | SQ | RQ | FP | FN | PQ easy | PQ medium | PQ hard |
|---|---|---|---|---|---|---|---|---|
| E1 | 0.3541 ± 0.0050 | 0.6337 | 0.5588 | 594 | 593 | 0.371 | 0.340 | 0.357 |
| E2 | 0.3514 ± 0.0043 | 0.6331 | 0.5550 | 632 | 585 | 0.371 | 0.340 | 0.349 |
| E3 | 0.3535 ± 0.0053 | 0.6325 | 0.5590 | 590 | 594 | 0.369 | 0.339 | 0.360 |
| E4 | 0.3476 ± 0.0046 | 0.6330 | 0.5492 | 647 | 590 | 0.359 | 0.334 | 0.356 |
| **E5** | **0.3578 ± 0.0052** | 0.6324 | **0.5657** | **568** | 590 | **0.373** | **0.345** | **0.361** |

Paired comparisons by seed (`reports/paired_comparisons.csv`; the bootstrap CI is computed over images on pooled predictions):

| Comparison | ΔPQ | wins / 5 | paired t p | Wilcoxon p | 95% bootstrap CI |
|---|---|---|---|---|---|
| E5 − E1 | +0.0037 | 5 | 0.022 | 0.0625 | [−0.0022, +0.0096] |
| E3 − E1 | −0.0006 | 2 | 0.89 | 1.0 | [−0.0060, +0.0051] |
| E2 − E1 | −0.0027 | 2 | 0.49 | 0.63 | [−0.0081, +0.0031] |
| E4 − E1 | −0.0064 | 0 | 0.0096 | 0.0625 | [−0.0119, −0.0007] |
| E3 − E2 | +0.0021 | 5 | 0.044 | 0.0625 | [−0.0035, +0.0078] |
| E5 − E3 | +0.0043 | 4 | 0.34 | 0.44 | [−0.0019, +0.0097] |

(0.0625 is the smallest Wilcoxon p-value possible with n = 5.)

**Interpretation.**
- **The three-stage easy→medium→all curriculum (E5) is the best schedule.** It beats the baseline on every seed. The
  gain is small (+0.004 PQ, about 1%), and almost all of it is recognition quality: fewer false positives
  (568 vs 594) at unchanged segmentation quality. It is consistent but **not robustly significant**. The image-level
  bootstrap CI includes 0, and the result would not survive a Holm correction over the 7 comparisons.
- **The two-stage curriculum (E3) is no better than random sampling.** It does beat its size-matched random-subset
  control E2 on all 5 seeds. So restricting the early pool by difficulty is better than restricting it at random,
  but neither beats training on everything.
- **The anti-curriculum (E4) reliably hurts.** It loses on 0/5 seeds, and its CI excludes 0. Starting on crowded,
  hard images produces more false positives later (647 vs 594).
- If each run's best intermediate checkpoint is chosen instead, all schedules land at ≈ 0.36 (0.359–0.365). The
  ranking is not stable to checkpoint choice, which is a reason to call the curriculum effect small
  (`reports/figures/val_curves_vs_updates.png`).

## 4. Post-study improvements (final submission model only)

These steps used only inference settings, more training budget, and seed ensembling. The E1–E5 study above was left
unchanged.

1. **Inference sweep** (`reports/infer_sweep/summary.csv`): imgsz ∈ {1024, 1280, 1536} × conf ∈ [0.10, 0.60] for
   the 5 E5 seeds. The best setting is imgsz 1536, conf 0.45: **PQ 0.399 ± 0.003** (mean over seeds), up from
   0.358 at the study protocol (1024 / 0.25). Higher resolution helps the thin, small filaments. A higher confidence
   threshold removes low-score false positives, which matters because PQ penalises FP and FN equally.
2. **Longer training at a higher resolution** (`configs/final_long_*.yaml`, `reports/infer_sweep_long`): the best
   result was PQ 0.377. This is worse than the short runs with tuned inference, so it was **not used**.
3. **5-seed mask-voting ensemble** (`src/evaluation/ensemble_sweep.py`, `reports/ensemble_sweep.csv`). Each model
   predicts at 1536 with a 0.2 confidence floor. Instances from different models are matched at IoU 0.5 and kept
   if at least 50% of the models agree. Masks are fused by pixel vote, then thresholded at conf 0.45 and passed through
   the standard overlap resolution. Result: **val PQ 0.417** (SQ 0.664, RQ 0.629). A vote fraction of 0.5 or 0.6
   gives the same score (0.4173 vs 0.4175).

**Caveat.** The inference and ensemble settings were selected on the same val split they are reported on, so
0.399 and 0.417 are optimistic. Only 3–4 parameters were tuned, and their effect is consistent across seeds,
so the bias should be small. The study numbers in §3 are not affected.

## 5. Final submission

`submissions/E5_ens5_1536.csv` comes from the 5 E5 models (`results/E5_s{0..4}/weights/last.pt`, copied to
`weights/final/`) at imgsz 1536, vote 0.5, conf 0.45. It has 1099 rows for 180 test images (176 with at least
one filament, 6.1 per image on average). It passes format validation with 0 errors: every test image is covered,
ids are unique, and the RLEs decode to 2048² masks. Metadata is in `submissions/E5_ens5_1536.json`.

**Leaderboard.** Public Kaggle leaderboard score: **0.70** (submitted 2026-11-14). A gap this large cannot be explained by tuning bias, which would push the other way. The likeliest cause is that the leaderboard scores predictions differently from our local per-annotator-record PQ (for example, against the best-matching or merged annotation instead of every annotator's set), or that the public test subset is easier. The local numbers are still the right basis for comparing schedules with each other, but they are not on the same scale as the leaderboard.

## 6. Qualitative analysis

`reports/figures/qualitative_val.png` shows easy, medium and hard val images. Common errors:
- Long filaments are split into pieces, or adjacent fragments are merged. The annotators themselves disagree on this.
- Faint, short filaments near the limb or in plage are missed.
- In crowded images the ensemble is conservative. It favours precision, which the metric rewards.

## 7. Limitations

- There is a single held-out fold (month-grouped). Fold-to-fold variance has not been measured.
- Curriculum effects are small compared with seed noise (±0.005). Five seeds are not enough to establish significance
  at this effect size.
- The difficulty proxy is the filament count only. Contrast, filament size and annotator agreement are not used.
- The public leaderboard score (0.70) is not on the same scale as local validation PQ (0.417). The scoring difference has not been pinned down, and the private score is not yet known.
