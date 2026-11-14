# Does Curriculum Learning Help Solar-Filament Instance Segmentation? A Controlled Study with YOLO11m-seg

*Draft competition report: Filament Segmentation Challenge 2026 (MAGFiLO 1.0)*

## Abstract

We test whether ordering training images from easy to hard improves instance segmentation of solar filaments in
full-disk H-alpha images. Five sampling schedules were trained with YOLO11m-seg under a strictly controlled protocol:
identical initialisation, update budget, augmentation and post-processing, with 5 seeds each. The schedules were
random sampling, a size-matched random subset, a two-stage curriculum, an anti-curriculum, and a three-stage
curriculum. The three-stage easy→medium→all curriculum gave the best Panoptic Quality (PQ 0.358 ± 0.005 vs 0.354 ± 0.005
for the baseline). It won on all 5 seeds, but the effect is within image-level bootstrap uncertainty. The
anti-curriculum was reliably worse (−0.006, 0/5 seeds). Inference resolution, a tuned confidence threshold and
5-seed mask-voting ensembling had a much larger effect: validation PQ rose to 0.417, above the measured
inter-annotator agreement of ≈ 0.34.

## 1. Introduction

Solar filaments are elongated, low-contrast absorption features. Automated catalogues need them segmented as separate
instances, and annotators disagree considerably about where one filament ends and the next begins. The challenge
scores dataset-level PQ at IoU > 0.5 against every annotator's record. Curriculum learning, which presents easier
examples first, is often reported to help with noisy or hard data. We ask whether it helps here once confounders
are controlled, in particular the number of optimizer updates and the "fewer distinct images early" effect.

## 2. Data and preprocessing

The training set contains 707 grayscale 2048×2048 images with 1154 annotator records (1–3 per image) and 8199
filament polygons. We converted polygons to RLE and YOLO polygon labels. Each image is trained on one representative
record, chosen as the record that agrees most with the other annotators. Evaluation always uses all records.
To prevent temporal leakage, images were grouped by observation month and one month-grouped fold (141 images,
218 records) was held out for validation. The other 566 images were used for training.

**Difficulty.** An image's difficulty is the median per-annotator filament count d. The tertile thresholds were
computed on training images only: easy d ≤ 5, medium ≤ 9, hard > 9. Crowded images have more adjacent and
fragmented filaments and more annotator disagreement.

## 3. Method

**Model and training.** YOLO11m-seg, COCO-pretrained, at input size 1024. SGD (lr 0.01, momentum 0.937), batch 16,
**1800 optimizer updates** for every schedule. Augmentation: flips, ±30° rotation, mild scale and brightness
changes. Mosaic, mixup and copy-paste were disabled so every sample comes from the active curriculum pool and
full-disk geometry is preserved.

**Schedules** (fractions of the update budget):
E1 random over all images ·
E2 30% on a random subset the size of easy+medium, then all ·
E3 30% easy+medium → all ·
E4 30% medium+hard → all (anti-curriculum) ·
E5 20% easy → 20% easy+medium → 60% all.
A custom sampler draws batches from the active pool so that each stage gets exactly its share of updates.

**Inference and post-processing** (identical for all schedules): conf 0.25, NMS IoU 0.5, full-resolution masks.
Pixels are assigned to the highest-confidence instance. Instances smaller than 30 px, or that lose more than half
their area to higher-scoring overlaps, are dropped. This produces the disjoint masks the submission format needs.

**Statistics.** 5 seeds per schedule. We report paired t and Wilcoxon tests across seeds and an image-level
bootstrap CI on the pooled PQ difference.

## 4. Results

### 4.1 Controlled comparison

| Schedule | PQ | SQ | RQ | FP | FN |
|---|---|---|---|---|---|
| E1 random | 0.354 ± 0.005 | 0.634 | 0.559 | 594 | 593 |
| E2 random subset | 0.351 ± 0.004 | 0.633 | 0.555 | 632 | 585 |
| E3 easy+med → all | 0.354 ± 0.005 | 0.633 | 0.559 | 590 | 594 |
| E4 med+hard → all | 0.348 ± 0.005 | 0.633 | 0.549 | 647 | 590 |
| **E5 easy → easy+med → all** | **0.358 ± 0.005** | 0.632 | **0.566** | **568** | 590 |

| Δ vs E1 | ΔPQ | seed wins | paired t p | 95% CI (bootstrap) |
|---|---|---|---|---|
| E5 | +0.0037 | 5/5 | 0.022 | [−0.002, +0.010] |
| E3 | −0.0006 | 2/5 | 0.89 | [−0.006, +0.005] |
| E2 | −0.0027 | 2/5 | 0.49 | [−0.008, +0.003] |
| E4 | −0.0064 | 0/5 | 0.010 | [−0.012, −0.001] |

Findings:
1. **The gradual three-stage curriculum gives a small, consistent gain.** E5 > E1 on every seed. The gain comes
   entirely from recognition quality (RQ): about 26 fewer false positives at the same segmentation quality. It is
   not significant after multiple-comparison correction, and the image-level CI includes zero.
2. **The coarse two-stage curriculum does nothing.** E3 ≈ E1. It does beat its size-matched random control E2
   on 5/5 seeds, so ordering by difficulty helps relative to an equally restricted random pool.
3. **Hard-first training hurts.** E4 is worse on every seed, and its CI excludes zero. It produces 9% more false
   positives than E1.
4. The gains are spread across easy, medium and hard images (E5 leads in all three groups), not concentrated
   on easy ones.

### 4.2 Final system

These choices were tuned on the validation split after the study, keeping the E5 models fixed:

| System | Val PQ |
|---|---|
| E5 single model, study protocol (1024, conf 0.25) | 0.358 |
| E5 longer training at higher resolution | 0.377 |
| E5 single model, imgsz 1536, conf 0.45 (mean of 5 seeds) | 0.399 ± 0.003 |
| **E5 × 5 seeds, mask-voting ensemble, 1536, conf 0.45** | **0.417** |

The ensemble matches instances across models at IoU 0.5 and keeps those predicted by at least half of the models.
Masks are fused by pixel vote, then the standard threshold and overlap resolution are applied. Higher input
resolution recovers thin and small filaments. The higher confidence threshold and the voting both trade recall
for precision, which pays off because PQ weights FP and FN equally and annotators disagree about faint filaments.
Because these settings were selected on the validation split, 0.417 is an optimistic estimate.

## 5. Discussion

Curriculum effects in this setting are about 1% PQ. This is comparable to seed-to-seed noise and roughly an order
of magnitude smaller than the effect of inference resolution or ensembling. Two observations suggest the
curriculum mainly acts as a regulariser on false positives early in training:
- the E5 benefit is entirely in RQ;
- the anti-curriculum fails in the mirror-image way, with more false positives.

The size-matched control (E2) matters. Without it, E3's gain over a restricted random pool could be mistaken for a
curriculum effect, when overall it only matches random sampling. Note also that the model scores above the
inter-annotator PQ (≈ 0.34) under the same metric. Part of this is because a model's consistent "consensus"
segmentation is penalised less than one annotator's idiosyncratic choices.

**Limitations.** Single held-out fold. Count-based difficulty proxy. 5 seeds is not enough for effects of this size.
Inference and ensemble settings were tuned on validation. No test-set (leaderboard) score is reported here.

## 6. Conclusion

A gradual easy→medium→all curriculum is a safe default. It costs nothing and never hurt in our runs, but its benefit
is small and not statistically established. Hard-first ordering should be avoided. For competition performance, input
resolution, the confidence threshold and seed ensembling matter far more. Our submission is a 5-seed ensemble of
E5 models at 1536 px with a validation PQ of 0.417.

## Reproducibility

Code, configs, per-run results (`experiment_results.csv`), statistics and figures are in the repository.
`notebooks/filament_curriculum_study.ipynb` regenerates every table from the saved outputs. Training uses
`scripts/run_queue.sh`, and the submission uses `src/inference/make_ensemble_submission.py`.
