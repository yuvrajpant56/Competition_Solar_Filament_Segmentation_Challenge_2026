# Final summary

## Best model

**A 5-seed ensemble of E5 models (easy→easy+medium→all curriculum, YOLO11m-seg) at imgsz 1536, vote 0.5, conf 0.45.**

- Weights: `weights/final/E5_s{0..4}.pt` (copies of `results/E5_s*/weights/last.pt`; not tracked in git).
- Submission: `submissions/E5_ens5_1536.csv`. It has 1099 rows for 180 test images and passes validation with 0 errors.
- Validation PQ is 0.417 (SQ 0.664, RQ 0.629). The inference settings were tuned on this same split, so the figure is optimistic.
- **Public Kaggle leaderboard score: 0.70.** It is much higher than validation PQ, most likely because the leaderboard scores predictions differently from our local per-annotator PQ. The private score is not yet known.

Why this model:
- E5 had the best mean PQ in the controlled study.
- Resolution 1536 with conf 0.45 was the best single-model setting, consistently across all 5 seeds (0.399 ± 0.003).
- The 5-seed ensemble added another +0.018 on top of that.
- Longer training at higher resolution (0.377) was tested and rejected.

## Completed experiments (measured)

| Item | Status |
|---|---|
| EDA, annotation validation, conversion, month-grouped split, difficulty tertiles | done; 10/10 tests pass |
| E1–E5 × 5 seeds (25 runs, identical 1800-update budget) | done |
| Paired statistics (t, Wilcoxon, bootstrap) | done |
| Inference sweep (3 resolutions × 8–10 confidence thresholds × 5 seeds) | done |
| Long-budget E5 (1024 / 1536) | done, not used |
| Ensemble sweep, final submission | done |
| Qualitative figure, notebook, report, 4-page draft | done |

## Measured results (val, mean ± std over 5 seeds, study protocol)

E1 0.3541 ± 0.0050 · E2 0.3514 ± 0.0043 · E3 0.3535 ± 0.0053 · E4 0.3476 ± 0.0046 · **E5 0.3578 ± 0.0052**

- **E5 vs E1:** +0.0037. E5 won on 5/5 seeds (paired t p = 0.022), but the image-bootstrap CI [−0.002, +0.010] includes 0. The gain is small and consistent, but not established.
- **E4 vs E1:** −0.0064. E4 lost on 0/5 seeds, and the CI excludes 0. The anti-curriculum hurts.
- **E3 vs E1:** ≈ 0. E3 does beat its size-matched random control E2 on 5/5 seeds.

## Not done / proposed future work

- **Leaderboard/local metric gap:** public LB 0.70 vs val PQ 0.417. Pin down how Kaggle scores predictions and add a matching local scorer.
- **Cross-validation:** only fold 0 was used. Running all 5 month-grouped folds would measure fold variance and give an untuned validation estimate.
- **More seeds:** at least 10 per schedule are needed to resolve effects of about 0.004 PQ.
- **Better difficulty proxies:** contrast, filament size, annotator agreement.
- **Multi-annotator training targets:** currently one representative record is used per image.
- **Test-time augmentation and a larger backbone (YOLO11l/x):** untested.

## Where things are

- Report: `reports/REPORT.md`
- Competition draft: `reports/competition_report_draft.md`
- Notebook: `notebooks/filament_curriculum_study.ipynb` (executed)
- Figures: `reports/figures/`
- Per-run table: `experiment_results.csv`
