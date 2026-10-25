"""Aggregate all runs: master CSV, paired statistics, grouped bootstrap CIs, difficulty-proxy check, plots.

Usage: python -m src.evaluation.aggregate
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation import official_metric as om
from src.evaluation.evaluate import get_overlap_df_fast

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = ROOT / "reports/figures"
EXPS = ["E1", "E2", "E3", "E4", "E5"]
STRATEGY = {"E1": "random (all)", "E2": "random subset -> all", "E3": "easy+medium -> all",
            "E4": "medium+hard -> all", "E5": "easy -> easy+medium -> all"}
GROUPS = ["easy", "medium", "hard"]


def per_record(run):
    """Per-annotator-record TP-IoU sum, TP, FP, FN for the final checkpoint (cached)."""
    out = run / "eval/last/per_record.csv"
    if out.exists():
        return pd.read_csv(out)
    pred = pd.read_csv(run / "eval/last/val_predictions.csv")
    gt = pd.read_csv(ROOT / "data/processed/val_gt.csv")
    ov = get_overlap_df_fast(gt, pred)
    rows = []
    for r in ov.itertuples(index=False):
        if r.n_pred == 0 or r.n_gt == 0:
            rows.append((r.annotator_image, 0.0, 0, r.n_pred, r.n_gt)); continue
        hit = r.iou_matrix > 0.5
        rows.append((r.annotator_image, float(r.iou_matrix[hit].sum()), int(hit.sum()),
                     om.fp_count_hit(hit), om.fn_count_hit(hit)))
    df = pd.DataFrame(rows, columns=["record", "iou_sum", "tp", "fp", "fn"])
    df["stem"] = df.record.str.split("-", n=1).str[1]
    df.to_csv(out, index=False)
    return df


def pq_of(df):
    den = df.tp.sum() + 0.5 * df.fp.sum() + 0.5 * df.fn.sum()
    return df.iou_sum.sum() / den if den else 0.0


def collect():
    rows = []
    for run in sorted(RES.glob("E*_s*")):
        if not (run / "val_summary.json").exists():
            continue
        exp, seed = run.name.split("_s")
        s = json.load(open(run / "val_summary.json"))
        cur = json.load(open(run / "curriculum.json"))
        f, b = s["final"], s["best"]
        rows.append(dict(experiment=exp, seed=int(seed), fold=0, strategy=STRATEGY[exp], update_budget=cur["total_updates"],
                         model="yolo11m-seg", resolution=1024,
                         PQ=f["all"]["PQ"], SQ=f["all"]["SQ"], RQ=f["all"]["RQ"], TP=f["all"]["TP"], FP=f["all"]["FP"], FN=f["all"]["FN"],
                         PQ_easy=f["easy"]["PQ"], PQ_medium=f["medium"]["PQ"], PQ_hard=f["hard"]["PQ"],
                         PQ_rep_only=f["rep_only"]["PQ"], mean_tp_dice=f["all"]["mean_tp_dice"],
                         gt_fragmented=f["all"]["gt_fragmented"], pred_merging=f["all"]["pred_merging"],
                         best_ckpt=s["best_checkpoint"], PQ_best_ckpt=b["all"]["PQ"],
                         train_seconds=json.load(open(run / "train_time.json"))["train_seconds"],
                         infer_s_per_img=f["seconds_per_image"], run_dir=str(run.relative_to(ROOT))))
    return pd.DataFrame(rows).sort_values(["experiment", "seed"])


def grouped_bootstrap(a, b, imgs, n=2000, seed=0):
    """CI for PQ(a)-PQ(b): resample observation dates (all records of an image/date move together)."""
    date = imgs.set_index("stem").date
    a = a.assign(date=a.stem.map(date)); b = b.assign(date=b.stem.map(date))
    dates = a.date.unique()
    ga, gb = dict(list(a.groupby("date"))), dict(list(b.groupby("date")))
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n):
        smp = rng.choice(dates, len(dates))
        da = pd.concat([ga[d] for d in smp]); db = pd.concat([gb[d] for d in smp])
        diffs.append(pq_of(da) - pq_of(db))
    return np.percentile(diffs, [2.5, 97.5])


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(ROOT / "experiment_results.csv", index=False)
    imgs = pd.read_csv(ROOT / "data/processed/images.csv")
    out = {}

    # --- mean/std per experiment
    agg = df.groupby("experiment")[["PQ", "SQ", "RQ", "FP", "FN", "PQ_easy", "PQ_medium", "PQ_hard", "PQ_best_ckpt", "train_seconds"]].agg(["mean", "std", "count"])
    agg.round(4).to_csv(ROOT / "reports/summary_by_experiment.csv")
    out["by_experiment"] = {e: {k: [round(df[df.experiment == e][k].mean(), 4), round(df[df.experiment == e][k].std(), 4)]
                                for k in ["PQ", "SQ", "RQ", "PQ_easy", "PQ_medium", "PQ_hard", "FP", "FN"]} for e in df.experiment.unique()}

    # --- paired differences across seeds + pooled grouped bootstrap per seed
    pr = {r.run_dir: per_record(ROOT / r.run_dir) for r in df.itertuples()}
    pairs = []
    for a, b in [("E3", "E1"), ("E3", "E2"), ("E2", "E1"), ("E4", "E1"), ("E5", "E1"), ("E3", "E4"), ("E5", "E3")]:
        m = df[df.experiment == a].merge(df[df.experiment == b], on="seed", suffixes=("_a", "_b"))
        if not len(m):
            continue
        d = m.PQ_a - m.PQ_b
        t = stats.ttest_rel(m.PQ_a, m.PQ_b) if len(m) > 1 else None
        w = stats.wilcoxon(m.PQ_a, m.PQ_b) if len(m) > 4 else None
        # bootstrap on seed-averaged per-record stats (seeds paired, dates resampled)
        A = pd.concat([pr[r] for r in m.run_dir_a]).groupby(["record", "stem"], as_index=False).sum(numeric_only=True)
        B = pd.concat([pr[r] for r in m.run_dir_b]).groupby(["record", "stem"], as_index=False).sum(numeric_only=True)
        ci = grouped_bootstrap(A, B, imgs)
        pairs.append(dict(comparison=f"{a}-{b}", n_seeds=len(m), mean_diff=d.mean(), std_diff=d.std(), wins=int((d > 0).sum()),
                          paired_t_p=t.pvalue if t else np.nan, wilcoxon_p=w.pvalue if w else np.nan,
                          pooled_PQ_diff=pq_of(A) - pq_of(B), boot_ci_low=ci[0], boot_ci_high=ci[1]))
    pairs = pd.DataFrame(pairs).round(4)
    pairs.to_csv(ROOT / "reports/paired_comparisons.csv", index=False)

    # --- is filament count a valid difficulty proxy? per-image PQ of E1 (pooled seeds) vs d_i
    e1 = pd.concat([pr[r] for r in df[df.experiment == "E1"].run_dir]).groupby("stem", as_index=False).sum(numeric_only=True)
    e1["pq"] = e1.iou_sum / (e1.tp + 0.5 * e1.fp + 0.5 * e1.fn).replace(0, np.nan)
    e1 = e1.merge(imgs[["stem", "d_median", "difficulty", "disk_std", "mean_area", "agreement_pq"]], on="stem")
    proxy = {c: dict(zip(["spearman_rho", "p"], map(float, stats.spearmanr(e1[c], e1.pq, nan_policy="omit")))) for c in ["d_median", "disk_std", "mean_area", "agreement_pq"]}
    out["difficulty_proxy_vs_E1_image_PQ"] = proxy
    e1.to_csv(ROOT / "reports/per_image_pq_E1.csv", index=False)
    json.dump(out, open(ROOT / "reports/analysis.json", "w"), indent=2)

    # --- plots
    col = dict(zip(EXPS, plt.cm.tab10.colors))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for e in sorted(df.experiment.unique()):
        cs = [pd.read_csv(ROOT / r / "pq_curve.csv") for r in df[df.experiment == e].run_dir]
        c = pd.concat(cs).groupby("updates")[["PQ", "SQ", "RQ"]].agg(["mean", "std"])
        for a_, k in zip(ax, ["PQ", "SQ", "RQ"]):
            a_.plot(c.index, c[k]["mean"], "-o", ms=3, color=col[e], label=f"{e} ({STRATEGY[e]})")
            a_.fill_between(c.index, c[k]["mean"] - c[k]["std"], c[k]["mean"] + c[k]["std"], color=col[e], alpha=.12)
            a_.set_xlabel("optimizer updates"); a_.set_title(f"Validation {k} (mean ± sd over seeds)")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "val_curves_vs_updates.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    es = sorted(df.experiment.unique())
    for i, e in enumerate(es):
        s = df[df.experiment == e]
        ax[0].bar(i, s.PQ.mean(), yerr=s.PQ.std(), color=col[e], alpha=.7, capsize=4); ax[0].plot([i] * len(s), s.PQ, "k.", ms=5)
        for j, g in enumerate(GROUPS):
            ax[1].bar(j * 6 + i, s[f"PQ_{g}"].mean(), yerr=s[f"PQ_{g}"].std(), color=col[e], capsize=2, label=e if j == 0 else None)
        ax[2].bar(i - .2, s.FP.mean(), .4, yerr=s.FP.std(), color=col[e], hatch="//", capsize=2)
        ax[2].bar(i + .2, s.FN.mean(), .4, yerr=s.FN.std(), color=col[e], capsize=2)
        ax[3].bar(i, s.train_seconds.mean() / 60, color=col[e])
    for a_ in [ax[0], ax[2], ax[3]]:
        a_.set_xticks(range(len(es))); a_.set_xticklabels(es)
    ax[0].set_ylim(df.PQ.min() - .03, df.PQ.max() + .02); ax[0].set_title("Final val PQ (dots = seeds)")
    ax[1].set_xticks([2, 8, 14]); ax[1].set_xticklabels(GROUPS); ax[1].set_title("Final PQ by difficulty"); ax[1].legend(fontsize=7)
    ax[2].set_title("FP (hatched) / FN (solid)"); ax[3].set_title("Training time [min]")
    fig.tight_layout(); fig.savefig(FIG / "experiment_comparison.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(e1.d_median, e1.pq, s=10, alpha=.6); ax.set_xlabel("d_i (median filament count)"); ax.set_ylabel("per-image PQ (E1, pooled seeds)")
    ax.set_title(f"Spearman rho={proxy['d_median']['spearman_rho']:.2f}"); fig.tight_layout(); fig.savefig(FIG / "difficulty_proxy.png", dpi=120); plt.close(fig)

    print(agg[["PQ", "PQ_best_ckpt", "FP", "FN"]].round(4).to_string()); print(pairs.to_string()); print(json.dumps(proxy, indent=1))


if __name__ == "__main__":
    main()
