"""Multi-seed mask-voting ensemble submission (settings chosen by src.evaluation.ensemble_sweep on val).

  python -m src.inference.make_ensemble_submission --weights results/E5_s{0,1,2,3,4}/weights/last.pt \
      --imgsz 1536 --vote 0.5 --conf 0.45 --out submissions/E5_ens5_1536.csv
"""
import argparse
import json
import pickle
from pathlib import Path

import torch
import yaml

from src.evaluation.ensemble_sweep import fuse, threshold
from src.evaluation.evaluate import to_submission_df
from src.evaluation.infer_sweep import predict_raw
from src.inference.make_submission import ROOT, TEST, validate_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1536); ap.add_argument("--pre", type=float, default=0.2)
    ap.add_argument("--vote", type=float, default=0.5); ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--config", default="configs/base.yaml"); ap.add_argument("--device", default="0")
    a = ap.parse_args()
    torch.set_num_threads(16)
    infer = yaml.safe_load(open(ROOT / a.config))["infer"]
    paths = sorted(TEST.glob("*.jp*g"))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = out.parent / "cache"; cache_dir.mkdir(exist_ok=True)
    raws = []
    for w in a.weights:
        cache = cache_dir / f"test_{Path(w).parts[-3]}_{Path(w).stem}_{a.imgsz}.pkl"
        if cache.exists():
            raw = pickle.load(open(cache, "rb"))
        else:
            raw = predict_raw(w, paths, a.imgsz, infer, a.device, a.pre)
            pickle.dump(raw, open(cache, "wb"))
        raws.append(raw); print(f"predicted {w}", flush=True)
    preds = threshold(fuse(raws, pre=a.pre, vote_frac=a.vote), a.conf, infer)
    df = to_submission_df(preds)
    errors = validate_submission(df, [p.stem for p in paths])
    df.to_csv(out, index=False)
    n_per = df.filament_id.str.rsplit("_", n=1).str[0].value_counts()
    meta = dict(weights=a.weights, imgsz=a.imgsz, pre=a.pre, vote=a.vote, conf=a.conf, infer=infer,
                n_rows=len(df), n_test_images=len(paths), images_with_predictions=int(len(n_per)),
                mean_filaments_per_image=float(len(df) / len(paths)), validation_errors=errors)
    json.dump(meta, open(out.with_suffix(".json"), "w"), indent=2)
    print(json.dumps(meta, indent=1))
    if errors:
        raise SystemExit("submission validation FAILED")


if __name__ == "__main__":
    main()
