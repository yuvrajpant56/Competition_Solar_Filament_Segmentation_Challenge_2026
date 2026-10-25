"""Train YOLO11m-seg with an update-indexed curriculum schedule.

Usage:
  python -m src.training.train --exp E3 --seed 0            # new run -> results/E3_s0
  python -m src.training.train --exp E3 --seed 0 --resume   # resume an interrupted run
  python -m src.training.train --exp E1 --seed 0 --epochs 1 --name smoke --fraction-check
"""
import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.segment import SegmentationTrainer

from src.curriculum.sampler import CurriculumSampler
from src.curriculum.schedules import SCHEDULES, build_pools

ROOT = Path(__file__).resolve().parents[2]
RUN_CFG = {}  # filled by main(); read by the trainer (Ultralytics instantiates the trainer class itself)


class CurriculumTrainer(SegmentationTrainer):
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        imgs = pd.read_csv(ROOT / "data/processed/images.csv").set_index("stem")
        diff = [imgs.loc[Path(f).stem, "difficulty"] for f in dataset.im_files]
        pools = build_pools(diff)
        upe = math.ceil(len(dataset) / batch_size)
        total = upe * self.args.epochs
        sampler = CurriculumSampler(pools, SCHEDULES[RUN_CFG["exp"]], batch_size, upe, total, RUN_CFG["seed"])
        self.curriculum = dict(sampler=sampler, difficulty=np.array(diff), files=dataset.im_files,
                               updates_per_epoch=upe, total_updates=total,
                               pool_sizes={k: int(len(v)) for k, v in pools.items()})
        g = torch.Generator()
        g.manual_seed(RUN_CFG["seed"])
        return InfiniteDataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                                  num_workers=self.args.workers, prefetch_factor=4, pin_memory=True,
                                  collate_fn=dataset.collate_fn, worker_init_fn=seed_worker, generator=g,
                                  drop_last=False)


def on_train_start(trainer):
    c = trainer.curriculum
    if trainer.start_epoch > 0:
        c["sampler"].set_epoch(trainer.start_epoch)
    out = Path(trainer.save_dir)
    json.dump(dict(exp=RUN_CFG["exp"], seed=RUN_CFG["seed"], schedule=SCHEDULES[RUN_CFG["exp"]],
                   updates_per_epoch=c["updates_per_epoch"], total_updates=c["total_updates"],
                   stage_boundaries=c["sampler"].bounds, pool_sizes=c["pool_sizes"],
                   n_train_images=len(c["files"]), accumulate=trainer.accumulate),
              open(out / "curriculum.json", "w"), indent=2)
    assert trainer.accumulate == 1, "optimizer updates must equal batches (set nbs == batch)"


def on_train_epoch_end(trainer):
    """Persist exposure log incrementally (trimmed to completed updates)."""
    c = trainer.curriculum
    done = (trainer.epoch + 1) * c["updates_per_epoch"]
    log = np.array([r for r in c["sampler"].log if r[0] < done], dtype=np.int32)
    np.save(Path(trainer.save_dir) / "exposure_log.npy", log)


def on_train_end(trainer):
    c = trainer.curriculum
    log = np.load(Path(trainer.save_dir) / "exposure_log.npy")
    df = pd.DataFrame(log, columns=["update", "stage", "index"])
    df["difficulty"] = c["difficulty"][df["index"]]
    summ = df.groupby(["stage", "difficulty"]).size().unstack(fill_value=0)
    summ["unique_images"] = df.groupby("stage")["index"].nunique()
    summ.to_csv(Path(trainer.save_dir) / "exposure_summary.csv")


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=sorted(SCHEDULES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override budget (smoke tests only)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    cfg.pop("infer")
    if args.epochs:
        cfg["epochs"] = args.epochs
    name = args.name or f"{args.exp}_s{args.seed}"
    run_dir = ROOT / "results" / name
    RUN_CFG.update(exp=args.exp, seed=args.seed)
    cb = dict(on_train_start=on_train_start, on_train_epoch_end=on_train_epoch_end, on_train_end=on_train_end)

    if args.resume:
        model = YOLO(run_dir / "weights/last.pt")
        for k, f in cb.items():
            model.add_callback(k, f)
        model.train(trainer=CurriculumTrainer, resume=True)
    else:
        if run_dir.exists():
            raise SystemExit(f"{run_dir} exists; use --resume or another --name (results are never overwritten)")
        model = YOLO(str(ROOT / "checkpoints/pretrained" / cfg.pop("model")))
        for k, f in cb.items():
            model.add_callback(k, f)
        run_dir.mkdir(parents=True)
        json.dump(dict(cfg=cfg, exp=args.exp, seed=args.seed, git=git_rev(), torch=torch.__version__,
                       gpu=torch.cuda.get_device_name(int(args.device)), host=platform.node(),
                       split=json.load(open(ROOT / "data/processed/split_meta.json"))),
                  open(run_dir / "run_meta.json", "w"), indent=2)
        cfg["data"] = str(ROOT / cfg["data"])
        t0 = time.time()
        model.train(trainer=CurriculumTrainer, project=str(ROOT / "results"), name=name, exist_ok=True,
                    seed=args.seed, device=args.device, **cfg)
        json.dump(dict(train_seconds=time.time() - t0), open(run_dir / "train_time.json", "w"))


if __name__ == "__main__":
    main()
