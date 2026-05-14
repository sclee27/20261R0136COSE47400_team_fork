"""Training entry point for the SeaDronesSee ablation.

Trains one of the four model variants on SeaDronesSee v2 using ultralytics'
Trainer (which handles dataloading, augmentation, optimizer, EMA, mAP eval).

Key design points:
  * Pretrained COCO weights are loaded by handing the official ``yolov8m.pt``
    to ultralytics' ``YOLO.load()`` which transfers only the keys whose shapes
    match between checkpoint and the current cfg. The classification head and
    (for the P2 variant) any newly added layers are left at random init and
    get learned during finetuning — exactly the standard transfer-learning
    recipe.
  * Hyperparameters are pinned in code, not pulled from CLI, so all four runs
    are guaranteed identical except for the model cfg. That makes the
    ablation comparison clean.

Run on the GPU instance (after `nvidia-smi` shows your A100):
    .venv/bin/python experiments/train.py --model baseline
    .venv/bin/python experiments/train.py --model p2
    .venv/bin/python experiments/train.py --model sppf-k3
    .venv/bin/python experiments/train.py --model p2-sppf-k3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Model registry — maps a short CLI name to a cfg yaml path.
# All paths are relative to the experiments/ directory.
# ---------------------------------------------------------------------------
# File names include the scale letter ("m") because ultralytics parses the
# scale from the filename when one isn't passed explicitly in the cfg dict.
MODEL_CFGS = {
    "baseline":     "cfg/yolov8m.yaml",
    "p2":           "cfg/yolov8m-p2.yaml",
    "sppf-k3":      "cfg/yolov8m-sppf-k3.yaml",
    "p2-sppf-k3":   "cfg/yolov8m-p2-sppf-k3.yaml",
}


# ---------------------------------------------------------------------------
# Shared training hyperparameters.
# These are pinned so every variant trains under the same recipe — the only
# difference between runs is the architecture cfg. Any change here affects
# all runs equally, which is what we want for an ablation.
# ---------------------------------------------------------------------------
TRAIN_KW = dict(
    # data + duration
    epochs=100,           # early stopping will usually cut this to ~70-80
    patience=30,
    imgsz=640,
    batch=16,
    workers=6,
    cache="ram",          # 96 GiB RAM is plenty for SeaDronesSee
    device=0,             # single-GPU MIG slice on Elice

    # optimization
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=True,

    # loss gains (ultralytics defaults — match what we verified locally)
    box=7.5,
    cls=0.5,
    dfl=1.5,

    # augmentation (ultralytics defaults, with flipud=0 since aerial imagery
    # has a meaningful up-down orientation)
    mosaic=1.0,
    close_mosaic=10,
    mixup=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    fliplr=0.5,
    flipud=0.0,

    # bookkeeping
    amp=True,
    seed=0,
    deterministic=True,
    plots=True,
    save=True,
    save_period=-1,       # only best + last
    verbose=True,
)


def build_model(variant: str, weights: str | None, experiments_dir: Path) -> YOLO:
    """Build a YOLOv8m model with optional COCO-pretrained weight transfer.

    Args:
        variant: key into MODEL_CFGS.
        weights: path to a .pt checkpoint to transfer (typically yolov8m.pt).
            If None, the model is built from cfg with random init.
        experiments_dir: directory the cfg paths are resolved against.

    Returns:
        ultralytics YOLO wrapper, ready for .train().
    """
    cfg_path = experiments_dir / MODEL_CFGS[variant]
    if not cfg_path.exists():
        sys.exit(f"cfg not found: {cfg_path}")

    # Building from cfg returns an architecture with random init.
    # `.load(weights)` then intersects state dicts: keys with matching shape
    # are copied, everything else is left at the random init the cfg builder
    # produced. ultralytics logs how many tensors were transferred.
    model = YOLO(str(cfg_path))
    if weights:
        weights_path = Path(weights)
        if not weights_path.exists():
            sys.exit(f"pretrained weights not found: {weights_path}")
        print(f"[+] loading COCO-pretrained weights from {weights_path}")
        model = model.load(str(weights_path))
    else:
        print("[!] no pretrained weights provided — training from scratch")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a YOLOv8m variant on SeaDronesSee.")
    ap.add_argument("--model", required=True, choices=list(MODEL_CFGS),
                    help="which architecture variant to train.")
    ap.add_argument("--data", default="data/sds.yaml",
                    help="ultralytics dataset yaml (relative to experiments/).")
    ap.add_argument("--weights", default="weights/yolov8m.pt",
                    help="COCO-pretrained .pt to transfer (relative to experiments/). "
                         "Set to '' to train from scratch.")
    ap.add_argument("--name", default=None,
                    help="run name under runs/detect/. Defaults to --model value.")
    ap.add_argument("--project", default="runs",
                    help="parent directory for runs (relative to experiments/).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the default epoch count.")
    ap.add_argument("--batch", type=int, default=None,
                    help="override the default batch size.")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="override the default image size (default 640).")
    args = ap.parse_args()

    experiments_dir = Path(__file__).resolve().parent

    weights = str(experiments_dir / args.weights) if args.weights else None
    model = build_model(args.model, weights, experiments_dir)

    # Apply per-run overrides on top of the pinned defaults.
    train_kw = dict(TRAIN_KW)
    if args.epochs is not None:
        train_kw["epochs"] = args.epochs
    if args.batch is not None:
        train_kw["batch"] = args.batch
    if args.imgsz is not None:
        train_kw["imgsz"] = args.imgsz
    train_kw["data"] = str(experiments_dir / args.data)
    train_kw["project"] = str(experiments_dir / args.project)
    train_kw["name"] = args.name or args.model

    print(f"\n=== training {args.model} ===")
    print(f"    data    : {train_kw['data']}")
    print(f"    epochs  : {train_kw['epochs']}  imgsz: {train_kw['imgsz']}  batch: {train_kw['batch']}")
    print(f"    output  : {train_kw['project']}/{train_kw['name']}")

    model.train(**train_kw)

    print(f"\n=== {args.model} done ===")
    print(f"    best.pt -> {train_kw['project']}/{train_kw['name']}/weights/best.pt")


if __name__ == "__main__":
    main()
