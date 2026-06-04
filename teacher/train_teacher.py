#!/usr/bin/env python3
"""Runner: train the teacher bbox evaluator on SeaDronesSee, driven by a YAML config.

Everything is a knob in teacher/configs/train.yaml (mode / levels / jitter /
label / backbone / train). Switch the sampler with --mode, drop P-levels via
levels.enabled, tune IoU thresholds via label.iou_pos / label.iou_neg, etc.

Usage:
  .venv/bin/python teacher/train_teacher.py                                   # default cfg
  .venv/bin/python teacher/train_teacher.py --mode jittering_v2 --epochs 20
  .venv/bin/python teacher/train_teacher.py --config teacher/configs/train_smoke.yaml
  .venv/bin/python teacher/train_teacher.py --set label.iou_pos=0.4 --set train.lr=0.005
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

_HERE = os.path.dirname(os.path.abspath(__file__))            # teacher/
sys.path.insert(0, _HERE)                                     # for sampling.* / training.* / backbone
sys.path.insert(0, os.path.dirname(_HERE))                    # repo root, for `import yolov8` (backbone)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from training.config import load_train_config  # noqa: E402
from training.dataset import TeacherBBoxDataset, collate_fn  # noqa: E402
from training.model import TeacherModel  # noqa: E402
from training.loop import resolve_device, train  # noqa: E402


def build_loader(cfg, split, max_images, shuffle):
    """Build a (dataset, DataLoader) pair for one split using the resolved config."""
    ds = TeacherBBoxDataset(
        sampling_cfg=cfg.sampling,
        images_dir=cfg.dataset.images_dir,
        split=split,
        max_images=max_images,
        mode=cfg.mode,
        num_fg_classes=cfg.dataset.num_classes,
        stratify=cfg.dataset.stratify,
        enabled_levels=cfg.sampling.levels.enabled,
        upper_short=cfg.sampling.levels.upper_short,
        pad_value=cfg.dataset.pad_value,
        seed=cfg.train.seed,
    )
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle,
                        num_workers=cfg.train.num_workers, collate_fn=collate_fn)
    return ds, loader


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--config", default=os.path.join(here, "configs", "train.yaml"))
    ap.add_argument("--mode", choices=["original", "jittering", "jittering_v2"], default=None)
    ap.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:0")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-images", type=int, default=None,
                    help="cap TRAIN images (0 = all); val cap is separate (val.max_images)")
    ap.add_argument("--exp-name", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="dotted override, e.g. --set label.iou_pos=0.4 (repeatable)")
    args = ap.parse_args()

    # Convenience flags are translated into dotted overrides (applied before
    # the dataclasses are built), so they compose with --set.
    overrides = list(args.overrides)
    if args.mode is not None:
        overrides.append(f"mode={args.mode}")
    if args.device is not None:
        overrides.append(f"train.device={args.device}")
    if args.epochs is not None:
        overrides.append(f"train.epochs={args.epochs}")
    if args.batch_size is not None:
        overrides.append(f"train.batch_size={args.batch_size}")
    if args.max_images is not None:
        overrides.append(f"data.max_images={args.max_images}")
    if args.exp_name is not None:
        overrides.append(f"train.exp_name={args.exp_name}")

    cfg = load_train_config(args.config, overrides)
    cfg_dict = asdict(cfg)

    # Pre-flight: fail fast with a clear message if the backbone checkpoint is
    # missing (otherwise the error surfaces deep inside from_pretrained).
    if not os.path.exists(cfg.backbone.weights):
        raise FileNotFoundError(
            f"backbone weights not found: {cfg.backbone.weights}\n"
            f"  set backbone.weights in {args.config} to your SDS fine-tuned best.pt "
            f"(the shipped default is a placeholder COCO yolov8m.pt).")

    device = resolve_device(cfg.train.device)
    print("=" * 72)
    print(f"teacher training  mode={cfg.mode}  enabled={cfg.sampling.levels.enabled}")
    print(f"  backbone={cfg.backbone.weights}  freeze={cfg.backbone.freeze}")
    print(f"  device={device}  optimizer={cfg.train.optimizer}  lr={cfg.train.lr}  "
          f"epochs={cfg.train.epochs}  batch={cfg.train.batch_size}")
    print(f"  label: iou_pos={cfg.sampling.label.iou_pos} iou_neg={cfg.sampling.label.iou_neg}  "
          f"jitter scale={cfg.sampling.jitter.scale} n_cand={cfg.sampling.jitter.n_candidates}")
    print("=" * 72)

    train_ds, train_loader = build_loader(
        cfg, cfg.sampling.data.split, cfg.sampling.data.max_images, shuffle=True)
    val_ds, val_loader = build_loader(
        cfg, cfg.val.split, cfg.val.max_images, shuffle=False)
    print(f"train images={len(train_ds)}  val images={len(val_ds)}")

    model = TeacherModel(
        weights=cfg.backbone.weights, scale=cfg.backbone.scale, cfg=cfg.backbone.cfg,
        freeze=cfg.backbone.freeze, num_fg_classes=cfg.dataset.num_classes,
        enabled_levels=cfg.sampling.levels.enabled,
        upper_short=cfg.sampling.levels.upper_short,
    )
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"trainable params (evaluator heads): {n_train:,}")

    train(model, train_loader, val_loader, cfg, cfg_dict)


if __name__ == "__main__":
    main()
