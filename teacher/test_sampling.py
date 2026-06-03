#!/usr/bin/env python3
"""Runner that tests two things, driven by a YAML config.

  TEST 1  Necessity of P3/P4/P5 -- how real GT distributes across levels (sampler-independent)
  TEST 2  Jitter sample quality  -- how well generated boxes enclose objects (IoU / coverage / level load)

Usage:
  .venv/bin/python teacher/test_sampling.py                         # default cfg
  .venv/bin/python teacher/test_sampling.py --mode stride           # compare with legacy
  .venv/bin/python teacher/test_sampling.py --config teacher/configs/sampling.yaml --plot
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # put teacher/ on the path

from sampling.config import load_config
from sampling.data import load_gt_by_image, NUM_CLASSES
from sampling.boxes import sample_boxes
from sampling.levels import assign_levels, level_counts, enabled_mask, LEVEL_ORDER
from sampling.metrics import iou_and_dominant
from sampling.labeling import label_boxes, label_summary, stratify_indices


def _bar(frac: float, width: int = 30) -> str:
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def _print_level_table(counts: dict, total: int, enabled: list, title: str):
    print(f"\n{title}")
    print(f"  {'level':<6}{'count':>10}{'pct':>8}   {'enabled':<8} distribution")
    for name in LEVEL_ORDER:
        c = counts[name]
        pct = c / total if total else 0.0
        flag = "ON" if name in enabled else "drop"
        print(f"  {name:<6}{c:>10}{pct*100:>7.1f}%   {flag:<8} {_bar(pct)}")


def run(cfg, plot_path: str | None):
    rng = np.random.default_rng(cfg.data.seed)
    images = load_gt_by_image(cfg.data.split, cfg.data.max_images, cfg.data.seed,
                              cfg.image_size, cfg.data.ann_dir)
    n_gt = sum(len(im["boxes"]) for im in images)
    print("=" * 72)
    print(f"split={cfg.data.split}  images={len(images)}  GT boxes={n_gt}  mode={cfg.mode}")
    print("=" * 72)

    # -- TEST 1: real GT level distribution (P3/P4/P5 necessity) -------------
    gt_level_ids = []
    for im in images:
        if len(im["boxes"]):
            gt_level_ids.append(assign_levels(im["boxes"], cfg.levels.upper_short))
    gt_level_ids = np.concatenate(gt_level_ids) if gt_level_ids else np.zeros(0, int)
    gt_counts = level_counts(gt_level_ids)
    _print_level_table(gt_counts, len(gt_level_ids), cfg.levels.enabled,
                       "[TEST 1] real GT level distribution  -> if P3/P4/P5 hold no objects, dropping is justified")

    # -- TEST 2: sampler box quality ----------------------------------------
    all_iou, all_coverage, box_level_ids = [], [], []
    all_labels = []
    n_boxes = 0
    n_strat = 0
    for im in images:
        boxes, _src = sample_boxes(im["boxes"], cfg, rng)
        if len(boxes) == 0:
            continue
        n_boxes += len(boxes)
        iou_dom, dom_idx, coverage = iou_and_dominant(boxes, im["boxes"])
        lvl = assign_levels(boxes, cfg.levels.upper_short)
        labels = label_boxes(iou_dom, dom_idx, im["cls"], cfg.label, NUM_CLASSES)
        all_iou.append(iou_dom)
        all_coverage.append(coverage)
        box_level_ids.append(lvl)
        all_labels.append(labels)
        n_strat += len(stratify_indices(iou_dom, cfg.label, rng))

    if n_boxes == 0:
        print("\n[TEST 2] no boxes were generated.")
        return

    iou = np.concatenate(all_iou)
    coverage = np.concatenate(all_coverage)
    box_level_ids = np.concatenate(box_level_ids)
    labels = np.concatenate(all_labels)
    box_counts = level_counts(box_level_ids)

    print(f"\n[TEST 2] sampler box quality  (mode={cfg.mode}, {n_boxes:,} boxes total)")
    print(f"  IoU      median {np.median(iou):.3f}   IoU>=0.5 {np.mean(iou>=0.5)*100:5.1f}%   "
          f"IoU<0.2 {np.mean(iou<0.2)*100:5.1f}%")
    print(f"  coverage  median {np.median(coverage):.3f}   coverage>=0.3 {np.mean(coverage>=0.3)*100:5.1f}%")
    print(f"  (ref) the legacy stride sampler had IoU<0.2 ~99%, coverage median 0.00 in EDA2")

    _print_level_table(box_counts, n_boxes, cfg.levels.enabled,
                       "  sampler box level load  -> for gt_linked it should concentrate on orig/p1/p2")

    enab = enabled_mask(box_level_ids, cfg.levels.enabled)
    print(f"\n  boxes inside enabled levels {tuple(cfg.levels.enabled)}: "
          f"{enab.sum():,} / {n_boxes:,} ({enab.mean()*100:.1f}%)")

    lab = label_summary(labels, NUM_CLASSES)
    print(f"  labeling: positive {lab['positive']:,}  background {lab['background']:,}  "
          f"ignore {lab['ignore']:,}")
    print(f"  stratified balanced-sample size: {n_strat:,} "
          f"(bins={cfg.label.stratify_bins}, n_per_bin={cfg.label.n_per_bin})")

    if plot_path:
        _plot(gt_counts, len(gt_level_ids), box_counts, n_boxes, iou, coverage, cfg, plot_path)


def _plot(gt_counts, gt_total, box_counts, box_total, iou, coverage, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(LEVEL_ORDER))

    g = [gt_counts[n] / gt_total * 100 if gt_total else 0 for n in LEVEL_ORDER]
    b = [box_counts[n] / box_total * 100 if box_total else 0 for n in LEVEL_ORDER]
    ax[0].bar(x, g, color="#3b7dd8")
    ax[0].set_xticks(x); ax[0].set_xticklabels(LEVEL_ORDER)
    ax[0].set_title("TEST1: real GT level dist (%)"); ax[0].set_ylabel("%")

    ax[1].bar(x, b, color="#d8623b")
    ax[1].set_xticks(x); ax[1].set_xticklabels(LEVEL_ORDER)
    ax[1].set_title(f"TEST2: sampler box level load (%) [{cfg.mode}]")

    ax[2].hist(iou, bins=20, range=(0, 1), color="#5db846", alpha=0.8, label="IoU")
    ax[2].hist(coverage, bins=20, range=(0, 1), color="#b86dd6", alpha=0.5, label="coverage")
    ax[2].axvline(cfg.label.iou_pos, color="k", ls="--", lw=1)
    ax[2].axvline(cfg.label.iou_neg, color="gray", ls="--", lw=1)
    ax[2].set_title("TEST2: IoU / coverage"); ax[2].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\n[plot] saved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--config", default=os.path.join(here, "configs", "sampling.yaml"))
    ap.add_argument("--mode", choices=["gt_linked", "stride"], default=None,
                    help="override cfg.mode (for comparison)")
    ap.add_argument("--split", default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--plot", nargs="?", const="auto", default=None,
                    help="png output path (auto-named if no value given)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mode:
        cfg.mode = args.mode
    if args.split:
        cfg.data.split = args.split
    if args.max_images is not None:
        cfg.data.max_images = args.max_images

    plot_path = None
    if args.plot:
        plot_path = (os.path.join(here, f"sampling_test_{cfg.mode}.png")
                     if args.plot == "auto" else args.plot)

    run(cfg, plot_path)


if __name__ == "__main__":
    main()
