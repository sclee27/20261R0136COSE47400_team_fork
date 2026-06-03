"""Block: IoU-based labeling + stratified sampling.

Turns the jitter box spectrum into a signal the teacher can learn:
    IoU >= iou_pos  -> positive (dominant GT class)
    IoU <  iou_neg  -> background
    in between      -> ignore (-1)

stratify: fill IoU buckets evenly so the good->bad monotonicity signal
is spread uniformly across the IoU range.
"""
from __future__ import annotations

import numpy as np

IGNORE = -1


def label_boxes(iou_dom: np.ndarray, dom_idx: np.ndarray, gt_cls: np.ndarray,
                label_cfg, num_classes: int) -> np.ndarray:
    """Return labels: -1=ignore, num_classes=background, otherwise class index."""
    bg = num_classes
    labels = np.full(len(iou_dom), IGNORE, dtype=int)
    pos = iou_dom >= label_cfg.iou_pos
    neg = iou_dom < label_cfg.iou_neg
    if len(gt_cls) > 0:
        labels[pos] = gt_cls[dom_idx[pos]]
    labels[neg] = bg
    return labels


def stratify_indices(iou_dom: np.ndarray, label_cfg, rng: np.random.Generator) -> np.ndarray:
    """Split IoU [0, 1] into stratify_bins buckets and sample n_per_bin from each.

    Returns: indices of the selected boxes (a balanced spectrum).
    """
    bins = np.linspace(0.0, 1.0, label_cfg.stratify_bins + 1)
    bucket = np.clip(np.digitize(iou_dom, bins) - 1, 0, label_cfg.stratify_bins - 1)
    chosen = []
    for b in range(label_cfg.stratify_bins):
        idx = np.where(bucket == b)[0]
        if len(idx) == 0:
            continue
        k = min(label_cfg.n_per_bin, len(idx))
        chosen.append(rng.choice(idx, size=k, replace=False))
    if not chosen:
        return np.zeros(0, int)
    return np.concatenate(chosen)


def label_summary(labels: np.ndarray, num_classes: int) -> dict:
    bg = num_classes
    return {
        "positive": int(np.sum((labels >= 0) & (labels < num_classes))),
        "background": int(np.sum(labels == bg)),
        "ignore": int(np.sum(labels == IGNORE)),
    }
