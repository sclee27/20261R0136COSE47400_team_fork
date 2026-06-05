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

def stratify_indices_per_gt(
        iou_dom: np.ndarray,
        dom_idx: np.ndarray,
        num_gts: int,
        label_cfg,
        rng: np.random.Generator,
) -> np.ndarray:
    """Stratified sampling performed independently for each GT instance.

    Candidates are partitioned by which specific GT instance is their
    dominant GT (dom_idx value).  Inside each GT's group the IoU spectrum
    is bucketed and sampled with the same n_per_bin / stratify_bins logic
    as stratify_indices.

    This gives every individual object in the image its own balanced
    training signal regardless of how many candidates happened to be
    generated near it -- a dense cluster of GTs cannot drown out an
    isolated one.

    Args:
        iou_dom:   (M,) IoU of each candidate with its dominant GT.
        dom_idx:   (M,) dominant GT index for each candidate (0..num_gts-1).
                   Candidates with dom_idx == -1 (no GT) are skipped.
        num_gts:   total number of GT instances in this image.
        label_cfg: config with .stratify_bins and .n_per_bin.
        rng:       numpy random Generator for reproducible sampling.

    Returns:
        indices (K,) into the original candidate array, one flat array
        concatenating the per-GT selections.
    """
    chosen = []
    bins   = np.linspace(0.0, 1.0, label_cfg.stratify_bins + 1)

    for gt_id in range(num_gts):
        group_idx = np.where(dom_idx == gt_id)[0]   # candidates dominated by this GT
        if len(group_idx) == 0:
            continue

        group_iou = iou_dom[group_idx]

        
        bucket = np.clip(np.digitize(group_iou, bins) - 1,
                         0, label_cfg.stratify_bins - 1)

        for b in range(label_cfg.stratify_bins):
            bin_pos = np.where(bucket == b)[0]
            if len(bin_pos) == 0:
                continue
            k = min(label_cfg.n_per_bin, len(bin_pos))
            sampled_positions = rng.choice(bin_pos, size=k, replace=False)
            chosen.append(group_idx[sampled_positions])  # map back to global indices

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
