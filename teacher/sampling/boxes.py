"""Block: box generation.

Two modes:
  - gt_linked : scale/position jitter around each GT (the proposed fix).
                Boxes tightly enclose the object.
  - stride    : drives the legacy anchor_box_generate_center_sample.py as-is
                (for comparison).
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Import the legacy stride sampler (teacher/anchor_box_generate_center_sample.py).
_TEACHER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TEACHER_DIR not in sys.path:
    sys.path.insert(0, _TEACHER_DIR)
import anchor_box_generate_center_sample as legacy  # noqa: E402


# ---------------------------------------------------------------------------
# Proposed fix: GT-linked jitter
# ---------------------------------------------------------------------------
def sample_boxes_gt_linked(gt_boxes: np.ndarray, jitter_cfg, image_size: int,
                           rng: np.random.Generator):
    """Generate n_candidates jitter boxes per GT.

    box_short = GT_short * scale,  center = GT_center + offset.
    Returns: boxes (M, 4) int32,  src_idx (M,) -- source GT index per box.
    """
    if len(gt_boxes) == 0:
        return np.zeros((0, 4), np.int32), np.zeros(0, int)

    w = (gt_boxes[:, 2] - gt_boxes[:, 0]).astype(float)
    h = (gt_boxes[:, 3] - gt_boxes[:, 1]).astype(float)
    cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0

    N = len(gt_boxes)
    n = jitter_cfg.n_candidates
    src = np.repeat(np.arange(N), n)
    M = N * n

    lo, hi = jitter_cfg.scale
    if jitter_cfg.log:
        s = np.exp(rng.uniform(np.log(lo), np.log(hi), M))
    else:
        s = rng.uniform(lo, hi, M)

    pf = jitter_cfg.pos_frac
    dx = rng.uniform(-pf, pf, M) * w[src]
    dy = rng.uniform(-pf, pf, M) * h[src]

    nw = np.maximum(w[src] * s, 1.0)
    nh = np.maximum(h[src] * s, 1.0)
    ncx = cx[src] + dx
    ncy = cy[src] + dy

    x1 = ncx - nw / 2.0
    y1 = ncy - nh / 2.0
    x2 = ncx + nw / 2.0
    y2 = ncy + nh / 2.0
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    boxes = np.clip(boxes, 0, image_size - 1).astype(np.int32)

    # Drop boxes whose width/height collapsed to 0 after edge clipping.
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep], src[keep]


def sample_boxes_gt_linked_v2(gt_boxes: np.ndarray, jitter_cfg, image_size: int,
                              rng: np.random.Generator):
    """Version 2: keep GT-linked size sampling but obtain centers from
    legacy center-sampling routines in `anchor_box_generate_center_sample.py`.

    For each GT we generate `n_candidates` scales (same as `sample_boxes_gt_linked`),
    then call `legacy.get_GToverlap_center_regions_SINGLE` with the candidate
    anchor shapes for that GT to obtain centers constrained to overlap regions.
    If the legacy sampler fails to produce a center for a candidate, we fall
    back to the original jittered center.

    Returns: boxes (M, 4) int32, src_idx (M,) -- source GT index per box.
    """
    if len(gt_boxes) == 0:
        return np.zeros((0, 4), np.int32), np.zeros(0, int)

    w = (gt_boxes[:, 2] - gt_boxes[:, 0]).astype(float)
    h = (gt_boxes[:, 3] - gt_boxes[:, 1]).astype(float)
    cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0

    N = len(gt_boxes)
    n = jitter_cfg.n_candidates
    src = np.repeat(np.arange(N), n)

    lo, hi = jitter_cfg.scale
    if jitter_cfg.log:
        s = np.exp(rng.uniform(np.log(lo), np.log(hi), N * n))
    else:
        s = rng.uniform(lo, hi, N * n)

    pf = jitter_cfg.pos_frac
    # per-candidate offsets (used as fallback if legacy sampling returns none)
    dx = rng.uniform(-pf, pf, N * n) * w[src]
    dy = rng.uniform(-pf, pf, N * n) * h[src]

    nw = np.maximum(w[src] * s, 1.0)
    nh = np.maximum(h[src] * s, 1.0)

    out_boxes = []
    out_src = []

    # For each GT, call legacy sampler with its candidate anchor shapes
    for i in range(N):
        # indices of candidates for this GT
        idxs = np.where(src == i)[0]
        if len(idxs) == 0:
            continue

        # anchor shapes expected as integers (w, h)
        anchors_int = np.stack([np.round(nw[idxs]).astype(np.int32),
                                 np.round(nh[idxs]).astype(np.int32)], axis=1)

        # legacy expects target coords as (T,4) int32; pass single GT
        target = gt_boxes[i:i+1].astype(np.int32)

        sampled_centers, no_sample_pairs = legacy.get_GToverlap_center_regions_SINGLE(
            target, anchors_int)

        # sampled_centers: (B, T, 2) where T=1 here
        for local_k, global_idx in enumerate(idxs):
            bw, bh = anchors_int[local_k]

            # fallback jittered center
            ncx = cx[i] + dx[global_idx]
            ncy = cy[i] + dy[global_idx]

            # legacy returns -1 for missing samples
            center = sampled_centers[local_k, 0]
            if center[0] < 0 or center[1] < 0:
                ccx, ccy = float(ncx), float(ncy)
            else:
                ccx, ccy = float(center[0]), float(center[1])

            x1 = ccx - bw / 2.0
            y1 = ccy - bh / 2.0
            x2 = ccx + bw / 2.0
            y2 = ccy + bh / 2.0

            out_boxes.append([x1, y1, x2, y2])
            out_src.append(i)

    if not out_boxes:
        return np.zeros((0, 4), np.int32), np.zeros(0, int)

    boxes = np.clip(np.array(out_boxes), 0, image_size - 1).astype(np.int32)
    src = np.array(out_src, dtype=int)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep], src[keep]


# ---------------------------------------------------------------------------
# Legacy: stride-based (anchor size tied to stride -- root cause in EDA2/3)
# ---------------------------------------------------------------------------
def sample_boxes_stride(gt_boxes: np.ndarray, stride_cfg, image_size: int):
    """Drive the legacy sampler as-is (multivariate variant) to produce boxes.

    For each stride, build anchor shapes with make_random_anchor_shapes, sample
    centers with get_GToverlap_center_regions_MULTIPLE_MULTIVARIATE, then recover
    boxes as (center +/- half-anchor).
    Returns: boxes (M, 4) int32, src_idx (M,) -- dominant GT is recomputed in labeling.
    """
    if len(gt_boxes) == 0:
        return np.zeros((0, 4), np.int32), np.zeros(0, int)

    legacy.SIG_SCALE = stride_cfg.sig_scale          # apply sig_scale from Block 4
    target = gt_boxes.astype(np.int32)

    out_boxes, out_src = [], []
    for stride in stride_cfg.strides:
        try:
            anchors, _ = legacy.make_random_anchor_shapes(
                stride, stride_cfg.n_short, stride_cfg.n_ratios)
        except ValueError:
            # Legacy bug: when a short side reaches image_size, IMAGE_SIZE/short == 1.0
            # -> triangular(left=1, mode=1, right=1) -> "left == right" crash.
            print(f"  [warn] stride={stride}: legacy sampler cannot form valid ratios "
                  f"(short side near {image_size}px) -> skip")
            continue
        anchors = anchors.astype(np.int32)           # (B, 2) = (w, h)
        samples = legacy.get_GToverlap_center_regions_MULTIPLE_MULTIVARIATE(target, anchors)
        for b_idx, pairs in samples.items():
            bw, bh = anchors[b_idx]
            for active_t_indices, center in pairs:
                ccx, ccy = float(center[0]), float(center[1])
                x1 = ccx - bw / 2.0
                y1 = ccy - bh / 2.0
                x2 = ccx + bw / 2.0
                y2 = ccy + bh / 2.0
                out_boxes.append([x1, y1, x2, y2])
                out_src.append(active_t_indices[0])

    if not out_boxes:
        return np.zeros((0, 4), np.int32), np.zeros(0, int)

    boxes = np.clip(np.array(out_boxes), 0, image_size - 1).astype(np.int32)
    src = np.array(out_src, dtype=int)
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[keep], src[keep]


def sample_boxes(gt_boxes, cfg, rng):
    """Dispatch on cfg.mode."""
    if cfg.mode == "gt_linked":
        return sample_boxes_gt_linked(gt_boxes, cfg.jitter, cfg.image_size, rng)
    elif cfg.mode == "stride":
        return sample_boxes_stride(gt_boxes, cfg.stride, cfg.image_size)
    raise ValueError(f"unknown mode: {cfg.mode}")
