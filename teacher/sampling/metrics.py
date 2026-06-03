"""Block: geometric metrics (IoU / intersection / coverage).

All boxes are (N, 4) = [x1, y1, x2, y2].
"""
from __future__ import annotations

import numpy as np


def box_area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def pairwise_inter(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Intersection area of A(M, 4) x B(N, 4) -> (M, N)."""
    ax1, ay1, ax2, ay2 = A[:, 0][:, None], A[:, 1][:, None], A[:, 2][:, None], A[:, 3][:, None]
    bx1, by1, bx2, by2 = B[:, 0][None], B[:, 1][None], B[:, 2][None], B[:, 3][None]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    return iw * ih


def iou_and_dominant(boxes: np.ndarray, gt_boxes: np.ndarray):
    """For each box, return the IoU with its dominant (most-overlapping) GT,
    that GT's index, and the coverage (= intersection / box area).

    Returns:
        iou_dom  (M,)  : IoU with the dominant GT
        dom_idx  (M,)  : dominant GT index
        coverage (M,)  : fraction of the box covered by the object
    """
    if len(gt_boxes) == 0:
        m = len(boxes)
        return np.zeros(m), np.full(m, -1), np.zeros(m)
    inter = pairwise_inter(boxes, gt_boxes)            # (M, N)
    ba = box_area(boxes)                               # (M,)
    ga = box_area(gt_boxes)                            # (N,)
    union = ba[:, None] + ga[None, :] - inter
    iou = inter / np.maximum(union, 1e-9)
    dom = iou.argmax(1)
    rows = np.arange(len(boxes))
    iou_dom = iou[rows, dom]
    coverage = inter[rows, dom] / np.maximum(ba, 1e-9)
    return iou_dom, dom, coverage
