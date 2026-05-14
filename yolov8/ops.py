"""Box ops and small numeric utilities used by model/loss.

Ported from ultralytics/utils/ops.py and ultralytics/utils/metrics.py — kept
intentionally tiny so the modeling math stays auditable.
"""

from __future__ import annotations

import math

import torch

__all__ = ["make_divisible", "xywh2xyxy", "xyxy2xywh", "bbox_iou"]


def make_divisible(x: float, divisor: int) -> int:
    """Return the smallest multiple of ``divisor`` that is >= ``x``."""
    return math.ceil(x / divisor) * divisor


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) -> (x1, y1, x2, y2). Last dim must be 4."""
    assert x.shape[-1] == 4, f"expected last dim = 4, got {x.shape}"
    y = torch.empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) -> (cx, cy, w, h). Last dim must be 4."""
    assert x.shape[-1] == 4, f"expected last dim = 4, got {x.shape}"
    y = torch.empty_like(x)
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2
    y[..., 1] = (y1 + y2) / 2
    y[..., 2] = x2 - x1
    y[..., 3] = y2 - y1
    return y


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Pairwise IoU / GIoU / DIoU / CIoU between matched boxes.

    The last dim of each input must be 4. All other dims must broadcast. Output
    shape is the broadcast shape with last dim 1. CIoU is the variant the
    YOLOv8 box-regression loss uses.
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw.pow(2) + ch.pow(2) + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
            if CIoU:
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou
