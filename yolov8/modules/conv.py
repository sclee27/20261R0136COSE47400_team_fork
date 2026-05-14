"""Convolution building blocks for YOLOv8.

Ported verbatim from ultralytics/nn/modules/conv.py — only the pieces YOLOv8
detection actually uses are kept here so the file is easy to read and modify.

Public API:
    autopad(k, p=None, d=1) -> int | list[int]
    Conv(c1, c2, k=1, s=1, p=None, g=1, d=1, act=True)
    Concat(dimension=1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["autopad", "Conv", "Concat"]


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs. k=kernel, p=pad, d=dilation."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv2d + BatchNorm2d + SiLU activation.

    The default building block of YOLOv8. ``forward_fuse`` is used after BN is
    folded into the conv weights (see ``yolov8.model.fuse``).
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along a given dimension."""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        return torch.cat(x, self.d)
