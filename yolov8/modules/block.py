"""YOLOv8 building blocks: Bottleneck, C2f, SPPF, DFL.

Ported verbatim from ultralytics/nn/modules/block.py — only the blocks YOLOv8
detection actually uses are kept here.

Mathematical equivalence with the official implementation is intentional:
* ``Bottleneck`` — two-conv residual block.
* ``C2f``       — split + n Bottlenecks + concat + 1x1.
* ``SPPF``      — 1x1 + 3 sequential MaxPools (k=5) + 1x1.
* ``DFL``       — softmax-then-weighted-sum used by ``Detect`` to decode
                  distribution-focal-loss outputs into 4 box distances.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv

__all__ = ["Bottleneck", "C2f", "SPPF", "DFL"]


class Bottleneck(nn.Module):
    """Standard YOLO bottleneck with optional residual connection."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP-style block with two convs + n Bottlenecks (YOLOv8 backbone / neck core)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels per branch
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast.

    Equivalent to the classic SPP(k=(5, 9, 13)) but implemented as three
    sequential MaxPool2d(k=5) operations sharing the same kernel.
    """

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class DFL(nn.Module):
    """Distribution Focal Loss integral module.

    Takes the predicted distance distribution of shape (B, 4*reg_max, A) and
    collapses it to (B, 4, A) via softmax over reg_max bins followed by a
    fixed linear combination with weights [0, 1, ..., reg_max-1].

    Reference: Generalized Focal Loss, https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
