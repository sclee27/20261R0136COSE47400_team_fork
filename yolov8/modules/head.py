"""YOLOv8 detection head.

Ported from ultralytics/nn/modules/head.py with ``legacy=True`` hardcoded:
that is the variant the published yolov8{n,s,m,l,x}.pt weights were trained
with (DWConv-based head only appears in YOLO11+). Anything related to
end2end / Segment / Pose / OBB / world / YOLOE has been removed.

Output convention (matches the official ``DetectionModel``):

* ``training=True``                 → ``dict(boxes=..., scores=..., feats=feats)``
* ``training=False`` (inference)    → ``(decoded, raw_dict)``
  where ``decoded`` has shape ``(B, 4 + nc, A)`` and contains xywh + sigmoid(scores).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..tal import dist2bbox, make_anchors
from .block import DFL
from .conv import Conv

__all__ = ["Detect"]


class Detect(nn.Module):
    """Anchor-free, DFL-based YOLOv8 detection head."""

    # Class-level flags kept for parity with the official implementation. They
    # are only consulted by the inference path; training uses the dict output.
    dynamic = False  # force grid reconstruction every call
    export = False  # set True to drop the (decoded, raw) tuple and return only decoded
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc: int = 80, reg_max: int = 16, ch: tuple = ()):
        """Build a YOLOv8 detect head.

        Args:
            nc: number of classes.
            reg_max: DFL bin count (16 in the official cfg).
            ch: tuple of channel sizes from the 3 feature levels feeding the head.
        """
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.zeros(self.nl)  # filled in by the model builder

        # box-regression branch: two 3x3 convs + 1x1 to (4*reg_max) channels.
        # ch[0]//4 ensures cv2 hidden channels scale with the model width.
        c2 = max(16, ch[0] // 4, reg_max * 4)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for x in ch
        )

        # classification branch: two 3x3 convs + 1x1 to nc channels.
        # Hidden channels are min(c, max(ch[0], min(nc, 100))) — same formula as upstream.
        c3 = max(ch[0], min(nc, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)) for x in ch
        )

        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

    # ------------------------------------------------------------------ heads

    def _forward_head(self, x: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        bs = x[0].shape[0]
        boxes = torch.cat([self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([self.cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return {"boxes": boxes, "scores": scores, "feats": x}

    def forward(self, x: list[torch.Tensor]):
        preds = self._forward_head(x)
        if self.training:
            return preds
        return self._inference(preds)

    # ------------------------------------------------------------------ infer

    def _inference(self, preds: dict[str, torch.Tensor]):
        """Decode raw head outputs into ``(decoded, preds)`` for inference."""
        shape = preds["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(preds["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = dist2bbox(self.dfl(preds["boxes"]), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, preds["scores"].sigmoid()), 1)
        return y if self.export else (y, preds)

    # ------------------------------------------------------------------ init

    def bias_init(self):
        """Initialize Detect biases. Requires ``self.stride`` to be set."""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 2.0  # box branch bias prior
            # cls prior: ~0.01 objects per cell at 640x640 with nc=80
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
