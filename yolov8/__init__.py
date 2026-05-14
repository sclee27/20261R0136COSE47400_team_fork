"""Slim YOLOv8 (detection-only) — easy-to-modify baseline.

Public top-level API:
    YOLOv8           — the detection model (n/s/m/l/x via ``scale=...``).
    v8DetectionLoss  — the official YOLOv8 loss (box CIoU + cls BCE + DFL).
    parse_model      — yaml → nn.Sequential (extend for new blocks).
    load_yaml        — read a model yaml relative to this package.
"""

from .loss import BboxLoss, DFLoss, v8DetectionLoss
from .model import YOLOv8, load_yaml, parse_model
from .tal import TaskAlignedAssigner, bbox2dist, dist2bbox, make_anchors

__all__ = [
    "BboxLoss",
    "DFLoss",
    "TaskAlignedAssigner",
    "YOLOv8",
    "bbox2dist",
    "dist2bbox",
    "load_yaml",
    "make_anchors",
    "parse_model",
    "v8DetectionLoss",
]
