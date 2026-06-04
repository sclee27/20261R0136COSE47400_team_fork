"""Block: model assembly for teacher-model training.

Combines two EXISTING pieces into one nn.Module:
    - YOLOv8Backbone    (teacher/backbone.py)       -- COCO-pretrained, frozen
    - YOLOBBoxEvaluator (teacher/bbox_evaluator.py) -- ROI-align bbox classifier

The backbone runs under torch.no_grad() (frozen feature extractor), and only
the evaluator heads receive gradients during training. The evaluator is built
with num_fg_classes + 1 outputs, where the extra channel is the background
class (index == num_fg_classes), matching labeling.label_boxes (bg=num_classes).

Level routing is kept consistent between the dataset and the evaluator: the
module-level `filter_to_enabled_levels` uses the SAME thresholds (upper_short)
that the evaluator's internal `_assign_levels` uses, so a box kept for level X
in the dataset is routed by the evaluator to head X. Boxes whose level is not
enabled are dropped before they ever reach the evaluator, so disabled heads
receive no gradient.

The runner puts teacher/ on sys.path; this module bootstraps the same path so
`import backbone`, `import bbox_evaluator`, and `from sampling.levels import ...`
resolve (see teacher/test_sampling.py for the pattern).
"""
from __future__ import annotations

import os
import sys

# -- sys.path bootstrap ----------------------------------------------------
# training/model.py lives in teacher/training/, so the teacher dir is the
# parent of this file's directory. Putting it on the path lets the flat
# `import backbone` / `import bbox_evaluator` / `from sampling...` resolve.
# We also add the REPO ROOT (parent of teacher/) because backbone.py does
# `from yolov8.model import ...`, and the yolov8 package lives at the root.
_TEACHER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_TEACHER_DIR)
for _p in (_TEACHER_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

import backbone as _backbone
import bbox_evaluator as _bbox_evaluator
from sampling.levels import assign_levels, enabled_mask

YOLOv8Backbone = _backbone.YOLOv8Backbone
YOLOBBoxEvaluator = _bbox_evaluator.YOLOBBoxEvaluator


def filter_to_enabled_levels(boxes_np, labels_np, enabled, upper_short):
    """Drop boxes whose assigned level is not in `enabled` (single source of truth).

    Args:
        boxes_np:    (A, 4) numpy array, xyxy at the model input scale.
        labels_np:   (A,) numpy array of integer labels (parallel to boxes).
        enabled:     list of enabled level names (subset of LEVEL_ORDER).
        upper_short: dict level-name -> shorter-side upper bound. MUST match the
                     evaluator's _assign_levels thresholds so kept boxes route to
                     their corresponding head.

    Returns:
        (boxes_np[mask], labels_np[mask]) -- the surviving boxes/labels.
        Empty input is returned unchanged.
    """
    # Nothing to filter -- avoid a level call on a zero-length array.
    if boxes_np is None or len(boxes_np) == 0:
        return boxes_np, labels_np

    level_ids = assign_levels(boxes_np, upper_short)   # (A,) int over LEVEL_ORDER
    mask = enabled_mask(level_ids, enabled)            # (A,) bool
    return boxes_np[mask], labels_np[mask]


class TeacherModel(nn.Module):
    """Frozen YOLOv8m backbone + trainable ROI-align bbox evaluator.

    forward(images, boxes) -> evaluator output dict:
        {"scores": (B, A, num_fg+1), "valid": (B, A) bool, "rejected": (B, A) bool}

    Only the evaluator heads are trainable; the backbone is a frozen feature
    extractor run under torch.no_grad().
    """

    def __init__(self, weights: str, scale: str, cfg: str, freeze: bool,
                 num_fg_classes: int, enabled_levels: list, upper_short: dict):
        super().__init__()

        # Frozen COCO-pretrained feature extractor (orig passthrough + P1..P5).
        self.backbone = YOLOv8Backbone.from_pretrained(
            weights, scale=scale, cfg=cfg, freeze=freeze,
        )
        self.backbone.eval()   # frozen -- keep BN/dropout in eval mode always

        # +1 output channel for the background class (index == num_fg_classes).
        self.evaluator = YOLOBBoxEvaluator(num_classes=num_fg_classes + 1)

        # Bookkeeping used by the training loop / dataset.
        self.num_fg_classes = num_fg_classes
        self.bg_index = num_fg_classes
        self.enabled_levels = enabled_levels
        self.upper_short = upper_short

    def forward(self, images: torch.Tensor, boxes: torch.Tensor) -> dict:
        """Args:
            images: (B, 3, 640, 640) float input.
            boxes:  (B, A, 4) float xyxy at the 640 input scale.
        Returns the evaluator output dict (scores/valid/rejected).
        """
        # Backbone is frozen: extract features without building its autograd graph.
        with torch.no_grad():
            feats = self.backbone(images)
        # Evaluator (trainable) reads orig/p1/p2/p3/p4 from feats; p5 is ignored.
        return self.evaluator(feats, boxes)

    @staticmethod
    def teacher_score(scores_logits: torch.Tensor, bg_index: int,
                      valid: torch.Tensor = None, eps: float = 1e-3) -> torch.Tensor:
        """Map raw class logits to an objectness-like score in (0, 1].

        score = 1 - P(background), where P comes from a softmax over the last
        dim. Clamped to (eps, 1]. Boxes flagged invalid (valid is False) are
        forced to eps. This is the future TAL hook value.

        Args:
            scores_logits: (..., num_fg+1) raw logits.
            bg_index:      index of the background class in the last dim.
            valid:         optional (...) bool; invalid boxes -> eps.
            eps:           lower clamp / invalid-box value.
        Returns:
            (...,) tensor of scores in (0, 1].
        """
        probs = torch.softmax(scores_logits, dim=-1)
        score = 1.0 - probs[..., bg_index]
        score = score.clamp(min=eps, max=1.0)
        if valid is not None:
            # Force invalid boxes to the floor value (eps).
            score = torch.where(valid, score, torch.full_like(score, eps))
        return score

    def trainable_parameters(self):
        """Parameters that should be optimized (evaluator heads only)."""
        return [p for p in self.evaluator.parameters() if p.requires_grad]
