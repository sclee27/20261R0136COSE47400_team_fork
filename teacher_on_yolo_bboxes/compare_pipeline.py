"""
compare_pipeline.py
====================
Loads YOLOv8 + YOLOBBoxEvaluator from separate directories,
runs a single image through both, and returns pd_scores / teacher_scores
in the same shape/space that TaskAlignedAssigner_With_Teacher uses.

Usage
-----
    from compare_pipeline import build_pipeline, run_image

    pipeline = build_pipeline(
        yolo_pkg_dir   = "/path/to/yolov8_package_dir",   # dir that contains __init__.py, model.py …
        teacher_dir    = "/path/to/bbox_evaluator_dir",   # dir that contains bbox_evaluator.py
        yolo_ckpt      = "/path/to/yolo_checkpoint.pt",
        teacher_ckpt   = "/path/to/teacher_checkpoint.pt",
        yolo_scale     = "m",
        yolo_nc        = 80,
        teacher_nc     = 80,
        device         = "cpu",
    )

    results = run_image(pipeline, image_path="image.jpg")
    # results keys:
    #   pd_scores      : (A, C)  float  – student sigmoid probs
    #   teacher_scores : (A, C)  float  – teacher softmax probs  (0 for rejected)
    #   teacher_valid  : (A,)    bool   – True = scored by teacher
    #   teacher_rejected:(A,)    bool   – True = bad aspect ratio / size
    #   pd_bboxes      : (A, 4)  float  – xyxy in image pixel coords
    #   image_size     : (H, W)  int
"""

from __future__ import annotations

import sys
import importlib
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Dataclass returned by run_image
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    pd_scores:        torch.Tensor   # (A, C)
    teacher_scores:   torch.Tensor   # (A, C)
    teacher_valid:    torch.Tensor   # (A,) bool
    teacher_rejected: torch.Tensor   # (A,) bool
    pd_bboxes:        torch.Tensor   # (A, 4)  xyxy image-pixel coords
    image_size:       tuple[int, int] # (H, W)
    anchor_count:     int             # A  (8400 for 640×640)
    # GT fields — populated when gt_info is passed to run_image()
    gt_bboxes:        torch.Tensor | None = None  # (N, 4) xyxy in 640-space
    gt_labels:        torch.Tensor | None = None  # (N,)   long class indices
    gt_mask:          torch.Tensor | None = None  # (A,)   bool: anchor center inside any GT box


# ---------------------------------------------------------------------------
# Dynamic importer  (handles files from different directories)
# ---------------------------------------------------------------------------

def _import_from_dir(module_name: str, directory: str | Path):
    """Import a .py file by path without polluting sys.modules permanently."""
    directory = Path(directory).resolve()
    spec = importlib.util.spec_from_file_location(
        module_name, directory / f"{module_name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod          # needed for intra-package imports
    spec.loader.exec_module(mod)
    return mod


def _import_package(pkg_dir: str | Path):
    """Add pkg_dir to sys.path and import the package via its __init__.py."""
    pkg_dir = Path(pkg_dir).resolve()
    pkg_name = pkg_dir.name
    if str(pkg_dir.parent) not in sys.path:
        sys.path.insert(0, str(pkg_dir.parent))
    return importlib.import_module(pkg_name)


# ---------------------------------------------------------------------------
# Feature-map hooks for the YOLO backbone
# ---------------------------------------------------------------------------

class _FeatureHook:
    """
    Registers forward hooks on named YOLO backbone layers to capture
    intermediate feature maps needed by the teacher evaluator.

    YOLOv8m backbone layer indices (identical for yolov8.yaml and yolov8m-p2.yaml):
        0  : Conv  (stride 2)  → P1,  48ch,  H/2   ← hook here
        1  : Conv  (stride 2)
        2  : C2f               → P2,  96ch,  H/4   ← hook here
        3  : Conv  (stride 2)
        4  : C2f               → P3,  192ch, H/8   ← hook here
        5  : Conv  (stride 2)
        6  : C2f               → P4,  384ch, H/16  ← hook here
        7  : Conv  (stride 2)
        8  : C2f               → P5 (not used by teacher)
        9  : SPPF

    Override via layer_map={level: layer_idx} if your yaml differs.
    """

    _DEFAULT_LAYER_MAP = {
        'p1': 0,   # Conv  P1/2   → 48ch,  H/2   (first stride-2 Conv)
        'p2': 2,   # C2f   P2/4   → 96ch,  H/4   (C2f after Conv@0,Conv@1)
        'p3': 4,   # C2f   P3/8   → 192ch, H/8   (C2f after Conv@3)
        'p4': 6,   # C2f   P4/16  → 384ch, H/16  (C2f after Conv@5)
        # Same indices for both yolov8.yaml and yolov8m-p2.yaml (identical backbones):
        # 0=Conv(P1), 1=Conv(P2), 2=C2f(P2), 3=Conv(P3), 4=C2f(P3),
        # 5=Conv(P4), 6=C2f(P4), 7=Conv(P5), 8=C2f(P5), 9=SPPF
        # Channels match YOLOBBoxEvaluator.LEVEL_IN_CHANNELS: p1=48,p2=96,p3=192,p4=384
    }

    def __init__(self, model: nn.Module, layer_map: dict[str, int] | None = None):
        self.layer_map = layer_map or self._DEFAULT_LAYER_MAP
        self._handles = []
        self._captured: dict[str, torch.Tensor] = {}

        seq = model.model          # nn.Sequential of all layers
        for level, idx in self.layer_map.items():
            handle = seq[idx].register_forward_hook(self._make_hook(level))
            self._handles.append(handle)

    def _make_hook(self, level: str):
        def hook(module, input, output):
            # output can be a tensor or a tuple; grab first tensor
            t = output[0] if isinstance(output, (list, tuple)) else output
            self._captured[level] = t.detach()
        return hook

    def get(self) -> dict[str, torch.Tensor]:
        return dict(self._captured)

    def clear(self):
        self._captured.clear()

    def remove(self):
        for h in self._handles:
            h.remove()


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

@dataclass
class Pipeline:
    yolo_model:      nn.Module
    teacher_model:   nn.Module
    feature_hook:    _FeatureHook
    device:          torch.device
    nc:              int
    stride_tensor:   torch.Tensor   # (A, 1) in image-pixel scale
    anchor_points:   torch.Tensor   # (A, 2) in image-pixel scale
    teacher_temperature: float
    bg_index:        int | None     # index of background class in teacher logits; None = no bg class


def build_pipeline(
    yolo_pkg_dir:    str | Path,
    teacher_dir:     str | Path,
    yolo_ckpt:       str | Path,
    teacher_ckpt:    str | Path,
    yolo_scale:      str = "m",
    yolo_cfg:        str | Path | None = None,
    yolo_nc:         int = 80,
    teacher_nc:      int | None = None,
    device:          str | torch.device = "cpu",
    teacher_temperature: float = 1.0,
    yolo_layer_map:  dict[str, int] | None = None,
) -> Pipeline:
    """
    Build and return the full pipeline.

    Parameters
    ----------
    yolo_pkg_dir   : directory containing the yolov8 package (__init__.py, model.py …)
    teacher_dir    : directory containing bbox_evaluator.py
    yolo_ckpt      : checkpoint path for YOLOv8 (state_dict or full ckpt dict)
    teacher_ckpt   : checkpoint path for YOLOBBoxEvaluator
    yolo_scale     : "n"/"s"/"m"/"l"/"x"
    yolo_cfg       : path to model yaml (e.g. yolov8m-p2.yaml); None = default yolov8.yaml
    yolo_nc        : number of classes YOLO was trained on
    teacher_nc     : number of fg classes teacher was trained on; None = read from checkpoint
    device         : torch device string or object
    teacher_temperature : softmax temperature applied to teacher logits
    yolo_layer_map : override default backbone layer indices for feature hooks
                     e.g. {"p1": 0, "p2": 2, "p3": 4, "p4": 6}
    """
    device = torch.device(device)

    # ── 1. Import yolo package ─────────────────────────────────────────────
    yolo_pkg = _import_package(yolo_pkg_dir)
    YOLOv8    = yolo_pkg.YOLOv8
    make_anchors = yolo_pkg.make_anchors

    # ── 2. Import teacher ──────────────────────────────────────────────────
    teacher_mod = _import_from_dir("bbox_evaluator", teacher_dir)
    YOLOBBoxEvaluator = teacher_mod.YOLOBBoxEvaluator

    # ── 3. Build YOLO ──────────────────────────────────────────────────────
    cfg_arg = str(Path(yolo_cfg).resolve()) if yolo_cfg is not None else "cfg/yolov8.yaml"
    print(f"[pipeline] Building YOLOv8{yolo_scale} (nc={yolo_nc}, cfg={Path(cfg_arg).name}) …")
    yolo = YOLOv8(cfg=cfg_arg, scale=yolo_scale, nc=yolo_nc, verbose=False).to(device)

    ckpt = torch.load(yolo_ckpt, map_location=device, weights_only=False)
    # Ultralytics saves the trained weights under "ema" (EMA copy), not "model".
    # Fall back chain: ema → model → state_dict → model_state_dict → bare dict
    if isinstance(ckpt, dict):
        if "ema" in ckpt and ckpt["ema"] is not None:
            sd = ckpt["ema"].float().state_dict()
        elif "model" in ckpt and ckpt["model"] is not None:
            sd = ckpt["model"].float().state_dict()
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt   # assume it's already a raw state_dict
    else:
        sd = ckpt.float().state_dict()   # live nn.Module passed directly
    missing, unexpected = yolo.load_state_dict(sd, strict=False)
    print(f"  YOLO loaded — missing:{len(missing)}  unexpected:{len(unexpected)}")
    yolo.eval()

    # ── 4. Pre-compute anchors for 640×640 ─────────────────────────────────
    #    We need a dummy forward in train mode to get feats (list of feature maps)
    #    then run make_anchors on them.
    yolo.model[-1].training = True          # force dict output from Detect
    with torch.no_grad():
        dummy_out = yolo(torch.zeros(1, 3, 640, 640, device=device))
    yolo.model[-1].training = False

    raw = dummy_out if isinstance(dummy_out, dict) else dummy_out[1]
    feats = raw["feats"]                    # list of feature tensors

    stride_tensor_raw = yolo.stride.to(device)   # (num_levels,)
    anchor_points_grid, stride_tensor_full = make_anchors(feats, stride_tensor_raw, 0.5)
    # anchor_points_grid : (A, 2)  in feature-grid units
    # stride_tensor_full : (A, 1)
    anchor_points_img = anchor_points_grid * stride_tensor_full   # image-pixel units

    # ── 5. Register feature hooks ──────────────────────────────────────────
    hook = _FeatureHook(yolo, layer_map=yolo_layer_map)

    # ── 6. Build teacher ───────────────────────────────────────────────────
    t_ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)

    # Read metadata saved by training/loop.py:save_checkpoint so we reconstruct
    # the evaluator with exactly the right num_classes and enabled_levels.
    ckpt_nc      = None
    ckpt_enabled = None
    ckpt_bg      = None
    if isinstance(t_ckpt, dict):
        ckpt_nc      = t_ckpt.get("num_classes")       # num_fg + 1 (includes bg class)
        ckpt_enabled = t_ckpt.get("enabled_levels")    # e.g. ['orig', 'p1', 'p2']
        ckpt_bg      = t_ckpt.get("bg_index")          # index of background class
        if "evaluator" in t_ckpt:
            t_sd = t_ckpt["evaluator"]
        elif "state_dict" in t_ckpt:
            t_sd = t_ckpt["state_dict"]
        elif "model_state_dict" in t_ckpt:
            t_sd = t_ckpt["model_state_dict"]
        else:
            t_sd = t_ckpt
    else:
        t_sd = t_ckpt.float().state_dict()

    # Resolve num_classes: checkpoint metadata > caller-supplied teacher_nc
    if ckpt_nc is not None:
        teacher_nc_actual = ckpt_nc
    elif teacher_nc is not None:
        teacher_nc_actual = teacher_nc
    else:
        raise ValueError(
            "Cannot determine teacher num_classes: checkpoint has no 'num_classes' key "
            "and teacher_nc was not provided."
        )
    bg_index = ckpt_bg  # None if checkpoint predates bg_index saving

    print(f"[pipeline] Building YOLOBBoxEvaluator "
          f"(nc={teacher_nc_actual}, enabled={ckpt_enabled}) …")
    teacher = YOLOBBoxEvaluator(num_classes=teacher_nc_actual,
                                enabled_levels=ckpt_enabled).to(device)

    missing_t, unexpected_t = teacher.load_state_dict(t_sd, strict=False)
    print(f"  Teacher loaded — missing:{len(missing_t)}  unexpected:{len(unexpected_t)}")
    teacher.eval()

    return Pipeline(
        yolo_model=yolo,
        teacher_model=teacher,
        feature_hook=hook,
        device=device,
        nc=yolo_nc,
        stride_tensor=stride_tensor_full,
        anchor_points=anchor_points_img,
        teacher_temperature=teacher_temperature,
        bg_index=bg_index,
    )


# ---------------------------------------------------------------------------
# GT mask helper  (mirrors TAL's _select_candidates_in_gts logic)
# ---------------------------------------------------------------------------

def _compute_gt_mask(
    anchor_points: torch.Tensor,   # (A, 2)  image-pixel coords
    gt_bboxes:     torch.Tensor,   # (N, 4)  xyxy image-pixel coords (640-space)
    device:        torch.device,
) -> torch.Tensor:                 # (A,)    bool — True if anchor center inside any GT box
    """Replicate the TAL `mask_in_gts` criterion used during teacher training.

    An anchor is considered "GT-overlapping" when its center point falls
    strictly inside at least one GT bounding box (positive gap on all 4 sides).
    This is the same criterion as TaskAlignedAssigner_With_Teacher._get_pos_mask.
    """
    if gt_bboxes.shape[0] == 0:
        return torch.zeros(anchor_points.shape[0], dtype=torch.bool, device=device)
    lt = gt_bboxes[:, :2].unsqueeze(1)                         # (N, 1, 2)
    rb = gt_bboxes[:, 2:].unsqueeze(1)                         # (N, 1, 2)
    xy = anchor_points.unsqueeze(0)                             # (1, A, 2)
    deltas = torch.cat([xy - lt, rb - xy], dim=-1)             # (N, A, 4)
    inside = deltas.amin(dim=-1) > 1e-9                        # (N, A) bool
    return inside.any(dim=0)                                    # (A,)   bool


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------

def _load_image(image_path: str | Path | np.ndarray, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Load an image from path or numpy array.
    Returns:
        img_tensor : (1, 3, 640, 640) float32 in [0, 1]
        orig_size  : (H, W) original image size
    """
    try:
        from PIL import Image
        import torchvision.transforms.functional as TF
    except ImportError:
        raise ImportError("pip install Pillow torchvision")

    if isinstance(image_path, np.ndarray):
        img = Image.fromarray(image_path)
    else:
        img = Image.open(image_path).convert("RGB")

    orig_size = (img.height, img.width)
    img_resized = img.resize((640, 640), Image.BILINEAR)
    img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)   # (1,3,640,640)
    return img_tensor, orig_size


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_image(
    pipeline:   Pipeline,
    image_path: str | Path | np.ndarray,
    gt_info:    dict | None = None,
) -> PipelineResult:
    """
    Run YOLO + teacher on a single image.

    Parameters
    ----------
    pipeline   : built by build_pipeline()
    image_path : file path, numpy RGB array, or Path
    gt_info    : optional dict with keys
                   "gt_bboxes" – array/tensor (N, 4) xyxy in 640×640 image space
                   "gt_labels" – array/tensor (N,)   integer class indices
                 When provided the teacher only evaluates anchors whose centers
                 fall inside a GT box (same criterion as TAL training), and the
                 result carries gt_bboxes / gt_labels / gt_mask fields.

    Returns a PipelineResult with pd_scores and teacher_scores both shaped (A, C).

    pd_scores      — student sigmoid probabilities, same as what TAL receives
    teacher_scores — teacher softmax probabilities (temp-scaled), 0 for rejected
    """
    p = pipeline
    device = p.device

    # ── Load + preprocess image ────────────────────────────────────────────
    img_tensor, orig_size = _load_image(image_path, device)

    # ── YOLO forward ───────────────────────────────────────────────────────
    p.feature_hook.clear()

    # Force train-mode Detect output (dict with boxes/scores/feats)
    p.yolo_model.model[-1].training = True
    raw = p.yolo_model(img_tensor)
    p.yolo_model.model[-1].training = False

    raw_dict = raw if isinstance(raw, dict) else raw[1]

    # pd_scores: (B, C, A) → (B, A, C) → (A, C) sigmoid  — exactly what TAL sees
    pd_scores_raw = raw_dict["scores"]                          # (1, C, A)
    pd_scores = pd_scores_raw.permute(0, 2, 1).squeeze(0).sigmoid()  # (A, C)

    # pd_bboxes: decoded xyxy in image-pixel coords
    # raw_dict["boxes"] is the raw DFL distribution (1, 4*reg_max, A)
    # We need the decoded bboxes — replicate loss.py's bbox_decode
    pred_distri = raw_dict["boxes"].permute(0, 2, 1).contiguous()  # (1, A, 4*reg_max)
    m = p.yolo_model.model[-1]   # Detect head
    reg_max = m.reg_max
    proj = torch.arange(reg_max, dtype=torch.float, device=device)
    b, a, c = pred_distri.shape
    pred_dist_soft = pred_distri.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_distri.dtype))

    # dist2bbox: anchor_points_grid needed (feature units), not image-pixel
    anchor_grid = p.anchor_points / p.stride_tensor   # (A, 2) feature-grid units
    lt, rb = pred_dist_soft.chunk(2, dim=-1)
    x1y1 = anchor_grid.unsqueeze(0) - lt
    x2y2 = anchor_grid.unsqueeze(0) + rb
    pd_bboxes_feat = torch.cat([x1y1, x2y2], dim=-1)              # (1, A, 4)
    pd_bboxes_img = pd_bboxes_feat * p.stride_tensor.unsqueeze(0) # (1, A, 4) image-pixel
    pd_bboxes = pd_bboxes_img.squeeze(0)                           # (A, 4)

    # ── Collect backbone feature maps ──────────────────────────────────────
    backbone_feats = p.feature_hook.get()
    # 'orig' is just the raw image tensor
    feature_maps = {'orig': img_tensor, **backbone_feats}

    # Validate required levels are present
    missing_levels = [k for k in ['p1', 'p2', 'p3', 'p4'] if k not in feature_maps]
    if missing_levels:
        raise RuntimeError(
            f"Feature maps missing: {missing_levels}. "
            f"Check yolo_layer_map — got hooks on: {list(backbone_feats.keys())}"
        )

    # ── GT mask (anchor-center-in-GT, same criterion as TAL training) ─────────
    if gt_info is not None:
        gt_bboxes_t = torch.as_tensor(
            np.asarray(gt_info["gt_bboxes"], dtype=np.float32), device=device
        )
        gt_labels_t = torch.as_tensor(
            np.asarray(gt_info["gt_labels"], dtype=np.int64), device=device
        )
        gt_mask = _compute_gt_mask(p.anchor_points, gt_bboxes_t, device)  # (A,)
        gt_mask_batched = gt_mask.unsqueeze(0)                             # (1, A)
    else:
        gt_bboxes_t = gt_labels_t = gt_mask = gt_mask_batched = None

    # ── Teacher forward ────────────────────────────────────────────────────
    # pd_bboxes needs shape (B, A, 4) for the evaluator
    pd_bboxes_batched = pd_bboxes.unsqueeze(0)                    # (1, A, 4)

    teacher_out = p.teacher_model(feature_maps, pd_bboxes_batched, gt_mask=gt_mask_batched)

    # teacher logits → temperature softmax → (A, num_classes)
    teacher_logits = teacher_out["scores"].squeeze(0)              # (A, num_classes)
    teacher_scores = (teacher_logits / p.teacher_temperature).softmax(dim=-1)

    # Drop background class so teacher_scores aligns with pd_scores (A, num_fg).
    # The training loop always appends bg as the last class (bg_index = num_fg_classes),
    # so we can strip it here and renormalize over the fg columns.
    if p.bg_index is not None:
        num_total = teacher_scores.shape[-1]
        fg_cols = [i for i in range(num_total) if i != p.bg_index]
        teacher_scores = teacher_scores[:, fg_cols]                # (A, num_fg)
        teacher_scores = teacher_scores / teacher_scores.sum(-1, keepdim=True).clamp(min=1e-6)

    teacher_valid    = teacher_out["valid"].squeeze(0)             # (A,)
    teacher_rejected = teacher_out["rejected"].squeeze(0)          # (A,)

    # Rejected boxes: fall back to student scores (matches tal.py behavior)
    teacher_scores = teacher_scores.clone()
    teacher_scores[teacher_rejected] = pd_scores[teacher_rejected]

    return PipelineResult(
        pd_scores=pd_scores,
        teacher_scores=teacher_scores,
        teacher_valid=teacher_valid,
        teacher_rejected=teacher_rejected,
        pd_bboxes=pd_bboxes,
        image_size=orig_size,
        anchor_count=pd_scores.shape[0],
        gt_bboxes=gt_bboxes_t,
        gt_labels=gt_labels_t,
        gt_mask=gt_mask,
    )