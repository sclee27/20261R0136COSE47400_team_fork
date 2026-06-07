"""Block: training YAML config -> dataclasses.

This is the config layer for training ``teacher/bbox_evaluator.py`` on
SeaDronesSee. It reuses the shared sampling blocks (levels/jitter/stride/
label/data) from ``sampling.config`` and adds the training-specific blocks
(backbone / dataset / train / val).

Everything in ``configs/train.yaml`` is an editable knob. The sampler the
dataset uses is selected by the FRIENDLY ``mode`` field:

    original     -> stride sampler (legacy uniform anchors)
    jittering    -> GT-linked jitter sampler
    jittering_v2 -> GT-linked jitter sampler, center-overlap variant

The dataset dispatches on the friendly mode directly; the legacy
``SamplingConfig.mode`` ("gt_linked"/"stride") is filled in only for
completeness.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import yaml

# -- sys.path bootstrap --------------------------------------------------
# training/config.py lives in teacher/training/, but imports from
# sampling.* which lives in teacher/sampling/. Put the TEACHER dir (the
# parent of this file's dir) on the path so the import works standalone too.
_TEACHER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TEACHER_DIR not in sys.path:
    sys.path.insert(0, _TEACHER_DIR)

from sampling.config import (  # noqa: E402  (after sys.path bootstrap)
    DataCfg,
    JitterCfg,
    LabelCfg,
    LevelsCfg,
    SamplingConfig,
    StrideCfg,
)


# -- friendly mode -> legacy SamplingConfig.mode -------------------------
# NOTE: this map is BOOKKEEPING only -- it fills SamplingConfig.mode for
# completeness. The AUTHORITATIVE friendly->sampler dispatch lives in
# dataset.py:FRIENDLY_TO_SAMPLER (which distinguishes gt_linked vs gt_linked_v2).
# Here jittering_v2 collapses to "gt_linked" because the legacy mode field has
# no v2 value; do NOT read v2-ness off SamplingConfig.mode.
FRIENDLY_MODES = ("original", "jittering", "jittering_v2")
_FRIENDLY_TO_LEGACY = {
    "original": "stride",
    "jittering": "gt_linked",
    "jittering_v2": "gt_linked",
}


# -- training-specific blocks -------------------------------------------
@dataclass
class BackboneCfg:
    weights: str            # path to the SDS fine-tuned best.pt (frozen feats)
    scale: str              # yolov8 scale tag: n/s/m/l/x
    cfg: str                # yolov8 architecture yaml
    freeze: bool            # freeze backbone weights during teacher training


@dataclass
class DatasetCfg:
    images_dir: str         # images live at <images_dir>/<split>/<file_name>
    pad_value: int          # letterbox pad color (114 = yolo grey)
    stratify: bool          # balance pos boxes by IoU bins (label.stratify_*)
    num_classes: int        # FOREGROUND classes; evaluator gets +1 for bg


@dataclass
class ValCfg:
    split: str              # which split to validate on
    max_images: int         # cap val images (0 -> all)


@dataclass
class TrainCfg:
    epochs: int             # MAX number of training epochs
    patience: int           # early stop after N epochs w/o val-acc improvement (0 = off)
    batch_size: int         # images per step (1 = per-image, variable #boxes)
    lr: float               # base learning rate
    weight_decay: float     # L2 / decoupled weight decay
    optimizer: str          # adamw | sgd
    momentum: float         # SGD momentum (ignored by adamw)
    device: str             # auto | cpu | cuda | cuda:0
    amp: bool               # automatic mixed precision (cuda only)
    num_workers: int        # dataloader worker processes
    out_dir: str            # root dir for run outputs
    exp_name: str           # run name -> <out_dir>/<exp_name>
    log_every: int          # log every N steps
    seed: int               # global RNG seed


@dataclass
class TeacherTrainConfig:
    image_size: int
    mode: str               # friendly: "original" | "jittering" | "jittering_v2"
    sampling: SamplingConfig  # built from the shared blocks
    backbone: BackboneCfg
    dataset: DatasetCfg
    train: TrainCfg
    val: ValCfg


# -- overrides ----------------------------------------------------------
def apply_overrides(d: dict, overrides: list[str]) -> dict:
    """Apply ``dotted.key=value`` overrides onto a nested dict in place.

    The value is parsed with ``yaml.safe_load`` so ints/floats/bools/lists
    behave (e.g. "label.iou_pos=0.4", "train.lr=0.005", "mode=stride",
    "levels.enabled=[orig, p1]").
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be 'dotted.key=value', got: {item!r}")
        key, raw = item.split("=", 1)
        keys = key.strip().split(".")
        value = yaml.safe_load(raw)
        node = d
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
    return d


# -- loader -------------------------------------------------------------
def load_train_config(path, overrides: list[str] | None = None) -> TeacherTrainConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    if overrides:
        d = apply_overrides(d, overrides)

    # -- friendly mode validation ---------------------------------------
    mode = str(d["mode"])
    if mode not in FRIENDLY_MODES:
        raise ValueError(
            f"mode must be one of {FRIENDLY_MODES}, got {mode!r}"
        )

    # -- shared sampling blocks (mirror sampling/config.py:load_config) --
    j = d["jitter"]
    sampling = SamplingConfig(
        image_size=int(d["image_size"]),
        mode=_FRIENDLY_TO_LEGACY[mode],   # legacy value, for completeness only
        levels=LevelsCfg(enabled=list(d["levels"]["enabled"]),
                         upper_short=dict(d["levels"]["upper_short"])),
        jitter=JitterCfg(scale=tuple(j["scale"]), log=bool(j.get("log", False)),
                         pos_frac=float(j["pos_frac"]),
                         n_candidates=int(j["n_candidates"])),
        stride=StrideCfg(strides=list(d["stride"]["strides"]),
                         n_short=int(d["stride"]["n_short"]),
                         n_ratios=int(d["stride"]["n_ratios"]),
                         sig_scale=float(d["stride"]["sig_scale"])),
        label=LabelCfg(iou_pos=float(d["label"]["iou_pos"]),
                       iou_neg=float(d["label"]["iou_neg"]),
                       stratify_bins=int(d["label"]["stratify_bins"]),
                       n_per_bin=int(d["label"]["n_per_bin"])),
        # max_images: 0 means "all" -> keep 0 (load_gt_by_image treats
        # falsy max_images as all).
        data=DataCfg(ann_dir=str(d["data"].get("ann_dir", "data_sds/annotations")),
                     split=str(d["data"]["split"]),
                     max_images=int(d["data"]["max_images"]),
                     seed=int(d["data"]["seed"])),
    )

    # -- training-specific blocks ---------------------------------------
    b = d["backbone"]
    backbone = BackboneCfg(weights=str(b["weights"]), scale=str(b["scale"]),
                           cfg=str(b["cfg"]), freeze=bool(b["freeze"]))

    ds = d["dataset"]
    dataset = DatasetCfg(images_dir=str(ds["images_dir"]),
                         pad_value=int(ds["pad_value"]),
                         stratify=bool(ds["stratify"]),
                         num_classes=int(ds["num_classes"]))

    t = d["train"]
    train = TrainCfg(epochs=int(t["epochs"]), patience=int(t.get("patience", 0)),
                     batch_size=int(t["batch_size"]),
                     lr=float(t["lr"]), weight_decay=float(t["weight_decay"]),
                     optimizer=str(t["optimizer"]), momentum=float(t["momentum"]),
                     device=str(t["device"]), amp=bool(t["amp"]),
                     num_workers=int(t["num_workers"]), out_dir=str(t["out_dir"]),
                     exp_name=str(t["exp_name"]), log_every=int(t["log_every"]),
                     seed=int(t["seed"]))

    v = d["val"]
    val = ValCfg(split=str(v["split"]), max_images=int(v["max_images"]))

    return TeacherTrainConfig(
        image_size=int(d["image_size"]),
        mode=mode,                         # friendly mode
        sampling=sampling,
        backbone=backbone,
        dataset=dataset,
        train=train,
        val=val,
    )
