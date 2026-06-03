"""Block: YAML config -> dataclass.

Mirrors the blocks (levels/jitter/stride/label/data) into dataclasses,
the same way a training config is structured.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class LevelsCfg:
    enabled: list           # level names to use (i.e. NOT dropped)
    upper_short: dict       # level name -> shorter-side upper bound (px)


@dataclass
class JitterCfg:
    scale: tuple            # (lo, hi) -- box_short = GT_short * U(lo, hi)
    log: bool               # True -> log-uniform scale
    pos_frac: float         # center offset = U(+/-pos_frac) * GT_size
    n_candidates: int       # number of candidate boxes per GT


@dataclass
class StrideCfg:
    strides: list
    n_short: int
    n_ratios: int
    sig_scale: float


@dataclass
class LabelCfg:
    iou_pos: float
    iou_neg: float
    stratify_bins: int
    n_per_bin: int


@dataclass
class DataCfg:
    ann_dir: str
    split: str
    max_images: int
    seed: int


@dataclass
class SamplingConfig:
    image_size: int
    mode: str               # gt_linked | stride
    levels: LevelsCfg
    jitter: JitterCfg
    stride: StrideCfg
    label: LabelCfg
    data: DataCfg


def load_config(path: str | Path) -> SamplingConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    j = d["jitter"]
    return SamplingConfig(
        image_size=int(d["image_size"]),
        mode=str(d["mode"]),
        levels=LevelsCfg(enabled=list(d["levels"]["enabled"]),
                         upper_short=dict(d["levels"]["upper_short"])),
        jitter=JitterCfg(scale=tuple(j["scale"]), log=bool(j.get("log", False)),
                         pos_frac=float(j["pos_frac"]), n_candidates=int(j["n_candidates"])),
        stride=StrideCfg(strides=list(d["stride"]["strides"]),
                         n_short=int(d["stride"]["n_short"]),
                         n_ratios=int(d["stride"]["n_ratios"]),
                         sig_scale=float(d["stride"]["sig_scale"])),
        label=LabelCfg(iou_pos=float(d["label"]["iou_pos"]),
                       iou_neg=float(d["label"]["iou_neg"]),
                       stratify_bins=int(d["label"]["stratify_bins"]),
                       n_per_bin=int(d["label"]["n_per_bin"])),
        data=DataCfg(ann_dir=str(d["data"].get("ann_dir", "data_sds/annotations")),
                     split=str(d["data"]["split"]),
                     max_images=int(d["data"]["max_images"]),
                     seed=int(d["data"]["seed"])),
    )
