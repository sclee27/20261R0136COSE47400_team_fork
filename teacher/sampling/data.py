"""Block: SeaDronesSee GT loading + letterbox(640) conversion.

Self-contained: parses the COCO-JSON annotations directly, with no dependency
on files outside teacher/ and no hardcoded absolute paths. The annotation dir
comes from the config (data.ann_dir) and can be overridden per-machine via the
SDS_ANN_DIR environment variable.

Image pixels are not needed -- the level/jitter tests work on GT boxes alone.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

# Project root = two levels up from teacher/sampling/data.py
_ROOT = Path(__file__).resolve().parents[2]

# Fixed class order (0..4). Background is set to num_classes (= 5).
CLASS_NAMES = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

SPLIT_FILES = {"train": "instances_train.json", "val": "instances_val.json"}


def resolve_ann_dir(ann_dir: str) -> Path:
    """Resolve the annotation dir.

    Priority: SDS_ANN_DIR env var > the given ann_dir. Relative paths resolve
    from the project root; absolute paths are used as-is.
    """
    p = Path(os.environ.get("SDS_ANN_DIR", ann_dir))
    return p if p.is_absolute() else (_ROOT / p)


def letterbox_boxes(xywh: np.ndarray, img_w: int, img_h: int, size: int = 640) -> np.ndarray:
    """COCO [x, y, w, h] (native resolution) -> [x1, y1, x2, y2] in the letterbox 640 frame."""
    r = min(size / img_w, size / img_h)
    pad_x = (size - img_w * r) / 2.0
    pad_y = (size - img_h * r) / 2.0
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = x * r + pad_x
    y1 = y * r + pad_y
    x2 = (x + w) * r + pad_x
    y2 = (y + h) * r + pad_y
    out = np.stack([x1, y1, x2, y2], axis=1)
    return np.clip(out, 0, size - 1)


def load_gt_by_image(split: str = "train", max_images: int = 500, seed: int = 0,
                     image_size: int = 640,
                     ann_dir: str = "data_sds/annotations") -> list[dict]:
    """Load per-image GT (COCO JSON) converted to the 640 frame.

    Returns: [{ "boxes": (N, 4) int32 x1y1x2y2, "cls": (N,) int }, ...]
    """
    path = resolve_ann_dir(ann_dir) / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"annotation file not found: {path}\n"
            f"  set data.ann_dir in the config, or export SDS_ANN_DIR=/path/to/annotations")

    with open(path) as f:
        coco = json.load(f)

    # categories -> id2name (drop the 'ignored' class)
    id2name = {c["id"]: c["name"] for c in coco["categories"] if c["name"] != "ignored"}
    # images -> id -> (w, h)
    img_wh = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}

    by_img: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        name = id2name.get(a["category_id"])
        if name in CLASS_TO_IDX and a["image_id"] in img_wh:
            by_img[a["image_id"]].append((a["bbox"], CLASS_TO_IDX[name]))

    img_ids = sorted(by_img.keys())
    rng = np.random.default_rng(seed)
    if max_images and len(img_ids) > max_images:
        img_ids = sorted(rng.choice(img_ids, size=max_images, replace=False).tolist())

    out = []
    for iid in img_ids:
        recs = by_img[iid]
        xywh = np.array([b for b, _ in recs], dtype=float)
        iw, ih = img_wh[iid]
        boxes = letterbox_boxes(xywh, iw, ih, image_size).astype(np.int32)
        cls = np.array([c for _, c in recs], dtype=int)
        # Drop zero-area boxes (sub-1px after conversion).
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        out.append({"boxes": boxes[keep], "cls": cls[keep]})
    return out
