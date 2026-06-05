"""Block: torch Dataset for teacher-model training.

Per image this Dataset:
  1. loads the JPEG and letterboxes it to 640 using the SAME r/pad math as
     `sampling.data.letterbox_boxes` (so image pixels align with GT/candidate
     boxes -- ROIs must land on the right features),
  2. loads GT boxes/classes (re-parsing the COCO JSON to recover file names,
     replicating `sampling.data.load_gt_by_image`'s parsing + subsampling so the
     image set matches what test_sampling.py analyzed),
  3. samples candidate boxes with the chosen mode (original/jittering/jittering_v2),
  4. computes IoU vs GT and labels the candidates (pos / background / ignore),
  5. optionally stratified-subsamples by IoU bucket,
  6. filters to the enabled P-levels,
  7. returns tensors ready for the evaluator.

The runner puts teacher/ on sys.path; this module also bootstraps it so that
`from sampling... import ...` and `from training.model import ...` resolve when
run directly.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

# --- sys.path bootstrap: insert TEACHER dir (= parent of training/) -----------
_TEACHER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TEACHER_DIR not in sys.path:
    sys.path.insert(0, _TEACHER_DIR)

from sampling.data import (  # noqa: E402
    CLASS_TO_IDX,
    SPLIT_FILES,
    letterbox_boxes,
    resolve_ann_dir,
)
from sampling import boxes as boxes_mod  # noqa: E402
from sampling.metrics import iou_and_dominant  # noqa: E402
from sampling.labeling import label_boxes, stratify_indices, stratify_indices_per_gt  # noqa: E402
from training.model import filter_to_enabled_levels  # noqa: E402

# Friendly experiment names -> internal sampler modes.
FRIENDLY_TO_SAMPLER = {
    "original": "stride",
    "jittering": "gt_linked",
    "jittering_v2": "gt_linked_v2",
}


class TeacherBBoxDataset(torch.utils.data.Dataset):
    """Per-image candidate boxes + IoU labels + letterboxed image tensor."""

    def __init__(self, sampling_cfg, images_dir: str, split: str, max_images: int,
                 mode: str, num_fg_classes: int, stratify: bool,
                 enabled_levels: list, upper_short: dict, pad_value: int = 114,
                 seed: int = 0):
        if mode not in FRIENDLY_TO_SAMPLER:
            raise ValueError(
                f"unknown mode {mode!r}; expected one of {list(FRIENDLY_TO_SAMPLER)}")

        self.sampling_cfg = sampling_cfg
        self.images_dir = images_dir
        self.split = split
        self.max_images = max_images
        self.mode = mode                       # friendly name
        self.sampler_mode = FRIENDLY_TO_SAMPLER[mode]
        self.num_fg_classes = num_fg_classes
        self.stratify = stratify
        self.enabled_levels = enabled_levels
        self.upper_short = upper_short
        self.pad_value = pad_value
        self.seed = seed
        self.image_size = sampling_cfg.image_size

        # --- parse COCO JSON (replicates load_gt_by_image parsing/subsampling) ---
        path = resolve_ann_dir(sampling_cfg.data.ann_dir) / SPLIT_FILES[split]
        if not path.exists():
            raise FileNotFoundError(
                f"annotation file not found: {path}\n"
                f"  set data.ann_dir in the config, or export SDS_ANN_DIR=/path/to/annotations")

        with open(path) as f:
            coco = json.load(f)

        # categories -> id2name (drop the 'ignored' class)
        id2name = {c["id"]: c["name"] for c in coco["categories"] if c["name"] != "ignored"}
        # images -> id -> (file_name, w, h)
        img_meta = {im["id"]: (im["file_name"], im["width"], im["height"])
                    for im in coco["images"]}

        by_img: dict[int, list] = defaultdict(list)
        for a in coco["annotations"]:
            name = id2name.get(a["category_id"])
            if name in CLASS_TO_IDX and a["image_id"] in img_meta:
                by_img[a["image_id"]].append((a["bbox"], CLASS_TO_IDX[name]))

        img_ids = sorted(by_img.keys())
        rng = np.random.default_rng(seed)
        if max_images and len(img_ids) > max_images:
            img_ids = sorted(rng.choice(img_ids, size=max_images, replace=False).tolist())

        # Build records (keep native COCO xywh + size; letterbox happens lazily).
        self.records = []
        for iid in img_ids:
            recs = by_img[iid]
            fname, iw, ih = img_meta[iid]
            xywh = np.array([b for b, _ in recs], dtype=float)
            cls = np.array([c for _, c in recs], dtype=int)
            self.records.append({
                "file_name": fname,
                "w": int(iw),
                "h": int(ih),
                "boxes_xywh": xywh,
                "cls": cls,
            })

    def __len__(self) -> int:
        return len(self.records)

    def _letterbox_image(self, img: np.ndarray, w: int, h: int) -> torch.Tensor:
        """Letterbox an HxWx3 RGB uint8 image with the SAME r/pad as letterbox_boxes.

        Returns a (3, size, size) float32 tensor in [0, 1].
        """
        size = self.image_size
        r = min(size / w, size / h)
        pad_x = (size - w * r) / 2.0
        pad_y = (size - h * r) / 2.0

        new_w = round(w * r)
        new_h = round(h * r)
        # Bilinear resize via PIL (matches typical detector preprocessing).
        resized = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
        resized = np.asarray(resized)

        canvas = np.full((size, size, 3), self.pad_value, dtype=np.uint8)
        off_x = round(pad_x)
        off_y = round(pad_y)
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

        # -> float32, /255, (3, H, W)
        t = torch.from_numpy(canvas).float().div_(255.0).permute(2, 0, 1).contiguous()
        return t

    def _empty_item(self, image: torch.Tensor) -> dict:
        return {
            "image": image,
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        w, h = rec["w"], rec["h"]
        size = self.image_size

        # 1-2. Load + letterbox the image (RGB).
        img_path = os.path.join(self.images_dir, self.split, rec["file_name"])
        with Image.open(img_path) as im:
            img = np.asarray(im.convert("RGB"))
        image = self._letterbox_image(img, w, h)

        # 3. GT boxes in the 640 frame; align cls; drop zero-area boxes.
        xywh = rec["boxes_xywh"]
        cls = rec["cls"]
        if len(xywh) == 0:
            return self._empty_item(image)
        gt = letterbox_boxes(xywh, w, h, size).astype(np.int32)
        keep = (gt[:, 2] > gt[:, 0]) & (gt[:, 3] > gt[:, 1])
        gt = gt[keep]
        cls = cls[keep]
        if len(gt) == 0:
            return self._empty_item(image)

        # 4. Sample candidate boxes. The per-item rng makes `jittering` fully
        #    reproducible; `original` (stride) and `jittering_v2` also draw box
        #    CENTERS from the legacy sampler's global np.random, so those two are
        #    NOT per-item reproducible (seed only fixes the GT-linked scale draw).
        rng = np.random.default_rng(self.seed + i)
        if self.sampler_mode == "gt_linked":
            cand, _src = boxes_mod.sample_boxes_gt_linked(
                gt, self.sampling_cfg.jitter, size, rng)
        elif self.sampler_mode == "gt_linked_v2":
            cand, _src = boxes_mod.sample_boxes_gt_linked_v2(
                gt, self.sampling_cfg.jitter, size, rng)
        elif self.sampler_mode == "stride":
            cand, _src = boxes_mod.sample_boxes_stride(
                gt, self.sampling_cfg.stride, size)
        else:  # defensive; __init__ already validated
            raise ValueError(f"unknown sampler mode: {self.sampler_mode}")

        if len(cand) == 0:
            return self._empty_item(image)

        # 5. IoU vs GT (dominant).
        iou_dom, dom_idx, _ = iou_and_dominant(cand, gt)

        # 6. Label: bg=num_fg_classes, ignore=-1, else class index.
        labels = label_boxes(iou_dom, dom_idx, cls, self.sampling_cfg.label,
                             self.num_fg_classes)

        # 7. Optional stratified subsampling across IoU buckets.
        if self.stratify:
            # keep = stratify_indices(iou_dom, self.sampling_cfg.label, rng) --> Original stratify 
            
            # New stratify (per GT)
            num_gts = len(cls) # Extract the number of ground truth objects
            keep = stratify_indices_per_gt(
                iou_dom=iou_dom, 
                dom_idx=dom_idx, 
                num_gts=num_gts, 
                label_cfg=self.sampling_cfg.label, 
                rng=rng
            )

            cand = cand[keep]
            labels = labels[keep]

        # 8. Keep only boxes whose level is enabled.
        cand, labels = filter_to_enabled_levels(
            cand, labels, self.enabled_levels, self.upper_short)

        if len(cand) == 0:
            return self._empty_item(image)

        return {
            "image": image,
            "boxes": torch.as_tensor(np.asarray(cand), dtype=torch.float32),
            "labels": torch.as_tensor(np.asarray(labels), dtype=torch.int64),
        }


def collate_fn(batch):
    """Collate items into a padded batch, dropping empties.

    Items with no boxes (A == 0) are dropped. If all items are dropped the batch
    is None (the training loop should skip it). Boxes are padded to A_max with a
    dummy valid box [0, 0, 10, 10] and labels with -1 (ignore) so the padding
    contributes no loss. For batch_size==1 this reduces to the trivial case.
    """
    items = [b for b in batch if b is not None and b["labels"].numel() > 0]
    if not items:
        return None

    images = torch.stack([b["image"] for b in items], dim=0)  # (B, 3, S, S)
    a_max = max(b["labels"].numel() for b in items)

    boxes_out = []
    labels_out = []
    for b in items:
        boxes = b["boxes"]
        labels = b["labels"]
        a = labels.numel()
        if a < a_max:
            pad_n = a_max - a
            pad_boxes = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32).repeat(pad_n, 1)
            pad_labels = torch.full((pad_n,), -1, dtype=torch.int64)
            boxes = torch.cat([boxes, pad_boxes], dim=0)
            labels = torch.cat([labels, pad_labels], dim=0)
        boxes_out.append(boxes)
        labels_out.append(labels)

    return {
        "images": images,
        "boxes": torch.stack(boxes_out, dim=0),    # (B, A_max, 4)
        "labels": torch.stack(labels_out, dim=0),  # (B, A_max)
    }
