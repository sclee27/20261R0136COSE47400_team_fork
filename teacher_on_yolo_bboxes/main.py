"""
main.py — run the p2-YOLO + teacher pipeline over SDS images.

Usage
-----
    python main.py                          # val split, 10 images, save to results/
    python main.py --split train --max-images 50
    python main.py --images-dir /path/to/sds/images   # if auto-detection fails
    python main.py --show                   # interactive matplotlib instead of saving
    python main.py --class-idx 1            # focus per-class plots on class 1 (boat)

Edit the CONFIGURATION block below to match your checkpoint / dataset paths.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURATION — edit these to match your local machine
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).resolve().parent          # teacher_on_yolo_bboxes/
_REPO_ROOT  = _HERE.parent                             # repo root

YOLO_PKG_DIR = _REPO_ROOT / "yolov8"
TEACHER_DIR  = _REPO_ROOT / "teacher"
YOLO_CKPT    = _HERE / "p2_yolo_ckpts" / "best.pt"
TEACHER_CKPT = _HERE / "teacher_ckpts"  / "best.pt"
YOLO_CFG     = _REPO_ROOT / "experiments" / "cfg" / "yolov8m-p2.yaml"
ANN_DIR      = _REPO_ROOT / "data_sds" / "annotations"

# Candidate directories for SDS images (tried in order).
# Add your own path first if it differs from these defaults.
_IMAGE_CANDIDATES: list[Path] = [
    _REPO_ROOT / "data_sds" / "images",            # images committed / symlinked in repo
    Path("/root/datasets/sds/images"),              # GPU instance (Elice)
    Path("data_sds/images"),                        # CWD-relative fallback
]

# SDS foreground class names (index matches CLASS_TO_IDX in sampling/data.py)
CLASS_NAMES = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]
# ---------------------------------------------------------------------------


_ANN_FILES = {"train": "instances_train.json", "val": "instances_val.json"}


def build_gt_index(ann_path: Path) -> dict:
    """Parse a COCO JSON and return a dict mapping image filename → GT info.

    Each value is a dict with:
        "bboxes_xywh" – np.ndarray (N, 4)  COCO [x, y, w, h] in original pixels
        "labels"      – np.ndarray (N,)    integer class index (into CLASS_NAMES)
        "w", "h"      – original image dimensions in pixels
    """
    with open(ann_path) as f:
        coco = json.load(f)

    id2name  = {c["id"]: c["name"] for c in coco["categories"]}
    name2idx = {n: i for i, n in enumerate(CLASS_NAMES)}
    img_meta = {im["id"]: (im["file_name"], im["width"], im["height"])
                for im in coco["images"]}

    by_fname: dict = defaultdict(lambda: {"bboxes_xywh": [], "labels": [], "w": 0, "h": 0})
    for ann in coco["annotations"]:
        entry = img_meta.get(ann["image_id"])
        if entry is None:
            continue
        fname, iw, ih = entry
        cname = id2name.get(ann["category_id"])
        if cname not in name2idx:
            continue
        by_fname[fname]["bboxes_xywh"].append(ann["bbox"])
        by_fname[fname]["labels"].append(name2idx[cname])
        by_fname[fname]["w"] = iw
        by_fname[fname]["h"] = ih

    return {
        fn: {
            "bboxes_xywh": np.array(d["bboxes_xywh"], dtype=float),
            "labels":      np.array(d["labels"],      dtype=int),
            "w": d["w"], "h": d["h"],
        }
        for fn, d in by_fname.items()
    }


def get_gt_for_image(img_path: Path, gt_index: dict) -> dict:
    """Return GT bboxes/labels in 640×640 pixel space (simple-resize transform).

    The image loader (_load_image) stretches the original to 640×640, so we
    apply the matching linear scale to GT boxes.

    Returns dict with:
        "gt_bboxes" – np.ndarray (N, 4)  xyxy in 640-space
        "gt_labels" – np.ndarray (N,)    integer class indices
    """
    d = gt_index.get(img_path.name)
    if d is None or len(d["bboxes_xywh"]) == 0:
        return {
            "gt_bboxes": np.zeros((0, 4), dtype=float),
            "gt_labels": np.zeros(0, dtype=int),
        }

    bboxes = d["bboxes_xywh"]   # (N, 4)  COCO xywh, original pixel coords
    sx, sy = 640.0 / d["w"], 640.0 / d["h"]
    xyxy = np.stack([
        bboxes[:, 0] * sx,
        bboxes[:, 1] * sy,
        (bboxes[:, 0] + bboxes[:, 2]) * sx,
        (bboxes[:, 1] + bboxes[:, 3]) * sy,
    ], axis=1)
    return {"gt_bboxes": xyxy, "gt_labels": d["labels"]}


def find_images_dir(override: str | None = None) -> Path:
    """Return the SDS images root (contains train/ and val/ subdirs).

    Uses `override` if given, otherwise walks _IMAGE_CANDIDATES.
    Raises FileNotFoundError with a helpful message if nothing is found.
    """
    if override:
        p = Path(override)
        if p.is_dir():
            return p
        raise FileNotFoundError(
            f"--images-dir not found: {p}\n"
            "Check that the path exists and contains train/ and val/ subdirs."
        )
    for p in _IMAGE_CANDIDATES:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        "Cannot find the SDS images directory automatically.\n"
        "Either pass  --images-dir /path/to/sds/images\n"
        "or add the correct path to _IMAGE_CANDIDATES at the top of main.py."
    )


def collect_image_paths(
    split: str,
    images_dir: Path,
    ann_dir: Path,
    max_images: int = 0,
) -> list[Path]:
    """Return existing image file paths for `split` by scanning the images folder,
    rather than following the annotation JSON."""
    img_dir = images_dir / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    paths = sorted(paths)  # consistent order

    if not paths:
        raise FileNotFoundError(f"No images found under {img_dir}")

    if max_images:
        paths = paths[:max_images]

    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare p2-YOLO and teacher scores on SDS images.")
    ap.add_argument("--images-dir", default=None,
                    help="Root SDS images dir (must contain train/ and val/ subdirs). "
                         "Auto-detected from _IMAGE_CANDIDATES if omitted.")
    ap.add_argument("--split", default="val", choices=["train", "val"],
                    help="Dataset split to iterate over (default: val).")
    ap.add_argument("--max-images", type=int, default=10,
                    help="Stop after this many images (0 = all, default: 10).")
    ap.add_argument("--class-idx", type=int, default=0,
                    help="Class index for per-class scatter/histogram plots (default: 0 = swimmer).")
    ap.add_argument("--save-dir", default="results",
                    help="Directory to write per-image PNG files (default: results/).")
    ap.add_argument("--show", action="store_true",
                    help="Show plots in an interactive window instead of saving to disk.")
    ap.add_argument("--device", default="cuda",
                    help="Torch device string (default: cuda).")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Top-K anchors shown in the bar chart panel (default: 20).")
    args = ap.parse_args()

    # ── imports (after path setup so compare_pipeline resolves correctly) ────
    sys.path.insert(0, str(_HERE))
    from compare_pipeline import build_pipeline, run_image
    from visualize_scores import visualize_result

    # ── build pipeline once ──────────────────────────────────────────────────
    pipeline = build_pipeline(
        yolo_pkg_dir = str(YOLO_PKG_DIR),
        teacher_dir  = str(TEACHER_DIR),
        yolo_ckpt    = str(YOLO_CKPT),
        teacher_ckpt = str(TEACHER_CKPT),
        yolo_cfg     = str(YOLO_CFG),
        yolo_scale   = "m",
        yolo_nc      = 5,          # SDS foreground classes
        device       = args.device,
        # teacher_nc auto-detected from checkpoint (num_fg + 1 = 6)
    )

    # ── discover image paths and build GT index ──────────────────────────────
    images_dir = find_images_dir(args.images_dir)
    image_paths = collect_image_paths(
        args.split, images_dir, ANN_DIR, args.max_images,
    )
    print(f"\n[main] {len(image_paths)} {args.split} images found under {images_dir / args.split}")

    ann_path = ANN_DIR / _ANN_FILES[args.split]
    gt_index = build_gt_index(ann_path)
    print(f"[main] GT index: {len(gt_index)} annotated images loaded from {ann_path.name}")

    if not args.show:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[main] saving figures to {save_dir}/\n")

    # ── per-image loop ───────────────────────────────────────────────────────
    for i, img_path in enumerate(image_paths):
        tag = f"[{i+1}/{len(image_paths)}]"
        print(f"{tag} {img_path.name}", end="", flush=True)

        gt_info = get_gt_for_image(img_path, gt_index)
        result  = run_image(pipeline, img_path, gt_info=gt_info)

        valid_n    = result.teacher_valid.sum().item()
        rejected_n = result.teacher_rejected.sum().item()
        unscored_n = result.anchor_count - valid_n - rejected_n
        gt_n       = int(result.gt_mask.sum()) if result.gt_mask is not None else 0
        print(f"  anchors={result.anchor_count}  gt_overlap={gt_n}  "
              f"valid={valid_n}  rejected={rejected_n}  unscored={unscored_n}")

        if args.show:
            visualize_result(
                result,
                class_idx=args.class_idx,
                class_names=CLASS_NAMES,
                top_k=args.top_k,
            )
        else:
            out_path = save_dir / f"{img_path.stem}_scores.png"
            visualize_result(
                result,
                class_idx=args.class_idx,
                class_names=CLASS_NAMES,
                top_k=args.top_k,
                save_path=str(out_path),
            )

    print(f"\n[main] done." + (f"  figures in {args.save_dir}/" if not args.show else ""))


if __name__ == "__main__":
    main()
