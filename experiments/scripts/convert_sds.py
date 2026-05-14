"""Convert SeaDronesSee v2 (COCO-format JSON) to YOLO txt labels.

SeaDronesSee distributes images plus a COCO-style annotation JSON per split
(train / val / (test — labels are private)). YOLO wants:

    images/{split}/<basename>.jpg
    labels/{split}/<basename>.txt

with each label line as:

    <class> <x_center> <y_center> <width> <height>      # all normalized to [0,1]

This script:
  1. Reads the COCO JSON for one split.
  2. Maps the original category_id values to a contiguous 0..nc-1 range.
  3. Writes a YOLO-format .txt for every image that has at least one bbox
     (images with no annotations are left without a .txt; YOLO treats those
     as background, which is correct).
  4. Optionally symlinks (or copies) the images into images/{split}/.

Usage (on the GPU box, after extracting SeaDronesSee under ~/datasets/sds-raw):

    python experiments/scripts/convert_sds.py \\
        --coco ~/datasets/sds-raw/annotations/instances_train.json \\
        --images ~/datasets/sds-raw/images/train \\
        --out ~/datasets/sds \\
        --split train

    python experiments/scripts/convert_sds.py \\
        --coco ~/datasets/sds-raw/annotations/instances_val.json \\
        --images ~/datasets/sds-raw/images/val \\
        --out ~/datasets/sds \\
        --split val

Adjust the paths to match wherever you extracted the SDS archive. Class IDs
in SeaDronesSee v2 are 1..6, which we map to 0..5 — see CLASS_MAP below.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


# SeaDronesSee v2 maps category_id -> name in the JSON. We map to a contiguous
# 0..5 order that matches the order in data/sds.yaml. Adjust if the JSON's
# actual category list differs (verify by inspecting the first 10 lines of
# the annotations JSON — `categories` field).
CLASS_MAP: dict[int, int] = {
    # original_id : yolo_id
    1: 0,   # swimmer
    2: 1,   # floater (a.k.a. "boat" in older v1 — double check via JSON)
    3: 2,   # life_jacket
    4: 3,   # boat
    5: 4,   # swimmer_on_boat
    6: 5,   # floater_on_boat
}


def convert_split(coco_path: Path, image_dir: Path, out_dir: Path, split: str, link_mode: str) -> None:
    print(f"\n== converting {split} ==")
    print(f"   coco json    : {coco_path}")
    print(f"   image source : {image_dir}")

    with open(coco_path) as f:
        coco = json.load(f)

    # Show the actual category list so a mismatch with CLASS_MAP is obvious.
    print(f"   categories in JSON ({len(coco['categories'])}):")
    for c in coco["categories"]:
        print(f"      id={c['id']:>3}  name={c.get('name', '?')}")
    print(f"   CLASS_MAP keys: {sorted(CLASS_MAP)} -> yolo ids 0..{max(CLASS_MAP.values())}")

    # Index images and group annotations by image_id.
    images = {img["id"]: img for img in coco["images"]}
    anns_by_img: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    out_img_dir = out_dir / "images" / split
    out_lbl_dir = out_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_imgs, n_with_labels, n_anns, n_skipped = 0, 0, 0, 0
    unknown_cats: set[int] = set()

    for img_id, img in images.items():
        fname = img["file_name"]
        W, H = img["width"], img["height"]
        src = image_dir / fname
        if not src.exists():
            # Some SDS releases nest images in subdirectories. Try a recursive search once.
            cands = list(image_dir.rglob(fname))
            if not cands:
                print(f"   [warn] image missing: {fname}")
                n_skipped += 1
                continue
            src = cands[0]

        # Image: symlink (fast, no disk dup) or copy.
        dst_img = out_img_dir / fname
        if not dst_img.exists():
            if link_mode == "symlink":
                os.symlink(src.resolve(), dst_img)
            else:  # copy
                from shutil import copy2
                copy2(src, dst_img)
        n_imgs += 1

        # Labels.
        anns = anns_by_img.get(img_id, [])
        if not anns:
            continue

        lines = []
        for a in anns:
            cat_id = a["category_id"]
            if cat_id not in CLASS_MAP:
                unknown_cats.add(cat_id)
                continue
            yolo_cls = CLASS_MAP[cat_id]

            x, y, w, h = a["bbox"]  # COCO format: top-left x, top-left y, w, h (absolute pixels)
            # Clamp to image bounds (some annotations slightly overflow).
            x = max(0.0, min(x, W))
            y = max(0.0, min(y, H))
            w = max(0.0, min(w, W - x))
            h = max(0.0, min(h, H - y))
            if w <= 1 or h <= 1:
                continue  # degenerate box

            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            nw, nh = w / W, h / H
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_anns += 1

        if lines:
            (out_lbl_dir / (Path(fname).stem + ".txt")).write_text("\n".join(lines) + "\n")
            n_with_labels += 1

    print(f"   images written : {n_imgs}")
    print(f"   labels written : {n_with_labels}   (images without annotations are treated as background)")
    print(f"   total bboxes   : {n_anns}")
    if n_skipped:
        print(f"   skipped (missing image) : {n_skipped}")
    if unknown_cats:
        print(f"   [warn] saw category_ids outside CLASS_MAP: {sorted(unknown_cats)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, type=Path, help="path to COCO-style JSON for the split")
    ap.add_argument("--images", required=True, type=Path, help="directory containing the split's images")
    ap.add_argument("--out", required=True, type=Path, help="output dataset root (will hold images/ and labels/)")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink",
                    help="how to bring images into the dataset directory")
    args = ap.parse_args()

    convert_split(args.coco, args.images, args.out, args.split, args.link_mode)


if __name__ == "__main__":
    main()
