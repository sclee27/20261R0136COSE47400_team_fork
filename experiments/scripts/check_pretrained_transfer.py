"""Sanity check: confirm COCO pretrained weights transfer into each variant.

For every variant we:
  1. Build the model from cfg with nc=6 (SeaDronesSee classes).
  2. Load yolov8m.pt (COCO, nc=80) with ultralytics' shape-aware transfer.
  3. Report how many tensors were copied vs left at random init.

Expected behavior:
  baseline    : ~99-100% of tensors transferred (only nc=80 -> nc=6 cls head
                output conv is dropped: 3 tensors).
  sppf-k3     : same as baseline. SPPF MaxPool has no params, cv1/cv2 shapes
                are unchanged by k.
  p2          : ~85-95% transferred. Newly added P2 branch + Detect.cv2[0]/cv3[0]
                input channels (128 instead of 192) and an extra Detect level
                are random init.
  p2-sppf-k3  : same coverage as p2.

Run on the GPU instance after uploading yolov8m.pt:
    python experiments/scripts/check_pretrained_transfer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from ultralytics import YOLO


HERE = Path(__file__).resolve().parent.parent      # .../experiments/
CFGS = {
    "baseline":   HERE / "cfg/yolov8m.yaml",
    "p2":         HERE / "cfg/yolov8m-p2.yaml",
    "sppf-k3":    HERE / "cfg/yolov8m-sppf-k3.yaml",
    "p2-sppf-k3": HERE / "cfg/yolov8m-p2-sppf-k3.yaml",
}
WEIGHTS = HERE / "weights/yolov8m.pt"
NC_SDS = 6                                          # SeaDronesSee class count


def check(variant: str, cfg_path: Path) -> None:
    print(f"\n=== {variant} ===")
    if not cfg_path.exists():
        print(f"   skip: cfg missing at {cfg_path}")
        return

    # Build the SDS-shaped model (nc=6) from cfg.
    model = YOLO(str(cfg_path))
    model.model.nc = NC_SDS
    # Rebuild the detect head if nc differs from cfg default.
    # (ultralytics' load() handles shape mismatch on the cls head automatically.)

    # Snapshot the random-init weights for a per-tensor before/after compare.
    sd_before = {k: v.detach().clone() for k, v in model.model.state_dict().items()}

    if not WEIGHTS.exists():
        print(f"   skip: pretrained weights missing at {WEIGHTS}")
        return
    model = model.load(str(WEIGHTS))

    sd_after = model.model.state_dict()

    # A tensor is "transferred" if it differs from the random-init snapshot.
    n_total = len(sd_after)
    n_transferred = sum(
        1 for k in sd_after
        if k in sd_before
        and sd_after[k].shape == sd_before[k].shape
        and not (sd_after[k] == sd_before[k]).all()
    )
    n_shape_mismatch = sum(
        1 for k in sd_after
        if k in sd_before and sd_after[k].shape != sd_before[k].shape
    )
    n_new = sum(1 for k in sd_after if k not in sd_before)

    pct = 100.0 * n_transferred / n_total if n_total else 0.0
    print(f"   transferred : {n_transferred:>4d} / {n_total} tensors  ({pct:.1f}%)")
    print(f"   random init : {n_total - n_transferred:>4d}   (new arch + cls head reshape)")
    if n_shape_mismatch:
        print(f"   shape diffs : {n_shape_mismatch:>4d}")
    if n_new:
        print(f"   brand-new   : {n_new:>4d}")


def main() -> None:
    if not WEIGHTS.exists():
        print(f"!! place yolov8m.pt at {WEIGHTS} before running.")
        print("   Download: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt")
        sys.exit(1)

    for v, p in CFGS.items():
        check(v, p)


if __name__ == "__main__":
    main()
