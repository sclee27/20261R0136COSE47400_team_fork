"""Forward-equivalence verifier.

Loads the official yolov8{scale}.pt weights into our slim re-implementation
and confirms that:

    1. Every parameter tensor has the same shape (so loading is unambiguous).
    2. Total parameter count matches the published numbers.
    3. For a fixed random input, the decoded predictions match the official
       ultralytics model to within ``atol``.

Run:
    .venv/bin/python -m yolov8.verify --scale m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

from .modules import C2f, Conv, Detect, SPPF
from .model import YOLOv8


# ---- expected parameter counts (from cfg/yolov8.yaml comments)
EXPECTED_PARAMS = {
    "n": 3_157_200,
    "s": 11_166_560,
    "m": 25_902_640,
    "l": 43_691_520,
    "x": 68_229_648,
}


# Official weight URLs
WEIGHT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8{}.pt"


def download_weights(scale: str, dest: Path) -> Path:
    """Download the official yolov8{scale}.pt checkpoint if not present."""
    url = WEIGHT_URL.format(scale)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[ok] cached weights at {dest}")
        return dest
    print(f"[..] downloading {url} -> {dest}")
    import urllib.request

    urllib.request.urlretrieve(url, dest)
    print(f"[ok] downloaded {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def load_official_state_dict(weight_path: Path) -> tuple[dict, "OfficialModelInfo"]:
    """Load the pickled ultralytics checkpoint and return:

    * a plain state_dict of float32 parameters (renamed nothing — keys are the
      *same* as in our re-implementation because both walk ``self.model[i]...``
      in the same yaml order).
    * an OfficialModelInfo bundle with the original (un-renamed) state dict +
      the live ultralytics model object, so we can also run a forward through
      it for output comparison.
    """
    # The official .pt is a torch.save of a dict containing the actual model.
    # ultralytics uses a custom unpickler — but for *loading parameters* we
    # only need the raw state_dict, which we can get by directly loading.
    # The pickled object references `ultralytics.*` classes, so we install
    # `safe_globals` shims for the bits we need.
    import io

    raw = torch.load(weight_path, map_location="cpu", weights_only=False)
    model_obj = raw["model"] if isinstance(raw, dict) else raw
    sd = {k: v.float() for k, v in model_obj.state_dict().items()}

    return sd, OfficialModelInfo(model_obj, sd)


class OfficialModelInfo:
    def __init__(self, model_obj, state_dict):
        self.model = model_obj
        self.state_dict = state_dict


def map_state_dict(ours_sd: dict, theirs_sd: dict) -> tuple[dict, list[str], list[str]]:
    """Match keys between the two state dicts.

    Since our model walks the *same* yaml in the *same* order, the natural
    naming for our params should align with theirs after a tiny rename:

    *   theirs:   ``model.0.conv.weight``
    *   ours:     ``model.0.conv.weight``     (identical for Conv/Bottleneck/C2f/SPPF)

    For the Detect head:
    *   theirs (legacy=True):  ``model.22.cv2.0.0.conv.weight`` etc.
    *   ours:                  ``model.22.cv2.0.0.conv.weight`` — identical.

    Returns (renamed_state_dict, missing_keys, unexpected_keys).
    """
    mapped = {}
    for k_theirs, v in theirs_sd.items():
        if k_theirs in ours_sd:
            mapped[k_theirs] = v
    missing = sorted(set(ours_sd) - set(mapped))
    unexpected = sorted(set(theirs_sd) - set(mapped) - {k for k in theirs_sd if k.endswith(".num_batches_tracked")})
    return mapped, missing, unexpected


def report_mismatched_shapes(ours_sd, mapped_sd) -> int:
    bad = 0
    for k, v in mapped_sd.items():
        if k in ours_sd and v.shape != ours_sd[k].shape:
            print(f"   shape mismatch on {k}: theirs={tuple(v.shape)}  ours={tuple(ours_sd[k].shape)}")
            bad += 1
    return bad


# ----------------------------------------------------------- verifier


def verify(scale: str, atol: float, weights_dir: Path) -> None:
    expected = EXPECTED_PARAMS[scale]

    # 1. Build our model and check param count.
    print(f"\n=== YOLOv8{scale} ===")
    ours = YOLOv8(cfg="cfg/yolov8.yaml", scale=scale, verbose=False)
    ours.eval()
    n_ours = sum(p.numel() for p in ours.parameters())
    print(f"param count: {n_ours:,}  expected {expected:,}  {'OK' if n_ours == expected else 'MISMATCH'}")
    assert n_ours == expected, "parameter count mismatch — model topology differs from upstream"

    # 2. Download / load the official checkpoint.
    weight_path = weights_dir / f"yolov8{scale}.pt"
    download_weights(scale, weight_path)
    theirs_sd, official_info = load_official_state_dict(weight_path)

    # 3. Map state dicts.
    ours_sd = ours.state_dict()
    mapped, missing, unexpected = map_state_dict(ours_sd, theirs_sd)
    bad_shapes = report_mismatched_shapes(ours_sd, mapped)
    print(f"state_dict: mapped={len(mapped)}/{len(ours_sd)}  missing={len(missing)}  unexpected={len(unexpected)}  shape_mismatch={bad_shapes}")
    if missing:
        print(f"   first missing: {missing[:5]}")
    if unexpected:
        print(f"   first unexpected: {unexpected[:5]}")
    assert bad_shapes == 0, "weight shape mismatch — can't proceed to forward comparison"
    assert not missing, f"missing keys when loading official weights: {missing[:5]}..."

    # 4. Load weights and run forward through both models.
    ours.load_state_dict(mapped, strict=False)
    ours.eval()

    theirs = official_info.model.float().eval()

    torch.manual_seed(0)
    x = torch.randn(2, 3, 640, 640)

    with torch.no_grad():
        out_ours = ours(x)
        out_theirs = theirs(x)

    # Both outputs are tuples: (decoded[B, 4+nc, A], raw_preds_dict).
    dec_ours = out_ours[0] if isinstance(out_ours, tuple) else out_ours
    dec_theirs = out_theirs[0] if isinstance(out_theirs, tuple) else out_theirs

    diff = (dec_ours - dec_theirs).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"forward decoded: shape={tuple(dec_ours.shape)}  max|Δ|={max_diff:.3e}  mean|Δ|={mean_diff:.3e}")

    ok = max_diff < atol
    print(f"{'PASS' if ok else 'FAIL'}: max|Δ|={max_diff:.3e}  atol={atol:.0e}")
    if not ok:
        # Show per-component breakdown to help debugging.
        box_d = (dec_ours[:, :4] - dec_theirs[:, :4]).abs().max().item()
        cls_d = (dec_ours[:, 4:] - dec_theirs[:, 4:]).abs().max().item()
        print(f"   box channels max|Δ|={box_d:.3e}")
        print(f"   cls channels max|Δ|={cls_d:.3e}")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="m", choices=list(EXPECTED_PARAMS))
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--weights-dir", type=Path, default=Path(__file__).parent / "weights")
    ap.add_argument("--all", action="store_true", help="verify every scale")
    args = ap.parse_args()

    scales = list(EXPECTED_PARAMS) if args.all else [args.scale]
    for s in scales:
        verify(s, args.atol, args.weights_dir)


if __name__ == "__main__":
    main()
