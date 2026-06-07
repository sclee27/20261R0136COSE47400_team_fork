"""backbone_extractor.py

Extracts COCO-pretrained YOLOv8m backbone as a pure feature extractor,
returning all 5 pyramid levels (P1–P5).

Layer indices from yolov8.yaml (m-scale, width=0.75):
    0  Conv  P1/2   → (B,  48, H/2,  W/2)
    1  Conv  P2/4
    2  C2f   P2/4   → (B,  96, H/4,  W/4)   ← P2 output
    3  Conv  P3/8
    4  C2f   P3/8   → (B, 192, H/8,  W/8)   ← P3 output
    5  Conv  P4/16
    6  C2f   P4/16  → (B, 384, H/16, W/16)  ← P4 output
    7  Conv  P5/32
    8  C2f   P5/32
    9  SPPF  P5/32  → (B, 576, H/32, W/32)  ← P5 output (post-SPPF)

P1 is after layer 0 only (single Conv, no C2f):
    0  Conv  P1/2   → (B,  48, H/2,  W/2)   ← P1 output
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn

from yolov8.model import YOLOv8, load_yaml
from yolov8.modules import Detect


# Backbone ends at layer 9 (SPPF).
_BACKBONE_END = 9

# Which layer index produces each pyramid level.
_PYRAMID_LAYERS = {
    0: "p1",   # after first Conv (stride 2)
    2: "p2",   # after C2f      (stride 4)
    4: "p3",   # after C2f      (stride 8)
    6: "p4",   # after C2f      (stride 16)
    9: "p5",   # after SPPF     (stride 32)
}


class YOLOv8Backbone(nn.Module):
    """COCO-pretrained YOLOv8m backbone — returns (P1, P2, P3, P4, P5).

    All 5 feature maps are returned as a dict keyed by level name.
    The head (layers 10+) is completely absent from this module —
    no Detect, no neck, no extra parameters.

    Usage:
        backbone = YOLOv8Backbone.from_pretrained("weights/yolov8m.pt")
        backbone.eval()
        with torch.no_grad():
            feats = backbone(images)   # dict with keys orig, p1..p5
        p2 = feats["p2"]   # (B, 96, H/4, W/4) for 640x640 input
    """

    def __init__(self, layers: nn.ModuleList, save_at: dict[int, str]):
        """
        Args:
            layers:   the backbone nn.ModuleList (layers 0–9).
            save_at:  mapping from layer index → pyramid level name.
        """
        super().__init__()
        self.layers = layers
        self.save_at = save_at          # {0: "p1", 2: "p2", ...}
        self._capture_indices = set(save_at.keys())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats: dict[str, torch.Tensor] = {"orig": x}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self._capture_indices:
                feats[self.save_at[i]] = x
        return feats

    # ---------------------------------------------------------------- factory

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str | Path,
        scale: str = "m",
        cfg: str = "cfg/yolov8.yaml",
        freeze: bool = True,
        verbose: bool = False,
    ) -> "YOLOv8Backbone":
        """Build backbone from a COCO yolov8m.pt checkpoint.

        Args:
            weights_path: path to yolov8m.pt (official ultralytics checkpoint).
            scale:        model scale letter — must match the checkpoint.
            cfg:          yaml path, resolved relative to the yolov8 package.
            freeze:       if True, all backbone parameters are frozen (no grad).
            verbose:      print parse_model table.

        Returns:
            YOLOv8Backbone with COCO-pretrained weights loaded.
        """
        weights_path = Path(weights_path)

        # 1. Build the full YOLOv8 (backbone + neck + head) so we can
        #    transfer weights the same way verify.py does.
        full_model = YOLOv8(cfg=cfg, scale=scale, verbose=verbose)

        # 2. Load COCO checkpoint — shape-aware, same as verify.py.
        raw = torch.load(weights_path, map_location="cpu", weights_only=False)
        official_obj = raw["model"] if isinstance(raw, dict) else raw
        theirs_sd = {k: v.float() for k, v in official_obj.state_dict().items()}

        ours_sd = full_model.state_dict()
        # Keep only keys that exist AND match shape, so a non-COCO checkpoint
        # (e.g. SDS-fine-tuned nc=5) loads its backbone cleanly while the
        # shape-mismatched Detect head (nc) is dropped -- we slice the head off
        # anyway. (strict=False does NOT tolerate size mismatches, only missing
        # keys, so the shape guard is required here.)
        mapped = {k: v for k, v in theirs_sd.items()
                  if k in ours_sd and v.shape == ours_sd[k].shape}
        missing = sorted(set(ours_sd) - set(mapped))

        # Only the Detect cls-head output conv may be missing (nc=80 mismatch
        # is irrelevant here since we drop the head entirely). Warn but proceed.
        backbone_missing = [k for k in missing if not k.startswith("model.2")]
        if backbone_missing:
            print(f"[warn] {len(backbone_missing)} backbone keys not found in checkpoint:")
            for k in backbone_missing[:5]:
                print(f"       {k}")

        full_model.load_state_dict(mapped, strict=False)
        print(f"[ok] loaded {len(mapped)}/{len(ours_sd)} tensors from {weights_path.name}")

        # 3. Slice out layers 0..9 (backbone only). The full model's Sequential
        #    is self.model[0..21+], where [22] is Detect. We want [0..9].
        all_layers = list(full_model.model.children())
        backbone_layers = nn.ModuleList(all_layers[: _BACKBONE_END + 1])

        # Sanity: none of these should be a Detect layer.
        for i, l in enumerate(backbone_layers):
            assert not isinstance(l, Detect), f"layer {i} is Detect — wrong cut point"

        # 4. Build and optionally freeze.
        backbone = cls(backbone_layers, save_at=dict(_PYRAMID_LAYERS))
        if freeze:
            for p in backbone.parameters():
                p.requires_grad_(False)
            print("[ok] backbone parameters frozen")

        return backbone

    def channel_dims(self, scale: str = "m") -> dict[str, int]:
        """Return output channel count per pyramid level for a given scale."""
        # width multipliers from yolov8.yaml scales
        width = {"n": 0.25, "s": 0.50, "m": 0.75, "l": 1.00, "x": 1.25}[scale]
        # base channel counts from yaml args (before width scaling)
        # P1: Conv[64], P2: C2f[128], P3: C2f[256], P4: C2f[512], P5: SPPF[1024]
        import math
        def w(c): return math.ceil(min(c, 768) * width / 8) * 8  # max_channels=768 for m
        return {"p1": w(64), "p2": w(128), "p3": w(256), "p4": w(512), "p5": w(1024)}
    
if __name__ == '__main__':
    print("Loading backbone.py..")