"""Loss-equivalence test against ultralytics' v8DetectionLoss.

Builds the official model + our slim re-impl with identical weights, feeds
the same head outputs into both losses, and confirms the scalar loss matches.

Run:
    .venv/bin/python -m yolov8.tests.test_loss_equivalence
"""

from __future__ import annotations

from pathlib import Path

import torch

from yolov8 import YOLOv8, v8DetectionLoss as OurLoss
from yolov8.verify import download_weights, load_official_state_dict, map_state_dict


def main():
    torch.manual_seed(0)

    # 1. Build both models with identical weights.
    weight_path = Path(__file__).parent.parent / "weights" / "yolov8m.pt"
    download_weights("m", weight_path)
    theirs_sd, official_info = load_official_state_dict(weight_path)
    theirs_model = official_info.model.float().train()

    ours = YOLOv8(cfg="cfg/yolov8.yaml", scale="m", verbose=False).train()
    mapped, missing, _ = map_state_dict(ours.state_dict(), theirs_sd)
    assert not missing, f"missing keys: {missing[:3]}"
    ours.load_state_dict(mapped, strict=False)

    # The official model's `args` is a dict; ultralytics v8DetectionLoss accesses
    # `.box / .cls / .dfl` as attributes, so we wrap it in a namespace clone and
    # make sure both models share identical gain hyper-params.
    from types import SimpleNamespace
    if isinstance(theirs_model.args, dict):
        theirs_model.args = SimpleNamespace(**theirs_model.args)
    for k in ("box", "cls", "dfl"):
        setattr(theirs_model.args, k, getattr(ours.args, k))

    # 2. Common forward in training mode -> dict preds.
    imgs = torch.randn(2, 3, 640, 640)
    preds_ours = ours(imgs)
    preds_theirs = theirs_model(imgs)
    # ultralytics returns either dict or {"one2many": ..., "one2one": ...}; we want the dict.
    if isinstance(preds_theirs, dict) and "one2many" in preds_theirs:
        preds_theirs = preds_theirs["one2many"]

    # 3. Common batch.
    batch = {
        "batch_idx": torch.tensor([0, 0, 1, 1, 1]),
        "cls": torch.tensor([[3.0], [17.0], [0.0], [56.0], [7.0]]),
        "bboxes": torch.tensor([
            [0.4, 0.5, 0.2, 0.3],
            [0.7, 0.2, 0.1, 0.15],
            [0.5, 0.5, 0.4, 0.4],
            [0.25, 0.75, 0.15, 0.2],
            [0.8, 0.8, 0.18, 0.22],
        ]),
    }

    # 4. Compute both losses.
    our_crit = OurLoss(ours)
    loss_ours, items_ours = our_crit(preds_ours, batch)

    from ultralytics.utils.loss import v8DetectionLoss as TheirLoss
    their_crit = TheirLoss(theirs_model)
    loss_theirs, items_theirs = their_crit(preds_theirs, batch)

    # Their loss returns a (3,) tensor (box, cls, dfl) * batch_size — we return
    # the sum scalar. Compare per-component AND the scalar sum.
    loss_theirs_scalar = loss_theirs.sum() if loss_theirs.ndim > 0 else loss_theirs
    print(f"ours   : loss={loss_ours.item():.6f}  box={items_ours[0]:.6f}  cls={items_ours[1]:.6f}  dfl={items_ours[2]:.6f}")
    print(f"theirs : loss={loss_theirs_scalar.item():.6f}  box={items_theirs[0]:.6f}  cls={items_theirs[1]:.6f}  dfl={items_theirs[2]:.6f}")
    delta = (loss_ours - loss_theirs_scalar).abs().item()
    per_comp = (items_ours - items_theirs).abs().max().item()
    print(f"|Δ loss| = {delta:.3e}   max|Δ component| = {per_comp:.3e}")
    assert delta < 1e-3, f"loss mismatch: {delta}"
    assert per_comp < 1e-3, f"per-component mismatch: {per_comp}"
    print("PASS")


if __name__ == "__main__":
    main()
