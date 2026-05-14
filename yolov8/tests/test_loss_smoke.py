"""Smoke test: build YOLOv8m, run training-mode forward, run v8DetectionLoss.

Confirms shapes wire end-to-end and the loss is a positive finite scalar.
Run:
    .venv/bin/python -m yolov8.tests.test_loss_smoke
"""

from __future__ import annotations

import torch

from yolov8 import YOLOv8, v8DetectionLoss


def main():
    torch.manual_seed(0)
    model = YOLOv8(cfg="cfg/yolov8.yaml", scale="m", verbose=False).train()
    criterion = v8DetectionLoss(model)

    bs = 2
    imgs = torch.randn(bs, 3, 640, 640)
    preds = model(imgs)  # training=True path returns dict(boxes, scores, feats)
    assert isinstance(preds, dict), f"expected training-mode dict, got {type(preds)}"
    print(
        f"preds dict: boxes={tuple(preds['boxes'].shape)}  "
        f"scores={tuple(preds['scores'].shape)}  feats=[{', '.join(str(tuple(f.shape)) for f in preds['feats'])}]"
    )

    # Fake batch in the standard ultralytics format.
    # 3 targets total: 2 in image 0, 1 in image 1.
    batch = {
        "batch_idx": torch.tensor([0, 0, 1]),
        "cls": torch.tensor([[2.0], [5.0], [11.0]]),
        # xywh normalized to [0, 1]
        "bboxes": torch.tensor([
            [0.4, 0.5, 0.2, 0.3],
            [0.7, 0.2, 0.1, 0.15],
            [0.5, 0.5, 0.4, 0.4],
        ]),
    }

    loss, loss_items = criterion(preds, batch)
    box, cls, dfl = loss_items.tolist()
    print(f"loss={loss.item():.4f}   box={box:.4f}  cls={cls:.4f}  dfl={dfl:.4f}")
    assert torch.isfinite(loss), "loss is not finite"
    assert loss.item() > 0, "loss is non-positive"

    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters())
    print(f"backward OK: {n_grad}/{n_total} parameters received non-zero gradients")
    print("PASS")


if __name__ == "__main__":
    main()
