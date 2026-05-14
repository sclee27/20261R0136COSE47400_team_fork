# yolov8 — slim detection-only baseline

A modular, easy-to-modify re-implementation of **YOLOv8 detection** extracted
from the official ultralytics repository (commit `main` as of clone time).
Forward pass and loss are byte-identical to the upstream implementation when
loading the published `yolov8{n,s,m,l,x}.pt` weights.

The goal is to keep the **mathematical baseline** intact while letting you
freely experiment with architecture, loss, or assignment changes — without the
sprawl of the full ultralytics package.

## What is included

| File                          | Contents                                                                 |
| ----------------------------- | ------------------------------------------------------------------------ |
| `cfg/yolov8.yaml`             | Backbone + head topology + n/s/m/l/x scales.                             |
| `modules/conv.py`             | `Conv` (Conv + BN + SiLU), `Concat`, `autopad`.                          |
| `modules/block.py`            | `Bottleneck`, `C2f`, `SPPF`, `DFL`.                                      |
| `modules/head.py`             | `Detect` head (DFL + cls branches, anchor-free).                         |
| `model.py`                    | `parse_model` (yaml → `nn.Sequential`) and `YOLOv8` model class.         |
| `tal.py`                      | `TaskAlignedAssigner`, `make_anchors`, `dist2bbox`, `bbox2dist`.         |
| `loss.py`                     | `v8DetectionLoss`, `BboxLoss` (CIoU), `DFLoss`.                          |
| `ops.py`                      | `bbox_iou` (IoU/GIoU/DIoU/CIoU), `xywh2xyxy`, `xyxy2xywh`, `make_divisible`. |
| `verify.py`                   | Forward-equivalence verifier vs. official `yolov8*.pt`.                  |
| `tests/test_loss_smoke.py`    | Build model → forward → loss → backward, end-to-end smoke test.          |
| `tests/test_loss_equivalence.py` | Confirms our loss == ultralytics' v8DetectionLoss on identical inputs.|

## What is **not** included (and why)

* Data pipeline (`YOLODataset`, Mosaic/MixUp/HSV augmentation, collate).
* Trainer / EMA / warmup / AMP / multi-scale (the `engine/trainer.py` machinery).
* Validation metrics (`ap_per_class`, `ConfusionMatrix`, NMS-based `DetMetrics`).
* Non-detection heads: Segment, Pose, OBB, Classify, World, YOLOE, RT-DETR.

These were intentionally cut to keep the baseline small. They can be added on
top later — every model/loss API in this package matches the upstream so an
external loop can use it directly.

## Quickstart

```bash
# 1. Environment (uv, Python 3.11)
uv venv --python 3.11 .venv
uv pip install torch torchvision pyyaml numpy ultralytics  # ultralytics only for verify.py

# 2. Build a model
.venv/bin/python -c "
from yolov8 import YOLOv8
import torch
m = YOLOv8(cfg='cfg/yolov8.yaml', scale='m', verbose=False).eval()
print(sum(p.numel() for p in m.parameters()), 'params')
print(m(torch.zeros(1, 3, 640, 640))[0].shape)  # (1, 84, 8400)
"

# 3. Verify equivalence with official weights (downloads ~50 MB the first time)
.venv/bin/python -m yolov8.verify --scale m --atol 1e-4

# 4. Verify all five scales (downloads ~300 MB total)
.venv/bin/python -m yolov8.verify --all --atol 1e-4

# 5. Run the end-to-end smoke test (forward + loss + backward)
.venv/bin/python -m yolov8.tests.test_loss_smoke

# 6. Confirm loss matches ultralytics' v8DetectionLoss exactly
.venv/bin/python -m yolov8.tests.test_loss_equivalence
```

## Verification results (recorded with torch 2.11.0, CPU)

| scale | params (ours) | params (expected) | state_dict keys mapped | max\|Δ output\|        |
| ----- | ------------- | ----------------- | ---------------------- | ---------------------- |
| n     |   3,157,200   |   3,157,200       | 355 / 355              | **0.000e+00**          |
| s     |  11,166,560   |  11,166,560       | 355 / 355              | **0.000e+00**          |
| m     |  25,902,640   |  25,902,640       | 475 / 475              | **0.000e+00**          |
| l     |  43,691,520   |  43,691,520       | 595 / 595              | **0.000e+00**          |
| x     |  68,229,648   |  68,229,648       | 595 / 595              | **0.000e+00**          |

Loss equivalence on yolov8m, fixed random input + 5 random GT boxes:

```
ours   : loss=30.958422  box=2.996883  cls=9.399422  dfl=3.082906
theirs : loss=30.958422  box=2.996883  cls=9.399422  dfl=3.082906
|Δ loss| = 0.000e+00   max|Δ component| = 0.000e+00
```

## API at a glance

### Model

```python
from yolov8 import YOLOv8
model = YOLOv8(cfg="cfg/yolov8.yaml", scale="m", nc=80)

# Training mode → dict of raw head outputs
model.train()
preds = model(images)   # {"boxes": (B, 4*reg_max, A), "scores": (B, nc, A), "feats": [...]}

# Eval mode → (decoded, raw_dict)
model.eval()
decoded, raw = model(images)   # decoded: (B, 4+nc, A) in xywh + sigmoid(cls)
```

### Loss

```python
from yolov8 import v8DetectionLoss
criterion = v8DetectionLoss(model)

batch = {
    "batch_idx": torch.tensor([0, 0, 1, ...]),       # which image in the batch
    "cls":       torch.tensor([[3.0], [17.0], ...]),  # class id per target
    "bboxes":    torch.tensor([[cx, cy, w, h], ...]), # xywh normalized to [0, 1]
}
loss, loss_items = criterion(preds, batch)
# loss_items[0] = box, [1] = cls, [2] = dfl  (gains already applied)
```

### Loss gains

The loss reads gains from `model.args.{box, cls, dfl}`. Defaults match
ultralytics `default.yaml` (`box=7.5, cls=0.5, dfl=1.5`). Tweak before
constructing the criterion:

```python
model.args.box = 5.0
criterion = v8DetectionLoss(model)
```

## How to modify

| Want to try                      | Edit                                                          |
| -------------------------------- | ------------------------------------------------------------- |
| New backbone block               | Add a class in `modules/block.py`, register it in `model.py:_MODULE_REGISTRY`, reference it from `cfg/yolov8.yaml`. |
| Different head architecture      | Edit / subclass `modules/head.py:Detect`. Output contract: dict with `boxes`, `scores`, `feats`. |
| Alternative assignment strategy  | Replace `tal.py:TaskAlignedAssigner` (or swap it in inside `loss.py:v8DetectionLoss`). |
| Different IoU formulation        | `ops.py:bbox_iou` — add a new flag or branch.                 |
| New loss term                    | Extend `loss.py:v8DetectionLoss.__call__`; add to the `loss[0..n]` tensor. |
| Different DFL reg_max            | Edit `cfg/yolov8.yaml` (`reg_max: N`) — propagates to head and loss. |

## Reference

The ultralytics source files this package was distilled from:

```
ultralytics/cfg/models/v8/yolov8.yaml
ultralytics/nn/modules/{conv,block,head}.py
ultralytics/nn/tasks.py             (parse_model, DetectionModel)
ultralytics/utils/tal.py
ultralytics/utils/loss.py
ultralytics/utils/ops.py            (xywh ↔ xyxy, make_divisible)
ultralytics/utils/metrics.py        (bbox_iou)
ultralytics/utils/torch_utils.py    (fuse_conv_and_bn, initialize_weights)
```

Upstream license: AGPL-3.0 (https://ultralytics.com/license). This re-package
preserves the same license.
