# teacher/

Teacher-guided anchor selection for YOLOv8 — backbone, bbox evaluator, and the
**sampling test** used to validate two design choices on SeaDronesSee:

1. **Necessity of P3/P4/P5** — do real objects ever land on those levels?
2. **GT-linked jitter sampling** — do generated boxes actually enclose objects?

---

## Setup (uv)

From the repo root:

```bash
uv sync                 # build the env from pyproject.toml + uv.lock
```

Run the sampling test:

```bash
uv run python teacher/test_sampling.py                 # default config (gt_linked)
uv run python teacher/test_sampling.py --mode stride   # legacy sampler, for comparison
uv run python teacher/test_sampling.py --plot          # also save a PNG
```

Useful flags: `--config <path>`, `--mode {gt_linked,stride}`, `--split {train,val}`,
`--max-images N`, `--plot [path]`.

---

## Data

The test reads COCO-JSON annotations only (no images needed):
`data_sds/annotations/instances_{train,val}.json`.

If your data lives elsewhere, point to it without editing any file:

```bash
SDS_ANN_DIR=/path/to/annotations uv run python teacher/test_sampling.py
```

---

## Tuning — edit `configs/sampling.yaml`

Everything is a knob in one YAML file. Common edits:

| Want to… | Change |
|---|---|
| Use the proposed fix vs the legacy sampler | `mode: gt_linked` \| `stride` |
| Drop / keep pyramid levels | `levels.enabled: [orig, p1, p2]` |
| Make jitter boxes tighter (more positives) | lower `jitter.scale` upper bound, e.g. `[0.7, 1.5]` |
| Move jitter centers more / less | `jitter.pos_frac` (0 = centered) |
| More boxes per object | `jitter.n_candidates` |
| Positive / background IoU cutoffs | `label.iou_pos`, `label.iou_neg` |
| Balanced IoU spectrum size | `label.stratify_bins`, `label.n_per_bin` |
| Dataset / split / sample size | `data.ann_dir`, `data.split`, `data.max_images` |

After editing, just re-run the command above.

---

## Files

```
teacher/
├─ configs/
│   ├─ sampling.yaml          # sampling-TEST knobs (levels / jitter / stride / label / data)
│   ├─ train.yaml             # TRAINING knobs (sampling + backbone / dataset / train / val)
│   └─ train_smoke.yaml       # tiny CPU wiring-test config (2 images, 1 epoch)
├─ sampling/
│   ├─ config.py    # YAML -> dataclass
│   ├─ data.py      # self-contained COCO-JSON loader + letterbox(640)
│   ├─ boxes.py     # box generation: gt_linked jitter | gt_linked_v2 | stride legacy
│   ├─ levels.py    # shorter-side level assignment + enabled filter
│   ├─ metrics.py   # IoU / coverage
│   └─ labeling.py  # IoU -> positive/background/ignore + stratify
├─ training/                   # teacher TRAINING pipeline (reuses sampling/ + backbone + evaluator)
│   ├─ config.py   # train.yaml -> dataclasses (reuses sampling dataclasses)
│   ├─ dataset.py  # image letterbox + GT + on-the-fly box sampling/labeling
│   ├─ model.py    # frozen backbone + YOLOBBoxEvaluator(num_fg+1) + teacher_score
│   └─ loop.py     # train/val loop, CE(ignore=-1), per-level metrics, checkpoints
├─ test_sampling.py            # sampling-test runner (TEST 1 + TEST 2 report, --plot)
├─ train_teacher.py            # TRAINING runner (CLI, YAML-driven)
│
├─ anchor_box_generate_center_sample.py   # legacy stride sampler (wrapped by boxes.py)
├─ backbone.py                # frozen YOLOv8m backbone (needs torch + ultralytics)
└─ bbox_evaluator.py          # ROI-align bbox evaluator heads (needs torch + torchvision)
```

> The sampling test (`test_sampling.py` + `sampling/`) is self-contained and needs
> only numpy / pyyaml / matplotlib. `train_teacher.py` + `training/` additionally
> require torch / torchvision / ultralytics (the backbone unpickles an ultralytics .pt).

---

## Training the teacher

Trains the ROI-align bbox evaluator (`bbox_evaluator.py`) on top of the **frozen**
backbone, using boxes sampled on-the-fly around GT. Everything is a knob in
`configs/train.yaml`.

```bash
uv run python teacher/train_teacher.py                                  # default (jittering)
uv run python teacher/train_teacher.py --mode jittering_v2 --epochs 20  # switch sampler
uv run python teacher/train_teacher.py --config teacher/configs/train_smoke.yaml  # CPU wiring test
```

**Point the backbone at your model.** The shipped default `backbone.weights:
experiments/weights/yolov8m.pt` is a **placeholder plain-COCO** model — replace it
with your MS-COCO-pretrained + SDS-fine-tuned `best.pt` (on the GPU box). It loads the
frozen backbone layers; only the evaluator heads are trained. Outputs (evaluator-only
`best.pt` / `last.pt`) go to `<train.out_dir>/<train.exp_name>/`. If the path is
missing the runner fails fast with a clear message.

> **Reproducibility:** `seed` fully fixes `jittering`. `original` and `jittering_v2`
> also draw box centers from the legacy sampler's global RNG, so they are not
> per-item reproducible. **Validation** runs on a `val.max_images`-capped subset
> (default 200) — keep it constant when comparing runs. `--max-images` caps the
> **train** split only (0 = all) and composes with `--set key=value` overrides.

### Tuning — edit `configs/train.yaml` (or `--set key=value` on the CLI)

| Want to… | Change |
|---|---|
| Switch sampler | `mode: original \| jittering \| jittering_v2` |
| Drop / keep pyramid levels | `levels.enabled: [orig, p1, p2]` |
| **IoU positive / background cutoffs** (hyperparameters) | `label.iou_pos`, `label.iou_neg` |
| Tighter jitter boxes (more positives) | `jitter.scale`, e.g. `[0.7, 1.5]` |
| Move jitter centers more / less | `jitter.pos_frac` (0 = centered) |
| Boxes per object | `jitter.n_candidates` |
| Balanced IoU spectrum size | `label.stratify_bins`, `label.n_per_bin` |
| Backbone checkpoint | `backbone.weights` |
| LR / epochs / optimizer / device | `train.lr`, `train.epochs`, `train.optimizer`, `train.device` |

CLI overrides compose with the YAML, e.g.:
```bash
uv run python teacher/train_teacher.py --set label.iou_pos=0.4 --set train.lr=0.005
```

> Background is class index `num_classes` (=5 for SDS), so the evaluator is built with
> `num_classes + 1` outputs and trained with `CrossEntropy(ignore_index=-1)`. The
> per-box `teacher_score = 1 - P(background)` (in `training/model.py`) is the value
> that will later multiply the YOLO TAL align metric.
