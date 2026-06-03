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
├─ configs/sampling.yaml      # all knobs (levels / jitter / stride / label / data)
├─ sampling/
│   ├─ config.py    # YAML -> dataclass
│   ├─ data.py      # self-contained COCO-JSON loader + letterbox(640)
│   ├─ boxes.py     # box generation: gt_linked jitter | stride legacy
│   ├─ levels.py    # shorter-side level assignment + enabled filter
│   ├─ metrics.py   # IoU / coverage
│   └─ labeling.py  # IoU -> positive/background/ignore + stratify
├─ test_sampling.py            # runner (TEST 1 + TEST 2 report, --plot)
│
├─ anchor_box_generate_center_sample.py   # legacy stride sampler (wrapped by boxes.py)
├─ backbone.py                # frozen YOLOv8m backbone (needs torch + ultralytics)
└─ bbox_evaluator.py          # ROI-align bbox evaluator heads (needs torch + torchvision)
```

> The sampling test (`test_sampling.py` + `sampling/`) is self-contained and needs
> only numpy / pyyaml / matplotlib. `backbone.py` and `bbox_evaluator.py` are for the
> later training/scoring phase and additionally require torch / torchvision / ultralytics.
