# experiments/ — SeaDronesSee Ablation Runner

End-to-end pipeline for fine-tuning four YOLOv8m variants on SeaDronesSee v2.
Built on top of the official `ultralytics` Trainer; our `yolov8/` package
served as the validation tool that confirmed the model+loss equivalence.

```
experiments/
├── cfg/                           model architecture yamls (4 variants × 2 names each)
│   ├── yolov8m.yaml               baseline (scale-tagged copy of yolov8/cfg/yolov8.yaml)
│   ├── yolov8m-p2.yaml            + P2 head (Detect P2/P3/P4/P5)
│   ├── yolov8m-sppf-k3.yaml       SPPF kernel 5 -> 3
│   └── yolov8m-p2-sppf-k3.yaml    P2 + SPPF k=3 combined
├── data/
│   └── sds.yaml                   ultralytics dataset descriptor for SeaDronesSee
├── scripts/
│   ├── convert_sds.py             COCO JSON -> YOLO txt converter
│   └── check_pretrained_transfer.py  sanity check for weight transfer
├── train.py                       single entry point: --model {baseline|p2|sppf-k3|p2-sppf-k3}
├── weights/                       drop yolov8m.pt here (downloaded by setup step)
└── runs/                          ultralytics writes checkpoints + plots here
```

---

## 1. SeaDronesSee v2 download

The dataset is hosted at **https://seadronessee.cs.uni-tuebingen.de/dataset**.

It requires creating a (free) account; the team approves access manually and
also has a Google Form fallback. The page lists three sub-datasets — we want
**"Object Detection v2"** (the newest, COCO-style annotations).

What you should end up with on disk:

```
sds-raw/
├── images/
│   ├── train/   (~5,630 jpgs)
│   ├── val/     (~859 jpgs)
│   └── test/    (~1,796 jpgs — test labels are withheld)
└── annotations/
    ├── instances_train.json   (COCO-format)
    ├── instances_val.json
    └── instances_test_info.json   (images only, no annotations)
```

The exact filenames may differ slightly between releases — adapt the paths in
the conversion command below if needed. The Roboflow mirror
(https://universe.roboflow.com/?q=seadronessee) hosts pre-split copies in
YOLO format that can be used as a fallback if the Tübingen download is slow,
but the official source is preferred for paper-comparable numbers.

Heads up: the test labels are private (used by the SeaDronesSee benchmark
leaderboard). For our midterm we evaluate on the **val** split.

---

## 2. Elice instance setup (one-time)

### 2.1 Create the instance

Choose **G-NAHPM-40** (A100 80GB MIG 3g-40GB) with **256 GiB** storage. Boot.

### 2.2 Connect

Either the web terminal that Elice provides, or via SSH from your laptop:

```bash
ssh <user>@<instance-ip>
```

(Elice usually shows the SSH command on the instance details page.)

### 2.3 Verify the GPU

```bash
nvidia-smi
```

You should see one A100 slice with ~40 GiB. If `nvidia-smi` fails, the
instance image is broken — recreate.

### 2.4 Install Python + uv + dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

mkdir -p ~/work && cd ~/work
# upload our SOD_Project directory here (see step 2.5)

cd SOD_Project
uv venv --python 3.11 .venv
source .venv/bin/activate

# torch with CUDA — pick the wheel that matches the instance's CUDA version.
# Elice's A100 image typically ships with CUDA 12.x, so the default index works:
uv pip install torch torchvision pyyaml numpy ultralytics

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True  NVIDIA A100 80GB PCIe MIG 3g.40gb
```

### 2.5 Get the code onto the instance

From your **laptop**:

```bash
scp -r /Users/erdembileg/Desktop/SOD_Project <user>@<instance-ip>:~/work/
```

Or, if you've pushed to GitHub:

```bash
cd ~/work
git clone <your-repo-url> SOD_Project
```

### 2.6 Download COCO-pretrained yolov8m.pt

```bash
cd ~/work/SOD_Project/experiments
mkdir -p weights
curl -L -o weights/yolov8m.pt \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt
ls -lh weights/yolov8m.pt   # should be ~52 MB
```

### 2.7 Sanity-check weight transfer for all 4 variants

```bash
python experiments/scripts/check_pretrained_transfer.py
```

Expected output (roughly):

```
baseline      : 474 / 475 tensors transferred  (99.8%)
p2            : 319 / 581 tensors transferred  (54.9%)   <- P2 branch is fresh
sppf-k3       : 474 / 475 tensors transferred  (99.8%)   <- SPPF k change is weight-free
p2-sppf-k3    : 319 / 581 tensors transferred  (54.9%)
```

If you see much lower transfer (<20%), the cfg's scale wasn't recognized.
Make sure the cfg filenames contain the `m` letter (`yolov8m*.yaml`).

---

## 3. Prepare SeaDronesSee data

```bash
# Put the raw SDS archive under ~/datasets/sds-raw with the layout shown in §1.
mkdir -p ~/datasets/sds-raw && cd ~/datasets/sds-raw
# ... extract whatever you downloaded from the SDS site here ...

# Run the converter for each split.
cd ~/work/SOD_Project
python experiments/scripts/convert_sds.py \
    --coco ~/datasets/sds-raw/annotations/instances_train.json \
    --images ~/datasets/sds-raw/images/train \
    --out ~/datasets/sds \
    --split train

python experiments/scripts/convert_sds.py \
    --coco ~/datasets/sds-raw/annotations/instances_val.json \
    --images ~/datasets/sds-raw/images/val \
    --out ~/datasets/sds \
    --split val
```

The converter prints the actual `categories` list from the JSON — **verify it
matches `CLASS_MAP` in convert_sds.py**. SeaDronesSee occasionally renumbers
ids; if the JSON's `categories` differ, edit `CLASS_MAP` and rerun.

After both splits convert, the layout under `~/datasets/sds` should be:

```
~/datasets/sds/
├── images/{train,val}/*.jpg
└── labels/{train,val}/*.txt
```

Then check that `experiments/data/sds.yaml`'s `path:` line points to
`~/datasets/sds` (already set to `/root/datasets/sds` — adjust if your
home dir differs).

---

## 4. Run training

Always launch inside `tmux` so the run survives SSH drops:

```bash
tmux new -s sds
cd ~/work/SOD_Project
source .venv/bin/activate

# Smoke test (1 epoch, prints if anything is wrong fast):
python experiments/train.py --model baseline --epochs 1 --name smoke
```

If that finishes without errors, kick off the real runs **one at a time**
(MIG slice has one logical GPU, so no parallel runs on this instance):

```bash
python experiments/train.py --model baseline
python experiments/train.py --model sppf-k3
python experiments/train.py --model p2
python experiments/train.py --model p2-sppf-k3      # optional, time permitting
```

Each run writes to `experiments/runs/<model_name>/` with:

* `weights/best.pt`, `weights/last.pt`
* `results.csv` — per-epoch metrics (mAP50, mAP50-95, losses)
* `results.png`, `confusion_matrix.png`, `val_batch*.jpg` etc.

Detach tmux with `Ctrl+B` then `D`; reattach with `tmux attach -t sds`.

### Pinned hyperparameters

All four runs use the *exact same* recipe — see `TRAIN_KW` at the top of
`train.py`. The only thing that varies across runs is the `--model` cfg.
That's what makes the comparison meaningful.

| key                | value | why |
|--------------------|-------|-----|
| `epochs`           | 100   | with `patience=30`, real runs usually stop ~70-80 |
| `imgsz`            | 640   | matches the most common SDS-baseline comparison |
| `batch`            | 16    | fits 40 GiB MIG + Mosaic comfortably |
| `optimizer`        | SGD   | ultralytics default; stable for finetune |
| `lr0`              | 0.01  | finetune from COCO weights with cosine decay |
| `mosaic` / `close` | 1.0 / 10 | mosaic on, disabled last 10 epochs for stable convergence |
| `flipud`           | 0.0   | aerial imagery has a real up/down |
| `cache`            | ram   | 96 GiB RAM → no I/O bottleneck after epoch 0 |

To override anything for a quick test: `--epochs 5 --batch 8`.

---

## 5. Pull results back to your laptop

After every model finishes, snapshot the outputs locally so an instance
mishap doesn't wipe them:

```bash
# from laptop
scp -r <user>@<instance-ip>:~/work/SOD_Project/experiments/runs ./runs-snapshot
```

The CSVs (`results.csv`) + the final `confusion_matrix.png` are what feeds
into the slides.

---

## 6. Stop the instance when not training

Elice bills per hour the instance is **running**. Storage is billed
separately and persists across stop/start, so:

* Training done? → **Stop** the instance.
* Need to keep iterating tomorrow? → Stop now, Start tomorrow.
* Project done? → Snapshot results to laptop **then** delete.

A 40 GiB MIG + 256 GiB storage idle for 8 h costs ~₩11k. Not free.

---

## 7. Quick troubleshooting

| symptom | likely cause / fix |
|---|---|
| `Transferred 95/355 items` (low %) | cfg scale not detected; use `yolov8m*.yaml` filenames |
| `CUDA out of memory` | drop `--batch` to 8 (or 4 for the p2 variants) |
| `dataset not found` | edit `experiments/data/sds.yaml:path` to the absolute dataset root |
| `categories in JSON ... CLASS_MAP keys` mismatch | edit `CLASS_MAP` in `convert_sds.py` to match the JSON's ids |
| training stalls at first epoch | check `nvidia-smi` — if util is 0%, dataloader is starved; lower `workers` |
| AMP NaNs | rerun with `--imgsz 608` or set `amp=False` in train.py |
