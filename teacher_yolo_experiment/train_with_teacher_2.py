"""Teacher-guided YOLOv8 student training — aligned with experiments/train.py.

Training recipe matches experiments/train.py TRAIN_KW exactly:
  SGD  lr0=0.01 → lrf=0.01  warmup_epochs=3  momentum=0.937
  EMA decay=0.9999
  Augmentation: mosaic=1.0 close_mosaic=10 mixup=0.0 + HSV/flip/translate
  Loss gains: box=7.5 cls=0.5 dfl=1.5

Teacher (YOLOBBoxEvaluator) is frozen throughout.  Backbone hooks are
registered inside the criterion so __call__(preds, batch) is 2-arg and
compatible with any standard trainer loop.

Usage
-----
    python teacher_yolo_experiment/train_with_teacher_2.py \
        --model baseline \
        --yolo-weights experiments/weights/yolov8m.pt \
        --teacher-ckpt  teacher/runs/best.pt \
        --data          experiments/data/sds.yaml \
        --gamma 1.0
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# ── sys.path bootstrap ─────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEACHER_DIR = _REPO_ROOT / "teacher"
for _p in (str(_REPO_ROOT), str(_TEACHER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from yolov8 import YOLOv8                                              # noqa: E402
from yolov8.loss import v8DetectionLoss                                # noqa: E402
from yolov8.tal import TaskAlignedAssigner_With_Teacher, make_anchors  # noqa: E402
from teacher.bbox_evaluator import YOLOBBoxEvaluator                   # noqa: E402


# ── constants ──────────────────────────────────────────────────────────────────

# Backbone hook layer indices — same as teacher/backbone.py
_BACKBONE_HOOK_LAYERS: dict[int, str] = {0: "p1", 2: "p2", 4: "p3", 6: "p4"}

# Model cfg registry — mirrors experiments/train.py MODEL_CFGS
MODEL_CFGS = {
    "baseline":   "cfg/yolov8m.yaml",
    "p2":         "cfg/yolov8m-p2.yaml",
    "sppf-k3":    "cfg/yolov8m-sppf-k3.yaml",
    "p2-sppf-k3": "cfg/yolov8m-p2-sppf-k3.yaml",
}

# Training defaults — matches experiments/train.py TRAIN_KW
TRAIN_DEFAULTS = dict(
    epochs=100, patience=30, imgsz=640, batch=16, workers=6,
    lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    box=7.5, cls=0.5, dfl=1.5,
    mosaic=1.0, close_mosaic=10, mixup=0.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5,
    shear=0.0, perspective=0.0, fliplr=0.5, flipud=0.0,
    amp=True, seed=0,
)


# ==============================================================================
# v8DetectionLossWithTeacher
# ==============================================================================

class v8DetectionLossWithTeacher(v8DetectionLoss):
    """v8DetectionLoss with teacher-augmented TAL.

    Backbone hooks are registered inside __init__ and auto-populate
    self._backbone_feats on every model forward pass.  __call__ is
    2-arg  (preds, batch) — backbone feats come from the hooks, so this
    criterion is a drop-in replacement anywhere criterion(preds, batch)
    is called.

    Args:
        model:          YOLOv8 student (provides stride / nc / hyp).
        teacher:        Frozen YOLOBBoxEvaluator.
        gamma:          Teacher score exponent γ.
        temperature:    Softmax temperature for teacher logits.
        tal_topk:       Top-k anchors per GT (default 10).
        enabled_levels: Pyramid levels the teacher uses.
    """

    def __init__(
        self,
        model: YOLOv8,
        teacher: YOLOBBoxEvaluator,
        gamma: float = 1.0,
        temperature: float = 1.0,
        tal_topk: int = 10,
        enabled_levels: list[str] | None = None,
    ) -> None:
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = TaskAlignedAssigner_With_Teacher(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            gamma=gamma,
            stride=self.stride.tolist(),
            teacher_network=teacher,
            temperature=temperature,
        )
        self._backbone_feats: dict[str, torch.Tensor] = {}
        self._hook_handles: list = []
        if enabled_levels:
            self._register_backbone_hooks(model, enabled_levels)

    def _register_backbone_hooks(self, model: YOLOv8, enabled_levels: list[str]) -> None:
        if "orig" in enabled_levels:
            def _pre_orig(m, inputs):
                self._backbone_feats["orig"] = inputs[0].detach()
            self._hook_handles.append(
                model.model[0].register_forward_pre_hook(_pre_orig)
            )
        for layer_idx, level_name in _BACKBONE_HOOK_LAYERS.items():
            if level_name not in enabled_levels:
                continue
            def _post(m, inp, out, _n=level_name):
                self._backbone_feats[_n] = out.detach()
            self._hook_handles.append(
                model.model[layer_idx].register_forward_hook(_post)
            )

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def __call__(  # type: ignore[override]
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Snapshot feats captured during model(imgs); clear for next step.
        backbone_feats = dict(self._backbone_feats)
        self._backbone_feats.clear()

        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
            backbone_feats,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()


# ==============================================================================
# EMA
# ==============================================================================

class ModelEMA:
    """Exponential Moving Average of model weights (mirrors ultralytics EMA)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000.0) -> None:
        self.ema = copy.deepcopy(model).eval()
        self.updates = 0
        self._decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self._decay(self.updates)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1.0 - d) * msd[k].detach()


# ==============================================================================
# Optimizer + LR schedule
# ==============================================================================

def build_optimizer(
    model: YOLOv8,
    lr0: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.SGD:
    """SGD with bias params in a separate group — no weight decay on biases.

    Bias group is also used for per-step warmup tracking (warmup_bias_lr starts
    higher than the weight group which warms up from 0).
    """
    bias_params, weight_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            bias_params.append(p)
        else:
            weight_params.append(p)
    return torch.optim.SGD(
        [
            {"params": weight_params, "is_bias": False},
            {"params": bias_params,   "is_bias": True,  "weight_decay": 0.0},
        ],
        lr=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )


def _lr_lambda(epoch: int, epochs: int, warmup_epochs: float, lrf: float) -> float:
    """Linear warmup then cosine decay, expressed as a multiplier on lr0."""
    if epoch < warmup_epochs:
        return max(1e-5, (epoch + 1) / warmup_epochs)
    progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
    return lrf + (1.0 - lrf) * (1.0 + math.cos(math.pi * progress)) / 2.0


def _apply_warmup_bias_lr(
    optimizer: torch.optim.SGD,
    global_step: int,
    warmup_steps: int,
    lr0: float,
    warmup_bias_lr: float,
    momentum: float,
    warmup_momentum: float,
) -> None:
    """Per-step interpolation during warmup (ultralytics-style)."""
    frac = min(global_step / max(warmup_steps, 1), 1.0)
    for g in optimizer.param_groups:
        if g.get("is_bias", False):
            g["lr"] = warmup_bias_lr + (lr0 - warmup_bias_lr) * frac
        else:
            g["lr"] = lr0 * frac
        g["momentum"] = warmup_momentum + (momentum - warmup_momentum) * frac


# ==============================================================================
# Data loading (ultralytics dataset — same augmentation as experiments/train.py)
# ==============================================================================

def build_yolo_dataloader(
    data_yaml: str,
    split: str,
    batch: int,
    imgsz: int = 640,
    workers: int = 4,
    hyp_overrides: dict | None = None,
    cache: str | bool = False,
):
    """Ultralytics YOLODataset loader — provides mosaic, mixup, HSV, flips, etc.

    hyp_overrides: augmentation knobs (mosaic, mixup, hsv_h, …).  Pass
        ``{"mosaic": 0.0, "mixup": 0.0}`` to disable mosaic for the last
        close_mosaic epochs.
    """
    from ultralytics.data import build_dataloader             # type: ignore
    from ultralytics.data.dataset import YOLODataset          # type: ignore
    from ultralytics.data.utils import check_det_dataset      # type: ignore
    from ultralytics.utils import DEFAULT_CFG                 # type: ignore

    data = check_det_dataset(data_yaml)
    img_path = data[split]
    is_train = split == "train"

    cfg = copy.copy(DEFAULT_CFG)
    cfg.imgsz = imgsz
    cfg.cache = cache
    if hyp_overrides:
        for k, v in hyp_overrides.items():
            setattr(cfg, k, v)

    dataset = YOLODataset(
        img_path=img_path,
        imgsz=imgsz,
        batch_size=batch,
        augment=is_train,
        hyp=cfg,
        rect=False,
        stride=32,
        pad=0.0 if is_train else 0.5,
        task="detect",
        data=data,
        classes=None,
        fraction=1.0,
    )
    return build_dataloader(dataset, batch, workers, shuffle=is_train, rank=-1)


# ==============================================================================
# Checkpoint helpers
# ==============================================================================

def load_teacher(
    ckpt_path: str, device: torch.device
) -> tuple[YOLOBBoxEvaluator, list[str]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enabled_levels: list[str] = ckpt["enabled_levels"]
    num_classes: int = ckpt["num_classes"]
    evaluator = YOLOBBoxEvaluator(num_classes=num_classes, enabled_levels=enabled_levels)
    evaluator.load_state_dict(ckpt["evaluator"], strict=False)
    evaluator.to(device).eval()
    for p in evaluator.parameters():
        p.requires_grad_(False)
    print(f"[teacher] nc={num_classes}  levels={enabled_levels}  <- {ckpt_path}")
    return evaluator, enabled_levels


def load_student(
    weights_path: str | None,
    cfg: str = "cfg/yolov8.yaml",
    scale: str = "m",
    nc: int | None = None,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> YOLOv8:
    """Build YOLOv8 with shape-aware weight transfer.

    Accepts ultralytics checkpoints (nested 'model' key) and plain state dicts.
    Shape-mismatched keys (e.g. head nc mismatch) are silently skipped.
    """
    model = YOLOv8(cfg=cfg, scale=scale, nc=nc, verbose=verbose)

    if weights_path:
        raw = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "model" in raw:
            src_sd = {k: v.float() for k, v in raw["model"].state_dict().items()}
        elif isinstance(raw, dict) and "ema" in raw and raw["ema"] is not None:
            src_sd = {k: v.float() for k, v in raw["ema"].items()}
        elif isinstance(raw, dict) and "state_dict" in raw:
            src_sd = {k: v.float() for k, v in raw["state_dict"].items()}
        elif isinstance(raw, dict):
            src_sd = {k: v.float() for k, v in raw.items() if isinstance(v, torch.Tensor)}
        else:
            src_sd = {k: v.float() for k, v in raw.state_dict().items()}

        our_sd = model.state_dict()
        mapped = {k: v for k, v in src_sd.items()
                  if k in our_sd and v.shape == our_sd[k].shape}
        model.load_state_dict(mapped, strict=False)
        print(f"[student] loaded {len(mapped)}/{len(our_sd)} tensors  <- {weights_path}")

    return model.to(device)


def save_checkpoint(
    path: str,
    model: YOLOv8,
    ema: ModelEMA | None,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model":     model.state_dict(),
            "ema":       ema.ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "epoch":     epoch,
            "metrics":   metrics,
        },
        path,
    )


# ==============================================================================
# Training / validation loops
# ==============================================================================

def train_one_epoch(
    model: YOLOv8,
    criterion: v8DetectionLossWithTeacher,
    loader,
    optimizer: torch.optim.SGD,
    ema: ModelEMA | None,
    device: torch.device,
    scaler,
    amp: bool,
    epoch: int,
    global_step: int,
    warmup_steps: int,
    lr0: float,
    warmup_bias_lr: float,
    momentum: float,
    warmup_momentum: float,
    log_every: int = 50,
) -> tuple[dict[str, float], int]:
    model.train()
    total_loss = box_l = cls_l = dfl_l = 0.0
    steps = 0

    for batch in loader:
        imgs = batch["img"].to(device, non_blocking=True).float().div_(255.0)
        for key in ("batch_idx", "cls", "bboxes"):
            batch[key] = batch[key].to(device, non_blocking=True)

        global_step += 1
        if global_step <= warmup_steps:
            _apply_warmup_bias_lr(optimizer, global_step, warmup_steps,
                                  lr0, warmup_bias_lr, momentum, warmup_momentum)

        optimizer.zero_grad(set_to_none=True)

        # Hooks auto-populate criterion._backbone_feats during model(imgs).
        with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            preds = model(imgs)
            loss, loss_items = criterion(preds, batch)

        if amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        steps += 1
        total_loss += float(loss.detach()) / imgs.shape[0]   # undo batch-size scaling
        box_l += float(loss_items[0])
        cls_l += float(loss_items[1])
        dfl_l += float(loss_items[2])

        if log_every and steps % log_every == 0:
            n = steps
            print(
                f"  [epoch {epoch}] step {steps:4d}"
                f"  loss {total_loss/n:.4f}"
                f"  box {box_l/n:.4f}  cls {cls_l/n:.4f}  dfl {dfl_l/n:.4f}"
                f"  lr {optimizer.param_groups[0]['lr']:.6f}"
            )

    n = max(steps, 1)
    return (
        {"loss": total_loss/n, "box": box_l/n, "cls": cls_l/n, "dfl": dfl_l/n},
        global_step,
    )


@torch.no_grad()
def validate(
    model: YOLOv8,
    criterion: v8DetectionLossWithTeacher,
    loader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = box_l = cls_l = dfl_l = 0.0
    steps = 0

    for batch in loader:
        imgs = batch["img"].to(device, non_blocking=True).float().div_(255.0)
        for key in ("batch_idx", "cls", "bboxes"):
            batch[key] = batch[key].to(device, non_blocking=True)
        # Hooks still fire in eval mode — backbone_feats are populated.
        preds = model(imgs)
        loss, loss_items = criterion(preds, batch)
        steps += 1
        total_loss += float(loss) / imgs.shape[0]
        box_l += float(loss_items[0])
        cls_l += float(loss_items[1])
        dfl_l += float(loss_items[2])

    n = max(steps, 1)
    return {"loss": total_loss/n, "box": box_l/n, "cls": cls_l/n, "dfl": dfl_l/n}


# ==============================================================================
# Full training driver
# ==============================================================================

def train(
    student_weights: str | None,
    teacher_ckpt: str,
    data_yaml: str,
    cfg: str = "cfg/yolov8.yaml",
    scale: str = "m",
    nc: int | None = None,
    # ── recipe (matches TRAIN_DEFAULTS / experiments/train.py) ──────────────
    epochs: int = 100,
    patience: int = 30,
    batch: int = 16,
    imgsz: int = 640,
    lr0: float = 0.01,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    warmup_momentum: float = 0.8,
    warmup_bias_lr: float = 0.1,
    box: float = 7.5,
    cls: float = 0.5,
    dfl: float = 1.5,
    mosaic: float = 1.0,
    close_mosaic: int = 10,
    mixup: float = 0.0,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale_aug: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    fliplr: float = 0.5,
    flipud: float = 0.0,
    # ── teacher knobs ─────────────────────────────────────────────────────
    gamma: float = 1.0,
    temperature: float = 1.0,
    tal_topk: int = 10,
    # ── infrastructure ────────────────────────────────────────────────────
    workers: int = 6,
    device_str: str = "auto",
    amp: bool = True,
    cache: str | bool = False,
    out_dir: str = "teacher_yolo_experiment/runs",
    exp_name: str = "teacher_tal",
    log_every: int = 50,
    seed: int = 0,
) -> float:
    # ── reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(seed)

    # ── device ────────────────────────────────────────────────────────────
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    use_amp = amp and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    print(f"[train] device={device}  amp={use_amp}")

    # ── teacher ───────────────────────────────────────────────────────────
    teacher, enabled_levels = load_teacher(teacher_ckpt, device)

    # ── student ───────────────────────────────────────────────────────────
    model = load_student(student_weights, cfg=cfg, scale=scale,
                         nc=nc, device=device, verbose=True)
    # Wire in loss-gain hyperparameters (model.args is read by v8DetectionLoss)
    model.args.box = box
    model.args.cls = cls
    model.args.dfl = dfl

    # ── criterion (hooks registered here) ────────────────────────────────
    criterion = v8DetectionLossWithTeacher(
        model, teacher,
        gamma=gamma, temperature=temperature,
        tal_topk=tal_topk, enabled_levels=enabled_levels,
    )

    # ── EMA ───────────────────────────────────────────────────────────────
    ema = ModelEMA(model, decay=0.9999)

    # ── optimizer + LR scheduler ──────────────────────────────────────────
    optimizer = build_optimizer(model, lr0, momentum, weight_decay)

    # Per-epoch LambdaLR handles warmup + cosine in one schedule.
    # During the warmup steps per-step overrides are applied on top;
    # the scheduler resets LR correctly after each epoch.
    lr_fn = lambda e: _lr_lambda(e, epochs, warmup_epochs, lrf)  # noqa: E731
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── augmentation hyperparameters ──────────────────────────────────────
    aug_hyp = dict(
        mosaic=mosaic, mixup=mixup,
        hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v,
        degrees=degrees, translate=translate, scale=scale_aug,
        shear=shear, perspective=perspective, fliplr=fliplr, flipud=flipud,
    )

    # ── data loaders ──────────────────────────────────────────────────────
    train_loader = build_yolo_dataloader(
        data_yaml, "train", batch, imgsz=imgsz, workers=workers,
        hyp_overrides=aug_hyp, cache=cache,
    )
    val_loader = build_yolo_dataloader(
        data_yaml, "val", batch, imgsz=imgsz, workers=max(workers // 2, 1),
        hyp_overrides={},
    )

    warmup_steps = max(round(warmup_epochs * len(train_loader)), 100)

    # ── output ────────────────────────────────────────────────────────────
    run_dir = os.path.join(out_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")
    print(f"[train] run -> {run_dir}")
    print(f"[train] epochs={epochs}  batch={batch}  imgsz={imgsz}"
          f"  warmup={warmup_epochs}  γ={gamma}  T={temperature}")

    best_val_loss = float("inf")
    no_improve = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        # Disable mosaic for the final close_mosaic epochs (same as ultralytics)
        if close_mosaic and epoch == epochs - close_mosaic + 1:
            print(f"[train] epoch {epoch}: disabling mosaic augmentation")
            train_loader = build_yolo_dataloader(
                data_yaml, "train", batch, imgsz=imgsz, workers=workers,
                hyp_overrides={**aug_hyp, "mosaic": 0.0, "mixup": 0.0},
            )

        tr, global_step = train_one_epoch(
            model, criterion, train_loader, optimizer, ema,
            device, scaler, amp, epoch, global_step,
            warmup_steps, lr0, warmup_bias_lr, momentum, warmup_momentum,
            log_every,
        )
        va = validate(model, criterion, val_loader, device)

        # Step scheduler after warmup period (per-step warmup owns the LR until then)
        if global_step > warmup_steps:
            scheduler.step()

        is_best = va["loss"] < best_val_loss
        tag = "  *best*" if is_best else ""
        print(
            f"\nepoch {epoch:03d}/{epochs}"
            f"  train {tr['loss']:.4f}"
            f" (box {tr['box']:.4f} cls {tr['cls']:.4f} dfl {tr['dfl']:.4f})"
            f"  |  val {va['loss']:.4f}"
            f" (box {va['box']:.4f} cls {va['cls']:.4f} dfl {va['dfl']:.4f})"
            f"  lr {optimizer.param_groups[0]['lr']:.6f}"
            f"{tag}"
        )

        save_checkpoint(last_path, model, ema, optimizer, epoch, {"train": tr, "val": va})
        if is_best:
            best_val_loss = va["loss"]
            no_improve = 0
            save_checkpoint(best_path, model, ema, optimizer, epoch, {"train": tr, "val": va})
        else:
            no_improve += 1

        if patience and no_improve >= patience:
            print(f"\n[early stop] no improvement for {patience} epochs — stopping at {epoch}.")
            break

    criterion.remove_hooks()
    print(f"\n[done] best val loss {best_val_loss:.4f}  ->  {best_path}")
    return best_val_loss


# ==============================================================================
# CLI — mirrors experiments/train.py interface
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="YOLOv8 + teacher-augmented TAL — training pipeline aligned with experiments/train.py"
    )
    # Model selection (same variants as experiments/train.py)
    ap.add_argument("--model", default="baseline", choices=list(MODEL_CFGS),
                    help="Architecture variant.")
    ap.add_argument("--yolo-weights", default=None, metavar="PATH",
                    help="Student init weights (.pt). Omit for random init.")
    ap.add_argument("--teacher-ckpt", required=True, metavar="PATH",
                    help="Teacher checkpoint (teacher/runs/…/best.pt).")
    ap.add_argument("--data", default="experiments/data/sds.yaml",
                    help="Ultralytics dataset yaml.")
    ap.add_argument("--scale", default="m")
    ap.add_argument("--nc",    type=int, default=None,
                    help="Num classes (reads from cfg yaml if omitted).")
    # Training recipe overrides (defaults match TRAIN_DEFAULTS / TRAIN_KW)
    ap.add_argument("--epochs",        type=int,   default=TRAIN_DEFAULTS["epochs"])
    ap.add_argument("--patience",      type=int,   default=TRAIN_DEFAULTS["patience"])
    ap.add_argument("--batch",         type=int,   default=TRAIN_DEFAULTS["batch"])
    ap.add_argument("--imgsz",         type=int,   default=TRAIN_DEFAULTS["imgsz"])
    ap.add_argument("--workers",       type=int,   default=TRAIN_DEFAULTS["workers"])
    ap.add_argument("--lr0",           type=float, default=TRAIN_DEFAULTS["lr0"])
    ap.add_argument("--lrf",           type=float, default=TRAIN_DEFAULTS["lrf"])
    ap.add_argument("--cache",         default="ram",
                    help="Dataset cache: ram | disk | false")
    # Teacher knobs
    ap.add_argument("--gamma",         type=float, default=1.0,
                    help="Teacher score exponent γ.")
    ap.add_argument("--temperature",   type=float, default=1.0,
                    help="Softmax temperature for teacher logits.")
    ap.add_argument("--tal-topk",      type=int,   default=10)
    # Infrastructure
    ap.add_argument("--device",        default="auto")
    ap.add_argument("--amp",           action="store_true",  default=True)
    ap.add_argument("--no-amp",        dest="amp", action="store_false")
    ap.add_argument("--out-dir",       default="teacher_yolo_experiment/runs")
    ap.add_argument("--name",          default="teacher_tal")
    ap.add_argument("--seed",          type=int,   default=TRAIN_DEFAULTS["seed"])
    ap.add_argument("--log-every",     type=int,   default=50)
    args = ap.parse_args()

    # Resolve cfg against experiments/ directory (same as experiments/train.py)
    experiments_dir = _REPO_ROOT / "experiments"
    cfg_path = experiments_dir / MODEL_CFGS[args.model]
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"cfg not found: {cfg_path}\n"
            f"  Run from the repo root, or check that experiments/cfg/ exists."
        )

    cache = False if args.cache.lower() == "false" else args.cache

    train(
        student_weights=args.yolo_weights,
        teacher_ckpt=args.teacher_ckpt,
        data_yaml=args.data,
        cfg=str(cfg_path),
        scale=args.scale,
        nc=args.nc,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        gamma=args.gamma,
        temperature=args.temperature,
        tal_topk=args.tal_topk,
        workers=args.workers,
        device_str=args.device,
        amp=args.amp,
        cache=cache,
        out_dir=args.out_dir,
        exp_name=args.name,
        log_every=args.log_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
