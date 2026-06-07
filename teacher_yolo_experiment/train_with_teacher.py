"""Teacher-guided YOLOv8 student training.

Trains a YOLOv8 student whose Task-Aligned Assigner is augmented with scores
from a frozen YOLOBBoxEvaluator teacher:

    align_metric = cls_score^α × IoU^β × teacher_cls_score^γ

Nothing in yolov8/ or teacher/ is modified — all changes live here.
Other training pipelines (experiments/train.py) are completely unaffected.

Usage
-----
    python teacher_yolo_experiment/train_with_teacher.py \
        --yolo-weights  experiments/weights/yolov8m.pt \
        --teacher-ckpt  teacher/runs/best.pt \
        --data          experiments/data/sds.yaml \
        --epochs 100 --batch 16 --gamma 1.0

Expected batch format from any DataLoader
------------------------------------------
    batch["img"]        (B, 3, H, W)  uint8 [0, 255]
    batch["batch_idx"]  (N,)          int64  image index per GT row
    batch["cls"]        (N, 1)        int64  class index per GT row
    batch["bboxes"]     (N, 4)        float32 xywh normalized [0, 1]

This is the standard ultralytics detection batch format.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── sys.path bootstrap ─────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEACHER_DIR = _REPO_ROOT / "teacher"
for _p in (str(_REPO_ROOT), str(_TEACHER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from yolov8 import YOLOv8                                          # noqa: E402
from yolov8.loss import v8DetectionLoss                            # noqa: E402
from yolov8.tal import TaskAlignedAssigner_With_Teacher, make_anchors  # noqa: E402
from teacher.bbox_evaluator import YOLOBBoxEvaluator               # noqa: E402


# ── backbone pyramid layer indices (mirrors teacher/backbone.py) ───────────────
# YOLOv8 backbone (all scales share the same layer order, only channel widths differ):
#   layer 0 → Conv  P1/2  (stride 2)
#   layer 2 → C2f   P2/4  (stride 4)
#   layer 4 → C2f   P3/8  (stride 8)
#   layer 6 → C2f   P4/16 (stride 16)
_BACKBONE_HOOK_LAYERS: dict[int, str] = {
    0: "p1",
    2: "p2",
    4: "p3",
    6: "p4",
}


# ==============================================================================
# BackboneHookCapture
# ==============================================================================

class BackboneHookCapture:
    """Registers forward hooks on the YOLOv8 backbone to capture raw feature maps.

    Called once after model construction; produces a `feats` dict that is
    repopulated on every forward pass and passed to the teacher evaluator.
    Tensors are detached — the teacher runs under torch.no_grad() anyway.

    Args:
        model:          Built YOLOv8 student model.
        enabled_levels: Levels to capture, e.g. ['p2','p3','p4'].
                        Must match the teacher checkpoint's enabled_levels.
    """

    def __init__(self, model: YOLOv8, enabled_levels: list[str]) -> None:
        self.enabled_levels = enabled_levels
        self._feats: dict[str, torch.Tensor] = {}
        self._handles: list = []

        # 'orig' = the raw input image, captured via a pre-hook on layer 0
        if "orig" in enabled_levels:
            def _capture_orig(module, inputs):
                self._feats["orig"] = inputs[0].detach()
            self._handles.append(
                model.model[0].register_forward_pre_hook(_capture_orig)
            )

        # p1-p4 via post-hooks on their respective backbone layers
        for layer_idx, level_name in _BACKBONE_HOOK_LAYERS.items():
            if level_name not in enabled_levels:
                continue
            def _capture_level(module, input, output, name=level_name):
                self._feats[name] = output.detach()
            self._handles.append(
                model.model[layer_idx].register_forward_hook(_capture_level)
            )

    @property
    def feats(self) -> dict[str, torch.Tensor]:
        return self._feats

    def clear(self) -> None:
        self._feats.clear()

    def remove(self) -> None:
        """Deregister all hooks (call before deleting the capture object)."""
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ==============================================================================
# v8DetectionLossWithTeacher
# ==============================================================================

class v8DetectionLossWithTeacher(v8DetectionLoss):
    """v8DetectionLoss whose TAL assigner is augmented with teacher scores.

    Only two differences from the base class:
      1. `self.assigner` is replaced with TaskAlignedAssigner_With_Teacher.
      2. `__call__` accepts a third positional arg `backbone_feats` which is
         forwarded to the teacher inside the assigner.

    yolov8/loss.py is NOT modified.

    Args:
        model:       YOLOv8 student (used to extract nc, stride, etc.)
        teacher:     Frozen YOLOBBoxEvaluator loaded from a checkpoint.
        gamma:       Exponent on the teacher score term (γ).
        temperature: Softmax temperature applied to teacher logits before pow.
        tal_topk:    Top-k anchors kept per GT (default 10).
    """

    def __init__(
        self,
        model: YOLOv8,
        teacher: YOLOBBoxEvaluator,
        gamma: float = 1.0,
        temperature: float = 1.0,
        tal_topk: int = 10,
    ) -> None:
        super().__init__(model, tal_topk=tal_topk)
        # Overwrite the base-class assigner with the teacher-augmented version.
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

    def __call__(  # type: ignore[override]
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, Any],
        backbone_feats: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        # assigner now receives backbone_feats as the extra 'feats' argument
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
            backbone_feats,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()


# ==============================================================================
# Checkpoint helpers
# ==============================================================================

def load_teacher(
    ckpt_path: str, device: torch.device
) -> tuple[YOLOBBoxEvaluator, list[str]]:
    """Load a frozen YOLOBBoxEvaluator from a teacher training checkpoint.

    Returns:
        evaluator:      Frozen, eval-mode YOLOBBoxEvaluator.
        enabled_levels: Pyramid levels the teacher was trained with.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enabled_levels: list[str] = ckpt["enabled_levels"]
    num_classes: int = ckpt["num_classes"]

    evaluator = YOLOBBoxEvaluator(num_classes=num_classes, enabled_levels=enabled_levels)
    evaluator.load_state_dict(ckpt["evaluator"], strict=False)
    evaluator.to(device).eval()
    for p in evaluator.parameters():
        p.requires_grad_(False)

    print(f"[teacher] {ckpt_path}  nc={num_classes}  levels={enabled_levels}")
    return evaluator, enabled_levels


def load_student(
    weights_path: str | None,
    cfg: str = "cfg/yolov8.yaml",
    scale: str = "m",
    nc: int | None = None,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> YOLOv8:
    """Build YOLOv8 and transfer weights with shape-aware loading.

    Accepts ultralytics checkpoints (nested under 'model' key) as well as
    plain state-dicts.  Shape-mismatched keys (e.g. head nc mismatch) are
    silently skipped so a COCO-pretrained or SDS-fine-tuned checkpoint loads
    cleanly into any nc configuration.
    """
    model = YOLOv8(cfg=cfg, scale=scale, nc=nc, verbose=verbose)

    if weights_path:
        raw = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "model" in raw:
            src_sd = {k: v.float() for k, v in raw["model"].state_dict().items()}
        elif isinstance(raw, dict) and "state_dict" in raw:
            src_sd = {k: v.float() for k, v in raw["state_dict"].items()}
        elif isinstance(raw, dict):
            src_sd = {k: v.float() for k, v in raw.items()
                      if isinstance(v, torch.Tensor)}
        else:
            src_sd = {k: v.float() for k, v in raw.state_dict().items()}

        our_sd = model.state_dict()
        mapped = {k: v for k, v in src_sd.items()
                  if k in our_sd and v.shape == our_sd[k].shape}
        model.load_state_dict(mapped, strict=False)
        print(f"[student] {weights_path}  loaded {len(mapped)}/{len(our_sd)} tensors")

    return model.to(device)


def save_checkpoint(
    path: str, model: YOLOv8, optimizer: torch.optim.Optimizer,
    epoch: int, metrics: dict
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }, path)


# ==============================================================================
# Data loading (ultralytics format)
# ==============================================================================

def build_ultralytics_loader(
    data_yaml: str,
    split: str,
    batch: int,
    imgsz: int = 640,
    workers: int = 4,
):
    """Build an ultralytics-format DataLoader for a YOLO dataset yaml.

    Requires ultralytics to be installed (already a project dependency).
    Produced batches follow the standard format documented at the top of this
    file.
    """
    from ultralytics.data import build_dataloader, YOLODataset          # type: ignore
    from ultralytics.data.utils import check_det_dataset                # type: ignore

    data = check_det_dataset(data_yaml)
    img_path = data[split]
    dataset = YOLODataset(
        img_path=img_path,
        imgsz=imgsz,
        augment=(split == "train"),
        rect=False,
    )
    return build_dataloader(
        dataset, batch=batch, workers=workers,
        shuffle=(split == "train"), rank=-1,
    )


# ==============================================================================
# Training / validation loops
# ==============================================================================

def train_one_epoch(
    model: YOLOv8,
    criterion: v8DetectionLossWithTeacher,
    loader,
    optimizer: torch.optim.Optimizer,
    hook_capture: BackboneHookCapture,
    device: torch.device,
    scaler,
    amp: bool,
    epoch: int,
    log_every: int = 50,
) -> dict[str, float]:
    model.train()
    total_loss = box_l = cls_l = dfl_l = 0.0
    steps = 0

    for batch in loader:
        # ultralytics batches ship img as uint8 [0, 255]
        imgs = batch["img"].to(device, non_blocking=True).float().div_(255.0)
        for key in ("batch_idx", "cls", "bboxes"):
            batch[key] = batch[key].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        hook_capture.clear()

        with torch.autocast(device_type=device.type,
                            enabled=(amp and device.type == "cuda")):
            preds = model(imgs)
            loss, loss_items = criterion(preds, batch, hook_capture.feats)

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

        steps += 1
        total_loss += float(loss.detach()) / imgs.shape[0]   # undo batch scaling
        box_l += float(loss_items[0])
        cls_l += float(loss_items[1])
        dfl_l += float(loss_items[2])

        if log_every and steps % log_every == 0:
            n = steps
            print(f"  [epoch {epoch}] step {steps:4d}"
                  f"  loss {total_loss/n:.4f}"
                  f"  box {box_l/n:.4f}  cls {cls_l/n:.4f}  dfl {dfl_l/n:.4f}")

    n = max(steps, 1)
    return {"loss": total_loss/n, "box": box_l/n, "cls": cls_l/n, "dfl": dfl_l/n}


@torch.no_grad()
def validate(
    model: YOLOv8,
    criterion: v8DetectionLossWithTeacher,
    loader,
    hook_capture: BackboneHookCapture,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = box_l = cls_l = dfl_l = 0.0
    steps = 0

    for batch in loader:
        imgs = batch["img"].to(device, non_blocking=True).float().div_(255.0)
        for key in ("batch_idx", "cls", "bboxes"):
            batch[key] = batch[key].to(device, non_blocking=True)

        hook_capture.clear()
        preds = model(imgs)
        loss, loss_items = criterion(preds, batch, hook_capture.feats)

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
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    gamma: float = 1.0,
    temperature: float = 1.0,
    tal_topk: int = 10,
    workers: int = 4,
    device_str: str = "auto",
    amp: bool = True,
    out_dir: str = "teacher_yolo_experiment/runs",
    exp_name: str = "teacher_tal",
    log_every: int = 50,
    patience: int = 20,
) -> float:
    # ── device ────────────────────────────────────────────────────────────────
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto" else torch.device(device_str)
    )
    use_amp = amp and device.type == "cuda"
    print(f"[train] device={device}  amp={use_amp}")

    # ── teacher (frozen) ──────────────────────────────────────────────────────
    teacher, enabled_levels = load_teacher(teacher_ckpt, device)

    # ── student model ─────────────────────────────────────────────────────────
    model = load_student(student_weights, cfg=cfg, scale=scale,
                         nc=nc, device=device, verbose=True)

    # ── backbone hooks ────────────────────────────────────────────────────────
    # Hooks fire on every model(imgs) call; feats are read by the loss.
    hook_capture = BackboneHookCapture(model, enabled_levels)

    # ── teacher-augmented loss ────────────────────────────────────────────────
    criterion = v8DetectionLossWithTeacher(
        model, teacher, gamma=gamma, temperature=temperature, tal_topk=tal_topk
    )

    # ── optimizer + LR scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.937,
        weight_decay=weight_decay, nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── data loaders ──────────────────────────────────────────────────────────
    train_loader = build_ultralytics_loader(
        data_yaml, "train", batch, imgsz=imgsz, workers=workers)
    val_loader = build_ultralytics_loader(
        data_yaml, "val", batch, imgsz=imgsz, workers=workers)

    # ── output dir ────────────────────────────────────────────────────────────
    run_dir = os.path.join(out_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")
    print(f"[train] outputs -> {run_dir}")

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(
            model, criterion, train_loader, optimizer,
            hook_capture, device, scaler, amp, epoch, log_every,
        )
        va = validate(model, criterion, val_loader, hook_capture, device)
        scheduler.step()

        is_best = va["loss"] < best_val_loss
        tag = "  *best*" if is_best else ""
        print(
            f"\nepoch {epoch:03d}/{epochs}"
            f"  train loss {tr['loss']:.4f}"
            f"  (box {tr['box']:.4f}  cls {tr['cls']:.4f}  dfl {tr['dfl']:.4f})"
            f"  |  val loss {va['loss']:.4f}"
            f"  (box {va['box']:.4f}  cls {va['cls']:.4f}  dfl {va['dfl']:.4f})"
            f"{tag}"
        )

        save_checkpoint(last_path, model, optimizer, epoch, {"train": tr, "val": va})
        if is_best:
            best_val_loss = va["loss"]
            no_improve = 0
            save_checkpoint(best_path, model, optimizer, epoch, {"train": tr, "val": va})
        else:
            no_improve += 1

        if patience and no_improve >= patience:
            print(f"\n[early stop] no val improvement for {patience} epochs"
                  f" — stopping at epoch {epoch}.")
            break

    hook_capture.remove()
    print(f"\n[done] best val loss {best_val_loss:.4f} -> {best_path}")
    return best_val_loss


# ==============================================================================
# CLI
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train YOLOv8 student with teacher-augmented TAL assignment."
    )
    ap.add_argument("--yolo-weights", default=None,
                    help="Student init checkpoint (.pt). Omit for random init.")
    ap.add_argument("--teacher-ckpt", required=True,
                    help="Teacher checkpoint (teacher/runs/…/best.pt).")
    ap.add_argument("--data",  required=True,
                    help="Ultralytics dataset yaml (e.g. experiments/data/sds.yaml).")
    ap.add_argument("--cfg",   default="cfg/yolov8.yaml",
                    help="YOLOv8 architecture yaml (relative to yolov8/ package).")
    ap.add_argument("--scale", default="m", choices=list("nsmix") + ["lx"],
                    help="Model scale letter.")
    ap.add_argument("--nc",    type=int, default=None,
                    help="Num classes (leave None to read from cfg yaml).")
    ap.add_argument("--epochs",        type=int,   default=100)
    ap.add_argument("--batch",         type=int,   default=16)
    ap.add_argument("--imgsz",         type=int,   default=640)
    ap.add_argument("--lr",            type=float, default=0.01)
    ap.add_argument("--weight-decay",  type=float, default=5e-4)
    ap.add_argument("--gamma",         type=float, default=1.0,
                    help="Teacher exponent γ in align_metric.")
    ap.add_argument("--temperature",   type=float, default=1.0,
                    help="Softmax temperature applied to teacher logits.")
    ap.add_argument("--tal-topk",      type=int,   default=10)
    ap.add_argument("--workers",       type=int,   default=4)
    ap.add_argument("--device",        default="auto")
    ap.add_argument("--amp",           action="store_true",  default=True)
    ap.add_argument("--no-amp",        dest="amp", action="store_false")
    ap.add_argument("--out-dir",       default="teacher_yolo_experiment/runs")
    ap.add_argument("--name",          default="teacher_tal")
    ap.add_argument("--patience",      type=int,   default=20)
    ap.add_argument("--log-every",     type=int,   default=50)
    args = ap.parse_args()

    train(
        student_weights=args.yolo_weights,
        teacher_ckpt=args.teacher_ckpt,
        data_yaml=args.data,
        cfg=args.cfg,
        scale=args.scale,
        nc=args.nc,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        temperature=args.temperature,
        tal_topk=args.tal_topk,
        workers=args.workers,
        device_str=args.device,
        amp=args.amp,
        out_dir=args.out_dir,
        exp_name=args.name,
        log_every=args.log_every,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
