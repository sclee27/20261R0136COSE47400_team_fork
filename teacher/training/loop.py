"""Block: training / validation loop for the teacher bbox evaluator.

Wires the frozen-backbone model (model.py) and the dataset (dataset.py) into a
standard train/val loop:

    backbone (frozen, no_grad) -> evaluator (trainable)
      -> mask out aspect-rejected boxes ("valid") and ignore labels (-1)
      -> CrossEntropy over (num_fg + 1) classes (background = num_fg index)
      -> optimizer steps only the evaluator heads
      -> per-level accuracy / positive-recall metrics
      -> save best checkpoint by val accuracy

Only metrics + checkpoints are produced; the backbone is the frozen input
artifact and is NOT saved (only evaluator weights are).
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_TEACHER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TEACHER_DIR not in sys.path:
    sys.path.insert(0, _TEACHER_DIR)

from sampling.levels import LEVEL_ORDER, assign_levels  # noqa: E402


def resolve_device(device: str) -> torch.device:
    """Map the config device string to a torch.device ('auto' -> cuda if present)."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_optimizer(params, train_cfg):
    """AdamW (default) or SGD over the given params."""
    if train_cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=train_cfg.lr,
                                 weight_decay=train_cfg.weight_decay)
    if train_cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=train_cfg.lr,
                               momentum=train_cfg.momentum,
                               weight_decay=train_cfg.weight_decay)
    raise ValueError(f"unknown optimizer: {train_cfg.optimizer}")


# ---------------------------------------------------------------------------
# Per-level metric accumulator
# ---------------------------------------------------------------------------
class LevelMeter:
    """Accumulates per-level accuracy and positive-recall.

    For each usable box (valid & label != -1):
        - acc:         predicted class == label
        - pos-recall:  among foreground boxes (label < num_fg), fraction whose
                       predicted class equals the true foreground class.
    Levels are computed from box geometry with the SAME thresholds the
    evaluator routes on (upper_short), so the table matches the heads that
    were actually trained.
    """

    def __init__(self, num_fg: int, upper_short: dict):
        self.num_fg = num_fg
        self.upper_short = upper_short
        self.n = {k: 0 for k in LEVEL_ORDER}          # usable boxes per level
        self.correct = {k: 0 for k in LEVEL_ORDER}    # correct predictions
        self.pos_n = {k: 0 for k in LEVEL_ORDER}      # foreground boxes
        self.pos_correct = {k: 0 for k in LEVEL_ORDER}  # foreground predicted right
        self.loss_sum = 0.0
        self.loss_steps = 0

    def add_loss(self, value: float):
        self.loss_sum += value
        self.loss_steps += 1

    def update(self, boxes_np: np.ndarray, labels_np: np.ndarray, preds_np: np.ndarray):
        """boxes/labels/preds are the already-masked (usable only) flat arrays."""
        if len(boxes_np) == 0:
            return
        level_ids = assign_levels(boxes_np, self.upper_short)
        for li, name in enumerate(LEVEL_ORDER):
            m = level_ids == li
            if not m.any():
                continue
            lab = labels_np[m]
            pred = preds_np[m]
            self.n[name] += int(m.sum())
            self.correct[name] += int((pred == lab).sum())
            fg = lab < self.num_fg
            self.pos_n[name] += int(fg.sum())
            self.pos_correct[name] += int(((pred == lab) & fg).sum())

    @property
    def total_n(self):
        return sum(self.n.values())

    @property
    def total_correct(self):
        return sum(self.correct.values())

    @property
    def acc(self):
        return self.total_correct / self.total_n if self.total_n else 0.0

    @property
    def avg_loss(self):
        return self.loss_sum / self.loss_steps if self.loss_steps else 0.0


def _bar(frac: float, width: int = 20) -> str:
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def print_level_table(meter: LevelMeter, enabled: list, indent: str = "    "):
    print(f"{indent}{'level':<6}{'acc':>8}{'pos-rec':>9}{'n':>9}   enabled")
    for name in LEVEL_ORDER:
        n = meter.n[name]
        acc = meter.correct[name] / n if n else 0.0
        pr = meter.pos_correct[name] / meter.pos_n[name] if meter.pos_n[name] else 0.0
        flag = "ON" if name in enabled else "drop"
        print(f"{indent}{name:<6}{acc:>8.3f}{pr:>9.3f}{n:>9}   {flag}")


# ---------------------------------------------------------------------------
# One epoch (train or eval)
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, criterion, num_fg, upper_short, enabled_levels,
              optimizer=None, scaler=None, amp=False, log_every=50, epoch=0,
              train=True, step_writer=None):
    """Run a single epoch; returns a LevelMeter with the accumulated metrics.

    If `step_writer` (a csv.writer) is given and we are training, one row per
    optimizer step is logged: [epoch, step, loss, running_acc].
    """
    model.evaluator.train(mode=train)
    model.backbone.eval()    # backbone stays frozen / eval always
    meter = LevelMeter(num_fg, upper_short)
    use_amp = amp and device.type == "cuda"

    step = 0
    for batch in loader:
        if batch is None:               # collate dropped an all-empty batch
            continue
        step += 1
        images = batch["images"].to(device, non_blocking=True)
        boxes = batch["boxes"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(images, boxes)               # scores/valid/rejected
                scores = out["scores"]                   # (B, A, num_fg+1)
                valid = out["valid"]                     # (B, A) bool

                B, A, C = scores.shape
                flat_scores = scores.reshape(B * A, C)
                flat_labels = labels.reshape(B * A)
                flat_valid = valid.reshape(B * A)
                # usable = scored (valid) AND not an ignore/pad label (-1)
                usable = flat_valid & (flat_labels != -1)

                if usable.any():
                    loss = criterion(flat_scores[usable], flat_labels[usable])
                else:
                    loss = None

        if train and loss is not None:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # -- metrics (no grad) -------------------------------------------
        if loss is not None:
            step_loss = float(loss.detach())
            meter.add_loss(step_loss)
            with torch.no_grad():
                u = usable
                preds = flat_scores[u].argmax(dim=-1)
                boxes_u = boxes.reshape(B * A, 4)[u].detach().cpu().numpy()
                labels_u = flat_labels[u].detach().cpu().numpy()
                preds_u = preds.detach().cpu().numpy()
            meter.update(boxes_u, labels_u, preds_u)
            if train and step_writer is not None:
                step_writer.writerow([epoch, step, round(step_loss, 6), round(meter.acc, 6)])

        if train and log_every and step % log_every == 0:
            print(f"    [epoch {epoch}] step {step}  loss {meter.avg_loss:.4f}  "
                  f"acc {meter.acc:.3f}")

    return meter


# ---------------------------------------------------------------------------
# CSV metric logging (per-epoch table -> graphs later)
# ---------------------------------------------------------------------------
def epoch_csv_fieldnames() -> list:
    """Columns for metrics.csv: aggregates + per-level val metrics."""
    cols = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    for name in LEVEL_ORDER:
        cols += [f"val_{name}_acc", f"val_{name}_posrec", f"val_{name}_n"]
    return cols


def epoch_csv_row(epoch: int, tr: "LevelMeter", va: "LevelMeter") -> dict:
    """One metrics.csv row from the train/val meters of an epoch."""
    row = {
        "epoch": epoch,
        "train_loss": round(tr.avg_loss, 6), "train_acc": round(tr.acc, 6),
        "val_loss": round(va.avg_loss, 6), "val_acc": round(va.acc, 6),
    }
    for name in LEVEL_ORDER:
        n = va.n[name]
        row[f"val_{name}_acc"] = round(va.correct[name] / n, 6) if n else 0.0
        row[f"val_{name}_posrec"] = (round(va.pos_correct[name] / va.pos_n[name], 6)
                                     if va.pos_n[name] else 0.0)
        row[f"val_{name}_n"] = n
    return row


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(path, model, cfg_dict, epoch, metrics):
    """Persist the trained EVALUATOR heads only (the backbone is the frozen input
    artifact and is not saved), plus the metadata needed to reload + score boxes.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "evaluator": model.evaluator.state_dict(),   # only the trained heads
        "num_classes": model.num_fg_classes + 1,
        "num_fg_classes": model.num_fg_classes,
        "bg_index": model.bg_index,
        "enabled_levels": model.enabled_levels,
        "upper_short": model.upper_short,
        "epoch": epoch,
        "metrics": metrics,
        "config": cfg_dict,
    }, path)


# ---------------------------------------------------------------------------
# Full training driver
# ---------------------------------------------------------------------------
def train(model, train_loader, val_loader, cfg, cfg_dict):
    """Run the full training schedule; save best/last checkpoints. Returns best metrics."""
    device = resolve_device(cfg.train.device)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = build_optimizer(model.trainable_parameters(), cfg.train)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    out_dir = os.path.join(cfg.train.out_dir, cfg.train.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    num_fg = model.num_fg_classes
    enabled = model.enabled_levels
    upper = model.upper_short

    # -- CSV logs (for plotting later); live under the run dir so renaming the
    #    run (train.exp_name) renames/relocates them too. -------------------
    metrics_path = os.path.join(out_dir, "metrics.csv")   # one row per epoch
    steps_path = os.path.join(out_dir, "steps.csv")       # one row per train step
    mfile = open(metrics_path, "w", newline="")
    sfile = open(steps_path, "w", newline="")
    mwriter = csv.DictWriter(mfile, fieldnames=epoch_csv_fieldnames())
    mwriter.writeheader()
    swriter = csv.writer(sfile)
    swriter.writerow(["epoch", "step", "loss", "running_acc"])
    print(f"[csv] per-epoch -> {metrics_path}\n[csv] per-step  -> {steps_path}")

    best_acc = -1.0
    best_metrics = None
    no_improve = 0          # epochs since the last val-acc improvement
    try:
        for epoch in range(1, cfg.train.epochs + 1):
            tr = run_epoch(model, train_loader, device, criterion, num_fg, upper, enabled,
                           optimizer=optimizer, scaler=scaler, amp=cfg.train.amp,
                           log_every=cfg.train.log_every, epoch=epoch, train=True,
                           step_writer=swriter)

            va = run_epoch(model, val_loader, device, criterion, num_fg, upper, enabled,
                           optimizer=None, scaler=None, amp=False, epoch=epoch, train=False)

            is_best = va.acc > best_acc
            tag = "  *best*" if is_best else ""
            print(f"\nepoch {epoch:02d}/{cfg.train.epochs}  "
                  f"train_loss {tr.avg_loss:.4f} acc {tr.acc:.3f} | "
                  f"val_loss {va.avg_loss:.4f} acc {va.acc:.3f}{tag}")
            print_level_table(va, enabled)

            # append + flush so the CSV is usable even mid-run / on crash
            mwriter.writerow(epoch_csv_row(epoch, tr, va))
            mfile.flush()
            sfile.flush()

            metrics = {"train_loss": tr.avg_loss, "train_acc": tr.acc,
                       "val_loss": va.avg_loss, "val_acc": va.acc}
            save_checkpoint(os.path.join(out_dir, "last.pt"), model, cfg_dict, epoch, metrics)
            if is_best:
                best_acc = va.acc
                best_metrics = metrics
                no_improve = 0
                save_checkpoint(os.path.join(out_dir, "best.pt"), model, cfg_dict, epoch, metrics)
            else:
                no_improve += 1

            # Early stopping: stop once val-acc has not improved for `patience`
            # epochs (patience == 0 disables it). best.pt already holds the peak.
            if cfg.train.patience and no_improve >= cfg.train.patience:
                print(f"\n[early stop] val-acc no improvement for {cfg.train.patience} epochs "
                      f"-> stopping at epoch {epoch} (best {best_acc:.3f} @ epoch {epoch - no_improve}).")
                break
    finally:
        mfile.close()
        sfile.close()

    print(f"\n[done] best val acc {best_acc:.3f} -> {os.path.join(out_dir, 'best.pt')}")
    return best_metrics
