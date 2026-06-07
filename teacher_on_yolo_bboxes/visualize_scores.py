"""
visualize_scores.py
====================
Interactive visualization of pd_scores vs teacher_scores from compare_pipeline.

When the PipelineResult carries a gt_mask (i.e., run_image was called with gt_info),
all five panels are restricted to GT-overlapping anchors — those whose predicted
bbox centers fall inside at least one GT bounding box, matching the TAL training
criterion.  GT bounding boxes are drawn as yellow outlines in the spatial-delta panel.

Usage (standalone — pass a pre-built PipelineResult):
    python visualize_scores.py           # launches matplotlib GUI

Or call directly from a notebook / script:
    from visualize_scores import visualize_result
    visualize_result(result, class_names=CLASS_NAMES, top_k=20, class_idx=0)

Plots produced
--------------
1. Scatter  — pd_score[gt_class] vs teacher_score[gt_class] per GT-overlapping anchor
2. Histogram — score distributions for GT-overlapping anchors
3. Top-K panel — bar chart for the top-K GT-overlapping anchors ranked by pd×iou×teacher
4. Spatial heatmap — teacher_score - pd_score delta, GT boxes overlaid in yellow
5. Monotonicity rank plot — Spearman correlation bar per class (GT-overlapping subset)
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import torch
from scipy.stats import spearmanr

from compare_pipeline import PipelineResult

# ── SDS class names ───────────────────────────────────────────────────────────
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.float().cpu().numpy()


def _iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """IoU between every pair in A×B.  Returns (A, B) array."""
    x1 = np.maximum(bboxes_a[:, None, 0], bboxes_b[None, :, 0])
    y1 = np.maximum(bboxes_a[:, None, 1], bboxes_b[None, :, 1])
    x2 = np.minimum(bboxes_a[:, None, 2], bboxes_b[None, :, 2])
    y2 = np.minimum(bboxes_a[:, None, 3], bboxes_b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    a_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])
    union = a_a[:, None] + a_b[None] - inter
    return inter / np.maximum(union, 1e-9)


def _active_mask(result: PipelineResult) -> np.ndarray:
    """Return the primary boolean filter for 'relevant' anchors.

    If the result has a gt_mask, return it (GT-overlapping anchors).
    Otherwise fall back to teacher_valid (backward compat).
    """
    if result.gt_mask is not None:
        return _to_np(result.gt_mask).astype(bool)
    return _to_np(result.teacher_valid).astype(bool)


def _max_iou_with_gt(bboxes: np.ndarray, gt_bboxes: np.ndarray) -> np.ndarray:
    """Max IoU of each predicted bbox with any GT bbox.  Returns (A,) float."""
    if gt_bboxes.shape[0] == 0:
        return np.ones(len(bboxes), dtype=float)
    return _iou_matrix(bboxes, gt_bboxes).max(axis=1)


def _per_anchor_gt_class_scores(
    result: PipelineResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """For each GT-overlapping anchor, return its pd/teacher score at the actual GT class.

    Assigns each anchor to the GT box with highest IoU — more robust than center
    containment, which silently falls back to class 0 for edge-touching anchors.

    Returns (pd_gt, tch_gt, assigned_classes, anchor_idx) or None when GT
    info is unavailable or there are no GT boxes.
    """
    if (result.gt_mask is None or result.gt_bboxes is None or
            result.gt_labels is None or result.gt_bboxes.shape[0] == 0):
        return None

    gt_mask   = _to_np(result.gt_mask).astype(bool)
    bboxes    = _to_np(result.pd_bboxes)                    # (A, 4)
    gt_bboxes = _to_np(result.gt_bboxes)                    # (N, 4)
    gt_labels = _to_np(result.gt_labels).astype(int)        # (N,)
    pd_all    = _to_np(result.pd_scores)                    # (A, C)
    tch_all   = _to_np(result.teacher_scores)               # (A, C)

    anchor_idx = np.where(gt_mask)[0]                       # (A_gt,)
    if len(anchor_idx) == 0:
        return (np.array([]), np.array([]),
                np.array([], dtype=int), anchor_idx)

    iou            = _iou_matrix(bboxes[anchor_idx], gt_bboxes)  # (A_gt, N)
    best_gt        = iou.argmax(axis=1)                     # (A_gt,)
    assigned_class = gt_labels[best_gt]                     # (A_gt,)

    pd_gt  = pd_all[anchor_idx, assigned_class]
    tch_gt = tch_all[anchor_idx, assigned_class]
    return pd_gt, tch_gt, assigned_class, anchor_idx


# ─────────────────────────────────────────────────────────────────────────────
# Per-class scatter (Plot 1)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(result: PipelineResult, class_idx: int,
                 class_names: list[str], ax: plt.Axes) -> None:
    # ── GT-class mode: score at actual GT class per anchor ────────────────────
    gt_scores = _per_anchor_gt_class_scores(result)
    if gt_scores is not None:
        pd_gt, tch_gt, assigned_classes, anchor_idx = gt_scores

        if len(pd_gt) == 0:
            ax.text(0.5, 0.5, "no GT-overlapping anchors",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#64748b")
            ax.set_title("Scatter — score at GT-assigned class", fontsize=9)
            return

        valid_a    = _to_np(result.teacher_valid).astype(bool)[anchor_idx]
        rejected_a = _to_np(result.teacher_rejected).astype(bool)[anchor_idx]

        unique_cls = np.unique(assigned_classes)
        n_cls      = len(unique_cls)
        cmap       = plt.cm.get_cmap("tab10" if n_cls <= 10 else "tab20")

        for ci, c in enumerate(unique_cls):
            c_mask = assigned_classes == c
            cname  = class_names[c] if c < len(class_names) else str(c)
            col    = cmap(ci % cmap.N)

            # All points for this GT class (circle = valid/unscored)
            ax.scatter(pd_gt[c_mask], tch_gt[c_mask], s=12, alpha=0.7,
                       color=col, label=f"GT:{cname} ({c_mask.sum()})", zorder=2)
            # Overlay rejected anchors with X marker (no extra legend entry)
            r_mask = c_mask & rejected_a
            if r_mask.any():
                ax.scatter(pd_gt[r_mask], tch_gt[r_mask], s=20, alpha=1.0,
                           color=col, marker="x", linewidths=1.5, zorder=4)

        lim = max(pd_gt.max(), tch_gt.max(), 0.05) + 0.02
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4, label="y=x")

        r, p_val = (spearmanr(pd_gt, tch_gt) if len(pd_gt) > 2 else (0.0, 1.0))
        ax.set_title(
            f"Scatter — score at GT-assigned class per anchor\n"
            f"Spearman ρ={r:.3f} (p={p_val:.2e})  n={len(pd_gt)}"
            f"  (x marker = teacher-rejected)",
            fontsize=9,
        )
        ax.set_xlabel("pd_score at GT class (student)", fontsize=8)
        ax.set_ylabel("teacher_score at GT class", fontsize=8)
        ax.legend(fontsize=7, markerscale=1.5)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        return

    # ── Fallback: fixed class_idx (no GT label info available) ───────────────
    pd       = _to_np(result.pd_scores[:, class_idx])
    tch      = _to_np(result.teacher_scores[:, class_idx])
    valid    = _to_np(result.teacher_valid).astype(bool)
    rejected = _to_np(result.teacher_rejected).astype(bool)

    if result.gt_mask is not None:
        gt = _to_np(result.gt_mask).astype(bool)

        in_gt_valid    = gt & valid
        in_gt_rejected = gt & rejected
        in_gt_unscored = gt & ~valid & ~rejected
        not_gt         = ~gt

        ax.scatter(pd[not_gt], tch[not_gt], s=2, alpha=0.08, c="#334155",
                   label=f"outside GT ({not_gt.sum()})", zorder=1)
        if in_gt_unscored.any():
            ax.scatter(pd[in_gt_unscored], tch[in_gt_unscored], s=10, alpha=0.6,
                       c="#a855f7", label=f"GT unscored ({in_gt_unscored.sum()})", zorder=2)
        ax.scatter(pd[in_gt_rejected], tch[in_gt_rejected], s=10, alpha=0.8,
                   c="#f97316", label=f"GT rejected→fallback ({in_gt_rejected.sum()})", zorder=3)
        ax.scatter(pd[in_gt_valid], tch[in_gt_valid], s=10, alpha=0.8,
                   c="#3b82f6", label=f"GT teacher-scored ({in_gt_valid.sum()})", zorder=4)

        spearman_mask = in_gt_valid
        subtitle = f"GT-overlap={gt.sum()}"
    else:
        neither = ~valid & ~rejected
        ax.scatter(pd[valid],    tch[valid],    s=6, alpha=0.5, c="#3b82f6",
                   label="teacher scored", zorder=3)
        ax.scatter(pd[rejected], tch[rejected], s=6, alpha=0.5, c="#f97316",
                   label="rejected→fallback", zorder=3)
        ax.scatter(pd[neither],  tch[neither],  s=3, alpha=0.2, c="#94a3b8",
                   label="outside GT", zorder=2)
        spearman_mask = valid
        subtitle = ""

    lim = max(pd.max(), tch.max()) + 0.02
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4, label="y=x")

    r, p_val = (spearmanr(pd[spearman_mask], tch[spearman_mask])
                if spearman_mask.sum() > 2 else (0.0, 1.0))
    title = f"Scatter — class {class_idx}: {class_names[class_idx]}"
    if subtitle:
        title += f"  [{subtitle}]"
    title += f"\nSpearman ρ={r:.3f} (p={p_val:.2e})  n_scored={spearman_mask.sum()}"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("pd_score (student sigmoid)", fontsize=8)
    ax.set_ylabel("teacher_score (softmax)", fontsize=8)
    ax.legend(fontsize=7, markerscale=2)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)


# ─────────────────────────────────────────────────────────────────────────────
# Score distributions (Plot 2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_distributions(result: PipelineResult, class_idx: int,
                        class_names: list[str], ax: plt.Axes) -> None:
    # ── GT-class mode ─────────────────────────────────────────────────────────
    gt_scores = _per_anchor_gt_class_scores(result)
    if gt_scores is not None:
        pd_gt, tch_gt, assigned_classes, anchor_idx = gt_scores

        valid_a  = _to_np(result.teacher_valid).astype(bool)[anchor_idx]
        tch_plot = tch_gt[valid_a]

        max_val = max(
            pd_gt.max()   if len(pd_gt)   else 0.0,
            tch_plot.max() if len(tch_plot) else 0.0,
        )
        bins = np.linspace(0, max_val + 0.01, 50)

        ax.hist(pd_gt,   bins=bins, alpha=0.55, color="#3b82f6",
                label=f"pd @ GT class ({len(pd_gt)})", density=True)
        ax.hist(tch_plot, bins=bins, alpha=0.55, color="#22c55e",
                label=f"teacher @ GT class, valid ({len(tch_plot)})", density=True)

        if len(pd_gt):
            ax.axvline(pd_gt.mean(), color="#1d4ed8", lw=1.2, ls="--",
                       label=f"pd mean={pd_gt.mean():.3f}")
        if len(tch_plot):
            ax.axvline(tch_plot.mean(), color="#15803d", lw=1.2, ls="--",
                       label=f"teacher mean={tch_plot.mean():.3f}")
        else:
            ax.text(0.5, 0.5, "no valid GT anchors", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#64748b")

        ax.set_title("Score distributions @ GT-assigned class\n"
                     "(GT-overlapping anchors only)", fontsize=9)
        ax.set_xlabel("Score at GT class", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7)
        return

    # ── Fallback: fixed class_idx ─────────────────────────────────────────────
    pd    = _to_np(result.pd_scores[:, class_idx])
    tch   = _to_np(result.teacher_scores[:, class_idx])
    valid = _to_np(result.teacher_valid).astype(bool)

    if result.gt_mask is not None:
        gt       = _to_np(result.gt_mask).astype(bool)
        pd_plot  = pd[gt]
        tch_plot = tch[gt & valid]
        title_suffix = " (GT-overlapping anchors)"
    else:
        pd_plot  = pd
        tch_plot = tch[valid]
        title_suffix = ""

    max_val = max(
        pd_plot.max()  if len(pd_plot)  else 0.0,
        tch_plot.max() if len(tch_plot) else 0.0,
    )
    bins = np.linspace(0, max_val + 0.01, 50)

    ax.hist(pd_plot,  bins=bins, alpha=0.55, color="#3b82f6",
            label=f"pd_score ({len(pd_plot)})", density=True)
    ax.hist(tch_plot, bins=bins, alpha=0.55, color="#22c55e",
            label=f"teacher_score valid ({len(tch_plot)})", density=True)

    if len(pd_plot):
        ax.axvline(pd_plot.mean(), color="#1d4ed8", lw=1.2, ls="--",
                   label=f"pd mean={pd_plot.mean():.3f}")
    if len(tch_plot):
        ax.axvline(tch_plot.mean(), color="#15803d", lw=1.2, ls="--",
                   label=f"teacher mean={tch_plot.mean():.3f}")
    else:
        ax.text(0.5, 0.5, "no valid anchors", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#64748b")

    ax.set_title(f"Score distributions — class {class_idx}: {class_names[class_idx]}"
                 f"{title_suffix}", fontsize=9)
    ax.set_xlabel("Score", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Top-K anchors (Plot 3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_topk(result: PipelineResult, class_idx: int,
              class_names: list[str], ax: plt.Axes, top_k: int = 20,
              alpha: float = 1.0, beta: float = 6.0, gamma: float = 1.0) -> None:
    pd       = _to_np(result.pd_scores[:, class_idx])
    tch      = _to_np(result.teacher_scores[:, class_idx])
    rejected = _to_np(result.teacher_rejected).astype(bool)

    if result.gt_mask is not None:
        gt         = _to_np(result.gt_mask).astype(bool)
        pool_idx   = np.where(gt)[0]
        title_suffix = " [GT-overlapping pool]"
    else:
        pool_idx   = np.arange(len(pd))
        title_suffix = ""

    if len(pool_idx) == 0:
        ax.text(0.5, 0.5, "no GT-overlapping anchors", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#64748b")
        ax.set_title(f"Top-{top_k} anchors — class {class_idx}: {class_names[class_idx]}"
                     f"{title_suffix}", fontsize=9)
        return

    pd_pool  = pd[pool_idx]
    tch_pool = tch[pool_idx]

    # Use real IoU with GT bboxes when available, else iou=1 placeholder
    if result.gt_bboxes is not None and result.gt_bboxes.shape[0] > 0:
        iou = _max_iou_with_gt(_to_np(result.pd_bboxes), _to_np(result.gt_bboxes))[pool_idx]
    else:
        iou = np.ones(len(pool_idx))

    align      = (pd_pool ** alpha) * (iou ** beta) * (tch_pool ** gamma)
    top_k_show = min(top_k, len(pool_idx))
    topk_pool  = np.argsort(align)[::-1][:top_k_show]
    topk_idx   = pool_idx[topk_pool]    # global anchor indices

    x = np.arange(top_k_show)
    w = 0.35
    ax.bar(x - w / 2, pd[topk_idx],  w, color="#3b82f6", alpha=0.8, label="pd_score")
    ax.bar(x + w / 2, tch[topk_idx], w, color="#22c55e", alpha=0.8, label="teacher_score")

    for i, idx in enumerate(topk_idx):
        if rejected[idx]:
            ax.text(x[i], max(pd[idx], tch[idx]) + 0.005, "R", ha="center",
                    va="bottom", fontsize=6, color="#f97316")

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in topk_idx], fontsize=5, rotation=60)
    ax.set_title(f"Top-{top_k_show} anchors by pd×iou×teacher{title_suffix}\n"
                 f"class {class_idx}: {class_names[class_idx]}  (R=rejected by teacher)",
                 fontsize=9)
    ax.set_xlabel("Anchor index", fontsize=8)
    ax.set_ylabel("Score", fontsize=8)
    ax.legend(fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial delta heatmap (Plot 4)
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial_delta(result: PipelineResult, class_idx: int,
                        class_names: list[str], ax: plt.Axes) -> None:
    """Project teacher_score - pd_score onto a 640×640 grid, GT boxes in yellow."""
    bboxes = _to_np(result.pd_bboxes)                   # (A, 4)
    pd     = _to_np(result.pd_scores[:, class_idx])
    tch    = _to_np(result.teacher_scores[:, class_idx])

    if result.gt_mask is not None:
        mask         = _to_np(result.gt_mask).astype(bool)
        title_suffix = " (GT-overlapping anchors)"
    else:
        mask         = _to_np(result.teacher_valid).astype(bool)
        title_suffix = ""

    cx = np.clip((bboxes[:, 0] + bboxes[:, 2]) / 2, 0, 639)
    cy = np.clip((bboxes[:, 1] + bboxes[:, 3]) / 2, 0, 639)
    delta = tch - pd

    grid_size = 40
    grid  = np.zeros((grid_size, grid_size))
    count = np.zeros((grid_size, grid_size))
    gx = np.floor(cx / 640 * grid_size).astype(int).clip(0, grid_size - 1)
    gy = np.floor(cy / 640 * grid_size).astype(int).clip(0, grid_size - 1)
    for i in range(len(delta)):
        if mask[i]:
            grid[gy[i], gx[i]] += delta[i]
            count[gy[i], gx[i]] += 1
    count[count == 0] = 1
    grid /= count

    vabs = max(abs(grid.min()), abs(grid.max()), 1e-6)
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vabs, vmax=vabs, origin="upper",
                   extent=[0, 640, 640, 0])
    plt.colorbar(im, ax=ax, fraction=0.03, label="teacher − student")

    # Overlay GT bounding boxes as yellow outlines
    if result.gt_bboxes is not None and result.gt_bboxes.shape[0] > 0:
        gt_bboxes = _to_np(result.gt_bboxes)
        gt_labels = (_to_np(result.gt_labels).astype(int)
                     if result.gt_labels is not None else None)
        for j, (x1, y1, x2, y2) in enumerate(gt_bboxes):
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=1.5, edgecolor="#facc15",
                                  facecolor="none", zorder=5)
            ax.add_patch(rect)
            if gt_labels is not None and class_names:
                lbl = (class_names[gt_labels[j]]
                       if gt_labels[j] < len(class_names) else str(gt_labels[j]))
                ax.text(x1, max(y1 - 2, 0), lbl, fontsize=5, color="#facc15",
                        zorder=6, ha="left", va="bottom")

    ax.set_title(
        f"Spatial Δ (teacher−student){title_suffix}\n"
        f"class {class_idx}: {class_names[class_idx]}\n"
        f"Blue=teacher>student  Red=student>teacher  Yellow=GT box",
        fontsize=9,
    )
    ax.set_xlabel("x (pixels)", fontsize=8)
    ax.set_ylabel("y (pixels)", fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Spearman rank correlation across all classes (Plot 5)
# ─────────────────────────────────────────────────────────────────────────────

def plot_spearman_all_classes(result: PipelineResult,
                               class_names: list[str], ax: plt.Axes,
                               top_n_classes: int = 30) -> None:
    """Bar chart of Spearman ρ between pd_score and teacher_score for each class."""
    nc      = result.pd_scores.shape[1]
    pd_all  = _to_np(result.pd_scores)
    tch_all = _to_np(result.teacher_scores)

    if result.gt_mask is not None:
        mask       = _to_np(result.gt_mask).astype(bool)
        mask_label = "GT-overlapping"
    else:
        mask       = _to_np(result.teacher_valid).astype(bool)
        mask_label = "teacher-valid"

    rhos, labels = [], []
    for c in range(nc):
        pd_v  = pd_all[mask, c]
        tch_v = tch_all[mask, c]
        r = spearmanr(pd_v, tch_v)[0] if len(pd_v) > 5 else 0.0
        rhos.append(r)
        labels.append(class_names[c] if c < len(class_names) else str(c))

    order    = np.argsort(np.abs(rhos))[::-1][:top_n_classes]
    rhos_s   = np.array(rhos)[order]
    labels_s = [labels[i] for i in order]

    colors = ["#3b82f6" if r >= 0 else "#ef4444" for r in rhos_s]
    ax.barh(range(len(rhos_s)), rhos_s, color=colors, alpha=0.8)
    ax.set_yticks(range(len(rhos_s)))
    ax.set_yticklabels(labels_s, fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman ρ (pd vs teacher)", fontsize=8)
    ax.set_title(
        f"Monotonicity rank — top {top_n_classes} classes\n"
        f"(computed on {mask_label} anchors only)",
        fontsize=9,
    )
    ax.set_xlim(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main visualize function
# ─────────────────────────────────────────────────────────────────────────────

def visualize_result(
    result:       PipelineResult,
    class_idx:    int = 0,
    class_names:  list[str] | None = None,
    top_k:        int = 20,
    save_path:    str | None = None,
) -> plt.Figure:
    """
    Produce the full 5-panel comparison figure.

    All panels focus on GT-overlapping anchors when result.gt_mask is available
    (i.e., run_image was called with gt_info).

    Parameters
    ----------
    result      : output of run_image()
    class_idx   : which class column to focus the per-class plots on
    class_names : list of class name strings (defaults to COCO 80)
    top_k       : how many anchors to show in the bar chart
    save_path   : if set, save figure to this path instead of showing

    Returns the Figure object.
    """
    if class_names is None:
        nc = result.pd_scores.shape[1]
        class_names = COCO_NAMES[:nc] if nc <= 80 else [str(i) for i in range(nc)]

    n_gt = int(result.gt_mask.sum()) if result.gt_mask is not None else None
    n_gt_boxes = (result.gt_bboxes.shape[0]
                  if result.gt_bboxes is not None else None)
    gt_str = (f"  |  GT-boxes={n_gt_boxes}  GT-overlap-anchors={n_gt}"
              if n_gt is not None else "")

    fig = plt.figure(figsize=(20, 14), facecolor="#0f172a")
    fig.suptitle(
        f"pd_scores vs teacher_scores  |  {result.anchor_count} anchors{gt_str}  |  "
        f"image {result.image_size[1]}×{result.image_size[0]}",
        fontsize=13, color="white", y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97, top=0.93, bottom=0.06,
    )

    _style_ax = lambda ax: (
        ax.set_facecolor("#1e293b"),
        ax.tick_params(colors="#94a3b8", labelsize=7),
        ax.xaxis.label.set_color("#94a3b8"),
        ax.yaxis.label.set_color("#94a3b8"),
        ax.title.set_color("#e2e8f0"),
        [s.set_color("#334155") for s in ax.spines.values()],
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1:]),
    ]
    for ax in axes:
        _style_ax(ax)

    plot_scatter(result, class_idx, class_names, axes[0])
    plot_distributions(result, class_idx, class_names, axes[1])
    plot_topk(result, class_idx, class_names, axes[2], top_k=top_k)
    plot_spatial_delta(result, class_idx, class_names, axes[3])
    plot_spearman_all_classes(result, class_names, axes[4])

    valid_count    = result.teacher_valid.sum().item()
    rejected_count = result.teacher_rejected.sum().item()
    gt_footer = (f"  |  GT-overlap: {n_gt}  GT-boxes: {n_gt_boxes}"
                 if n_gt is not None else "")
    fig.text(
        0.01, 0.01,
        f"valid: {valid_count}  |  rejected (fallback): {rejected_count}  |  "
        f"unscored: {result.anchor_count - valid_count - rejected_count}{gt_footer}",
        fontsize=8, color="#64748b",
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[viz] saved → {save_path}")
    else:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point (demo with random data — no real model needed)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo",       action="store_true",
                    help="Run with synthetic random data (no real models needed)")
    ap.add_argument("--yolo-dir",   default=None)
    ap.add_argument("--teacher-dir",default=None)
    ap.add_argument("--yolo-ckpt",  default=None)
    ap.add_argument("--teacher-ckpt", default=None)
    ap.add_argument("--image",      default=None)
    ap.add_argument("--class-idx",  type=int, default=0)
    ap.add_argument("--top-k",      type=int, default=20)
    ap.add_argument("--save",       default=None)
    ap.add_argument("--device",     default="cpu")
    args = ap.parse_args()

    if args.demo:
        print("[demo] generating synthetic PipelineResult …")
        A, C = 8400, 5
        np.random.seed(42)
        pd_s  = torch.sigmoid(torch.randn(A, C) * 2 - 3)
        t_log = torch.randn(A, C) * 2 - 3
        t_log[:, 0] += pd_s[:, 0] * 5 + torch.randn(A) * 1.5
        t_s   = t_log.softmax(dim=-1)

        valid_mask    = torch.zeros(A, dtype=torch.bool)
        valid_mask[:500] = True
        rejected_mask = torch.zeros(A, dtype=torch.bool)
        rejected_mask[500:550] = True
        t_s[rejected_mask] = pd_s[rejected_mask]

        gt_mask = torch.zeros(A, dtype=torch.bool)
        gt_mask[:550] = True                           # first 550 are "in GT"

        cx = torch.randint(5, 635, (A,)).float()
        cy = torch.randint(5, 635, (A,)).float()
        w  = torch.randint(5, 200, (A,)).float()
        h  = torch.randint(5, 200, (A,)).float()
        bboxes = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=1).clamp(0, 640)

        gt_boxes = torch.tensor([[50, 50, 300, 300], [350, 200, 600, 580]], dtype=torch.float)
        gt_lbls  = torch.tensor([0, 2], dtype=torch.long)

        result = PipelineResult(
            pd_scores=pd_s,
            teacher_scores=t_s,
            teacher_valid=valid_mask,
            teacher_rejected=rejected_mask,
            pd_bboxes=bboxes,
            image_size=(640, 640),
            anchor_count=A,
            gt_bboxes=gt_boxes,
            gt_labels=gt_lbls,
            gt_mask=gt_mask,
        )
        CLASS_NAMES_DEMO = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]
    else:
        if not all([args.yolo_dir, args.teacher_dir, args.yolo_ckpt, args.teacher_ckpt, args.image]):
            ap.error("Provide --yolo-dir, --teacher-dir, --yolo-ckpt, --teacher-ckpt, --image  OR  --demo")

        from compare_pipeline import build_pipeline, run_image
        pipeline = build_pipeline(
            yolo_pkg_dir=args.yolo_dir,
            teacher_dir=args.teacher_dir,
            yolo_ckpt=args.yolo_ckpt,
            teacher_ckpt=args.teacher_ckpt,
            device=args.device,
        )
        result = run_image(pipeline, args.image)
        print(f"[done] pd_scores: {result.pd_scores.shape}  teacher_scores: {result.teacher_scores.shape}")
        print(f"       valid: {result.teacher_valid.sum()}  rejected: {result.teacher_rejected.sum()}")
        CLASS_NAMES_DEMO = COCO_NAMES

    visualize_result(
        result,
        class_idx=args.class_idx,
        class_names=CLASS_NAMES_DEMO,
        top_k=args.top_k,
        save_path=args.save,
    )
