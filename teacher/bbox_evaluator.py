"""
YOLOv8m BBox Evaluator Networks
================================
Evaluates YOLO predicted bboxes using backbone feature maps.

Level assignment (S = shorter side in original image scale):
    Original image : 10 < S <= 20   (raw RGB, stride=1, 3ch)
    P1             : 20 < S <= 40   (YOLO P1 stem, stride=2, 48ch)
    P2             : 40 < S <= 80   (stride=4,  96ch)
    P3             : 80 < S <= 160  (stride=8,  192ch)
    P4             : 160 < S <= 320 (stride=16, 384ch)

Feature map patch sizes (after stride normalization):
    orig : 8-20px short side (raw pixels, no stride reduction)
    P1   : 10-20px short side in feature space  (stride=2)
    P2   : 10-20px short side in feature space (stride=4)
    P3   : 10-20px short side in feature space (stride=8)
    P4   : 10-20px short side in feature space (stride=16)

Constraints:
    - Aspect ratio > 1:4 -> rejected (flagged, no score assigned)
    - roi_align output size: ROI_K x ROI_K (7x7) for all levels

Design decisions:
    - Conv-first: full-map convs applied BEFORE ROI crop (preserves spatial context)
    - Custom stride=1 conv for orig level (864 params, trained from scratch)
    - P1 uses YOLO actual P1 stem output (48ch, stride=2)
    - Vectorized aspect ratio filter and level assignment (no Python loops over bboxes)
    - Batched ROI align per level (one forward pass per level, not per bbox)
    - roi_align spatial pooling replaces AdaptiveAvgPool -- no redundant pooling
    - Dropout(0.3) before final FC -- correlated patches risk overfitting
"""

import torch
import torch.nn as nn
from torchvision.ops import roi_align


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ASPECT_RATIO = 4.0
ROI_K            = 7
NUM_CLASSES      = 80

LEVEL_NAMES = ['orig', 'p1', 'p2', 'p3', 'p4']

LEVEL_STRIDES = {
    'orig': 1,
    'p1':   2,
    'p2':   4,
    'p3':   8,
    'p4':   16,
}

# shorter side upper bounds per level
LEVEL_UPPER_S = {
    'orig': 16,
    'p1':   40,
    'p2':   80,
    'p3':   160,
    'p4':   320,
}

# YOLOv8m backbone output channels per level
LEVEL_IN_CHANNELS = {
    'orig': 3,     # raw RGB -> OriginalImageConv -> 32ch
    'p1':   48,    # YOLO P1 stem
    'p2':   96,
    'p3':   192,
    'p4':   384
}


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def conv_bn_relu(in_ch, out_ch, k=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ClassifierHead(nn.Module):
    """Shared FC classification head used by all evaluator nets."""

    def __init__(self, in_features: int, hidden: int = 256,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Level 0 -- Original image (S <= 16px, raw RGB)
# ---------------------------------------------------------------------------

class OriginalImageConv(nn.Module):
    """
    Stride=1 conv applied to the FULL original image before ROI crop.

    Only 864 parameters -- trained from scratch alongside evaluator heads.
    Replaces P1 stem for very small bboxes (S <= 16px) where stride=2
    would reduce patches to an unusable 4-8px short side.

    No spatial downsampling -- patches at this level are already tiny.
    """

    def __init__(self):
        super().__init__()
        self.conv = conv_bn_relu(3, 32, k=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, 32, H, W)
        return self.conv(x)


class OriginalImageNet(nn.Module):
    """
    Evaluator head for orig level.
    Input patch (post ROI crop): 32ch, short=8-16px, ratio 1:1-1:4.

    OriginalImageConv already ran on full map before ROI crop.
    Two additional 3x3 convs post-crop -- raw pixel features need
    more extraction even after the full-map conv.
    Padding=1, stride=1 throughout -- patches too small for downsampling.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.convs = nn.Sequential(
            conv_bn_relu(32, 64,  k=3, padding=1),
            conv_bn_relu(64, 128, k=3, padding=1),
        )
        self.head = ClassifierHead(128 * ROI_K * ROI_K,
                                   num_classes=num_classes)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        x = self.convs(patch)
        x = x.flatten(1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Level 1 -- P1 (16 < S <= 40px, YOLO P1 stem, stride=2, 48ch)
# ---------------------------------------------------------------------------

class P1FullMapConv(nn.Module):
    """
    Full-map conv for P1 level. Applied to entire P1 map before ROI crop.

    P1 has low-level features (edges, gradients) from YOLO stem.
    1x1 projection + 2x 3x3 spatial convs -- needs meaningful extraction.
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            conv_bn_relu(48,  128, k=1, padding=0),
            conv_bn_relu(128, 128, k=3, padding=1),
            conv_bn_relu(128, 128, k=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class P1Net(nn.Module):
    """
    Evaluator head for P1 level.
    Input patch (post ROI crop): 128ch, short=8-20px in feature space.
    Full-map conv already ran -- head just flattens and classifies.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.head = ClassifierHead(128 * ROI_K * ROI_K,
                                   num_classes=num_classes)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        x = patch.flatten(1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Level 2 -- P2 (40 < S <= 80px, stride=4, 96ch)
# ---------------------------------------------------------------------------

class P2FullMapConv(nn.Module):
    """
    Full-map conv for P2 level.
    Low-level features -- 1x1 proj + 2x 3x3 spatial convs.
    Same depth as P1 -- P2 is still shallow semantically.
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            conv_bn_relu(96,  128, k=1, padding=0),
            conv_bn_relu(128, 128, k=3, padding=1),
            conv_bn_relu(128, 128, k=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class P2Net(nn.Module):
    """
    Evaluator head for P2 level.
    Input patch (post ROI crop): 128ch, short=10-20px in feature space.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.head = ClassifierHead(128 * ROI_K * ROI_K,
                                   num_classes=num_classes)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        x = patch.flatten(1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Level 3 -- P3 (80 < S <= 160px, stride=8, 192ch)
# ---------------------------------------------------------------------------

class P3FullMapConv(nn.Module):
    """
    Full-map conv for P3 level.
    Mid-level features -- 1x1 proj + 1x 3x3 spatial conv.
    One fewer spatial conv than P1/P2 -- features are richer here.
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            conv_bn_relu(192, 128, k=1, padding=0),
            conv_bn_relu(128, 128, k=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class P3Net(nn.Module):
    """
    Evaluator head for P3 level.
    Input patch (post ROI crop): 128ch, short=10-20px in feature space.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.head = ClassifierHead(128 * ROI_K * ROI_K,
                                   num_classes=num_classes)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        x = patch.flatten(1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Level 4 -- P4 (160 < S <= 320px, stride=16, 384ch)
# ---------------------------------------------------------------------------

class P4FullMapConv(nn.Module):
    """
    Full-map conv for P4 level.
    Deep features -- 1x1 projection only. No spatial conv needed.
    Adding 3x3 on already-semantic features risks over-processing.
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            conv_bn_relu(384, 128, k=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class P4Net(nn.Module):
    """
    Evaluator head for P4 level.
    Input patch (post ROI crop): 128ch, short=10-20px in feature space.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.head = ClassifierHead(128 * ROI_K * ROI_K,
                                   num_classes=num_classes)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        x = patch.flatten(1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Full Evaluator Pipeline
# ---------------------------------------------------------------------------

class YOLOBBoxEvaluator(nn.Module):
    """
    Full evaluator pipeline for YOLOv8m predicted bboxes.

    Forward pass is vectorized and batched:
        1. Apply full-map convs to each feature map once per image
        2. Vectorized aspect ratio filter (tensor ops, no Python loop over bboxes)
        3. Vectorized level assignment via threshold comparisons
        4. Batched ROI align per level -- one roi_align call per level
        5. Batched net forward per level -- one forward pass per level
        6. Gather scores back into original bbox order

    Expected feature_maps keys and shapes (batch size B):
        'orig': (B, 3,   640, 640)  -- raw RGB image
        'p1':   (B, 48,  320, 320)  -- YOLO P1 stem output
        'p2':   (B, 96,  160, 160)
        'p3':   (B, 192,  80,  80)
        'p4':   (B, 384,  40,  40)

    bboxes: (B, A, 4) float tensor [x1, y1, x2, y2] original image scale
            A = number of anchor/predicted boxes (fixed across the batch)

    Returns: dict:
        'scores':   (B, A, num_classes) -- class logits; zero-filled for rejected
        'valid':    (B, A) bool         -- True if bbox was scored
        'rejected': (B, A) bool         -- True if ratio > 1:4 (not scored)
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 enabled_levels: list = None):
        super().__init__()
        self.num_classes = num_classes
        self.enabled_levels = enabled_levels if enabled_levels is not None else list(LEVEL_NAMES)

        # Full-map convs -- only instantiate for enabled levels
        if 'orig' in self.enabled_levels:
            self.orig_conv = OriginalImageConv()
        self.full_map_convs = nn.ModuleDict({
            lvl: cls() for lvl, cls in [
                ('p1', P1FullMapConv),
                ('p2', P2FullMapConv),
                ('p3', P3FullMapConv),
                ('p4', P4FullMapConv),
            ] if lvl in self.enabled_levels
        })

        # Evaluator heads -- only instantiate for enabled levels
        _head_ctors = {
            'orig': lambda: OriginalImageNet(num_classes),
            'p1':   lambda: P1Net(num_classes),
            'p2':   lambda: P2Net(num_classes),
            'p3':   lambda: P3Net(num_classes),
            'p4':   lambda: P4Net(num_classes),
        }
        self.nets = nn.ModuleDict({
            lvl: _head_ctors[lvl]() for lvl in self.enabled_levels
        })

    # ------------------------------------------------------------------
    # Static helpers -- vectorized, no Python loops over bboxes
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_aspect_ratio(bboxes: torch.Tensor,
                             max_ratio: float = MAX_ASPECT_RATIO
                             ) -> torch.Tensor:
        """
        Args:
            bboxes: (N, 4) [x1, y1, x2, y2]
        Returns:
            valid_mask: (N,) bool
        """
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        longer  = torch.max(w, h)
        shorter = torch.min(w, h)

        return ((longer / (shorter + 1e-6)) <= max_ratio) & (shorter <= 320)

    def _assign_levels(self, bboxes: torch.Tensor) -> torch.Tensor:
        """
        Vectorized level assignment based on shorter side S.

        Args:
            bboxes: (N, 4) [x1, y1, x2, y2]
        Returns:
            level_ids: (N,) long tensor
                0=orig, 1=p1, 2=p2, 3=p3, 4=p4, -1=disabled level
        """
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        shorter = torch.min(w, h)

        ids = torch.zeros(len(bboxes), dtype=torch.long, device=bboxes.device)
        ids[shorter >  20] = 1   # p1
        ids[shorter >  40] = 2   # p2
        ids[shorter >  80] = 3   # p3
        ids[shorter > 160] = 4   # p4

        # Mask out disabled levels -- -1 will be excluded by valid_mask downstream
        enabled_ids = {LEVEL_NAMES.index(lvl) for lvl in self.enabled_levels}
        all_ids = set(range(len(LEVEL_NAMES)))
        for disabled_id in (all_ids - enabled_ids):
            ids[ids == disabled_id] = -1

        return ids

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    # gt_mask if not None, it should be (B, A)
    # no gt overlap boxes will have False
    def forward(self, feature_maps: dict, bboxes: torch.Tensor, gt_mask = None) -> dict:
        B, A, _ = bboxes.shape
        device = next(iter(feature_maps.values())).device
        bboxes   = bboxes.to(device)
        # (B, A)
        if gt_mask is not None:
            gt_overlapping_box_mask = gt_mask.flatten()
        else:
            gt_overlapping_box_mask = torch.ones((B * A), dtype=bool, device=device)
            
        # ── Step 1: Full-map convs (batched over B, once per image) ────────
        processed = {}
        if 'orig' in self.enabled_levels:
            processed['orig'] = self.orig_conv(feature_maps['orig'])
        for level in ('p1', 'p2', 'p3', 'p4'):
            if level in self.enabled_levels:
                processed[level] = self.full_map_convs[level](feature_maps[level])

        # ── Step 2: Vectorized filter and level assignment ──────────────────
        flat = bboxes.view(B * A, 4)                              # (B*A, 4)
        level_ids   = self._assign_levels(flat)                   # (B*A,) long
        ratio_size_mask = self._filter_aspect_ratio(flat) & (level_ids != -1)
        
        valid_mask  = ratio_size_mask & gt_overlapping_box_mask     # (B*A,) bool
        valid_but_rejected_mask = (~ratio_size_mask) & gt_overlapping_box_mask

        # ROI format: [batch_idx, x1, y1, x2, y2] in image coords
        b_idx    = torch.arange(B, device=device).repeat_interleave(A)  # (B*A,)
        rois_all = torch.cat([b_idx.float().unsqueeze(1), flat], dim=1)  # (B*A, 5)

        valid_rois   = rois_all[valid_mask]      # (N_valid, 5)
        valid_levels = level_ids[valid_mask]     # (N_valid,)
        N_valid      = int(valid_mask.sum())

        # ── Step 3: Batched ROI align + net forward per level ───────────────
        all_logits = torch.zeros(B * A, self.num_classes, device=device)

        if N_valid > 0:
            level_logits = torch.zeros(N_valid, self.num_classes, device=device)

            for level_name in self.enabled_levels:
                level_id = LEVEL_NAMES.index(level_name)
                lvl_mask = (valid_levels == level_id)
                if not lvl_mask.any():
                    continue

                stride     = LEVEL_STRIDES[level_name]
                level_rois = valid_rois[lvl_mask].clone()
                level_rois[:, 1:] = level_rois[:, 1:] / stride

                patches = roi_align(
                    processed[level_name],
                    level_rois,
                    output_size=(ROI_K, ROI_K),
                    spatial_scale=1.0,
                    aligned=True,
                )   # (N_level, C, ROI_K, ROI_K)

                level_logits[lvl_mask] = self.nets[level_name](patches)

            all_logits[valid_mask] = level_logits

        # ── Step 4: Reshape to (B, A, num_classes) ──────────────────────────
        return {
            'scores':   all_logits.view(B, A, self.num_classes),  # (B, A, num_classes)
            'valid':    valid_mask.view(B, A),                    # (B, A) bool
            'rejected': valid_but_rejected_mask.view(B, A),       # (B, A) bool
        }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = torch.device('cpu')

    evaluator = YOLOBBoxEvaluator(num_classes=80).to(device)
    evaluator.eval()

    # Dummy feature maps -- as would come from frozen YOLOv8m backbone
    feature_maps = {
        'orig': torch.randn(2, 3,   640, 640),
        'p1':   torch.randn(2, 48,  320, 320),
        'p2':   torch.randn(2, 96,  160, 160),
        'p3':   torch.randn(2, 192,  80,  80),
        'p4':   torch.randn(2, 384,  40,  40),
        'p5':   torch.randn(2, 576,  20,  20),
    }

    # bboxes: (B=2, A=7, 4) -- same number of boxes per image, as in YOLO training
    bboxes = torch.tensor([
        [   # image 0: one bbox per level + one rejected
            [100, 100, 110, 115],    # S=10  -> orig  (10x15,  1:1.5)  valid
            [200, 200, 250, 235],    # S=30  -> p1    (35x50,  1:1.4)  valid
            [10,  10,  80,  70],     # S=60  -> p2    (60x70,  1:1.2)  valid
            [50,  50,  250, 170],    # S=120 -> p3    (120x200,1:1.7)  valid
            [0,   0,   300, 200],    # S=200 -> p4    (200x300,1:1.5)  valid
            [0,   0,   640, 400],    # S=400 -> p5    (400x640,1:1.6)  valid
            [10,  10,  20,  170],    # S=10  -> orig  (10x160, 1:16)   REJECTED
        ],
        [   # image 1
            [100, 100, 115, 112],    # S=12  -> orig  valid
            [50,  50,  450, 400],    # S=350 -> p5    REJECTED
            [30,  30,  100,  90],    # S=60  -> p2    valid
            [10,  10,  200, 180],    # S=170 -> p4    valid
            [20,  20,   90,  80],    # S=60  -> p2    valid
            [0,   0,   500, 340],    # S=340 -> p5    REJECTED
            [5,   5,   50,  45],     # S=40  -> p1    valid
        ],
    ], dtype=torch.float32)   # (2, 7, 4)

    with torch.no_grad():
        results = evaluator(feature_maps, bboxes)

    print(f'valid    : {results["valid"].tolist()}')
    print(f'rejected : {results["rejected"].tolist()}')
    print(f'scores   : {results["scores"].shape}')

    # Parameter breakdown
    p_orig  = sum(p.numel() for p in evaluator.orig_conv.parameters())
    p_fmc   = sum(p.numel() for p in evaluator.full_map_convs.parameters())
    p_heads = sum(p.numel() for p in evaluator.nets.parameters())
    p_total = sum(p.numel() for p in evaluator.parameters())

    print('\nParameter counts:')
    print(f'  OriginalImageConv (stride=1)  : {p_orig:>10,}')
    print(f'  Full-map convs (all levels)   : {p_fmc:>10,}')
    print(f'  Evaluator heads (all levels)  : {p_heads:>10,}')
    print(f'  Total (evaluator only)        : {p_total:>10,}')
    print(f'  Frozen YOLOv8m backbone       :          0  (not included)')
