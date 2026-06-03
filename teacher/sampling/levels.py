"""Block: level assignment + enabled filter.

A box's level is decided by its shorter side (orig < p1 < ... < p5).
Levels absent from the enabled list are treated as 'dropped'.
"""
from __future__ import annotations

import numpy as np

LEVEL_ORDER = ["orig", "p1", "p2", "p3", "p4", "p5"]


def assign_levels(boxes: np.ndarray, upper_short: dict) -> np.ndarray:
    """Assign each box to a level index (0..5 over LEVEL_ORDER).

    shorter <= upper[orig] -> orig, next band -> p1, and so on.
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    shorter = np.minimum(w, h)
    uppers = np.array([upper_short[n] for n in LEVEL_ORDER], dtype=float)  # ascending
    ids = np.searchsorted(uppers, shorter, side="left")
    return np.clip(ids, 0, len(LEVEL_ORDER) - 1).astype(int)


def enabled_mask(level_ids: np.ndarray, enabled: list) -> np.ndarray:
    """True for boxes whose level is in the enabled list."""
    enabled_ids = {LEVEL_ORDER.index(n) for n in enabled}
    return np.array([i in enabled_ids for i in level_ids], dtype=bool)


def level_counts(level_ids: np.ndarray) -> dict:
    """level name -> count (all 6 levels included)."""
    counts = {n: 0 for n in LEVEL_ORDER}
    for i in level_ids:
        counts[LEVEL_ORDER[i]] += 1
    return counts
