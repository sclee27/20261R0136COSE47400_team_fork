"""YOLOv8 model builder.

Builds a YOLOv8{n,s,m,l,x} ``nn.Module`` from ``cfg/yolov8.yaml`` + a scale
letter, mirroring ``ultralytics/nn/tasks.py`` but trimmed to detection only.

Key entry points:
    parse_model(cfg, ch=3, scale="m", verbose=True)
        Build the bare ``nn.Sequential`` + savelist from a parsed yaml dict.

    YOLOv8(cfg="...", scale="m", ch=3, nc=None, verbose=True)
        Full detection model. ``forward`` returns whatever the Detect head
        returns:

        * training=True            → dict(boxes=..., scores=..., feats=feats)
        * training=False (eval)    → (decoded, raw_dict)
"""

from __future__ import annotations

import ast
import contextlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from .modules import DFL, SPPF, Bottleneck, C2f, Concat, Conv, Detect
from .ops import make_divisible

__all__ = ["YOLOv8", "parse_model", "load_yaml"]

# All modules eligible to appear in cfg/yolov8.yaml. Extending the model with
# new blocks is just a matter of adding them here (plus to the channel-handling
# branches in parse_model below).
_BASE_MODULES = frozenset({Conv, Bottleneck, SPPF, C2f})
_REPEAT_MODULES = frozenset({C2f})


# ---------------------------------------------------------------- yaml loader


def load_yaml(path: str | Path) -> dict:
    """Load a YOLO yaml from disk (relative to the package or absolute)."""
    p = Path(path)
    if not p.is_absolute():
        # Resolve against this file's directory so `yolov8.yaml` works from anywhere.
        p = Path(__file__).parent / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------ parse_model


def parse_model(cfg: dict, ch: int = 3, scale: str = "m", verbose: bool = True):
    """Translate the parsed yaml into a Sequential model + savelist.

    Args:
        cfg: parsed yaml dict (must have ``backbone``, ``head``, ``nc`` and
            ``scales``).
        ch: input channel count.
        scale: model scale letter (n/s/m/l/x).
        verbose: whether to log layer summaries.

    Returns:
        model:       ``nn.Sequential`` of all layers (each with ``.i``, ``.f``,
                     ``.type``, ``.np`` attributes attached for routing & info).
        save:        sorted list of layer indices whose outputs are reused as
                     ``from`` arguments by later layers.
    """
    cfg = deepcopy(cfg)
    nc = cfg["nc"]
    scales = cfg["scales"]
    if scale not in scales:
        raise KeyError(f"scale={scale!r} not in {list(scales)}")
    depth, width, max_channels = scales[scale]
    reg_max = cfg.get("reg_max", 16)

    if verbose:
        print(f"{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<35}{'arguments'}")

    channels = [ch]
    layers: list[nn.Module] = []
    save: list[int] = []

    for i, (f, n, m_name, args) in enumerate(cfg["backbone"] + cfg["head"]):
        # ---- resolve module class --------------------------------------
        if isinstance(m_name, str) and m_name.startswith("nn."):
            m = getattr(nn, m_name[3:])
        else:
            m = _MODULE_REGISTRY[m_name]

        # ---- evaluate string args -------------------------------------
        # Two cases:
        #   * "nc" / "reg_max" — refer to scalars defined above (mirrors ultralytics' locals() lookup).
        #   * "None" / numeric literals — parsed with ast.literal_eval.
        args = list(args)
        _name_table = {"nc": nc, "reg_max": reg_max}
        for j, a in enumerate(args):
            if isinstance(a, str):
                if a in _name_table:
                    args[j] = _name_table[a]
                else:
                    with contextlib.suppress(ValueError, SyntaxError):
                        args[j] = ast.literal_eval(a)

        # ---- depth multiplier -----------------------------------------
        n = n_repeat_display = max(round(n * depth), 1) if n > 1 else n

        # ---- per-module-class channel handling ------------------------
        if m in _BASE_MODULES:
            c1, c2 = channels[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in _REPEAT_MODULES:
                args.insert(2, n)  # number of repeats handled internally
                n = 1
        elif m is nn.Upsample:
            c2 = channels[f]
        elif m is Concat:
            c2 = sum(channels[x] for x in f)
        elif m is Detect:
            # head args: [nc] from yaml → extend with [reg_max, ch_tuple]
            args.extend([reg_max, [channels[x] for x in f]])
            c2 = None  # head produces dict / tensor, not a feature map
        else:
            c2 = channels[f]

        # ---- instantiate ----------------------------------------------
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        type_str = m.__name__ if hasattr(m, "__name__") else str(m)
        m_.i, m_.f, m_.type = i, f, type_str
        m_.np = sum(x.numel() for x in m_.parameters())
        if verbose:
            print(f"{i:>3}{str(f):>20}{n_repeat_display:>3}{m_.np:10.0f}  {type_str:<35}{args}")

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)

        if i == 0:
            channels = []  # discard input ch so channels[f] indexing matches layer indices
        channels.append(c2 if c2 is not None else channels[-1])

    return nn.Sequential(*layers), sorted(set(save))


_MODULE_REGISTRY = {
    "Conv": Conv,
    "Bottleneck": Bottleneck,
    "C2f": C2f,
    "SPPF": SPPF,
    "DFL": DFL,
    "Concat": Concat,
    "Detect": Detect,
}


# --------------------------------------------------------------- YOLOv8 class


def _initialize_weights(model: nn.Module) -> None:
    """Mirror ultralytics initialize_weights (BN momentum/eps + activation inplace)."""
    for m in model.modules():
        t = type(m)
        if t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


class YOLOv8(nn.Module):
    """YOLOv8 detection model — backbone + neck + Detect head.

    ``args`` (an ``argparse.Namespace``-like object exposing ``box``, ``cls``,
    ``dfl``) is what the loss reads its gain weights from; default values match
    the official ``ultralytics/cfg/default.yaml`` so a from-scratch run matches
    the published recipe.
    """

    def __init__(
        self,
        cfg: str | dict = "cfg/yolov8.yaml",
        scale: str = "m",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else load_yaml(cfg)
        if nc is not None and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc
        self.nc = self.yaml["nc"]
        self.scale = scale

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, scale=scale, verbose=verbose)
        self.names = {i: str(i) for i in range(self.nc)}

        # ---- compute strides via a dummy forward (eval mode) ----------
        m = self.model[-1]
        if not isinstance(m, Detect):
            raise RuntimeError("Last layer must be Detect")
        s = 256
        self.model.eval()
        m.training = True  # ensures _forward_head dict is returned, regardless of self.training
        with torch.no_grad():
            feats = self._forward_for_stride(torch.zeros(1, ch, s, s))["feats"]
        m.stride = torch.tensor([s / f.shape[-2] for f in feats])
        self.stride = m.stride
        self.model.train()
        m.bias_init()

        _initialize_weights(self)

        # Default loss-gain hyperparameters (matches ultralytics default.yaml).
        # Override via `self.args.box = ...` etc. before constructing v8DetectionLoss.
        self.args = _Args(box=7.5, cls=0.5, dfl=1.5)

        if verbose:
            print(
                f"YOLOv8{scale} built: nc={self.nc}, "
                f"params={sum(p.numel() for p in self.parameters()):,}, "
                f"strides={self.stride.tolist()}"
            )

    # ------------------------------------------------------------------ run

    def forward(self, x: torch.Tensor):
        return self._predict_once(x)

    def _predict_once(self, x: torch.Tensor):
        y: list[Any] = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def _forward_for_stride(self, x: torch.Tensor) -> dict:
        """Forward used only at build-time to discover strides. Returns the raw head dict."""
        return self._predict_once(x)


class _Args:
    """Tiny stand-in for ``ultralytics.utils.IterableSimpleNamespace`` used only
    to feed loss-gain hyperparameters to v8DetectionLoss."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
