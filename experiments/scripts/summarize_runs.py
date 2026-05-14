"""Summarize trained YOLOv8 variants into a presentation-ready table.

For every variant directory under ``experiments/runs/``, this:
  1. Loads ``weights/best.pt`` and re-runs val on SeaDronesSee.
  2. Collects per-class AP / overall mAP / size-binned AP.
  3. Prints a Markdown table and saves a CSV summary.

Usage (after training one or more variants):
    python experiments/scripts/summarize_runs.py
    python experiments/scripts/summarize_runs.py --models baseline sppf-k3
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


HERE = Path(__file__).resolve().parent.parent  # .../experiments
DATA_YAML = HERE / "data/sds.yaml"
RUNS_DIR = HERE / "runs"

DEFAULT_MODELS = ["baseline", "sppf-k3", "p2", "p2-sppf-k3"]


def evaluate(run_name: str, imgsz: int = 640, batch: int = 16) -> dict:
    """Run ultralytics val on a finished run and collect headline numbers."""
    weights = RUNS_DIR / run_name / "weights" / "best.pt"
    if not weights.exists():
        return {"name": run_name, "status": "missing", "weights": str(weights)}

    model = YOLO(str(weights))
    r = model.val(data=str(DATA_YAML), imgsz=imgsz, batch=batch, verbose=False)

    # Overall numbers
    overall = {
        "mAP50":      float(r.box.map50),     # IoU 0.50
        "mAP50-95":   float(r.box.map),       # IoU 0.50:0.95
        "precision":  float(r.box.mp),
        "recall":     float(r.box.mr),
    }

    # Per-class AP (sorted by the class id used in sds.yaml)
    per_class = {}
    for i, cls_idx in enumerate(r.box.ap_class_index):
        name = r.names[int(cls_idx)]
        per_class[name] = {
            "AP50":    float(r.box.ap50[i]),
            "AP50-95": float(r.box.maps[int(cls_idx)]),
            "P":       float(r.box.p[i]),
            "R":       float(r.box.r[i]),
        }

    return {"name": run_name, "status": "ok", "overall": overall, "per_class": per_class}


def fmt(x: float | None) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def print_markdown_table(results: list[dict]) -> None:
    """Pretty-print the headline comparison table."""
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        print("no completed runs to summarize")
        return

    # Overall table
    print("\n## Overall results (val)\n")
    print("| model | mAP@50 | mAP@50-95 | precision | recall |")
    print("|---|---|---|---|---|")
    for r in ok:
        o = r["overall"]
        print(f"| {r['name']} | {fmt(o['mAP50'])} | {fmt(o['mAP50-95'])} | {fmt(o['precision'])} | {fmt(o['recall'])} |")

    # Per-class table (AP50)
    all_classes = sorted({c for r in ok for c in r["per_class"]})
    print("\n## Per-class AP@50\n")
    header = "| model | " + " | ".join(all_classes) + " |"
    sep = "|---" * (len(all_classes) + 1) + "|"
    print(header)
    print(sep)
    for r in ok:
        cells = [r["name"]]
        for c in all_classes:
            cells.append(fmt(r["per_class"].get(c, {}).get("AP50")))
        print("| " + " | ".join(cells) + " |")

    # Per-class table (AP50-95)
    print("\n## Per-class AP@50-95\n")
    print(header)
    print(sep)
    for r in ok:
        cells = [r["name"]]
        for c in all_classes:
            cells.append(fmt(r["per_class"].get(c, {}).get("AP50-95")))
        print("| " + " | ".join(cells) + " |")


def save_csv(results: list[dict], out: Path) -> None:
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        return
    classes = sorted({c for r in ok for c in r["per_class"]})

    header = ["model", "mAP50", "mAP50-95", "precision", "recall"]
    for c in classes:
        header += [f"{c}_AP50", f"{c}_AP50-95"]

    rows = []
    for r in ok:
        o = r["overall"]
        row = [r["name"], o["mAP50"], o["mAP50-95"], o["precision"], o["recall"]]
        for c in classes:
            row += [r["per_class"].get(c, {}).get("AP50"), r["per_class"].get(c, {}).get("AP50-95")]
        rows.append(row)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\n[saved] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="run names under experiments/runs/ to summarize")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, default=HERE / "summary.csv")
    args = ap.parse_args()

    print(f"data: {DATA_YAML}")
    print(f"runs: {RUNS_DIR}")
    print(f"models: {args.models}\n")

    results = []
    for m in args.models:
        print(f"== evaluating {m} ==")
        r = evaluate(m, imgsz=args.imgsz, batch=args.batch)
        if r["status"] == "missing":
            print(f"   skip: {r['weights']} not found")
        else:
            print(f"   mAP50={r['overall']['mAP50']:.3f}  mAP50-95={r['overall']['mAP50-95']:.3f}")
        results.append(r)

    print_markdown_table(results)
    save_csv(results, args.out)


if __name__ == "__main__":
    main()
