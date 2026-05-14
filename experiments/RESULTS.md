# SeaDronesSee ODV2 — Experiment Results

Comparison of YOLOv8m architecture variants on SeaDronesSee Object Detection v2.
All numbers below are **val-set results at the best epoch**, measured by re-running
`model.val()` on the saved `best.pt` (see `scripts/summarize_runs.py`).

---

## Setup

| Item | Value |
|---|---|
| Base model | YOLOv8m (25.9M params) |
| Pretrained | COCO `yolov8m.pt` (transferred into each variant; shape-incompatible layers re-init) |
| Dataset | SeaDronesSee Object Detection V2|
| Train / Val images | 8,930 / 1,547 |
| Total bboxes (train) | 57,760 |
| Classes (5, `ignored` skipped) | swimmer, boat, jetski, life_saving_appliances, buoy |
| Image size | 640 |
| Batch | 16 |
| Optimizer | SGD (lr0=0.01, momentum=0.937, weight_decay=0.0005) |
| Schedule | warmup 3 ep → cosine; 100 epochs with `patience=30` |
| Augmentation | Mosaic + HSV + scale/translate + fliplr (no flipud) |
| Precision | AMP (mixed precision) |
| Hardware | A100 80GB PCIe MIG 3g.40gb (Elice) |

Every variant trains under **identical** hyperparameters; only the architecture cfg differs.

---

## Variants

| ID | Variant | Cfg file | Change vs baseline |
|---|---|---|---|
| **M0** | baseline | `cfg/yolov8m.yaml` | — (unmodified YOLOv8m) |
| **M1** | sppf-k3 | `cfg/yolov8m-sppf-k3.yaml` | SPPF kernel 5 → 3 (smaller receptive field) |
| M2 | p2 | `cfg/yolov8m-p2.yaml` | + P2 detection head (stride 4) — *pending* |
| M3 | p2-sppf-k3 | `cfg/yolov8m-p2-sppf-k3.yaml` | P2 + SPPF k=3 combined — *pending* |

---

## Overall Results (val)

| Model | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| baseline | **0.732** | **0.439** | 0.889 | 0.708 |
| sppf-k3 | 0.731 | 0.439 | 0.885 | **0.719** |

**Δ (sppf-k3 − baseline)**: mAP@50 −0.001, mAP@50-95 ±0.000, P −0.004, R **+0.011**

Overall performance is essentially **tied** — but recall edges up slightly with SPPF k=3, which is the metric most relevant to small-object detection (catching all instances).

---

## Per-Class AP@50

| Model | boat | buoy | jetski | life_saving_appliances | swimmer |
|---|---|---|---|---|---|
| baseline | **0.965** | 0.639 | 0.911 | **0.378** | **0.766** |
| sppf-k3 | 0.963 | **0.658** | **0.924** | 0.356 | 0.753 |

**Δ (sppf-k3 − baseline)** per class:
- boat: −0.002 (큰 객체, 이미 거의 ceiling)
- **buoy: +0.019** (작은 객체, 개선)
- **jetski: +0.013** (중간 객체, 개선)
- life_saving_appliances: **−0.022** (작은 객체지만 악화)
- swimmer: −0.013 (작은 객체, 약간 악화)

---

## Per-Class AP@50-95 (stricter, COCO-style)

| Model | boat | buoy | jetski | life_saving_appliances | swimmer |
|---|---|---|---|---|---|
| baseline | **0.724** | **0.390** | 0.596 | **0.181** | **0.307** |
| sppf-k3 | 0.721 | 0.388 | **0.632** | 0.156 | 0.300 |

**Δ (sppf-k3 − baseline)** per class:
- boat: −0.003
- buoy: −0.002
- **jetski: +0.036** (가장 큰 개선)
- life_saving_appliances: **−0.025** (가장 큰 악화)
- swimmer: −0.007

---

## Analysis

### Aggregate level — SPPF k=3 makes no overall difference

전체 mAP@50 / mAP@50-95 가 **거의 동일** (Δ < 0.002). "SPPF 커널만 줄여서 작은 객체 탐지를 개선한다"는 단순 가설은 **이 결과로는 지지되지 않음**.

### Per-class level — Trade-off가 명확히 보임

| 효과 | 클래스 |
|---|---|
| 의미있게 개선 (AP +1% 이상) | **buoy** (+0.019 @50), **jetski** (+0.013 @50, +0.036 @50-95) |
| 거의 무변화 | boat |
| 의미있게 악화 (AP −1% 이상) | **life_saving_appliances** (−0.022 @50, −0.025 @50-95), swimmer (−0.013 @50) |

**해석**:
- 작은 receptive field가 *어떤* 작은 객체엔 도움 (buoy — 형태가 단순한 부유물)
- 다른 작은 객체엔 오히려 손해 (life_saving_appliances, swimmer — 더 큰 컨텍스트 정보가 필요한 객체)
- 즉 "작은 객체 = 작은 RF 유리"라는 단순 매핑은 **틀림**

### Class imbalance caveat — life_saving_appliances

이 클래스는 val 인스턴스가 적어 (~330개) AP 변동성이 큼. ±0.02 정도의 차이는 통계적으로 유의하지 않을 수도. 추가 실험 (seed 변경, 더 긴 학습) 으로 robust한지 확인 필요.

### Takeaway for next experiments

1. **SPPF k 변경 단독으로는 약한 기여** — ablation의 한 항목으로 두되, main contribution으로 삼기 어려움
2. **P2 head 추가 (M2) 가 진짜 효과 확인 포인트** — feature map 해상도 자체를 늘리는 변화라 작은 객체에 더 본질적
3. **클래스별 trade-off가 보인다는 것 자체가 발표 메시지** — "수정이 모든 클래스에 일률적으로 적용되지 않는다"는 관찰은 후속 실험 방향 (클래스/크기별 differentiated module) 의 motivation

---

## Status

- [x] M0 baseline trained + evaluated
- [x] M1 sppf-k3 trained + evaluated
- [ ] M2 p2 — *training pending*
- [ ] M3 p2-sppf-k3 — *training pending (시간 여유 시)*

---

## Reproduction

```bash
# train
python experiments/train.py --model baseline
python experiments/train.py --model sppf-k3
python experiments/train.py --model p2
python experiments/train.py --model p2-sppf-k3

# evaluate (best.pt val 재평가)
python experiments/scripts/summarize_runs.py --models baseline sppf-k3 p2 p2-sppf-k3
```

산출물:
- `experiments/runs/<model>/weights/best.pt` — 체크포인트
- `experiments/runs/<model>/results.csv` — epoch별 metric
- `experiments/runs/<model>/results.png` — 학습 곡선
- `experiments/runs/<model>/confusion_matrix.png` — confusion matrix
- `experiments/summary.csv` — 4개 모델 일괄 비교
