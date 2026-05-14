# Small Object Detection on SeaDronesSee

고려대학교 2026년 1학기 COSE 474 00분반 딥러닝 프로젝트.

YOLOv8m을 베이스라인으로 두고 **작은 객체 탐지 (Small Object Detection)** 성능을 개선하기 위한 아키텍처 변형 실험. 해상 드론 영상 데이터셋 **SeaDronesSee Object Detection v2 (ODV2)** 에서 4가지 변형을 비교한다.

## Motivation

해상 SAR (Search and Rescue) 시나리오에서 드론은 사람 (`swimmer`), 구조 장비 (`life_saving_appliances`), 부표 (`buoy`) 같은 작은 객체를 정확히 찾아야 한다. 그러나 일반적인 YOLO 베이스라인은 큰 객체 (`boat`) 에선 잘 작동하지만 작은 객체에선:

- bbox 위치 정확도가 떨어지고 (AP@50 vs AP@50-95 격차가 큼)
- 정답을 놓치는 비율이 높다 (낮은 Recall)

본 프로젝트는 **receptive field 와 feature map 해상도** 두 측면에서 변형을 시도해 이 한계를 정량적으로 분석한다.

## Variants

| ID | Variant | Change | Hypothesis |
|---|---|---|---|
| M0 | baseline | unmodified YOLOv8m | 비교 기준 |
| M1 | sppf-k3 | SPPF kernel 5 → 3 | 작은 RF 가 작은 객체에 유리 |
| M2 | p2 | + P2 detection head (stride 4) | 고해상도 feature map 이 본질적 해결책 |
| M3 | p2-sppf-k3 | P2 + SPPF k=3 결합 | 두 변경의 시너지/충돌 검증 |

## Repository Layout

```
yolov8/              # detection-only YOLOv8 reimplementation
                     # (모델/로스 수학을 공식 코드와 비트 단위 검증)
  ├── model.py       # parse_model + YOLOv8 class
  ├── loss.py        # v8DetectionLoss, BboxLoss, DFLoss
  ├── tal.py         # TaskAlignedAssigner, anchors, dist↔bbox
  ├── modules/       # Conv, C2f, Bottleneck, SPPF, DFL, Detect
  ├── cfg/           # yolov8 yaml
  ├── tests/         # forward/loss equivalence vs official weights
  └── verify.py      # COCO weight transfer + forward allclose

experiments/         # SeaDronesSee 학습 파이프라인 (ultralytics Trainer 기반)
  ├── train.py       # 4가지 변형 통합 entry point
  ├── cfg/           # yolov8m, +p2, +sppf-k3, +p2-sppf-k3 yamls
  ├── data/sds.yaml  # SeaDronesSee ODV2 dataset descriptor (5 classes)
  ├── scripts/
  │   ├── convert_sds.py             # COCO JSON → YOLO txt
  │   ├── check_pretrained_transfer.py  # weight transfer sanity check
  │   └── summarize_runs.py          # 변형별 결과 일괄 평가
  ├── README.md      # 인스턴스 셋업 + 학습 가이드
  └── RESULTS.md     # 학습 결과 표 + 분석
```

## Method 요약

- **Base**: YOLOv8m (25.9M params, COCO `yolov8m.pt` 로 transfer learning)
- **Dataset**: SeaDronesSee ODV2 — 5 classes (swimmer, boat, jetski, life_saving_appliances, buoy)
- **Training**: SGD (lr0=0.01) + cosine decay, 100 epochs, patience=30, batch=16, imgsz=640, AMP
- **모든 변형이 동일 하이퍼파라미터** — 아키텍처 차이만 비교
- **Hardware**: NVIDIA A100 80GB PCIe MIG 3g.40gb (Elice)

## Status

자세한 결과/분석은 [`experiments/RESULTS.md`](experiments/RESULTS.md) 참고.

- [x] M0 baseline — trained + evaluated
- [x] M1 sppf-k3 — trained + evaluated
- [ ] M2 p2 — training in progress
- [ ] M3 p2-sppf-k3 — pending

## Reproduction

```bash
# 1. 환경
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install torch torchvision pyyaml numpy ultralytics opencv-python-headless

# 2. 데이터 준비 (SeaDronesSee ODV2)
#    https://www.kaggle.com/datasets/ubiratanfilho/sds-dataset
python experiments/scripts/convert_sds.py \
    --coco /path/to/instances_train.json \
    --images /path/to/images/train \
    --out /path/to/sds --split train
# val 도 동일

# 3. COCO pretrained 가중치 다운로드
curl -L -o experiments/weights/yolov8m.pt \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt

# 4. transfer 검증
python experiments/scripts/check_pretrained_transfer.py

# 5. 학습 (4가지 변형 각각)
python experiments/train.py --model baseline
python experiments/train.py --model sppf-k3
python experiments/train.py --model p2
python experiments/train.py --model p2-sppf-k3

# 6. 결과 비교 표 자동 생성
python experiments/scripts/summarize_runs.py
```

## License

코드는 ultralytics (AGPL-3.0) 기반이므로 동일 라이선스를 따른다.
