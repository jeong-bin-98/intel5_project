# 🔌 전기기능사 릴레이 소켓 (8핀/12핀) 자동 라벨링 파이프라인

## 왜 이 방법을 쓰는가?

릴레이 소켓(MY2 8핀, MY4 14핀 등)은 일반 AI 모델이 학습한 적 없는 **산업용 특수 부품**입니다.
→ "socket"이라고 텍스트 프롬프트를 줘도 Grounded SAM/DINO가 못 찾음
→ **직접 소량 라벨링 → 학습 → 자동 확장** 전략이 가장 현실적

## 3단계 부트스트래핑 전략

```
[1단계] D435 depth로 배경 제거 + SAM 클릭 라벨링 (50~100장)
   ↓
[2단계] YOLOv8 1차 학습 (소량 데이터로 모델 생성)
   ↓
[3단계] 1차 모델로 나머지 이미지 자동 라벨링 (Pseudo-labeling)
   ↓
[반복] 자동 라벨 검수 → 2차 학습 → 더 정확한 모델
```

## 환경 설정

```bash
# 1. 가상환경 만들기
python -m venv binpick
source binpick/bin/activate  # Windows: binpick\Scripts\activate

# 2. 필수 패키지 설치
pip install pyrealsense2          # D435 카메라 드라이버
pip install opencv-python          # 이미지 처리
pip install numpy                  # 수치 연산
pip install ultralytics            # YOLOv8 (학습 + 추론)
pip install segment-anything       # Meta SAM (세그멘테이션)
pip install torch torchvision      # PyTorch (SAM/YOLO 공통)
pip install matplotlib             # 시각화

# 3. SAM 모델 가중치 다운로드 (한 번만)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## 폴더 구조

```
bin_picking_labeling/
├── README.md                  ← 지금 이 파일
├── step1_capture.py           ← D435로 이미지 촬영
├── step2_depth_mask.py        ← depth로 배경 제거
├── step3_sam_labeler.py       ← SAM 클릭 라벨링 도구
├── step4_train_yolo.py        ← YOLOv8 1차 학습
├── step5_pseudo_label.py      ← 자동 라벨링 (Pseudo-labeling)
├── sam_vit_h_4b8939.pth       ← SAM 가중치 (다운로드)
├── dataset/
│   ├── images/
│   │   ├── train/             ← 학습용 이미지
│   │   └── val/               ← 검증용 이미지
│   └── labels/
│       ├── train/             ← 학습용 라벨 (YOLO txt)
│       └── val/               ← 검증용 라벨
└── dataset.yaml               ← YOLOv8 데이터셋 설정
```

## 각 단계 상세 설명

### 1단계: 이미지 촬영 + depth 배경 제거 + SAM 라벨링

**목표**: 50~100장의 정확한 라벨 데이터 확보

D435 카메라는 일반 RGB 사진 + depth(거리) 정보를 동시에 줍니다.
이 depth를 활용하면:
- 작업대까지 거리: 약 80cm → depth > 0.7m인 픽셀은 전부 배경
- 소켓까지 거리: 약 50cm → 0.2m < depth < 0.7m인 픽셀만 전경(소켓)

이렇게 배경을 날린 깨끗한 이미지에서 SAM으로 클릭 한 번 하면
소켓 윤곽이 자동으로 잡히고, 사람은 "8pin" / "12pin"만 지정하면 됩니다.

### 2단계: YOLOv8 1차 학습

**목표**: 소량 데이터로 1차 탐지 모델 생성

50~100장이 적어 보이지만, 클래스가 2개뿐이고 크기 차이가 명확해서
YOLOv8n(nano) 모델로도 충분히 높은 정확도가 나옵니다.
학습 시간도 GPU 있으면 5~10분, CPU로도 30분 내외입니다.

### 3단계: Pseudo-labeling (자동 라벨 확장)

**목표**: 수백~수천 장으로 데이터 확장

1차 모델이 새 이미지를 추론 → confidence 80% 이상인 결과만 자동 라벨로 저장
→ 사람이 빠르게 검수(틀린 것만 삭제) → 2차 학습
→ 반복할수록 모델이 좋아지고, 라벨링 속도도 빨라짐
