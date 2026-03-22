# 릴레이 소켓 3D Bin-Picking 시스템

RealSense D435i + YOLOv8 + Indy7 기반 소켓(8핀/12핀) 자동 빈픽킹 시스템

## 시스템 개요

```
카메라(D435i) → YOLOv8 탐지 → 3D 좌표 계산 → 표면 법선 추정 → 접근 벡터 → Indy7 픽앤플레이스
```

## 프로젝트 구조

```
Intel5/
├── config/                         # 설정
│   ├── paths.py                    # 경로 설정
│   ├── robot_config.py             # Indy7 로봇 설정 (IP, 속도, 캘리브레이션)
│   └── calibration_data/           # Hand-eye 캘리브레이션 데이터 (gitignore)
│
├── src/                            # 소스 코드
│   ├── detection/                  # 탐지 & 빈픽킹
│   │   ├── binpicking_3d.py        # 3D 탐지 + 빈픽킹 메인 시스템
│   │   └── test_detect.py          # 실시간 탐지 테스트
│   │
│   ├── pipeline/                   # 데이터 수집 & 학습 파이프라인
│   │   ├── step1_capture.py        # D435 RGB+Depth 촬영
│   │   ├── step2_depth_mask.py     # Depth 기반 배경 제거
│   │   ├── step3_sam_labeler.py    # SAM 클릭 라벨링
│   │   ├── step3_1_auto_labeler.py # 자동 라벨러
│   │   ├── step4_train_yolo.py     # YOLOv8 학습
│   │   └── step5_pseudo_label.py   # Pseudo-labeling 자동 확장
│   │
│   ├── robot/                      # 로봇 제어
│   │   └── indy_controller.py      # Indy7 래퍼 (픽/플레이스/진공)
│   │
│   └── calibration/                # 캘리브레이션
│       └── hand_eye_calibration.py # Hand-Eye 캘리브레이션 (ArUco)
│
├── robot/                          # Indy7 SDK & 예제
│   ├── indy_utils/                 # IndyDCP 클라이언트 라이브러리
│   ├── src/                        # 기존 로봇 예제 코드
│   └── 250213/                     # 추가 예제
│
├── models/                         # 모델 가중치 (gitignore)
├── data/                           # 이미지 데이터 (gitignore)
├── runs/                           # YOLO 학습 결과
└── dataset.yaml                    # YOLOv8 데이터셋 설정 (8pin, 12pin)
```

## 실행 방법

### 1. 데이터 수집 & 학습

```bash
# 이미지 촬영
python src/pipeline/step1_capture.py

# 배경 제거
python src/pipeline/step2_depth_mask.py

# SAM 라벨링
python src/pipeline/step3_sam_labeler.py

# YOLOv8 학습
python src/pipeline/step4_train_yolo.py

# Pseudo-labeling 확장
python src/pipeline/step5_pseudo_label.py
```

### 2. 3D 탐지 & 빈픽킹

```bash
# 실시간 3D 탐지 (시각화)
python src/detection/binpicking_3d.py

# 빈픽킹 모드 (시뮬레이션)
python src/detection/binpicking_3d.py --pick

# 빈픽킹 모드 (실제 Indy7 연결)
python src/detection/binpicking_3d.py --pick --robot
```

### 3. Hand-Eye 캘리브레이션

```bash
# ArUco 마커 생성 (인쇄 후 그리퍼에 부착)
python src/calibration/hand_eye_calibration.py --generate-marker

# 캘리브레이션 데이터 수집 (로봇 연결)
python src/calibration/hand_eye_calibration.py --collect --robot

# 변환 행렬 계산
python src/calibration/hand_eye_calibration.py --calibrate

# 결과 검증
python src/calibration/hand_eye_calibration.py --verify
```

## 환경

- Python 3.12 + CUDA 12.4
- Intel RealSense D435i (pyrealsense2)
- YOLOv8 (ultralytics)
- Neuromeka Indy7 (IndyDCP)
- OpenCV, NumPy, SciPy, SAM
