# 릴레이 소켓 3D Bin-Picking 시스템

RealSense D435i + YOLO11-OBB + Indy7 기반 소켓(8핀/12핀) 자동 흡착 빈픽킹 시스템

## 시스템 개요

```
카메라(D435i) → YOLO11-OBB (객체 탐지 및 Yaw 각도 추출) → RANSAC 3D 평면 피팅 → 흡착점 및 법선 계산 → Indy7 픽앤플레이스
```

## 프로젝트 구조

```
Intel5/
├── config/                         # 설정
│   ├── paths.py                    # 경로 설정 (전체 프로젝트 공통 경로)
│   ├── robot_config.py             # Indy7 로봇 설정 (IP, 속도, 캘리브레이션)
│   └── calibration_data/           # Hand-eye 캘리브레이션 데이터 (gitignore)
│
├── src/                            # 소스 코드
│   ├── detection/                  # 탐지 & 빈픽킹 (모듈화)
│   │   ├── binpicking_3d.py        # 3D 탐지 + 빈픽킹 메인 시스템
│   │   ├── constants.py            # 장치/신뢰도/클래스명 상수
│   │   ├── geometry.py             # 3D 좌표·법선·접근벡터 계산
│   │   ├── realsense_config.py     # RealSense 카메라 설정
│   │   ├── tracker.py              # 객체 추적기
│   │   ├── visualization.py        # 시각화 유틸리티
│   │   ├── fast_brisk_pose.py      # BRISK 특징점 기반 자세 추정
│   │   └── tests/                  # 단위 테스트
│   │       ├── test_detect.py      # 실시간 탐지 테스트
│   │       ├── test_3d_pose.py     # 3D 자세 추정 테스트
│   │       ├── test_fast_brisk.py  # BRISK 알고리즘 테스트
│   │       └── benchmark/          # 추론 속도 벤치마크 (CPU/OpenVINO/XPU)
│   │
│   ├── pipeline/                   # OBB 데이터 수집 & 학습 파이프라인
│   │   ├── step1_capture.py        # D435 RGB+Depth 촬영
│   │   ├── step2_sam_labeler.py    # OBB 라벨링 (SAM 보조, 클릭 자동 윤곽)
│   │   ├── step2_obb_labeler.py    # OBB 라벨링 (SAM + 수동 4-코너)
│   │   └── step3_train_yolo_obb.py # YOLO11-OBB 학습 (Intel XPU 지원)
│   │
│   ├── robot/                      # 로봇 제어
│   │   └── indy_controller.py      # Indy7 래퍼 (픽/플레이스/진공)
│   │
│   ├── calibration/                # 캘리브레이션
│   │   └── hand_eye_calibration.py # Hand-Eye 캘리브레이션 (ArUco)
│   │
│   └── gputest.py                  # Intel XPU 동작 확인 스크립트
│
├── robot/                          # Indy7 SDK & 예제
├── models/                         # OBB 모델 가중치 (gitignore)
├── data/                           # 이미지 데이터 (gitignore)
├── runs/                           # YOLO 학습 결과
├── step4_train_yolo.py             # (루트) 추가 학습 실험용 스크립트
├── dataset_obb.yaml                # YOLO OBB 데이터셋 설정 (8pin, 12pin)
└── install.sh                      # Intel XPU 환경 패키지 설치 스크립트
```

## 초기 설정 (Setup)

다른 컴퓨터나 로봇 제어기 환경에서 프로젝트를 그대로 복원하기 위한 세팅 방법입니다.

### 1. 패키지 라이브러리 설치

동봉된 `requirements.txt`를 사용하여 동일한 파이썬 환경을 구성합니다.

> **주의:** `torch`, `intel_extension_for_pytorch` 등 `+xpu` 빌드는 PyPI에 없고 별도 인덱스에서 제공됩니다.
> 반드시 아래 방법으로 설치하세요.

```bash
# 가상환경 생성 및 활성화 (Linux)
python -m venv .venv
source .venv/bin/activate

# 의존성 설치 (install.sh 사용)
bash install.sh
```

`install.sh`는 아래 명령어와 동일합니다:

```bash
pip install -r requirements.txt \
  --index-url https://download.pytorch.org/whl/xpu \
  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
  --extra-index-url https://pypi.org/simple/
```

### 2. 베이스 모델 가중치 오프라인 다운로드

인터넷이 불안정한 산업 현장에서 파이프라인이 다운로드 지연 없이 곧바로 실행될 수 있도록 기초 모델 가중치들을 미리 다운로드해 둡니다.

```bash
cd models

# 1) 학습용 YOLO11-OBB 기초 가중치 다운로드
python -c "from ultralytics import YOLO; YOLO('yolo11n-obb.pt')"

# 2) 라벨링용 SAM (Segment Anything) 가중치 다운로드
#    vit_b (빠름, 기본)
wget -O sam_vit_b_01ec64.pth "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
#    vit_h (고정밀, 선택)
# wget -O sam_vit_h_4b8939.pth "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

cd ..
```

### 3. Intel XPU 동작 확인

```bash
python src/gputest.py
```

## 실행 방법

### 1. 데이터 수집 & 학습

```bash
# Step 1: D435i로 이미지 촬영 (RGB + Depth)
python src/pipeline/step1_capture.py

# Step 2: OBB 라벨링 (택 1)
#   SAM 클릭 자동 윤곽 방식
python src/pipeline/step2_sam_labeler.py          # vit_b (빠름)
python src/pipeline/step2_sam_labeler.py --heavy  # vit_h (정확)
#   SAM + 수동 4-코너 방식
python src/pipeline/step2_obb_labeler.py
python src/pipeline/step2_obb_labeler.py --heavy

# Step 3: YOLO11-OBB 학습 (Intel XPU 자동 감지)
python src/pipeline/step3_train_yolo_obb.py
```

#### 라벨링 조작법 (step2_obb_labeler.py)

| 키 | 동작 |
|----|------|
| 마우스 좌클릭 (SAM 모드) | 소켓 클릭 → 윤곽 자동 탐지 + 회전박스 계산 |
| 마우스 좌클릭 4번 (수동 모드) | 코너 4점 순서대로 클릭 → OBB 완성 |
| `1` | 현재 박스를 **8pin** (클래스 0)으로 저장 |
| `2` | 현재 박스를 **12pin** (클래스 1)으로 저장 |
| `u` | 마지막 라벨 취소 (undo) |
| `r` | 현재 작업 리셋 |
| `m` | SAM 모드 ↔ 수동 모드 전환 |
| `t` | 현재 이미지 저장 후 다음 이미지 |
| `s` | 현재 이미지 스킵 |
| `q` | 저장 후 종료 |

#### 학습 파라미터 (step3_train_yolo_obb.py)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `epochs` | 20 | 최대 에폭 (early stopping 적용) |
| `imgsz` | 640 | 입력 이미지 크기 |
| `batch` | 8 | 배치 크기 |
| `patience` | 30 | 조기 종료 대기 에폭 |
| `degrees` | 180 | 회전 증강 (OBB 각도 학습 핵심) |
| `device` | XPU / CPU | Intel XPU 자동 감지, 없으면 CPU |
| `amp` | False | Intel XPU AMP CUDA 검사 우회 |

학습 결과: `runs/detect/socket_detector_obb/weights/best.pt`

### 2. 3D 탐지 & 빈픽킹

```bash
# 실시간 3D 탐지 (시각화)
python src/detection/binpicking_3d.py

# 빈픽킹 모드 (시뮬레이션)
python src/detection/binpicking_3d.py --pick

# 빈픽킹 모드 (실제 Indy7 연결)
python src/detection/binpicking_3d.py --pick --robot

# 카메라 없이 단일 이미지 테스트
python src/detection/binpicking_3d.py --test
```

#### 실시간 조작키

| 키 | 동작 |
|----|------|
| `q` | 종료 |
| `s` | 현재 프레임 결과 저장 |
| `d` | 탐지 정보 콘솔 출력 |
| `p` | 픽킹 1회 실행 (`--pick` 모드) |

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

| 항목 | 버전 |
|------|------|
| Python | 3.12 |
| OS | Linux (Ubuntu) |
| 가속기 | Intel XPU (Arc / Iris Xe) |
| torch | 2.8.0+xpu |
| intel_extension_for_pytorch | 2.8.10+xpu |
| ultralytics | 8.4.23 |
| pyrealsense2 | 2.56.5 |
| opencv-python | 4.13.0 |
| numpy | 2.4.3 |
| scipy | 1.17.1 |
| segment_anything | (GitHub 최신) |
| 로봇 | Neuromeka Indy7 (IndyDCP) |
