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
│   ├── paths.py                    # 경로 설정
│   ├── robot_config.py             # Indy7 로봇 설정 (IP, 속도, 캘리브레이션)
│   └── calibration_data/           # Hand-eye 캘리브레이션 데이터 (gitignore)
│
├── src/                            # 소스 코드
│   ├── detection/                  # 탐지 & 빈픽킹
│   │   ├── binpicking_3d.py        # 3D 탐지 + 빈픽킹 메인 시스템
│   │   └── test_detect.py          # 실시간 탐지 테스트
│   │
│   ├── pipeline/                   # OBB 데이터 수집 & 학습 파이프라인
│   │   ├── step1_capture.py        # D435 RGB+Depth 촬영
│   │   ├── step2_obb_labeler.py    # 통합 OBB 라벨링 (SAM + 수동 박스 모드 지원)
│   │   └── step3_train_yolo_obb.py # YOLO11-OBB 최신 학습 스크립트
│   │
│   ├── robot/                      # 로봇 제어
│   │   └── indy_controller.py      # Indy7 래퍼 (픽/플레이스/진공)
│   │
│   └── calibration/                # 캘리브레이션
│       └── hand_eye_calibration.py # Hand-Eye 캘리브레이션 (ArUco)
│
├── robot/                          # Indy7 SDK & 예제
├── models/                         # OBB 모델 가중치 (gitignore)
├── data/                           # 이미지 데이터 (gitignore)
├── runs/                           # YOLO 학습 결과
└── dataset.yaml                    # YOLO OBB 데이터셋 설정 (8pin, 12pin)
```

## 다운로드 및 설치 (Installation)

본 프로젝트는 하드웨어 연산 장치(GPU / XPU)에 따라 브랜치(Branch)가 나누어져 있습니다. 다른 컴퓨터에서 가져다 쓰실 땐 본인의 하드웨어 환경에 맞춰 아래 명령어 중 하나를 골라 다운로드(Clone)해 주세요.

```bash
# 옵션 A. NVIDIA GPU 기반 환경 (CUDA 최적화 버젼)
git clone -b gpu/yolo11-obb-RANSAC https://github.com/jeong-bin-98/intel5_project.git

# 옵션 B. Intel XPU / 일반 CPU 환경 (코어 울트라, ARC 등 NPU/XPU 최적화 버젼)
git clone -b xpu/yolo11-obb-RANSAC https://github.com/jeong-bin-98/intel5_project.git

# 다운로드 완료 후 프로젝트 폴더로 진입
cd intel5_project
```

## 초기 설정 (Setup)

코드를 다운로드하신 뒤, 해당 컴퓨터에서 프로젝트를 구동하기 위한 초기 세팅 방법입니다.

### 1. 패키지 라이브러리 설치
동봉된 `requirements.txt`를 사용하여 동일한 파이썬 환경을 구성합니다.

```bash
# 가상환경 생성 (권장)
python -m venv intel5
.\intel5\Scripts\activate  # Windows 환경

# 의존성 설치
pip install -r requirements.txt
```

### 2. 베이스 모델 가중치 오프라인 다운로드 
인터넷이 불안정한 산업 현장에서 파이프라인이 다운로드 지연 없이 곧바로 실행될 수 있도록 기초 모델 가중치들을 미리 다운로드해 둡니다.

```bash
cd models

# 1) 학습용 YOLO11-OBB 기초 가중치 다운로드
python -c "from ultralytics import YOLO; YOLO('yolo11n-obb.pt')"

# 2) 라벨링용 SAM (Segment Anything) 가중치 다운로드 (Windows PowerShell 전용)
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "sam_vit_b_01ec64.pth"
# (고해상도 모델이 필요한 경우 아래 명령어 사용)
# Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -OutFile "sam_vit_h_4b8939.pth"

cd ..
```

## 실행 방법

### 1. 데이터 수집 & 학습

```bash
# 이미지 촬영
python src/pipeline/step1_capture.py

# 통합 OBB 라벨링 도구 실행 (클릭 or 수동 4포인트)
python src/pipeline/step2_obb_labeler.py

# YOLO11-OBB 학습
python src/pipeline/step3_train_yolo_obb.py
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
- YOLO11 OBB (ultralytics 최신)
- Neuromeka Indy7 (IndyDCP)
- OpenCV, NumPy, SciPy, SAM
