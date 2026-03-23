"""
Step 3: YOLO11-OBB 학습 (회전 바운딩 박스)

사용법:
    python step3_train_yolo_obb.py

설명:
    Step 2에서 만든 OBB 라벨 데이터로 YOLO11 회전 박스 모델을 학습합니다.

    학습 완료 후 runs/detect/socket_detector_obb/weights/best.pt 파일이 생성됩니다.
    이 모델은 [중심X, 중심Y, 가로, 세로, 회전각도(Yaw)]를 통째로 출력하여
    흡착툴 기반 픽킹에 완벽한 2D 방향성을 제공합니다.

사전 준비:
    1. dataset/images/train/ 에 학습 이미지
    2. dataset/labels/train/ 에 YOLO OBB 라벨 (.txt)
    3. dataset/images/val/ 에 검증 이미지
    4. dataset/labels/val/ 에 검증 라벨
    5. dataset.yaml 파일
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.paths import DATASET_OBB_DIR, DATASET_LABELS_OBB_TRAIN, DATASET_OBB_YAML, RUNS_DIR

import shutil
import random
import torch
from ultralytics import YOLO
from ultralytics.utils import torch_utils
from ultralytics.engine import validator as _validator
from ultralytics.engine import predictor as _predictor

# XPU 장치를 ultralytics select_device가 거부하지 않도록 패치
# validator/trainer가 from ... import select_device로 직접 가져오므로 모든 모듈에 패치 필요
_orig_select_device = torch_utils.select_device

def _patched_select_device(device="", batch=0, newline=False, verbose=True):
    d = str(device) if not hasattr(device, "type") else device.type
    if d.startswith("xpu"):
        return torch.device("xpu")
    return _orig_select_device(device, batch, newline, verbose)

torch_utils.select_device = _patched_select_device
_validator.select_device = _patched_select_device
_predictor.select_device = _patched_select_device

# XPU용 메모리 함수 패치
def _get_memory_xpu(self, fraction=False):
    if self.device.type == "xpu":
        mem = torch.xpu.memory_reserved(self.device)
        total = torch.xpu.get_device_properties(self.device).total_memory
        return mem / total if fraction else mem
    return 0

def _clear_memory_xpu(self, threshold=0.5):
    if self.device.type == "xpu":
        torch.xpu.empty_cache()

if torch.xpu.is_available():
    DEVICE = torch.device("xpu")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  XPU 없음 → CPU로 학습합니다 (느릴 수 있음)")


def split_train_val(train_img_dir, train_lbl_dir, val_ratio=0.2):
    """
    train 폴더에서 일부를 val 폴더로 자동 분리합니다.
    (이미 val에 파일이 있으면 스킵)

    Args:
        train_img_dir: 학습 이미지 폴더
        train_lbl_dir: 학습 라벨 폴더
        val_ratio: 검증용 비율 (기본 20%)
    """
    val_img_dir = train_img_dir.replace("/train", "/val")
    val_lbl_dir = train_lbl_dir.replace("/train", "/val")

    # val 폴더에 이미 파일이 있으면 스킵
    if os.path.exists(val_img_dir) and len(os.listdir(val_img_dir)) > 0:
        print(f"val 폴더에 이미 {len(os.listdir(val_img_dir))}장 있음 → 분리 스킵")
        return

    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # 학습 이미지 목록
    images = [f for f in os.listdir(train_img_dir) if f.endswith('.png')]
    random.shuffle(images)

    val_count = max(1, int(len(images) * val_ratio))
    val_images = images[:val_count]

    for img_name in val_images:
        lbl_name = img_name.replace('.png', '.txt')

        # 이미지 이동
        shutil.move(
            os.path.join(train_img_dir, img_name),
            os.path.join(val_img_dir, img_name)
        )
        # 라벨 이동
        lbl_src = os.path.join(train_lbl_dir, lbl_name)
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, os.path.join(val_lbl_dir, lbl_name))

    print(f"Train/Val 분리 완료: train {len(images) - val_count}장, val {val_count}장")


def train():
    from ultralytics.engine import trainer as _trainer
    _trainer.BaseTrainer._get_memory = _get_memory_xpu
    _trainer.BaseTrainer._clear_memory = _clear_memory_xpu

    # ===== Train/Val 자동 분리 =====
    obb_images_train = os.path.join(DATASET_OBB_DIR, "images", "train")
    split_train_val(obb_images_train, DATASET_LABELS_OBB_TRAIN)

    # ===== 학습 데이터 확인 =====
    train_images = os.listdir(obb_images_train)
    train_labels = os.listdir(DATASET_LABELS_OBB_TRAIN)
    print(f"\n학습 이미지: {len(train_images)}장")
    print(f"학습 라벨:   {len(train_labels)}개")

    if len(train_labels) < 10:
        print("⚠️  라벨이 10개 미만입니다. Step 3에서 더 라벨링하세요.")
        return

    # ===== YOLO11-OBB 학습 =====
    print("\n" + "=" * 50)
    print("YOLO11n-OBB 학습 시작")
    print("=" * 50)

    model = YOLO("yolo11n-obb.pt")  # 사전학습된 YOLO11 OBB (네오) 모델 자동 다운로드

    results = model.train(
        task="obb",             # OBB(회전 바운딩 박스) 모드 필수 설정
        data=DATASET_OBB_YAML,  # OBB 데이터셋 설정
        project=os.path.join(RUNS_DIR, "detect"),  # 결과 저장 위치
        epochs=20,             # 최대 100 에폭 (early stopping으로 자동 종료됨)
        imgsz=640,              # 이미지 크기
        batch=8,                # 배치 크기 (GPU 메모리에 맞게 조절)
        patience=30,            # 30 에폭 동안 개선 없으면 조기 종료
        device=DEVICE,          # Intel XPU 또는 CPU (자동 선택)
        amp=False,              # Intel XPU는 AMP CUDA 검사 우회 필요
        workers=0,              # XPU에서는 0 권장
        name="socket_detector_obb", # 결과 저장 폴더 이름 (OBB 명시)

        # === 데이터 증강 ===
        augment=True,
        hsv_h=0.015,            # 색상 변동
        hsv_s=0.7,              # 채도 변동
        hsv_v=0.4,              # 밝기 변동
        degrees=180,            # 회전 (OBB에서는 각도 학습을 위해 180도 회전 증강이 핵심입니다)
        translate=0.1,          # 평행 이동
        scale=0.3,              # 크기 변동
        flipud=0.5,             # 상하 반전
        fliplr=0.5,             # 좌우 반전
        mosaic=0.5,             # 모자이크 증강
        mixup=0.0,              # 믹스업 증강
    )

    # ===== 결과 확인 =====
    print("\n" + "=" * 50)
    print("학습 완료!")
    print("=" * 50)
    print(f"최적 모델: runs/detect/socket_detector_obb/weights/best.pt")
    print(f"학습 로그: runs/detect/socket_detector_obb/results.csv")
    print(f"\n이제 이 OBB 모델을 사용하여 RANSAC 3D 추론을 시작할 수 있습니다!")


if __name__ == "__main__":
    train()
