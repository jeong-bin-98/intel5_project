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
import time
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


def _delete_cache_files(labels_dir):
    """labels 폴더 내 stale .cache 파일을 삭제합니다."""
    parent = os.path.dirname(labels_dir)
    for f in os.listdir(parent) if os.path.isdir(parent) else []:
        if f.endswith(".cache"):
            cache_path = os.path.join(parent, f)
            os.remove(cache_path)
            print(f"  stale cache 삭제: {cache_path}")


def split_train_val(train_img_dir, train_lbl_dir, val_ratio=0.2):
    """
    train 폴더에서 일부를 val 폴더로 자동 분리합니다.
    라벨이 있는 이미지를 우선 분배하여 val에도 라벨이 반드시 포함되도록 합니다.

    이미 val에 라벨이 있으면 스킵합니다.
    val에 이미지만 있고 라벨이 없으면 재분배를 수행합니다.

    Args:
        train_img_dir: 학습 이미지 폴더
        train_lbl_dir: 학습 라벨 폴더
        val_ratio: 검증용 비율 (기본 20%)
    """
    val_img_dir = train_img_dir.replace("/train", "/val")
    val_lbl_dir = train_lbl_dir.replace("/train", "/val")

    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # val에 라벨이 이미 있으면 스킵
    val_labels = [f for f in os.listdir(val_lbl_dir) if f.endswith('.txt')]
    if len(val_labels) > 0:
        print(f"val 라벨 {len(val_labels)}개 확인 → 분리 스킵")
        return

    # val에 이미지만 있고 라벨이 없으면 → train으로 되돌린 후 재분배
    val_existing_imgs = [f for f in os.listdir(val_img_dir) if f.endswith('.png')]
    if len(val_existing_imgs) > 0:
        print(f"val에 이미지 {len(val_existing_imgs)}장 있지만 라벨 0개 → 재분배 수행")
        for img_name in val_existing_imgs:
            src = os.path.join(val_img_dir, img_name)
            dst = os.path.join(train_img_dir, img_name)
            if not os.path.exists(dst):
                shutil.move(src, dst)
            else:
                os.remove(src)

    # stale cache 삭제
    _delete_cache_files(train_lbl_dir)

    # 라벨 있는 이미지 / 없는 이미지 분리
    all_images = [f for f in os.listdir(train_img_dir) if f.endswith('.png')]
    labeled = []
    unlabeled = []
    for img in all_images:
        lbl = img.replace('.png', '.txt')
        if os.path.exists(os.path.join(train_lbl_dir, lbl)):
            labeled.append(img)
        else:
            unlabeled.append(img)

    random.shuffle(labeled)
    random.shuffle(unlabeled)

    print(f"전체 이미지: {len(all_images)}장 (라벨 있음: {len(labeled)}, 없음: {len(unlabeled)})")

    # 라벨 있는 이미지에서 val_ratio만큼 val로 분배 (최소 1장)
    val_labeled_count = max(1, int(len(labeled) * val_ratio))
    # 라벨 없는 이미지에서도 val_ratio만큼 val로 분배
    val_unlabeled_count = int(len(unlabeled) * val_ratio)

    val_images = labeled[:val_labeled_count] + unlabeled[:val_unlabeled_count]

    for img_name in val_images:
        lbl_name = img_name.replace('.png', '.txt')

        # 이미지 이동
        shutil.move(
            os.path.join(train_img_dir, img_name),
            os.path.join(val_img_dir, img_name)
        )
        # 라벨 이동 (있는 경우)
        lbl_src = os.path.join(train_lbl_dir, lbl_name)
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, os.path.join(val_lbl_dir, lbl_name))

    train_remaining = len(all_images) - len(val_images)
    val_lbl_final = len([f for f in os.listdir(val_lbl_dir) if f.endswith('.txt')])
    print(f"Train/Val 분리 완료: train {train_remaining}장, val {len(val_images)}장 (val 라벨: {val_lbl_final}개)")


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
    print("YOLO11s-OBB 학습 시작 (빈피킹 최적화)")
    print("=" * 50)

    model = YOLO("yolo11s-obb.pt")

    # === 에폭별 남은 시간 표시 콜백 ===
    train_start = time.time()

    def _on_train_epoch_end(trainer):
        epoch = trainer.epoch + 1
        total_epochs = trainer.epochs
        elapsed = time.time() - train_start
        avg_per_epoch = elapsed / epoch
        remaining = avg_per_epoch * (total_epochs - epoch)

        def _fmt(s):
            m, s = divmod(int(s), 60)
            h, m = divmod(m, 60)
            return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

        print(f"  [{epoch}/{total_epochs}] 경과: {_fmt(elapsed)} | "
              f"에폭당: {_fmt(avg_per_epoch)} | 남은 시간: {_fmt(remaining)}")

    model.add_callback("on_train_epoch_end", _on_train_epoch_end)

    results = model.train(
        task="obb",             # OBB(회전 바운딩 박스) 모드 필수 설정
        data=DATASET_OBB_YAML,  # OBB 데이터셋 설정
        project=os.path.join(RUNS_DIR, "detect"),  # 결과 저장 위치
        device=DEVICE,          # Intel XPU 또는 CPU (자동 선택)
        amp=False,              # Intel XPU는 AMP CUDA 검사 우회 필요
        workers=0,              # XPU에서는 0 권장
        name="socket_detector_obb", # 결과 저장 폴더 이름 (OBB 명시)

        # === 학습 스케줄 ===
        epochs=500,             # 최대 300 에폭 (early stopping으로 자동 종료)
        patience=150,            # 50 에폭 동안 개선 없으면 조기 종료 (충분한 수렴 기회)
        imgsz=640,              # 이미지 크기
        batch=16,               # 배치 크기 (Arc A770 16GB VRAM 활용)

        # === 옵티마이저 (소규모 데이터셋 + 정밀 OBB에 AdamW 적합) ===
        optimizer="AdamW",
        lr0=0.001,              # AdamW 초기 학습률
        lrf=0.01,               # 최종 학습률 비율 (lr0 * lrf)
        weight_decay=0.0005,    # 과적합 방지
        warmup_epochs=5,        # 안정적 학습 시작을 위한 웜업
        warmup_momentum=0.8,
        cos_lr=True,            # 코사인 학습률 스케줄 (안정적 수렴)

        # === 손실 가중치 (빈피킹: 위치/각도 정밀도 > 분류) ===
        box=10.0,               # 박스 회귀 가중치 ↑ (OBB 위치+각도 정밀도 강화)
        cls=0.5,                # 분류 가중치 (2클래스라 기본값 충분)
        dfl=1.5,                # 분포 초점 손실 (기본값)

        # === 정규화 ===
        label_smoothing=0.05,   # 과신 방지 (소규모 데이터셋에 효과적)

        # === 데이터 증강 (빈피킹 환경 최적화) ===
        augment=True,
        # 색상/조명: 산업용 조명 환경 변동 대응
        hsv_h=0.015,            # 색상 변동 (플라스틱 색상 일관적 → 소폭)
        hsv_s=0.5,              # 채도 변동 (산업 조명 하 적당히)
        hsv_v=0.4,              # 밝기 변동 (빈 내부 그림자/반사 대응)
        # 기하학적 변환: 빈 피킹 핵심 — 임의 자세 대응
        degrees=180,            # 360° 회전 (OBB 각도 학습의 핵심, 빈 내 임의 방향)
        translate=0.15,         # 평행 이동 (빈 내 다양한 위치)
        scale=0.4,              # 크기 변동 ↑ (빈 내 깊이 차이로 크기 변화 큼)
        shear=2.0,              # 전단 변형 (약간의 시점 변화 시뮬레이션)
        perspective=0.001,      # 원근 변형 (빈 내 깊이에 따른 왜곡)
        # 반전: 빈 피킹에서 상하좌우 무관
        flipud=0.5,             # 상하 반전
        fliplr=0.5,             # 좌우 반전
        # 합성 증강: 다수 객체 겹침 시뮬레이션
        mosaic=1.0,             # 모자이크 ↑ (빈 내 다수 소켓 밀집 환경 학습)
        close_mosaic=20,        # 마지막 20에폭은 모자이크 off (정밀 미세조정)
        mixup=0.1,              # 약간의 믹스업 (겹침/반투명 소켓 대응)
        copy_paste=0.2,         # 복사-붙여넣기 (빈 내 소켓 겹침 시뮬레이션)
        erasing=0.3,            # 랜덤 지우기 (부분 가림/겹침 강건성)
    )

    # ===== 결과 확인 =====
    elapsed = time.time() - train_start
    minutes, seconds = divmod(int(elapsed), 60)
    hours, minutes = divmod(minutes, 60)

    print("\n" + "=" * 50)
    print("학습 완료!")
    print(f"총 학습 시간: {hours}시간 {minutes}분 {seconds}초")
    print("=" * 50)
    print(f"최적 모델: runs/detect/socket_detector_obb/weights/best.pt")
    print(f"학습 로그: runs/detect/socket_detector_obb/results.csv")
    print(f"\n이제 이 OBB 모델을 사용하여 RANSAC 3D 추론을 시작할 수 있습니다!")


if __name__ == "__main__":
    train()
