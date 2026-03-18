"""
Step 4: YOLOv8 1차 학습

사용법:
    python step4_train_yolo.py

설명:
    Step 3에서 만든 라벨 데이터(50~100장)로 YOLOv8 모델을 학습합니다.

    YOLOv8n(nano)은 가장 작은 모델이라:
    - GPU: 약 5~10분
    - CPU: 약 20~30분
    으로 학습이 끝납니다.

    학습 완료 후 runs/detect/train/weights/best.pt 파일이 생기며,
    이것이 Step 5에서 자동 라벨링에 사용할 모델입니다.

사전 준비:
    1. dataset/images/train/ 에 학습 이미지
    2. dataset/labels/train/ 에 YOLO 라벨 (.txt)
    3. dataset/images/val/ 에 검증 이미지 (train의 20% 정도를 옮기세요)
    4. dataset/labels/val/ 에 검증 라벨
    5. dataset.yaml 파일
"""

import os
import shutil
import random
from ultralytics import YOLO


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
    # ===== Train/Val 자동 분리 =====
    split_train_val("dataset/images/train", "dataset/labels/train")

    # ===== 학습 데이터 확인 =====
    train_images = os.listdir("dataset/images/train")
    train_labels = os.listdir("dataset/labels/train")
    print(f"\n학습 이미지: {len(train_images)}장")
    print(f"학습 라벨:   {len(train_labels)}개")

    if len(train_labels) < 10:
        print("⚠️  라벨이 10개 미만입니다. Step 3에서 더 라벨링하세요.")
        return

    # ===== YOLOv8 학습 =====
    print("\n" + "=" * 50)
    print("YOLOv8n 학습 시작")
    print("=" * 50)

    model = YOLO("yolov8n.pt")  # 사전학습된 nano 모델 다운로드 + 로드

    results = model.train(
        data="dataset.yaml",    # 데이터셋 설정
        epochs=100,             # 최대 100 에폭 (early stopping으로 자동 종료됨)
        imgsz=640,              # 이미지 크기
        batch=8,               # 배치 크기 (GPU 메모리에 맞게 조절)
        patience=30,            # 20 에폭 동안 개선 없으면 조기 종료
        device="0",             # GPU 사용 ("cpu"로 바꾸면 CPU 사용)
        workers=4,              # 데이터 로딩 워커 수
        name="socket_detector", # 결과 저장 폴더 이름

        # === 데이터 증강 (적은 데이터를 뻥튀기) ===
        augment=True,
        hsv_h=0.015,            # 색상 변동
        hsv_s=0.7,              # 채도 변동
        hsv_v=0.4,              # 밝기 변동
        degrees=180,            # 회전 (빈픽킹은 360도 무작위이므로 크게)
        translate=0.1,          # 평행 이동
        scale=0.3,              # 크기 변동
        flipud=0.5,             # 상하 반전
        fliplr=0.5,             # 좌우 반전
        mosaic=0.5,             # 모자이크 증강 (4장을 하나로 합침)
        mixup=0.0,              # 믹스업 증강 (2장을 겹침)
    )

    # ===== 결과 확인 =====
    print("\n" + "=" * 50)
    print("학습 완료!")
    print("=" * 50)
    print(f"최적 모델: runs/detect/socket_detector/weights/best.pt")
    print(f"학습 로그: runs/detect/socket_detector/results.csv")
    print(f"혼동 행렬: runs/detect/socket_detector/confusion_matrix.png")
    print(f"\n이 best.pt를 Step 5 (자동 라벨링)에 사용합니다.")


if __name__ == "__main__":
    train()
