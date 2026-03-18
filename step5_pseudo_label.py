"""
Step 5: Pseudo-labeling (자동 라벨링)

사용법:
    python step5_pseudo_label.py                    # 자동 라벨링 실행
    python step5_pseudo_label.py --review           # 결과 시각적 검수

설명:
    Step 4에서 학습한 모델이 새 이미지를 보고 자동으로 라벨을 만듭니다.

    Pseudo-labeling 이란?
    ─────────────────────
    "가짜 라벨링"이라는 뜻이지만, 실제로는 매우 효과적인 기법입니다.

    1. 모델이 새 이미지를 추론
    2. confidence(확신도) 80% 이상인 결과만 "라벨"로 채택
    3. 이 자동 라벨로 다시 학습 → 더 좋은 모델
    4. 반복

    왜 80%?
    - 너무 높으면 (95%): 라벨이 너무 적게 생성됨
    - 너무 낮으면 (50%): 틀린 라벨이 섞여서 모델이 오염됨
    - 80%가 정확도와 수량의 균형점

    주의:
    자동 라벨링 후 반드시 --review로 검수하세요!
    틀린 라벨 1~2개는 괜찮지만, 10% 이상 틀리면 threshold를 올려야 합니다.
"""

import os
import glob
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

# ===== 설정 =====
MODEL_PATH = "runs/detect/socket_detector2/weights/best.pt"   # Step 4 학습 결과
NEW_IMAGES_DIR = "new_images"               # 라벨 없는 새 이미지들
CONFIDENCE_THRESHOLD = 0.10                 # 이 이상만 자동 라벨로 채택
OUTPUT_IMAGE_DIR = "dataset/images/train"   # 기존 학습 데이터에 합침
OUTPUT_LABEL_DIR = "dataset/labels/train"

CLASS_NAMES = {0: "8pin_socket", 1: "12pin_socket"}


def auto_label():
    """새 이미지에 대해 자동 라벨링을 수행합니다."""

    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일이 없습니다: {MODEL_PATH}")
        print("Step 4를 먼저 실행하세요.")
        return

    # 모델 로드
    model = YOLO(MODEL_PATH)
    print(f"모델 로드: {MODEL_PATH}")
    print(f"Confidence 임계값: {CONFIDENCE_THRESHOLD}")

    # 새 이미지 목록
    new_images = sorted(glob.glob(f"{NEW_IMAGES_DIR}/*.png"))
    if not new_images:
        print(f"\n{NEW_IMAGES_DIR}/ 에 이미지가 없습니다.")
        print("D435로 새 이미지를 촬영해서 넣어주세요.")
        return

    print(f"자동 라벨링할 이미지: {len(new_images)}장\n")

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    stats = {"total": 0, "labeled": 0, "skipped": 0, "objects": 0}

    for img_path in new_images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        stats["total"] += 1

        # ===== 추론 =====
        results = model(img_path, verbose=False)[0]

        # confidence 필터링
        labels = []
        boxes = results.boxes

        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])

            if conf >= CONFIDENCE_THRESHOLD:
                # xyxy → YOLO 정규화 좌표 변환
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                img_h, img_w = results.orig_shape

                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                labels.append((cls_id, x_center, y_center, width, height, conf))

        # ===== 저장 =====
        if labels:
            # 이미지 복사
            shutil.copy2(img_path, f"{OUTPUT_IMAGE_DIR}/{basename}.png")

            # 라벨 저장
            with open(f"{OUTPUT_LABEL_DIR}/{basename}.txt", 'w') as f:
                for cls_id, xc, yc, w, h, conf in labels:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            class_summary = ", ".join(
                f"{CLASS_NAMES[int(l[0])]}({l[5]:.0%})" for l in labels
            )
            print(f"  [✓] {basename}: {len(labels)}개 → {class_summary}")
            stats["labeled"] += 1
            stats["objects"] += len(labels)
        else:
            print(f"  [✗] {basename}: 탐지 없음 또는 confidence 미달")
            stats["skipped"] += 1

    # ===== 통계 =====
    print(f"\n{'=' * 50}")
    print(f"자동 라벨링 완료!")
    print(f"  전체 이미지:   {stats['total']}장")
    print(f"  라벨 생성:     {stats['labeled']}장 ({stats['objects']}개 객체)")
    print(f"  스킵 (미달):   {stats['skipped']}장")
    print(f"{'=' * 50}")
    print(f"\n다음 단계:")
    print(f"  1. python step5_pseudo_label.py --review  (검수)")
    print(f"  2. 틀린 라벨 삭제")
    print(f"  3. python step4_train_yolo.py  (2차 학습)")


def review_labels():
    """자동 라벨링 결과를 시각적으로 검수합니다."""

    image_files = sorted(glob.glob(f"{OUTPUT_IMAGE_DIR}/*.png"))
    print(f"검수할 이미지: {len(image_files)}장")
    print("  'd' = 이 이미지의 라벨 삭제  |  아무키 = 다음  |  'q' = 종료\n")

    deleted = 0
    for idx, img_path in enumerate(image_files):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = f"{OUTPUT_LABEL_DIR}/{basename}.txt"

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # 라벨 읽기 + 그리기
        with open(label_path) as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            color = (255, 0, 0) if cls_id == 0 else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, CLASS_NAMES[cls_id], (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(img, f"[{idx+1}/{len(image_files)}] {basename} | 'd'=삭제 'q'=종료",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Review", img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('d'):
            os.remove(label_path)
            os.remove(img_path)
            deleted += 1
            print(f"  [삭제] {basename}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"\n검수 완료. {deleted}장 삭제됨.")


if __name__ == "__main__":
    import sys
    if "--review" in sys.argv:
        review_labels()
    else:
        auto_label()
