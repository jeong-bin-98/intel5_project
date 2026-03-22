"""
Step 3-1: SAM 자동 라벨링 도구 (클릭 없이!)

사용법:
    python step3_1_auto_labeler.py
    python step3_1_auto_labeler.py --heavy    # vit_h 모델 사용 (더 정확, 더 느림)

조작법:
    - '1' 키: 현재 표시된 영역을 "8pin" (클래스 0)으로 저장
    - '2' 키: 현재 표시된 영역을 "12pin" (클래스 1)으로 저장
    - 'd' 키: 현재 영역 스킵 (소켓이 아닌 것)
    - 'n' 키: 이 이미지 끝내고 다음 이미지로
    - 'q' 키: 종료

원리:
    기존 step3은 소켓 위를 일일이 클릭해야 했지만,
    이 도구는 SAM의 SamAutomaticMaskGenerator를 사용해
    이미지 속 모든 물체를 자동으로 찾아냅니다.

    사람이 할 일:
    1. SAM이 자동으로 찾은 영역을 확인
    2. 소켓이면 '1' 또는 '2'로 클래스 지정
    3. 소켓이 아니면 'd'로 스킵

    → 클릭 0번! 키보드만 탁탁 누르면 끝!

출력:
    YOLO 형식 라벨 파일 (.txt)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.paths import SAM_VIT_H, SAM_VIT_B, CLEANED_IMAGES_DIR, DATASET_IMAGES_TRAIN, DATASET_LABELS_TRAIN

import numpy as np
import cv2
import glob
import torch
from threading import Thread
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ===== 설정 =====
# 기본: vit_b (빠름), --heavy 옵션: vit_h (정확하지만 느림)
if "--heavy" in sys.argv:
    SAM_CHECKPOINT = SAM_VIT_H
    SAM_MODEL_TYPE = "vit_h"
else:
    SAM_CHECKPOINT = SAM_VIT_B
    SAM_MODEL_TYPE = "vit_b"
INPUT_DIR = CLEANED_IMAGES_DIR
OUTPUT_IMAGE_DIR = DATASET_IMAGES_TRAIN
OUTPUT_LABEL_DIR = DATASET_LABELS_TRAIN

# 마스크 필터 설정
MIN_AREA_RATIO = 0.001    # 이미지 대비 최소 면적 비율 (너무 작은 건 노이즈)
MAX_AREA_RATIO = 0.5      # 이미지 대비 최대 면적 비율 (너무 큰 건 배경)
MIN_AREA_PX = 500         # 최소 픽셀 수

CLASS_NAMES = {0: "8pin_socket", 1: "12pin_socket"}


def mask_to_yolo_bbox(mask, img_h, img_w):
    """바이너리 마스크 → YOLO 바운딩 박스 (정규화 좌표)"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h

    return x_center, y_center, width, height


def main():
    # ===== 폴더 생성 =====
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    # ===== SAM 모델 로드 =====
    print(f"SAM 모델 로딩 중... (모델: {SAM_MODEL_TYPE})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device)

    # 자동 마스크 생성기 설정
    pts = 16 if SAM_MODEL_TYPE == "vit_b" else 32
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=pts,            # vit_b: 16 (빠름), vit_h: 32 (정확)
        pred_iou_thresh=0.86,          # IoU 임계값
        stability_score_thresh=0.92,   # 안정성 점수 임계값
        min_mask_region_area=MIN_AREA_PX,  # 최소 마스크 영역
    )
    print(f"SAM 로드 완료 (device: {device}, points_per_side: {pts})")

    # ===== 이미지 목록 =====
    image_files = sorted(glob.glob(f"{INPUT_DIR}/*.png"))
    image_files = [f for f in image_files if "_mask" not in f]
    print(f"라벨링할 이미지: {len(image_files)}장")
    print("=" * 50)

    cv2.namedWindow("Auto Labeler")
    total_labeled = 0

    # ===== 백그라운드 마스크 미리 계산 함수 =====
    prefetch_result = {}  # {idx: filtered_masks}

    def prefetch_masks(img_idx, img_file):
        """백그라운드 스레드에서 다음 이미지의 마스크를 미리 계산"""
        try:
            img = cv2.imread(img_file)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            total = h * w
            raw_masks = mask_generator.generate(rgb)
            filt = []
            for m in raw_masks:
                area = m["area"]
                ratio = area / total
                if MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO and area >= MIN_AREA_PX:
                    filt.append(m)
            filt.sort(key=lambda x: x["area"], reverse=True)
            prefetch_result[img_idx] = filt
        except Exception as e:
            print(f"  [prefetch 오류] {e}")
            prefetch_result[img_idx] = []

    # 첫 번째 이미지는 미리 계산할 수 없으니 직접 계산
    for idx, img_path in enumerate(image_files):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[{idx + 1}/{len(image_files)}] {basename}")

        image_bgr = cv2.imread(img_path)
        img_h, img_w = image_bgr.shape[:2]

        # ===== 마스크 가져오기 (미리 계산됐으면 그거 쓰고, 아니면 직접 계산) =====
        if idx in prefetch_result:
            filtered = prefetch_result.pop(idx)
            print(f"  (미리 계산됨) 필터 후 {len(filtered)}개")
        else:
            print("  SAM 자동 세그멘테이션 중...")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            total_area = img_h * img_w
            masks = mask_generator.generate(image_rgb)
            filtered = []
            for m in masks:
                area = m["area"]
                ratio = area / total_area
                if MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO and area >= MIN_AREA_PX:
                    filtered.append(m)
            filtered.sort(key=lambda x: x["area"], reverse=True)
            print(f"  총 {len(masks)}개 검출 → 필터 후 {len(filtered)}개")

        # 다음 이미지 백그라운드 미리 계산 시작
        if idx + 1 < len(image_files):
            t = Thread(target=prefetch_masks, args=(idx + 1, image_files[idx + 1]))
            t.start()

        if not filtered:
            print("  유효한 영역 없음 → 스킵")
            continue

        # ===== 각 마스크를 하나씩 보여주며 클래스 지정 =====
        current_labels = []

        # 이미 저장된 바운딩 박스들을 보여주기 위한 리스트
        for m_idx, mask_data in enumerate(filtered):
            mask = mask_data["segmentation"]  # (H, W) boolean

            # 현재 마스크 오버레이
            display = image_bgr.copy()

            # 이미 저장된 바운딩 박스 표시
            for cls_id, xc, yc, bw, bh in current_labels:
                x1 = int((xc - bw / 2) * img_w)
                y1 = int((yc - bh / 2) * img_h)
                x2 = int((xc + bw / 2) * img_w)
                y2 = int((yc + bh / 2) * img_h)
                color = (255, 0, 0) if cls_id == 0 else (0, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, CLASS_NAMES[cls_id], (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 현재 마스크 초록색 오버레이
            overlay = display.copy()
            overlay[mask] = [0, 255, 0]
            display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

            # 현재 마스크의 바운딩 박스 표시
            bbox = mask_to_yolo_bbox(mask, img_h, img_w)
            if bbox:
                xc, yc, bw, bh = bbox
                x1 = int((xc - bw / 2) * img_w)
                y1 = int((yc - bh / 2) * img_h)
                x2 = int((xc + bw / 2) * img_w)
                y2 = int((yc + bh / 2) * img_h)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 안내 텍스트
            area_pct = mask_data["area"] / total_area * 100
            info1 = f"[Img {idx+1}/{len(image_files)}] Mask {m_idx+1}/{len(filtered)} ({area_pct:.1f}%)"
            info2 = f"1=8pin  2=12pin  d=skip  n=next_img  q=quit"
            cv2.putText(display, info1, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, info2, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display, f"Saved: {len(current_labels)}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Auto Labeler", display)

            # ===== 키 입력 대기 =====
            while True:
                key = cv2.waitKey(0) & 0xFF

                if key == ord('1') and bbox:
                    current_labels.append((0, *bbox))
                    print(f"    → 8핀 소켓 (총 {len(current_labels)}개)")
                    break

                elif key == ord('2') and bbox:
                    current_labels.append((1, *bbox))
                    print(f"    → 12핀 소켓 (총 {len(current_labels)}개)")
                    break

                elif key == ord('d'):
                    print(f"    → 스킵")
                    break

                elif key == ord('n'):
                    break

                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    print(f"\n라벨링 종료. 총 {total_labeled}장 완료.")
                    return

            if key == ord('n') or key == ord('q'):
                break

        # ===== 이 이미지의 라벨 저장 =====
        if current_labels:
            cv2.imwrite(f"{OUTPUT_IMAGE_DIR}/{basename}.png", image_bgr)
            label_path = f"{OUTPUT_LABEL_DIR}/{basename}.txt"
            with open(label_path, 'w') as f:
                for cls_id, xc, yc, bw, bh in current_labels:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            total_labeled += 1
            print(f"  저장 완료: {len(current_labels)}개 라벨 → {label_path}")
        else:
            print("  라벨 없음, 스킵")

    cv2.destroyAllWindows()
    print(f"\n전체 라벨링 완료!")
    print(f"  이미지: {OUTPUT_IMAGE_DIR}/")
    print(f"  라벨:   {OUTPUT_LABEL_DIR}/")


if __name__ == "__main__":
    main()
