"""
Step 3: SAM 클릭 라벨링 도구

사용법:
    python step3_sam_labeler.py

조작법:
    - 마우스 왼쪽 클릭: 소켓 위를 클릭하면 SAM이 자동으로 윤곽을 잡음
    - '1' 키: 현재 마스크를 "8pin" (클래스 0)으로 저장
    - '2' 키: 현재 마스크를 "12pin" (클래스 1)으로 저장
    - 'r' 키: 현재 마스크 리셋 (다시 클릭)
    - 'n' 키: 다음 이미지로 넘어감
    - 'q' 키: 종료

원리:
    SAM (Segment Anything Model)은 Meta가 만든 세그멘테이션 AI입니다.
    이미지 위의 한 점(point)을 찍어주면 그 점이 속한 물체의 윤곽을 자동으로 찾습니다.

    사람이 할 일:
    1. 소켓 위를 클릭 (1초)
    2. SAM이 윤곽을 자동으로 잡아줌
    3. '1' 또는 '2'로 클래스 지정 (1초)

    → 한 장에 소켓 5개 있으면 약 10초면 끝!
    → 수동으로 바운딩 박스 그리는 것보다 5배 이상 빠름

출력:
    YOLO 형식 라벨 파일 (.txt)
    각 줄: class_id x_center y_center width height (정규화 좌표)
"""

import numpy as np
import cv2
import os
import glob
import sys
import torch
from segment_anything import sam_model_registry, SamPredictor

# ===== 설정 =====
# 기본: vit_b (빠름), --heavy 옵션: vit_h (정확하지만 느림)
if "--heavy" in sys.argv:
    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
    SAM_MODEL_TYPE = "vit_h"
else:
    SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
    SAM_MODEL_TYPE = "vit_b"
INPUT_DIR = "cleaned_images"                # step2에서 만든 배경 제거 이미지
OUTPUT_IMAGE_DIR = "dataset/images/train"
OUTPUT_LABEL_DIR = "dataset/labels/train"

CLASS_NAMES = {0: "8pin_socket", 1: "12pin_socket"}

# ===== 전역 변수 (마우스 콜백용) =====
click_point = None
current_mask = None
current_image = None
predictor = None


def mouse_callback(event, x, y, flags, param):
    """마우스 클릭 시 SAM으로 세그멘테이션 실행"""
    global click_point, current_mask

    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)

        # SAM에 클릭 좌표 전달 → 마스크 예측
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # 1 = 전경(소켓)

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True  # 3개의 후보 마스크 생성
        )

        # 가장 점수 높은 마스크 선택
        best_idx = np.argmax(scores)
        current_mask = masks[best_idx]

        print(f"  클릭 ({x}, {y}) → 마스크 점수: {scores[best_idx]:.3f}")


def mask_to_yolo_bbox(mask, img_h, img_w):
    """
    바이너리 마스크 → YOLO 바운딩 박스 (정규화 좌표)

    YOLO 형식: x_center y_center width height (0~1 사이 값)

    예시:
        이미지 크기 640x480, 소켓이 (100,200)~(200,300)에 있으면
        x_center = 150/640 = 0.234
        y_center = 250/480 = 0.521
        width = 100/640 = 0.156
        height = 100/480 = 0.208
    """
    # 마스크에서 흰 픽셀의 좌표 찾기
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # YOLO 정규화 좌표로 변환
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h

    return x_center, y_center, width, height


def main():
    global current_mask, current_image, predictor, click_point

    # ===== 폴더 생성 =====
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    # ===== SAM 모델 로드 =====
    print("SAM 모델 로딩 중... (처음 한 번만 오래 걸림)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device)
    predictor = SamPredictor(sam)
    print(f"SAM 로드 완료 (device: {device})")

    # ===== 이미지 목록 =====
    image_files = sorted(glob.glob(f"{INPUT_DIR}/*.png"))
    # _mask.png 파일은 제외
    image_files = [f for f in image_files if "_mask" not in f]
    print(f"라벨링할 이미지: {len(image_files)}장\n")

    cv2.namedWindow("SAM Labeler")
    cv2.setMouseCallback("SAM Labeler", mouse_callback)

    # 이 이미지에서 지금까지 저장한 라벨들 (한 이미지에 소켓 여러 개)
    current_labels = []

    for idx, img_path in enumerate(image_files):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[{idx + 1}/{len(image_files)}] {basename}")
        print("  소켓 위를 클릭 → '1'(8핀) 또는 '2'(12핀) → 반복 → 'n'(다음)")

        current_image = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

        # SAM에 이미지 설정 (임베딩 계산, 이미지당 1번)
        predictor.set_image(img_rgb)

        current_labels = []
        current_mask = None

        while True:
            # ===== 화면 그리기 =====
            display = current_image.copy()

            # 현재 마스크가 있으면 반투명 오버레이
            if current_mask is not None:
                overlay = display.copy()
                overlay[current_mask] = [0, 255, 0]  # 초록색
                display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

            # 클릭 포인트 표시
            if click_point is not None:
                cv2.circle(display, click_point, 5, (0, 0, 255), -1)

            # 이미 저장된 바운딩 박스 표시
            h, w = current_image.shape[:2]
            for cls_id, xc, yc, bw, bh in current_labels:
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                color = (255, 0, 0) if cls_id == 0 else (0, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, CLASS_NAMES[cls_id], (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 안내 텍스트
            info = f"[{idx+1}/{len(image_files)}] Labels: {len(current_labels)} | 1=8pin 2=12pin r=reset n=next q=quit"
            cv2.putText(display, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("SAM Labeler", display)
            key = cv2.waitKey(30) & 0xFF

            # ----- 클래스 지정 + 저장 -----
            if key == ord('1') and current_mask is not None:
                bbox = mask_to_yolo_bbox(current_mask, h, w)
                if bbox:
                    current_labels.append((0, *bbox))  # 0 = 8pin
                    print(f"    → 8핀 소켓 저장 (총 {len(current_labels)}개)")
                current_mask = None
                click_point = None

            elif key == ord('2') and current_mask is not None:
                bbox = mask_to_yolo_bbox(current_mask, h, w)
                if bbox:
                    current_labels.append((1, *bbox))  # 1 = 12pin
                    print(f"    → 12핀 소켓 저장 (총 {len(current_labels)}개)")
                current_mask = None
                click_point = None

            # ----- 리셋 -----
            elif key == ord('r'):
                current_mask = None
                click_point = None
                print("    마스크 리셋")

            # ----- 다음 이미지 -----
            elif key == ord('n'):
                if current_labels:
                    # 이미지 복사
                    cv2.imwrite(f"{OUTPUT_IMAGE_DIR}/{basename}.png", current_image)

                    # 라벨 파일 저장 (YOLO 형식)
                    label_path = f"{OUTPUT_LABEL_DIR}/{basename}.txt"
                    with open(label_path, 'w') as f:
                        for cls_id, xc, yc, bw, bh in current_labels:
                            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

                    print(f"  저장 완료: {len(current_labels)}개 라벨 → {label_path}")
                else:
                    print("  라벨 없음, 스킵")
                break

            # ----- 종료 -----
            elif key == ord('q'):
                cv2.destroyAllWindows()
                print(f"\n라벨링 종료. 총 {idx}장 완료.")
                return

    cv2.destroyAllWindows()
    print(f"\n전체 라벨링 완료!")
    print(f"  이미지: {OUTPUT_IMAGE_DIR}/")
    print(f"  라벨:   {OUTPUT_LABEL_DIR}/")


if __name__ == "__main__":
    main()
