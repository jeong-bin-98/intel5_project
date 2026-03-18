"""
학습된 모델로 실시간 탐지 테스트

사용법:
    python test_detect.py                          # D435 카메라 실시간 탐지
    python test_detect.py --image test.png         # 단일 이미지 탐지
    python test_detect.py --dir new_images         # 폴더 내 전체 이미지 탐지

조작법 (카메라 모드):
    - 's' 키: 현재 탐지 결과 스크린샷 저장
    - 'q' 키: 종료

설명:
    Step 4에서 학습한 best.pt 모델로 소켓을 탐지합니다.
    바운딩 박스, 클래스, confidence가 화면에 표시됩니다.
"""

import cv2
import numpy as np
import os
import sys
import glob
from ultralytics import YOLO

# ===== 설정 =====
MODEL_PATH = "runs/detect/socket_detector3/weights/best.pt"
CONFIDENCE = 0.7
CLASS_NAMES = {0: "8pin", 1: "12pin"}
COLORS = {0: (255, 100, 0), 1: (0, 100, 255)}  # 8pin=파랑, 12pin=빨강


def draw_detections(image, results):d
    """탐지 결과를 이미지 위에 그립니다."""
    boxes = results[0].boxes
    count = 0

    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        cls_id = int(boxes.cls[i])
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())

        color = COLORS.get(cls_id, (0, 255, 0))
        label = f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.0%}"

        # 바운딩 박스
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 라벨 배경
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        count += 1

    return image, count


def test_camera():
    """D435 카메라로 실시간 탐지"""
    import pyrealsense2 as rs

    model = YOLO(MODEL_PATH)
    print(f"모델 로드: {MODEL_PATH}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    save_count = 0
    print("실시간 탐지 시작! ('s'=저장, 'q'=종료)")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # YOLO 추론
            results = model(color_image, conf=CONFIDENCE, verbose=False)
            display, count = draw_detections(color_image.copy(), results)

            # depth 컬러맵
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # 탐지 정보 표시
            cv2.putText(display, f"Detected: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            combined = np.hstack([display, depth_colormap])
            cv2.imshow("Socket Detection (Left: RGB | Right: Depth)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                save_count += 1
                fname = f"detection_result_{save_count}.png"
                cv2.imwrite(fname, combined)
                print(f"  저장: {fname}")
            elif key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def test_image(image_path):
    """단일 이미지 탐지"""
    model = YOLO(MODEL_PATH)
    print(f"모델 로드: {MODEL_PATH}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 열 수 없습니다: {image_path}")
        return

    results = model(image, conf=CONFIDENCE, verbose=False)
    display, count = draw_detections(image.copy(), results)

    print(f"탐지 결과: {count}개 객체")
    for i in range(len(results[0].boxes)):
        cls_id = int(results[0].boxes.cls[i])
        conf = float(results[0].boxes.conf[i])
        print(f"  - {CLASS_NAMES.get(cls_id, cls_id)}: {conf:.1%}")

    cv2.imshow("Detection Result", display)
    print("아무 키나 누르면 종료")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_directory(dir_path):
    """폴더 내 모든 이미지 탐지"""
    model = YOLO(MODEL_PATH)
    print(f"모델 로드: {MODEL_PATH}")

    image_files = sorted(glob.glob(f"{dir_path}/*.png"))
    image_files = [f for f in image_files if "_mask" not in f]
    print(f"이미지: {len(image_files)}장\n")

    os.makedirs("detection_results", exist_ok=True)

    for idx, img_path in enumerate(image_files):
        basename = os.path.basename(img_path)
        image = cv2.imread(img_path)

        results = model(image, conf=CONFIDENCE, verbose=False)
        display, count = draw_detections(image.copy(), results)

        cv2.imwrite(f"detection_results/{basename}", display)

        classes = []
        for i in range(len(results[0].boxes)):
            cls_id = int(results[0].boxes.cls[i])
            conf = float(results[0].boxes.conf[i])
            classes.append(f"{CLASS_NAMES.get(cls_id, cls_id)}({conf:.0%})")

        summary = ", ".join(classes) if classes else "탐지 없음"
        print(f"  [{idx+1}/{len(image_files)}] {basename}: {count}개 → {summary}")

    print(f"\n결과 저장 → detection_results/")


if __name__ == "__main__":
    if "--image" in sys.argv:
        idx = sys.argv.index("--image")
        test_image(sys.argv[idx + 1])
    elif "--dir" in sys.argv:
        idx = sys.argv.index("--dir")
        test_directory(sys.argv[idx + 1])
    else:
        test_camera()
