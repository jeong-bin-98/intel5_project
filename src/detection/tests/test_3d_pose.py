"""
3D 자세 추정 테스트
====================

RealSense D435 + YOLOv8 기반 소켓 3D 좌표 + 자세 추정을 테스트합니다.

릴레이 소켓은 금속 재질이라 텍스처 기반 특징점(FAST+BRISK)이 부족합니다.
대신 depth 기반 방법을 사용합니다:
    - 3D 좌표 (X, Y, Z): depth + 카메라 intrinsics → rs2_deproject_pixel_to_point
    - Roll, Pitch: depth 점군에 PCA 평면 피팅 → 표면 법선
    - Yaw: depth 마스크 + 윤곽선 minAreaRect

사용법:
    python src/detection/tests/test_3d_pose.py              # 실시간 3D 좌표 표시
    python src/detection/tests/test_3d_pose.py --verbose     # 콘솔에 좌표 실시간 출력
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import cv2
import numpy as np
import time
from ultralytics import YOLO

from config.paths import YOLO_BEST_PT
from src.detection.constants import DEVICE, CONFIDENCE
from src.detection.tracker import ObjectSmoother, select_pick_target
from src.detection.visualization import draw_3d_detections, draw_depth_analysis
from src.detection.binpicking_3d import detect_3d
from src.detection.realsense_config import create_pipeline, apply_filters


def test_3d_live(verbose=False):
    """실시간 3D 좌표 테스트.

    화면에 각 소켓의 XYZ 좌표(mm)와 자세(roll, pitch, yaw)를 표시합니다.
    --verbose 옵션 시 콘솔에도 실시간 출력합니다.
    """
    print(f"YOLOv8 모델 로드... (device={DEVICE})")
    model = YOLO(YOLO_BEST_PT)
    model.to(DEVICE)

    smoother = ObjectSmoother()

    # RealSense 초기화 (센서 튜닝 + 필터 체인 + 워밍업 포함)
    # 화면은 640x480(4:3), depth는 848x480(최적 해상도) 유지
    pipeline, align, filters, intrinsics = create_pipeline(
        color_res=(640, 480)
    )

    frame_count = 0

    print("\n" + "=" * 60)
    print("  3D Socket Detection Test")
    print("  Depth 기반 3D 좌표 + 윤곽선 기반 Yaw")
    print("=" * 60)
    print("  'q' = 종료  |  's' = 저장  |  'd' = 상세 출력")
    print("=" * 60)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Depth 필터 (Threshold → Disparity → Spatial → Temporal → Depth → Hole)
            depth_frame = apply_filters(depth_frame, filters)

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 3D 탐지 + 스무딩
            t0 = time.time()
            raw_objects = detect_3d(model, color_image, depth_image, intrinsics)
            objects = smoother.smooth(raw_objects)
            detect_time = (time.time() - t0) * 1000

            # 시각화
            display = draw_3d_detections(color_image, objects, intrinsics)
            depth_panel = draw_depth_analysis(depth_image, objects)

            # 처리 시간
            cv2.putText(display, f"Time: {detect_time:.0f}ms", (490, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # depth 패널 높이 맞추기
            if depth_panel.shape[0] != display.shape[0]:
                depth_panel = cv2.resize(
                    depth_panel, (depth_panel.shape[1], display.shape[0])
                )

            combined = np.hstack([display, depth_panel])
            cv2.imshow("3D Socket Detection (RGB+3D | Depth Analysis)", combined)

            # verbose: 콘솔에 좌표 출력
            frame_count += 1
            if verbose and objects and frame_count % 15 == 0:
                print(f"\n--- Frame #{frame_count} ({detect_time:.0f}ms) ---")
                target = select_pick_target(objects)
                for i, obj in enumerate(objects):
                    marker = " >> " if obj is target else "    "
                    print(f"{marker}[{i}] {obj.class_name} {obj.confidence:.0%}")
                    print(f"        XYZ: ({obj.pos_3d[0]:.1f}, {obj.pos_3d[1]:.1f}, {obj.pos_3d[2]:.1f}) mm")
                    print(f"        RPY: ({obj.orientation[0]:.1f}, {obj.orientation[1]:.1f}, {obj.orientation[2]:.1f}) deg")
                    if obj.approach:
                        ap, ar = obj.approach
                        print(f"        Approach: ({ap[0]:.1f}, {ap[1]:.1f}, {ap[2]:.1f}) mm")

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"3d_test_{int(time.time())}.png"
                cv2.imwrite(fname, combined)
                print(f"  저장: {fname}")
            elif key == ord('d'):
                print(f"\n{'=' * 50}")
                print(f"Frame #{frame_count} | {len(objects)}개 탐지 | {detect_time:.1f}ms")
                print(f"{'=' * 50}")
                target = select_pick_target(objects)
                for i, obj in enumerate(objects):
                    is_target = " [PICK TARGET]" if obj is target else ""
                    print(f"\n  [{i}] {obj.class_name} {obj.confidence:.0%}{is_target}")
                    print(f"      3D 좌표:  X={obj.pos_3d[0]:>8.1f}  Y={obj.pos_3d[1]:>8.1f}  Z={obj.pos_3d[2]:>8.1f} mm")
                    print(f"      자세:     R={obj.orientation[0]:>8.1f}  P={obj.orientation[1]:>8.1f}  Y={obj.orientation[2]:>8.1f} deg")
                    if obj.approach:
                        ap, ar = obj.approach
                        print(f"      접근위치: X={ap[0]:>8.1f}  Y={ap[1]:>8.1f}  Z={ap[2]:>8.1f} mm")
                        print(f"      접근자세: R={ar[0]:>8.1f}  P={ar[1]:>8.1f}  Y={ar[2]:>8.1f} deg")
                print()

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("테스트 종료.")


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    test_3d_live(verbose=verbose)
