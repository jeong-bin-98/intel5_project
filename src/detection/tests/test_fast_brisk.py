"""
FAST+BRISK 3D 자세 추정 테스트
================================

FAST+BRISK 기반 소켓 3D 자세 추정을 다양한 방식으로 테스트합니다.
depth 결합 3D 결과와 기존 PCA 방식을 비교할 수 있습니다.

테스트 모드:
    1. --live         실시간 카메라 (D435 + YOLO + FAST+BRISK 3D)
    2. --keypoints    단일 이미지에서 FAST 키포인트 시각화
    3. --accuracy     합성 회전 정확도 측정
    4. --match        두 이미지 간 매칭 테스트

사용법:
    python src/detection/tests/test_fast_brisk.py --live
    python src/detection/tests/test_fast_brisk.py --keypoints image.png
    python src/detection/tests/test_fast_brisk.py --accuracy image.png
    python src/detection/tests/test_fast_brisk.py --match template.png query.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import cv2
import numpy as np
import time

from src.detection.fast_brisk_pose import FastBriskPoseEstimator, extract_roi


CLASS_NAMES = {0: "8pin", 1: "12pin"}
COLORS = {0: (255, 100, 0), 1: (0, 100, 255)}


def test_keypoints(image_path):
    """단일 이미지에서 FAST 키포인트를 다양한 threshold로 비교 시각화."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 열 수 없습니다: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholds = [10, 20, 30, 50]
    results = []

    for thresh in thresholds:
        fast = cv2.FastFeatureDetector_create(
            threshold=thresh, nonmaxSuppression=True,
            type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
        )
        brisk = cv2.BRISK_create()

        kp = fast.detect(gray, None)
        kp, des = brisk.compute(gray, kp)

        vis = cv2.drawKeypoints(
            image, kp, None, color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        label = f"FAST thresh={thresh}: {len(kp)} kp"
        cv2.putText(vis, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if des is not None:
            cv2.putText(vis, f"BRISK desc: {des.shape}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        results.append(vis)
        print(f"  {label}" + (f" desc={des.shape}" if des is not None else ""))

    # 2x2 그리드
    if len(results) == 4:
        h, w = results[0].shape[:2]
        grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        grid[0:h, 0:w] = results[0]
        grid[0:h, w:2*w] = results[1]
        grid[h:2*h, 0:w] = results[2]
        grid[h:2*h, w:2*w] = results[3]

        scale = min(1920 / (w * 2), 1080 / (h * 2), 1.0)
        if scale < 1.0:
            grid = cv2.resize(grid, None, fx=scale, fy=scale)
        cv2.imshow("FAST Keypoints Comparison", grid)
    else:
        for i, vis in enumerate(results):
            cv2.imshow(f"Result {i}", vis)

    print("\n아무 키나 누르면 종료합니다...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_match(template_path, query_path):
    """두 이미지 간 FAST+BRISK 매칭 테스트."""
    template = cv2.imread(template_path)
    query = cv2.imread(query_path)
    if template is None or query is None:
        print("이미지를 열 수 없습니다")
        return

    estimator = FastBriskPoseEstimator()
    estimator.register_template_from_image(template, class_id=0, name="template")

    t_start = time.time()
    result = estimator.estimate_pose(query, class_id=0)
    elapsed = (time.time() - t_start) * 1000

    if result:
        print(f"\n--- 매칭 결과 (2D) ---")
        print(f"  회전각:    {result['angle']:.1f}°")
        print(f"  스케일:    {result['scale']:.3f}")
        print(f"  중심:      ({result['center'][0]:.1f}, {result['center'][1]:.1f})")
        print(f"  매칭:      {result['num_inliers']}/{result['num_matches']} 인라이어")
        print(f"  신뢰도:    {result['confidence']:.1%}")
        print(f"  처리 시간: {elapsed:.1f}ms")

        t_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        display = estimator.draw_matches(t_gray, query, result)
        cv2.imshow("FAST+BRISK Matching", display)
        cv2.waitKey(0)
    else:
        print(f"매칭 실패 ({elapsed:.1f}ms)")

    cv2.destroyAllWindows()


def test_accuracy(image_path):
    """합성 회전 테스트로 yaw 추정 정확도를 측정합니다."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 열 수 없습니다: {image_path}")
        return

    estimator = FastBriskPoseEstimator(min_matches=6)
    estimator.register_template_from_image(image, class_id=0, name="original")

    test_angles = list(range(0, 360, 15))
    errors = []
    successes = 0

    print(f"\n{'실제':>8} {'추정':>8} {'오차':>8} {'매칭':>6} {'결과':>6}")
    print("-" * 44)

    for true_angle in test_angles:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -true_angle, 1.0)

        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(image, M, (new_w, new_h),
                                  borderMode=cv2.BORDER_REFLECT)

        result = estimator.estimate_pose(rotated, class_id=0)

        if result:
            estimated = result['angle']
            error = abs(estimated - true_angle)
            if error > 180:
                error = 360 - error
            errors.append(error)
            successes += 1
            status = "OK" if error < 10 else "WARN"
            print(f"  {true_angle:>5.0f}°  {estimated:>6.1f}°  {error:>6.1f}°  "
                  f"{result['num_inliers']:>4}   {status}")
        else:
            print(f"  {true_angle:>5.0f}°  {'FAIL':>6}  {'-':>6}  {'-':>4}   FAIL")

    print(f"\n{'=' * 44}")
    print(f"성공: {successes}/{len(test_angles)} ({successes/len(test_angles):.0%})")
    if errors:
        errors = np.array(errors)
        print(f"평균 오차: {errors.mean():.2f}°")
        print(f"최대 오차: {errors.max():.2f}°")
        print(f"5° 이내:  {(errors < 5).sum()}/{len(errors)} ({(errors < 5).mean():.0%})")
        print(f"10° 이내: {(errors < 10).sum()}/{len(errors)} ({(errors < 10).mean():.0%})")


def test_live():
    """실시간 카메라 테스트: YOLO → FAST+BRISK 3D 자세 추정.

    PCA 방식과 FAST+BRISK 3D 방식을 나란히 비교합니다.
    depth 정보를 결합한 3D 좌표(X,Y,Z mm) + 자세(roll,pitch,yaw)가 표시됩니다.
    """
    import pyrealsense2 as rs
    from ultralytics import YOLO
    from config.paths import YOLO_BEST_PT
    from src.detection.binpicking_3d import (
        estimate_surface_normal, estimate_yaw_from_bbox,
        get_depth_at_pixel, pixel_to_3d, CONFIDENCE
    )

    print("YOLOv8 모델 로드...")
    model = YOLO(YOLO_BEST_PT)

    estimator = FastBriskPoseEstimator(fast_threshold=20, min_matches=6)

    template_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'config', 'templates'
    )
    if os.path.isdir(template_dir):
        estimator.load_templates(template_dir)
    else:
        print(f"템플릿 없음 → 첫 탐지를 자동 등록합니다.")

    # RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    intrinsics = None
    auto_registered = {}
    frame_count = 0
    total_fb_time = 0
    total_pca_time = 0

    print("\n" + "=" * 60)
    print("  FAST+BRISK 3D vs PCA 실시간 비교")
    print("=" * 60)
    print("  'q' = 종료  |  't' = 수동 템플릿 등록  |  's' = 저장  |  'd' = 상세")
    print("=" * 60)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            if intrinsics is None:
                intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            display = color_image.copy()

            # YOLO
            results = model(color_image, conf=CONFIDENCE, verbose=False)
            boxes = results[0].boxes

            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                color = COLORS.get(cls_id, (0, 255, 0))
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{CLASS_NAMES.get(cls_id, '?')} {conf:.0%}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # --- PCA (기존) ---
                t0 = time.time()
                _, pca_roll, pca_pitch, _ = estimate_surface_normal(
                    depth_image, cx, cy, intrinsics
                )
                pca_yaw = estimate_yaw_from_bbox(x1, y1, x2, y2)
                depth_mm = get_depth_at_pixel(depth_image, cx, cy)
                pca_pos = pixel_to_3d(cx, cy, depth_mm, intrinsics) if depth_mm > 0 else (0, 0, 0)
                pca_time = (time.time() - t0) * 1000

                # --- FAST+BRISK 3D ---
                # 자동 템플릿 등록
                if cls_id not in auto_registered:
                    roi, _ = extract_roi(color_image, (x1, y1, x2, y2), margin=15)
                    if estimator.register_template_from_image(roi, cls_id, name=f"auto_{cls_id}"):
                        auto_registered[cls_id] = True
                        print(f"  [자동등록] {CLASS_NAMES.get(cls_id, '?')} 템플릿 등록")

                t0 = time.time()
                fb_3d = estimator.estimate_pose_3d(
                    color_image, depth_image, (x1, y1, x2, y2),
                    intrinsics, class_id=cls_id
                )
                fb_time = (time.time() - t0) * 1000

                total_fb_time += fb_time
                total_pca_time += pca_time
                frame_count += 1

                # --- 결과 표시 ---
                # PCA 결과 (노란색)
                pca_text = (f"PCA: X{pca_pos[0]:.0f} Y{pca_pos[1]:.0f} Z{pca_pos[2]:.0f} "
                            f"R{pca_roll:.0f} P{pca_pitch:.0f} Y{pca_yaw:.0f} "
                            f"({pca_time:.1f}ms)")
                cv2.putText(display, pca_text, (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

                # FAST+BRISK 3D 결과 (초록색)
                if fb_3d:
                    p = fb_3d['pos_3d']
                    fb_text = (f"FB3D: X{p[0]:.0f} Y{p[1]:.0f} Z{p[2]:.0f} "
                               f"R{fb_3d['roll']:.0f} P{fb_3d['pitch']:.0f} Y{fb_3d['yaw']:.0f} "
                               f"M{fb_3d['num_inliers']} ({fb_time:.1f}ms)")
                    cv2.putText(display, fb_text, (x1, y2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

                    estimator.draw_pose_on_image(display, (x1, y1, x2, y2), fb_3d)
                else:
                    cv2.putText(display, f"FB3D: no match ({fb_time:.1f}ms)",
                                (x1, y2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # 상단 정보
            cv2.putText(display, f"Detections: {len(boxes)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if frame_count > 0:
                cv2.putText(display,
                            f"Avg: PCA {total_pca_time/frame_count:.1f}ms / "
                            f"FB3D {total_fb_time/frame_count:.1f}ms",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("FAST+BRISK 3D vs PCA", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('t') and len(boxes) > 0:
                cls_id = int(boxes.cls[0])
                x1, y1, x2, y2 = map(int, boxes.xyxy[0].cpu().numpy())
                roi, _ = extract_roi(color_image, (x1, y1, x2, y2), margin=15)
                name = f"manual_{CLASS_NAMES.get(cls_id, str(cls_id))}"
                if estimator.register_template_from_image(roi, cls_id, name=name):
                    print(f"  [수동등록] {name}")
            elif key == ord('s'):
                fname = f"fb3d_test_{int(time.time())}.png"
                cv2.imwrite(fname, display)
                print(f"  저장: {fname}")
            elif key == ord('d'):
                print(f"\n--- 상세 ---")
                print(f"  프레임: {frame_count}")
                if frame_count > 0:
                    print(f"  PCA 평균: {total_pca_time/frame_count:.2f}ms")
                    print(f"  FB3D 평균: {total_fb_time/frame_count:.2f}ms")
                for cid, tmpls in estimator.templates.items():
                    for t in tmpls:
                        print(f"  템플릿: {CLASS_NAMES.get(cid, cid)}/{t['name']} "
                              f"({len(t['keypoints'])} kp)")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("테스트 종료.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("옵션:")
        print("  --keypoints <image>           FAST 키포인트 시각화")
        print("  --match <template> <query>    두 이미지 매칭 테스트")
        print("  --accuracy <image>            합성 회전 정확도 테스트")
        print("  --live                        실시간 3D 비교 (PCA vs FB3D)")
        sys.exit(0)

    mode = sys.argv[1]

    if mode == "--keypoints":
        if len(sys.argv) < 3:
            print("사용법: --keypoints <image_path>")
            sys.exit(1)
        test_keypoints(sys.argv[2])
    elif mode == "--match":
        if len(sys.argv) < 4:
            print("사용법: --match <template> <query>")
            sys.exit(1)
        test_match(sys.argv[2], sys.argv[3])
    elif mode == "--accuracy":
        if len(sys.argv) < 3:
            print("사용법: --accuracy <image_path>")
            sys.exit(1)
        test_accuracy(sys.argv[2])
    elif mode == "--live":
        test_live()
    else:
        print(f"알 수 없는 모드: {mode}")
