"""
3D Object Detection & Pose Estimation for Bin-Picking
=====================================================

YOLOv8 + Intel RealSense D435 기반 3D 소켓 탐지 및 자세 추정 시스템

사용법:
    python src/detection/binpicking_3d.py                  # 실시간 3D 탐지 + 시각화
    python src/detection/binpicking_3d.py --pick           # 빈픽킹 모드 (시뮬레이션)
    python src/detection/binpicking_3d.py --pick --robot   # 빈픽킹 모드 (Indy7 실제 연결)
    python src/detection/binpicking_3d.py --test           # 카메라 없이 단일 이미지 테스트

기능:
    1. YOLOv8으로 소켓(8pin/12pin) 2D 탐지
    2. RealSense depth로 3D 좌표(X,Y,Z) 계산 (카메라 intrinsics 기반)
    3. Depth 기반 표면 법선 추정으로 소켓 기울기(orientation) 계산
    4. 로봇 접근 벡터(approach vector) 자동 계산
    5. 가장 높은 소켓 우선 픽킹 (bin-picking 전략)

조작법:
    - 'q': 종료
    - 's': 현재 프레임 결과 저장
    - 'd': 탐지 정보 콘솔 출력
    - 'p': 픽킹 1회 실행 (--pick 모드)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.paths import YOLO_BEST_PT, NEW_IMAGES_DIR

import numpy as np
import cv2
from datetime import datetime
from ultralytics import YOLO

# 모듈화된 컴포넌트 임포트
from src.detection.constants import (
    DEVICE, CONFIDENCE, CLASS_NAMES, COLORS, DetectedObject
)
from src.detection.geometry import (
    get_depth_at_pixel, pixel_to_3d,
    estimate_surface_normal, estimate_yaw_from_contour,
    compute_approach_vector
)
from src.detection.tracker import ObjectSmoother, select_pick_target
from src.detection.visualization import (
    draw_3d_detections, draw_depth_analysis
)
from src.detection.realsense_config import (
    create_pipeline, apply_filters, capture_median_depth
)


# ===== 설정 =====
MODEL_PATH = YOLO_BEST_PT


# =============================================================================
# 메인 파이프라인
# =============================================================================

def detect_3d(model, color_image, depth_image, intrinsics):
    """
    YOLOv8 탐지 + depth → 3D 좌표 & pose 추정 전체 파이프라인

    레퍼런스의 yolo_order.py + node_order.py의 핵심 로직을 통합.

    Args:
        model: YOLOv8 모델
        color_image: RGB 이미지 (H, W, 3)
        depth_image: depth 이미지 (H, W) - mm
        intrinsics: RealSense 카메라 내부 파라미터

    Returns:
        objects: DetectedObject 리스트
    """
    # 1) YOLOv8 추론 (GPU)
    results = model(color_image, conf=CONFIDENCE, verbose=False, device=DEVICE)
    boxes = results[0].boxes

    objects = []

    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        cls_id = int(boxes.cls[i])
        x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())

        # 2) 바운딩 박스 중심 계산
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 3) 표면 법선 추정 및 RANSAC 평면 기반 정교한 3D 좌표 획득
        estimate_res = estimate_surface_normal(
            depth_image, cx, cy, intrinsics, bbox=(x1, y1, x2, y2)
        )
        normal, roll, pitch, yaw, ransac_pos_3d = estimate_res

        # 평면 추출에 실패했거나 깊이가 없는 경우 기존 Median 기반 백업 로직 사용
        if ransac_pos_3d[2] == 0.0:
            depth_mm = get_depth_at_pixel(depth_image, cx, cy)
            if depth_mm == 0:
                continue  # 유효한 depth가 없으면 스킵
            pos_3d = pixel_to_3d(cx, cy, depth_mm, intrinsics)
        else:
            pos_3d = ransac_pos_3d

        # 4) Yaw 추정: depth 마스크 + 윤곽선 기반
        yaw = estimate_yaw_from_contour(color_image, depth_image, x1, y1, x2, y2)

        # 6) 접근 벡터 계산
        approach = compute_approach_vector(
            pos_3d[0], pos_3d[1], pos_3d[2],
            roll, pitch, yaw,
            offset_mm=100  # 소켓 위 100mm에서 접근
        )

        obj = DetectedObject(
            class_id=cls_id,
            confidence=conf,
            bbox=(x1, y1, x2, y2),
            pos_3d=pos_3d,
            orientation=(roll, pitch, yaw),
            approach=approach,
            normal=normal
        )
        objects.append(obj)

    return objects


# =============================================================================
# 실시간 모드
# =============================================================================

def run_realtime():
    """
    실시간 3D 소켓 탐지 모드

    레퍼런스의 yolo_order.py 메인 루프에 해당하지만,
    ROS 없이 pyrealsense2로 직접 구현.
    """
    # 모델 로드 (GPU)
    print(f"YOLOv8 모델 로드: {MODEL_PATH} (device={DEVICE})")
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)

    # 시간축 스무딩 (칼만 필터 기반 노이즈 안정화)
    smoother = ObjectSmoother()

    # RealSense 초기화 (센서 튜닝 + 필터 체인 + 워밍업 포함)
    pipeline, align, filters, intrinsics = create_pipeline()

    save_count = 0

    print("=" * 60)
    print("3D Socket Detection - Bin Picking System")
    print("=" * 60)
    print("  'q' = 종료  |  's' = 저장  |  'd' = 상세 출력")
    print("=" * 60)

    import time
    fps_time = time.time()
    fps_count = 0
    fps_display = 0.0

    try:
        while True:
            # 프레임 읽기
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Depth 필터 적용 (Threshold → Disparity → Spatial → Temporal → Depth → Hole)
            depth_frame = apply_filters(depth_frame, filters)

            # numpy 변환
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 3D 탐지 + 칼만 필터 스무딩
            raw_objects = detect_3d(model, color_image, depth_image, intrinsics)
            objects = smoother.smooth(raw_objects)

            # 시각화
            display = draw_3d_detections(color_image, objects, intrinsics)
            depth_panel = draw_depth_analysis(depth_image, objects)

            # FPS 계산
            fps_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps_display = fps_count / elapsed
                fps_count = 0
                fps_time = time.time()
            cv2.putText(display, f"FPS: {fps_display:.1f} ({DEVICE})", (450, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # depth 패널 높이를 RGB와 맞추기
            if depth_panel.shape[0] != display.shape[0]:
                depth_panel = cv2.resize(
                    depth_panel, (depth_panel.shape[1],  display.shape[0])
                )

            combined = np.hstack([display, depth_panel])
            cv2.imshow("3D Socket Detection (RGB+3D | Depth Analysis)", combined)

            # 키 입력
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('s'):
                save_count += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"detection_3d_{ts}.png"
                cv2.imwrite(fname, combined)
                print(f"  [저장] {fname}")

            elif key == ord('d'):
                print(f"\n--- Frame Detection Results ({len(objects)} objects) ---")
                for i, obj in enumerate(objects):
                    print(f"  [{i}] {obj}")
                    if obj.approach:
                        ap, ar = obj.approach
                        print(f"      Approach: pos=({ap[0]:.1f}, {ap[1]:.1f}, {ap[2]:.1f}) "
                              f"rot=({ar[0]:.1f}, {ar[1]:.1f}, {ar[2]:.1f})")

                target = select_pick_target(objects)
                if target:
                    print(f"  >> Pick target: {target}")
                print()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[오류 발생] {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("종료.")


# =============================================================================
# 빈픽킹 모드
# =============================================================================

def run_binpicking(simulation=True):
    """
    빈픽킹 제어 루프 (Indy7 로봇 통합)

    레퍼런스의 node_order.py → start_order()에 해당.
    로봇 없이도 detect/stream 명령으로 3D 좌표 테스트 가능.

    Args:
        simulation: True=시뮬레이션 모드 (기본), False=실제 로봇 연결
    """
    print(f"YOLOv8 모델 로드: {MODEL_PATH} (device={DEVICE})")
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)

    # 로봇은 필요할 때 연결 (detect/stream은 로봇 없이 가능)
    robot = None
    robot_connected = False

    mode_str = "시뮬레이션" if simulation else "실제 로봇"

    # RealSense 초기화 (센서 튜닝 + 필터 체인 + 워밍업 포함)
    pipeline, align, filters, intrinsics = create_pipeline()

    def _ensure_robot():
        """로봇이 필요한 명령에서만 연결을 시도합니다."""
        nonlocal robot, robot_connected
        if robot_connected:
            return True
        try:
            robot = get_robot_controller(simulation=simulation)
            robot_connected = True
            return True
        except Exception as e:
            print(f"로봇 연결 실패: {e}")
            print("  detect/stream 명령은 로봇 없이 사용 가능합니다.")
            return False

    try:
        print("=" * 60)
        print(f"  BIN PICKING MODE ({mode_str})")
        print("=" * 60)
        print("\n명령어:")
        print("  detect    - 소켓 탐지 (1회, 로봇 불필요)")
        print("  stream    - 실시간 영상 보기 (로봇 불필요)")
        print("  pick      - 가장 높은 소켓 픽앤플레이스 실행")
        print("  home      - 로봇 홈 위치로 이동")
        print("  loop      - 연속 픽킹 루프 (q로 중지)")
        print("  quit      - 종료")
        print()

        while True:
            cmd = input("binpicking> ").strip().lower()

            if cmd == "quit" or cmd == "q":
                break

            elif cmd == "detect":
                # Multi-frame median으로 안정적 캡처
                color_image, depth_image, intr = capture_median_depth(
                    pipeline, align, filters
                )
                if color_image is None:
                    print("프레임 캡처 실패.")
                    continue

                objects = detect_3d(model, color_image, depth_image, intr)

                print(f"\n탐지 결과: {len(objects)}개")
                for i, obj in enumerate(objects):
                    print(f"  [{i}] {obj}")
                    if obj.approach:
                        ap_mm, ar = obj.approach
                        rob_m = camera_to_robot(ap_mm[0], ap_mm[1], ap_mm[2])
                        print(f"      Robot pos: ({rob_m[0]:.4f}, {rob_m[1]:.4f}, {rob_m[2]:.4f}) m")

                target = select_pick_target(objects)
                if target:
                    print(f"\n  >> Pick target: {target.class_name} "
                          f"Z={target.pos_3d[2]:.1f}mm")

                if objects:
                    display = draw_3d_detections(color_image, objects, intr)
                    cv2.imshow("Detection Result", display)
                    cv2.waitKey(1)

            elif cmd == "stream":
                run_realtime()

            elif cmd == "pick":
                if not _ensure_robot():
                    continue

                color_image, depth_image, intr = capture_median_depth(
                    pipeline, align, filters
                )
                if color_image is None:
                    print("프레임 캡처 실패.")
                    continue

                objects = detect_3d(model, color_image, depth_image, intr)
                target = select_pick_target(objects)
                if target and target.approach:
                    execute_pick_and_place(robot, target)
                else:
                    print("소켓을 찾지 못했습니다.")

            elif cmd == "home":
                if not _ensure_robot():
                    continue
                robot.go_home()
                print("홈 위치 도착.")

            elif cmd == "loop":
                if not _ensure_robot():
                    continue

                print("연속 픽킹 루프 시작 (Ctrl+C 또는 'q'로 중지)")
                pick_count = 0

                while True:
                    color_image, depth_image, intr = capture_median_depth(
                        pipeline, align, filters
                    )
                    if color_image is None:
                        continue

                    objects = detect_3d(model, color_image, depth_image, intr)
                    target = select_pick_target(objects)
                    if target and target.approach:
                        pick_count += 1
                        print(f"\n--- Pick #{pick_count} ---")
                        try:
                            execute_pick_and_place(robot, target)
                        except InterruptedError as e:
                            print(f"  [경고] {e}")
                            robot.set_vacuum(False)
                            break
                    else:
                        print("  소켓 없음. 대기 중...")

                    display = draw_3d_detections(color_image, objects, intr)
                    cv2.imshow("Bin Picking Loop", display)
                    key = cv2.waitKey(500) & 0xFF
                    if key == ord('q'):
                        break

                print(f"루프 종료. 총 {pick_count}회 픽킹.")

            else:
                print(f"알 수 없는 명령: {cmd}")

    except KeyboardInterrupt:
        pass
    except InterruptedError as e:
        print(f"\n[에러] {e}")
    except Exception as e:
        print(f"\n[오류 발생] {e}")
    finally:
        if robot_connected:
            robot.disconnect()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("빈픽킹 모드 종료.")


# =============================================================================
# Indy7 로봇 제어
# =============================================================================

def get_robot_controller(simulation=True):
    """Indy7Controller 인스턴스를 생성하고 연결합니다.

    Args:
        simulation: True=시뮬레이션 모드, False=실제 로봇 연결

    Returns:
        Indy7Controller 인스턴스
    """
    from src.robot.indy_controller import Indy7Controller
    from config.robot_config import ROBOT_IP, ROBOT_NAME

    robot = Indy7Controller(ROBOT_IP, ROBOT_NAME, simulation=simulation)
    robot.connect()
    return robot


def camera_to_robot(cam_x_mm, cam_y_mm, cam_z_mm, transform_matrix=None):
    """
    카메라 좌표(mm) → 로봇 좌표(미터) 변환

    1. 4x4 Hand-eye 캘리브레이션 행렬 적용 (카메라→로봇 좌표 변환)
    2. mm → 미터 변환 (Indy7은 미터 단위 사용)

    Args:
        cam_x_mm, cam_y_mm, cam_z_mm: 카메라 좌표계 3D 좌표 (mm)
        transform_matrix: 4x4 변환 행렬 (None이면 config에서 로드)

    Returns:
        (robot_x_m, robot_y_m, robot_z_m): 로봇 좌표계 (미터)
    """
    if transform_matrix is None:
        from config.robot_config import CAMERA_TO_ROBOT_MATRIX
        transform_matrix = CAMERA_TO_ROBOT_MATRIX

    if transform_matrix is None:
        transform_matrix = np.eye(4)

    cam_point = np.array([cam_x_mm, cam_y_mm, cam_z_mm, 1.0])
    robot_point = transform_matrix @ cam_point

    # mm → 미터 변환
    return (robot_point[0] / 1000.0, robot_point[1] / 1000.0, robot_point[2] / 1000.0)


def execute_pick_and_place(robot, target):
    """탐지된 소켓에 대해 픽앤플레이스를 실행합니다.

    Args:
        robot: Indy7Controller 인스턴스
        target: DetectedObject (select_pick_target에서 선택된 대상)
    """
    from config.robot_config import PLACE_POS, PLACE_ROT, RETRACT_Z_OFFSET

    ap_mm, ar_deg = target.approach

    # 카메라 좌표 → 로봇 좌표 (mm → 미터)
    pick_pos_m = list(camera_to_robot(ap_mm[0], ap_mm[1], ap_mm[2]))
    pick_rot_deg = list(ar_deg)

    print(f"\n=== PICK TARGET ===")
    print(f"  Class: {target.class_name}")
    print(f"  Camera pos: X={target.pos_3d[0]:.1f} Y={target.pos_3d[1]:.1f} Z={target.pos_3d[2]:.1f} mm")
    print(f"  Robot pos:  X={pick_pos_m[0]:.4f} Y={pick_pos_m[1]:.4f} Z={pick_pos_m[2]:.4f} m")
    print(f"  Rotation:   RX={pick_rot_deg[0]:.1f} RY={pick_rot_deg[1]:.1f} RZ={pick_rot_deg[2]:.1f} deg")
    print(f"===================\n")

    # 홈 → 픽 → 플레이스 → 홈
    robot.go_home()
    robot.pick(pick_pos_m, pick_rot_deg, RETRACT_Z_OFFSET)
    robot.place(PLACE_POS, PLACE_ROT, RETRACT_Z_OFFSET)
    robot.go_home()

    print("픽앤플레이스 완료!")


# =============================================================================
# 엔트리 포인트
# =============================================================================

if __name__ == "__main__":
    if "--pick" in sys.argv or "--detect" in sys.argv:
        simulation = "--robot" not in sys.argv
        run_binpicking(simulation=simulation)
    elif "--test" in sys.argv:
        # 카메라 없이 이미지로 테스트 (depth 없이 2D만)
        print("테스트 모드: 단일 이미지 2D 탐지만 수행")
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
        if len(sys.argv) > sys.argv.index("--test") + 1:
            img_path = sys.argv[sys.argv.index("--test") + 1]
        else:
            img_path = NEW_IMAGES_DIR
            import glob
            files = glob.glob(f"{img_path}/*.png")
            if files:
                img_path = files[0]
            else:
                print("테스트할 이미지가 없습니다.")
                sys.exit(1)

        image = cv2.imread(img_path)
        results = model(image, conf=CONFIDENCE, verbose=False, device=DEVICE)
        boxes = results[0].boxes
        print(f"\n탐지 결과 ({img_path}): {len(boxes)}개")
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            print(f"  [{i}] {CLASS_NAMES.get(cls_id, cls_id)} {conf:.0%}")

        display = image.copy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            color = COLORS.get(cls_id, (0, 255, 0))
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.0%}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Test Detection", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        run_realtime()
