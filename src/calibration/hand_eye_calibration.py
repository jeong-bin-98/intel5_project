"""
Hand-Eye Calibration (Eye-to-Hand)
===================================
RealSense D435i + Indy7 로봇의 카메라-로봇 좌표 변환 행렬을 구합니다.

구성:
    - 카메라: 작업대 위에 고정 (Eye-to-Hand 구성)
    - ArUco 마커: 로봇 그리퍼(말단)에 부착
    - 로봇이 여러 자세를 취하면서 카메라가 마커를 촬영

사용법:
    1. ArUco 마커 출력:
        python src/calibration/hand_eye_calibration.py --generate-marker

    2. 캘리브레이션 데이터 수집 (인터랙티브):
        python src/calibration/hand_eye_calibration.py --collect
        python src/calibration/hand_eye_calibration.py --collect --robot  (실제 로봇)

    3. 저장된 데이터로 캘리브레이션 계산:
        python src/calibration/hand_eye_calibration.py --calibrate

    4. 결과 검증:
        python src/calibration/hand_eye_calibration.py --verify

절차:
    1. --generate-marker 로 ArUco 마커를 출력하여 로봇 그리퍼에 부착
    2. --collect 실행 → 로봇을 다양한 자세로 이동
    3. 각 자세에서 's' 키로 (로봇 TCP 좌표 + 카메라 마커 좌표) 쌍을 저장
    4. 최소 10쌍 이상 수집 (15~20쌍 권장)
    5. --calibrate 로 변환 행렬 계산 → config/robot_config.py 에 자동 저장
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import cv2
import json
import pyrealsense2 as rs
from datetime import datetime
from config.paths import PROJECT_ROOT


# ===== 설정 =====
# ArUco 마커 설정
ARUCO_DICT_TYPE = cv2.aruco.DICT_5X5_50
MARKER_ID = 0               # 사용할 마커 ID
MARKER_SIZE_M = 0.05         # 마커 실제 크기 (미터) — 출력 후 실측하여 정확히 입력

# 캘리브레이션 데이터 저장 경로
CALIB_DATA_DIR = os.path.join(PROJECT_ROOT, "config", "calibration_data")
CALIB_DATA_FILE = os.path.join(CALIB_DATA_DIR, "hand_eye_poses.json")
CALIB_RESULT_FILE = os.path.join(CALIB_DATA_DIR, "calibration_result.json")


# =============================================================================
# ArUco 마커 생성
# =============================================================================

def generate_marker(marker_id=MARKER_ID, size_px=500):
    """ArUco 마커를 생성하고 PNG로 저장합니다.

    출력된 마커를 인쇄하여 로봇 그리퍼에 단단히 부착하세요.
    인쇄 후 실제 크기를 자로 측정하여 MARKER_SIZE_M 값을 수정하세요.

    Args:
        marker_id: 마커 번호 (0~49)
        size_px: 이미지 크기 (pixels)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)

    # 여백 추가 (인쇄 시 잘림 방지)
    border = 50
    canvas = np.ones((size_px + border * 2, size_px + border * 2), dtype=np.uint8) * 255
    canvas[border:border + size_px, border:border + size_px] = marker_img

    output_path = os.path.join(PROJECT_ROOT, f"aruco_marker_{marker_id}.png")
    cv2.imwrite(output_path, canvas)

    print(f"ArUco 마커 생성 완료:")
    print(f"  파일: {output_path}")
    print(f"  마커 ID: {marker_id}")
    print(f"  Dictionary: DICT_5X5_50")
    print(f"\n사용법:")
    print(f"  1. 이 이미지를 출력하세요")
    print(f"  2. 마커의 검은 테두리 한 변 길이를 자로 측정하세요")
    print(f"  3. 이 스크립트의 MARKER_SIZE_M 값을 측정값(미터)으로 수정하세요")
    print(f"     현재 설정: {MARKER_SIZE_M}m ({MARKER_SIZE_M*100}cm)")
    print(f"  4. 로봇 그리퍼에 평평하게 부착하세요")


# =============================================================================
# RealSense 카메라 intrinsics
# =============================================================================

def get_camera_intrinsics(pipeline, align):
    """RealSense에서 카메라 내부 파라미터를 가져옵니다.

    Returns:
        camera_matrix: 3x3 카메라 행렬
        dist_coeffs: 왜곡 계수
        intrinsics: RealSense intrinsics 객체
    """
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()

    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float64)

    # D435i는 Brown-Conrady 왜곡 모델 사용
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)

    return camera_matrix, dist_coeffs, intrinsics


# =============================================================================
# ArUco 마커 검출 + 자세 추정
# =============================================================================

def detect_marker_pose(color_image, camera_matrix, dist_coeffs):
    """이미지에서 ArUco 마커를 검출하고 카메라 기준 자세를 추정합니다.

    Args:
        color_image: BGR 이미지
        camera_matrix: 3x3 카메라 행렬
        dist_coeffs: 왜곡 계수

    Returns:
        success: 마커 검출 여부
        rvec: 회전 벡터 (Rodrigues) — 카메라→마커
        tvec: 병진 벡터 — 카메라→마커 (미터)
        corners: 마커 코너 좌표 (시각화용)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or MARKER_ID not in ids.flatten():
        return False, None, None, None

    # 원하는 마커 ID의 인덱스 찾기
    idx = np.where(ids.flatten() == MARKER_ID)[0][0]
    marker_corners = corners[idx]

    # 마커의 3D 좌표 (마커 로컬 좌표계, 원점 = 마커 중심)
    obj_points = np.array([
        [-MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
        [ MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
        [ MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
        [-MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
    ], dtype=np.float64)

    # solvePnP로 마커 자세 추정
    success, rvec, tvec = cv2.solvePnP(
        obj_points,
        marker_corners.reshape(-1, 2),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    return success, rvec, tvec, marker_corners


def draw_marker_info(image, corners, rvec, tvec, camera_matrix, dist_coeffs):
    """검출된 마커와 좌표축을 이미지 위에 그립니다."""
    cv2.aruco.drawDetectedMarkers(image, [corners])
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE_M * 0.5)

    # 마커 위치 텍스트
    t = tvec.flatten()
    pos_text = f"Marker: X={t[0]*1000:.1f} Y={t[1]*1000:.1f} Z={t[2]*1000:.1f} mm"
    cv2.putText(image, pos_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image


# =============================================================================
# 데이터 수집
# =============================================================================

def collect_data(use_robot=False):
    """카메라+로봇 자세 쌍을 인터랙티브하게 수집합니다.

    Args:
        use_robot: True이면 Indy7에서 실시간 TCP 좌표를 읽음
                   False이면 수동으로 좌표를 입력
    """
    os.makedirs(CALIB_DATA_DIR, exist_ok=True)

    # 기존 데이터 로드
    pose_pairs = []
    if os.path.exists(CALIB_DATA_FILE):
        with open(CALIB_DATA_FILE, 'r') as f:
            pose_pairs = json.load(f)
        print(f"기존 데이터 {len(pose_pairs)}쌍 로드됨")

    # 로봇 연결
    robot = None
    if use_robot:
        from src.robot.indy_controller import Indy7Controller
        from config.robot_config import ROBOT_IP, ROBOT_NAME
        robot = Indy7Controller(ROBOT_IP, ROBOT_NAME, simulation=False)
        robot.connect()
        print(f"Indy7 연결 완료: {ROBOT_IP}")

    # RealSense 초기화
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    camera_matrix, dist_coeffs, intrinsics = get_camera_intrinsics(pipeline, align)
    print(f"\nCamera intrinsics loaded:")
    print(f"  fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
    print(f"  ppx={intrinsics.ppx:.1f}, ppy={intrinsics.ppy:.1f}")

    print(f"\n{'=' * 60}")
    print(f"  Hand-Eye Calibration — 데이터 수집")
    print(f"  수집된 자세: {len(pose_pairs)}쌍")
    print(f"{'=' * 60}")
    print(f"  조작법:")
    print(f"    's' = 현재 자세 저장")
    print(f"    'u' = 마지막 자세 삭제 (undo)")
    print(f"    'q' = 종료 + 저장")
    print(f"{'=' * 60}")
    print(f"\n  로봇을 다양한 자세로 이동시키며 's'를 누르세요.")
    print(f"  마커가 카메라에 잘 보여야 합니다.")
    print(f"  다양한 위치/각도로 최소 10쌍, 권장 15~20쌍 수집하세요.\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            display = color_image.copy()

            # 마커 검출
            success, rvec, tvec, corners = detect_marker_pose(
                color_image, camera_matrix, dist_coeffs
            )

            if success:
                display = draw_marker_info(display, corners, rvec, tvec,
                                           camera_matrix, dist_coeffs)
                status_color = (0, 255, 0)
                status_text = "Marker DETECTED"
            else:
                status_color = (0, 0, 255)
                status_text = "Marker NOT found"

            # 상태 표시
            cv2.putText(display, status_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(display, f"Collected: {len(pose_pairs)} pairs (need >= 10)",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if use_robot and robot and robot.connected:
                try:
                    tcp = robot.indy.get_task_pos()
                    tcp_text = (f"Robot TCP: X={tcp[0]:.4f} Y={tcp[1]:.4f} Z={tcp[2]:.4f} "
                                f"RX={tcp[3]:.1f} RY={tcp[4]:.1f} RZ={tcp[5]:.1f}")
                    cv2.putText(display, tcp_text, (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
                except Exception:
                    pass

            cv2.imshow("Hand-Eye Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and success:
                # 로봇 TCP 좌표 가져오기
                if use_robot and robot:
                    tcp_pose = robot.indy.get_task_pos()  # [x,y,z,rx,ry,rz]
                else:
                    print("\n  로봇 TCP 좌표를 입력하세요 (미터, 도):")
                    try:
                        x = float(input("    X (m): "))
                        y = float(input("    Y (m): "))
                        z = float(input("    Z (m): "))
                        rx = float(input("    RX (deg): "))
                        ry = float(input("    RY (deg): "))
                        rz = float(input("    RZ (deg): "))
                        tcp_pose = [x, y, z, rx, ry, rz]
                    except ValueError:
                        print("  잘못된 입력. 다시 시도하세요.")
                        continue

                pair = {
                    "robot_tcp": list(tcp_pose),           # [x,y,z,rx,ry,rz] 미터/도
                    "marker_rvec": rvec.flatten().tolist(), # 회전 벡터 (Rodrigues)
                    "marker_tvec": tvec.flatten().tolist(), # 병진 벡터 (미터)
                    "timestamp": datetime.now().isoformat()
                }
                pose_pairs.append(pair)

                t = tvec.flatten()
                print(f"  [{len(pose_pairs)}] 저장 완료!")
                print(f"    Robot TCP: [{tcp_pose[0]:.4f}, {tcp_pose[1]:.4f}, {tcp_pose[2]:.4f}, "
                      f"{tcp_pose[3]:.1f}, {tcp_pose[4]:.1f}, {tcp_pose[5]:.1f}]")
                print(f"    Marker:    [{t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}] mm")

                # 즉시 파일 저장 (데이터 손실 방지)
                with open(CALIB_DATA_FILE, 'w') as f:
                    json.dump(pose_pairs, f, indent=2)

            elif key == ord('s') and not success:
                print("  마커가 검출되지 않았습니다. 마커가 카메라에 보이는지 확인하세요.")

            elif key == ord('u') and pose_pairs:
                removed = pose_pairs.pop()
                print(f"  마지막 자세 삭제됨 (남은 {len(pose_pairs)}쌍)")
                with open(CALIB_DATA_FILE, 'w') as f:
                    json.dump(pose_pairs, f, indent=2)

            elif key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if robot:
            robot.disconnect()

        # 최종 저장
        with open(CALIB_DATA_FILE, 'w') as f:
            json.dump(pose_pairs, f, indent=2)

        print(f"\n데이터 수집 완료: {len(pose_pairs)}쌍")
        print(f"저장 위치: {CALIB_DATA_FILE}")
        if len(pose_pairs) >= 10:
            print(f"\n다음 단계: python src/calibration/hand_eye_calibration.py --calibrate")
        else:
            print(f"\n최소 10쌍이 필요합니다. 더 수집하세요.")


# =============================================================================
# 캘리브레이션 계산
# =============================================================================

def robot_pose_to_matrix(tcp_pose):
    """로봇 TCP 자세 [x,y,z,rx,ry,rz] → 4x4 동차 변환 행렬.

    Indy7 좌표계: [x,y,z] 미터, [rx,ry,rz] 도 (XYZ Euler)

    Args:
        tcp_pose: [x, y, z, rx, ry, rz]

    Returns:
        4x4 변환 행렬 (base → gripper)
    """
    x, y, z, rx, ry, rz = tcp_pose

    # 오일러 각 → 회전 행렬 (degrees → radians)
    rx_rad = np.radians(rx)
    ry_rad = np.radians(ry)
    rz_rad = np.radians(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx_rad), -np.sin(rx_rad)],
        [0, np.sin(rx_rad),  np.cos(rx_rad)]
    ])
    Ry = np.array([
        [ np.cos(ry_rad), 0, np.sin(ry_rad)],
        [0, 1, 0],
        [-np.sin(ry_rad), 0, np.cos(ry_rad)]
    ])
    Rz = np.array([
        [np.cos(rz_rad), -np.sin(rz_rad), 0],
        [np.sin(rz_rad),  np.cos(rz_rad), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx  # ZYX 순서 (Indy7 기본)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T


def calibrate():
    """수집된 데이터로 Hand-Eye 캘리브레이션을 수행합니다.

    Eye-to-Hand 구성:
        AX = XB 문제를 풀어 카메라↔로봇 변환 행렬을 구합니다.
        - A: 로봇 그리퍼 자세 (base→gripper)
        - B: 카메라가 본 마커 자세 (camera→marker)
        - X: 카메라→로봇 베이스 변환 행렬 (구하려는 것)

    OpenCV의 calibrateHandEye를 사용하며, 여러 알고리즘의 결과를 비교합니다.
    """
    if not os.path.exists(CALIB_DATA_FILE):
        print(f"캘리브레이션 데이터가 없습니다: {CALIB_DATA_FILE}")
        print(f"먼저 --collect 로 데이터를 수집하세요.")
        return

    with open(CALIB_DATA_FILE, 'r') as f:
        pose_pairs = json.load(f)

    if len(pose_pairs) < 3:
        print(f"데이터가 부족합니다: {len(pose_pairs)}쌍 (최소 3, 권장 10+)")
        return

    print(f"캘리브레이션 데이터: {len(pose_pairs)}쌍")
    print(f"{'=' * 60}")

    # 데이터 변환
    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    for pair in pose_pairs:
        # 로봇 자세 → 행렬
        tcp = pair["robot_tcp"]
        T_base2gripper = robot_pose_to_matrix(tcp)

        # Eye-to-Hand: gripper2base = inverse(base2gripper)
        T_gripper2base = np.linalg.inv(T_base2gripper)
        R_gripper2base_list.append(T_gripper2base[:3, :3])
        t_gripper2base_list.append(T_gripper2base[:3, 3].reshape(3, 1))

        # 마커 자세 (카메라→마커)
        rvec = np.array(pair["marker_rvec"])
        tvec = np.array(pair["marker_tvec"])
        R_target2cam, _ = cv2.Rodrigues(rvec)
        R_target2cam_list.append(R_target2cam)
        t_target2cam_list.append(tvec.reshape(3, 1))

    # 여러 알고리즘으로 비교
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    results = {}
    print(f"\n--- 캘리브레이션 결과 ---\n")

    for name, method in methods.items():
        try:
            R_cam2base, t_cam2base = cv2.calibrateHandEye(
                R_gripper2base_list,
                t_gripper2base_list,
                R_target2cam_list,
                t_target2cam_list,
                method=method
            )

            T = np.eye(4)
            T[:3, :3] = R_cam2base
            T[:3, 3] = t_cam2base.flatten()

            # 회전 행렬 유효성 검사 (det(R) ≈ 1, R^T R ≈ I)
            det = np.linalg.det(R_cam2base)
            ortho_err = np.linalg.norm(R_cam2base.T @ R_cam2base - np.eye(3))

            results[name] = {
                "matrix": T,
                "det": det,
                "ortho_err": ortho_err,
                "translation": t_cam2base.flatten()
            }

            tx, ty, tz = t_cam2base.flatten() * 1000  # mm로 표시
            print(f"  [{name}]")
            print(f"    Translation: X={tx:.1f} Y={ty:.1f} Z={tz:.1f} mm")
            print(f"    det(R)={det:.6f}, ortho_err={ortho_err:.6f}")
            print()

        except Exception as e:
            print(f"  [{name}] 실패: {e}\n")

    if not results:
        print("모든 알고리즘이 실패했습니다. 데이터를 확인하세요.")
        return

    # 가장 좋은 결과 선택 (det(R)이 1에 가깝고 직교 에러가 작은 것)
    best_name = min(results, key=lambda k: abs(results[k]["det"] - 1.0) + results[k]["ortho_err"])
    best_matrix = results[best_name]["matrix"]

    print(f"{'=' * 60}")
    print(f"최적 알고리즘: {best_name}")
    print(f"\n카메라→로봇 변환 행렬 (4x4):")
    print(np.array2string(best_matrix, precision=6, suppress_small=True))

    # 결과 저장 (JSON)
    result_data = {
        "method": best_name,
        "matrix": best_matrix.tolist(),
        "num_poses": len(pose_pairs),
        "timestamp": datetime.now().isoformat(),
        "marker_size_m": MARKER_SIZE_M,
        "all_results": {
            name: {
                "matrix": r["matrix"].tolist(),
                "det": r["det"],
                "ortho_err": r["ortho_err"]
            }
            for name, r in results.items()
        }
    }

    with open(CALIB_RESULT_FILE, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"\n결과 저장: {CALIB_RESULT_FILE}")

    # config/robot_config.py 에 자동 반영
    _update_robot_config(best_matrix)

    print(f"\nconfig/robot_config.py 에 CAMERA_TO_ROBOT_MATRIX 업데이트 완료!")
    print(f"\n검증: python src/calibration/hand_eye_calibration.py --verify")


def _update_robot_config(matrix):
    """config/robot_config.py의 CAMERA_TO_ROBOT_MATRIX를 업데이트합니다."""
    config_path = os.path.join(PROJECT_ROOT, "config", "robot_config.py")

    with open(config_path, 'r') as f:
        content = f.read()

    # 행렬을 Python 코드로 포맷팅
    m = matrix
    matrix_str = (
        f"CAMERA_TO_ROBOT_MATRIX = np.array([\n"
        f"    [{m[0,0]:12.8f}, {m[0,1]:12.8f}, {m[0,2]:12.8f}, {m[0,3]:12.8f}],\n"
        f"    [{m[1,0]:12.8f}, {m[1,1]:12.8f}, {m[1,2]:12.8f}, {m[1,3]:12.8f}],\n"
        f"    [{m[2,0]:12.8f}, {m[2,1]:12.8f}, {m[2,2]:12.8f}, {m[2,3]:12.8f}],\n"
        f"    [{m[3,0]:12.8f}, {m[3,1]:12.8f}, {m[3,2]:12.8f}, {m[3,3]:12.8f}],\n"
        f"])"
    )

    # 기존 CAMERA_TO_ROBOT_MATRIX 라인을 교체
    import re
    # None 또는 np.array(...) 형태 모두 매칭
    pattern = r'CAMERA_TO_ROBOT_MATRIX\s*=\s*(?:None|np\.array\(\[.*?\]\))'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, matrix_str, content, flags=re.DOTALL)
    else:
        # 패턴을 찾지 못하면 파일 끝에 추가
        content += f"\n{matrix_str}\n"

    with open(config_path, 'w') as f:
        f.write(content)


# =============================================================================
# 검증
# =============================================================================

def verify():
    """캘리브레이션 결과를 실시간으로 검증합니다.

    카메라가 검출한 마커 좌표를 로봇 좌표로 변환하여 표시합니다.
    로봇의 실제 TCP 좌표와 비교하여 정확도를 확인하세요.
    """
    if not os.path.exists(CALIB_RESULT_FILE):
        print("캘리브레이션 결과가 없습니다. --calibrate 를 먼저 실행하세요.")
        return

    with open(CALIB_RESULT_FILE, 'r') as f:
        result = json.load(f)

    T_cam2base = np.array(result["matrix"])
    print(f"캘리브레이션 결과 로드 (method: {result['method']})")
    print(f"  사용 자세 수: {result['num_poses']}")
    print(f"  마커 크기: {result['marker_size_m']}m")
    print(f"\n변환 행렬:")
    print(np.array2string(T_cam2base, precision=6))

    # RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    camera_matrix, dist_coeffs, _ = get_camera_intrinsics(pipeline, align)

    print(f"\n실시간 검증 시작 ('q'로 종료)")
    print(f"  마커의 카메라 좌표와 로봇 좌표가 표시됩니다.")
    print(f"  로봇의 실제 TCP 좌표와 비교하세요.\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            display = color_image.copy()

            success, rvec, tvec, corners = detect_marker_pose(
                color_image, camera_matrix, dist_coeffs
            )

            if success:
                display = draw_marker_info(display, corners, rvec, tvec,
                                           camera_matrix, dist_coeffs)

                # 카메라→로봇 좌표 변환
                t_cam = tvec.flatten()  # 미터
                cam_point = np.array([t_cam[0], t_cam[1], t_cam[2], 1.0])
                robot_point = T_cam2base @ cam_point

                # 로봇 좌표 표시 (미터)
                rx, ry, rz = robot_point[0], robot_point[1], robot_point[2]
                robot_text = f"Robot frame: X={rx*1000:.1f} Y={ry*1000:.1f} Z={rz*1000:.1f} mm"
                cv2.putText(display, robot_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                cam_text = f"Camera frame: X={t_cam[0]*1000:.1f} Y={t_cam[1]*1000:.1f} Z={t_cam[2]*1000:.1f} mm"
                cv2.putText(display, cam_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(display, "Marker NOT found", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(display, "Verification Mode | 'q' to quit", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Calibration Verification", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("검증 종료.")


# =============================================================================
# 엔트리 포인트
# =============================================================================

if __name__ == "__main__":
    if "--generate-marker" in sys.argv:
        generate_marker()
    elif "--collect" in sys.argv:
        use_robot = "--robot" in sys.argv
        collect_data(use_robot=use_robot)
    elif "--calibrate" in sys.argv:
        calibrate()
    elif "--verify" in sys.argv:
        verify()
    else:
        print(__doc__)
        print("옵션:")
        print("  --generate-marker   ArUco 마커 이미지 생성")
        print("  --collect           데이터 수집 (--robot: 실제 로봇 연결)")
        print("  --calibrate         변환 행렬 계산")
        print("  --verify            결과 실시간 검증")
