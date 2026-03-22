"""
3D 기하학 계산
==============

Depth 이미지 기반 3D 좌표 변환, 표면 법선 추정, 접근 벡터 계산.
"""

import numpy as np
import cv2
import math
import pyrealsense2 as rs

from src.detection.constants import (
    DEPTH_SAMPLE_RADIUS, DEPTH_MIN_MM, DEPTH_MAX_MM, NORMAL_PATCH_SIZE
)


# =============================================================================
# 3D 좌표 계산
# =============================================================================

def get_depth_at_pixel(depth_image, cx, cy, radius=DEPTH_SAMPLE_RADIUS):
    """
    중심 픽셀 주변의 안정적인 depth 값을 구합니다.

    단일 픽셀 depth는 노이즈가 크므로, 주변 원형 영역의 중앙값을 사용합니다.
    (레퍼런스에서도 5~7 픽셀 반경 샘플링 사용)

    Args:
        depth_image: (H, W) depth 이미지 (단위: mm)
        cx, cy: 중심 픽셀 좌표
        radius: 샘플링 반경 (pixels)

    Returns:
        depth_mm: 중앙값 depth (mm), 유효값 없으면 0
    """
    h, w = depth_image.shape[:2]

    # 원형 마스크 생성
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius + 1)
    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius + 1)

    patch = depth_image[y1:y2, x1:x2]

    # 유효 depth만 사용
    valid = patch[(patch > DEPTH_MIN_MM) & (patch < DEPTH_MAX_MM)]

    if len(valid) == 0:
        return 0

    return int(np.median(valid))


def pixel_to_3d(cx, cy, depth_mm, intrinsics):
    """
    2D 픽셀 + depth → 3D 카메라 좌표 변환

    RealSense 카메라의 intrinsics(초점거리, 주점)를 사용하여
    depth 값을 실제 3D 좌표로 변환합니다.

    Args:
        cx, cy: 픽셀 좌표
        depth_mm: depth (mm)
        intrinsics: RealSense 카메라 내부 파라미터

    Returns:
        (X, Y, Z): 3D 좌표 (mm, 카메라 좌표계)
            X: 오른쪽 양수
            Y: 아래쪽 양수
            Z: 카메라에서 멀어지는 방향 양수
    """
    depth_m = depth_mm / 1000.0
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m)
    # m → mm 변환
    return (point_3d[0] * 1000.0, point_3d[1] * 1000.0, point_3d[2] * 1000.0)


# =============================================================================
# 표면 법선 & 자세 추정
# =============================================================================

def estimate_surface_normal(depth_image, cx, cy, intrinsics,
                            bbox=None, patch_size=NORMAL_PATCH_SIZE):
    """
    Depth 이미지에서 소켓 주변의 표면 법선(surface normal)을 추정합니다.

    bbox가 주어지면 bbox 단축의 50%를 패치 크기로 사용합니다.
    - 고정 15px: 너무 작아서 depth 노이즈에 ±15° 흔들림
    - bbox 전체: 너무 커서 바닥이 PCA를 지배 → 기울기 감지 불가
    - bbox 50%: 물체 중심에 집중하면서도 충분한 크기 → 안정적

    Args:
        depth_image: depth 이미지 (H, W) - mm
        cx, cy: 소켓 중심 픽셀
        intrinsics: 카메라 내부 파라미터
        bbox: (x1, y1, x2, y2) YOLO bbox — 패치 크기 결정에 사용
        patch_size: bbox 없을 때 사용할 패치 크기

    Returns:
        normal: (nx, ny, nz) 정규화된 법선 벡터
        roll, pitch, yaw: 오일러 각 (degrees)
    """
    h, w = depth_image.shape[:2]

    # bbox가 있으면 bbox 단축의 50%를 패치로 사용 (최소 20px)
    if bbox is not None:
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        half = max(10, int(min(bbox_w, bbox_h) * 0.25))
    else:
        half = patch_size // 2

    y1 = max(0, cy - half)
    y2 = min(h, cy + half + 1)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half + 1)

    # 2픽셀 간격 샘플링 (벡터화)
    ys = np.arange(y1, y2, 2)
    xs = np.arange(x1, x2, 2)
    grid_x, grid_y = np.meshgrid(xs, ys)
    px_flat = grid_x.ravel()
    py_flat = grid_y.ravel()

    # depth 값 한꺼번에 가져오기
    depths = depth_image[py_flat, px_flat].astype(np.float64)

    # 유효 depth 필터
    valid = (depths > DEPTH_MIN_MM) & (depths < DEPTH_MAX_MM)
    if valid.sum() < 10:
        return (0, 0, -1), 0.0, 0.0, 0.0, (0.0, 0.0, 0.0)

    # depth outlier 제거: 중앙값 ± 20mm 이내만 (바닥 vs 물체 분리)
    med_depth = np.median(depths[valid])
    valid = valid & (np.abs(depths - med_depth) < 20)
    if valid.sum() < 10:
        return (0, 0, -1), 0.0, 0.0, 0.0, (0.0, 0.0, 0.0)

    px_v = px_flat[valid].astype(np.float64)
    py_v = py_flat[valid].astype(np.float64)
    dv = depths[valid] / 1000.0  # mm → m

    # 배치 deprojection (intrinsics 수식 직접 적용 — 개별 호출 대비 ~50배 빠름)
    x_3d = (px_v - intrinsics.ppx) * dv / intrinsics.fx * 1000.0  # m → mm
    y_3d = (py_v - intrinsics.ppy) * dv / intrinsics.fy * 1000.0
    z_3d = dv * 1000.0

    points = np.column_stack([x_3d, y_3d, z_3d])

    # --- RANSAC 평면 피팅 (NumPy 기반) ---
    num_points = points.shape[0]
    best_inliers = np.ones(num_points, dtype=bool)

    if num_points >= 3:
        max_inliers = 0
        n_iters = min(100, num_points * 2)
        np.random.seed(42)
        distance_threshold = 3.0  # 평면 위아래 3mm 이내면 Inlier 인정

        for _ in range(n_iters):
            sample_idx = np.random.choice(num_points, 3, replace=False)
            p1, p2, p3 = points[sample_idx]

            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            n = n / norm

            distances = np.abs(np.dot(points - p1, n))
            inlier_mask = distances < distance_threshold
            inlier_count = np.sum(inlier_mask)

            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_inliers = inlier_mask

    # RANSAC으로 걸러진 Inlier들만 사용하여 최종 PCA (정교한 법선 계산)
    inlier_points = points[best_inliers] if np.sum(best_inliers) >= 3 else points
    centroid = inlier_points.mean(axis=0)
    centered = inlier_points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 가장 작은 고유값에 해당하는 고유벡터 = 법선
    normal = eigenvectors[:, 0]  # eigh는 오름차순 정렬

    # 법선이 카메라 방향(+Z)을 가리키도록 보정
    if normal[2] > 0:
        normal = -normal

    # 법선 → 오일러 각 변환
    # 법선 벡터가 (0, 0, -1)에서 얼마나 기울어졌는지
    roll = math.degrees(math.atan2(normal[1], -normal[2]))
    pitch = math.degrees(math.atan2(normal[0], -normal[2]))
    yaw = 0.0  # depth만으로는 Z축 회전을 알 수 없음

    # RANSAC으로 구한 평면 방정식을 이용해 2D 중심 픽셀(cx, cy)의 깊이 Z를 수학적으로 역산
    # Ray: X_ray = (cx - ppx)/fx, Y_ray = (cy - ppy)/fy, Z_ray = 1
    ray_x = (cx - intrinsics.ppx) / intrinsics.fx
    ray_y = (cy - intrinsics.ppy) / intrinsics.fy
    ray_vector = np.array([ray_x, ray_y, 1.0])
    
    dot_product = np.dot(normal, ray_vector)
    
    if abs(dot_product) > 1e-6:
        Z = np.dot(normal, centroid) / dot_product
        robust_pos_3d = (ray_x * Z, ray_y * Z, Z)
    else:
        # 평면과 카메라 광선이 수평일 경우 폴백 (물리적으로 극히 드묾)
        robust_pos_3d = tuple(centroid)

    return tuple(normal), roll, pitch, yaw, robust_pos_3d


def estimate_yaw_from_contour(color_image, depth_image, x1, y1, x2, y2):
    """
    Depth 마스크 + 윤곽선 분석으로 소켓의 yaw(Z축 회전)를 추정합니다.

    릴레이 소켓은 금속 재질이라 FAST+BRISK 특징점이 부족합니다.
    대신 depth를 이용하여 소켓 영역을 분리하고,
    minAreaRect로 최소 외접 회전 사각형의 각도를 구합니다.

    Args:
        color_image: 전체 컬러 이미지
        depth_image: 전체 depth 이미지 (mm)
        x1, y1, x2, y2: YOLO bbox

    Returns:
        yaw_deg: 추정된 yaw 각도 (degrees, -90~90)
    """
    # bbox 영역 크롭
    margin = 5
    h, w = depth_image.shape[:2]
    rx1 = max(0, x1 - margin)
    ry1 = max(0, y1 - margin)
    rx2 = min(w, x2 + margin)
    ry2 = min(h, y2 + margin)

    depth_roi = depth_image[ry1:ry2, rx1:rx2]
    color_roi = color_image[ry1:ry2, rx1:rx2]

    if depth_roi.size == 0 or color_roi.size == 0:
        return 0.0

    # 1) depth 기반 마스크: 유효 depth인 픽셀만 전경
    mask = np.zeros(depth_roi.shape, dtype=np.uint8)
    mask[(depth_roi > DEPTH_MIN_MM) & (depth_roi < DEPTH_MAX_MM)] = 255

    # 모폴로지로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 2) 윤곽선 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # depth 마스크 실패 시 컬러 에지로 fallback
        gray_roi = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    # 가장 큰 윤곽선 선택
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:  # 너무 작으면 무시
        return 0.0

    # 3) 최소 외접 회전 사각형
    rect = cv2.minAreaRect(largest)
    angle = rect[2]  # -90 ~ 0
    rect_w, rect_h = rect[1]

    # minAreaRect 각도를 사용 가능한 yaw로 변환
    # 긴 변이 기준 방향
    if rect_w < rect_h:
        angle = angle + 90

    return angle


# =============================================================================
# 접근 벡터 계산 (레퍼런스의 compute_vector 함수 대응)
# =============================================================================

def compute_approach_vector(x, y, z, roll, pitch, yaw, offset_mm=100):
    """
    소켓 위치와 자세로부터 로봇 접근 벡터를 계산합니다.

    레퍼런스의 compute_vector_100/120/150()을 통합 & 일반화한 버전.

    원리:
        1. 소켓의 표면 법선 방향으로 offset_mm만큼 위에 로봇을 위치
        2. 오일러 각 → 회전 행렬 → 법선 방향 오프셋 적용

    Args:
        x, y, z: 소켓 3D 좌표 (mm, 카메라 좌표계)
        roll, pitch, yaw: 소켓 자세 (degrees)
        offset_mm: 소켓 표면으로부터의 접근 거리 (mm)

    Returns:
        (ax, ay, az): 접근 위치 (mm)
        (rx, ry, rz): 로봇 자세 (degrees)
    """
    # 오일러 각 → 라디안
    rx = math.radians(roll)
    ry = math.radians(pitch)
    rz = math.radians(yaw)

    # 회전 행렬 (ZYX 순서)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)],
                   [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz), math.cos(rz), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx

    # 소켓 표면 법선 방향으로 offset만큼 위 (소켓 로컬 -Z 방향)
    offset_local = np.array([0, 0, -offset_mm])
    offset_world = R @ offset_local

    # 접근 위치
    approach_pos = np.array([x, y, z]) + offset_world

    # 로봇 자세 (레퍼런스 방식: roll+180, -(pitch-180), yaw+180)
    robot_rx = roll + 180
    robot_ry = -(pitch - 180)
    robot_rz = yaw + 180

    return (approach_pos[0], approach_pos[1], approach_pos[2]), (robot_rx, robot_ry, robot_rz)
