"""
시각화 함수
===========

3D 좌표축, depth 분석, 탐지 결과 오버레이 등 시각화 기능.
"""

import math
import numpy as np
import cv2

from src.detection.constants import (
    DEPTH_MIN_MM, DEPTH_MAX_MM, CLASS_NAMES, COLORS
)
from src.detection.tracker import select_pick_target


def _draw_3d_axes(display, intrinsics, pos_3d, axis_length_mm=30,
                   normal=None, obb_corners=None):
    """소켓 중심에 3D 좌표축 (X=빨강, Y=초록, Z=파랑)을 그립니다.

    X축 = OBB width 방향 (corners에서 직접 추출)
    Y축 = OBB height 방향 (corners에서 직접 추출)
    Z축 = 표면 법선 방향 (3D→2D 투영)

    Args:
        display: 그릴 이미지
        intrinsics: RealSense 카메라 intrinsics
        pos_3d: (X, Y, Z) mm 소켓 3D 위치
        axis_length_mm: 축 길이 (mm)
        normal: (nx, ny, nz) 표면 법선 벡터
        obb_corners: OBB 꼭짓점 (4, 2) ndarray
    """
    if intrinsics is None or pos_3d is None or pos_3d[2] <= 0:
        return

    h_img, w_img = display.shape[:2]
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)

    # depth에 비례한 축 픽셀 길이 (일정한 물리적 크기)
    axis_px = axis_length_mm * intrinsics.fx / pos_3d[2]

    if obb_corners is not None:
        # OBB 중심 (pixel)
        center = obb_corners.mean(axis=0)
        origin = (int(center[0]), int(center[1]))

        # X축 (width 방향): corner[0]-corner[3] = 2*vec1 (YOLO OBB 규약)
        width_dir = obb_corners[0] - obb_corners[3]
        w_len = np.linalg.norm(width_dir)

        # Y축 (height 방향): corner[0]-corner[1] = 2*vec2
        height_dir = obb_corners[0] - obb_corners[1]
        h_len = np.linalg.norm(height_dir)

        axes = []  # (end_px, end_py, color, label)

        if w_len > 0:
            x_end = center + width_dir / w_len * axis_px
            axes.append((int(x_end[0]), int(x_end[1]), (0, 0, 255), "X"))

        if h_len > 0:
            y_end = center + height_dir / h_len * axis_px
            axes.append((int(y_end[0]), int(y_end[1]), (0, 255, 0), "Y"))

        # Z축: 법선 방향을 3D→2D 투영하여 방향 벡터 획득
        if normal is not None:
            n = np.array(normal, dtype=np.float64)
            n_norm = np.linalg.norm(n)
            if n_norm > 0:
                n = n / n_norm
                p1 = np.array(pos_3d, dtype=np.float64) / 1000.0
                p2 = p1 + n * axis_length_mm / 1000.0
                pts_3d = np.float32([list(p1), list(p2)])
                rvec_zero = np.zeros(3, dtype=np.float64)
                tvec_zero = np.zeros(3, dtype=np.float64)
                pts_2d, _ = cv2.projectPoints(
                    pts_3d, rvec_zero, tvec_zero, camera_matrix, dist_coeffs
                )
                z_dir = pts_2d[1][0] - pts_2d[0][0]
                z_end = center + z_dir
                axes.append((int(z_end[0]), int(z_end[1]), (255, 0, 0), "Z"))

        # 그리기
        for px, py, color, label in axes:
            if (-w_img < px < 2 * w_img and -h_img < py < 2 * h_img):
                cv2.line(display, origin, (px, py), color, 2)
                if 0 <= px < w_img and 0 <= py < h_img:
                    cv2.putText(display, label, (px + 3, py - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    else:
        # 폴백: OBB corners 없을 때 — 3D 좌표축만 표시
        if normal is not None:
            p_center = np.array(pos_3d, dtype=np.float64) / 1000.0
            rvec_zero = np.zeros(3, dtype=np.float64)
            tvec_zero = np.zeros(3, dtype=np.float64)
            pts_2d, _ = cv2.projectPoints(
                np.float32([list(p_center)]), rvec_zero, tvec_zero,
                camera_matrix, dist_coeffs
            )
            origin = (int(pts_2d[0][0][0]), int(pts_2d[0][0][1]))
            cv2.circle(display, origin, 4, (255, 0, 0), -1)


def draw_depth_analysis(depth_image, objects):
    """Depth 정보를 분석하기 쉬운 형태로 시각화합니다.

    표시 내용:
    - 거리별 컬러맵 + mm 단위 스케일바
    - 탐지된 소켓 위치에 depth 값 표시
    - 상단: depth 히스토그램 (거리 분포)
    - 하단: 측면 뷰 (XZ 평면) — 소켓들의 위치를 위에서 본 시점으로

    Args:
        depth_image: (H, W) depth 이미지 (mm)
        objects: DetectedObject 리스트

    Returns:
        depth_panel: 분석 이미지
    """
    h, w = depth_image.shape[:2]

    # === 1) 컬러맵 (유효 범위만 하이라이트) ===
    # 유효 depth 범위를 0~255로 정규화
    valid_mask = (depth_image > DEPTH_MIN_MM) & (depth_image < DEPTH_MAX_MM)
    depth_norm = np.zeros_like(depth_image, dtype=np.uint8)
    if valid_mask.any():
        d_min = depth_image[valid_mask].min()
        d_max = depth_image[valid_mask].max()
        d_range = max(d_max - d_min, 1)
        depth_norm[valid_mask] = (
            (depth_image[valid_mask].astype(float) - d_min) / d_range * 255
        ).astype(np.uint8)

    colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    # 유효하지 않은 영역은 어둡게
    colormap[~valid_mask] = (30, 30, 30)

    # === 2) 스케일바 (오른쪽 세로, 벡터화) ===
    bar_w = 30
    bar_x = w - bar_w - 10
    if valid_mask.any():
        bar_height = h - 80
        gradient = np.linspace(0, 255, bar_height).astype(np.uint8)
        gradient_img = gradient.reshape(-1, 1)
        gradient_color = cv2.applyColorMap(gradient_img, cv2.COLORMAP_TURBO)
        # 스케일바 영역에 한번에 복사
        bar_strip = np.repeat(gradient_color, bar_w, axis=1)
        colormap[50:50 + bar_height, bar_x:bar_x + bar_w] = bar_strip

        # 스케일바 눈금 (5단계)
        for i in range(6):
            ratio = i / 5.0
            y_pos = int(50 + ratio * (h - 80))
            val = d_min + ratio * d_range
            cv2.line(colormap, (bar_x - 5, y_pos), (bar_x, y_pos), (255, 255, 255), 1)
            cv2.putText(colormap, f"{val:.0f}", (bar_x - 50, y_pos + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.putText(colormap, "mm", (bar_x - 10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # === 3) 탐지된 소켓에 depth 값 표시 ===
    for obj in objects:
        x1, y1, x2, y2 = obj.bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        color = COLORS.get(obj.class_id, (0, 255, 0))

        # bbox (OBB 꼭짓점이 있으면 회전 박스, 없으면 축 정렬 박스)
        if obj.obb_corners is not None:
            corners = obj.obb_corners.astype(int)
            cv2.polylines(colormap, [corners], isClosed=True, color=color, thickness=1)
        else:
            cv2.rectangle(colormap, (x1, y1), (x2, y2), color, 1)
        # 중심 십자
        cv2.drawMarker(colormap, (cx, cy), (255, 255, 255),
                       cv2.MARKER_CROSS, 10, 1)

        # depth 값
        if obj.pos_3d:
            z_text = f"{obj.pos_3d[2]:.0f}mm"
            cv2.putText(colormap, z_text, (cx + 8, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(colormap, z_text, (cx + 8, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(colormap, "Depth Map", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return colormap


def draw_3d_detections(image, objects, intrinsics=None, depth_colormap=None):
    """
    3D 탐지 결과를 이미지 위에 시각화합니다.

    각 소켓마다 표시:
    - 바운딩 박스 + 클래스 + confidence
    - 3D 좌표축 (X=빨강, Y=초록, Z=파랑) — 자세에 따라 회전
    - 3D 좌표 (X, Y, Z mm) 텍스트
    - 자세 (Roll, Pitch, Yaw deg)
    - 픽킹 대상 하이라이트

    Args:
        image: 원본 컬러 이미지
        objects: DetectedObject 리스트
        intrinsics: RealSense 카메라 intrinsics (좌표축 그리기용)
        depth_colormap: depth 시각화 (optional)

    Returns:
        display: 시각화된 이미지
    """
    display = image.copy()

    # 픽킹 대상 선택 (하이라이트용)
    pick_target = select_pick_target(objects)

    for i, obj in enumerate(objects):
        x1, y1, x2, y2 = obj.bbox
        color = COLORS.get(obj.class_id, (0, 255, 0))
        is_target = (obj is pick_target)
        thickness = 3 if is_target else 2

        # OBB 꼭짓점이 있으면 회전 바운딩 박스, 없으면 축 정렬 박스
        if obj.obb_corners is not None:
            corners = obj.obb_corners.astype(int)
            cv2.polylines(display, [corners], isClosed=True, color=color, thickness=thickness)
        else:
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

        # 픽킹 대상 표시
        if is_target:
            cv2.putText(display, "PICK", (x1, y1 - 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 중심점
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(display, (cx, cy), 5, (0, 255, 255), -1)

        # 클래스 + confidence (bbox 위)
        label = f"{obj.class_name} {obj.confidence:.0%}"
        cv2.putText(display, label, (x1, y1 - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # === 3D 좌표축 그리기 (X=빨강, Y=초록, Z=파랑) ===
        if obj.pos_3d and intrinsics:
            _draw_3d_axes(display, intrinsics, obj.pos_3d,
                          axis_length_mm=30, normal=obj.normal,
                          obb_corners=obj.obb_corners)

        # === 3D 좌표 (XYZ mm) 텍스트 ===
        if obj.pos_3d:
            x_mm, y_mm, z_mm = obj.pos_3d
            xyz_text = f"X:{x_mm:.1f} Y:{y_mm:.1f} Z:{z_mm:.1f} mm"
            (tw, th), _ = cv2.getTextSize(xyz_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(display, (x1, y1 - 14), (x1 + tw + 4, y1), (0, 0, 0), -1)
            cv2.putText(display, xyz_text, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # === 자세 (Roll, Pitch, Yaw) — bbox 아래 ===
        if obj.orientation:
            roll, pitch, yaw = obj.orientation
            ori_text = f"R:{roll:.1f} P:{pitch:.1f} Y:{yaw:.1f} deg"
            cv2.putText(display, ori_text, (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    # 상단 정보 — y=20 라인은 binpicking_3d.py의 FPS/Pipeline용으로 비워둠
    cv2.putText(display, f"Detected: {len(objects)}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    return display
