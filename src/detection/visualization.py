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


def _draw_3d_axes(display, roll, pitch, yaw, intrinsics, pos_3d,
                   axis_length_mm=30, normal=None):
    """소켓 중심에 3D 좌표축 (X=빨강, Y=초록, Z=파랑)을 그립니다.

    Z축 = 표면 법선 방향 (depth point cloud의 2D 평면에 수직)
    X, Y축 = 표면 평면 위의 두 방향

    법선 벡터가 주어지면 직접 회전 행렬을 구성하여
    Z축이 정확히 표면 법선 방향을 가리킵니다.

    Args:
        display: 그릴 이미지
        roll, pitch, yaw: 자세 (degrees) — 법선이 없을 때 폴백
        intrinsics: RealSense 카메라 intrinsics
        pos_3d: (X, Y, Z) mm 소켓 3D 위치
        axis_length_mm: 축 길이 (mm)
        normal: (nx, ny, nz) 표면 법선 벡터 (있으면 이것으로 Z축 결정)
    """
    if intrinsics is None or pos_3d is None:
        return

    # 카메라 행렬
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)

    if normal is not None:
        # 법선 벡터로 직접 회전 행렬 구성
        # Z축 = 법선 방향 (표면에 수직, 카메라 쪽을 향함)
        z_axis = np.array(normal, dtype=np.float64)
        z_norm = np.linalg.norm(z_axis)
        if z_norm > 0:
            z_axis = z_axis / z_norm

        # X축: Z축과 카메라 Y축(아래 방향)의 외적
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(z_axis, up)) > 0.99:
            # Z축이 거의 Y축과 평행하면 다른 축 사용
            up = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Y축: Z축과 X축의 외적 (오른손 좌표계)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Yaw 회전 적용 (Z축 기준 회전)
        if yaw != 0.0:
            rz = math.radians(yaw)
            cos_rz, sin_rz = math.cos(rz), math.sin(rz)
            x_rot = cos_rz * x_axis + sin_rz * y_axis
            y_rot = -sin_rz * x_axis + cos_rz * y_axis
            x_axis = x_rot
            y_axis = y_rot

        R = np.column_stack([x_axis, y_axis, z_axis])
    else:
        # 폴백: Euler 각으로 회전 행렬 구성
        rx = math.radians(roll)
        ry = math.radians(pitch)
        rz = math.radians(yaw)

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

    # 회전 행렬 → Rodrigues 벡터
    rvec, _ = cv2.Rodrigues(R)

    # 소켓 3D 위치 (mm → m)
    tvec = np.array([[pos_3d[0] / 1000.0],
                     [pos_3d[1] / 1000.0],
                     [pos_3d[2] / 1000.0]], dtype=np.float64)

    # drawFrameAxes로 XYZ 좌표축 그리기
    axis_m = axis_length_mm / 1000.0
    cv2.drawFrameAxes(display, camera_matrix, dist_coeffs,
                      rvec, tvec, axis_m, thickness=2)

    # 축 끝에 라벨 표시
    # 각 축의 끝점을 3D → 2D 투영
    axes_3d = np.float32([
        [axis_m, 0, 0],   # X축 끝
        [0, axis_m, 0],   # Y축 끝
        [0, 0, axis_m],   # Z축 끝
    ])
    pts_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, camera_matrix, dist_coeffs)
    labels = [("X", (0, 0, 255)), ("Y", (0, 255, 0)), ("Z", (255, 0, 0))]

    for pt, (name, lcolor) in zip(pts_2d, labels):
        px, py = int(pt[0][0]), int(pt[0][1])
        cv2.putText(display, name, (px + 3, py - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, lcolor, 2)


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

        # bbox
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

    # === 4) 측면 뷰 (XZ 평면) — 하단 패널 ===
    side_h = 120
    side_panel = np.zeros((side_h, w, 3), dtype=np.uint8)
    side_panel[:] = (40, 40, 40)

    cv2.putText(side_panel, "Side View (XZ)", (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if objects:
        # XZ 범위 계산
        xs = [obj.pos_3d[0] for obj in objects if obj.pos_3d]
        zs = [obj.pos_3d[2] for obj in objects if obj.pos_3d]
        if xs and zs:
            x_min, x_max = min(xs) - 50, max(xs) + 50
            z_min, z_max = min(zs) - 30, max(zs) + 30
            x_range = max(x_max - x_min, 1)
            z_range = max(z_max - z_min, 1)

            # 그리드
            for gz in range(int(z_min), int(z_max), 50):
                sy = int(20 + (gz - z_min) / z_range * (side_h - 30))
                if 20 <= sy < side_h:
                    cv2.line(side_panel, (60, sy), (w - 10, sy), (60, 60, 60), 1)
                    cv2.putText(side_panel, f"{gz:.0f}", (5, sy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

            # 소켓 위치
            pick_target = select_pick_target(objects)
            for obj in objects:
                if not obj.pos_3d:
                    continue
                sx = int(60 + (obj.pos_3d[0] - x_min) / x_range * (w - 70))
                sy = int(20 + (obj.pos_3d[2] - z_min) / z_range * (side_h - 30))
                color = COLORS.get(obj.class_id, (0, 255, 0))
                radius = 8 if obj is pick_target else 5
                cv2.circle(side_panel, (sx, sy), radius, color, -1)
                
                # RANSAC 평면 시각화 (XZ 단면선 노란색으로 표시)
                if hasattr(obj, 'normal') and obj.normal:
                    nx, ny, nz = obj.normal
                    # 반경 20mm 넓이의 RANSAC 가상 평면 선 그리기
                    ratio_x = (w - 70) / x_range
                    ratio_z = (side_h - 30) / z_range
                    dir_x_mm = -nz * 20 
                    dir_z_mm = nx * 20
                    dx = int(dir_x_mm * ratio_x)
                    dz = int(dir_z_mm * ratio_z)
                    cv2.line(side_panel, (sx - dx, sy - dz), (sx + dx, sy + dz), (0, 255, 255), 2)
                    
                cv2.putText(side_panel, f"{obj.class_name}", (sx + 10, sy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                cv2.putText(side_panel, f"Z={obj.pos_3d[2]:.0f}", (sx + 10, sy + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

            # 축 라벨
            cv2.putText(side_panel, "X->", (w // 2, side_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            cv2.putText(side_panel, "Z(depth)", (5, side_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    # 합치기
    depth_panel = np.vstack([colormap, side_panel])

    return depth_panel


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

        # 바운딩 박스 (픽킹 대상은 두꺼운 테두리)
        thickness = 3 if is_target else 2
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
        if obj.pos_3d and obj.orientation and intrinsics:
            roll, pitch, yaw = obj.orientation
            _draw_3d_axes(display, roll, pitch, yaw,
                          intrinsics, obj.pos_3d, axis_length_mm=30,
                          normal=obj.normal)

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

    # 상단 정보
    cv2.putText(display, f"Detected: {len(objects)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 좌측 하단에 좌표계 범례
    legend_x, legend_y = 10, image.shape[0] - 60
    cv2.putText(display, "Axes:", (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.arrowedLine(display, (legend_x + 50, legend_y), (legend_x + 90, legend_y),
                    (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(display, "X", (legend_x + 92, legend_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.arrowedLine(display, (legend_x + 50, legend_y), (legend_x + 50, legend_y - 40),
                    (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(display, "Y", (legend_x + 42, legend_y - 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(display, "Z(depth)", (legend_x + 100, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

    # 3D 좌표 요약 패널
    if objects:
        panel_y = 60
        cv2.putText(display, "--- 3D Coordinates ---", (10, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        for i, obj in enumerate(objects):
            if obj.pos_3d:
                panel_y += 18
                marker = ">>" if obj is pick_target else "  "
                txt = (f"{marker} [{i}] {obj.class_name}: "
                       f"X={obj.pos_3d[0]:.1f} Y={obj.pos_3d[1]:.1f} Z={obj.pos_3d[2]:.1f}mm")
                text_color = (0, 255, 0) if obj is pick_target else (200, 200, 200)
                cv2.putText(display, txt, (10, panel_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)

    return display
