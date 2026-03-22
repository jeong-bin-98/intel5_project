"""
FAST + BRISK 기반 3D 소켓 자세 추정
=====================================

레퍼런스 논문의 Phase 2에 해당하는 기능을 독립 모듈로 구현.
2D 특징점 매칭 결과를 RealSense depth와 결합하여 3D 좌표+자세를 출력합니다.

파이프라인:
    1. FAST로 키포인트(코너) 검출
    2. BRISK로 디스크립터 계산
    3. BFMatcher로 템플릿-쿼리 매칭
    4. Homography → 2D 자세 (yaw)
    5. 매칭된 인라이어 키포인트 + depth → 3D 점군
    6. 3D 점군에 PCA 평면 피팅 → 표면 법선 (roll, pitch)
    7. 최종 출력: 3D 위치 (X, Y, Z mm) + 자세 (roll, pitch, yaw deg)

사용법:
    from src.detection.fast_brisk_pose import FastBriskPoseEstimator

    estimator = FastBriskPoseEstimator()
    estimator.load_templates("config/templates")

    # 2D 매칭만 (depth 없이)
    result = estimator.estimate_pose(cropped_image, class_id=0)

    # 3D 자세 추정 (depth + intrinsics 결합)
    result = estimator.estimate_pose_3d(
        color_image, depth_image, bbox, intrinsics, class_id=0
    )
    if result:
        pos_3d = result['pos_3d']       # (X, Y, Z) mm
        roll   = result['roll']          # degrees
        pitch  = result['pitch']         # degrees
        yaw    = result['yaw']           # degrees
"""

import os
import cv2
import numpy as np
import math
import pyrealsense2 as rs
from typing import Optional


# depth 유효 범위
DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 1000


class FastBriskPoseEstimator:
    """FAST 키포인트 + BRISK 디스크립터 기반 3D 자세 추정기.

    2D 특징점 매칭으로 yaw를, depth 기반 평면 피팅으로 roll/pitch를 추정합니다.
    """

    def __init__(self, fast_threshold=25, brisk_threshold=30,
                 min_matches=8, match_ratio=0.75):
        """
        Args:
            fast_threshold: FAST 코너 검출 임계값 (높을수록 엄격)
            brisk_threshold: BRISK 디스크립터 임계값
            min_matches: 유효한 매칭으로 간주할 최소 매칭 수
            match_ratio: Lowe's ratio test 비율 (낮을수록 엄격)
        """
        self.fast = cv2.FastFeatureDetector_create(
            threshold=fast_threshold,
            nonmaxSuppression=True,
            type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
        )
        self.brisk = cv2.BRISK_create(thresh=brisk_threshold)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.min_matches = min_matches
        self.match_ratio = match_ratio

        # 템플릿: {class_id: [{'keypoints', 'descriptors', 'image', 'name'}, ...]}
        self.templates = {}

    # =========================================================================
    # 템플릿 관리
    # =========================================================================

    def load_templates(self, template_dir):
        """템플릿 이미지를 로드하고 특징점을 미리 계산합니다.

        폴더 구조:
            template_dir/
                8pin/    (class_id=0)
                12pin/   (class_id=1)
        """
        class_map = {"8pin": 0, "12pin": 1}
        total = 0

        for class_name, class_id in class_map.items():
            class_dir = os.path.join(template_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            self.templates[class_id] = []

            for fname in os.listdir(class_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue

                path = os.path.join(class_dir, fname)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                kp, des = self._detect_and_compute(img)
                if des is not None and len(kp) >= 4:
                    self.templates[class_id].append({
                        'keypoints': kp,
                        'descriptors': des,
                        'image': img,
                        'name': fname,
                        'path': path
                    })
                    total += 1
                    print(f"  템플릿 로드: {class_name}/{fname} ({len(kp)} keypoints)")

        print(f"총 {total}개 템플릿 로드 완료")
        return total

    def register_template_from_image(self, image, class_id, name="manual"):
        """이미지를 직접 템플릿으로 등록합니다 (파일 없이)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        kp, des = self._detect_and_compute(gray)
        if des is None or len(kp) < 4:
            return False

        if class_id not in self.templates:
            self.templates[class_id] = []

        self.templates[class_id].append({
            'keypoints': kp,
            'descriptors': des,
            'image': gray,
            'name': name,
            'path': None
        })
        return True

    # =========================================================================
    # 특징점 검출/매칭
    # =========================================================================

    def _detect_and_compute(self, gray_image):
        """FAST 키포인트 검출 + BRISK 디스크립터 계산."""
        keypoints = self.fast.detect(gray_image, None)
        if len(keypoints) == 0:
            return [], None
        keypoints, descriptors = self.brisk.compute(gray_image, keypoints)
        return keypoints, descriptors

    def _match_template(self, template, q_kp, q_des, query_shape):
        """단일 템플릿과 매칭을 수행합니다."""
        t_kp = template['keypoints']
        t_des = template['descriptors']

        if t_des is None or len(t_kp) < 4:
            return None

        try:
            matches = self.matcher.knnMatch(t_des, q_des, k=2)
        except cv2.error:
            return None

        # Lowe's ratio test
        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.min_matches:
            return None

        src_pts = np.float32(
            [t_kp[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [q_kp[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        inlier_mask = mask.ravel().tolist()
        num_inliers = sum(inlier_mask)

        if num_inliers < self.min_matches:
            return None

        angle, scale, center = self._decompose_homography(
            H, template['image'].shape, query_shape
        )

        confidence = num_inliers / len(good_matches) if good_matches else 0

        return {
            'angle': angle,
            'scale': scale,
            'center': center,
            'confidence': confidence,
            'num_matches': len(good_matches),
            'num_inliers': num_inliers,
            'homography': H,
            'good_matches': good_matches,
            'inlier_mask': inlier_mask,
            'src_pts': src_pts,
            'dst_pts': dst_pts,
        }

    def _decompose_homography(self, H, template_shape, query_shape):
        """호모그래피 → 회전각, 스케일, 중심."""
        th, tw = template_shape[:2]

        corners = np.float32([
            [0, 0], [tw, 0], [tw, th], [0, th]
        ]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)
        tc = transformed.reshape(-1, 2)

        center = tc.mean(axis=0)

        dx = tc[1][0] - tc[0][0]
        dy = tc[1][1] - tc[0][1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360

        side1 = np.linalg.norm(tc[1] - tc[0])
        side2 = np.linalg.norm(tc[2] - tc[1])
        scale = math.sqrt(side1 * side2) / math.sqrt(tw * th)

        return angle, scale, tuple(center)

    # =========================================================================
    # 2D 자세 추정 (depth 없이)
    # =========================================================================

    def estimate_pose(self, query_image, class_id=None):
        """2D 자세 추정 (기존 호환). depth 없이 yaw 각도만 반환."""
        if len(query_image.shape) == 3:
            gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = query_image.copy()

        q_kp, q_des = self._detect_and_compute(gray)
        if q_des is None or len(q_kp) < 4:
            return None

        if class_id is not None:
            classes_to_check = [class_id] if class_id in self.templates else []
        else:
            classes_to_check = list(self.templates.keys())

        best_result = None
        best_inliers = 0

        for cid in classes_to_check:
            for tmpl in self.templates[cid]:
                result = self._match_template(tmpl, q_kp, q_des, gray.shape)
                if result and result['num_inliers'] > best_inliers:
                    best_result = result
                    best_result['class_id'] = cid
                    best_result['template_name'] = tmpl['name']
                    best_inliers = result['num_inliers']

        return best_result

    # =========================================================================
    # 3D 자세 추정 (depth + intrinsics 결합)
    # =========================================================================

    def estimate_pose_3d(self, color_image, depth_image, bbox, intrinsics,
                         class_id=None):
        """3D 자세 추정: FAST+BRISK 매칭 + depth → 3D 좌표 + 자세.

        파이프라인:
            1. YOLO bbox → ROI 크롭
            2. ROI에서 FAST+BRISK 매칭 → 인라이어 키포인트 + yaw
            3. 인라이어 키포인트를 원본 이미지 좌표로 변환
            4. 각 키포인트의 depth → rs2_deproject_pixel_to_point → 3D
            5. 3D 인라이어 점군에 PCA 평면 피팅 → 표면 법선 → roll, pitch
            6. 오브젝트 중심(bbox center)의 depth → 3D 위치

        Args:
            color_image: 전체 컬러 이미지 (H, W, 3)
            depth_image: 전체 depth 이미지 (H, W) - mm
            bbox: (x1, y1, x2, y2) YOLO 바운딩 박스
            intrinsics: RealSense 카메라 내부 파라미터
            class_id: 매칭할 클래스 (None이면 전체)

        Returns:
            dict 또는 None:
                pos_3d: (X, Y, Z) mm — 카메라 좌표계 3D 위치
                roll: degrees — 표면 기울기 (X축 회전)
                pitch: degrees — 표면 기울기 (Y축 회전)
                yaw: degrees — 평면 회전 (Z축, homography에서)
                normal: (nx, ny, nz) — 표면 법선 벡터
                keypoints_3d: Nx3 array — 매칭된 인라이어의 3D 좌표
                confidence: 매칭 신뢰도
                num_inliers: 인라이어 수
                homography: 3x3 행렬
                center_px: (cx, cy) — 원본 이미지 기준 중심 픽셀
        """
        x1, y1, x2, y2 = bbox

        # 1) ROI 크롭 + 2D 매칭
        roi, (ox, oy) = extract_roi(color_image, bbox, margin=15)
        match_2d = self.estimate_pose(roi, class_id=class_id)

        if match_2d is None:
            return None

        # 2) 인라이어 키포인트를 원본 이미지 좌표로 변환
        inlier_pixels = []
        dst_pts = match_2d['dst_pts']
        inlier_mask = match_2d['inlier_mask']

        for idx, is_inlier in enumerate(inlier_mask):
            if is_inlier:
                # ROI 내 좌표 → 원본 이미지 좌표
                px = int(dst_pts[idx][0][0] + ox)
                py = int(dst_pts[idx][0][1] + oy)
                inlier_pixels.append((px, py))

        # 3) 인라이어 키포인트 → 3D 점군
        h, w = depth_image.shape[:2]
        keypoints_3d = []

        for px, py in inlier_pixels:
            if 0 <= px < w and 0 <= py < h:
                d = depth_image[py, px]
                if DEPTH_MIN_MM < d < DEPTH_MAX_MM:
                    pt = _pixel_to_3d(px, py, d, intrinsics)
                    keypoints_3d.append(pt)

        # 최소 점 수 확인
        if len(keypoints_3d) < 4:
            # 인라이어 3D 점이 부족하면 bbox 중심 주변으로 보충
            cx_px = (x1 + x2) // 2
            cy_px = (y1 + y2) // 2
            keypoints_3d = _sample_3d_points(
                depth_image, cx_px, cy_px, intrinsics, radius=15
            )
            if len(keypoints_3d) < 4:
                return None

        kp3d = np.array(keypoints_3d)

        # 4) PCA 평면 피팅 → 표면 법선 → roll, pitch
        normal, roll, pitch = _fit_plane_pca(kp3d)

        # 5) yaw는 homography에서 (2D 평면 회전)
        yaw = match_2d['angle']

        # 6) 오브젝트 중심의 3D 위치 (bbox center + depth)
        cx_px = (x1 + x2) // 2
        cy_px = (y1 + y2) // 2
        center_depth = _robust_depth(depth_image, cx_px, cy_px, radius=5)

        if center_depth == 0:
            # bbox center의 depth가 없으면 인라이어 3D 점의 중심 사용
            pos_3d = tuple(kp3d.mean(axis=0))
        else:
            pos_3d = _pixel_to_3d(cx_px, cy_px, center_depth, intrinsics)

        return {
            'pos_3d': pos_3d,              # (X, Y, Z) mm
            'roll': roll,                   # degrees
            'pitch': pitch,                 # degrees
            'yaw': yaw,                     # degrees
            'normal': tuple(normal),        # (nx, ny, nz)
            'keypoints_3d': kp3d,           # Nx3 array (mm)
            'confidence': match_2d['confidence'],
            'num_inliers': match_2d['num_inliers'],
            'num_matches': match_2d['num_matches'],
            'homography': match_2d['homography'],
            'center_px': (cx_px, cy_px),
            'class_id': match_2d.get('class_id'),
            'template_name': match_2d.get('template_name'),
            # 2D 결과도 보존 (시각화용)
            '_match_2d': match_2d,
        }

    # =========================================================================
    # 시각화
    # =========================================================================

    def draw_matches(self, template_image, query_image, result,
                     draw_homography=True):
        """매칭 결과를 시각화합니다."""
        if result is None:
            return query_image.copy()

        # 3D 결과면 내부 2D 결과 사용
        match_2d = result.get('_match_2d', result)

        if len(template_image.shape) == 2:
            t_color = cv2.cvtColor(template_image, cv2.COLOR_GRAY2BGR)
        else:
            t_color = template_image.copy()

        if len(query_image.shape) == 2:
            q_color = cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR)
        else:
            q_color = query_image.copy()

        matches_mask = [[1, 0] if m else [0, 0]
                        for m in match_2d['inlier_mask']]

        good = match_2d['good_matches']
        knn_matches = [[m] for m in good]

        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matches_mask,
            flags=cv2.DrawMatchesFlags_DEFAULT
        )

        display = cv2.drawMatchesKnn(
            t_color, [cv2.KeyPoint(p[0][0], p[0][1], 5) for p in match_2d['src_pts']],
            q_color, [cv2.KeyPoint(p[0][0], p[0][1], 5) for p in match_2d['dst_pts']],
            knn_matches, None, **draw_params
        )

        if draw_homography and match_2d.get('homography') is not None:
            H = match_2d['homography']
            th, tw = template_image.shape[:2]
            corners = np.float32([
                [0, 0], [tw, 0], [tw, th], [0, th]
            ]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, H)

            offset_x = t_color.shape[1]
            transformed_shifted = transformed + np.array([offset_x, 0], dtype=np.float32)
            cv2.polylines(display,
                          [np.int32(transformed_shifted)],
                          True, (0, 255, 255), 3)

        # 정보 텍스트
        if 'pos_3d' in result:
            p = result['pos_3d']
            info = (f"3D: X={p[0]:.0f} Y={p[1]:.0f} Z={p[2]:.0f}mm | "
                    f"R={result['roll']:.1f} P={result['pitch']:.1f} Y={result['yaw']:.1f} | "
                    f"Inliers:{result['num_inliers']}")
        else:
            info = (f"Angle: {match_2d['angle']:.1f}deg  "
                    f"Matches: {match_2d['num_inliers']}/{match_2d['num_matches']}  "
                    f"Conf: {match_2d['confidence']:.0%}")
        cv2.putText(display, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        return display

    def draw_pose_on_image(self, image, bbox, result):
        """원본 이미지 위에 3D 자세 추정 결과를 표시합니다."""
        if result is None:
            return image

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # yaw (2D 결과이든 3D 결과이든)
        yaw = result.get('yaw', result.get('angle', 0))

        # yaw 방향 화살표
        arrow_len = min(x2 - x1, y2 - y1) // 2
        ax = int(cx + arrow_len * math.cos(math.radians(yaw)))
        ay = int(cy + arrow_len * math.sin(math.radians(yaw)))
        cv2.arrowedLine(image, (cx, cy), (ax, ay), (0, 255, 255), 2, tipLength=0.3)

        # 법선 방향 화살표 (3D 결과일 때)
        if 'normal' in result:
            nx, ny, nz = result['normal']
            # 법선의 xy 성분을 2D 화살표로 투영
            n_len = 25
            ndx = int(n_len * nx)
            ndy = int(n_len * ny)
            cv2.arrowedLine(image, (cx, cy), (cx + ndx, cy + ndy),
                            (255, 0, 255), 2, tipLength=0.4)

        # 텍스트
        if 'pos_3d' in result:
            p = result['pos_3d']
            text1 = f"X:{p[0]:.0f} Y:{p[1]:.0f} Z:{p[2]:.0f}mm"
            text2 = f"R:{result['roll']:.1f} P:{result['pitch']:.1f} Y:{yaw:.1f}"
            cv2.putText(image, text1, (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(image, text2, (x1, y2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            text = f"Yaw:{yaw:.1f}"
            cv2.putText(image, text, (x1, y2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return image


# =============================================================================
# 3D 유틸리티 함수들
# =============================================================================

def _pixel_to_3d(px, py, depth_mm, intrinsics):
    """2D 픽셀 + depth → 3D 카메라 좌표 (mm).

    Args:
        px, py: 픽셀 좌표
        depth_mm: depth 값 (mm)
        intrinsics: RealSense intrinsics

    Returns:
        (X, Y, Z) mm
    """
    depth_m = depth_mm / 1000.0
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth_m)
    return (point[0] * 1000.0, point[1] * 1000.0, point[2] * 1000.0)


def _robust_depth(depth_image, cx, cy, radius=5):
    """중심 주변 원형 영역의 중앙값 depth (mm).

    Args:
        depth_image: (H, W) depth 이미지
        cx, cy: 중심 픽셀
        radius: 샘플링 반경

    Returns:
        중앙값 depth (mm), 실패 시 0
    """
    h, w = depth_image.shape[:2]
    depths = []

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            py, px = cy + dy, cx + dx
            if 0 <= py < h and 0 <= px < w:
                d = depth_image[py, px]
                if DEPTH_MIN_MM < d < DEPTH_MAX_MM:
                    depths.append(d)

    return int(np.median(depths)) if depths else 0


def _sample_3d_points(depth_image, cx, cy, intrinsics, radius=15):
    """중심 주변에서 3D 점군을 샘플링합니다.

    인라이어 3D 점이 부족할 때 fallback으로 사용합니다.

    Args:
        depth_image: (H, W) depth
        cx, cy: 중심 픽셀
        intrinsics: RealSense intrinsics
        radius: 샘플링 반경

    Returns:
        list of (X, Y, Z) mm
    """
    h, w = depth_image.shape[:2]
    points = []

    for dy in range(-radius, radius + 1, 2):
        for dx in range(-radius, radius + 1, 2):
            py, px = cy + dy, cx + dx
            if 0 <= py < h and 0 <= px < w:
                d = depth_image[py, px]
                if DEPTH_MIN_MM < d < DEPTH_MAX_MM:
                    points.append(_pixel_to_3d(px, py, d, intrinsics))

    return points


def _fit_plane_pca(points_3d):
    """3D 점군에 PCA 평면 피팅 → 표면 법선 + roll/pitch.

    매칭된 인라이어 키포인트의 3D 좌표로 소켓 표면 평면을 추정합니다.
    가장 작은 분산 방향 = 법선 벡터.

    Args:
        points_3d: Nx3 numpy array (mm)

    Returns:
        normal: (nx, ny, nz) 정규화된 법선 벡터
        roll: degrees
        pitch: degrees
    """
    centroid = points_3d.mean(axis=0)
    centered = points_3d - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 가장 작은 고유값 → 법선
    normal = eigenvectors[:, 0]

    # 법선이 카메라 방향(-Z)을 향하도록 보정
    if normal[2] > 0:
        normal = -normal

    roll = math.degrees(math.atan2(normal[1], -normal[2]))
    pitch = math.degrees(math.atan2(normal[0], -normal[2]))

    return normal, roll, pitch


# =============================================================================
# ROI 추출
# =============================================================================

def extract_roi(image, bbox, margin=10):
    """YOLO bbox에서 ROI를 크롭합니다 (여백 포함).

    Args:
        image: 원본 이미지
        bbox: (x1, y1, x2, y2)
        margin: 여백 픽셀

    Returns:
        cropped: 크롭된 이미지
        offset: (ox, oy) 크롭 오프셋 (원본 좌표 복원용)
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    x1m = max(0, x1 - margin)
    y1m = max(0, y1 - margin)
    x2m = min(w, x2 + margin)
    y2m = min(h, y2 + margin)

    cropped = image[y1m:y2m, x1m:x2m]
    return cropped, (x1m, y1m)
