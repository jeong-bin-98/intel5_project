"""
칼만 필터 기반 오브젝트 트래킹
==============================

시간축 스무딩으로 depth 노이즈에 의한 좌표 떨림을 제거합니다.
"""

import math
import numpy as np
import cv2

from src.detection.constants import SMOOTHING_ALPHA, DetectedObject
from src.detection.geometry import compute_approach_vector


class _KalmanTrack:
    """개별 오브젝트의 칼만 필터 트래커.

    상태 벡터 (6D): [x, y, z, vx, vy, vz] — 위치 + 속도
    자세는 별도 EMA로 처리 (각도 도메인은 칼만보다 EMA가 적절).
    """

    def __init__(self, pos_3d, orientation, cx, cy, class_id,
                 process_noise=0.5, measurement_noise=5.0, normal=None):
        self.class_id = class_id
        self.cx = cx
        self.cy = cy
        self.last_seen = 0
        self.orientation = np.array(orientation, dtype=np.float64)
        self.normal = np.array(normal if normal is not None else (0, 0, -1),
                               dtype=np.float64)

        # 칼만 필터 초기화 (6 상태, 3 관측)
        self.kf = cv2.KalmanFilter(6, 3)

        # 상태 전이 행렬 (등속 모델): x' = x + vx*dt
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf.transitionMatrix[0, 3] = 1.0  # x += vx
        self.kf.transitionMatrix[1, 4] = 1.0  # y += vy
        self.kf.transitionMatrix[2, 5] = 1.0  # z += vz

        # 관측 행렬: 위치만 관측
        self.kf.measurementMatrix = np.zeros((3, 6), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1.0
        self.kf.measurementMatrix[1, 1] = 1.0
        self.kf.measurementMatrix[2, 2] = 1.0

        # 프로세스 노이즈 (작을수록 안정적, 클수록 반응 빠름)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        # 속도 성분의 프로세스 노이즈를 약간 더 크게
        self.kf.processNoiseCov[3, 3] = process_noise * 2
        self.kf.processNoiseCov[4, 4] = process_noise * 2
        self.kf.processNoiseCov[5, 5] = process_noise * 2

        # 관측 노이즈 (depth 센서 노이즈 수준)
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise

        # 초기 상태
        self.kf.statePost = np.array(
            [pos_3d[0], pos_3d[1], pos_3d[2], 0, 0, 0], dtype=np.float32
        )
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 50

    def predict(self):
        """예측 단계 — 미탐지 프레임에서도 호출"""
        return self.kf.predict()

    def update(self, pos_3d, orientation, cx, cy, normal=None):
        """관측값으로 보정"""
        measurement = np.array(
            [pos_3d[0], pos_3d[1], pos_3d[2]], dtype=np.float32
        )
        self.kf.correct(measurement)
        self.cx = cx
        self.cy = cy
        self.last_seen = 0

        # 자세는 EMA (alpha=0.4) — 각도는 칼만보다 단순 스무딩이 적절
        alpha = SMOOTHING_ALPHA
        self.orientation = alpha * np.array(orientation) + (1 - alpha) * self.orientation

        # 법선 벡터 EMA 스무딩
        if normal is not None:
            self.normal = alpha * np.array(normal) + (1 - alpha) * self.normal
            # 정규화 (EMA 후 단위 벡터 유지)
            norm = np.linalg.norm(self.normal)
            if norm > 0:
                self.normal = self.normal / norm

    def get_state(self):
        """현재 추정 위치"""
        s = self.kf.statePost
        return (float(s[0]), float(s[1]), float(s[2]))

    def get_orientation(self):
        return tuple(self.orientation)

    def get_normal(self):
        return tuple(self.normal)


class ObjectSmoother:
    """칼만 필터 기반 시간축 스무딩으로 depth 노이즈에 의한 좌표 떨림을 제거합니다.

    각 오브젝트를 등속(constant velocity) 칼만 필터로 트래킹합니다:
    - 위치(X,Y,Z): 칼만 필터 (속도 모델링, 노이즈 적응적 가중)
    - 자세(roll,pitch,yaw): EMA 스무딩 (각도 도메인)
    - 미탐지 프레임: 예측값 사용 (predict-only)
    """

    def __init__(self, max_dist_px=60, timeout_frames=15,
                 process_noise=0.5, measurement_noise=5.0):
        """
        Args:
            max_dist_px: 동일 오브젝트로 간주할 최대 픽셀 거리
            timeout_frames: 이 프레임 수 동안 미탐지되면 트래킹 삭제
            process_noise: 칼만 프로세스 노이즈 (작을수록 안정적)
            measurement_noise: 칼만 관측 노이즈 (depth 센서 노이즈)
        """
        self.max_dist_px = max_dist_px
        self.timeout = timeout_frames
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.tracks = {}  # {track_id: _KalmanTrack}
        self._next_id = 0

    def smooth(self, objects):
        """탐지 결과에 칼만 필터 스무딩을 적용합니다.

        Args:
            objects: DetectedObject 리스트

        Returns:
            smoothed: 스무딩된 DetectedObject 리스트
        """
        # 모든 트랙 예측 단계 실행
        for track in self.tracks.values():
            track.predict()

        # 매칭: 현재 탐지 ↔ 기존 트랙 (bbox 중심 거리 기준)
        used_tracks = set()
        results = []

        for obj in objects:
            cx = (obj.bbox[0] + obj.bbox[2]) // 2
            cy = (obj.bbox[1] + obj.bbox[3]) // 2

            # 가장 가까운 기존 트랙 찾기
            best_tid = None
            best_dist = self.max_dist_px

            for tid, track in self.tracks.items():
                if tid in used_tracks:
                    continue
                if track.class_id != obj.class_id:
                    continue
                dx = cx - track.cx
                dy = cy - track.cy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                # 기존 트랙 업데이트 (칼만 보정)
                track = self.tracks[best_tid]
                track.update(obj.pos_3d, obj.orientation, cx, cy,
                             normal=obj.normal)
                used_tracks.add(best_tid)

                s_pos = track.get_state()
                s_ori = track.get_orientation()
                s_normal = track.get_normal()

                # 스무딩된 좌표로 approach 재계산
                approach = compute_approach_vector(
                    s_pos[0], s_pos[1], s_pos[2],
                    s_ori[0], s_ori[1], s_ori[2],
                    offset_mm=100
                )

                results.append(DetectedObject(
                    obj.class_id, obj.confidence, obj.bbox,
                    s_pos, s_ori, approach, normal=s_normal
                ))
            else:
                # 새 트랙 생성
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = _KalmanTrack(
                    obj.pos_3d, obj.orientation, cx, cy, obj.class_id,
                    self.process_noise, self.measurement_noise,
                    normal=obj.normal
                )
                used_tracks.add(tid)
                results.append(obj)

        # 타임아웃된 트랙 삭제
        expired = []
        for tid, track in self.tracks.items():
            if tid not in used_tracks:
                track.last_seen += 1
                if track.last_seen > self.timeout:
                    expired.append(tid)
        for tid in expired:
            del self.tracks[tid]

        return results


def select_pick_target(objects):
    """
    탐지된 소켓들 중 픽킹 대상을 선택합니다.

    레퍼런스의 전략과 동일: Z값이 가장 높은(카메라에 가장 가까운) 소켓을 선택.
    빈(상자) 안에서 가장 위에 있는 소켓이 잡기 쉬우므로.

    Args:
        objects: DetectedObject 리스트

    Returns:
        DetectedObject 또는 None
    """
    if not objects:
        return None

    valid = [obj for obj in objects if obj.pos_3d and obj.pos_3d[2] > 0]
    if not valid:
        return None

    return min(valid, key=lambda obj: obj.pos_3d[2])
