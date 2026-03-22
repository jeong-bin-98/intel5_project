"""
상수 및 데이터 모델
==================

탐지 시스템의 설정값과 DetectedObject 데이터 클래스를 정의합니다.
"""

import torch

# GPU 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# YOLO 설정
CONFIDENCE = 0.7
CLASS_NAMES = {0: "8pin", 1: "12pin"}
COLORS = {0: (255, 100, 0), 1: (0, 100, 255)}

# Depth 기반 3D 좌표 계산 설정
DEPTH_SAMPLE_RADIUS = 5       # 중심점 주변 depth 샘플링 반경 (pixels)
DEPTH_MIN_MM = 100            # 최소 유효 depth (mm) - 노이즈 제거
DEPTH_MAX_MM = 1000           # 최대 유효 depth (mm)

# 표면 법선 추정 설정
NORMAL_PATCH_SIZE = 15        # 법선 추정용 패치 크기 (pixels)

# 시간축 스무딩 (depth 노이즈 안정화)
SMOOTHING_ALPHA = 0.4         # EMA 가중치 (0=느림/안정, 1=즉시반응)


class DetectedObject:
    """탐지된 소켓 한 개의 정보를 담는 클래스"""

    def __init__(self, class_id, confidence, bbox, pos_3d, orientation, approach,
                 normal=None):
        self.class_id = class_id        # 0=8pin, 1=12pin
        self.class_name = CLASS_NAMES.get(class_id, str(class_id))
        self.confidence = confidence     # 0~1
        self.bbox = bbox                 # (x1, y1, x2, y2) 픽셀
        self.pos_3d = pos_3d             # (X, Y, Z) mm 카메라 좌표계
        self.orientation = orientation   # (roll, pitch, yaw) degrees
        self.approach = approach         # ((ax,ay,az), (rx,ry,rz))
        self.normal = normal             # (nx, ny, nz) 표면 법선 벡터

    @property
    def depth_mm(self):
        return self.pos_3d[2] if self.pos_3d else 0

    def __repr__(self):
        return (f"{self.class_name} conf={self.confidence:.0%} "
                f"pos=({self.pos_3d[0]:.1f}, {self.pos_3d[1]:.1f}, {self.pos_3d[2]:.1f})mm "
                f"ori=({self.orientation[0]:.1f}, {self.orientation[1]:.1f}, {self.orientation[2]:.1f})deg")
