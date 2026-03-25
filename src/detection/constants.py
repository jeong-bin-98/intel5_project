"""
상수 및 데이터 모델
==================

탐지 시스템의 설정값과 DetectedObject 데이터 클래스를 정의합니다.
"""

import torch
from ultralytics.utils import torch_utils
from ultralytics.engine import predictor as _predictor

# Ultralytics select_device 패치: XPU 디바이스 인식
_orig_select_device = torch_utils.select_device

def _patched_select_device(device="", newline=False, verbose=True):
    d = str(device) if not hasattr(device, "type") else device.type
    if d.startswith("xpu"):
        return torch.device("xpu")
    return _orig_select_device(device, newline, verbose)

torch_utils.select_device = _patched_select_device
_predictor.select_device = _patched_select_device

# GPU 설정 (Intel Arc A770 XPU)
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = 'xpu'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

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
                 normal=None, obb_corners=None, obb_angle=None):
        self.class_id = class_id        # 0=8pin, 1=12pin
        self.class_name = CLASS_NAMES.get(class_id, str(class_id))
        self.confidence = confidence     # 0~1
        self.bbox = bbox                 # (x1, y1, x2, y2) 축 정렬 바운딩 박스 (호환용)
        self.pos_3d = pos_3d             # (X, Y, Z) mm 카메라 좌표계
        self.orientation = orientation   # (roll, pitch, yaw) degrees
        self.approach = approach         # ((ax,ay,az), (rx,ry,rz))
        self.normal = normal             # (nx, ny, nz) 표면 법선 벡터
        self.obb_corners = obb_corners   # OBB 꼭짓점 (4, 2) ndarray or None
        self.obb_angle = obb_angle       # OBB 회전 각도 (degrees) or None

    @property
    def depth_mm(self):
        return self.pos_3d[2] if self.pos_3d else 0

    def __repr__(self):
        return (f"{self.class_name} conf={self.confidence:.0%} "
                f"pos=({self.pos_3d[0]:.1f}, {self.pos_3d[1]:.1f}, {self.pos_3d[2]:.1f})mm "
                f"ori=({self.orientation[0]:.1f}, {self.orientation[1]:.1f}, {self.orientation[2]:.1f})deg")
