"""
Indy7 로봇 설정
================
로봇 연결, 속도, 진공, 캘리브레이션 등의 설정값을 관리합니다.
"""

import numpy as np

# ===== 로봇 연결 =====
ROBOT_IP = "192.168.3.7"
ROBOT_NAME = "NRMK-Indy7"

# ===== 진공 (Digital Output) =====
VACUUM_DO_CHANNEL = 2
VACUUM_SETTLE_TIME = 2.0  # 진공 ON/OFF 후 대기 시간 (초)

# ===== 속도 (1~10) =====
DEFAULT_JOINT_VEL = 5
DEFAULT_TASK_VEL = 5

# ===== 동작 체크 =====
MOVE_CHECK_INTERVAL = 0.1  # 상태 폴링 간격 (초)
MOVE_TIMEOUT = 30.0        # 이동 타임아웃 (초)

# ===== 픽킹 =====
RETRACT_Z_OFFSET = 0.08   # 접근/후퇴 높이 오프셋 (미터)

# ===== 플레이스 위치 (로봇 좌표계, 미터/도) =====
PLACE_POS = [0.109, 0.260, 0.198]  # [x, y, z] 미터
PLACE_ROT = [0, 180, 90]           # [roll, pitch, yaw] 도

# ===== 카메라-로봇 변환 행렬 (Hand-Eye Calibration) =====
# Hand-eye calibration 결과로 얻은 4x4 변환 행렬
# None이면 단위 행렬(변환 없음)을 사용합니다.
#
# 실제 사용 시 캘리브레이션 결과를 아래 형식으로 입력하세요:
# CAMERA_TO_ROBOT_MATRIX = np.array([
#     [ r00, r01, r02, tx],
#     [ r10, r11, r12, ty],
#     [ r20, r21, r22, tz],
#     [   0,   0,   0,  1]
# ])
CAMERA_TO_ROBOT_MATRIX = None
