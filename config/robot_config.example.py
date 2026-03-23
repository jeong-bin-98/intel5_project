"""
Indy7 로봇 설정 (템플릿)
========================
이 파일을 복사하여 config/robot_config.py 로 저장한 뒤 현장에 맞게 수정하세요.
robot_config.py 는 .gitignore 에 포함되어 있습니다 (IP/좌표 보안).

  cp config/robot_config.example.py config/robot_config.py
"""

import numpy as np

# ===== 로봇 연결 =====
ROBOT_IP = "192.168.x.x"       # ← 현장 로봇 IP로 변경
ROBOT_NAME = "NRMK-Indy7"

# ===== 진공 (Digital Output) =====
VACUUM_DO_CHANNEL = 2
VACUUM_SETTLE_TIME = 2.0        # 진공 ON/OFF 후 대기 시간 (초)

# ===== 속도 (1~10) =====
DEFAULT_JOINT_VEL = 5
DEFAULT_TASK_VEL = 5

# ===== 동작 체크 =====
MOVE_CHECK_INTERVAL = 0.1       # 상태 폴링 간격 (초)
MOVE_TIMEOUT = 30.0             # 이동 타임아웃 (초)

# ===== 픽킹 =====
RETRACT_Z_OFFSET = 0.08         # 접근/후퇴 높이 오프셋 (미터)

# ===== 플레이스 위치 (로봇 좌표계, 미터/도) =====
PLACE_POS = [0.0, 0.0, 0.0]    # ← 현장 플레이스 위치로 변경 [x, y, z] 미터
PLACE_ROT = [0, 180, 90]        # ← 현장 플레이스 자세로 변경 [roll, pitch, yaw] 도

# ===== 카메라-로봇 변환 행렬 (Hand-Eye Calibration) =====
# hand_eye_calibration.py 실행 후 결과를 아래에 입력하세요.
# None이면 단위 행렬(변환 없음)을 사용합니다.
#
# CAMERA_TO_ROBOT_MATRIX = np.array([
#     [ r00, r01, r02, tx],
#     [ r10, r11, r12, ty],
#     [ r20, r21, r22, tz],
#     [   0,   0,   0,  1]
# ])
CAMERA_TO_ROBOT_MATRIX = None
