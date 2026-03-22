"""
Indy7 로봇 컨트롤러
====================
Neuromeka Indy7 로봇의 빈픽킹 제어를 위한 래퍼 클래스.

기존 robot/src/indy_2dan.py, robot/src/gui/gui_main/robot_logic.py의
패턴을 재사용하여 구현.

시뮬레이션 모드를 지원하여 로봇 없이도 좌표 확인이 가능합니다.
"""

import sys
import os
import time
import numpy as np

# robot/indy_utils 패키지를 import할 수 있도록 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'robot'))


class Indy7Controller:
    """Indy7 로봇 빈픽킹 컨트롤러.

    Args:
        robot_ip: 로봇 IP 주소
        robot_name: 로봇 모델명
        simulation: True이면 실제 로봇에 연결하지 않고 명령만 출력
    """

    def __init__(self, robot_ip="192.168.3.7", robot_name="NRMK-Indy7", simulation=False):
        self.robot_ip = robot_ip
        self.robot_name = robot_name
        self.simulation = simulation
        self.indy = None
        self.connected = False

    def connect(self):
        """로봇에 연결합니다."""
        if self.simulation:
            print(f"[SIM] 로봇 연결 시뮬레이션 ({self.robot_ip})")
            self.connected = True
            return

        from indy_utils import indydcp_client as client
        self.indy = client.IndyDCPClient(self.robot_ip, self.robot_name)
        self.indy.connect()
        self.connected = True
        print(f"[ROBOT] 연결 완료: {self.robot_ip}")

    def disconnect(self):
        """안전하게 연결 해제합니다. 진공을 먼저 끕니다."""
        if self.connected and not self.simulation:
            try:
                self.set_vacuum(False)
            except Exception:
                pass
            if self.indy:
                self.indy.disconnect()
            print("[ROBOT] 연결 해제")
        elif self.simulation:
            print("[SIM] 연결 해제")
        self.connected = False

    def move_done_check(self, timeout=30.0):
        """현재 동작이 완료될 때까지 대기합니다.

        robot/src/gui/gui_main/robot_logic.py의 move_done_check() 패턴 재사용.
        에러/충돌 발생 시 리셋 후 InterruptedError를 발생시킵니다.

        Args:
            timeout: 타임아웃 (초)

        Raises:
            InterruptedError: 로봇 에러 또는 충돌 발생 시
            TimeoutError: 동작 타임아웃 시
        """
        if self.simulation:
            return

        time.sleep(0.2)  # 초기 안정화 대기
        start = time.time()

        while True:
            status = self.indy.get_robot_status()

            if status.get('error') == 1 or status.get('collision') == 1:
                print("[ROBOT] 에러/충돌 감지! 리셋 중...")
                self.indy.reset_robot()
                time.sleep(2.0)
                raise InterruptedError("로봇 에러/충돌 발생")

            if status['movedone'] == 1:
                break

            if time.time() - start > timeout:
                raise TimeoutError(f"동작 타임아웃 ({timeout}초)")

            time.sleep(0.1)

    def go_home(self):
        """홈 위치로 이동합니다."""
        if self.simulation:
            print("[SIM] go_home()")
            return
        self.indy.go_home()
        self.move_done_check()

    def set_velocity(self, joint_level=5, task_level=5):
        """속도 레벨을 설정합니다 (1~10).

        Args:
            joint_level: 관절 이동 속도 레벨
            task_level: 직교 이동 속도 레벨
        """
        if self.simulation:
            print(f"[SIM] set_velocity(joint={joint_level}, task={task_level})")
            return
        self.indy.set_joint_vel_level(joint_level)
        self.indy.set_task_vel_level(task_level)

    def task_move_to(self, pos_meters, rot_degrees):
        """직교 좌표로 절대 이동합니다.

        Args:
            pos_meters: [x, y, z] 위치 (미터)
            rot_degrees: [roll, pitch, yaw] 자세 (도)
        """
        pose = [pos_meters[0], pos_meters[1], pos_meters[2],
                rot_degrees[0], rot_degrees[1], rot_degrees[2]]
        if self.simulation:
            print(f"[SIM] task_move_to({[f'{v:.4f}' for v in pose]})")
            return
        self.indy.task_move_to(pose)
        self.move_done_check()

    def set_vacuum(self, on, channel=2, settle_time=2.0):
        """진공 흡착컵을 제어합니다.

        robot/src/indy_2dan.py의 set_do(2, True/False) 패턴 재사용.

        Args:
            on: True=흡착, False=해제
            channel: 디지털 출력 채널 번호
            settle_time: 진공 안정화 대기 시간 (초)
        """
        state = "ON" if on else "OFF"
        if self.simulation:
            print(f"[SIM] vacuum {state} (channel={channel})")
            return
        self.indy.set_do(channel, on)
        print(f"[ROBOT] 진공 {state}")
        time.sleep(settle_time)

    def pick(self, pick_pos_m, pick_rot_deg, retract_offset_m=0.08):
        """완전한 픽킹 시퀀스를 실행합니다.

        robot/src/indy_2dan.py의 pick 패턴 재사용:
        접근(위) → 하강 → 진공 ON → 후퇴(위)

        Args:
            pick_pos_m: [x, y, z] 픽 위치 (미터, 로봇 좌표계)
            pick_rot_deg: [roll, pitch, yaw] 픽 자세 (도)
            retract_offset_m: 접근/후퇴 높이 오프셋 (미터)
        """
        retract_pos = self._compute_retraction(pick_pos_m, pick_rot_deg, retract_offset_m)

        print(f"[PICK] 접근 위치로 이동...")
        self.task_move_to(retract_pos, pick_rot_deg)

        print(f"[PICK] 픽 위치로 하강...")
        self.task_move_to(pick_pos_m, pick_rot_deg)

        print(f"[PICK] 진공 ON")
        self.set_vacuum(True)

        print(f"[PICK] 후퇴...")
        self.task_move_to(retract_pos, pick_rot_deg)

    def place(self, place_pos_m, place_rot_deg, retract_offset_m=0.08):
        """완전한 플레이스 시퀀스를 실행합니다.

        접근(위) → 하강 → 진공 OFF → 후퇴(위)

        Args:
            place_pos_m: [x, y, z] 플레이스 위치 (미터, 로봇 좌표계)
            place_rot_deg: [roll, pitch, yaw] 플레이스 자세 (도)
            retract_offset_m: 접근/후퇴 높이 오프셋 (미터)
        """
        retract_pos = self._compute_retraction(place_pos_m, place_rot_deg, retract_offset_m)

        print(f"[PLACE] 접근 위치로 이동...")
        self.task_move_to(retract_pos, place_rot_deg)

        print(f"[PLACE] 플레이스 위치로 하강...")
        self.task_move_to(place_pos_m, place_rot_deg)

        print(f"[PLACE] 진공 OFF")
        self.set_vacuum(False)

        print(f"[PLACE] 후퇴...")
        self.task_move_to(retract_pos, place_rot_deg)

    def _compute_retraction(self, pos, rot_deg, dz_offset):
        """회전을 고려한 후퇴 위치를 계산합니다.

        robot/src/indy_2dan.py의 retractionPick() 함수 재사용.
        로봇 자세(회전)를 고려하여 로컬 -Z 방향으로 오프셋을 적용합니다.

        Args:
            pos: [x, y, z] 위치 (미터)
            rot_deg: [roll, pitch, yaw] 자세 (도)
            dz_offset: 후퇴 거리 (미터)

        Returns:
            [x, y, z] 후퇴 위치 (미터)
        """
        from scipy.spatial.transform import Rotation as R

        pos_arr = np.array(pos)
        rotation = R.from_euler('xyz', rot_deg, degrees=True)
        local_move = np.array([0, 0, -dz_offset])
        global_move = rotation.apply(local_move)
        return (pos_arr + global_move).tolist()
