
# from indy_utils import indydcp_client as client
# from pymcprotocol import Type3E
# from time import sleep
# import threading

# # -----------------------------
# # Indy7 연결
# # -----------------------------
# ROBOT_IP = "192.168.3.7"
# ROBOT_NAME = "NRMK-Indy7"

# indy = client.IndyDCPClient(ROBOT_IP, ROBOT_NAME)
# indy.connect()
# print("Indy7 연결 성공")

# # -----------------------------
# # PLC 연결
# # -----------------------------
# PLC_IP = "192.168.3.140"
# PLC_PORT = 1025

# plc = Type3E()
# plc.connect(PLC_IP, PLC_PORT)
# print("PLC 연결 성공")

# # -----------------------------
# # PLC 주소 정의
# # -----------------------------
# ADDR_START = "M100"   # PLC → Python
# ADDR_DONE  = "M101"   # Python → PLC

# # -----------------------------
# # 로봇 동작 완료 체크
# # -----------------------------
# def motion_done_check():
#     while True:
#         status = indy.get_robot_status()
#         if status["movedone"] == 1:
#             break
#         sleep(0.1)

# # -----------------------------
# # PLC 감시 루프
# # -----------------------------
# def plc_monitor():
#     print("PLC 감시 시작 (M100 대기중...)")

#     while True:
#         try:
#             start_signal = plc.batchread_bitunits(ADDR_START, 1)[0]

#             if start_signal == 1:
#                 print("[PLC] Start 신호 감지 → 홈 이동")

#                 # 로봇 홈 이동
#                 indy.go_zero()
#                 motion_done_check()

#                 print("[Robot] 홈 위치 도착 완료")

#                 # Done 신호 ON
#                 plc.batchwrite_bitunits(ADDR_DONE, [1])
#                 print("[PLC] Done 신호 ON")

#                 sleep(1)

#                 # Done OFF
#                 plc.batchwrite_bitunits(ADDR_DONE, [0])
#                 print("[PLC] Done 신호 OFF")

#                 # Start 신호 OFF 될 때까지 대기
#                 while plc.batchread_bitunits(ADDR_START, 1)[0] == 1:
#                     sleep(0.1)

#                 print("[PLC] Start 신호 OFF 확인 → 다음 대기")

#             else:
#                 sleep(0.1)

#         except Exception as e:
#             print(f"[오류] PLC 감시 중 예외 발생: {e}")
#             sleep(1)

# # -----------------------------
# # 메인 실행
# # -----------------------------
# try:
#     plc_thread = threading.Thread(target=plc_monitor)
#     plc_thread.start()

#     while True:
#         sleep(1)

# except KeyboardInterrupt:
#     print("프로그램 종료")

# finally:
#     indy.disconnect()
#     plc.close()
#     print("Indy7 및 PLC 연결 종료")
    
# from indy_utils import indydcp_client as client
# from pymcprotocol import Type3E
# from time import sleep
# import threading

# ROBOT_IP = "192.168.3.7"
# ROBOT_NAME = "NRMK-Indy7"

# indy = client.IndyDCPClient(ROBOT_IP, ROBOT_NAME)
# indy.connect()
# print("Indy7 연결 성공")


# PLC_IP = "192.168.3.140"
# PLC_PORT = 1025

# plc = Type3E()
# plc.connect(PLC_IP, PLC_PORT)
# print("PLC 연결 성공")

# ADDR_START_ZERO = "M100"
# ADDR_DONE_ZERO  = "M200"

# ADDR_START_HOME = "M101"
# ADDR_DONE_HOME  = "M201"

# def motion_done_check():
#     while True:
#         status = indy.get_robot_status()
#         if status["movedone"] == 1:
#             break
#         sleep(0.1)

# def plc_monitor():

#     print("PLC 감시 시작 (M100 / M200 대기중...)")

#     while True:
#         try:
#             # =============================
#             # Zero 이동 처리
#             # =============================
#             if plc.batchread_bitunits(ADDR_START_ZERO, 1)[0] == 1:

#                 print("[PLC] M100 감지 → Zero 이동 실행")
#                 indy.go_zero()
#                 motion_done_check()

#                 print("[Robot] Zero 이동 완료")

#                 # Done ON
#                 plc.batchwrite_bitunits(ADDR_DONE_ZERO, [1])
#                 sleep(1)

#                 # Done OFF
#                 plc.batchwrite_bitunits(ADDR_DONE_ZERO, [0])

#                 # Start OFF 대기
#                 while plc.batchread_bitunits(ADDR_START_ZERO, 1)[0] == 1:
#                     sleep(0.1)

#                 print("[PLC] M100 OFF 확인 → 대기 상태")

#             # =============================
#             # Home 이동 처리
#             # =============================
#             if plc.batchread_bitunits(ADDR_START_HOME, 1)[0] == 1:

#                 print("[PLC] M200 감지 → Home 이동 실행")

#                 indy.go_home()
#                 motion_done_check()

#                 print("[Robot] Home 이동 완료")

#                 # Done ON
#                 plc.batchwrite_bitunits(ADDR_DONE_HOME, [1])
#                 sleep(1)

#                 # Done OFF
#                 plc.batchwrite_bitunits(ADDR_DONE_HOME, [0])

#                 # Start OFF 대기
#                 while plc.batchread_bitunits(ADDR_START_HOME, 1)[0] == 1:
#                     sleep(0.1)

#                 print("[PLC] M200 OFF 확인 → 대기 상태")

#             sleep(0.1)

#         except Exception as e:
#             print(f"[오류] PLC 감시 중 예외 발생: {e}")
#             sleep(1)

#         except KeyboardInterrupt:
#             print("프로그램 종료")
#             break
# # -----------------------------
# # 메인 실행
# # -----------------------------
# try:
#     plc_thread = threading.Thread(target=plc_monitor, daemon=True)
#     plc_thread.start()

#     while True:
#         sleep(1)

# except KeyboardInterrupt:
#     print("프로그램 종료")

# finally:
#     indy.disconnect()
#     plc.close()
#     print("Indy7 및 PLC 연결 종료")   

from indy_utils import indydcp_client as client
from pymcprotocol import Type3E
from time import sleep
import threading

ROBOT_IP = "192.168.3.7"
ROBOT_NAME = "NRMK-Indy7"

indy = client.IndyDCPClient(ROBOT_IP, ROBOT_NAME)
indy.connect()
print("Indy7 연결 성공")

PLC_IP = "192.168.3.140"
PLC_PORT = 1025


plc = Type3E()
plc.connect(PLC_IP, PLC_PORT)
print("PLC 연결 성공")

# -----------------------------
# PLC 주소 정의
# -----------------------------
ADDR_START_ZERO = "M100"   # PLC → Python (Zero 이동)
ADDR_DONE_ZERO  = "M101"   # Python → PLC

ADDR_START_HOME = "M110"   # PLC → Python (Home 이동)
ADDR_DONE_HOME  = "M111"   # Python → PLC


def motion_done_check():
    while True:
        status = indy.get_robot_status()
        if status["movedone"] == 1:
            break
        sleep(0.1)

# -----------------------------
# 공통 동작 함수
# -----------------------------
def robot_motion(start_addr, done_addr, motion_func, motion_name):

    if plc.batchread_bitunits(start_addr, 1)[0] == 1:

        print(f"[PLC] {start_addr} 감지 → {motion_name} 실행")

        motion_func()
        motion_done_check()

        print(f"[Robot] {motion_name} 완료")

        # Done ON
        plc.batchwrite_bitunits(done_addr, [1])
        print(f"[PLC] {done_addr} ON")

        sleep(1)

        # Done OFF
        plc.batchwrite_bitunits(done_addr, [0])
        print(f"[PLC] {done_addr} OFF")

        # Start OFF 대기
        while plc.batchread_bitunits(start_addr, 1)[0] == 1:
            sleep(0.1)

        print(f"[PLC] {start_addr} OFF 확인 → 대기 상태")

# -----------------------------
# PLC 감시 루프
# -----------------------------
def plc_monitor():
    print("PLC 감시 시작 (M100 / M200 대기중...)")

    while True:
        try:
            # Zero 이동 감시
            robot_motion(
                ADDR_START_ZERO,
                ADDR_DONE_ZERO,
                indy.go_zero,
                "Zero 이동"
            )

            # Home 이동 감시
            robot_motion(
                ADDR_START_HOME,
                ADDR_DONE_HOME,
                indy.go_home,
                "Home 이동"
            )

            sleep(0.1)

        except Exception as e:
            print(f"[오류] PLC 감시 중 예외 발생: {e}")
            sleep(1)

# -----------------------------
# 메인 실행
# -----------------------------
try:
    plc_thread = threading.Thread(target=plc_monitor)
    plc_thread.start()

    while True:
        sleep(1)

except KeyboardInterrupt:
    print("프로그램 종료")

finally:
    indy.disconnect()
    plc.close()
    print("Indy7 및 PLC 연결 종료")

# #동시 입력 되었을 때 인터록 프로그램
# from indy_utils import indydcp_client as client
# from pymcprotocol import Type3E
# from time import sleep
# import threading

# # -----------------------------
# # 로봇 연결
# # -----------------------------
# ROBOT_IP = "192.168.3.7"
# ROBOT_NAME = "NRMK-Indy7"

# indy = client.IndyDCPClient(ROBOT_IP, ROBOT_NAME)
# indy.connect()
# print("Indy7 연결 성공")

# # -----------------------------
# # PLC 연결
# # -----------------------------
# PLC_IP = "192.168.3.140"
# PLC_PORT = 1025

# plc = Type3E()
# plc.connect(PLC_IP, PLC_PORT)
# print("PLC 연결 성공")

# # -----------------------------
# # PLC 주소 정의
# # -----------------------------
# ADDR_START_ZERO = "M100"
# ADDR_DONE_ZERO  = "M200"

# ADDR_START_HOME = "M101"
# ADDR_DONE_HOME  = "M201"

# # 선택사항: 에러 출력 비트
# ADDR_ERROR = "M900"

# # -----------------------------
# # 이동 완료 체크
# # -----------------------------
# def motion_done_check():
#     while True:
#         status = indy.get_robot_status()
#         if status["movedone"] == 1:
#             break
#         sleep(0.1)

# # -----------------------------
# # PLC 감시 루프
# # -----------------------------
# def plc_monitor():

#     print("PLC 감시 시작 (M100 / M200 대기중...)")

#     while True:
#         try:
#             zero_signal = plc.batchread_bitunits(ADDR_START_ZERO, 1)[0]
#             home_signal = plc.batchread_bitunits(ADDR_START_HOME, 1)[0]

#             # ==========================================
#             #  인터락 : 두 신호 동시 ON 방지
#             # ==========================================
#             if zero_signal == 1 and home_signal == 1:
#                 print("[인터락] Zero와 Home이 동시에 ON → 동작 차단")

#                 # 에러 비트 ON
#                 plc.batchwrite_bitunits(ADDR_ERROR, [1])

#                 # 둘 중 하나가 OFF 될 때까지 대기
#                 while True:
#                     zero_signal = plc.batchread_bitunits(ADDR_START_ZERO, 1)[0]
#                     home_signal = plc.batchread_bitunits(ADDR_START_HOME, 1)[0]
#                     if not (zero_signal == 1 and home_signal == 1):
#                         break
#                     sleep(0.1)

#                 # 에러 비트 OFF
#                 plc.batchwrite_bitunits(ADDR_ERROR, [0])
#                 print("[인터락 해제] 정상 감시 복귀")

#             # ==========================================
#             # Zero 이동 처리
#             # ==========================================
#             elif zero_signal == 1:

#                 print("[PLC] M100 감지 → Zero 이동 실행")

#                 indy.go_zero()
#                 motion_done_check()

#                 print("[Robot] Zero 이동 완료")

#                 plc.batchwrite_bitunits(ADDR_DONE_ZERO, [1])
#                 sleep(1)
#                 plc.batchwrite_bitunits(ADDR_DONE_ZERO, [0])

#                 while plc.batchread_bitunits(ADDR_START_ZERO, 1)[0] == 1:
#                     sleep(0.1)

#                 print("[PLC] M100 OFF 확인 → 대기 상태")

#             # ==========================================
#             # Home 이동 처리
#             # ==========================================
#             elif home_signal == 1:

#                 print("[PLC] M200 감지 → Home 이동 실행")

#                 indy.go_home()
#                 motion_done_check()

#                 print("[Robot] Home 이동 완료")

#                 plc.batchwrite_bitunits(ADDR_DONE_HOME, [1])
#                 sleep(1)
#                 plc.batchwrite_bitunits(ADDR_DONE_HOME, [0])

#                 while plc.batchread_bitunits(ADDR_START_HOME, 1)[0] == 1:
#                     sleep(0.1)

#                 print("[PLC] M200 OFF 확인 → 대기 상태")

#             sleep(0.1)

#         except Exception as e:
#             print(f"[오류] PLC 감시 중 예외 발생: {e}")
#             sleep(1)

# # -----------------------------
# # 메인 실행
# # -----------------------------
# try:
#     plc_thread = threading.Thread(target=plc_monitor, daemon=True)
#     plc_thread.start()

#     while True:
#         sleep(1)

# except KeyboardInterrupt:
#     print("프로그램 종료")

# finally:
#     indy.disconnect()
#     plc.close()
#     print("Indy7 및 PLC 연결 종료")
    



# from indy_utils import indydcp_client as client
# from pymcprotocol import Type3E
# from time import sleep
# import threading
# import time

# # -----------------------------
# # Indy7 연결
# # -----------------------------
# ROBOT_IP = "192.168.3.7"
# ROBOT_NAME = "NRMK-Indy7"

# indy = client.IndyDCPClient(ROBOT_IP, ROBOT_NAME)
# indy.connect()
# print("Indy7 연결 성공")

# # -----------------------------
# # PLC 연결
# # -----------------------------
# PLC_IP = "192.168.3.140"
# PLC_PORT = 1025

# plc = Type3E()
# plc.connect(PLC_IP, PLC_PORT)
# print("PLC 연결 성공")

# # -----------------------------
# # PLC 주소 정의
# # -----------------------------
# ADDR_START_ZERO = "M100"
# ADDR_DONE_ZERO  = "M200"

# ADDR_Y20 = "Y20"
# ADDR_Y21 = "Y21"


# # -----------------------------
# # 로봇 이동 완료 확인
# # -----------------------------
# def motion_done_check():
#     while True:
#         status = indy.get_robot_status()
#         if status["movedone"] == 1:
#             break
#         sleep(0.1)


# # -----------------------------
# # PLC 램프 점멸 함수
# # -----------------------------
# def blink_lamp(addr, duration):
#     """
#     addr : PLC 출력 주소 (예: Y20)
#     duration : 총 점멸 시간(초)
#     """
#     end_time = time.time() + duration

#     while time.time() < end_time:
#         plc.batchwrite_bitunits(addr, [1])
#         sleep(0.5)
#         plc.batchwrite_bitunits(addr, [0])
#         sleep(0.5)

#     # 종료 시 OFF 보장
#     plc.batchwrite_bitunits(addr, [0])


# # -----------------------------
# # PLC 감시 스레드
# # -----------------------------
# def plc_monitor():

#     print("PLC 감시 시작 (M100 대기중...)")

#     while True:
#         try:
#             # =============================
#             # Zero 이동 처리
#             # =============================
#             if plc.batchread_bitunits(ADDR_START_ZERO, 1)[0] == 1:

#                 print("[PLC] M100 감지")

#                 # 1️⃣ Y20 1초 점멸
#                 print("[PLC] Y20 1초 점멸")
#                 blink_lamp(ADDR_Y20, 1)

#                 # 2️⃣ 로봇 Zero 이동
#                 print("[Robot] Zero 이동 실행")
#                 indy.go_zero()
#                 motion_done_check()
#                 print("[Robot] Zero 이동 완료")

#                 # 3️⃣ Y21 2초 점멸
#                 print("[PLC] Y21 2초 점멸")
#                 blink_lamp(ADDR_Y21, 2)

#                 # 4️⃣ Done ON (1초)
#                 plc.batchwrite_bitunits(ADDR_DONE_ZERO, [1])
#                 sleep(1)
#                 plc.batchwrite_bitunits(ADDR_DONE_ZERO, [0])

#                 # 5️⃣ Start OFF 대기
#                 while plc.batchread_bitunits(ADDR_START_ZERO, 1)[0] == 1:
#                     sleep(0.1)

#                 print("[PLC] M100 OFF 확인 → 대기 상태")

#             sleep(0.1)

#         except Exception as e:
#             print(f"[오류] PLC 감시 중 예외 발생: {e}")
#             sleep(1)


# # -----------------------------
# # 메인 실행
# # -----------------------------
# try:
#     plc_thread = threading.Thread(target=plc_monitor)
#     plc_thread.start()

#     plc_thread.join()

# except KeyboardInterrupt:
#     print("프로그램 종료")

# finally:
#     indy.disconnect()
#     plc.close()
#     print("Indy7 및 PLC 연결 종료")
     
     
     
# 이제 현장에서 실제 사용하는 수준의 완전체 구조로

# ✅ 4-Phase Handshake

# ✅ 타임아웃 감시

# ✅ Alarm 처리

# ✅ Reset 구조

# PLC → Robot
# | 디바이스 | 의미          |
# | ---- | ----------- |
# | M100 | Cycle Start |
# | M105 | PLC Ack     |
# | M400 | Alarm Reset |
# Robot → PLC
# 디바이스	의미
# M200	Robot Busy
# M101	Cycle Complete
# M102	Robot Alarm  

# PLC 내부
# 디바이스	의미
# M110	Step1 (Start Sent)
# M120	Step2 (Running)
# M130	Step3 (Complete)
# M300	Timeout Alarm
# M500	System Alarm
# T10	Timeout Timer (10초)
# 🧩 2️⃣ PLC 전체 래더 구조 (Mitsubishi 기준)
# 🔹 (1) Start 입력
# |---[ X0 START ]------------------------( SET M100 )
# 🔹 (2) Robot Busy 응답 확인
# |---[ M100 ]---[ M200 ]-----------------( SET M110 )

# 👉 Robot이 Busy ON 하면 정상 수신

# 🔹 (3) Timeout 감시 (10초)
# |---[ M110 ]----------------------------( T10 K100 )

# 100ms × 100 = 10초

# 🔹 (4) Timeout 발생 → Alarm
# |---[ T10 ]-----------------------------( SET M300 )
# 🔹 (5) Robot 완료 감지
# |---[ M110 ]---[ M101 ]-----------------( SET M120 )
# 🔹 (6) PLC Ack 출력
# |---[ M120 ]----------------------------( SET M105 )
# 🔹 (7) Robot 신호 OFF 확인 후 초기화
# |---[ /M200 ]---[ /M101 ]---------------( SET M130 )
# |---[ M130 ]----------------------------( RST M100 M105 M110 M120 M130 )
# 🔹 (8) Alarm 통합
# |---[ M102 ]----------------------------( SET M500 )
# |---[ M300 ]----------------------------( SET M500 )
# 🔹 (9) Alarm Reset
# |---[ X1 RESET ]------------------------( RST M500 M300 M102 )
# 🤖 3️⃣ Python 전체 프로그램 (4-Phase + Timeout + Alarm)
# from indy_utils import indydcp_client as client
# from pymcprotocol import Type3E
# from time import sleep
# import time

# # -----------------------------
# # Indy 연결
# # -----------------------------
# ROBOT_IP = "192.168.3.7"
# ROBOT_NAME = "NRMK-Indy7"

# indy = client.IndyDCPClient(ROBOT_IP, ROBOT_NAME)
# indy.connect()
# print("Robot Connected")

# # -----------------------------
# # PLC 연결
# # -----------------------------
# PLC_IP = "192.168.3.140"
# PLC_PORT = 1025

# plc = Type3E()
# plc.connect(PLC_IP, PLC_PORT)
# print("PLC Connected")

# # -----------------------------
# # Motion Done 체크
# # -----------------------------
# def motion_done_check(timeout=10):

#     start_time = time.time()

#     while True:
#         status = indy.get_robot_status()

#         # 정상 완료
#         if status["movedone"] == 1:
#             return True

#         # 타임아웃
#         if time.time() - start_time > timeout:
#             return False

#         sleep(0.05)

# # -----------------------------
# # 메인 루프
# # -----------------------------
# try:
#     while True:

#         start_zero = plc.batchread_bitunits("M100", 1)[0]
#         start_home = plc.batchread_bitunits("M103", 1)[0]
#         if start_zero == 1:

#             print("Cycle Start 감지")

#             # Phase 1 : Busy ON
#             plc.batchwrite_bitunits("M200", [1])

#             # 실제 로봇 동작
            
#             indy.go_zero()

#             success = motion_done_check(timeout=10)

#             if success:
#                 print("Motion Complete")

#                 # Phase 2 : Complete ON
#                 plc.batchwrite_bitunits("M101", [1])

#             else:
#                 print("Timeout 발생 → Alarm")

#                 # Robot Alarm
#                 plc.batchwrite_bitunits("M201", [1])
#                 plc.batchwrite_bitunits("M200", [0])
#                 continue

#             # Phase 3 : PLC Ack 대기
#             while plc.batchread_bitunits("M105", 1)[0] == 0:
#                 sleep(0.05)

#             print("PLC Ack 수신")

#             # Phase 4 : 신호 정리
#             plc.batchwrite_bitunits("M101", [0])
#             plc.batchwrite_bitunits("M200", [0])

#             # PLC가 Start OFF 할 때까지 대기
#             while plc.batchread_bitunits("M100", 1)[0] == 1:
#                 sleep(0.05)

#             print("Cycle 종료")
        
#         if start_home == 1:

#             print("Cycle Start 감지")

#             # Phase 1 : Busy ON
#             plc.batchwrite_bitunits("M200", [1])

#             # 실제 로봇 동작
            
#             indy.go_home()

#             success = motion_done_check(timeout=10)

#             if success:
#                 print("Motion Complete")

#                 # Phase 2 : Complete ON
#                 plc.batchwrite_bitunits("M101", [1])

#             else:
#                 print("Timeout 발생 → Alarm")

#                 # Robot Alarm
#                 plc.batchwrite_bitunits("M201", [1])
#                 plc.batchwrite_bitunits("M200", [0])
#                 continue

#             # Phase 3 : PLC Ack 대기
#             while plc.batchread_bitunits("M105", 1)[0] == 0:
#                 sleep(0.05)

#             print("PLC Ack 수신")

#             # Phase 4 : 신호 정리
#             plc.batchwrite_bitunits("M101", [0])
#             plc.batchwrite_bitunits("M200", [0])

#             # PLC가 Start OFF 할 때까지 대기
#             while plc.batchread_bitunits("M100", 1)[0] == 1:
#                 sleep(0.05)

#             print("Cycle 종료")
#         sleep(0.05)

# except KeyboardInterrupt:
#     print("프로그램 종료")

# finally:
#     indy.disconnect()
#     plc.close()
#     print("연결 종료")
