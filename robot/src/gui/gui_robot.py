
from indy_utils import indydcp_client as client
import tkinter as tk
from time import sleep
import threading

# -----------------------
# Indy 연결 설정
# -----------------------
ROBOT_IP = "192.168.3.7"
ROBOT_NAME = "NRMK-Indy7"

indy = client.IndyDCPClient(ROBOT_IP, ROBOT_NAME)
indy.connect()

# -----------------------
# 로봇 동작 함수
# -----------------------
def motion_done_check():
    """로봇이 동작 완료될 때까지 대기"""
    while True:
        status = indy.get_robot_status()
        if status['movedone'] == 1:
            break
        sleep(0.1)

def move_home():
    """로봇을 홈 위치로 이동"""
    def task():
        print("홈 위치로 이동 중...")
        indy.go_home()
        motion_done_check()
        print("홈 위치 도착 완료!")
    threading.Thread(target=task).start()  # GUI 멈춤 방지

def move_zero():
    """로봇을 영점 위치로 이동"""
    def task():
        print("영점 위치로 이동 중...")
        indy.go_zero()
        motion_done_check()
        print("영점 위치 도착 완료!")
    threading.Thread(target=task).start()  # GUI 멈춤 방지 , 쓰레드는 지정된 함수가 끝날때 까지 존재

# -----------------------
# GUI 구성
# -----------------------
root = tk.Tk()
root.title("Indy7 제어 패널")
root.geometry("300x200")

label = tk.Label(root, text="Indy7 로봇 제어", font=("Arial", 16))
label.pack(pady=10)

btn_home = tk.Button(root, text="1. 홈으로 이동", command=move_home, bg="lightblue", width=20, height=2)
btn_home.pack(pady=10)

btn_zero = tk.Button(root, text="2. 영점으로 이동", command=move_zero, bg="lightgreen", width=20, height=2)
btn_zero.pack(pady=10)

def on_closing():
    indy.disconnect()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()


# Tkinter는 GUI 변경은 반드시 메인 스레드에서 해야 합니다.
# 즉, 백그라운드 스레드에서 label.config(text="...") 같은 코드를 직접 호출하면 오류가 납니다.
# (→ 해결: root.after() 또는 queue 이용)

# 스레드가 많아지면 자원 관리나 예외 처리 필요
# → 로봇 제어는 한 번에 하나의 스레드만 동작하도록 관리해야 함
# [Main Thread]
#  ├─ Tkinter mainloop()
#  │   ├─ 버튼 클릭 감지
#  │   ├─ 창 그리기
#  │   └─ 이벤트 처리 유지
#  │
#  └─ (버튼 클릭 시)
#       └─ [New Thread]
#            ├─ indy.go_home()
#            ├─ motion_done_check()
#            └─ print("완료")
# 이처럼 move_home()은 즉시 끝나고,
# 실제 로봇 이동은 별도의 task 스레드가 처리하므로
# Tkinter는 계속 “살아 있는” 상태로 유지됩니다.


# Tkinter는 GUI 변경은 반드시 메인 스레드에서 해야 합니다.
# 즉, 백그라운드 스레드에서 label.config(text="...") 같은 코드를 직접 호출하면 오류가 납니다.
# (→ 해결: root.after() 또는 queue 이용)

# 스레드가 많아지면 자원 관리나 예외 처리 필요
# → 로봇 제어는 한 번에 하나의 스레드만 동작하도록 관리해야 함

# ✅ 정리

# 구분	내용
# threading.Thread 사용 전	GUI 멈춤 (mainloop 블로킹)
# threading.Thread 사용 후	GUI는 계속 응답, 로봇은 백그라운드에서 동작
# 원리	Tkinter mainloop와 로봇 제어 코드를 별도 스레드로 분리

# 원하신다면 이 스레드 동작을 그림(스레드 흐름 다이어그램) 으로 시각화해드릴까요?







