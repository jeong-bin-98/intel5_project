# main_gui.py
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import sys
import time

# 분리해둔 로봇 제어 모듈에서 필요한 객체와 함수들을 가져옵니다.
from robot_logic import (
    indy, 
    stop_event, 
    task_pal_1by2_2layer, 
    task_pick2by2_place2by2, 
    move_done_check
)

# 파이썬의 기존 print() 함수를 낚아채서 Tkinter 창에 띄워주는 헬퍼 클래스
class PrintLogger:
    def __init__(self, app):
        self.app = app

    def write(self, text):
        if text.strip() or text == '\n':
            # GUI 요소 변경은 메인 스레드에서 해야 안전하므로 after 사용
            self.app.root.after(0, self._insert_text, text)

    def _insert_text(self, text):
        self.app.log_text.insert(tk.END, text)
        self.app.log_text.see(tk.END) # 자동 스크롤

    def flush(self):
        pass

# GUI 메인 클래스
class RobotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Indy7 로봇 제어 패널")
        self.root.geometry("550x550")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.current_task_thread = None

        # 버튼 프레임
        btn_frame = tk.Frame(self.root, pady=10)
        btn_frame.pack(fill="x")

        # 폰트 설정
        btn_font = ("Helvetica", 12, "bold")

        # 작업 1 버튼
        self.btn_task1 = tk.Button(btn_frame, text="작업 1 (1x2 2단)", bg="#4CAF50", fg="white", font=btn_font, width=20, height=2, command=self.run_task1)
        self.btn_task1.grid(row=0, column=0, padx=10, pady=5)

        # 작업 2 버튼
        self.btn_task2 = tk.Button(btn_frame, text="작업 2 (2x2 Pick)", bg="#2196F3", fg="white", font=btn_font, width=20, height=2, command=self.run_task2)
        self.btn_task2.grid(row=0, column=1, padx=10, pady=5)

        # 홈 버튼
        self.btn_home = tk.Button(btn_frame, text="홈(Home) 이동", bg="#FF9800", fg="white", font=btn_font, width=20, height=2, command=self.run_home)
        self.btn_home.grid(row=1, column=0, padx=10, pady=5)

        # 중지(Q) 버튼
        self.btn_stop = tk.Button(btn_frame, text="동작 중지 (Q)", bg="#F44336", fg="white", font=btn_font, width=20, height=2, command=self.run_stop)
        self.btn_stop.grid(row=1, column=1, padx=10, pady=5)

        # 로그 출력창 (Terminal 역할을 대신함)
        log_frame = tk.Frame(self.root)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 10), bg="#2d2d2d", fg="#cccccc")
        self.log_text.pack(fill="both", expand=True)

        # 이제부터 파이썬의 print()는 콘솔이 아닌 GUI 화면에 찍힙니다.
        sys.stdout = PrintLogger(self)

        print("=== 로봇 제어 패널 대기중 ===")
        print("버튼을 클릭하여 작업을 지시하세요.")

    def run_task1(self):
        if self.current_task_thread and self.current_task_thread.is_alive():
            print("⚠️ 이미 다른 작업이 진행 중입니다. (중지 버튼을 먼저 눌러주세요)")
            return
        stop_event.clear()
        self.current_task_thread = threading.Thread(target=task_pal_1by2_2layer, daemon=True)
        self.current_task_thread.start()

    def run_task2(self):
        if self.current_task_thread and self.current_task_thread.is_alive():
            print("⚠️ 이미 다른 작업이 진행 중입니다. (중지 버튼을 먼저 눌러주세요)")
            return
        stop_event.clear()
        self.current_task_thread = threading.Thread(target=task_pick2by2_place2by2, daemon=True)
        self.current_task_thread.start()

    def run_home(self):
        def _home_logic():
            if self.current_task_thread and self.current_task_thread.is_alive():
                print(">> 🛑 진행 중인 동작을 중단하고 홈으로 복귀합니다...")
                stop_event.set()
                self.current_task_thread.join(timeout=2.0)
            else:
                print(">> 홈으로 이동합니다...")
            
            stop_event.clear()
            indy.set_do(2, False) # 진공 흡착 끄기
            
            # 리셋 후 2초 대기하여 로봇 제어기가 정상화되도록 함
            indy.reset_robot()
            time.sleep(2.0)
            
            indy.go_home()
            try:
                move_done_check()
                print(">> 홈 이동 완료")
            except InterruptedError:
                pass

        # 홈 이동 과정도 대기 시간이 필요하므로 GUI가 멈추지 않게 별도 스레드로 실행
        threading.Thread(target=_home_logic, daemon=True).start()

    def run_stop(self):
        if self.current_task_thread and self.current_task_thread.is_alive():
            print(">> 🛑 진행 중인 동작을 즉시 중단합니다...")
            stop_event.set()
        else:
            print(">> 현재 진행 중인 작업이 없습니다.")

    def on_closing(self):
        if messagebox.askokcancel("종료", "프로그램을 종료하고 로봇 연결을 끊으시겠습니까?"):
            if self.current_task_thread and self.current_task_thread.is_alive():
                stop_event.set()
                self.current_task_thread.join(timeout=1.0)
            print(">> 로봇 연결 종료 중...")
            try:
                indy.disconnect()
            except Exception as e:
                pass
            self.root.destroy()

def main():
    root = tk.Tk()
    app = RobotApp(root)
    
    try:
        # 정상적인 GUI 실행 루프
        root.mainloop()
        
    except KeyboardInterrupt:
        # 터미널에서 강제로 Ctrl+C를 눌러서 껐을 때
        print("\n>> [알림] 터미널에서 강제 종료 요청을 받았습니다.")
        
    except Exception as e:
        # 그 외에 알 수 없는 에러로 프로그램이 튕겼을 때
        print(f"\n>> [치명적 에러] 프로그램에 예기치 못한 오류가 발생했습니다: {e}")
        
    finally:
        # ⭐ GUI를 끄든, 강제 종료하든, 에러가 나든 "무조건" 여기를 거칩니다! ⭐
        print("\n>> [시스템] 안전 종료 시퀀스 진입: 로봇과의 통신을 해제합니다...")
        try:
            # 확실하게 정지 신호를 한 번 더 보냄 (안전장치)
            stop_event.set() 
            time.sleep(0.1)
            
            # 대망의 Disconnect
            indy.disconnect()
            print(">> [시스템] 로봇 연결이 안전하게 해제되었습니다. 안녕히 가세요!")
        except Exception as e:
            print(f">> [시스템] 연결 해제 중 오류가 있었으나 프로그램을 강제 종료합니다: {e}")
        finally:
            # 파이썬 프로세스 자체를 완전히 메모리에서 죽임
            indy.disconnect()
            sys.exit(0)

if __name__ == "__main__":
    main()