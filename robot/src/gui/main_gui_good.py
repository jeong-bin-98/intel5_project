# main_gui.py
import customtkinter as ctk
import tkinter.messagebox as messagebox
import threading
import sys
import time

# 3D 그래픽 출력을 위한 matplotlib 임포트
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

# ⭐ 로봇 제어 모듈 가져오기 (robot_logic.py 필수)
from robot_logic import (
    indy, 
    stop_event, 
    task_pal_1by2_2layer, 
    task_pick2by2_place2by2, 
    move_done_check
)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class PrintLogger:
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, text):
        if text.strip() or text == '\n':
            self.textbox.after(0, self._insert_text, text)

    def _insert_text(self, text):
        self.textbox.insert(ctk.END, text)
        self.textbox.see(ctk.END)

    def flush(self): pass

class RobotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Indy7 3D Digital Twin Center")
        self.geometry("1200x850") # 3D 뷰어를 위해 창 크기를 넓혔습니다.
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.current_task_thread = None
        self.task_pos = None  
        self.joint_pos = None 
        self.is_jogging = False 

        # 3D 궤적 저장을 위한 리스트
        self.history_x, self.history_y, self.history_z = [], [], []

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # 1. 왼쪽 사이드바 (기본 컨트롤 및 로그)
        # ==========================================
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="🤖 Indy7 System", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        # 자동 작업 버튼
        self.btn_task1 = ctk.CTkButton(self.sidebar, text="📦 작업 1 (1x2 2단)", font=ctk.CTkFont(size=14, weight="bold"), height=45, fg_color="#2E7D32", hover_color="#1B5E20", command=self.run_task1)
        self.btn_task1.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.btn_task2 = ctk.CTkButton(self.sidebar, text="⚡ 작업 2 (2x2 Pick)", font=ctk.CTkFont(size=14, weight="bold"), height=45, fg_color="#1976D2", hover_color="#0D47A1", command=self.run_task2)
        self.btn_task2.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        # 제어 버튼
        self.btn_home = ctk.CTkButton(self.sidebar, text="🏠 홈 이동", font=ctk.CTkFont(size=14, weight="bold"), height=45, fg_color="#F57C00", hover_color="#E65100", command=self.run_home)
        self.btn_home.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        self.btn_stop = ctk.CTkButton(self.sidebar, text="🛑 긴급 중지 (Q)", font=ctk.CTkFont(size=14, weight="bold"), height=45, fg_color="#D32F2F", hover_color="#B71C1C", command=self.run_stop)
        self.btn_stop.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        # 로그 터미널
        self.log_frame = ctk.CTkFrame(self.sidebar, fg_color="#121212")
        self.log_frame.grid(row=5, column=0, padx=10, pady=20, sticky="nsew")
        self.log_text = ctk.CTkTextbox(self.log_frame, font=ctk.CTkFont(family="Consolas", size=12), fg_color="transparent", text_color="#00FF41", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        sys.stdout = PrintLogger(self.log_text)

        # ==========================================
        # 2. 오른쪽 메인 영역 (3D 뷰어, 좌표, 조그)
        # ==========================================
        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        self.main_area.grid_columnconfigure(0, weight=3) # 3D 뷰어가 넓게
        self.main_area.grid_columnconfigure(1, weight=1) # 텍스트/조그는 좁게
        self.main_area.grid_rowconfigure(0, weight=1)    # 위쪽 영역 꽉 차게
        
        # ---------------- (1) 3D 실시간 모니터링 뷰어 ----------------
        self.plot_frame = ctk.CTkFrame(self.main_area, corner_radius=10, fg_color="#1E1E1E")
        self.plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        
        ctk.CTkLabel(self.plot_frame, text="🛰️ 3D Digital Twin Viewer", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        # Matplotlib 3D 설정
        self.fig = plt.Figure(figsize=(6, 5), facecolor='#1E1E1E')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1E1E1E')
        
        # 축 디자인 (다크 모드 어울리게)
        self.ax.xaxis.set_pane_color((0.15, 0.15, 0.15, 1.0))
        self.ax.yaxis.set_pane_color((0.15, 0.15, 0.15, 1.0))
        self.ax.zaxis.set_pane_color((0.15, 0.15, 0.15, 1.0))
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.set_xlabel('X (m)', color='white')
        self.ax.set_ylabel('Y (m)', color='white')
        self.ax.set_zlabel('Z (m)', color='white')
        
        # 로봇이 주로 움직이는 공간(0.8m 반경)으로 축 제한 고정
        self.ax.set_xlim([-0.8, 0.8])
        self.ax.set_ylim([-0.8, 0.8])
        self.ax.set_zlim([0, 1.0])

        # 그리기 객체 초기화 (선, 점, 궤적)
        self.robot_arm_line, = self.ax.plot([], [], [], 'o-', color='#00FF41', lw=3, markersize=8)
        self.robot_trail, = self.ax.plot([], [], [], '-', color='#4FC3F7', alpha=0.5, lw=2)

        # Tkinter에 그래프 붙이기
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # ---------------- (2) 수치 데이터 & Jog 패드 영역 ----------------
        self.info_frame = ctk.CTkFrame(self.main_area, fg_color="transparent")
        self.info_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
        self.info_frame.grid_rowconfigure(2, weight=1)

        # [수치 좌표 디스플레이]
        self.coord_frame = ctk.CTkFrame(self.info_frame, fg_color="#2A2D34", corner_radius=10)
        self.coord_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(self.coord_frame, text="[ Task 좌표 (Base ➔ Tool) ]", font=ctk.CTkFont(size=12, weight="bold"), text_color="#AAAAAA").pack(pady=(15, 0))
        self.task_label = ctk.CTkLabel(self.coord_frame, text="수신 대기 중...", font=ctk.CTkFont(family="Consolas", size=14, weight="bold"), text_color="#4FC3F7")
        self.task_label.pack(pady=(5, 15))

        ctk.CTkLabel(self.coord_frame, text="[ Joint 각도 (J1 ~ J6) ]", font=ctk.CTkFont(size=12, weight="bold"), text_color="#AAAAAA").pack(pady=(5, 0))
        self.joint_label = ctk.CTkLabel(self.coord_frame, text="수신 대기 중...", font=ctk.CTkFont(family="Consolas", size=14, weight="bold"), text_color="#A5D6A7")
        self.joint_label.pack(pady=(5, 15))

        # [수동 조그 패드]
        self.jog_frame = ctk.CTkFrame(self.info_frame, corner_radius=10)
        self.jog_frame.pack(fill="both", expand=True)
        
        ctk.CTkLabel(self.jog_frame, text="🕹️ Hold & Jog (수동 조작)", font=ctk.CTkFont(size=15, weight="bold")).pack(pady=15)

        # 위치 조그
        pos_frame = ctk.CTkFrame(self.jog_frame, fg_color="transparent")
        pos_frame.pack(pady=5)
        for row, (ax_name, ax_idx) in enumerate([("X축", 0), ("Y축", 1), ("Z축", 2)]):
            self.create_jog_buttons(pos_frame, row, ax_name, ax_idx, "#2E7D32", "#1B5E20")

        # 회전 조그
        rot_frame = ctk.CTkFrame(self.jog_frame, fg_color="transparent")
        rot_frame.pack(pady=15)
        for row, (ax_name, ax_idx) in enumerate([("Rx", 3), ("Ry", 4), ("Rz", 5)]):
            self.create_jog_buttons(rot_frame, row, ax_name, ax_idx, "#6A1B9A", "#4A148C")

        print("=== 3D Digital Twin Ready ===")

        # 실시간 모니터링 시작
        threading.Thread(target=self._poll_position, daemon=True).start()
        self.update_gui_coordinates()

    # ---------------- 조그 버튼 생성 도우미 ----------------
    def create_jog_buttons(self, parent, row, ax_name, ax_idx, fg, h_color):
        jog_font = ctk.CTkFont(size=16, weight="bold")
        btn_minus = ctk.CTkButton(parent, text="-", width=50, height=40, font=jog_font, fg_color=fg, hover_color=h_color)
        btn_minus.grid(row=row, column=0, padx=5, pady=2)
        btn_minus.bind("<ButtonPress-1>", lambda e, idx=ax_idx: self.start_jog(idx, -1))
        btn_minus.bind("<ButtonRelease-1>", self.stop_jog)
        btn_minus.bind("<Leave>", self.stop_jog)

        ctk.CTkLabel(parent, text=ax_name, font=ctk.CTkFont(size=13, weight="bold"), width=40).grid(row=row, column=1)

        btn_plus = ctk.CTkButton(parent, text="+", width=50, height=40, font=jog_font, fg_color=fg, hover_color=h_color)
        btn_plus.grid(row=row, column=2, padx=5, pady=2)
        btn_plus.bind("<ButtonPress-1>", lambda e, idx=ax_idx: self.start_jog(idx, 1))
        btn_plus.bind("<ButtonRelease-1>", self.stop_jog)
        btn_plus.bind("<Leave>", self.stop_jog)

    # ---------------- 조그 로직 ----------------
    def start_jog(self, axis_idx, direction):
        if self.current_task_thread and self.current_task_thread.is_alive(): return
        if not self.task_pos: return

        self.is_jogging = True
        stop_event.clear()
        self.current_task_thread = threading.Thread(target=self._jog_loop, args=(axis_idx, direction), daemon=True)
        self.current_task_thread.start()

    def _jog_loop(self, axis_idx, direction):
        target_pos = list(self.task_pos)
        if axis_idx < 3: target_pos[axis_idx] += direction * 1.0 
        else: target_pos[axis_idx] += direction * 90.0 
            
        print(">> [조그] 이동 중...")
        try: indy.task_move_to(target_pos); move_done_check() 
        except InterruptedError: print(">> [조그] 정지됨.")
        except Exception: pass
        finally: self.is_jogging = False

    def stop_jog(self, event=None):
        if self.is_jogging:
            self.is_jogging = False
            stop_event.set() 

    # ---------------- 📊 실시간 모니터링 (텍스트 + 3D 업데이트) ----------------
    def _poll_position(self):
        while True:
            if indy is not None:
                try: 
                    self.task_pos = indy.get_task_pos()
                    self.joint_pos = indy.get_joint_pos()
                except Exception: 
                    self.task_pos = None
                    self.joint_pos = None
            time.sleep(0.05) # 3D 뷰의 부드러움을 위해 수집 주기를 짧게(50ms)

    def update_gui_coordinates(self):
        if self.task_pos and self.joint_pos:
            t = self.task_pos
            j = self.joint_pos

            # 1. 텍스트 업데이트
            task_str = f"X: {t[0]:.4f}  Y: {t[1]:.4f}  Z: {t[2]:.4f}\nRx:{t[3]:.2f}° Ry:{t[4]:.2f}° Rz:{t[5]:.2f}°"
            self.task_label.configure(text=task_str, text_color="#4FC3F7")

            joint_str = f"J1:{j[0]:.2f}° J2:{j[1]:.2f}° J3:{j[2]:.2f}°\nJ4:{j[3]:.2f}° J5:{j[4]:.2f}° J6:{j[5]:.2f}°"
            self.joint_label.configure(text=joint_str, text_color="#A5D6A7")

            # 2. 3D 그래픽 업데이트 (Base 0,0,0 -> 어깨 0,0,0.2 -> Tool t[x,y,z])
            x, y, z = t[0], t[1], t[2]
            
            # 로봇 팔 관절 모양 그리기 (가상의 어깨 포인트 거침)
            self.robot_arm_line.set_data([0, 0, x], [0, 0, y])
            self.robot_arm_line.set_3d_properties([0, 0.2, z])

            # 궤적(Trail) 남기기 (최근 30개의 이동 포인트 기억)
            self.history_x.append(x)
            self.history_y.append(y)
            self.history_z.append(z)
            if len(self.history_x) > 30:
                self.history_x.pop(0)
                self.history_y.pop(0)
                self.history_z.pop(0)

            self.robot_trail.set_data(self.history_x, self.history_y)
            self.robot_trail.set_3d_properties(self.history_z)
            
            # 캔버스 다시 그리기 (화면 갱신)
            self.canvas.draw_idle()

        else:
            self.task_label.configure(text="연결 끊김", text_color="#F44336")
            self.joint_label.configure(text="연결 끊김", text_color="#F44336")
            
        # 100ms 마다 그래픽 갱신 (10fps 수준)
        self.after(100, self.update_gui_coordinates)

    # ---------------- 기본 제어 기능들 ----------------
    def run_task1(self):
        if self.current_task_thread and self.current_task_thread.is_alive(): return
        stop_event.clear(); self.current_task_thread = threading.Thread(target=task_pal_1by2_2layer, daemon=True); self.current_task_thread.start()

    def run_task2(self):
        if self.current_task_thread and self.current_task_thread.is_alive(): return
        stop_event.clear(); self.current_task_thread = threading.Thread(target=task_pick2by2_place2by2, daemon=True); self.current_task_thread.start()

    def run_home(self):
        def _home_logic():
            if self.current_task_thread and self.current_task_thread.is_alive():
                print(">> 🛑 진행 중인 동작을 중단하고 홈으로 복귀합니다...")
                stop_event.set(); self.current_task_thread.join(timeout=2.0)
            else: print(">> 홈으로 이동합니다...")
            stop_event.clear(); indy.set_do(2, False); indy.reset_robot(); time.sleep(1.0)
            indy.go_home()
            try: move_done_check(); print(">> 홈 이동 완료")
            except InterruptedError: pass
        threading.Thread(target=_home_logic, daemon=True).start()

    def run_stop(self):
        if self.current_task_thread and self.current_task_thread.is_alive():
            print(">> 🛑 모든 동작을 즉시 중단합니다...")
            stop_event.set()

    def on_closing(self):
        if messagebox.askokcancel("종료", "프로그램을 종료하시겠습니까?"):
            if self.current_task_thread and self.current_task_thread.is_alive():
                stop_event.set(); self.current_task_thread.join(timeout=1.0)
            self.quit(); self.destroy()

def main():
    app = RobotApp()
    try: app.mainloop()
    except Exception as e: print(f"\n>> 시스템 오류 발생: {e}")
    finally:
        print("\n>> 통신을 해제합니다...")
        if indy is not None:
            try: stop_event.set(); time.sleep(0.1); indy.disconnect(); print(">> 로봇 연결 해제 완료.")
            except: pass
        sys.exit(0)

if __name__ == "__main__":
    main()