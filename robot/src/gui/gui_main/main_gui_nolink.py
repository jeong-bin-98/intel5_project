# main_gui.py
import customtkinter as ctk
import tkinter.messagebox as messagebox
import threading
import sys
import time
import numpy as np # ⭐ 행렬 계산을 위한 numpy 추가

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
        
        self.title("Indy7 6-Axis Digital Twin Center")
        self.geometry("1200x850") 
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.current_task_thread = None
        self.task_pos = None  
        self.joint_pos = None 
        self.is_jogging = False 

        # ----------------------------------------------------
        # ⭐ [핵심] Indy7 (또는 유사 6축 로봇) DH 파라미터 테이블 
        # (실제 뉴로메카 매뉴얼의 a, alpha, d, theta 오프셋 값으로 수정하세요!)
        # 단위: 미터(m), 라디안(rad)
        # ----------------------------------------------------
        self.dh_params = [
            # Link 1: a=0, alpha=0, d=0.3, theta=q1
            {"a": 0.0,    "alpha": 0.0,       "d": 0.3,    "theta_offset": 0.0},
            
            # Link 2: a=0, alpha=pi/2, d=0, theta=q2 + pi/2
            {"a": 0.0,    "alpha": np.pi/2,   "d": 0.0,    "theta_offset": np.pi/2},
            
            # Link 3: a=0.45, alpha=0, d=0.0035, theta=q3 + pi/2
            {"a": 0.45,   "alpha": 0.0,       "d": 0.0035, "theta_offset": np.pi/2},
            
            # Link 4: a=0, alpha=pi/2, d=0.35, theta=q4 + pi
            {"a": 0.0,    "alpha": np.pi/2,   "d": 0.35,   "theta_offset": np.pi},
            
            # Link 5: a=0, alpha=pi/2, d=0.1835, theta=q5
            {"a": 0.0,    "alpha": np.pi/2,   "d": 0.1835, "theta_offset": 0.0},
            
            # Link 6: a=0, alpha=-pi/2, d=0.228, theta=q6
            {"a": 0.0,    "alpha": -np.pi/2,  "d": 0.228,  "theta_offset": 0.0}
        ]

        # 3D 궤적 리스트
        self.history_x, self.history_y, self.history_z = [], [], []

        # (UI 초기화 코드는 이전과 동일합니다)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # === 1. 왼쪽 사이드바 ===
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="🤖 Indy7 System", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        self.btn_task1 = ctk.CTkButton(self.sidebar, text="📦 작업 1 (1x2 2단)", font=ctk.CTkFont(size=14, weight="bold"), height=45, fg_color="#2E7D32", hover_color="#1B5E20", command=self.run_task1)
        self.btn_task1.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.btn_task2 = ctk.CTkButton(self.sidebar, text="⚡ 작업 2 (2x2 Pick)", font=ctk.CTkFont(size=14, weight="bold"), height=45, fg_color="#1976D2", hover_color="#0D47A1", command=self.run_task2)
        self.btn_task2.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.btn_home = ctk.CTkButton(self.sidebar, text="🏠 홈 이동", font=ctk.CTkFont(size=14, weight="bold"), height=45, fg_color="#F57C00", hover_color="#E65100", command=self.run_home)
        self.btn_home.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        self.btn_stop = ctk.CTkButton(self.sidebar, text="🛑 긴급 중지 (Q)", font=ctk.CTkFont(size=14, weight="bold"), height=45, fg_color="#D32F2F", hover_color="#B71C1C", command=self.run_stop)
        self.btn_stop.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        self.log_frame = ctk.CTkFrame(self.sidebar, fg_color="#121212")
        self.log_frame.grid(row=5, column=0, padx=10, pady=20, sticky="nsew")
        self.log_text = ctk.CTkTextbox(self.log_frame, font=ctk.CTkFont(family="Consolas", size=12), fg_color="transparent", text_color="#00FF41", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        sys.stdout = PrintLogger(self.log_text)

        # === 2. 오른쪽 메인 영역 ===
        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_area.grid_columnconfigure(0, weight=3) 
        self.main_area.grid_columnconfigure(1, weight=1) 
        self.main_area.grid_rowconfigure(0, weight=1)    
        
        # --- (1) 3D 실시간 모니터링 뷰어 ---
        self.plot_frame = ctk.CTkFrame(self.main_area, corner_radius=10, fg_color="#1E1E1E")
        self.plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        
        ctk.CTkLabel(self.plot_frame, text="🛰️ 6-Axis Digital Twin Viewer", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.fig = plt.Figure(figsize=(6, 5), facecolor='#1E1E1E')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1E1E1E')
        
        self.ax.xaxis.set_pane_color((0.15, 0.15, 0.15, 1.0))
        self.ax.yaxis.set_pane_color((0.15, 0.15, 0.15, 1.0))
        self.ax.zaxis.set_pane_color((0.15, 0.15, 0.15, 1.0))
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.set_xlabel('X (m)', color='white')
        self.ax.set_ylabel('Y (m)', color='white')
        self.ax.set_zlabel('Z (m)', color='white')
        
        self.ax.set_xlim([-0.8, 0.8])
        self.ax.set_ylim([-0.8, 0.8])
        self.ax.set_zlim([0, 1.0])

        # 6개의 관절을 이을 로봇 뼈대 (관절 포인트에 흰색 마커 추가)
        self.robot_arm_line, = self.ax.plot([], [], [], 'o-', color='#00FF41', lw=4, markersize=7, markerfacecolor='white', markeredgecolor='#00FF41')
        self.robot_trail, = self.ax.plot([], [], [], '-', color='#4FC3F7', alpha=0.5, lw=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # --- (2) 수치 데이터 & Jog 패드 ---
        self.info_frame = ctk.CTkFrame(self.main_area, fg_color="transparent")
        self.info_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
        self.info_frame.grid_rowconfigure(2, weight=1)

        self.coord_frame = ctk.CTkFrame(self.info_frame, fg_color="#2A2D34", corner_radius=10)
        self.coord_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(self.coord_frame, text="[ Task 좌표 (Base ➔ Tool) ]", font=ctk.CTkFont(size=12, weight="bold"), text_color="#AAAAAA").pack(pady=(15, 0))
        self.task_label = ctk.CTkLabel(self.coord_frame, text="수신 대기 중...", font=ctk.CTkFont(family="Consolas", size=14, weight="bold"), text_color="#4FC3F7")
        self.task_label.pack(pady=(5, 15))

        ctk.CTkLabel(self.coord_frame, text="[ Joint 각도 (J1 ~ J6) ]", font=ctk.CTkFont(size=12, weight="bold"), text_color="#AAAAAA").pack(pady=(5, 0))
        self.joint_label = ctk.CTkLabel(self.coord_frame, text="수신 대기 중...", font=ctk.CTkFont(family="Consolas", size=14, weight="bold"), text_color="#A5D6A7")
        self.joint_label.pack(pady=(5, 15))

        self.jog_frame = ctk.CTkFrame(self.info_frame, corner_radius=10)
        self.jog_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(self.jog_frame, text="🕹️ Hold & Jog (수동 조작)", font=ctk.CTkFont(size=15, weight="bold")).pack(pady=15)

        pos_frame = ctk.CTkFrame(self.jog_frame, fg_color="transparent")
        pos_frame.pack(pady=5)
        for row, (ax_name, ax_idx) in enumerate([("X축", 0), ("Y축", 1), ("Z축", 2)]):
            self.create_jog_buttons(pos_frame, row, ax_name, ax_idx, "#2E7D32", "#1B5E20")

        rot_frame = ctk.CTkFrame(self.jog_frame, fg_color="transparent")
        rot_frame.pack(pady=15)
        for row, (ax_name, ax_idx) in enumerate([("Rx", 3), ("Ry", 4), ("Rz", 5)]):
            self.create_jog_buttons(rot_frame, row, ax_name, ax_idx, "#6A1B9A", "#4A148C")

        print("=== 6-Axis Digital Twin Ready ===")
        threading.Thread(target=self._poll_position, daemon=True).start()
        self.update_gui_coordinates()

    # ----------------------------------------------------
    # ⭐ [핵심 추가] 정기구학 (Forward Kinematics) 연산 함수
    # ----------------------------------------------------
    def compute_forward_kinematics(self, joint_angles):
        points = [[0, 0, 0]] # Base 좌표
        T = np.eye(4)
        
        for i in range(6):
            # 주어진 표의 q_i 에 해당하는 실제 모터 각도 + 표의 오프셋
            theta = np.radians(joint_angles[i]) + self.dh_params[i]['theta_offset']
            a = self.dh_params[i]['a']
            alpha = self.dh_params[i]['alpha']
            d = self.dh_params[i]['d']
            
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            
            # 💡 일반 DH가 아닌 Modified DH용 변환 행렬 공식 적용
            T_i = np.array([
                [           ct,            -st,   0,             a],
                [        st*ca,          ct*ca, -sa,         -d*sa],
                [        st*sa,          ct*sa,  ca,          d*ca],
                [            0,              0,   0,             1]
            ])
            
            T = T @ T_i 
            points.append([T[0,3], T[1,3], T[2,3]])
            
        return np.array(points)

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

    def _poll_position(self):
        while True:
            if indy is not None:
                try: 
                    self.task_pos = indy.get_task_pos()
                    self.joint_pos = indy.get_joint_pos()
                except Exception: 
                    self.task_pos = None; self.joint_pos = None
            time.sleep(0.05)

    def update_gui_coordinates(self):
        if self.task_pos and self.joint_pos:
            t = self.task_pos
            j = self.joint_pos

            task_str = f"X: {t[0]:.4f}  Y: {t[1]:.4f}  Z: {t[2]:.4f}\nRx:{t[3]:.2f}° Ry:{t[4]:.2f}° Rz:{t[5]:.2f}°"
            self.task_label.configure(text=task_str, text_color="#4FC3F7")

            joint_str = f"J1:{j[0]:.2f}° J2:{j[1]:.2f}° J3:{j[2]:.2f}°\nJ4:{j[3]:.2f}° J5:{j[4]:.2f}° J6:{j[5]:.2f}°"
            self.joint_label.configure(text=joint_str, text_color="#A5D6A7")

            # ⭐ 정기구학 연산을 통해 6개 관절의 3D 꺾임 위치 계산
            points = self.compute_forward_kinematics(j)
            xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]

            # 6축 뼈대 그리기
            self.robot_arm_line.set_data(xs, ys)
            self.robot_arm_line.set_3d_properties(zs)

            # 궤적 남기기 (가장 마지막 좌표인 Tool 끝단 기준)
            tool_x, tool_y, tool_z = xs[-1], ys[-1], zs[-1]
            self.history_x.append(tool_x)
            self.history_y.append(tool_y)
            self.history_z.append(tool_z)
            if len(self.history_x) > 30:
                self.history_x.pop(0); self.history_y.pop(0); self.history_z.pop(0)

            self.robot_trail.set_data(self.history_x, self.history_y)
            self.robot_trail.set_3d_properties(self.history_z)
            
            self.canvas.draw_idle()

        else:
            self.task_label.configure(text="연결 끊김", text_color="#F44336")
            self.joint_label.configure(text="연결 끊김", text_color="#F44336")
            
        self.after(100, self.update_gui_coordinates)

    def run_task1(self):
        if self.current_task_thread and self.current_task_thread.is_alive(): return
        stop_event.clear(); self.current_task_thread = threading.Thread(target=task_pal_1by2_2layer, daemon=True); self.current_task_thread.start()

    def run_task2(self):
        if self.current_task_thread and self.current_task_thread.is_alive(): return
        stop_event.clear(); self.current_task_thread = threading.Thread(target=task_pick2by2_place2by2, daemon=True); self.current_task_thread.start()

    def run_home(self):
        def _home_logic():
            if self.current_task_thread and self.current_task_thread.is_alive():
                stop_event.set(); self.current_task_thread.join(timeout=2.0)
            stop_event.clear(); indy.set_do(2, False); indy.reset_robot(); time.sleep(1.0)
            indy.go_home()
            try: move_done_check()
            except InterruptedError: pass
        threading.Thread(target=_home_logic, daemon=True).start()

    def run_stop(self):
        if self.current_task_thread and self.current_task_thread.is_alive(): stop_event.set()

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
        if indy is not None:
            try: stop_event.set(); time.sleep(0.1); indy.disconnect()
            except: pass
        sys.exit(0)

if __name__ == "__main__":
    main()