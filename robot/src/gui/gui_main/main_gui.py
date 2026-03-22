# main_gui.py
import customtkinter as ctk
import tkinter.messagebox as messagebox
import threading
import sys
import time
import numpy as np 

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from robot_logic import (
    indy, stop_event, task_pal_1by2_2layer, task_pick2by2_place2by2, move_done_check
)

# ---- 전역 테마 설정 ----
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# UI 색상 팔레트
BG_COLOR = "#0D0D0F"        
PANEL_COLOR = "#18181B"     
ACCENT_GREEN = "#00FF41"    
ACCENT_BLUE = "#00E5FF"
ACCENT_RED = "#FF1744"
TEXT_MUTED = "#8B8B96"

class PrintLogger:
    def __init__(self, textbox): self.textbox = textbox
    def write(self, text):
        if text.strip() or text == '\n': self.textbox.after(0, self._insert_text, text)
    def _insert_text(self, text):
        self.textbox.insert(ctk.END, text); self.textbox.see(ctk.END)
    def flush(self): pass

class RobotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Indy7 Command Center (Digital Twin HUD)")
        self.geometry("1380x880") 
        self.configure(fg_color=BG_COLOR)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.current_task_thread = None
        self.task_pos = None  
        self.joint_pos = None 
        self.is_jogging = False 

        # ----------------------------------------------------
        # Modified DH 파라미터 
        # ----------------------------------------------------
        self.dh_params = [
            {"a": 0.0,    "alpha": 0.0,       "d": 0.3,    "theta_offset": 0.0},
            {"a": 0.0,    "alpha": np.pi/2,   "d": 0.0,    "theta_offset": np.pi/2},
            {"a": 0.45,   "alpha": 0.0,       "d": 0.0035, "theta_offset": np.pi/2},
            {"a": 0.0,    "alpha": np.pi/2,   "d": 0.35,   "theta_offset": np.pi},
            {"a": 0.0,    "alpha": np.pi/2,   "d": 0.1835, "theta_offset": 0.0},
            {"a": 0.0,    "alpha": -np.pi/2,  "d": 0.228,  "theta_offset": 0.0}
        ]
        self.history_x, self.history_y, self.history_z = [], [], []

        # ==========================================
        # 3단 와이드 레이아웃 그리드 설정
        # ==========================================
        self.grid_columnconfigure(0, weight=0, minsize=270) 
        self.grid_columnconfigure(1, weight=1)             
        self.grid_columnconfigure(2, weight=0, minsize=330) 
        self.grid_rowconfigure(0, weight=4)                 
        self.grid_rowconfigure(1, weight=1, minsize=160)
        self.grid_rowconfigure(2, weight=0) 

        # ------------------------------------------
        # 1. 좌측 패널: 액션 제어 (Scrollable)
        # ------------------------------------------
        # 일반 Frame에서 ScrollableFrame으로 변경하고 스크롤바 색상을 어둡게 설정
        self.left_panel = ctk.CTkScrollableFrame(self, fg_color=PANEL_COLOR, corner_radius=12, scrollbar_button_color="#2A2A35", scrollbar_button_hover_color="#3F3F46")
        self.left_panel.grid(row=0, column=0, padx=(15, 10), pady=(15, 10), sticky="nsew")
        self.left_panel.grid_columnconfigure(0, weight=1)

        logo_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        logo_frame.grid(row=0, column=0, pady=(15, 20), sticky="ew")
        ctk.CTkLabel(logo_frame, text="⚡ INDY7", font=ctk.CTkFont(family="Arial Black", size=24, weight="bold"), text_color=ACCENT_BLUE).pack()
        ctk.CTkLabel(logo_frame, text="SYSTEM ONLINE", font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT_GREEN).pack()

        btn_font = ctk.CTkFont(size=14, weight="bold")
        
        ctk.CTkLabel(self.left_panel, text="AUTO TASKS", font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_MUTED).grid(row=1, column=0, pady=(10, 5), padx=10, sticky="w")
        self.btn_task1 = ctk.CTkButton(self.left_panel, text="📦 작업 1 (1x2 2단)", font=btn_font, height=45, corner_radius=8, fg_color="#2E7D32", hover_color="#1B5E20", command=self.run_task1)
        self.btn_task1.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.btn_task2 = ctk.CTkButton(self.left_panel, text="⚡ 작업 2 (2x2 Pick)", font=btn_font, height=45, corner_radius=8, fg_color="#1976D2", hover_color="#0D47A1", command=self.run_task2)
        self.btn_task2.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")

        ctk.CTkFrame(self.left_panel, height=30, fg_color="transparent").grid(row=4, column=0) 

        ctk.CTkLabel(self.left_panel, text="SYSTEM OVERRIDE", font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_MUTED).grid(row=5, column=0, pady=(0, 5), padx=10, sticky="w")
        self.btn_home = ctk.CTkButton(self.left_panel, text="🏠 홈 위치 복귀", font=btn_font, height=45, corner_radius=8, fg_color="#F57C00", hover_color="#E65100", command=self.run_home)
        self.btn_home.grid(row=6, column=0, padx=10, pady=(0, 10), sticky="ew")
        
        self.btn_stop = ctk.CTkButton(self.left_panel, text="🛑 긴급 정지 (E-STOP)", font=ctk.CTkFont(size=15, weight="bold"), height=55, corner_radius=8, fg_color=ACCENT_RED, hover_color="#B71C1C", command=self.run_stop)
        self.btn_stop.grid(row=7, column=0, padx=10, pady=(0, 10), sticky="ew")

        ctk.CTkFrame(self.left_panel, height=30, fg_color="transparent").grid(row=8, column=0) 

        ctk.CTkLabel(self.left_panel, text="DISPLAY ZOOM", font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_MUTED).grid(row=9, column=0, pady=(0, 5), padx=10, sticky="w")
        self.scale_menu = ctk.CTkSegmentedButton(self.left_panel, values=["100%", "125%"], command=self.change_scaling, selected_color=ACCENT_BLUE, selected_hover_color="#00B8D4")
        self.scale_menu.grid(row=10, column=0, padx=10, pady=(0, 20), sticky="ew")
        self.scale_menu.set("100%")

        # ------------------------------------------
        # 2. 중앙 패널: 3D 디지털 트윈
        # ------------------------------------------
        self.center_panel = ctk.CTkFrame(self, fg_color=PANEL_COLOR, corner_radius=12)
        self.center_panel.grid(row=0, column=1, padx=5, pady=(15, 10), sticky="nsew")
        
        header_frame = ctk.CTkFrame(self.center_panel, fg_color="transparent", height=40)
        header_frame.pack(fill="x", padx=15, pady=(10, 0))
        ctk.CTkLabel(header_frame, text="🛰️ 3D DIGITAL TWIN VIEWER", font=ctk.CTkFont(size=14, weight="bold"), text_color="#FFFFFF").pack(side="left")
        ctk.CTkLabel(header_frame, text="LIVE", font=ctk.CTkFont(size=12, weight="bold"), text_color=ACCENT_RED).pack(side="right")

        self.fig = plt.Figure(figsize=(6, 5), facecolor=PANEL_COLOR)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor(PANEL_COLOR)
        
        self.ax.xaxis.set_pane_color((0.09, 0.09, 0.11, 1.0))
        self.ax.yaxis.set_pane_color((0.09, 0.09, 0.11, 1.0))
        self.ax.zaxis.set_pane_color((0.09, 0.09, 0.11, 1.0))
        self.ax.grid(color='#2A2A35', linestyle=':', linewidth=0.5)
        self.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        self.ax.set_xlabel('X (m)', color=TEXT_MUTED, labelpad=5)
        self.ax.set_ylabel('Y (m)', color=TEXT_MUTED, labelpad=5)
        self.ax.set_zlabel('Z (m)', color=TEXT_MUTED, labelpad=5)
        
        self.ax.set_xlim([-0.8, 0.8])
        self.ax.set_ylim([-0.8, 0.8])
        self.ax.set_zlim([0, 1.0])

        self.robot_arm_line, = self.ax.plot([], [], [], '-', color=ACCENT_GREEN, lw=3)
        self.robot_joints, = self.ax.plot([], [], [], 'o', color=ACCENT_GREEN, markersize=7, markerfacecolor='white', markeredgecolor=ACCENT_GREEN, markeredgewidth=2)
        self.robot_trail, = self.ax.plot([], [], [], '-', color=ACCENT_BLUE, alpha=0.4, lw=1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.center_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=(0, 10))

        # ------------------------------------------
        # 3. 우측 패널: 텔레메트리 & 조그
        # ------------------------------------------
        self.right_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.right_panel.grid(row=0, column=2, padx=(10, 15), pady=(15, 10), sticky="nsew")
        self.right_panel.grid_rowconfigure(1, weight=1)

        self.telemetry_frame = ctk.CTkFrame(self.right_panel, fg_color=PANEL_COLOR, corner_radius=12)
        self.telemetry_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(self.telemetry_frame, text="📡 TELEMETRY DATA", font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_MUTED).pack(pady=(15, 5))
        
        task_info_box = ctk.CTkFrame(self.telemetry_frame, fg_color="#121215", corner_radius=6)
        task_info_box.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(task_info_box, text="TCP POS (Base ➔ Tool)", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED).pack(anchor="w", padx=10, pady=(5,0))
        self.task_label = ctk.CTkLabel(task_info_box, text="WAITING SIGNAL...", font=ctk.CTkFont(family="Consolas", size=14, weight="bold"), text_color=ACCENT_BLUE, justify="left")
        self.task_label.pack(anchor="w", padx=10, pady=(0, 8))

        joint_info_box = ctk.CTkFrame(self.telemetry_frame, fg_color="#121215", corner_radius=6)
        joint_info_box.pack(fill="x", padx=15, pady=(5, 15))
        ctk.CTkLabel(joint_info_box, text="JOINT ANGLES (J1 ~ J6)", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED).pack(anchor="w", padx=10, pady=(5,0))
        self.joint_label = ctk.CTkLabel(joint_info_box, text="WAITING SIGNAL...", font=ctk.CTkFont(family="Consolas", size=14, weight="bold"), text_color=ACCENT_GREEN, justify="left")
        self.joint_label.pack(anchor="w", padx=10, pady=(0, 8))

        # 조그 탭 뷰어 생성
        self.jog_tabs = ctk.CTkTabview(self.right_panel, corner_radius=12, fg_color=PANEL_COLOR, segmented_button_selected_color="#3F3F46", segmented_button_selected_hover_color="#52525B")
        self.jog_tabs.pack(fill="both", expand=True)
        
        self.jog_tabs.add("  Task Jog (X,Y,Z)  ")
        self.jog_tabs.add("  Joint Jog (J1~J6)  ")

        task_tab = self.jog_tabs.tab("  Task Jog (X,Y,Z)  ")
        task_scroll = ctk.CTkScrollableFrame(task_tab, fg_color="transparent", scrollbar_button_color="#2A2A35", scrollbar_button_hover_color="#3F3F46")
        task_scroll.pack(fill="both", expand=True)

        pos_frame = ctk.CTkFrame(task_scroll, fg_color="transparent")
        pos_frame.pack(pady=(15, 10))
        for row, (ax_name, ax_idx) in enumerate([("X축", 0), ("Y축", 1), ("Z축", 2)]):
            self.create_jog_buttons(pos_frame, row, ax_name, ax_idx, 'task', "#27272A", "#3F3F46", ACCENT_BLUE)

        rot_frame = ctk.CTkFrame(task_scroll, fg_color="transparent")
        rot_frame.pack(pady=(5, 10))
        for row, (ax_name, ax_idx) in enumerate([("Rx", 3), ("Ry", 4), ("Rz", 5)]):
            self.create_jog_buttons(rot_frame, row, ax_name, ax_idx, 'task', "#27272A", "#3F3F46", "#B388FF")

        joint_tab = self.jog_tabs.tab("  Joint Jog (J1~J6)  ")
        joint_scroll = ctk.CTkScrollableFrame(joint_tab, fg_color="transparent", scrollbar_button_color="#2A2A35", scrollbar_button_hover_color="#3F3F46")
        joint_scroll.pack(fill="both", expand=True)

        j_frame_1 = ctk.CTkFrame(joint_scroll, fg_color="transparent")
        j_frame_1.pack(pady=(15, 10))
        for row, (ax_name, ax_idx) in enumerate([("J1", 0), ("J2", 1), ("J3", 2)]):
            self.create_jog_buttons(j_frame_1, row, ax_name, ax_idx, 'joint', "#27272A", "#3F3F46", ACCENT_GREEN)

        j_frame_2 = ctk.CTkFrame(joint_scroll, fg_color="transparent")
        j_frame_2.pack(pady=(5, 10))
        for row, (ax_name, ax_idx) in enumerate([("J4", 3), ("J5", 4), ("J6", 5)]):
            self.create_jog_buttons(j_frame_2, row, ax_name, ax_idx, 'joint', "#27272A", "#3F3F46", ACCENT_GREEN)

        # ------------------------------------------
        # 4. 하단 패널: 시스템 로그 터미널
        # ------------------------------------------
        self.bottom_panel = ctk.CTkFrame(self, fg_color=PANEL_COLOR, corner_radius=12)
        self.bottom_panel.grid(row=1, column=0, columnspan=3, padx=15, pady=(5, 15), sticky="nsew")
        
        ctk.CTkLabel(self.bottom_panel, text=">_ SYSTEM TERMINAL", font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_MUTED).pack(anchor="w", padx=15, pady=(10, 0))
        
        self.log_text = ctk.CTkTextbox(self.bottom_panel, font=ctk.CTkFont(family="Consolas", size=13), fg_color="#0A0A0C", text_color=ACCENT_GREEN, wrap="word", corner_radius=8)
        self.log_text.pack(fill="both", expand=True, padx=15, pady=(5, 10))
        sys.stdout = PrintLogger(self.log_text)

        footer_frame = ctk.CTkFrame(self, fg_color="transparent")
        footer_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=20, pady=(0, 10))
        ctk.CTkLabel(footer_frame, text="Made by 정빈", font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_MUTED).pack(side="right")

        print("=== Indy7 Command Center Initialized ===")
        print(">> 시스템 준비 완료. 실시간 모니터링을 시작합니다.")

        threading.Thread(target=self._poll_position, daemon=True).start()
        self.update_gui_coordinates()

    # ----------------------------------------------------
    # UI 배율 줌(Zoom) 기능
    # ----------------------------------------------------
    def change_scaling(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100.0
        ctk.set_widget_scaling(new_scaling_float)

    def compute_forward_kinematics(self, joint_angles):
        T_matrices = [np.eye(4)] 
        T = np.eye(4)
        for i in range(6):
            theta = np.radians(joint_angles[i]) + self.dh_params[i]['theta_offset']
            a = self.dh_params[i]['a']
            alpha = self.dh_params[i]['alpha']
            d = self.dh_params[i]['d']
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            T_i = np.array([
                [           ct,            -st,   0,             a],
                [        st*ca,          ct*ca, -sa,         -d*sa],
                [        st*sa,          ct*sa,  ca,          d*ca],
                [            0,              0,   0,             1]
            ])
            T = T @ T_i 
            T_matrices.append(T)
        return T_matrices

    def create_jog_buttons(self, parent, row, ax_name, ax_idx, mode, bg, h_color, text_col):
        jog_font = ctk.CTkFont(size=18, weight="bold")
        
        btn_minus = ctk.CTkButton(parent, text="-", width=50, height=40, font=jog_font, fg_color=bg, text_color="#FFFFFF", hover_color=h_color, corner_radius=6)
        btn_minus.grid(row=row, column=0, padx=8, pady=4)
        btn_minus.bind("<ButtonPress-1>", lambda e, idx=ax_idx, m=mode: self.start_jog(idx, -1, m))
        btn_minus.bind("<ButtonRelease-1>", self.stop_jog)
        btn_minus.bind("<Leave>", self.stop_jog)

        ctk.CTkLabel(parent, text=ax_name, font=ctk.CTkFont(size=13, weight="bold"), text_color=text_col).grid(row=row, column=1, padx=5)

        btn_plus = ctk.CTkButton(parent, text="+", width=50, height=40, font=jog_font, fg_color=bg, text_color="#FFFFFF", hover_color=h_color, corner_radius=6)
        btn_plus.grid(row=row, column=2, padx=8, pady=4)
        btn_plus.bind("<ButtonPress-1>", lambda e, idx=ax_idx, m=mode: self.start_jog(idx, 1, m))
        btn_plus.bind("<ButtonRelease-1>", self.stop_jog)
        btn_plus.bind("<Leave>", self.stop_jog)

    def start_jog(self, axis_idx, direction, mode):
        if self.current_task_thread and self.current_task_thread.is_alive(): return
        if not self.task_pos or not self.joint_pos: return
        self.is_jogging = True
        stop_event.clear()
        self.current_task_thread = threading.Thread(target=self._jog_loop, args=(axis_idx, direction, mode), daemon=True)
        self.current_task_thread.start()

    def _jog_loop(self, axis_idx, direction, mode):
        if mode == 'task':
            target_pos = list(self.task_pos)
            if axis_idx < 3: target_pos[axis_idx] += direction * 1.0 
            else: target_pos[axis_idx] += direction * 90.0 
            print(">> [Task] 직교 제어 이동 중...")
            try: indy.task_move_to(target_pos); move_done_check() 
            except InterruptedError: print(">> [정지] 조그 이동 완료.")
            except Exception: pass
            finally: self.is_jogging = False
            
        elif mode == 'joint':
            target_joint = list(self.joint_pos)
            target_joint[axis_idx] += direction * 90.0 
            print(f">> [Joint] J{axis_idx+1} 모터 단독 회전 중...")
            try: indy.joint_move_to(target_joint); move_done_check() 
            except InterruptedError: print(">> [정지] 관절 이동 완료.")
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

            task_str = f"X: {t[0]: 7.4f}   Y: {t[1]: 7.4f}   Z: {t[2]: 7.4f}\nRx:{t[3]: 7.2f}°  Ry:{t[4]: 7.2f}°  Rz:{t[5]: 7.2f}°"
            self.task_label.configure(text=task_str, text_color=ACCENT_BLUE)

            joint_str = f"J1:{j[0]: 7.2f}°  J2:{j[1]: 7.2f}°  J3:{j[2]: 7.2f}°\nJ4:{j[3]: 7.2f}°  J5:{j[4]: 7.2f}°  J6:{j[5]: 7.2f}°"
            self.joint_label.configure(text=joint_str, text_color=ACCENT_GREEN)

            T = self.compute_forward_kinematics(j)
            offset_y = 0.1835  

            P0 = T[0][:3, 3]; P1 = T[1][:3, 3] 
            P2 = (T[2] @ np.array([0, 0, offset_y, 1]))[:3] 
            P3 = (T[3] @ np.array([0, 0, offset_y, 1]))[:3] 
            P3_corner = T[3][:3, 3]; P4 = T[4][:3, 3]; P5 = T[5][:3, 3]; P6 = T[6][:3, 3] 

            line_xs = [P0[0], P1[0], P2[0], P3[0], P3_corner[0], P4[0], P5[0], P6[0]]
            line_ys = [P0[1], P1[1], P2[1], P3[1], P3_corner[1], P4[1], P5[1], P6[1]]
            line_zs = [P0[2], P1[2], P2[2], P3[2], P3_corner[2], P4[2], P5[2], P6[2]]
            self.robot_arm_line.set_data(line_xs, line_ys)
            self.robot_arm_line.set_3d_properties(line_zs)

            joint_xs = [P0[0], P2[0], P3[0], P4[0], P5[0], P6[0]]
            joint_ys = [P0[1], P2[1], P3[1], P4[1], P5[1], P6[1]]
            joint_zs = [P0[2], P2[2], P3[2], P4[2], P5[2], P6[2]]
            self.robot_joints.set_data(joint_xs, joint_ys)
            self.robot_joints.set_3d_properties(joint_zs)

            self.history_x.append(P6[0]); self.history_y.append(P6[1]); self.history_z.append(P6[2])
            if len(self.history_x) > 35:
                self.history_x.pop(0); self.history_y.pop(0); self.history_z.pop(0)

            self.robot_trail.set_data(self.history_x, self.history_y)
            self.robot_trail.set_3d_properties(self.history_z)
            self.canvas.draw_idle()

        else:
            self.task_label.configure(text="연결 대기중 / 로봇 전원 확인", text_color=ACCENT_RED)
            self.joint_label.configure(text="연결 대기중 / 로봇 전원 확인", text_color=ACCENT_RED)
            
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
        if messagebox.askokcancel("시스템 종료", "디지털 트윈 제어 센터를 종료하시겠습니까?"):
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