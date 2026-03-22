# indy_drawing_gui.py
# Indy7 그림 그리기 GUI - Blend Radius를 활용한 부드러운 경유점 경로
import customtkinter as ctk
import tkinter.messagebox as messagebox
import threading
import sys
import time
import numpy as np
import math

# ============================================================
# 로봇 연결 설정
# ============================================================
ROBOT_IP = "192.168.3.7"
ROBOT_NAME = "NRMK-Indy7"

# ============================================================
# 그리기 초기 위치 설정
# ============================================================
DRAW_POS = [0.109, 0.300, 0.198]   # [m] X, Y, Z
DRAW_ROT = [0, 180, 90]            # [deg] Rx, Ry, Rz

# ============================================================
# 캔버스 ↔ 로봇 좌표 매핑 설정
# 캔버스 (500x500 px) → 로봇 작업 영역 (0.15m x 0.15m)
# ============================================================
CANVAS_SIZE = 500
WORKSPACE_SIZE = 0.15   # [m] 캔버스가 커버하는 실제 작업 영역 크기

# 캔버스 원점(좌상단)이 매핑되는 로봇 좌표
WORKSPACE_ORIGIN_X = DRAW_POS[0] - WORKSPACE_SIZE / 2
WORKSPACE_ORIGIN_Y = DRAW_POS[1] - WORKSPACE_SIZE / 2

# ============================================================
# 중단 신호 이벤트
# ============================================================
stop_event = threading.Event()

# ============================================================
# 로봇 유틸 함수
# ============================================================
def move_done_check(indy):
    """로봇 동작 완료 대기 (stop_event로 중단 가능)"""
    time.sleep(0.2)
    while True:
        if stop_event.is_set():
            indy.stop_motion()
            time.sleep(0.5)
            raise InterruptedError("강제 중단되었습니다.")

        status = indy.get_robot_status()

        if status.get('error') == 1 or status.get('collision') == 1:
            print("\n🚨 [경고] 로봇 에러(또는 충돌) 감지! 에러를 초기화합니다.")
            indy.reset_robot()
            time.sleep(2.0)
            raise InterruptedError("로봇 에러 발생으로 동작 취소.")

        if status['movedone'] == 1:
            break
        time.sleep(0.1)


def stoppable_sleep(duration):
    """stop_event 감시하면서 대기"""
    start = time.time()
    while time.time() - start < duration:
        if stop_event.is_set():
            raise InterruptedError("강제 중단되었습니다.")
        time.sleep(0.1)


# ============================================================
# Douglas-Peucker 알고리즘 (경로 단순화)
# ============================================================
def douglas_peucker(points, epsilon):
    """경로의 핵심 꺾임점만 추출"""
    if len(points) <= 2:
        return points

    # 시작~끝 직선에서 가장 먼 점 찾기
    start = np.array(points[0])
    end = np.array(points[-1])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return [points[0], points[-1]]

    line_unit = line_vec / line_len

    max_dist = 0
    max_idx = 0
    for i in range(1, len(points) - 1):
        pt = np.array(points[i])
        proj = np.dot(pt - start, line_unit)
        proj = np.clip(proj, 0, line_len)
        closest = start + proj * line_unit
        dist = np.linalg.norm(pt - closest)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > epsilon:
        left = douglas_peucker(points[:max_idx + 1], epsilon)
        right = douglas_peucker(points[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


# ============================================================
# 캔버스 좌표 → 로봇 Task 좌표 변환
# ============================================================
def canvas_to_robot(cx, cy, draw_z, draw_rot):
    """
    캔버스 좌표(px)를 로봇 Task 좌표(m)로 변환
    캔버스 X → 로봇 X, 캔버스 Y → 로봇 Y (부호 반전: 캔버스 아래 = Y 증가)
    """
    robot_x = WORKSPACE_ORIGIN_X + (cx / CANVAS_SIZE) * WORKSPACE_SIZE
    robot_y = WORKSPACE_ORIGIN_Y + ((CANVAS_SIZE - cy) / CANVAS_SIZE) * WORKSPACE_SIZE
    return [robot_x, robot_y, draw_z, *draw_rot]


# ============================================================
# GUI 애플리케이션
# ============================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class PrintLogger:
    """stdout을 GUI 로그 텍스트박스로 리다이렉트"""
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, text):
        if text.strip() or text == '\n':
            self.textbox.after(0, self._insert_text, text)

    def _insert_text(self, text):
        self.textbox.insert(ctk.END, text)
        self.textbox.see(ctk.END)

    def flush(self):
        pass


class DrawingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🎨 Indy7 Drawing GUI - Smooth Waypoint")
        self.geometry("1100x750")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.indy = None
        self.is_connected = False
        self.current_task_thread = None

        # 그리기 데이터
        self.strokes = []          # 획 목록: 각 획 = [(cx, cy), ...]
        self.current_stroke = []   # 현재 그리고 있는 획

        # ============================================================
        # 레이아웃: 왼쪽(설정+로그) | 가운데(캔버스) | 오른쪽(경유점 미리보기)
        # ============================================================
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- 왼쪽 사이드바 (외부 고정 프레임) ----
        self.sidebar_outer = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_outer.grid(row=0, column=0, sticky="nsew")
        self.sidebar_outer.grid_rowconfigure(1, weight=1)  # 스크롤 영역 확장
        self.sidebar_outer.grid_rowconfigure(2, weight=1)  # 로그 확장

        # ---- 스크롤 가능한 사이드바 내부 ----
        self.sidebar = ctk.CTkScrollableFrame(self.sidebar_outer, width=260,
                                               corner_radius=0, fg_color="transparent")
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")

        # 로고
        ctk.CTkLabel(self.sidebar, text="🎨 Indy7 Drawing",
                     font=ctk.CTkFont(size=20, weight="bold")
                     ).pack(padx=20, pady=(20, 15))

        # --- 연결 설정 프레임 ---
        conn_frame = ctk.CTkFrame(self.sidebar, fg_color="#2A2D34", corner_radius=8)
        conn_frame.pack(padx=15, pady=(0, 10), fill="x")

        ctk.CTkLabel(conn_frame, text="로봇 IP:", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, padx=(10, 5), pady=(10, 5), sticky="w")
        self.ip_entry = ctk.CTkEntry(conn_frame, width=140, font=ctk.CTkFont(size=12))
        self.ip_entry.insert(0, ROBOT_IP)
        self.ip_entry.grid(row=0, column=1, padx=(0, 10), pady=(10, 5))

        self.btn_connect = ctk.CTkButton(
            conn_frame, text="🔌 연결", height=35,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1976D2", hover_color="#0D47A1",
            command=self.toggle_connection)
        self.btn_connect.grid(row=1, column=0, columnspan=2, padx=10, pady=(5, 10), sticky="ew")

        # --- 그리기 설정 프레임 ---
        draw_frame = ctk.CTkFrame(self.sidebar, fg_color="#2A2D34", corner_radius=8)
        draw_frame.pack(padx=15, pady=(0, 10), fill="x")

        ctk.CTkLabel(draw_frame, text="[ 그리기 설정 ]",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#AAAAAA").grid(row=0, column=0, columnspan=2, pady=(10, 5))

        # Z 높이
        ctk.CTkLabel(draw_frame, text="Z 높이(m):", font=ctk.CTkFont(size=12)).grid(
            row=1, column=0, padx=(10, 5), pady=3, sticky="w")
        self.z_entry = ctk.CTkEntry(draw_frame, width=100, font=ctk.CTkFont(size=12))
        self.z_entry.insert(0, str(DRAW_POS[2]))
        self.z_entry.grid(row=1, column=1, padx=(0, 10), pady=3)

        # 펜 업 높이
        ctk.CTkLabel(draw_frame, text="펜 업(m):", font=ctk.CTkFont(size=12)).grid(
            row=2, column=0, padx=(10, 5), pady=3, sticky="w")
        self.pen_up_entry = ctk.CTkEntry(draw_frame, width=100, font=ctk.CTkFont(size=12))
        self.pen_up_entry.insert(0, "0.03")
        self.pen_up_entry.grid(row=2, column=1, padx=(0, 10), pady=3)

        # Blend Radius
        ctk.CTkLabel(draw_frame, text="Blend R(m):", font=ctk.CTkFont(size=12)).grid(
            row=3, column=0, padx=(10, 5), pady=3, sticky="w")
        self.blend_label = ctk.CTkLabel(draw_frame, text="0.05", font=ctk.CTkFont(size=12),
                                        text_color="#4FC3F7")
        self.blend_label.grid(row=3, column=1, padx=(0, 10), pady=3)
        self.blend_slider = ctk.CTkSlider(draw_frame, from_=0.02, to=0.2,
                                          number_of_steps=18,
                                          command=self.on_blend_changed)
        self.blend_slider.set(0.05)
        self.blend_slider.grid(row=4, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="ew")

        # 단순화 강도 (epsilon)
        ctk.CTkLabel(draw_frame, text="단순화(px):", font=ctk.CTkFont(size=12)).grid(
            row=5, column=0, padx=(10, 5), pady=3, sticky="w")
        self.epsilon_entry = ctk.CTkEntry(draw_frame, width=100, font=ctk.CTkFont(size=12))
        self.epsilon_entry.insert(0, "8")
        self.epsilon_entry.grid(row=5, column=1, padx=(0, 10), pady=3)

        # 획당 최대 경유점 수
        ctk.CTkLabel(draw_frame, text="최대 점 수:", font=ctk.CTkFont(size=12)).grid(
            row=6, column=0, padx=(10, 5), pady=3, sticky="w")
        self.max_pts_entry = ctk.CTkEntry(draw_frame, width=100, font=ctk.CTkFont(size=12))
        self.max_pts_entry.insert(0, "20")
        self.max_pts_entry.grid(row=6, column=1, padx=(0, 10), pady=(3, 10))

        # --- 동작 버튼 ---
        self.btn_draw = ctk.CTkButton(
            self.sidebar, text="🖊️ 그리기 실행", height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2E7D32", hover_color="#1B5E20",
            command=self.run_drawing)
        self.btn_draw.pack(padx=15, pady=5, fill="x")

        self.btn_preview = ctk.CTkButton(
            self.sidebar, text="👁️ 경유점 미리보기", height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#6A1B9A", hover_color="#4A148C",
            command=self.preview_waypoints)
        self.btn_preview.pack(padx=15, pady=5, fill="x")

        self.btn_clear = ctk.CTkButton(
            self.sidebar, text="🗑️ 캔버스 초기화", height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#455A64", hover_color="#37474F",
            command=self.clear_canvas)
        self.btn_clear.pack(padx=15, pady=5, fill="x")

        self.btn_home = ctk.CTkButton(
            self.sidebar, text="🏠 홈 이동", height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#F57C00", hover_color="#E65100",
            command=self.run_home)
        self.btn_home.pack(padx=15, pady=5, fill="x")

        self.btn_stop = ctk.CTkButton(
            self.sidebar, text="🛑 긴급 중지 (Q)", height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#D32F2F", hover_color="#B71C1C",
            command=self.run_stop)
        self.btn_stop.pack(padx=15, pady=5, fill="x")

        # 상태 표시
        self.status_label = ctk.CTkLabel(
            self.sidebar, text="⚪ 미연결",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#F44336")
        self.status_label.pack(padx=15, pady=(10, 5))

        # 경유점 수 표시
        self.wp_count_label = ctk.CTkLabel(
            self.sidebar, text="경유점: 0개",
            font=ctk.CTkFont(size=12), text_color="#AAAAAA")
        self.wp_count_label.pack(padx=15, pady=(0, 10))

        # --- 로그 창 (스크롤 밖 고정) ---
        self.log_frame = ctk.CTkFrame(self.sidebar_outer, fg_color="#121212")
        self.log_frame.grid(row=2, column=0, padx=10, pady=(5, 15), sticky="nsew")
        self.log_text = ctk.CTkTextbox(
            self.log_frame, font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="transparent", text_color="#00FF41", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        sys.stdout = PrintLogger(self.log_text)

        # ---- 메인 영역: 캔버스 ----
        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.main_area.grid_rowconfigure(1, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.main_area,
                     text="🖌️ 마우스로 그림을 그리세요 (드래그)",
                     font=ctk.CTkFont(size=15, weight="bold")
                     ).grid(row=0, column=0, pady=(0, 10))

        # Tkinter 캔버스 (customtkinter 위에 일반 tkinter Canvas 사용)
        canvas_frame = ctk.CTkFrame(self.main_area, fg_color="#1E1E1E", corner_radius=10)
        canvas_frame.grid(row=1, column=0, sticky="nsew")

        import tkinter as tk
        self.canvas = tk.Canvas(canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="#1E1E1E", highlightthickness=1,
                                highlightbackground="#444444", cursor="crosshair")
        self.canvas.pack(padx=15, pady=15)

        # 그리드 라인 (보조선)
        self._draw_grid()

        # 캔버스 이벤트 바인딩
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # 키보드 바인딩
        self.bind("<q>", lambda e: self.run_stop())
        self.bind("<Q>", lambda e: self.run_stop())
        self.bind("<Delete>", lambda e: self.clear_canvas())

        print("=== Indy7 Drawing GUI 시작 ===")
        print(f"초기 위치: X={DRAW_POS[0]}, Y={DRAW_POS[1]}, Z={DRAW_POS[2]}")
        print(f"자세: Rx={DRAW_ROT[0]}°, Ry={DRAW_ROT[1]}°, Rz={DRAW_ROT[2]}°")
        print(f"작업 영역: {WORKSPACE_SIZE*1000:.0f}mm x {WORKSPACE_SIZE*1000:.0f}mm")
        print("캔버스에 마우스로 그림을 그린 뒤 '그리기 실행' 버튼을 누르세요.\n")

    # ============================================================
    # 캔버스 그리드 보조선
    # ============================================================
    def _draw_grid(self):
        step = CANVAS_SIZE // 10
        for i in range(0, CANVAS_SIZE + 1, step):
            self.canvas.create_line(i, 0, i, CANVAS_SIZE, fill="#333333", dash=(2, 4))
            self.canvas.create_line(0, i, CANVAS_SIZE, i, fill="#333333", dash=(2, 4))
        # 중심선
        mid = CANVAS_SIZE // 2
        self.canvas.create_line(mid, 0, mid, CANVAS_SIZE, fill="#555555", width=1)
        self.canvas.create_line(0, mid, CANVAS_SIZE, mid, fill="#555555", width=1)
        # 중심점 표시
        self.canvas.create_oval(mid - 4, mid - 4, mid + 4, mid + 4,
                                fill="#4FC3F7", outline="#4FC3F7")

    # ============================================================
    # 마우스 이벤트: 그리기
    # ============================================================
    def on_mouse_down(self, event):
        self.current_stroke = [(event.x, event.y)]

    def on_mouse_drag(self, event):
        if not self.current_stroke:
            return
        x, y = event.x, event.y
        # 캔버스 범위 클리핑
        x = max(0, min(CANVAS_SIZE, x))
        y = max(0, min(CANVAS_SIZE, y))

        last_x, last_y = self.current_stroke[-1]
        self.canvas.create_line(last_x, last_y, x, y,
                                fill="#00FF41", width=2, smooth=True,
                                tags="drawing")
        self.current_stroke.append((x, y))

    def on_mouse_up(self, event):
        if self.current_stroke and len(self.current_stroke) >= 2:
            self.strokes.append(self.current_stroke)
            print(f"획 추가 ({len(self.current_stroke)}개 포인트)")
        self.current_stroke = []
        self._update_wp_count()

    # ============================================================
    # Blend Radius 슬라이더 변경
    # ============================================================
    def on_blend_changed(self, value):
        self.blend_label.configure(text=f"{value:.3f}")

    # ============================================================
    # 경유점 추출
    # ============================================================
    def _extract_waypoints(self):
        """모든 획에서 경유점 추출 (Douglas-Peucker 단순화 + 최대 개수 제한)"""
        epsilon = float(self.epsilon_entry.get())
        max_pts = int(self.max_pts_entry.get())
        all_waypoints = []

        for stroke in self.strokes:
            if len(stroke) < 2:
                continue
            simplified = douglas_peucker(stroke, epsilon)

            # 최대 개수 초과 시 균등 리샘플링
            if max_pts > 0 and len(simplified) > max_pts:
                simplified = self._resample_points(simplified, max_pts)

            all_waypoints.append(simplified)

        return all_waypoints

    def _resample_points(self, points, num):
        """경로를 num개 점으로 균등 리샘플링 (시작/끝 포함)"""
        if num < 2 or len(points) < 2:
            return points

        # 누적 거리 계산
        dists = [0.0]
        for i in range(1, len(points)):
            d = math.sqrt((points[i][0] - points[i-1][0])**2 +
                          (points[i][1] - points[i-1][1])**2)
            dists.append(dists[-1] + d)
        total = dists[-1]
        if total == 0:
            return [points[0], points[-1]]

        # 균등 간격으로 보간
        result = [points[0]]
        for k in range(1, num - 1):
            target = total * k / (num - 1)
            # target이 위치하는 구간 찾기
            for j in range(1, len(dists)):
                if dists[j] >= target:
                    ratio = (target - dists[j-1]) / (dists[j] - dists[j-1])
                    x = points[j-1][0] + ratio * (points[j][0] - points[j-1][0])
                    y = points[j-1][1] + ratio * (points[j][1] - points[j-1][1])
                    result.append((x, y))
                    break
        result.append(points[-1])
        return result

    def _update_wp_count(self):
        wp_strokes = self._extract_waypoints()
        total = sum(len(s) for s in wp_strokes)
        self.wp_count_label.configure(text=f"경유점: {total}개 ({len(self.strokes)}획)")

    # ============================================================
    # 경유점 미리보기
    # ============================================================
    def preview_waypoints(self):
        self.canvas.delete("preview")
        wp_strokes = self._extract_waypoints()

        total = 0
        for stroke_wps in wp_strokes:
            for i, (cx, cy) in enumerate(stroke_wps):
                r = 4
                color = "#FF5722" if (i == 0 or i == len(stroke_wps) - 1) else "#FFEB3B"
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                        fill=color, outline="white",
                                        tags="preview")
                total += 1

            # 경유점 사이를 선으로 연결
            if len(stroke_wps) >= 2:
                coords = []
                for pt in stroke_wps:
                    coords.extend(pt)
                self.canvas.create_line(*coords, fill="#FF9800", width=1,
                                       dash=(4, 4), tags="preview")

        self._update_wp_count()
        print(f"경유점 미리보기: 총 {total}개 포인트 ({len(wp_strokes)}획)")

    # ============================================================
    # 캔버스 초기화
    # ============================================================
    def clear_canvas(self):
        self.canvas.delete("drawing")
        self.canvas.delete("preview")
        self.strokes.clear()
        self.current_stroke.clear()
        self.wp_count_label.configure(text="경유점: 0개")
        print("캔버스 초기화 완료")

    # ============================================================
    # 로봇 연결 토글
    # ============================================================
    def toggle_connection(self):
        if self.is_connected:
            self._disconnect_robot()
        else:
            self._connect_robot()

    def _connect_robot(self):
        ip = self.ip_entry.get().strip()
        if not ip:
            print("❌ IP 주소를 입력하세요")
            return

        def _connect_logic():
            try:
                from indy_utils import indydcp_client as client
                print(f">> {ip} 연결 시도 중...")
                self.indy = client.IndyDCPClient(ip, ROBOT_NAME)
                self.indy.connect()
                self.is_connected = True
                self.after(0, lambda: self.status_label.configure(
                    text="🟢 연결됨", text_color="#4CAF50"))
                self.after(0, lambda: self.btn_connect.configure(
                    text="🔌 연결 해제", fg_color="#D32F2F"))
                print(f">> 로봇 연결 성공 ({ip})")
            except Exception as e:
                print(f"❌ 연결 실패: {e}")
                self.indy = None
                self.is_connected = False

        threading.Thread(target=_connect_logic, daemon=True).start()

    def _disconnect_robot(self):
        try:
            if self.indy:
                self.indy.disconnect()
            print(">> 로봇 연결 해제")
        except Exception as e:
            print(f"연결 해제 오류: {e}")
        finally:
            self.indy = None
            self.is_connected = False
            self.status_label.configure(text="⚪ 미연결", text_color="#F44336")
            self.btn_connect.configure(text="🔌 연결", fg_color="#1976D2")

    # ============================================================
    # 홈 이동
    # ============================================================
    def run_home(self):
        if not self.is_connected or not self.indy:
            print("❌ 로봇이 연결되지 않았습니다")
            return

        def _home_logic():
            try:
                # 진행 중인 작업이 있으면 먼저 정지
                if self.current_task_thread and self.current_task_thread.is_alive():
                    stop_event.set()
                    self.current_task_thread.join(timeout=2.0)

                stop_event.clear()
                self.indy.reset_robot()
                time.sleep(1.0)
                print(">> 홈 이동 시작...")
                self.indy.go_home()
                move_done_check(self.indy)
                print(">> 홈 위치 도착 완료")
            except InterruptedError as e:
                print(f"[알림] {e}")
            except Exception as e:
                print(f"❌ 홈 이동 오류: {e}")

        # 홈 스레드는 current_task_thread에 할당하지 않음
        # (자기 자신을 join 시도하는 버그 방지 - main_gui.py 패턴 따름)
        threading.Thread(target=_home_logic, daemon=True).start()

    # ============================================================
    # 긴급 정지
    # ============================================================
    def run_stop(self):
        print("🛑 긴급 정지!")
        stop_event.set()
        if self.indy and self.is_connected:
            try:
                self.indy.stop_motion()
            except Exception:
                pass

    # ============================================================
    # 그리기 실행 (핵심 로직)
    # ============================================================
    def run_drawing(self):
        if not self.is_connected or not self.indy:
            print("❌ 로봇이 연결되지 않았습니다")
            return
        if self.current_task_thread and self.current_task_thread.is_alive():
            print("⚠️ 다른 작업이 실행 중입니다")
            return
        if not self.strokes:
            print("❌ 먼저 캔버스에 그림을 그려주세요")
            return

        stop_event.clear()
        self.current_task_thread = threading.Thread(
            target=self._drawing_task, daemon=True)
        self.current_task_thread.start()

    def _drawing_task(self):
        """
        그리기 실행 로직:
        1. 각 획(stroke)에서 경유점 추출
        2. 획 시작 시 펜 다운 (Z 하강), 획 끝나면 펜 업 (Z 상승)
        3. 각 획 내의 경유점은 blend_radius를 사용하여 부드럽게 연결
        """
        try:
            draw_z = float(self.z_entry.get())
            pen_up_offset = float(self.pen_up_entry.get())
            blend_r = self.blend_slider.get()
            pen_up_z = draw_z + pen_up_offset

            wp_strokes = self._extract_waypoints()
            total_strokes = len(wp_strokes)
            total_wps = sum(len(s) for s in wp_strokes)

            print(f"\n🖊️ 그리기 시작! ({total_strokes}획, {total_wps}개 경유점)")
            print(f"   Z={draw_z}m, 펜업={pen_up_z}m, BlendR={blend_r:.3f}m")
            print("=" * 50)

            for stroke_idx, stroke_wps in enumerate(wp_strokes):
                if stop_event.is_set():
                    raise InterruptedError("강제 중단")

                print(f"\n--- 획 {stroke_idx + 1}/{total_strokes} ({len(stroke_wps)}개 포인트) ---")

                if len(stroke_wps) < 2:
                    print("  (포인트 부족, 건너뜀)")
                    continue

                # (1) 획 시작점 위로 이동 (펜 업 상태)
                first_pt = stroke_wps[0]
                start_pose = canvas_to_robot(first_pt[0], first_pt[1], pen_up_z, DRAW_ROT)
                print(f"  >> 시작점 위로 이동: X={start_pose[0]:.4f}, Y={start_pose[1]:.4f}")
                self.indy.task_move_to(start_pose)
                move_done_check(self.indy)

                # (2) 펜 다운 (Z 하강)
                down_pose = canvas_to_robot(first_pt[0], first_pt[1], draw_z, DRAW_ROT)
                print(f"  >> 펜 다운 (Z={draw_z}m)")
                self.indy.task_move_to(down_pose)
                move_done_check(self.indy)

                # (3) Waypoint Set으로 경유점 등록 + 블렌딩 실행
                if len(stroke_wps) >= 2:
                    self.indy.task_waypoint_clean()
                    time.sleep(0.1)

                    for i, (cx, cy) in enumerate(stroke_wps):
                        if stop_event.is_set():
                            raise InterruptedError("강제 중단")

                        pose = canvas_to_robot(cx, cy, draw_z, DRAW_ROT)

                        # 첫 번째와 마지막 포인트는 blend_radius=0 (정확히 도달)
                        # 중간 경유점은 blend_radius 적용 (부드럽게 통과)
                        if i == 0 or i == len(stroke_wps) - 1:
                            br = 0
                        else:
                            br = blend_r

                        self.indy.task_waypoint_append(pose, 0, br)  # (p, wp_type=0, blend_radius)

                    print(f"  >> {len(stroke_wps)}개 경유점 등록 완료, 실행 중...")
                    self.indy.task_waypoint_execute(0)  # policy=0 (stop on collision)
                    move_done_check(self.indy)
                    print(f"  >> 획 {stroke_idx + 1} 완료!")

                # (4) 펜 업 (Z 상승)
                last_pt = stroke_wps[-1]
                up_pose = canvas_to_robot(last_pt[0], last_pt[1], pen_up_z, DRAW_ROT)
                print(f"  >> 펜 업 (Z={pen_up_z}m)")
                self.indy.task_move_to(up_pose)
                move_done_check(self.indy)

            print("\n" + "=" * 50)
            print("✅ 그리기 완료!")

        except InterruptedError as e:
            print(f"\n🛑 [중단] {e}")
        except Exception as e:
            print(f"\n❌ [오류] {e}")

    # ============================================================
    # 종료
    # ============================================================
    def on_closing(self):
        if messagebox.askokcancel("종료", "프로그램을 종료하시겠습니까?"):
            if self.current_task_thread and self.current_task_thread.is_alive():
                stop_event.set()
                self.current_task_thread.join(timeout=1.0)
            if self.indy:
                try:
                    self.indy.disconnect()
                except:
                    pass
            self.quit()
            self.destroy()


# ============================================================
# 메인 실행
# ============================================================
def main():
    app = DrawingApp()
    try:
        app.mainloop()
    except Exception as e:
        print(f"\n>> 시스템 오류: {e}")
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
