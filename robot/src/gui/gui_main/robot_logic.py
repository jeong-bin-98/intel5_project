# robot_logic.py
from indy_utils import indydcp_client as client
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading

# ---- 로봇 연결 ----
robot_ip = "192.168.3.7"
robot_name = "NRMK-Indy7"

print(">> 로봇 연결을 시도합니다...")
try:
    indy = client.IndyDCPClient(robot_ip, robot_name)
    indy.connect()
    print(">> 로봇 연결 성공")
except Exception as e:
    print(f">> 로봇 연결 실패: {e}")

# ---- palletizing 속성 ----
LAYER_HEIGHT = 0.03
OFFSET_X = 0.04
OFFSET_Y = 0.04
RETRACT_Z = 0.08

# ---- 중단 신호를 관리할 이벤트 객체 생성 ----
stop_event = threading.Event()

# ---- 동작 확인 함수 ----
def move_done_check():
    time.sleep(0.2)
    
    while True:
        if stop_event.is_set():
            indy.stop_motion() 
            time.sleep(0.5)    
            raise InterruptedError("강제 중단되었습니다.")
        
        status = indy.get_robot_status()
        
        if status.get('error') == 1 or status.get('collision') == 1:
            print("\n🚨 [경고] 로봇 에러(또는 충돌)가 감지되었습니다! 에러를 초기화합니다.")
            indy.reset_robot() 
            time.sleep(2.0)
            raise InterruptedError("로봇 에러 발생으로 인해 동작이 취소되었습니다.")

        if status['movedone'] == 1:
            break
            
        time.sleep(0.1)

# ---- sleep 중에 중단 수행 함수 ----
def stoppable_sleep(duration):
    start = time.time()
    while time.time() - start < duration:
        if stop_event.is_set():
            indy.stop_motion()
            time.sleep(0.5)  
            raise InterruptedError("강제 중단되었습니다.")
        time.sleep(0.1)

def generate_grid(base, grid_x, grid_y, offset_x, offset_y, num_layers = 1, layer_height = 0):
    coords = []
    for layer in range(num_layers):
        z = base[2] + layer * layer_height
        for i in range(grid_y):
            for j in range(grid_x):
                x = base[0] + j * offset_x
                y = base[1] + i * offset_y
                coords.append([x,y,z])
    return coords

def retractionPick(pick_pos, pick_rot, dz_offset, seq='xyz'):
    pos = np.array(pick_pos)
    rotation = R.from_euler(seq, pick_rot, degrees=True)
    local_move_vector = np.array([0, 0, -dz_offset])
    global_move_vector = rotation.apply(local_move_vector)
    retract_pos = pos + global_move_vector
    return retract_pos.tolist()

# ---- task 1번 ----
def task_pal_1by2_2layer():
    try:
        print("\n>> [작업1] 1x2 2단 팔레타이징 시작")
        PICK_POS = [0.201, 0.521, 0.15]
        PICK_ROT = [-25, -180, 90]
        PLACE_POS = [0.109, 0.260, 0.198]
        PLACE_ROT = [0, 180, 90]
        GRID_X, GRID_Y = 2, 2

        place_position = generate_grid(PLACE_POS, GRID_X, GRID_Y, OFFSET_X, OFFSET_Y)
        
        indy.go_home()
        move_done_check()
        print("홈 위치 도착 완료.")
        stoppable_sleep(1)
        
        new_pos = retractionPick(PICK_POS, PICK_ROT, RETRACT_Z)
        
        for i, place in enumerate(place_position):
            print(f"==== Step {i+1} ====")
            indy.task_move_to([new_pos[0], new_pos[1], new_pos[2], *PICK_ROT]); move_done_check()
            indy.task_move_to([PICK_POS[0], PICK_POS[1], PICK_POS[2], *PICK_ROT]); move_done_check()
            print("진공 ON (흡착)")
            indy.set_do(2, True); stoppable_sleep(2)
            indy.task_move_to([new_pos[0], new_pos[1], new_pos[2], *PICK_ROT]); move_done_check()
            
            indy.task_move_to([place[0], place[1], place[2] + RETRACT_Z, *PLACE_ROT]); move_done_check()
            indy.task_move_to([place[0], place[1], place[2], *PLACE_ROT]); move_done_check()
            print("진공 OFF (해제 중)")
            indy.set_do(2, False); stoppable_sleep(2)
            indy.task_move_to([place[0], place[1], place[2] + RETRACT_Z, *PLACE_ROT]); move_done_check()

        indy.go_home(); move_done_check()
        print("작업1 완료")

    except InterruptedError as e:
        print(f"\n[알림] {e}")
        indy.set_do(2, False) 

# ---- task 2번 ----
def task_pick2by2_place2by2():
    try:
        print("\n>> [작업2] 2x2 Pick & Place 시작")
        indy.set_joint_vel_level(3)

        PICK_POS = [0.109, 0.260, 0.190]
        PICK_ROT = [0, 180, 90]
        PLACE_POS = [0.109, 0.340, 0.190]
        PLACE_ROT = [0, 180, 90]

        PICK_GRID_X, PICK_GRID_Y = 2, 2
        PLACE_GRID_X, PLACE_GRID_Y = 2, 2

        pick_positions = generate_grid(PICK_POS, PICK_GRID_X, PICK_GRID_Y, OFFSET_X, OFFSET_Y)
        place_positions = generate_grid(PLACE_POS, PLACE_GRID_X, PLACE_GRID_Y, OFFSET_X, OFFSET_Y)
        
        indy.go_home()
        move_done_check()
        print("홈 위치 도착 완료.")
        stoppable_sleep(1)

        for i in range(len(pick_positions)):
            pick = pick_positions[i]
            place = place_positions[i]

            print(f"==== Step {i+1} ====")
            indy.task_move_to([pick[0], pick[1], pick[2] + RETRACT_Z, *PICK_ROT]); move_done_check()
            indy.task_move_to([pick[0], pick[1], pick[2], *PICK_ROT]); move_done_check()
            indy.set_do(2, True); stoppable_sleep(0.5)
            indy.task_move_to([pick[0], pick[1], pick[2] + RETRACT_Z, *PICK_ROT]); move_done_check()

            indy.task_move_to([place[0], place[1], place[2] + RETRACT_Z, *PLACE_ROT]); move_done_check()
            indy.task_move_to([place[0], place[1], place[2], *PLACE_ROT]); move_done_check()
            indy.set_do(2, False); stoppable_sleep(0.5)
            indy.task_move_to([place[0], place[1], place[2] + RETRACT_Z, *PLACE_ROT]); move_done_check()
        
        indy.go_home()
        move_done_check()
        print("작업2 완료")

    except InterruptedError as e:
        print(f"\n[알림] {e}")
        indy.set_do(2, False)