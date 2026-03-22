from indy_utils import indydcp_client as client
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading

# ---- 로봇 연결 ----
robot_ip = "192.168.3.7"
robot_name = "NRMK-Indy7"
indy = client.IndyDCPClient(robot_ip, robot_name)
indy.connect()

print(">> 로봇 연결 성공")

# ---- palletizing 속성 ----
LAYER_HEIGHT = 0.03
OFFSET_X = 0.04
OFFSET_Y = 0.04
RETRACT_Z = 0.08

# ---- 중단 신호를 관리할 이벤트 객체 생성 ----
stop_event = threading.Event()

# ---- 동작 확인 함수 ----
def move_done_check():
    while True:
        # 1. 만약 중지 요청이 들어올 경우
        if stop_event.is_set():
            indy.stop_motion() 
            time.sleep(0.5)    
            raise InterruptedError("강제 중단되었습니다.")
        
        # 2. 로봇 상태 확인
        status = indy.get_robot_status()
        
        # [핵심 추가] 로봇이 에러나 충돌 상태인지 확인하고 스스로 복구하기
        if status.get('error') == 1 or status.get('collision') == 1:
            print("\n🚨 [경고] 로봇 에러(또는 충돌)가 감지되었습니다! 에러를 초기화합니다.")
            indy.reset_robot() # 로봇 에러 초기화 (Reset)
            time.sleep(0.5)
            raise InterruptedError("로봇 에러 발생으로 인해 동작이 취소되었습니다.")

        # 3. 정상적으로 이동을 마친 경우 탈출
        if status['movedone'] == 1:
            break

# ---- sleep 중에 중단 수행 함수 ----
def stoppable_sleep(duration):
    start = time.time()
    while time.time() - start < duration:
        if stop_event.is_set():
            indy.stop_motion()
            time.sleep(0.5)  
            raise InterruptedError("강제 중단되었습니다.")
        time.sleep(0.1)

# ---- palletizing 공간 생성 함수 ----
"""
LAYER_HEIGHT = 0.03
OFFSET_X = 0.04
OFFSET_Y = 0.04
RETRACT_Z = 0.08
"""
def generate_grid(base, grid_x, grid_y, offset_x, offset_y, num_layers = 1, layer_height = 0):
    coords = []
    for layer in range(num_layers):
        z = base[2] + layer * layer_height
        for i in range(grid_y):
            for j in range(grid_x):
                x = base[0] + j * offset_x
                y = base[1] + i * offset_y
                coords.append([x,y,z])
                print(f"생성된 좌표 (층 {layer+1}): {coords[-1]}")
    return coords

# ---- 기존의 retraction 모션을 구현하는 함수 ----
def retractionPick(pick_pos, pick_rot, dz_offset, seq='xyz'):
    # ---- 1. pick 좌표에 대해서 행렬 변환 ----
    pos = np.array(pick_pos)
    rotation = R.from_euler(seq, pick_rot, degrees=True)

    # ---- 2. pick rotation으로 행렬 변환 ----
    local_move_vector = np.array([0, 0, -dz_offset])
    global_move_vector = rotation.apply(local_move_vector)
    
    # ---- 3. 기존 좌표에 행렬 변환 된 offset 값을 더 해주어서 retration 좌표를 생성 ----
    retract_pos = pos + global_move_vector
    
    return retract_pos.tolist()

# ---- task 1번 ----
def task_pal_1by2_2layer():
    try:
        print(">> 작업1 시작")
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
            print(f"==== Step {i+1}====")
            # ----------PICK--------------
            indy.task_move_to([new_pos[0], new_pos[1], new_pos[2], *PICK_ROT]); move_done_check()
            indy.task_move_to([PICK_POS[0], PICK_POS[1], PICK_POS[2], *PICK_ROT]); move_done_check()
            print("진공 ON (흡착)")
            indy.set_do(2, True); stoppable_sleep(2)
            indy.task_move_to([new_pos[0], new_pos[1], new_pos[2], *PICK_ROT]); move_done_check()
            
            # ----------PLACE--------------
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
        print(">> 작업2 시작")
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

            print(f"==== Step {i+1}====")
            # ----------PICK--------------
            indy.task_move_to([pick[0], pick[1], pick[2] + RETRACT_Z, *PICK_ROT]); move_done_check()
            indy.task_move_to([pick[0], pick[1], pick[2], *PICK_ROT]); move_done_check()
            indy.set_do(2, True); stoppable_sleep(0.5)
            indy.task_move_to([pick[0], pick[1], pick[2] + RETRACT_Z, *PICK_ROT]); move_done_check()

            # ----------PLACE--------------
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

def main():
    current_task_thread = None

    try:
        print("\n=== 로봇 제어 대기중 ===")
        while True:
            # ---- 1. 사용자로부터 입력 신호를 get ----
            cmd = input(">> 동작 수행 입력(1: 작업1, 2: 작업2, h: 홈, q: 작업 취소/종료): ").strip()

            # ---- 2. 입력 신호에 해당하는 동작 수행 ----
            # 작업 1 수행
            if cmd == "1":
                if current_task_thread and current_task_thread.is_alive():
                    print(">> 이미 다른 작업이 진행 중입니다. (취소하려면 q 입력 후 Enter)")
                    continue
                stop_event.clear()
                current_task_thread = threading.Thread(target=task_pal_1by2_2layer)
                current_task_thread.start()

            # 작업 2 실행
            elif cmd == "2":
                if current_task_thread and current_task_thread.is_alive():
                    print(">> 이미 다른 작업이 진행 중입니다. (취소하려면 q 입력 후 Enter)")
                    continue
                stop_event.clear()
                current_task_thread = threading.Thread(target=task_pick2by2_place2by2)
                current_task_thread.start()
            
            # h 입력 시: 작업 도중, q 중지 중, 그리고 다른 명령 없을 때 home 모션 수행
            elif cmd == "h":
                if current_task_thread and current_task_thread.is_alive():
                    print(">> 🛑 진행 중인 동작을 중단하고 홈으로 복귀합니다...")
                    stop_event.set()           
                    current_task_thread.join(timeout=2.0) 
                    stop_event.clear()         
                else:
                    print(">> 홈으로 이동합니다...")

                stop_event.clear()
                indy.set_do(2, False)
                indy.reset_robot() 
                indy.go_home()
                move_done_check()
                print(">> 홈 이동 완료")

            # q 입력 시: 작업 중이면 중단, 대기 중이면 프로그램 종료
            elif cmd == "q":
                if current_task_thread and current_task_thread.is_alive():
                    print(">> 진행 중인 동작을 즉시 중단합니다...")
                    stop_event.set() 
                    current_task_thread.join(timeout=2.0) 
                    stop_event.clear() 
                    print(">> 동작이 취소되었습니다. 새로운 명령을 입력하세요.")
                else:
                    print(">> 프로그램을 종료합니다.")
                    break
            
            time.sleep(0.05)

    # ---- 3. 키보드 ctrl + c 중지 요청 ----
    except KeyboardInterrupt:
        print("\n>> 강제 종료 요청받음")
        if current_task_thread and current_task_thread.is_alive():
            stop_event.set()
            current_task_thread.join()
        indy.go_home()
        indy.set_do(2, False)

    # ---- 4. 확실한 연결을 끊기 위한 방지 ----
    finally:
        indy.disconnect()
        print(">> 로봇 연결 종료")

if __name__ =="__main__":
    main()