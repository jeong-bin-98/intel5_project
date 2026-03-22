from indy_utils import indydcp_client as client
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import serial

# ---- 아두이노 연결 ----
arduino = serial.Serial('COM10', 9600, timeout=1)
time.sleep(2)

# ---- 로봇 연결 ----
robot_ip = "192.168.3.7"
robot_name = "NRMK-Indy7"
indy = client.IndyDCPClient(robot_ip, robot_name)
indy.connect()

print(">> 로봇 연결 성공")

# pick and place palletizing configuration
LAYER_HEIGHT = 0.03
OFFSET_X = 0.04
OFFSET_Y = 0.04
RETRACT_Z = 0.08

def lcd_send(msg):
    arduino.write((msg + "\n").encode())

def move_done_check():
    while True:
        status = indy.get_robot_status()
        if status['movedone'] == 1:
            break
        time.sleep(0.2)

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

def retractionPick(pick_pos, pick_rot, dz_offset, seq='xyz'):
    pos = np.array(pick_pos)
    rotation = R.from_euler(seq, pick_rot, degrees=True)

    local_move_vector = np.array([0, 0, -dz_offset])
    global_move_vector = rotation.apply(local_move_vector)
    
    retract_pos = pos + global_move_vector
    
    return retract_pos.tolist()

def task_pick2by2_place2by2():
    """
    작업2 (2X2 Pick & Place)
    """
    lcd_send("OP2_START")
    print(">> 작업2 시작")

    indy.set_joint_vel_level(3)

    # pick and place POSE
    PICK_POS = [0.109, 0.260, 0.190]
    PICK_ROT = [0, 180, 90]
    PLACE_POS = [0.109, 0.340, 0.190]
    PLACE_ROT = [0, 180, 90]

    # Grid configuration
    PICK_GRID_X = 2
    PICK_GRID_Y = 2
    PLACE_GRID_X = 2
    PLACE_GRID_Y = 2

    pick_positions = generate_grid(PICK_POS, PICK_GRID_X, PICK_GRID_Y, OFFSET_X, OFFSET_Y)
    place_positions = generate_grid(PLACE_POS, PLACE_GRID_X, PLACE_GRID_Y, OFFSET_X, OFFSET_Y)
    print(f"픽 고정 위치: {PICK_POS}")
    print(f"팔레타이징 위치 수: {len(pick_positions)}")

    indy.go_home()
    move_done_check()
    print("홈 위치 도착 완료.")
    time.sleep(1)

    for i in range(len(pick_positions)):
        pick = pick_positions[i]
        place = place_positions[i]

        print(f"==== Step {i+1}====")
        print(f"Pick: {PICK_POS}")
        print(f"Place: {place}")
        # ----------PICK--------------
        indy.task_move_to([pick[0], pick[1], pick[2] + RETRACT_Z, *PICK_ROT]); move_done_check()
        indy.task_move_to([pick[0], pick[1], pick[2], *PICK_ROT]); move_done_check()
        indy.set_do(2, True); time.sleep(0.5)
        indy.task_move_to([pick[0], pick[1], pick[2] + RETRACT_Z, *PICK_ROT]); move_done_check()

        # ----------PLACE--------------
        indy.task_move_to([place[0], place[1], place[2] + RETRACT_Z, *PLACE_ROT]); move_done_check()
        indy.task_move_to([place[0], place[1], place[2], *PLACE_ROT]); move_done_check()
        indy.set_do(2, False); time.sleep(0.5)
        indy.task_move_to([place[0], place[1], place[2] + RETRACT_Z, *PLACE_ROT]); move_done_check()
    
    indy.go_home()
    move_done_check()

    lcd_send("OP2_FINISH")
    print("작업2 완료")

def task_pal_1by2_2layer():
    """
    작업1 (Pick & Place 1X2 2단)
    """
    lcd_send("OP1_START")
    print(">> 작업1 시작")

    # pick and place POSE
    PICK_POS = [0.201, 0.521, 0.15]
    PICK_ROT = [-25, -180, 90]
    PLACE_POS = [0.109, 0.260, 0.198]
    PLACE_ROT = [0, 180, 90]

    # Grid configuration
    GRID_X = 2
    GRID_Y = 2

    place_position = generate_grid(PLACE_POS, GRID_X, GRID_Y, OFFSET_X, OFFSET_Y)
    print(f"픽 고정 위치: {PICK_POS}")
    print(f"팔레타이징 위치 수: {len(place_position)}")

    indy.go_home()
    move_done_check()
    print("홈 위치 도착 완료.")
    time.sleep(1)

    new_pos = retractionPick(PICK_POS, PICK_ROT, RETRACT_Z)
    
    for i, place in enumerate(place_position):
        print(f"==== Step {i+1}====")
        print(f"Pick: {PICK_POS}")
        print(f"Place: {place}")
        print(f"새로운 절대 위치 (Retract): {np.round(new_pos, 4).tolist()}")
        # ----------PICK--------------
        indy.task_move_to([new_pos[0], new_pos[1], new_pos[2], *PICK_ROT]); move_done_check()
        indy.task_move_to([PICK_POS[0], PICK_POS[1], PICK_POS[2], *PICK_ROT]); move_done_check()
        print("진공 ON (흡착)")
        indy.set_do(2, True); time.sleep(2)
        indy.task_move_to([new_pos[0], new_pos[1], new_pos[2], *PICK_ROT]); move_done_check()

        
        # ----------PLACE--------------
        indy.task_move_to([place[0], place[1], place[2] + RETRACT_Z, *PLACE_ROT]); move_done_check()
        indy.task_move_to([place[0], place[1], place[2], *PLACE_ROT]); move_done_check()
        print("진공 OFF (해제 중)")
        indy.set_do(2, False); time.sleep(2)
        indy.task_move_to([place[0], place[1], place[2] + RETRACT_Z, *PLACE_ROT]); move_done_check()

    indy.go_home(); move_done_check()
    
    lcd_send("OP1_FINISH")
    print("작업1 완료")

def main():
    try:
        print("\n=== 아두이노 버튼 대기중… ===")
        while True:
            if arduino.in_waiting:
                cmd = arduino.readline().decode().strip()
                if cmd == "OP2":
                    task_pick2by2_place2by2()
                elif cmd == "OP1":
                    task_pal_1by2_2layer()
            time.sleep(0.05)

    except KeyboardInterrupt:
        indy.set_do(2, False)
        print(">>종료 요청받음")

    finally:
        indy.disconnect()
        print(">>로봇 연결 종료")

if __name__ =="__main__":
    main()