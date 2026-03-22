from indy_utils import indydcp_client as client
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

robot_ip = "192.168.3.7"
robot_name = "NRMK-Indy7"

indy = client.IndyDCPClient(robot_ip, robot_name)
indy.connect()

# pick and place POSE
PICK_POS = [0.201, 0.521, 0.15]
PICK_ROT = [-25, -180, 90]
PLACE_POS = [0.109, 0.260, 0.198]
PLACE_ROT = [0, 180, 90]

# pick and place palletizing configuration
GRID_X = 1
GRID_Y = 2
NUM_LAYERS = 2
LAYER_HEIGHT = 0.03
OFFSET_X = 0.04
OFFSET_Y = 0.04
RETRACT_Z = 0.08

def move_done_check():
    while True:
        status = indy.get_robot_status()
        if status['movedone'] == 1:
            break
        time.sleep(0.2)

def generate_grid(base, grid_x, grid_y, offset_x, offset_y, num_layers, layer_height):
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

def excute_pick_and_place():
    place_position = generate_grid(PLACE_POS, GRID_X, GRID_Y, OFFSET_X, OFFSET_Y, NUM_LAYERS, LAYER_HEIGHT)
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
        indy.task_move_to([new_pos[0], new_pos[1], new_pos[2], *PICK_ROT])
        move_done_check()
        indy.task_move_to([PICK_POS[0], PICK_POS[1], PICK_POS[2], *PICK_ROT])
        move_done_check()

        indy.set_do(2, True)
        print("진공 ON (흡착)")
        time.sleep(2)

        indy.task_move_to([new_pos[0], new_pos[1], new_pos[2], *PICK_ROT])
        move_done_check()

        # ----------PLACE--------------
        indy.task_move_to([place[0], place[1], place[2] + RETRACT_Z, *PLACE_ROT])
        move_done_check()
        indy.task_move_to([place[0], place[1], place[2], *PLACE_ROT])
        move_done_check()

        indy.set_do(2, False)
        print("진공 OFF (해제 중)")
        time.sleep(2)

        indy.task_move_to([place[0], place[1], place[2] + RETRACT_Z, *PLACE_ROT])
        move_done_check()
    indy.go_home()
    move_done_check()
    print("모든 Pick & Place 작업 완료!")

if __name__ =="__main__":
    try:
        excute_pick_and_place()

    except:
        KeyboardInterrupt
        print("종료")
    finally:
        indy.disconnect()
        print(">>로봇 연결 종료")