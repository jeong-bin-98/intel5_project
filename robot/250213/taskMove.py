from indy_utils import indydcp_client as client
from socket_server import SocketServer

import json 
import threading
import numpy as np
from time import sleep

robot_ip = "192.168.3.7"  # Robot (Indy) IP
robot_name = "NRMK-Indy7"  # Robot name (Indy7)

def motion_done(indy):
    while True:
        status = indy.get_robot_status()
        
        if status['movedone'] == 1:
            break
        sleep(0.1)

def main():
    indy = client.IndyDCPClient(robot_ip, robot_name)
    
    HOST = "127.0.0.1"
    PORT = 5555

    target_pos = np.array([0.186, 0.35, 0.522])
    target_rot = np.array([0.0, 180.0, 90.0])

    print(">>접속 시도 중...")
    indy.connect()
    print(">>연결 성공")

    socket_server = SocketServer(HOST, PORT)
    socket_server.start(target_pos, target_rot)

    print(f"[안내] 소켓 서버 대기중 ({HOST}:{PORT})...")
    print("다른 터미널에서 'python target_input.py' 실행하여 좌표 입력\n")
    sleep(1)

    try:
        while True:
            final_task_pose = np.concatenate((target_pos, target_rot)).tolist()

            print(f">> 이동 시작: {final_task_pose}")
            
            indy.task_move_to(final_task_pose)
            
            motion_done(indy)
            print(f">> 목표 위치로 이동 완료")

            sleep(1)

    except Exception as e:
        print(f">>에러 발생: {e}")

    except KeyboardInterrupt:
        print("\n>> 종료 요청 받음.")

    finally:
        indy.disconnect()
        print(">>연결 해제 완료")

if __name__ == "__main__":
    main()