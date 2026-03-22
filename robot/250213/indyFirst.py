from indy_utils import indydcp_client as client

import json
from time import sleep  
import threading
import numpy as np

robot_ip = "192.168.3.7"  # Robot (Indy) IP
robot_name = "NRMK-Indy7"  # Robot name (Indy7)

def motion_done(indy):
    while True:
        status = indy.get_robot_status()
        if status['movedone'] == 1:
            print(status['movedone'])
            break
        sleep(0.1)

def main():
    indy = client.IndyDCPClient(robot_ip, robot_name)

    try:
        print(">>접속 시도 중...")
        indy.connect()
        print(">>연결 성공")

        while True:
            cmd = input("명령 입력 (숫자: 프로그램 번호, h: Home, z: Zero, q: 종료): ").strip().lower()

            if cmd == "h":
                indy.go_home()
                motion_done(indy)
                print(">> 홈으로 이동 완료")

            elif cmd == "z":
                indy.go_zero()
                motion_done(indy)
                print(">> 0으로 이동 완료")

            elif cmd == "q":
                indy.stop_current_program()
                print(">> 프로그램 종료")
            

            elif cmd.isdigit():
                program_num = int(cmd)
                indy.set_default_program(program_num)
                print(f">> {program_num}번 프로그램 설정 (index: {indy.get_default_program_idx()})")

                print(f">> {program_num}번 프로그램 실행 중...")
                indy.start_default_program()
                motion_done(indy)
                print(f">> {program_num}번 프로그램 실행 완료")
            
            else:
                print(">> 잘못된 입력입니다. 숫자, h, z, q 중 하나를 입력하세요.")

    except Exception as e:
        print(f">>에러 발생: {e}")

    except KeyboardInterrupt:
        print("\n>> 종료 요청 받음.")

    finally:
        indy.disconnect()
        print(">>연결 해제 완료")

if __name__ == "__main__":
    main()