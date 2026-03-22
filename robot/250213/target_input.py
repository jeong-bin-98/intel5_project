"""
타겟 좌표 입력 클라이언트
터미널에서 실행하여 좌표를 입력하면 analyticSolution.py로 전송
"""
import socket
import struct

# 소켓 설정
HOST = "127.0.0.1"
PORT =  5555

import math

def main():
    print("=" * 60)
    print("원하는 위치(m), 회전(degree) 값을 입력하세요")
    print("=" * 60)
    print("[사용법] x y z roll pitch yaw")
    print("(예: -0.2 -0.2 0.4 0 -90 0)")
    print("[종료] Ctrl+C")
    print()

    while True:
        try:
            user_input = input(">> 좌표(m)/회전(°) 입력: ").strip()
            
            # 입력값 검증
            coords = user_input.split()
            if len(coords) != 6:
                print("[오류] 형식: x y z roll pitch yaw (6개 값 필요)")
                continue
            
            try:
                # 입력받은 값 파싱
                raw_data = [float(val) for val in coords]
                
                # 위치(x,y,z)는 그대로, 회전은 Degree -> Radian 변환
                pos_data = raw_data[:3]
                rot_deg = raw_data[3:]
                # rot_rad = [math.radians(deg) for deg in rot_deg]
                
                # 전송할 데이터 (x, y, z, r_rad, p_rad, y_rad)
                send_data = pos_data + rot_deg
                
            except ValueError:
                print("[오류] 숫자를 입력해주세요.")
                continue
            
            # 서버로 전송
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2)
                    s.connect((HOST, PORT))
                    # L-V(Length-Value) 프로토콜로 전송
                    # Value: 좌표+회전 데이터 (6 floats)
                    value_data = struct.pack('ffffff', *send_data)   
                    length_data = struct.pack('I', len(value_data))  # Length: 데이터 길이
                    s.sendall(length_data + value_data)

                    print(f"[전송 완료] Pos({pos_data[0]}, {pos_data[1]}, {pos_data[2]}) / Rot(deg)({rot_deg[0]}, {rot_deg[1]}, {rot_deg[2]})")
            except ConnectionRefusedError:
                print("[오류] 시뮬레이션 서버가 실행 중이 아닙니다.")
            except socket.timeout:
                print("[오류] 연결 시간 초과")
                
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break

if __name__ == "__main__":
    main()
