"""
소켓 서버 모듈
L-V(Length-Value) 프로토콜로 클라이언트로부터 좌표를 수신하여 목표 타겟 위치를 업데이트
"""
import socket
import threading
import os
import struct

class SocketServer:
    """소켓 서버: 클라이언트로부터 좌표를 수신"""
    
    def __init__(self, host="127.0.0.1", port=5555):
        self.host = host
        self.port = port
        self.running = False
        self.server = None
        self.thread = None
        self._desired_pos = None  # numpy array (x, y, z)
        self._desired_rot = None  # numpy array (roll, pitch, yaw)
    
    def start(self, desired_pos, desired_rot):
        """
        소켓 서버 쓰레드 시작
        desired_pos: 목표 좌표를 저장할 numpy array
        desired_rot: 목표 회전을 저장할 numpy array
        """
        self._desired_pos = desired_pos
        self._desired_rot = desired_rot
        self.running = True
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """소켓 서버 종료"""
        self.running = False
        if self.server:
            self.server.close()  # 소켓 먼저 닫기
        if self.thread:
            self.thread.join(timeout=2)
        print("[서버] 소켓 서버 종료")
    
    def _server_loop(self):
        """소켓 서버 메인 루프"""
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        self.server.settimeout(1)
        
        print(f"[서버] 소켓 서버 시작 (포트: {self.port})")
        
        while self.running:
            try:
                conn, addr = self.server.accept()
                with conn:
                    length_data = conn.recv(4)
                    if not length_data:
                        continue

                    value_length = struct.unpack('I', length_data)[0]
                    value_data = conn.recv(value_length)

                    if value_data and len(value_data) == value_length:
                        # 6개의 float (x, y, z, r, p, y) 수신
                        coords = struct.unpack('ffffff', value_data)
                        if len(coords) == 6:
                            # Numpy array 업데이트 (크기가 6이어야 함)
                            for i in range(3):
                                self._desired_pos[i] = float(coords[i])
                            for i in range(3, 6):
                                self._desired_rot[i-3] = float(coords[i])
                            
                            print(f"\n[수신] Position: ({coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}) | Rotation: ({coords[3]:.3f}, {coords[4]:.3f}, {coords[5]:.3f})")
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[서버 오류] {e}")
        
        self.server.close()
