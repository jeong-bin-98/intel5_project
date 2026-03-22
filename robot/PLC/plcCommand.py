from pymcprotocol import Type3E
from time import sleep

PLC_IP = "192.168.3.140"
PLC_PORT = 1025
plc = Type3E()
plc.connect(PLC_IP, PLC_PORT)

print("PLC 연결 성공")

ADDR_START = "M100"
ADDR_DONE = "M101"

print("PLC I/O 테스트 프로그램 실행 (Ctrl + C로 종료)")

def main():
    try:
        while True:
            start_signal = plc.batchread_bitunits(ADDR_START, 1)[0]
            if start_signal == 1:
                print("Start 신호 감지 -> 처리 시작")
                sleep(2)
                
                # Done 신호 PLC로 출력
                plc.batchwrite_bitunits(ADDR_DONE, [1])
                print("Done 신호 출력")
                sleep(1)

                # Done 신호 유지 후 초기화
                plc.batchwrite_bitunits(ADDR_DONE, [0])
                print("Done 신호 초기화 완료")

                # Start 신호 OFF 될 때까지 대기
                while plc.batchread_bitunits(ADDR_START, 1)[0] == 1:
                    sleep(0.1)
            else:
                sleep(0.1)
                
    except KeyboardInterrupt:
        print(">> 프로그램 종료")

    finally:
        plc.close()
        print(">> PLC 연결 종료")

if __name__ == "__main__":
    main()
    