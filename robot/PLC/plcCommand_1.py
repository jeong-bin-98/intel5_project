from pymcprotocol import Type3E
from time import sleep

PLC_IP = "192.168.3.140"
PLC_PORT = 1025
plc = Type3E()
plc.connect(PLC_IP, PLC_PORT)
print("PLC 연결 성공")

INPUT_ADDRS = {"STAR1": "M100", "STAR2": "M101", "STAR3": "M102", "STAR4": "M103",}
OUTPUT_ADDRS = {"STAR1": "M200", "STAR2": "M201", "STAR3": "M202", "STAR4": "M203",}
print("PLC I/O 멀티 입력 테스트 프로그램 실행 (Ctrl + C로 종료)")

# 작업 로직
def process_input(name):
    print(f"{name} 신호 감지 -> 작업 시작")
    sleep(2)
    
    out_addr = OUTPUT_ADDRS[name]
    plc.batchwrite_bitunits(out_addr, [1])
    print(f"{name} -> DONE 출력 ({out_addr})")
    sleep(0.5)

    plc.batchwrite_bitunits(out_addr, [0])
    print(f"{name} 작업 완료 / 대기 중")

def main():
    try:
        while True:
            for name, addr in INPUT_ADDRS.items():
                # 입력 읽기 (1비트)
                value = plc.batchread_bitunits(addr, 1)[0]
                if value == 1:
                    process_input(name)
                    # 입력 OFF 될 때까지 대기 (채터링 방지)
                    while plc.batchread_bitunits(addr, 1)[0] == 1:
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
    