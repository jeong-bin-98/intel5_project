from pymcprotocol import Type3E
from time import sleep

PLC_IP = "192.168.3.140"
PLC_PORT = 1025
plc = Type3E()
plc.connect(PLC_IP, PLC_PORT)
print("PLC 연결 성공") 

INPUT_WORDS = {"CMD1": "D100", "CMD2": "D101", "CMD3": "D102"}
OUTPUT_WORDS = {"CMD1": "D200", "CMD2": "D201", "CMD3": "D202"}
print("PLC I/O 멀티 입력 테스트 프로그램 실행 (Ctrl + C로 종료)")

# 작업 로직
def process_input(name, value):
    print(f"{name} 신호 감지 : 값 = {value}")
    sleep(1)
    
    processed_value = value * 10
    out_addr = OUTPUT_WORDS[name]
    plc.batchwrite_wordunits(out_addr, [processed_value])
    print(f"{name} -> D200 ~ D205 출력: {processed_value}")
    print("작업완료")

def main():
    try:
        while True:
            for name, addr in INPUT_WORDS.items():
                # 입력 읽기 (1비트)
                value = plc.batchread_wordunits(addr, 1)[0]
                if value != 0:
                    process_input(name, value)
                    # 입력 OFF 될 때까지 대기 (채터링 방지)
                    plc.batchwrite_wordunits(addr, [0])
            sleep(0.05)
                
    except KeyboardInterrupt:
        print(">> 프로그램 종료")

    finally:
        plc.close()
        print(">> PLC 연결 종료")

if __name__ == "__main__": 
    main()
    