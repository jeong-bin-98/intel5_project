"""
Step 1: D435 카메라로 RGB + Depth 이미지 촬영

사용법:
    python step1_capture.py

조작법:
    - 's' 키: 현재 프레임 저장 (RGB + Depth)
    - 'q' 키: 종료

설명:
    D435 카메라가 보여주는 실시간 화면에서
    소켓이 잘 보이는 순간에 's'를 누르면 이미지가 저장됩니다.
    다양한 각도, 조명, 배치로 50~100장 촬영하세요.

팁:
    - 소켓을 빈(상자) 안에 여러 개 무작위로 넣고 촬영
    - 8핀, 12핀을 섞어서 넣기
    - 촬영할 때마다 소켓 위치/방향을 바꾸기
    - 조명 조건도 약간씩 바꾸면 더 좋음
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime


def main():
    # ===== 저장 폴더 생성 =====
    save_dir = "raw_captures"
    os.makedirs(f"{save_dir}/color", exist_ok=True)
    os.makedirs(f"{save_dir}/depth", exist_ok=True)

    # ===== D435 카메라 초기화 =====
    pipeline = rs.pipeline()
    config = rs.config()

    # 해상도 설정 (640x480이 가장 안정적)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)   # RGB
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)    # Depth

    # 카메라 시작
    profile = pipeline.start(config)

    # depth → color 정렬 (두 이미지의 픽셀이 1:1 대응되게)
    align = rs.align(rs.stream.color)

    # depth 노이즈 줄이기 위한 필터
    spatial_filter = rs.spatial_filter()       # 공간 필터 (주변 픽셀 참고)
    temporal_filter = rs.temporal_filter()     # 시간 필터 (이전 프레임 참고)
    hole_filling = rs.hole_filling_filter()    # 빈 구멍 메우기

    count = 0
    print("=" * 50)
    print("D435 이미지 캡처 도구")
    print("  's' = 저장  |  'q' = 종료")
    print("=" * 50)

    try:
        while True:
            # ===== 프레임 읽기 =====
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)         # depth를 color에 정렬

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # depth 필터 적용 (노이즈 제거)
            depth_frame = spatial_filter.process(depth_frame)
            depth_frame = temporal_filter.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            # numpy 배열로 변환
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # ===== 화면 표시 =====
            # depth를 컬러맵으로 시각화 (가까우면 빨강, 멀면 파랑)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # RGB와 depth를 나란히 표시
            display = np.hstack([color_image, depth_colormap])
            cv2.putText(display, f"Captured: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("D435 Capture (Left: RGB | Right: Depth)", display)

            # ===== 키 입력 처리 =====
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # 타임스탬프로 파일명 생성
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(f"{save_dir}/color/{ts}.png", color_image)
                np.save(f"{save_dir}/depth/{ts}.npy", depth_image)
                count += 1
                print(f"  [저장 #{count}] {ts}.png")

            elif key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n총 {count}장 저장 완료 → {save_dir}/")


if __name__ == "__main__":
    main()
