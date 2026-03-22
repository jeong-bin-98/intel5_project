"""
Step 2: Depth 정보로 배경 제거

사용법:
    python step2_depth_mask.py            # 자동 depth 추정 + 일괄 처리
    python step2_depth_mask.py --tune     # 트랙바로 수동 조절
    python step2_depth_mask.py --manual 200 700   # 수동 min/max 지정

설명:
    D435는 각 픽셀까지의 거리(depth)를 mm 단위로 알려줍니다.
    이걸 이용하면 소켓이 있는 영역만 깔끔하게 잘라낼 수 있어요.

    카메라를 고정하지 않고 촬영한 경우에도,
    이미지마다 자동으로 depth 범위를 추정하므로 문제없습니다.

원리 (자동 모드):
    1. 각 이미지의 depth 히스토그램을 분석
    2. 가장 큰 피크(= 바닥/배경)를 찾음
    3. 바닥보다 가까운 영역 = 소켓으로 판단
    4. min_depth, max_depth를 이미지별로 자동 설정
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.paths import RAW_CAPTURES_DIR, CLEANED_IMAGES_DIR

import numpy as np
import cv2
import glob
from scipy.signal import find_peaks


# ===== 설정값 =====
NOISE_MIN_MM = 100        # 이보다 가까운 depth는 센서 노이즈로 무시
MARGIN_MM = 50            # 바닥 피크에서 이만큼 여유를 두고 잘라냄
HIST_BIN_SIZE = 10        # 히스토그램 bin 크기 (mm)

INPUT_COLOR_DIR = os.path.join(RAW_CAPTURES_DIR, "color")
INPUT_DEPTH_DIR = os.path.join(RAW_CAPTURES_DIR, "depth")
OUTPUT_DIR = CLEANED_IMAGES_DIR


def auto_estimate_depth_range(depth_image, noise_min=NOISE_MIN_MM, margin=MARGIN_MM):
    """
    단일 depth 이미지에서 히스토그램을 분석하여
    min_depth, max_depth를 자동으로 추정합니다.

    원리:
        - depth 히스토그램에서 가장 큰 피크 = 바닥(배경)
        - 바닥 피크보다 가까운 쪽 = 소켓(전경)
        → max_depth = 바닥 피크 - margin
        → min_depth = noise_min (센서 노이즈 이후)

    Args:
        depth_image: depth 이미지 (H, W) — 단위: mm
        noise_min: 센서 노이즈 최소 거리 (mm)
        margin: 바닥 피크에서 빼줄 여유 (mm)

    Returns:
        (min_depth, max_depth): 추정된 depth 범위 (mm)
    """
    # 유효한 depth 값만 추출 (0 = 측정 실패, noise_min 이하 = 노이즈)
    valid = depth_image[depth_image > noise_min].flatten()

    if len(valid) == 0:
        # 유효한 데이터가 없으면 기본값 반환
        return noise_min, 1000

    # 히스토그램 생성
    max_val = int(valid.max())
    bins = np.arange(noise_min, max_val + HIST_BIN_SIZE, HIST_BIN_SIZE)

    if len(bins) < 3:
        return noise_min, max_val

    hist, bin_edges = np.histogram(valid, bins=bins)

    # 피크 찾기 (prominence로 의미있는 피크만)
    peaks, properties = find_peaks(hist, prominence=len(valid) * 0.01)

    if len(peaks) == 0:
        # 피크를 못 찾으면 히스토그램 최대값 위치 사용
        peak_idx = np.argmax(hist)
    else:
        # 가장 큰 피크 = 바닥(배경)이라고 판단
        # (배경이 보통 가장 넓은 면적을 차지하므로)
        tallest = np.argmax(hist[peaks])
        peak_idx = peaks[tallest]

    # 바닥 피크의 depth 값 (mm)
    floor_depth = bin_edges[peak_idx]

    # max_depth = 바닥보다 margin만큼 앞
    max_depth = max(floor_depth - margin, noise_min + 50)
    min_depth = noise_min

    return int(min_depth), int(max_depth)


def remove_background(color_image, depth_image, min_depth, max_depth):
    """
    depth 범위 밖의 픽셀을 검은색으로 만들어 배경을 제거합니다.

    Args:
        color_image: RGB 이미지 (H, W, 3)
        depth_image: depth 이미지 (H, W) — 단위: mm
        min_depth: 최소 거리 (mm)
        max_depth: 최대 거리 (mm)

    Returns:
        masked_image: 배경이 제거된 RGB 이미지
        mask: 전경 마스크 (0 또는 255)
    """
    # depth가 범위 안에 있는 픽셀만 True
    # (depth == 0은 센서가 측정 실패한 곳이므로 제외)
    mask = (depth_image > min_depth) & (depth_image < max_depth)

    # boolean → uint8 (0 또는 255)
    mask_uint8 = (mask * 255).astype(np.uint8)

    # 노이즈 제거: 작은 점들 없애기
    kernel = np.ones((5, 5), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)   # 작은 점 제거
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)  # 작은 구멍 메우기

    # 마스크를 RGB에 적용
    masked_image = cv2.bitwise_and(color_image, color_image, mask=mask_uint8)

    return masked_image, mask_uint8


def process_all(manual_min=None, manual_max=None):
    """
    저장된 모든 이미지에 배경 제거를 일괄 적용합니다.

    manual_min, manual_max가 주어지면 모든 이미지에 동일한 값을 사용하고,
    None이면 이미지마다 자동으로 depth 범위를 추정합니다.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    color_files = sorted(glob.glob(f"{INPUT_COLOR_DIR}/*.png"))
    print(f"처리할 이미지: {len(color_files)}장")

    if manual_min is not None and manual_max is not None:
        print(f"모드: 수동 지정 (min={manual_min}mm, max={manual_max}mm)")
    else:
        print(f"모드: 이미지별 자동 추정")

    print("-" * 50)

    for color_path in color_files:
        # 같은 이름의 depth 파일 찾기
        basename = os.path.splitext(os.path.basename(color_path))[0]
        depth_path = f"{INPUT_DEPTH_DIR}/{basename}.npy"

        if not os.path.exists(depth_path):
            print(f"  [스킵] depth 없음: {basename}")
            continue

        # 이미지 로드
        color = cv2.imread(color_path)
        depth = np.load(depth_path)  # mm 단위의 uint16

        # depth 범위 결정
        if manual_min is not None and manual_max is not None:
            min_d, max_d = manual_min, manual_max
        else:
            min_d, max_d = auto_estimate_depth_range(depth)

        # 배경 제거
        cleaned, mask = remove_background(color, depth, min_d, max_d)

        # 저장
        cv2.imwrite(f"{OUTPUT_DIR}/{basename}.png", cleaned)
        cv2.imwrite(f"{OUTPUT_DIR}/{basename}_mask.png", mask)
        print(f"  [완료] {basename}.png  (depth: {min_d}~{max_d}mm)")

    print(f"\n배경 제거 완료 → {OUTPUT_DIR}/")


def interactive_tuning():
    """
    트랙바로 min/max depth를 실시간 조절하며 최적값을 찾는 도구입니다.
    자동 추정값을 초기값으로 사용합니다.
    """
    color_files = sorted(glob.glob(f"{INPUT_COLOR_DIR}/*.png"))
    if not color_files:
        print(f"{INPUT_COLOR_DIR} 에 이미지가 없습니다. step1_capture.py를 먼저 실행하세요.")
        return

    # 첫 번째 이미지로 테스트
    basename = os.path.splitext(os.path.basename(color_files[0]))[0]
    color = cv2.imread(color_files[0])
    depth = np.load(f"{INPUT_DEPTH_DIR}/{basename}.npy")

    # 자동 추정값을 초기값으로 사용
    auto_min, auto_max = auto_estimate_depth_range(depth)
    print(f"자동 추정값: min={auto_min}mm, max={auto_max}mm")

    cv2.namedWindow("Depth Tuning")
    cv2.createTrackbar("Min (mm)", "Depth Tuning", auto_min, 2000, lambda x: None)
    cv2.createTrackbar("Max (mm)", "Depth Tuning", auto_max, 2000, lambda x: None)

    print("트랙바를 조절해서 소켓만 남는 범위를 찾으세요.")
    print("'q' = 종료")

    while True:
        min_d = cv2.getTrackbarPos("Min (mm)", "Depth Tuning")
        max_d = cv2.getTrackbarPos("Max (mm)", "Depth Tuning")

        cleaned, mask = remove_background(color, depth, min_d, max_d)

        # 원본 / 마스크 / 결과를 나란히 표시
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display = np.hstack([color, mask_color, cleaned])
        display = cv2.resize(display, (0, 0), fx=0.7, fy=0.7)

        cv2.putText(display, f"Min: {min_d}mm  Max: {max_d}mm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Depth Tuning", display)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print(f"\n최적값: MIN={min_d}mm, MAX={max_d}mm")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--tune":
        interactive_tuning()
    elif len(sys.argv) > 2 and sys.argv[1] == "--manual":
        manual_min = int(sys.argv[2])
        manual_max = int(sys.argv[3])
        process_all(manual_min, manual_max)
    else:
        process_all()
