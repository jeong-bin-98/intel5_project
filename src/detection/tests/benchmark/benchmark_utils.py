"""
벤치마크 공통 유틸리티
=====================

모든 벤치마크 스크립트에서 공유하는 함수들을 정의합니다.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import glob
from config.paths import YOLO_BEST_PT, DATASET_IMAGES_OBB_VAL, RUNS_DIR


def find_model():
    """학습된 YOLO best.pt 모델을 찾습니다.

    1순위: config/paths.py의 YOLO_BEST_PT
    2순위: runs/detect/ 하위에서 가장 최근 best.pt를 자동 탐색

    Returns:
        str: best.pt 경로
    """
    if os.path.isfile(YOLO_BEST_PT):
        return YOLO_BEST_PT

    # fallback: runs/detect 하위 검색
    detect_dir = os.path.join(RUNS_DIR, "detect")
    if os.path.isdir(detect_dir):
        pattern = os.path.join(detect_dir, "*", "weights", "best.pt")
        candidates = glob.glob(pattern)
        if candidates:
            # 가장 최근 수정된 파일 선택
            latest = max(candidates, key=os.path.getmtime)
            print(f"⚠️  기본 모델 경로에 best.pt 없음 → 자동 탐지: {latest}")
            return latest

    print("❌ best.pt 모델을 찾을 수 없습니다.")
    print(f"   기본 경로: {YOLO_BEST_PT}")
    print(f"   검색 경로: {detect_dir}")
    sys.exit(1)


def find_test_image(img_path=None):
    """테스트 이미지를 찾습니다.

    Args:
        img_path: 사용자 지정 경로 (None이면 val 폴더에서 자동 선택)

    Returns:
        str: 이미지 파일 경로
    """
    if img_path and os.path.isfile(img_path):
        return img_path

    if os.path.isdir(DATASET_IMAGES_OBB_VAL):
        imgs = [f for f in os.listdir(DATASET_IMAGES_OBB_VAL)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if imgs:
            return os.path.join(DATASET_IMAGES_OBB_VAL, imgs[0])

    print("❌ 테스트 이미지를 찾을 수 없습니다.")
    print(f"   확인 경로: {DATASET_IMAGES_OBB_VAL}")
    sys.exit(1)
