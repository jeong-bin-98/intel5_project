"""
벤치마크 공통 유틸리티
=====================

모든 벤치마크 스크립트에서 공유하는 함수들을 정의합니다.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import json
import glob
import numpy as np
from config.paths import YOLO_BEST_PT, DATASET_IMAGES_OBB_VAL, RUNS_DIR, BENCHMARK_DIR


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


# =============================================================================
# 파이프라인 벤치마크용 테스트 프레임 저장/로드
# =============================================================================

FRAME_DIR = os.path.join(BENCHMARK_DIR, "test_frame")


def capture_test_frame():
    """RealSense에서 테스트 프레임(color + depth + intrinsics)을 캡처하고 저장합니다.

    Returns:
        (color_image, depth_image, intrinsics) 튜플
    """
    from src.detection.realsense_config import create_pipeline, apply_filters, capture_median_depth

    print("RealSense 카메라 연결 중...")
    pipeline, align, filters, intrinsics = create_pipeline()

    print("프레임 캡처 중 (multi-frame median)...")
    color_image, depth_image, intr = capture_median_depth(pipeline, align, filters)
    pipeline.stop()

    if color_image is None:
        print("❌ 프레임 캡처 실패")
        sys.exit(1)

    os.makedirs(FRAME_DIR, exist_ok=True)
    np.save(os.path.join(FRAME_DIR, "color.npy"), color_image)
    np.save(os.path.join(FRAME_DIR, "depth.npy"), depth_image)

    intr_data = {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "coeffs": list(intr.coeffs),
    }
    with open(os.path.join(FRAME_DIR, "intrinsics.json"), "w") as f:
        json.dump(intr_data, f, indent=2)

    print(f"✅ 테스트 프레임 저장: {FRAME_DIR}")
    print(f"   color: {color_image.shape}, depth: {depth_image.shape}")
    print(f"   intrinsics: {intr.width}x{intr.height} fx={intr.fx:.1f} fy={intr.fy:.1f}")

    return color_image, depth_image, intr


def load_test_frame():
    """저장된 테스트 프레임을 로드합니다.

    Returns:
        (color_image, depth_image, intrinsics) 튜플
    """
    import pyrealsense2 as rs

    color_path = os.path.join(FRAME_DIR, "color.npy")
    depth_path = os.path.join(FRAME_DIR, "depth.npy")
    intr_path = os.path.join(FRAME_DIR, "intrinsics.json")

    if not all(os.path.isfile(p) for p in [color_path, depth_path, intr_path]):
        print("❌ 저장된 테스트 프레임이 없습니다.")
        print("   먼저 --capture 옵션으로 프레임을 캡처하세요:")
        print("   python src/detection/tests/benchmark/test_benchmark_pipeline.py --capture")
        sys.exit(1)

    color_image = np.load(color_path)
    depth_image = np.load(depth_path)

    with open(intr_path, "r") as f:
        intr_data = json.load(f)

    intr = rs.intrinsics()
    intr.width = intr_data["width"]
    intr.height = intr_data["height"]
    intr.fx = intr_data["fx"]
    intr.fy = intr_data["fy"]
    intr.ppx = intr_data["ppx"]
    intr.ppy = intr_data["ppy"]
    intr.model = rs.distortion.inverse_brown_conrady
    intr.coeffs = intr_data["coeffs"]

    print(f"📷 테스트 프레임 로드: color={color_image.shape}, depth={depth_image.shape}")
    print(f"   intrinsics: {intr.width}x{intr.height} fx={intr.fx:.1f} fy={intr.fy:.1f}")

    return color_image, depth_image, intr


def has_test_frame():
    """저장된 테스트 프레임이 있는지 확인합니다."""
    return all(os.path.isfile(os.path.join(FRAME_DIR, f))
               for f in ["color.npy", "depth.npy", "intrinsics.json"])
