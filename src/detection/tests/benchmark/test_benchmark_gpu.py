"""
GPU (CUDA) Inference 벤치마크
==============================

YOLO11n-OBB 모델을 CUDA GPU에서 추론하는 속도를 측정합니다.
torch.cuda.synchronize()를 사용하여 비동기 연산 완료를 보장합니다.

사용법:
    python src/detection/tests/benchmark/test_benchmark_gpu.py --runs 50
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import argparse
import json
import time
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO
from config.paths import BENCHMARK_DIR
from src.detection.tests.benchmark.benchmark_utils import find_model, find_test_image


def run_benchmark(runs=50, warmup=10, img_path=None):
    """GPU (CUDA) 벤치마크를 실행합니다."""
    import cv2

    # CUDA 확인
    if not torch.cuda.is_available():
        print("❌ CUDA GPU를 사용할 수 없습니다.")
        print("   torch.cuda.is_available() = False")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

    # 이미지 준비
    test_img = find_test_image(img_path)
    image = cv2.imread(test_img)
    print(f"📷 테스트 이미지: {os.path.basename(test_img)} ({image.shape[1]}x{image.shape[0]})")

    # 모델 로드
    model_path = find_model()
    print(f"🔧 모델 로드: {os.path.basename(model_path)}")
    print(f"🖥️  디바이스: GPU ({gpu_name}, {vram_mb:.0f}MB VRAM)")
    model = YOLO(model_path)

    # Warm-up (GPU 커널 컴파일 + VRAM 할당 안정화)
    print(f"\n⏳ Warm-up ({warmup}회)...")
    for i in range(warmup):
        model(image, device="0", verbose=False)
        torch.cuda.synchronize()
    print("   Warm-up 완료")

    # 본 측정 (CUDA sync로 정밀 측정)
    print(f"\n🏃 벤치마크 실행 ({runs}회)...")
    times = []
    for i in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model(image, device="0", verbose=False)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"   {i + 1}/{runs} 완료 (현재 평균: {np.mean(times):.1f}ms)")

    # 결과 계산
    times_arr = np.array(times)
    result = {
        "device": f"GPU ({gpu_name})",
        "gpu_name": gpu_name,
        "vram_mb": round(vram_mb),
        "model": os.path.basename(model_path),
        "image": os.path.basename(test_img),
        "image_size": f"{image.shape[1]}x{image.shape[0]}",
        "runs": runs,
        "warmup": warmup,
        "avg_ms": round(float(np.mean(times_arr)), 2),
        "min_ms": round(float(np.min(times_arr)), 2),
        "max_ms": round(float(np.max(times_arr)), 2),
        "std_ms": round(float(np.std(times_arr)), 2),
        "median_ms": round(float(np.median(times_arr)), 2),
        "times_ms": [round(t, 2) for t in times],
        "timestamp": datetime.now().isoformat(),
    }

    # 결과 출력
    print("\n" + "=" * 50)
    print(f"📊 GPU ({gpu_name}) 벤치마크 결과")
    print("=" * 50)
    print(f"  평균: {result['avg_ms']:.2f} ms")
    print(f"  중앙: {result['median_ms']:.2f} ms")
    print(f"  최소: {result['min_ms']:.2f} ms")
    print(f"  최대: {result['max_ms']:.2f} ms")
    print(f"  표준편차: {result['std_ms']:.2f} ms")
    print("=" * 50)

    # JSON 저장
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    out_path = os.path.join(BENCHMARK_DIR, "result_gpu.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 결과 저장: {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU (CUDA) Inference Benchmark")
    parser.add_argument("--runs", type=int, default=50, help="반복 추론 횟수 (기본: 50)")
    parser.add_argument("--warmup", type=int, default=10, help="워밍업 횟수 (기본: 10)")
    parser.add_argument("--img", type=str, default=None, help="테스트 이미지 경로")
    args = parser.parse_args()

    run_benchmark(runs=args.runs, warmup=args.warmup, img_path=args.img)
